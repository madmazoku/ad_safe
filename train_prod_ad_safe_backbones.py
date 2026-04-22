#!/usr/bin/env python3

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import ad_safe_lib as ad_safe


@dataclass(frozen=True)
class PhaseConfig:
    name: str
    epochs: int
    resplit_runs: int
    batch_size: int
    learning_rate: float
    learning_rate_multiplier: float
    patience: int
    seed: int
    unfreeze_all: bool

    def to_training_config(self, backbone_name: str) -> ad_safe.TrainingConfig:
        return ad_safe.TrainingConfig(
            base_model=backbone_name,
            epochs=self.epochs,
            patience=self.patience,
            batch_size=self.batch_size,
            learning_rate=(self.learning_rate,),
            learning_rate_multiplier=self.learning_rate_multiplier,
            resplit_runs=self.resplit_runs,
            unfreeze=(),
            adversarial=False,
            adv_epsilon=ad_safe.DEFAULT_ADV_EPSILON,
            adv_steps=ad_safe.DEFAULT_ADV_STEPS,
            teacher_model_path=None,
            distillation_alpha=ad_safe.DEFAULT_DISTILLATION_ALPHA,
            distillation_temperature=ad_safe.DEFAULT_DISTILLATION_TEMPERATURE,
        )

    def to_json(self) -> dict[str, object]:
        return {
            "epochs": self.epochs,
            "resplit_runs": self.resplit_runs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "learning_rate_multiplier": self.learning_rate_multiplier,
            "patience": self.patience,
            "seed": self.seed,
            "unfreeze_all": self.unfreeze_all,
        }


def parse_csv_values(value: str, *, name: str, cast: type) -> tuple:
    parts = tuple(part.strip() for part in value.split(",") if part.strip())
    if len(parts) not in {1, 2}:
        raise ValueError(f"{name} must contain one value or two comma-separated phase values")
    return tuple(cast(part) for part in parts)


def expand_phase_values(values: tuple) -> tuple:
    return (values[0], values[0]) if len(values) == 1 else values


def validate_fraction(value: object, *, field_name: str) -> float:
    fraction = float(value)
    if fraction <= 0 or fraction > 1:
        raise ValueError(f"{field_name} must be in the range (0, 1]")
    return fraction


def parse_backbones(values: list[str]) -> list[str]:
    backbone_names: list[str] = []
    for value in values:
        backbone_names.extend(part.strip() for part in value.split(",") if part.strip())
    if not backbone_names:
        raise ValueError("At least one --backbone must be provided")

    unknown = [name for name in backbone_names if name not in ad_safe.SUPPORTED_BACKBONES]
    if unknown:
        raise ValueError(
            f"Unknown backbone(s): {', '.join(unknown)}. "
            f"Supported: {', '.join(sorted(ad_safe.SUPPORTED_BACKBONES))}"
        )
    return list(dict.fromkeys(backbone_names))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train production ad-safety models for selected backbones")
    parser.add_argument(
        "--backbone",
        action="append",
        required=True,
        help="Backbone name or comma-separated backbone names. May be passed multiple times.",
    )
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--train-fraction", type=float, default=1.0)
    parser.add_argument("--epochs", default=str(ad_safe.DEFAULT_EPOCHS))
    parser.add_argument("--resplit-runs", default=str(ad_safe.DEFAULT_RESPLIT_RUNS))
    parser.add_argument("--batch-size", default=str(ad_safe.DEFAULT_BATCH_SIZE))
    parser.add_argument("--learning-rate", default=str(ad_safe.DEFAULT_LR))
    parser.add_argument("--learning-rate-multiplier", default="1.0")
    parser.add_argument("--patience", default=str(ad_safe.DEFAULT_PATIENCE))
    parser.add_argument(
        "--seed",
        default="0",
        help="Random seed, or two comma-separated phase seeds. Use 0 for a fresh random seed.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ad_safe.PROD_MODELS_DIR,
        help="Root directory for timestamped production model runs.",
    )
    parser.add_argument(
        "--cooldown-every-epochs",
        type=int,
        default=ad_safe.DEFAULT_COOLDOWN_EVERY_EPOCHS,
        help="Run cooldown after every N global epochs. Use 0 to disable periodic cooldown.",
    )
    parser.add_argument(
        "--cooldown-seconds",
        type=float,
        default=ad_safe.DEFAULT_COOLDOWN_SECONDS,
        help="Maximum cooldown duration in seconds. Required for enabled cooldown.",
    )
    parser.add_argument(
        "--gpu-max-temp",
        type=int,
        default=ad_safe.DEFAULT_GPU_MAX_TEMP,
        help="Start cooldown after an epoch when GPU temperature is at least this Celsius value. Use 0 to disable.",
    )
    parser.add_argument(
        "--gpu-resume-temp",
        type=int,
        default=ad_safe.DEFAULT_GPU_RESUME_TEMP,
        help="Resume early when GPU temperature is at or below this Celsius value. Defaults to --gpu-max-temp - 5.",
    )
    parser.add_argument(
        "--gpu-temp-check-seconds",
        type=float,
        default=ad_safe.DEFAULT_GPU_TEMP_CHECK_SECONDS,
        help="Seconds between GPU temperature checks during cooldown.",
    )
    return parser.parse_args()


def build_phase_configs(args: argparse.Namespace) -> tuple[PhaseConfig, PhaseConfig]:
    epochs = expand_phase_values(parse_csv_values(args.epochs, name="--epochs", cast=int))
    resplit_runs = expand_phase_values(parse_csv_values(args.resplit_runs, name="--resplit-runs", cast=int))
    batch_sizes = expand_phase_values(parse_csv_values(args.batch_size, name="--batch-size", cast=int))
    learning_rates = expand_phase_values(parse_csv_values(args.learning_rate, name="--learning-rate", cast=float))
    learning_rate_multipliers = expand_phase_values(
        parse_csv_values(args.learning_rate_multiplier, name="--learning-rate-multiplier", cast=float)
    )
    patience = expand_phase_values(parse_csv_values(args.patience, name="--patience", cast=int))
    seeds = tuple(
        ad_safe.resolve_effective_seed(seed)
        for seed in expand_phase_values(parse_csv_values(args.seed, name="--seed", cast=int))
    )

    phases = (
        PhaseConfig(
            name="phase1",
            epochs=epochs[0],
            resplit_runs=resplit_runs[0],
            batch_size=batch_sizes[0],
            learning_rate=learning_rates[0],
            learning_rate_multiplier=learning_rate_multipliers[0],
            patience=patience[0],
            seed=seeds[0],
            unfreeze_all=False,
        ),
        PhaseConfig(
            name="phase2",
            epochs=epochs[1],
            resplit_runs=resplit_runs[1],
            batch_size=batch_sizes[1],
            learning_rate=learning_rates[1],
            learning_rate_multiplier=learning_rate_multipliers[1],
            patience=patience[1],
            seed=seeds[1],
            unfreeze_all=True,
        ),
    )
    for phase in phases:
        if phase.epochs <= 0:
            raise ValueError(f"{phase.name} epochs must be positive")
        if phase.resplit_runs <= 0:
            raise ValueError(f"{phase.name} resplit-runs must be positive")
        if phase.batch_size <= 0:
            raise ValueError(f"{phase.name} batch-size must be positive")
        if phase.learning_rate <= 0:
            raise ValueError(f"{phase.name} learning-rate must be positive")
        if phase.learning_rate_multiplier <= 0:
            raise ValueError(f"{phase.name} learning-rate-multiplier must be positive")
        if phase.patience < 0:
            raise ValueError(f"{phase.name} patience must be non-negative")
    return phases


def discover_dataset_splits() -> list[str]:
    if not ad_safe.DATA_DIR.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {ad_safe.DATA_DIR}")
    return sorted(
        path.name
        for path in ad_safe.DATA_DIR.iterdir()
        if path.is_dir() and any(sample_path.is_file() for sample_path in path.glob("*/*"))
    )


def validate_train_split(train_split: str, split_names: list[str]) -> None:
    if train_split not in split_names:
        raise ValueError(
            f"Unknown train split: {train_split}. Available: {', '.join(split_names)}"
        )


def build_job_specs(
    *,
    backbone_names: list[str],
    phases: tuple[PhaseConfig, PhaseConfig],
) -> tuple[ad_safe.JobSpec, ...]:
    jobs: list[ad_safe.JobSpec] = []
    for job_index, backbone_name in enumerate(backbone_names):
        phase_specs = (
            ad_safe.PhaseSpec(
                job_index=job_index,
                phase_index=0,
                prefix=f"{backbone_name}_phase1",
                name="phase1",
                requested_seed=phases[0].seed,
                config=phases[0].to_training_config(backbone_name),
                unfreeze_all=phases[0].unfreeze_all,
                model_filename=f"{backbone_name}_phase1_model.pt",
                history_filename=f"{backbone_name}_phase1_training_history.png",
                json_filename=f"{backbone_name}_phase1.json",
                signature={"builder": "backbones", "backbone": backbone_name},
            ),
            ad_safe.PhaseSpec(
                job_index=job_index,
                phase_index=1,
                prefix=backbone_name,
                name="phase2",
                requested_seed=phases[1].seed,
                config=phases[1].to_training_config(backbone_name),
                unfreeze_all=phases[1].unfreeze_all,
                model_filename=f"{backbone_name}_model.pt",
                history_filename=f"{backbone_name}_phase2_training_history.png",
                json_filename=f"{backbone_name}_phase2.json",
                signature={"builder": "backbones", "backbone": backbone_name},
            ),
        )
        jobs.append(
            ad_safe.JobSpec(
                job_index=job_index,
                job_id=backbone_name,
                display_name=backbone_name,
                backbone=backbone_name,
                phases=phase_specs,
                metadata={"builder": "backbones"},
            )
        )
    return tuple(jobs)


def build_run_plan(args: argparse.Namespace) -> ad_safe.RunPlan:
    backbone_names = parse_backbones(args.backbone)
    phases = build_phase_configs(args)
    split_names = discover_dataset_splits()
    validate_train_split(args.train_split, split_names)
    train_fraction = validate_fraction(args.train_fraction, field_name="--train-fraction")
    cooldown_config = ad_safe.build_cooldown_config(
        every_epochs=args.cooldown_every_epochs,
        seconds=args.cooldown_seconds,
        gpu_max_temp=args.gpu_max_temp,
        gpu_resume_temp=args.gpu_resume_temp,
        gpu_temp_check_seconds=args.gpu_temp_check_seconds,
    )
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    output_dir = args.output_root / timestamp
    return ad_safe.RunPlan(
        output_dir=output_dir,
        run_id=timestamp,
        train_split=args.train_split,
        eval_splits=tuple(split_names),
        jobs=build_job_specs(backbone_names=backbone_names, phases=phases),
        cooldown=cooldown_config,
        resume=False,
        force=True,
        setup_path=output_dir / "setup.json",
        check_foreign_contract=True,
        train_source=ad_safe.DatasetSourceSpec(
            name=args.train_split,
            fraction=train_fraction,
            seed=phases[0].seed,
        ),
        metadata={
            "builder": "backbones",
            "backbones": backbone_names,
            "phase1": phases[0].to_json(),
            "phase2": phases[1].to_json(),
            "train_fraction": train_fraction,
        },
    )


def main() -> None:
    plan = build_run_plan(parse_args())
    print(f"Backbones: {', '.join(job.job_id for job in plan.jobs)}")
    ad_safe.run_training_plan(plan)


if __name__ == "__main__":
    main()
