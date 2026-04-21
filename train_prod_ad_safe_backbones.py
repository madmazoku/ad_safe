#!/usr/bin/env python3

import argparse
import csv
import gc
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch
from torch import nn

import ad_safe


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "prod_models"
METRIC_CSV_FIELDS = (
    ("accuracy", "acc"),
    ("auc", "auc"),
    ("nll", "nll"),
    ("avg_conf", "avg_conf"),
    ("avg_margin", "avg_margin"),
    ("avg_correct_conf", "avg_correct_conf"),
    ("avg_wrong_conf", "avg_wrong_conf"),
    ("safe_recall", "safe_recall"),
    ("unsafe_recall", "unsafe_recall"),
)


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

    def to_training_config(self, backbone_name: str, unfreeze: tuple[str, ...]) -> ad_safe.TrainingConfig:
        return ad_safe.TrainingConfig(
            base_model=backbone_name,
            epochs=self.epochs,
            patience=self.patience,
            batch_size=self.batch_size,
            learning_rate=(self.learning_rate,),
            learning_rate_multiplier=self.learning_rate_multiplier,
            resplit_runs=self.resplit_runs,
            unfreeze=unfreeze,
            adversarial=False,
            adv_epsilon=ad_safe.DEFAULT_ADV_EPSILON,
            adv_steps=ad_safe.DEFAULT_ADV_STEPS,
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
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory for timestamped production model runs.",
    )
    parser.add_argument(
        "--cooldown-every-epochs",
        type=int,
        default=0,
        help="Run cooldown after every N global epochs. Use 0 to disable periodic cooldown.",
    )
    parser.add_argument(
        "--cooldown-seconds",
        type=float,
        default=0.0,
        help="Maximum cooldown duration in seconds. Required for enabled cooldown.",
    )
    parser.add_argument(
        "--gpu-max-temp",
        type=int,
        default=0,
        help="Start cooldown after an epoch when GPU temperature is at least this Celsius value. Use 0 to disable.",
    )
    parser.add_argument(
        "--gpu-resume-temp",
        type=int,
        default=0,
        help="Resume early when GPU temperature is at or below this Celsius value. Defaults to --gpu-max-temp - 5.",
    )
    parser.add_argument(
        "--gpu-temp-check-seconds",
        type=float,
        default=15.0,
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


def build_cooldown_config(args: argparse.Namespace) -> ad_safe.CooldownConfig:
    gpu_resume_temp = args.gpu_resume_temp
    if args.gpu_max_temp > 0 and gpu_resume_temp == 0:
        gpu_resume_temp = args.gpu_max_temp - 5

    config = ad_safe.CooldownConfig(
        every_epochs=args.cooldown_every_epochs,
        seconds=args.cooldown_seconds,
        gpu_max_temp=args.gpu_max_temp,
        gpu_resume_temp=gpu_resume_temp,
        gpu_temp_check_seconds=args.gpu_temp_check_seconds,
    )
    if config.every_epochs < 0:
        raise ValueError("--cooldown-every-epochs must be non-negative")
    if config.seconds < 0:
        raise ValueError("--cooldown-seconds must be non-negative")
    if config.gpu_max_temp < 0:
        raise ValueError("--gpu-max-temp must be non-negative")
    if config.gpu_resume_temp < 0:
        raise ValueError("--gpu-resume-temp must be non-negative")
    if config.gpu_temp_check_seconds <= 0:
        raise ValueError("--gpu-temp-check-seconds must be positive")
    if config.enabled and config.seconds <= 0:
        raise ValueError("--cooldown-seconds must be positive when cooldown is enabled")
    if config.uses_temperature and config.gpu_resume_temp >= config.gpu_max_temp:
        raise ValueError("--gpu-resume-temp must be lower than --gpu-max-temp")
    return config


def discover_dataset_splits() -> list[str]:
    if not ad_safe.DATA_DIR.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {ad_safe.DATA_DIR}")
    return sorted(
        path.name
        for path in ad_safe.DATA_DIR.iterdir()
        if path.is_dir() and any(sample_path.is_file() for sample_path in path.glob("*/*"))
    )


def make_model(backbone_name: str) -> nn.Module:
    model = ad_safe.make_model(backbone_name)
    return model.to(ad_safe.DEVICE)


def resolve_phase_unfreeze(model: nn.Module, phase: PhaseConfig) -> tuple[str, ...]:
    if not phase.unfreeze_all:
        return ()
    return ad_safe.resolve_unfreeze_blocks(
        model,
        unfreeze_all=True,
        unfreeze_top=0,
        unfreeze=(),
    )


def run_training_phase(
    *,
    model: nn.Module,
    backbone_name: str,
    phase: PhaseConfig,
    full_train_dataset: object,
    train_split: str,
    model_path: Path,
    history_figure_path: Path,
    input_model_path: Path | None,
    cooldown_config: ad_safe.CooldownConfig,
) -> tuple[nn.Module, list[ad_safe.TrainingHistoryEntry], ad_safe.TrainingConfig]:
    unfreeze = resolve_phase_unfreeze(model, phase)
    config = phase.to_training_config(backbone_name, unfreeze)
    ad_safe.set_seed(phase.seed)
    ad_safe.configure_trainable_layers(model, unfreeze=config.unfreeze)
    print(f"\n{backbone_name} {phase.name}")
    ad_safe.describe_model(model, config)

    epoch_end_handlers: tuple[ad_safe.EpochEndHandler, ...] = ()
    if cooldown_config.enabled:
        epoch_end_handlers = (
            ad_safe.CooldownEpochEndHandler(
                config=cooldown_config,
                backbone_name=backbone_name,
                phase_name=phase.name,
            ),
        )
    model, history = ad_safe.train_model_across_resplits(
        model=model,
        full_train_dataset=full_train_dataset,
        config=config,
        best_model_path=model_path,
        split_log_prefix=f"{backbone_name} {phase.name}",
        epoch_end_handlers=epoch_end_handlers,
    )

    ad_safe.save_figure(
        ad_safe.generate_training_history_figure(
            history=history,
            config=config,
            train_split=train_split,
            seed=phase.seed,
            input_model_path=input_model_path,
            output_model_path=model_path,
        ),
        history_figure_path,
    )
    return model, history, config


def run_foreign_contract_check(model_path: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT_DIR / "check_ad_safe_contract.py"),
            str(model_path),
        ],
        check=True,
    )


def release_torch_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def train_backbone(
    *,
    backbone_name: str,
    phases: tuple[PhaseConfig, PhaseConfig],
    full_train_dataset: object,
    train_split: str,
    output_dir: Path,
    cooldown_config: ad_safe.CooldownConfig,
) -> dict[str, object]:
    print(f"\n=== Training {backbone_name} ===")
    ad_safe.set_seed(phases[0].seed)
    model = make_model(backbone_name)
    model_path = output_dir / f"{backbone_name}_model.pt"
    phase1_history_path = output_dir / f"{backbone_name}_phase1_training_history.png"
    phase2_history_path = output_dir / f"{backbone_name}_phase2_training_history.png"

    ad_safe.save_model(model, model_path)
    model, phase1_history, phase1_config = run_training_phase(
        model=model,
        backbone_name=backbone_name,
        phase=phases[0],
        full_train_dataset=full_train_dataset,
        train_split=train_split,
        model_path=model_path,
        history_figure_path=phase1_history_path,
        input_model_path=None,
        cooldown_config=cooldown_config,
    )
    ad_safe.save_model(model, model_path)
    model, phase2_history, phase2_config = run_training_phase(
        model=model,
        backbone_name=backbone_name,
        phase=phases[1],
        full_train_dataset=full_train_dataset,
        train_split=train_split,
        model_path=model_path,
        history_figure_path=phase2_history_path,
        input_model_path=model_path,
        cooldown_config=cooldown_config,
    )
    ad_safe.save_model(model, model_path)
    del model
    release_torch_memory()
    run_foreign_contract_check(model_path)
    return {
        "backbone": backbone_name,
        "model_path": model_path,
        "phase1_history_path": phase1_history_path,
        "phase2_history_path": phase2_history_path,
        "phase1_config": ad_safe.config_to_json_dict(phase1_config),
        "phase2_config": ad_safe.config_to_json_dict(phase2_config),
        "phase1_history_entries": len(phase1_history),
        "phase2_history_entries": len(phase2_history),
    }


def load_eval_loaders(split_names: list[str], preloaded_datasets: dict[str, object]) -> dict[str, object]:
    split_loaders = {}
    for split_name in split_names:
        dataset = preloaded_datasets.get(split_name)
        if dataset is None:
            dataset = ad_safe.load_dataset(split_name)
        split_loaders[split_name], = ad_safe.make_data_loader(
            dataset,
            batch_size=ad_safe.DEFAULT_BATCH_SIZE,
            shuffle=False,
        )
    return split_loaders


def evaluate_model(
    *,
    backbone_name: str,
    model_path: Path,
    split_names: list[str],
    split_loaders: dict[str, object],
) -> dict[str, object]:
    model = ad_safe.load_model(model_path)
    row: dict[str, object] = {
        "backbone": backbone_name,
        "model_path": str(model_path.resolve()),
    }
    for split_name in split_names:
        metrics = ad_safe.evaluate_metrics(model, split_loaders[split_name], split_name)
        for metric_name, csv_name in METRIC_CSV_FIELDS:
            row[f"{csv_name}_{split_name}"] = getattr(metrics, metric_name)
    del model
    release_torch_memory()
    return row


def build_accuracy_csv_fieldnames(split_names: list[str]) -> list[str]:
    return [
        "backbone",
        "model_path",
        *[
            f"{csv_name}_{split_name}"
            for split_name in split_names
            for _, csv_name in METRIC_CSV_FIELDS
        ],
    ]


def write_accuracy_csv(rows: list[dict[str, object]], split_names: list[str], output_path: Path) -> None:
    fieldnames = build_accuracy_csv_fieldnames(split_names)
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Accuracy CSV saved to {output_path}")


def print_accuracy_table(rows: list[dict[str, object]], split_names: list[str]) -> None:
    headers = ["backbone", *(f"acc_{split_name}" for split_name in split_names)]
    widths = {
        header: max(
            len(header),
            *(len(str(row[header])) if header == "backbone" else len(f"{float(row[header]):.4f}") for row in rows),
        )
        for header in headers
    }
    print("\nAccuracy:")
    print("  ".join(header.ljust(widths[header]) for header in headers))
    print("  ".join("-" * widths[header] for header in headers))
    for row in rows:
        values = [str(row["backbone"]).ljust(widths["backbone"])]
        values.extend(
            f"{float(row[f'acc_{split_name}']):.4f}".ljust(widths[f"acc_{split_name}"])
            for split_name in split_names
        )
        print("  ".join(values))


def write_setup_json(
    *,
    output_path: Path,
    timestamp: str,
    args: argparse.Namespace,
    backbone_names: list[str],
    phases: tuple[PhaseConfig, PhaseConfig],
    cooldown_config: ad_safe.CooldownConfig,
    split_names: list[str],
    models: list[dict[str, object]] | None = None,
) -> None:
    payload = {
        "timestamp": timestamp,
        "argv": sys.argv[1:],
        "device": str(ad_safe.DEVICE),
        "dataset_dir": str(ad_safe.DATA_DIR.resolve()),
        "train_split": args.train_split,
        "eval_splits": split_names,
        "backbones": backbone_names,
        "output_dir": str(output_path.parent.resolve()),
        "phase1": phases[0].to_json(),
        "phase2": phases[1].to_json(),
        "cooldown": cooldown_config.to_json(),
        "models": []
        if models is None
        else [
            {
                "backbone": model_info["backbone"],
                "model_path": str(Path(model_info["model_path"]).resolve()),
                "phase1_training_history_figure_path": str(Path(model_info["phase1_history_path"]).resolve()),
                "phase2_training_history_figure_path": str(Path(model_info["phase2_history_path"]).resolve()),
                "phase1_config": model_info["phase1_config"],
                "phase2_config": model_info["phase2_config"],
                "phase1_history_entries": model_info["phase1_history_entries"],
                "phase2_history_entries": model_info["phase2_history_entries"],
            }
            for model_info in models
        ],
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Setup saved to {output_path}")


def main() -> None:
    args = parse_args()
    backbone_names = parse_backbones(args.backbone)
    phases = build_phase_configs(args)
    cooldown_config = build_cooldown_config(args)
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    output_dir = args.output_root / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output dir: {output_dir}")
    print(f"Phase seeds: {phases[0].seed}, {phases[1].seed}")
    print(f"Cooldown: {cooldown_config.to_json()}")
    print(f"Backbones: {', '.join(backbone_names)}")
    print(f"Train split: {args.train_split}")

    split_names = discover_dataset_splits()
    write_setup_json(
        output_path=output_dir / "setup.json",
        timestamp=timestamp,
        args=args,
        backbone_names=backbone_names,
        phases=phases,
        cooldown_config=cooldown_config,
        split_names=split_names,
        models=None,
    )
    full_train_dataset = ad_safe.load_dataset(args.train_split)
    split_loaders = load_eval_loaders(split_names, {args.train_split: full_train_dataset})
    model_infos: list[dict[str, object]] = []
    accuracy_rows: list[dict[str, object]] = []
    for backbone_name in backbone_names:
        model_info = train_backbone(
            backbone_name=backbone_name,
            phases=phases,
            full_train_dataset=full_train_dataset,
            train_split=args.train_split,
            output_dir=output_dir,
            cooldown_config=cooldown_config,
        )
        model_infos.append(model_info)
        accuracy_rows.append(
            evaluate_model(
                backbone_name=str(model_info["backbone"]),
                model_path=Path(model_info["model_path"]),
                split_names=split_names,
                split_loaders=split_loaders,
            )
        )
        write_accuracy_csv(accuracy_rows, split_names, output_dir / "accuracy.csv")
        write_setup_json(
            output_path=output_dir / "setup.json",
            timestamp=timestamp,
            args=args,
            backbone_names=backbone_names,
            phases=phases,
            cooldown_config=cooldown_config,
            split_names=split_names,
            models=model_infos,
        )

    print_accuracy_table(accuracy_rows, split_names)


if __name__ == "__main__":
    main()
