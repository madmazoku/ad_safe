#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import ad_safe_lib as ad_safe


DEFAULT_COOLDOWN = {
    "every_epochs": ad_safe.DEFAULT_COOLDOWN_EVERY_EPOCHS,
    "seconds": ad_safe.DEFAULT_COOLDOWN_SECONDS,
    "gpu_max_temp": ad_safe.DEFAULT_GPU_MAX_TEMP,
    "gpu_resume_temp": ad_safe.DEFAULT_GPU_RESUME_TEMP,
    "gpu_temp_check_seconds": ad_safe.DEFAULT_GPU_TEMP_CHECK_SECONDS,
}
DEFAULT_PHASE = {
    "epochs": ad_safe.DEFAULT_EPOCHS,
    "resplit_runs": ad_safe.DEFAULT_RESPLIT_RUNS,
    "batch_size": ad_safe.DEFAULT_BATCH_SIZE,
    "learning_rate": ad_safe.DEFAULT_LR,
    "learning_rate_multiplier": 1.0,
    "patience": ad_safe.DEFAULT_PATIENCE,
    "seed": 0,
    "unfreeze_all": False,
    "adversarial": False,
    "adv_epsilon": ad_safe.DEFAULT_ADV_EPSILON,
    "adv_steps": ad_safe.DEFAULT_ADV_STEPS,
    "teacher_model_path": None,
    "distillation_alpha": ad_safe.DEFAULT_DISTILLATION_ALPHA,
    "distillation_temperature": ad_safe.DEFAULT_DISTILLATION_TEMPERATURE,
}
PHASE_CONFIG_FIELDS = frozenset(DEFAULT_PHASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one ad-safety backbone across an explicit JSON phase sweep."
    )
    parser.add_argument("config", type=Path, help="Path to sweep config JSON")
    return parser.parse_args()


def validate_fraction(value: object, *, field_name: str) -> float:
    fraction = float(value)
    if fraction <= 0 or fraction > 1:
        raise ValueError(f"{field_name} must be in the range (0, 1]")
    return fraction


def load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Config JSON is invalid: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Config JSON root must be an object")
    return payload


def resolve_config_path(config_path: Path) -> Path:
    if config_path.is_absolute():
        return config_path
    for root in (Path.cwd(), ad_safe.SWEEP_CONFIGS_DIR, ad_safe.ARTEFACTS_DIR, ad_safe.CHALLENGE_DIR):
        candidate = root / config_path
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Sweep config does not exist: {config_path}")


def resolve_config_relative_path(path_value: Any, *, config_dir: Path, field_name: str) -> Path:
    if path_value is None:
        raise ValueError(f"{field_name} must not be null")
    if not isinstance(path_value, str):
        raise ValueError(f"{field_name} must be a string")
    path = Path(path_value)
    return path if path.is_absolute() else (config_dir / path).resolve()


def get_object(container: dict[str, Any], field_name: str, default: dict[str, Any] | None = None) -> dict[str, Any]:
    value = container.get(field_name, default if default is not None else {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be an object")
    return dict(value)


def get_bool(container: dict[str, Any], field_name: str, default: bool) -> bool:
    value = container.get(field_name, default)
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean")
    return value


def get_string(container: dict[str, Any], field_name: str, default: str | None = None) -> str:
    value = container.get(field_name, default)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def discover_dataset_splits() -> list[str]:
    if not ad_safe.DATA_DIR.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {ad_safe.DATA_DIR}")
    return sorted(
        path.name
        for path in ad_safe.DATA_DIR.iterdir()
        if path.is_dir() and any(sample_path.is_file() for sample_path in path.glob("*/*"))
    )


def normalize_eval_splits(value: Any) -> tuple[str, ...]:
    if value is None:
        return tuple(discover_dataset_splits())
    if isinstance(value, str):
        splits = tuple(part.strip() for part in value.split(",") if part.strip())
    elif isinstance(value, list):
        if not all(isinstance(item, str) for item in value):
            raise ValueError("eval_splits must contain only strings")
        splits = tuple(item.strip() for item in value if item.strip())
    else:
        raise ValueError("eval_splits must be a string, list of strings, or omitted")
    if not splits:
        raise ValueError("eval_splits must contain at least one split")
    return tuple(dict.fromkeys(splits))


def validate_dataset_splits(train_split: str, eval_splits: tuple[str, ...]) -> None:
    available = set(discover_dataset_splits())
    requested = {train_split, *eval_splits}
    unknown = sorted(requested - available)
    if unknown:
        raise ValueError(
            f"Unknown dataset split(s): {', '.join(unknown)}. "
            f"Available: {', '.join(sorted(available))}"
        )


def build_cooldown_config(payload: dict[str, Any]) -> ad_safe.CooldownConfig:
    cooldown = {**DEFAULT_COOLDOWN, **payload}
    gpu_resume_temp = int(cooldown["gpu_resume_temp"])
    gpu_max_temp = int(cooldown["gpu_max_temp"])
    if gpu_max_temp > 0 and gpu_resume_temp == 0:
        gpu_resume_temp = gpu_max_temp - 5

    config = ad_safe.CooldownConfig(
        every_epochs=int(cooldown["every_epochs"]),
        seconds=float(cooldown["seconds"]),
        gpu_max_temp=gpu_max_temp,
        gpu_resume_temp=gpu_resume_temp,
        gpu_temp_check_seconds=float(cooldown["gpu_temp_check_seconds"]),
    )
    if config.every_epochs < 0:
        raise ValueError("cooldown.every_epochs must be non-negative")
    if config.seconds < 0:
        raise ValueError("cooldown.seconds must be non-negative")
    if config.gpu_max_temp < 0:
        raise ValueError("cooldown.gpu_max_temp must be non-negative")
    if config.gpu_resume_temp < 0:
        raise ValueError("cooldown.gpu_resume_temp must be non-negative")
    if config.gpu_temp_check_seconds <= 0:
        raise ValueError("cooldown.gpu_temp_check_seconds must be positive")
    if config.enabled and config.seconds <= 0:
        raise ValueError("cooldown.seconds must be positive when cooldown is enabled")
    if config.uses_temperature and config.gpu_resume_temp >= config.gpu_max_temp:
        raise ValueError("cooldown.gpu_resume_temp must be lower than cooldown.gpu_max_temp")
    return config


def ensure_no_unknown_phase_fields(container: dict[str, Any], *, context: str) -> None:
    unknown = sorted(set(container) - PHASE_CONFIG_FIELDS)
    if unknown:
        raise ValueError(f"Unknown {context} field(s): {', '.join(unknown)}")


def normalize_requested_seed(value: Any) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int):
        raise ValueError("seed must be an integer or null")
    return value


def normalize_teacher_model_path(value: Any, *, config_dir: Path) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("teacher_model_path must be a non-empty string or null")
    path = ad_safe.resolve_existing_path(
        value.strip(),
        search_roots=(
            config_dir,
            ad_safe.AD_SAFE_RUNS_DIR,
            ad_safe.ARTEFACTS_DIR,
            ad_safe.CHALLENGE_DIR,
        ),
    )
    if path is None:
        raise FileNotFoundError(f"Specified teacher_model_path does not exist: {value}")
    return str(path.resolve())


def build_training_config(backbone: str, values: dict[str, Any]) -> ad_safe.TrainingConfig:
    batch_size = int(values["batch_size"])
    epochs = int(values["epochs"])
    resplit_runs = int(values["resplit_runs"])
    patience = int(values["patience"])
    learning_rates = ad_safe.normalize_learning_rates_value(values["learning_rate"])
    learning_rate_multiplier = float(values["learning_rate_multiplier"])
    adv_epsilon = float(values["adv_epsilon"])
    adv_steps = int(values["adv_steps"])
    distillation_alpha = float(values["distillation_alpha"])
    distillation_temperature = float(values["distillation_temperature"])

    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if resplit_runs <= 0:
        raise ValueError("resplit_runs must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if patience < 0:
        raise ValueError("patience must be non-negative")
    if learning_rate_multiplier <= 0:
        raise ValueError("learning_rate_multiplier must be positive")
    if len(learning_rates) > 1 and learning_rate_multiplier != 1.0:
        raise ValueError("learning_rate_multiplier cannot be used with multiple learning_rate values")
    if adv_epsilon < 0:
        raise ValueError("adv_epsilon must be non-negative")
    if adv_steps < 0:
        raise ValueError("adv_steps must be non-negative")
    if distillation_alpha < 0 or distillation_alpha > 1:
        raise ValueError("distillation_alpha must be between 0 and 1")
    if distillation_temperature <= 0:
        raise ValueError("distillation_temperature must be positive")

    return ad_safe.TrainingConfig(
        base_model=backbone,
        epochs=epochs,
        patience=patience,
        batch_size=batch_size,
        learning_rate=learning_rates,
        learning_rate_multiplier=learning_rate_multiplier,
        resplit_runs=resplit_runs,
        unfreeze=(),
        adversarial=bool(values["adversarial"]),
        adv_epsilon=adv_epsilon,
        adv_steps=adv_steps,
        teacher_model_path=values["teacher_model_path"],
        distillation_alpha=distillation_alpha,
        distillation_temperature=distillation_temperature,
    )


def build_job_specs(backbone: str, raw: dict[str, Any], *, config_dir: Path) -> tuple[ad_safe.JobSpec, ...]:
    defaults = {**DEFAULT_PHASE, **get_object(raw, "defaults")}
    ensure_no_unknown_phase_fields(defaults, context="defaults")

    jobs_value = raw.get("jobs")
    if not isinstance(jobs_value, list) or not jobs_value:
        raise ValueError("jobs must be a non-empty list")

    jobs: list[ad_safe.JobSpec] = []
    for job_index, job_value in enumerate(jobs_value):
        if not isinstance(job_value, dict):
            raise ValueError(f"jobs[{job_index}] must be an object")
        job_overrides = {key: value for key, value in job_value.items() if key != "phases"}
        ensure_no_unknown_phase_fields(job_overrides, context=f"jobs[{job_index}]")
        phases_value = job_value.get("phases")
        if not isinstance(phases_value, list) or not phases_value:
            raise ValueError(f"jobs[{job_index}].phases must be a non-empty list")

        phases: list[ad_safe.PhaseSpec] = []
        for phase_index, phase_value in enumerate(phases_value):
            if not isinstance(phase_value, dict):
                raise ValueError(f"jobs[{job_index}].phases[{phase_index}] must be an object")
            ensure_no_unknown_phase_fields(phase_value, context=f"jobs[{job_index}].phases[{phase_index}]")
            values = {**defaults, **job_overrides, **phase_value}
            if not isinstance(values["unfreeze_all"], bool):
                raise ValueError("unfreeze_all must be a boolean")
            requested_seed = normalize_requested_seed(values["seed"])
            values["teacher_model_path"] = normalize_teacher_model_path(
                values["teacher_model_path"],
                config_dir=config_dir,
            )
            training_config = build_training_config(backbone, values)
            prefix = f"{job_index:03d}-{phase_index:03d}"
            phases.append(
                ad_safe.PhaseSpec(
                    job_index=job_index,
                    phase_index=phase_index,
                    prefix=prefix,
                    name=prefix,
                    requested_seed=requested_seed,
                    config=training_config,
                    unfreeze_all=bool(values["unfreeze_all"]),
                    signature={
                        "builder": "sweep",
                        "backbone": backbone,
                        "job_index": job_index,
                        "phase_index": phase_index,
                    },
                )
            )
        jobs.append(
            ad_safe.JobSpec(
                job_index=job_index,
                job_id=f"{job_index:03d}",
                display_name=f"{backbone}-{job_index:03d}",
                backbone=backbone,
                phases=tuple(phases),
            )
        )
    return tuple(jobs)


def build_run_plan(config_arg: Path) -> tuple[Path, str, ad_safe.RunPlan]:
    config_path = resolve_config_path(config_arg)
    raw = load_json_object(config_path)
    config_dir = config_path.parent
    backbone = get_string(raw, "backbone")
    if backbone not in ad_safe.SUPPORTED_BACKBONES:
        raise ValueError(
            f"Unknown backbone: {backbone}. Supported: {', '.join(sorted(ad_safe.SUPPORTED_BACKBONES))}"
        )

    output_root = (
        resolve_config_relative_path(raw["output_root"], config_dir=config_dir, field_name="output_root")
        if "output_root" in raw
        else ad_safe.ARTEFACTS_DIR / f"prod_{backbone}_model"
    )
    run_id_value = raw.get("run_id")
    if run_id_value is None:
        run_id = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    elif isinstance(run_id_value, str) and run_id_value.strip():
        run_id = run_id_value.strip()
    else:
        raise ValueError("run_id must be a non-empty string or null")
    if "/" in run_id or "\\" in run_id:
        raise ValueError("run_id must not contain path separators")

    train_split = get_string(raw, "train_split", "train")
    train_fraction = validate_fraction(raw.get("train_fraction", 1.0), field_name="train_fraction")
    eval_splits = normalize_eval_splits(raw.get("eval_splits"))
    validate_dataset_splits(train_split, eval_splits)
    output_dir = output_root / run_id

    return (
        config_path,
        backbone,
        ad_safe.RunPlan(
            output_dir=output_dir,
            run_id=run_id,
            train_split=train_split,
            eval_splits=eval_splits,
            jobs=build_job_specs(backbone, raw, config_dir=config_dir),
            cooldown=build_cooldown_config(get_object(raw, "cooldown")),
            resume=get_bool(raw, "resume", True),
            force=get_bool(raw, "force", False),
            source_config_path=config_path,
            setup_path=output_dir / "setup.json",
            check_foreign_contract=True,
            train_source=ad_safe.DatasetSourceSpec(
                name=train_split,
                fraction=train_fraction,
                seed=0,
            ),
            metadata={
                "builder": "sweep",
                "backbone": backbone,
                "train_fraction": train_fraction,
                "raw_config": raw,
            },
        ),
    )


def main() -> None:
    args = parse_args()
    config_path, backbone, run_plan = build_run_plan(args.config)
    print(f"Config: {config_path}")
    print(f"Backbone: {backbone}")
    ad_safe.run_training_plan(run_plan)


if __name__ == "__main__":
    main()
