#!/usr/bin/env python3

import argparse
import csv
import gc
import json
import subprocess
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch import nn

import ad_safe


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_COOLDOWN = {
    "every_epochs": 0,
    "seconds": 0.0,
    "gpu_max_temp": 0,
    "gpu_resume_temp": 0,
    "gpu_temp_check_seconds": 15.0,
}
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


@dataclass(frozen=True)
class PhaseSpec:
    job_index: int
    phase_index: int
    prefix: str
    requested_seed: int | None
    config: ad_safe.TrainingConfig
    unfreeze_all: bool
    signature: dict[str, Any]


@dataclass(frozen=True)
class SweepConfig:
    config_path: Path
    raw: dict[str, Any]
    backbone: str
    output_dir: Path
    run_id: str
    train_split: str
    eval_splits: tuple[str, ...]
    resume: bool
    force: bool
    cooldown: ad_safe.CooldownConfig
    jobs: tuple[tuple[PhaseSpec, ...], ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one ad-safety backbone across an explicit JSON phase sweep."
    )
    parser.add_argument("config", type=Path, help="Path to sweep config JSON")
    return parser.parse_args()


def load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Config JSON is invalid: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Config JSON root must be an object")
    return payload


def resolve_config_path(config_path: Path) -> Path:
    return config_path if config_path.is_absolute() else (Path.cwd() / config_path).resolve()


def resolve_config_relative_path(path_value: Any, *, config_dir: Path, field_name: str) -> Path:
    if path_value is None:
        raise ValueError(f"{field_name} must not be null")
    if not isinstance(path_value, str):
        raise ValueError(f"{field_name} must be a string")
    path = Path(path_value)
    return path if path.is_absolute() else (config_dir / path).resolve()


def path_for_json(path: Path | None) -> str | None:
    if path is None:
        return None
    path = path.resolve()
    try:
        return path.relative_to(SCRIPT_DIR).as_posix()
    except ValueError:
        return str(path)


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


def discover_dataset_splits() -> list[str]:
    if not ad_safe.DATA_DIR.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {ad_safe.DATA_DIR}")
    return sorted(
        path.name
        for path in ad_safe.DATA_DIR.iterdir()
        if path.is_dir() and any(sample_path.is_file() for sample_path in path.glob("*/*"))
    )


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
    path = resolve_config_relative_path(
        value.strip(),
        config_dir=config_dir,
        field_name="teacher_model_path",
    )
    if not path.exists():
        raise FileNotFoundError(f"Specified teacher_model_path does not exist: {path}")
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


def build_phase_specs(backbone: str, raw: dict[str, Any], *, config_dir: Path) -> tuple[tuple[PhaseSpec, ...], ...]:
    defaults = {**DEFAULT_PHASE, **get_object(raw, "defaults")}
    ensure_no_unknown_phase_fields(defaults, context="defaults")

    jobs_value = raw.get("jobs")
    if not isinstance(jobs_value, list) or not jobs_value:
        raise ValueError("jobs must be a non-empty list")

    jobs: list[tuple[PhaseSpec, ...]] = []
    for job_index, job_value in enumerate(jobs_value):
        if not isinstance(job_value, dict):
            raise ValueError(f"jobs[{job_index}] must be an object")
        job_overrides = {key: value for key, value in job_value.items() if key != "phases"}
        ensure_no_unknown_phase_fields(job_overrides, context=f"jobs[{job_index}]")
        phases_value = job_value.get("phases")
        if not isinstance(phases_value, list) or not phases_value:
            raise ValueError(f"jobs[{job_index}].phases must be a non-empty list")

        phase_specs: list[PhaseSpec] = []
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
            signature = {
                "backbone": backbone,
                "job_index": job_index,
                "phase_index": phase_index,
                "requested_seed": requested_seed,
                "unfreeze_all": bool(values["unfreeze_all"]),
                "training_config": ad_safe.config_to_json_dict(training_config),
            }
            phase_specs.append(
                PhaseSpec(
                    job_index=job_index,
                    phase_index=phase_index,
                    prefix=prefix,
                    requested_seed=requested_seed,
                    config=training_config,
                    unfreeze_all=bool(values["unfreeze_all"]),
                    signature=signature,
                )
            )
        jobs.append(tuple(phase_specs))
    return tuple(jobs)


def load_sweep_config(config_arg: Path) -> SweepConfig:
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
        else SCRIPT_DIR / f"prod_{backbone}_model"
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
    eval_splits = normalize_eval_splits(raw.get("eval_splits"))
    validate_dataset_splits(train_split, eval_splits)

    return SweepConfig(
        config_path=config_path,
        raw=raw,
        backbone=backbone,
        output_dir=output_root / run_id,
        run_id=run_id,
        train_split=train_split,
        eval_splits=eval_splits,
        resume=get_bool(raw, "resume", True),
        force=get_bool(raw, "force", False),
        cooldown=build_cooldown_config(get_object(raw, "cooldown")),
        jobs=build_phase_specs(backbone, raw, config_dir=config_dir),
    )


def resolve_phase_unfreeze(model: nn.Module, phase: PhaseSpec) -> tuple[str, ...]:
    if not phase.unfreeze_all:
        return ()
    return ad_safe.resolve_unfreeze_blocks(
        model,
        unfreeze_all=True,
        unfreeze_top=0,
        unfreeze=(),
    )


def finalize_phase_config(model: nn.Module, phase: PhaseSpec) -> ad_safe.TrainingConfig:
    return ad_safe.TrainingConfig(
        base_model=phase.config.base_model,
        epochs=phase.config.epochs,
        patience=phase.config.patience,
        batch_size=phase.config.batch_size,
        learning_rate=phase.config.learning_rate,
        learning_rate_multiplier=phase.config.learning_rate_multiplier,
        resplit_runs=phase.config.resplit_runs,
        unfreeze=resolve_phase_unfreeze(model, phase),
        adversarial=phase.config.adversarial,
        adv_epsilon=phase.config.adv_epsilon,
        adv_steps=phase.config.adv_steps,
        teacher_model_path=phase.config.teacher_model_path,
        distillation_alpha=phase.config.distillation_alpha,
        distillation_temperature=phase.config.distillation_temperature,
    )


def build_phase_signature(phase: PhaseSpec, config: ad_safe.TrainingConfig) -> dict[str, Any]:
    return {
        **phase.signature,
        "training_config": ad_safe.config_to_json_dict(config),
    }


def load_existing_phase_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Existing phase JSON is invalid: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Existing phase JSON root must be an object: {path}")
    return payload


def should_skip_phase(
    *,
    phase: PhaseSpec,
    config: ad_safe.TrainingConfig,
    model_path: Path,
    json_path: Path,
    sweep: SweepConfig,
) -> bool:
    if sweep.force or not sweep.resume:
        return False
    existing = load_existing_phase_json(json_path)
    if existing is None or not model_path.exists():
        return False
    current_signature = build_phase_signature(phase, config)
    existing_signature = existing.get("sweep_signature")
    if existing.get("status") == "completed" and existing_signature == current_signature:
        print(f"Skipping completed phase {phase.prefix}")
        return True
    if existing.get("status") == "completed" and existing_signature != current_signature:
        raise ValueError(
            f"Existing completed artifact {json_path} does not match the current config. "
            "Use a new run_id/output directory or force=true to retrain."
        )
    return False


def should_skip_phase_before_model_load(
    *,
    phase: PhaseSpec,
    model_path: Path,
    json_path: Path,
    sweep: SweepConfig,
) -> bool:
    if sweep.force or not sweep.resume:
        return False
    existing = load_existing_phase_json(json_path)
    if existing is None or not model_path.exists():
        return False
    existing_signature = existing.get("sweep_requested_signature")
    if existing.get("status") == "completed" and existing_signature == phase.signature:
        print(f"Skipping completed phase {phase.prefix}")
        return True
    if existing.get("status") == "completed" and existing_signature != phase.signature:
        raise ValueError(
            f"Existing completed artifact {json_path} does not match the current config. "
            "Use a new run_id/output directory or force=true to retrain."
        )
    return False


def build_phase_payload(
    *,
    sweep: SweepConfig,
    phase: PhaseSpec,
    status: str,
    seed: int,
    config: ad_safe.TrainingConfig,
    model_path: Path,
    history_path: Path,
    previous_model_path: Path | None,
    accuracy: dict[str, float] | None = None,
    metrics: dict[str, dict[str, float | None]] | None = None,
    error: str | None = None,
    error_traceback: str | None = None,
) -> dict[str, Any]:
    eval_split = sweep.eval_splits[0]
    return {
        "timestamp": sweep.run_id,
        "seed": seed,
        "train_split": sweep.train_split,
        "eval_split": eval_split,
        "base_model": sweep.backbone,
        "original_model_path": path_for_json(previous_model_path),
        "training_checkpoint_path": path_for_json(model_path),
        "training_history_figure_path": path_for_json(history_path),
        "training_config": ad_safe.config_to_json_dict(config),
        "status": status,
        "job_index": phase.job_index,
        "phase_index": phase.phase_index,
        "phase_prefix": phase.prefix,
        "output_model_path": path_for_json(model_path),
        "accuracy": accuracy or {},
        "metrics": metrics or {},
        "error": error,
        "error_traceback": error_traceback,
        "sweep": {
            "config_path": path_for_json(sweep.config_path),
            "output_dir": path_for_json(sweep.output_dir),
            "eval_splits": list(sweep.eval_splits),
            "resume": sweep.resume,
            "force": sweep.force,
            "cooldown": sweep.cooldown.to_json(),
            "requested_seed": phase.requested_seed,
            "unfreeze_all": phase.unfreeze_all,
        },
        "sweep_signature": build_phase_signature(phase, config),
        "sweep_requested_signature": phase.signature,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"JSON saved to {path}")


def release_torch_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_foreign_contract_check(model_path: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT_DIR / "check_ad_safe_contract.py"),
            str(model_path),
        ],
        check=True,
    )


def load_eval_loaders(
    *,
    split_names: tuple[str, ...],
    batch_size: int,
    preloaded_datasets: dict[str, Any],
) -> dict[str, Any]:
    split_loaders = {}
    for split_name in split_names:
        dataset = preloaded_datasets.get(split_name)
        if dataset is None:
            dataset = ad_safe.load_dataset(split_name)
        split_loaders[split_name], = ad_safe.make_data_loader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
        )
    return split_loaders


def evaluate_model(
    *,
    model_path: Path,
    split_loaders: dict[str, Any],
) -> dict[str, dict[str, float | None]]:
    model = ad_safe.load_model(model_path)
    metrics_by_split: dict[str, dict[str, float | None]] = {}
    for split_name, split_loader in split_loaders.items():
        metrics_by_split[split_name] = ad_safe.evaluate_metrics(
            model,
            split_loader,
            split_name,
        ).to_json_dict()
    del model
    release_torch_memory()
    return metrics_by_split


def read_accuracy_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def build_accuracy_csv_fieldnames(eval_splits: tuple[str, ...]) -> list[str]:
    return [
        "prefix",
        "job_index",
        "phase_index",
        "model_path",
        *[
            f"{csv_name}_{split_name}"
            for split_name in eval_splits
            for _, csv_name in METRIC_CSV_FIELDS
        ],
    ]


def write_accuracy_csv(path: Path, rows: list[dict[str, Any]], eval_splits: tuple[str, ...]) -> None:
    fieldnames = build_accuracy_csv_fieldnames(eval_splits)
    normalized_rows = []
    for row in rows:
        normalized_rows.append(
            {
                **{field: "" for field in fieldnames},
                **{key: value for key, value in row.items() if key in fieldnames},
            }
        )
    normalized_rows.sort(key=lambda row: str(row["prefix"]))
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(normalized_rows)
    print(f"Accuracy CSV saved to {path}")


def update_accuracy_csv(
    *,
    path: Path,
    phase: PhaseSpec,
    model_path: Path,
    metrics: dict[str, dict[str, float | None]],
    eval_splits: tuple[str, ...],
) -> None:
    rows = [row for row in read_accuracy_rows(path) if row.get("prefix") != phase.prefix]
    row: dict[str, Any] = {
        "prefix": phase.prefix,
        "job_index": phase.job_index,
        "phase_index": phase.phase_index,
        "model_path": path_for_json(model_path),
    }
    for split_name in eval_splits:
        split_metrics = metrics[split_name]
        for metric_name, csv_name in METRIC_CSV_FIELDS:
            value = split_metrics.get(metric_name)
            row[f"{csv_name}_{split_name}"] = "" if value is None else f"{float(value):.8f}"
    rows.append(row)
    write_accuracy_csv(path, rows, eval_splits)


def make_model(backbone: str) -> nn.Module:
    return ad_safe.make_model(backbone).to(ad_safe.DEVICE)


def train_phase(
    *,
    sweep: SweepConfig,
    phase: PhaseSpec,
    previous_model_path: Path | None,
    full_train_dataset: Any,
    split_loaders: dict[str, Any],
    accuracy_csv_path: Path,
) -> Path:
    model_path = sweep.output_dir / f"{phase.prefix}.pt"
    json_path = sweep.output_dir / f"{phase.prefix}.json"
    history_path = sweep.output_dir / f"{phase.prefix}.png"

    print(f"\n=== Phase {phase.prefix} ===")
    if should_skip_phase_before_model_load(
        phase=phase,
        model_path=model_path,
        json_path=json_path,
        sweep=sweep,
    ):
        existing = load_existing_phase_json(json_path) or {}
        metrics = existing.get("metrics")
        if isinstance(metrics, dict) and all(split in metrics for split in sweep.eval_splits):
            update_accuracy_csv(
                path=accuracy_csv_path,
                phase=phase,
                model_path=model_path,
                metrics=metrics,
                eval_splits=sweep.eval_splits,
            )
        return model_path

    model = ad_safe.load_model(previous_model_path) if previous_model_path is not None else make_model(sweep.backbone)
    config = finalize_phase_config(model, phase)

    if should_skip_phase(
        phase=phase,
        config=config,
        model_path=model_path,
        json_path=json_path,
        sweep=sweep,
    ):
        existing = load_existing_phase_json(json_path) or {}
        metrics = existing.get("metrics")
        if isinstance(metrics, dict) and all(split in metrics for split in sweep.eval_splits):
            update_accuracy_csv(
                path=accuracy_csv_path,
                phase=phase,
                model_path=model_path,
                metrics=metrics,
                eval_splits=sweep.eval_splits,
            )
        del model
        release_torch_memory()
        return model_path

    seed = ad_safe.resolve_effective_seed(phase.requested_seed)
    ad_safe.set_seed(seed)
    ad_safe.configure_trainable_layers(model, unfreeze=config.unfreeze)
    ad_safe.describe_model(model, config)

    write_json(
        json_path,
        build_phase_payload(
            sweep=sweep,
            phase=phase,
            status="running",
            seed=seed,
            config=config,
            model_path=model_path,
            history_path=history_path,
            previous_model_path=previous_model_path,
        ),
    )

    try:
        ad_safe.save_model(model, model_path)
        epoch_end_handlers = ()
        if sweep.cooldown.enabled:
            epoch_end_handlers = (
                ad_safe.CooldownEpochEndHandler(
                    config=sweep.cooldown,
                    backbone_name=sweep.backbone,
                    phase_name=phase.prefix,
                ),
            )
        model, history = ad_safe.train_model_across_resplits(
            model=model,
            full_train_dataset=full_train_dataset,
            config=config,
            best_model_path=model_path,
            split_log_prefix=phase.prefix,
            epoch_end_handlers=epoch_end_handlers,
        )
        ad_safe.save_model(model, model_path)
        ad_safe.save_figure(
            ad_safe.generate_training_history_figure(
                history=history,
                config=config,
                train_split=sweep.train_split,
                seed=seed,
                input_model_path=previous_model_path,
                output_model_path=model_path,
            ),
            history_path,
        )
        del model
        release_torch_memory()

        run_foreign_contract_check(model_path)
        metrics = evaluate_model(model_path=model_path, split_loaders=split_loaders)
        accuracy = {
            split_name: float(split_metrics["accuracy"])
            for split_name, split_metrics in metrics.items()
        }
        write_json(
            json_path,
            build_phase_payload(
                sweep=sweep,
                phase=phase,
                status="completed",
                seed=seed,
                config=config,
                model_path=model_path,
                history_path=history_path,
                previous_model_path=previous_model_path,
                accuracy=accuracy,
                metrics=metrics,
            ),
        )
        update_accuracy_csv(
            path=accuracy_csv_path,
            phase=phase,
            model_path=model_path,
            metrics=metrics,
            eval_splits=sweep.eval_splits,
        )
        return model_path
    except Exception as exc:
        error_traceback = traceback.format_exc()
        write_json(
            json_path,
            build_phase_payload(
                sweep=sweep,
                phase=phase,
                status="failed",
                seed=seed,
                config=config,
                model_path=model_path,
                history_path=history_path,
                previous_model_path=previous_model_path,
                error=str(exc),
                error_traceback=error_traceback,
            ),
        )
        raise


def print_accuracy_table(path: Path, eval_splits: tuple[str, ...]) -> None:
    rows = read_accuracy_rows(path)
    if not rows:
        return
    headers = ["prefix", *(f"acc_{split_name}" for split_name in eval_splits)]
    widths = {
        header: max(len(header), *(len(str(row.get(header, ""))) for row in rows))
        for header in headers
    }
    print("\nAccuracy:")
    print("  ".join(header.ljust(widths[header]) for header in headers))
    print("  ".join("-" * widths[header] for header in headers))
    for row in rows:
        print("  ".join(str(row.get(header, "")).ljust(widths[header]) for header in headers))


def main() -> None:
    args = parse_args()
    sweep = load_sweep_config(args.config)
    sweep.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {ad_safe.DEVICE}")
    print(f"Config: {sweep.config_path}")
    print(f"Backbone: {sweep.backbone}")
    print(f"Output dir: {sweep.output_dir}")
    print(f"Train split: {sweep.train_split}")
    print(f"Eval splits: {', '.join(sweep.eval_splits)}")
    print(f"Resume: {sweep.resume}")
    print(f"Force: {sweep.force}")
    print(f"Cooldown: {sweep.cooldown.to_json()}")

    full_train_dataset = ad_safe.load_dataset(sweep.train_split)
    max_eval_batch_size = max(phase.config.batch_size for job in sweep.jobs for phase in job)
    split_loaders = load_eval_loaders(
        split_names=sweep.eval_splits,
        batch_size=max_eval_batch_size,
        preloaded_datasets={sweep.train_split: full_train_dataset},
    )
    accuracy_csv_path = sweep.output_dir / "accuracy.csv"

    for job in sweep.jobs:
        previous_model_path: Path | None = None
        for phase in job:
            previous_model_path = train_phase(
                sweep=sweep,
                phase=phase,
                previous_model_path=previous_model_path,
                full_train_dataset=full_train_dataset,
                split_loaders=split_loaders,
                accuracy_csv_path=accuracy_csv_path,
            )

    print_accuracy_table(accuracy_csv_path, sweep.eval_splits)


if __name__ == "__main__":
    main()
