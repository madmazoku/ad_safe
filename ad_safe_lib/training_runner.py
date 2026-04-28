from __future__ import annotations

import csv
import traceback
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Sequence

from torch.utils.data import DataLoader, Dataset

from .artifacts import (
    json_ready,
    load_json_file,
    load_model,
    path_to_json,
    release_torch_memory,
    save_model,
    write_json_file,
    write_setup_file,
)
from .backbones import configure_trainable_layers, describe_model, finalize_training_config, make_model
from .config import DATA_DIR, DEVICE, TrainingConfig, config_to_json_dict, resolve_effective_seed, set_seed
from .contract import run_foreign_contract_check
from .cooldown import CooldownConfig, CooldownEpochEndHandler, EpochEndHandler
from .data import DatasetSourceSpec, load_dataset_source, make_data_loader
from .enrichment import EnrichmentJobSpec
from .evaluation_runner import evaluate_model_checkpoint
from .figures import generate_training_history_figure, save_figure
from .metrics import ClassificationMetrics
from .reporting import (
    MetricsCsvRow,
    MetricsMatrixRow,
    classification_metrics_from_mapping,
    metrics_from_flat_csv_row,
    print_metrics_matrix,
    write_metrics_csv_rows,
)
from .training import train_model_across_resplits
from .training import build_teacher_logits_cache_key


@dataclass(frozen=True)
class PhaseSpec:
    job_index: int
    phase_index: int
    prefix: str
    config: TrainingConfig
    requested_seed: int | None = None
    unfreeze_all: bool = False
    unfreeze_top: int = 0
    title: str | None = None
    signature: dict[str, Any] = field(default_factory=dict)
    enrichment_jobs: tuple[EnrichmentJobSpec, ...] = ()
    enrichment_jobs_payload: tuple[dict[str, Any], ...] = ()
    model_filename: str | None = None
    history_filename: str | None = None
    json_filename: str | None = None

    def display_title(self) -> str:
        return self.title or self.prefix


@dataclass(frozen=True)
class JobSpec:
    job_index: int
    job_id: str
    backbone: str
    phases: tuple[PhaseSpec, ...]
    title: str | None = None
    initial_model_path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    enrichment_jobs: tuple[EnrichmentJobSpec, ...] = ()

    def display_title(self) -> str:
        return self.title or self.job_id


@dataclass(frozen=True)
class RunPlan:
    output_dir: Path
    run_id: str
    train_split: str
    eval_splits: tuple[str, ...]
    jobs: tuple[JobSpec, ...]
    cooldown: CooldownConfig = CooldownConfig()
    resume: bool = True
    force: bool = False
    source_config_path: Path | None = None
    setup_path: Path | None = None
    metrics_csv_path: Path | None = None
    check_foreign_contract: bool = True
    train_source: DatasetSourceSpec | None = None
    eval_sources: dict[str, DatasetSourceSpec] = field(default_factory=dict)
    train_dataset: Dataset | None = None
    eval_datasets: dict[str, Dataset] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    teacher_logits_cache: dict[tuple[Any, ...], Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainingPhaseResult:
    job_id: str
    job_index: int
    phase_index: int
    prefix: str
    phase_title: str
    config: TrainingConfig
    model_path: Path
    json_path: Path
    history_path: Path
    metrics_by_split: dict[str, ClassificationMetrics]
    skipped: bool = False


@dataclass(frozen=True)
class TrainingRunResult:
    phase_results: tuple[TrainingPhaseResult, ...]
    metrics_csv_path: Path


def dataset_source_to_json(source: DatasetSourceSpec | None) -> dict[str, object] | None:
    return None if source is None else source.to_json()


def eval_sources_to_json(sources: dict[str, DatasetSourceSpec]) -> dict[str, dict[str, object]]:
    return {
        split_name: source.to_json()
        for split_name, source in sources.items()
    }


def phase_model_path(plan: RunPlan, phase: PhaseSpec) -> Path:
    return plan.output_dir / (phase.model_filename or f"{phase.prefix}.pt")


def phase_history_path(plan: RunPlan, phase: PhaseSpec) -> Path:
    return plan.output_dir / (phase.history_filename or f"{phase.prefix}.png")


def phase_json_path(plan: RunPlan, phase: PhaseSpec) -> Path:
    return plan.output_dir / (phase.json_filename or f"{phase.prefix}.json")


def build_phase_signature(
    *,
    plan: RunPlan,
    job: JobSpec,
    phase: PhaseSpec,
    config: TrainingConfig,
    seed: int,
) -> dict[str, Any]:
    return {
        "run_id": plan.run_id,
        "train_split": plan.train_split,
        "eval_splits": list(plan.eval_splits),
        "train_source": dataset_source_to_json(plan.train_source),
        "eval_sources": eval_sources_to_json(plan.eval_sources),
        "train_dataset_size": len(plan.train_dataset) if plan.train_dataset is not None else None,
        "eval_dataset_sizes": {
            split_name: len(dataset)
            for split_name, dataset in plan.eval_datasets.items()
        },
        "job_index": job.job_index,
        "job_id": job.job_id,
        "backbone": job.backbone,
        "phase_index": phase.phase_index,
        "phase_prefix": phase.prefix,
        "phase_title": phase.display_title(),
        "effective_seed": seed,
        "requested_seed": phase.requested_seed,
        "unfreeze_all": phase.unfreeze_all,
        "unfreeze_top": phase.unfreeze_top,
        "training_config": config_to_json_dict(config),
        "enrichment_jobs": list(phase.enrichment_jobs_payload),
        "extra": json_ready(phase.signature),
    }


def metrics_objects_to_json(
    metrics_by_split: dict[str, ClassificationMetrics] | None,
) -> dict[str, dict[str, float | None]]:
    if metrics_by_split is None:
        return {}
    return {
        split_name: metrics.to_json_dict()
        for split_name, metrics in metrics_by_split.items()
    }


def metrics_objects_from_json(
    payload: object,
    split_names: Sequence[str],
) -> dict[str, ClassificationMetrics] | None:
    if not isinstance(payload, dict):
        return None
    metrics_by_split: dict[str, ClassificationMetrics] = {}
    for split_name in split_names:
        split_metrics = payload.get(split_name)
        if not isinstance(split_metrics, dict):
            return None
        metrics_by_split[split_name] = classification_metrics_from_mapping(split_metrics)
    return metrics_by_split


def build_training_phase_payload(
    *,
    plan: RunPlan,
    job: JobSpec,
    phase: PhaseSpec,
    status: str,
    seed: int,
    config: TrainingConfig,
    model_path: Path,
    history_path: Path,
    previous_model_path: Path | None,
    metrics_by_split: dict[str, ClassificationMetrics] | None = None,
    error: str | None = None,
    error_traceback: str | None = None,
) -> dict[str, Any]:
    metrics_json = metrics_objects_to_json(metrics_by_split)
    return {
        "timestamp": plan.run_id,
        "seed": seed,
        "train_split": plan.train_split,
        "eval_split": plan.eval_splits[0] if plan.eval_splits else None,
        "eval_splits": list(plan.eval_splits),
        "train_source": dataset_source_to_json(plan.train_source),
        "train_fraction": plan.train_source.fraction if plan.train_source is not None else 1.0,
        "eval_sources": eval_sources_to_json(plan.eval_sources),
        "train_dataset_size": len(plan.train_dataset) if plan.train_dataset is not None else None,
        "eval_dataset_sizes": {
            split_name: len(dataset)
            for split_name, dataset in plan.eval_datasets.items()
        },
        "base_model": job.backbone,
        "original_model_path": path_to_json(previous_model_path),
        "training_checkpoint_path": path_to_json(model_path),
        "training_history_figure_path": path_to_json(history_path),
        "training_config": config_to_json_dict(config),
        "enrichment_jobs": list(phase.enrichment_jobs_payload),
        "status": status,
        "job_id": job.job_id,
        "job_index": job.job_index,
        "phase_index": phase.phase_index,
        "phase_prefix": phase.prefix,
        "phase_title": phase.display_title(),
        "output_model_path": path_to_json(model_path),
        "accuracy": {
            split_name: metrics.accuracy
            for split_name, metrics in (metrics_by_split or {}).items()
        },
        "metrics": metrics_json,
        "error": error,
        "error_traceback": error_traceback,
        "run": {
            "run_id": plan.run_id,
            "output_dir": path_to_json(plan.output_dir),
            "source_config_path": path_to_json(plan.source_config_path),
            "resume": plan.resume,
            "force": plan.force,
            "cooldown": plan.cooldown.to_json(),
            "metadata": plan.metadata,
        },
        "run_signature": build_phase_signature(
            plan=plan,
            job=job,
            phase=phase,
            config=config,
            seed=seed,
        ),
    }


def read_training_metrics_csv_rows(
    path: Path,
    eval_splits: Sequence[str],
) -> list[MetricsCsvRow]:
    if not path.exists():
        return []
    rows: list[MetricsCsvRow] = []
    with path.open(newline="") as csv_file:
        for row in csv.DictReader(csv_file):
            metrics_by_split = metrics_from_flat_csv_row(row, eval_splits)
            metadata = {
                key: value
                for key, value in row.items()
                if not any(key.endswith(f"_{split_name}") for split_name in eval_splits)
            }
            rows.append(MetricsCsvRow(metadata=metadata, metrics_by_dataset=metrics_by_split))
    return rows


def training_metrics_metadata_fields() -> tuple[str, ...]:
    return ("row_id", "job_id", "phase_title", "model_path")


def update_training_metrics_csv(
    *,
    path: Path,
    result: TrainingPhaseResult,
    eval_splits: Sequence[str],
) -> None:
    existing_rows = [
        row
        for row in read_training_metrics_csv_rows(path, eval_splits)
        if row.metadata.get("row_id") != result.prefix
    ]
    existing_rows.append(
        MetricsCsvRow(
            metadata={
                "row_id": result.prefix,
                "job_id": result.job_id,
                "phase_title": result.phase_title,
                "model_path": path_to_json(result.model_path),
            },
            metrics_by_dataset=result.metrics_by_split,
        )
    )
    write_metrics_csv_rows(
        path=path,
        rows=existing_rows,
        dataset_names=eval_splits,
        metadata_fields=training_metrics_metadata_fields(),
        sort_metadata_field="row_id",
    )


def training_results_to_matrix_rows(
    results: Sequence[TrainingPhaseResult],
) -> list[MetricsMatrixRow]:
    return [
        MetricsMatrixRow(
            row_id=result.prefix,
            metrics_by_dataset=result.metrics_by_split,
            metadata={
                "job_id": result.job_id,
                "phase_title": result.phase_title,
                "model_path": path_to_json(result.model_path),
            },
        )
        for result in results
    ]


def metrics_matrix_rows_from_training_csv(
    path: Path,
    eval_splits: Sequence[str],
) -> list[MetricsMatrixRow]:
    return [
        MetricsMatrixRow(
            row_id=str(row.metadata.get("row_id", "")),
            metrics_by_dataset=row.metrics_by_dataset,
            metadata=row.metadata,
        )
        for row in read_training_metrics_csv_rows(path, eval_splits)
    ]


def write_training_run_setup(
    *,
    plan: RunPlan,
    results: Sequence[TrainingPhaseResult] = (),
) -> None:
    if plan.setup_path is None:
        return

    payload = {
        "timestamp": plan.run_id,
        "device": str(DEVICE),
        "dataset_dir": path_to_json(DATA_DIR),
        "train_split": plan.train_split,
        "eval_splits": list(plan.eval_splits),
        "train_source": dataset_source_to_json(plan.train_source),
        "train_fraction": plan.train_source.fraction if plan.train_source is not None else 1.0,
        "eval_sources": eval_sources_to_json(plan.eval_sources),
        "train_dataset_size": len(plan.train_dataset) if plan.train_dataset is not None else None,
        "eval_dataset_sizes": {
            split_name: len(dataset)
            for split_name, dataset in plan.eval_datasets.items()
        },
        "output_dir": path_to_json(plan.output_dir),
        "source_config_path": path_to_json(plan.source_config_path),
        "resume": plan.resume,
        "force": plan.force,
        "metrics_csv_path": path_to_json(plan.metrics_csv_path or (plan.output_dir / "accuracy.csv")),
        "cooldown": plan.cooldown.to_json(),
        "metadata": plan.metadata,
        "jobs": [
            {
                "job_id": job.job_id,
                "job_index": job.job_index,
                "title": job.title,
                "backbone": job.backbone,
                "initial_model_path": path_to_json(job.initial_model_path),
                "metadata": job.metadata,
                "phases": [
                    {
                        "phase_index": phase.phase_index,
                        "prefix": phase.prefix,
                        "title": phase.title,
                        "requested_seed": phase.requested_seed,
                        "unfreeze_all": phase.unfreeze_all,
                        "unfreeze_top": phase.unfreeze_top,
                        "training_config": config_to_json_dict(phase.config),
                        "enrichment_jobs": list(phase.enrichment_jobs_payload),
                        "signature": phase.signature,
                    }
                    for phase in job.phases
                ],
            }
            for job in plan.jobs
        ],
        "results": [
            {
                "job_id": result.job_id,
                "job_index": result.job_index,
                "phase_index": result.phase_index,
                "prefix": result.prefix,
                "phase_title": result.phase_title,
                "model_path": path_to_json(result.model_path),
                "history_path": path_to_json(result.history_path),
                "json_path": path_to_json(result.json_path),
                "training_config": config_to_json_dict(result.config),
                "metrics": metrics_objects_to_json(result.metrics_by_split),
                "skipped": result.skipped,
            }
            for result in results
        ],
    }
    if len(plan.jobs) == 1 and len(plan.jobs[0].phases) == 1:
        job = plan.jobs[0]
        phase = job.phases[0]
        result = results[-1] if results else None
        setup_config = result.config if result is not None else phase.config
        payload.update(
            {
                "seed": phase.requested_seed,
                "eval_split": plan.eval_splits[0],
                "base_model": setup_config.base_model,
                "original_model_path": path_to_json(job.initial_model_path),
                "training_checkpoint_path": path_to_json(
                    result.model_path if result is not None else phase_model_path(plan, phase)
                ),
                "training_history_figure_path": path_to_json(
                    result.history_path if result is not None else phase_history_path(plan, phase)
                ),
                "training_config": config_to_json_dict(setup_config),
            }
        )
    write_setup_file(payload, plan.setup_path)


def load_eval_loaders_for_splits(
    *,
    split_names: Sequence[str],
    batch_size: int,
    preloaded_datasets: dict[str, Dataset],
    source_specs: dict[str, DatasetSourceSpec],
) -> dict[str, DataLoader]:
    split_loaders: dict[str, DataLoader] = {}
    for split_name in split_names:
        dataset = preloaded_datasets.get(split_name)
        if dataset is None:
            dataset = load_dataset_source(
                source_specs.get(split_name, DatasetSourceSpec(name=split_name))
            )
        split_loaders[split_name], = make_data_loader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
        )
    return split_loaders


def should_skip_training_phase(
    *,
    plan: RunPlan,
    signature: dict[str, Any],
    model_path: Path,
    json_path: Path,
) -> tuple[bool, dict[str, ClassificationMetrics] | None]:
    if plan.force or not plan.resume:
        return False, None
    existing = load_json_file(json_path)
    if existing is None or not model_path.exists():
        return False, None
    existing_signature = existing.get("run_signature")
    if existing.get("status") == "completed" and existing_signature == json_ready(signature):
        metrics = metrics_objects_from_json(existing.get("metrics"), plan.eval_splits)
        if metrics is None:
            return False, None
        print(f"Skipping completed phase {json_path.stem}")
        return True, metrics
    if existing.get("status") == "completed" and existing_signature != json_ready(signature):
        raise ValueError(
            f"Existing completed artifact {json_path} does not match the current config. "
            "Use a new run_id/output directory or force=true to retrain."
        )
    return False, None


def run_training_phase(
    *,
    plan: RunPlan,
    job: JobSpec,
    phase: PhaseSpec,
    previous_model_path: Path | None,
    full_train_dataset: Dataset,
    split_loaders: dict[str, DataLoader],
    metrics_csv_path: Path,
) -> TrainingPhaseResult:
    model_path = phase_model_path(plan, phase)
    json_path = phase_json_path(plan, phase)
    history_path = phase_history_path(plan, phase)

    print(f"\n=== Phase {phase.prefix} ===")
    if phase.title:
        print(f"Phase title: {phase.title}")
    seed = resolve_effective_seed(phase.requested_seed)
    set_seed(seed)
    model = load_model(previous_model_path) if previous_model_path is not None else make_model(job.backbone)
    config = finalize_training_config(
        model,
        phase.config,
        unfreeze_all=phase.unfreeze_all,
        unfreeze_top=phase.unfreeze_top,
    )
    signature = build_phase_signature(
        plan=plan,
        job=job,
        phase=phase,
        config=config,
        seed=seed,
    )

    skipped, skipped_metrics = should_skip_training_phase(
        plan=plan,
        signature=signature,
        model_path=model_path,
        json_path=json_path,
    )
    if skipped and skipped_metrics is not None:
        result = TrainingPhaseResult(
            job_id=job.job_id,
            job_index=job.job_index,
            phase_index=phase.phase_index,
            prefix=phase.prefix,
            phase_title=phase.display_title(),
            config=config,
            model_path=model_path,
            json_path=json_path,
            history_path=history_path,
            metrics_by_split=skipped_metrics,
            skipped=True,
        )
        update_training_metrics_csv(
            path=metrics_csv_path,
            result=result,
            eval_splits=plan.eval_splits,
        )
        del model
        release_torch_memory()
        return result

    configure_trainable_layers(model, unfreeze=config.unfreeze)
    describe_model(model, config)

    write_json_file(
        json_path,
        build_training_phase_payload(
            plan=plan,
            job=job,
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
        save_model(model, model_path)
        epoch_end_handlers: tuple[EpochEndHandler, ...] = ()
        if plan.cooldown.enabled:
            epoch_end_handlers = (
                CooldownEpochEndHandler(
                    config=plan.cooldown,
                    backbone_name=job.backbone,
                    phase_title=phase.display_title(),
                ),
            )
        model, history = train_model_across_resplits(
            model=model,
            full_train_dataset=full_train_dataset,
            config=config,
            best_model_path=model_path,
            split_log_prefix=f"{job.display_title()} {phase.display_title()}",
            epoch_end_handlers=epoch_end_handlers,
            enrichment_jobs=phase.enrichment_jobs,
            teacher_model=None,
            replay_seed=seed,
            teacher_logits_cache=plan.teacher_logits_cache,
            teacher_logits_cache_key=build_teacher_logits_cache_key(
                teacher_model_path=config.teacher_model_path,
                train_source=plan.train_source,
                dataset=full_train_dataset,
            ),
        )
        save_model(model, model_path)
        save_figure(
            generate_training_history_figure(
                history=history,
                config=config,
                train_split=plan.train_split,
                seed=seed,
                input_model_path=previous_model_path,
                output_model_path=model_path,
            ),
            history_path,
        )
        del model
        release_torch_memory()

        if plan.check_foreign_contract:
            run_foreign_contract_check(model_path)
        metrics_by_split = evaluate_model_checkpoint(
            model_path=model_path,
            split_loaders=split_loaders,
        )
        write_json_file(
            json_path,
            build_training_phase_payload(
                plan=plan,
                job=job,
                phase=phase,
                status="completed",
                seed=seed,
                config=config,
                model_path=model_path,
                history_path=history_path,
                previous_model_path=previous_model_path,
                metrics_by_split=metrics_by_split,
            ),
        )
        result = TrainingPhaseResult(
            job_id=job.job_id,
            job_index=job.job_index,
            phase_index=phase.phase_index,
            prefix=phase.prefix,
                phase_title=phase.display_title(),
            config=config,
            model_path=model_path,
            json_path=json_path,
            history_path=history_path,
            metrics_by_split=metrics_by_split,
            skipped=False,
        )
        update_training_metrics_csv(
            path=metrics_csv_path,
            result=result,
            eval_splits=plan.eval_splits,
        )
        return result
    except Exception as exc:
        error_traceback = traceback.format_exc()
        write_json_file(
            json_path,
            build_training_phase_payload(
                plan=plan,
                job=job,
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


def run_training_plan(plan: RunPlan) -> TrainingRunResult:
    if not plan.jobs:
        raise ValueError("RunPlan must contain at least one job")
    if not plan.eval_splits:
        raise ValueError("RunPlan must contain at least one eval split")

    plan.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Using device: {DEVICE}")
    print(f"Run id: {plan.run_id}")
    print(f"Output dir: {plan.output_dir}")
    print(f"Train split: {plan.train_split}")
    print(f"Eval splits: {', '.join(plan.eval_splits)}")
    print(f"Resume: {plan.resume}")
    print(f"Force: {plan.force}")
    print(f"Cooldown: {plan.cooldown.to_json()}")

    max_batch_size = max(phase.config.batch_size for job in plan.jobs for phase in job.phases)
    full_train_dataset = plan.train_dataset
    if full_train_dataset is None:
        full_train_dataset = load_dataset_source(
            plan.train_source or DatasetSourceSpec(name=plan.train_split)
        )
    else:
        print(f"Using provided train dataset: {len(full_train_dataset)} samples")
    preloaded_eval_datasets = {
        plan.train_split: full_train_dataset,
        **plan.eval_datasets,
    }
    split_loaders = load_eval_loaders_for_splits(
        split_names=plan.eval_splits,
        batch_size=max_batch_size,
        preloaded_datasets=preloaded_eval_datasets,
        source_specs=plan.eval_sources,
    )
    execution_plan = replace(
        plan,
        train_dataset=full_train_dataset,
        eval_datasets={
            **plan.eval_datasets,
            **{
                split_name: split_loaders[split_name].dataset
                for split_name in plan.eval_splits
                if split_name not in plan.eval_datasets
            },
        },
        teacher_logits_cache={},
    )
    metrics_csv_path = plan.metrics_csv_path or (plan.output_dir / "accuracy.csv")
    phase_results: list[TrainingPhaseResult] = []
    write_training_run_setup(plan=execution_plan, results=phase_results)

    for job in execution_plan.jobs:
        print(f"\n=== Job {job.job_id} ===")
        if job.title:
            print(f"Job title: {job.title}")
        previous_model_path = job.initial_model_path
        for phase in job.phases:
            result = run_training_phase(
                plan=execution_plan,
                job=job,
                phase=phase,
                previous_model_path=previous_model_path,
                full_train_dataset=full_train_dataset,
                split_loaders=split_loaders,
                metrics_csv_path=metrics_csv_path,
            )
            phase_results.append(result)
            previous_model_path = result.model_path
            write_training_run_setup(plan=execution_plan, results=phase_results)

    matrix_rows = metrics_matrix_rows_from_training_csv(metrics_csv_path, execution_plan.eval_splits)
    print_metrics_matrix(
        matrix_rows,
        execution_plan.eval_splits,
        row_header="prefix",
        title="Results",
    )
    return TrainingRunResult(
        phase_results=tuple(phase_results),
        metrics_csv_path=metrics_csv_path,
    )
