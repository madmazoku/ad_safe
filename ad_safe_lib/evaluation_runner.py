from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from torch.utils.data import DataLoader, Dataset

from .artifacts import load_model, release_torch_memory
from .config import get_default_batch_size
from .data import DatasetSourceSpec, load_dataset, load_dataset_source, make_data_loader
from .metrics import DEFAULT_PRINT_METRICS, ClassificationMetrics, evaluate_metrics
from .reporting import (
    MetricsCsvRow,
    MetricsMatrixRow,
    print_metrics_matrix,
    sort_metrics_matrix_rows,
    write_metrics_csv_rows,
)


@dataclass(frozen=True)
class ModelEvalSpec:
    path: Path
    title: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.title is None:
            object.__setattr__(self, 'title', self.path.name)

    def display_title(self) -> str:
        return self.title or self.path.name


@dataclass(frozen=True)
class DatasetEvalSpec:
    name: str
    batch_size: int | None = None
    dataset: Dataset | None = None
    source: DatasetSourceSpec | None = None


@dataclass(frozen=True)
class EvaluationPlan:
    models: tuple[ModelEvalSpec, ...]
    datasets: tuple[DatasetEvalSpec, ...]
    output_dir: Path | None = None
    csv_filename: str | None = None
    sort_key: str | None = None
    write_csv: bool = True
    print_results: bool = True
    title: str = "Results"
    print_metrics: tuple[str, ...] = DEFAULT_PRINT_METRICS
    metadata_fields: tuple[str, ...] = ("rank", "model_name", "model_path")


@dataclass(frozen=True)
class EvaluationRunResult:
    rows: tuple[MetricsMatrixRow, ...]
    csv_path: Path | None


def evaluate_model_checkpoint(
    *,
    model_path: Path,
    split_loaders: dict[str, DataLoader],
) -> dict[str, ClassificationMetrics]:
    model = load_model(model_path)
    metrics_by_split: dict[str, ClassificationMetrics] = {}
    for split_name, split_loader in split_loaders.items():
        metrics_by_split[split_name] = evaluate_metrics(
            model,
            split_loader,
            split_name,
        )
    del model
    release_torch_memory()
    return metrics_by_split


def run_evaluation_plan(plan: EvaluationPlan) -> EvaluationRunResult:
    if not plan.models:
        raise ValueError("EvaluationPlan must contain at least one model")
    if not plan.datasets:
        raise ValueError("EvaluationPlan must contain at least one dataset")

    dataset_loaders = {}
    for dataset_spec in plan.datasets:
        batch_size = dataset_spec.batch_size or get_default_batch_size()
        if batch_size <= 0:
            raise ValueError("Evaluation batch size must be positive")
        dataset = dataset_spec.dataset
        if dataset is None:
            dataset = (
                load_dataset_source(dataset_spec.source)
                if dataset_spec.source is not None
                else load_dataset(dataset_spec.name)
            )
        dataset_loaders[dataset_spec.name], = make_data_loader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
        )

    rows: list[MetricsMatrixRow] = []
    for model_spec in plan.models:
        print(f"\nEvaluating {model_spec.display_title()}...")
        model = load_model(model_spec.path)
        metrics_by_dataset: dict[str, ClassificationMetrics] = {}
        for dataset_spec in plan.datasets:
            metrics = evaluate_metrics(
                model,
                dataset_loaders[dataset_spec.name],
                dataset_spec.name,
            )
            metrics_by_dataset[dataset_spec.name] = metrics
            print(f"{dataset_spec.name}: {metrics}")
        rows.append(
            MetricsMatrixRow(
                row_id=model_spec.display_title(),
                metrics_by_dataset=metrics_by_dataset,
                metadata={
                    "model_name": model_spec.display_title(),
                    "model_path": str(model_spec.path.resolve()),
                    **model_spec.metadata,
                },
            )
        )
        del model
        release_torch_memory()

    if plan.sort_key is not None:
        sort_metrics_matrix_rows(rows, plan.sort_key)

    for rank, row in enumerate(rows, start=1):
        row.metadata["rank"] = rank

    dataset_names = tuple(dataset.name for dataset in plan.datasets)
    if plan.print_results:
        print_metrics_matrix(
            rows,
            dataset_names,
            row_header="model",
            title=plan.title,
            metric_names=plan.print_metrics,
        )

    csv_path = None
    if plan.write_csv:
        if plan.output_dir is None:
            raise ValueError("EvaluationPlan.output_dir is required when write_csv is enabled")
        csv_filename = plan.csv_filename or f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}-ad-safe-test-metrics.csv"
        csv_path = Path(plan.output_dir) / csv_filename
        write_metrics_csv_rows(
            path=csv_path,
            rows=[
                MetricsCsvRow(metadata=row.metadata, metrics_by_dataset=row.metrics_by_dataset)
                for row in rows
            ],
            dataset_names=dataset_names,
            metadata_fields=plan.metadata_fields,
        )

    return EvaluationRunResult(rows=tuple(rows), csv_path=csv_path)
