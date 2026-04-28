from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from .metrics import (
    DEFAULT_PRINT_METRICS,
    LOWER_IS_BETTER_METRICS,
    METRIC_CSV_FIELDS,
    ClassificationMetrics,
)


@dataclass(frozen=True)
class MetricsMatrixRow:
    row_id: str
    metrics_by_dataset: dict[str, ClassificationMetrics]
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class MetricsCsvRow:
    metadata: dict[str, object]
    metrics_by_dataset: dict[str, ClassificationMetrics]


def metric_attr_name(metric_name: str) -> str:
    if metric_name == "acc":
        return "accuracy"
    for attr_name, csv_name in METRIC_CSV_FIELDS:
        if metric_name == csv_name:
            return attr_name
    raise ValueError(f"Unknown metric: {metric_name}")


def metric_csv_name(metric_name: str) -> str:
    if metric_name == "accuracy":
        return "acc"
    for attr_name, csv_name in METRIC_CSV_FIELDS:
        if metric_name == attr_name or metric_name == csv_name:
            return csv_name
    raise ValueError(f"Unknown metric: {metric_name}")


def metric_value(metrics: ClassificationMetrics, metric_name: str) -> float | None:
    return getattr(metrics, metric_attr_name(metric_name))


def format_metric(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{value:.4f}"


def format_csv_metric(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return ""
    return f"{float(value):.8f}"


def make_metrics_cell(
    metrics: ClassificationMetrics,
    *,
    metric_names: Sequence[str] = DEFAULT_PRINT_METRICS,
) -> str:
    return "\n".join(
        f"{metric_csv_name(metric_name)}={format_metric(metric_value(metrics, metric_name))}"
        for metric_name in metric_names
    )


def print_metrics_matrix(
    rows: Sequence[MetricsMatrixRow],
    dataset_names: Sequence[str],
    *,
    row_header: str = "model",
    title: str = "Results",
    metric_names: Sequence[str] = DEFAULT_PRINT_METRICS,
) -> None:
    if not rows:
        return
    headers = [row_header, *dataset_names]
    table_rows = [
        [
            row.row_id,
            *(
                make_metrics_cell(row.metrics_by_dataset[dataset_name], metric_names=metric_names)
                for dataset_name in dataset_names
            ),
        ]
        for row in rows
    ]
    widths = {
        header: max(
            len(header),
            *(
                max(len(line) for line in row[index].splitlines())
                for row in table_rows
            ),
        )
        for index, header in enumerate(headers)
    }

    print(f"\n{title}")
    print("  ".join(header.ljust(widths[header]) for header in headers))
    print("  ".join("-" * widths[header] for header in headers))
    for row in table_rows:
        cell_lines = [value.splitlines() for value in row]
        row_height = max(len(lines) for lines in cell_lines)
        for line_index in range(row_height):
            print(
                "  ".join(
                    (
                        cell_lines[index][line_index]
                        if line_index < len(cell_lines[index])
                        else ""
                    ).ljust(widths[headers[index]])
                    for index in range(len(headers))
                )
            )
        print()


def flatten_metrics(
    metrics_by_dataset: dict[str, ClassificationMetrics],
    dataset_names: Sequence[str] | None = None,
) -> dict[str, float | None]:
    dataset_order = dataset_names or tuple(metrics_by_dataset)
    row: dict[str, float | None] = {}
    for dataset_name in dataset_order:
        metric_values = metrics_by_dataset[dataset_name].to_json_dict()
        for attr_name, csv_name in METRIC_CSV_FIELDS:
            row[f"{csv_name}_{dataset_name}"] = metric_values[attr_name]
    return row


def metrics_csv_fieldnames(
    *,
    metadata_fields: Sequence[str],
    dataset_names: Sequence[str],
) -> list[str]:
    return [
        *metadata_fields,
        *[
            f"{csv_name}_{dataset_name}"
            for dataset_name in dataset_names
            for _, csv_name in METRIC_CSV_FIELDS
        ],
    ]


def write_metrics_csv_rows(
    *,
    path: Path,
    rows: Sequence[MetricsCsvRow],
    dataset_names: Sequence[str],
    metadata_fields: Sequence[str],
    sort_metadata_field: str | None = None,
) -> None:
    path = Path(path)
    fieldnames = metrics_csv_fieldnames(
        metadata_fields=metadata_fields,
        dataset_names=dataset_names,
    )
    serialized_rows = []
    for row in rows:
        flat_metrics = flatten_metrics(row.metrics_by_dataset, dataset_names)
        serialized_rows.append(
            {
                **{field: "" for field in fieldnames},
                **{key: value for key, value in row.metadata.items() if key in fieldnames},
                **{
                    key: format_csv_metric(value)
                    for key, value in flat_metrics.items()
                    if key in fieldnames
                },
            }
        )
    if sort_metadata_field is not None:
        serialized_rows.sort(key=lambda row: str(row.get(sort_metadata_field, "")))

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(serialized_rows)
    print(f"Metrics CSV saved to {path}")


def parse_optional_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def classification_metrics_from_mapping(values: dict[str, object]) -> ClassificationMetrics:
    def read_metric(attr_name: str) -> float | None:
        csv_name = metric_csv_name(attr_name)
        return parse_optional_float(values.get(attr_name, values.get(csv_name)))

    accuracy = read_metric("accuracy")
    nll = read_metric("nll")
    avg_conf = read_metric("avg_conf")
    avg_margin = read_metric("avg_margin")
    avg_correct_conf = read_metric("avg_correct_conf")
    if accuracy is None:
        raise ValueError("Metrics row is missing accuracy")
    if nll is None:
        raise ValueError("Metrics row is missing nll")
    if avg_conf is None:
        raise ValueError("Metrics row is missing avg_conf")
    if avg_margin is None:
        raise ValueError("Metrics row is missing avg_margin")
    if avg_correct_conf is None:
        raise ValueError("Metrics row is missing avg_correct_conf")

    return ClassificationMetrics(
        accuracy=accuracy,
        auc=read_metric("auc"),
        nll=nll,
        avg_conf=avg_conf,
        avg_margin=avg_margin,
        avg_correct_conf=avg_correct_conf,
        avg_wrong_conf=read_metric("avg_wrong_conf"),
        safe_recall=read_metric("safe_recall"),
        unsafe_recall=read_metric("unsafe_recall"),
    )


def metrics_from_flat_csv_row(
    row: dict[str, object],
    dataset_names: Sequence[str],
) -> dict[str, ClassificationMetrics]:
    metrics_by_dataset: dict[str, ClassificationMetrics] = {}
    for dataset_name in dataset_names:
        values = {
            csv_name: row.get(f"{csv_name}_{dataset_name}")
            for _, csv_name in METRIC_CSV_FIELDS
        }
        metrics_by_dataset[dataset_name] = classification_metrics_from_mapping(values)
    return metrics_by_dataset


def sort_metrics_matrix_rows(
    rows: list[MetricsMatrixRow],
    sort_key: str,
) -> None:
    if sort_key == "name":
        rows.sort(key=lambda row: row.row_id)
        return

    metric_name, separator, dataset_name = sort_key.rpartition("_")
    if not separator:
        raise ValueError(f"Invalid metric sort key: {sort_key}")

    def key(row: MetricsMatrixRow) -> tuple[bool, float]:
        metrics = row.metrics_by_dataset.get(dataset_name)
        value = None if metrics is None else metric_value(metrics, metric_name)
        if value is None or not math.isfinite(value):
            return (True, 0.0)
        numeric_value = float(value)
        if metric_csv_name(metric_name) not in LOWER_IS_BETTER_METRICS:
            numeric_value = -numeric_value
        return (False, numeric_value)

    rows.sort(key=key)
