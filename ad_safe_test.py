#!/usr/bin/env python3

import argparse
import csv
import math
from datetime import datetime
from pathlib import Path
from typing import Sequence

import ad_safe


METRIC_NAMES = (
    "acc",
    "auc",
    "nll",
    "avg_conf",
    "avg_margin",
    "avg_correct_conf",
    "avg_wrong_conf",
    "safe_recall",
    "unsafe_recall",
)
LOWER_IS_BETTER_METRICS = {"nll", "avg_wrong_conf"}
PRINT_METRICS = ("acc", "auc", "nll", "avg_wrong_conf")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate all saved ad-safety models on one or more dataset splits."
    )
    parser.add_argument(
        "dataset",
        nargs="+",
        help="Dataset split folder(s) to evaluate against, e.g. 'test', 'val,test', or 'train val test'",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for evaluation; defaults to ad_safe heuristic",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on how many newest models to evaluate",
    )
    parser.add_argument(
        "--model-path",
        default="*.pt",
        help="Model path or glob pattern, e.g. '*-fx.pt' or 'challenge/*-fx.pt'",
    )
    parser.add_argument(
        "--sort",
        default=None,
        help=(
            "How to order printed results: 'name' or '<metric>_<dataset>', "
            "e.g. 'acc_test', 'auc_val', or 'nll_test'"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ad_safe.SCRIPT_DIR,
        help="Directory where the timestamp-prefixed metrics CSV will be written",
    )
    return parser.parse_args()


def expand_model_path_pattern(pattern: str) -> list[Path]:
    path = Path(pattern)
    if path.is_absolute():
        parent = path.parent
        name_pattern = path.name
    elif path.parent == Path("."):
        parent = ad_safe.SCRIPT_DIR
        name_pattern = path.name
    else:
        parent = path.parent
        name_pattern = path.name

    if any(character in name_pattern for character in "*?["):
        return sorted(parent.glob(name_pattern))
    return [path if path.is_absolute() else ad_safe.SCRIPT_DIR / path]


def find_model_paths(model_path_arg: str, limit: int | None) -> list[Path]:
    model_paths: list[Path] = []
    for pattern in (part.strip() for part in model_path_arg.split(",") if part.strip()):
        model_paths.extend(expand_model_path_pattern(pattern))

    model_paths = sorted(dict.fromkeys(model_paths))
    if not model_paths:
        raise FileNotFoundError(
            f"No model checkpoints matching '{model_path_arg}' found in {ad_safe.SCRIPT_DIR}"
        )
    missing_paths = [path for path in model_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(
            "Model path(s) do not exist: "
            + ", ".join(str(path) for path in missing_paths)
        )
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be positive")
        model_paths = model_paths[-limit:]
    return model_paths


def parse_dataset_names(dataset_arg: Sequence[str]) -> tuple[str, ...]:
    dataset_names = tuple(
        dataset_name.strip()
        for dataset_part in dataset_arg
        for dataset_name in dataset_part.split(",")
        if dataset_name.strip()
    )
    if not dataset_names:
        raise ValueError("dataset must contain at least one split name")

    invalid_names = [name for name in dataset_names if name not in {"train", "val", "test"}]
    if invalid_names:
        raise ValueError(
            f"Unknown dataset split(s): {', '.join(invalid_names)}. Expected train, val, or test"
        )
    return dataset_names


def resolve_sort_key(sort_arg: str | None, dataset_names: tuple[str, ...]) -> str:
    if sort_arg is None:
        return f"acc_{dataset_names[0]}"
    if sort_arg == "name":
        return sort_arg
    metric_name, separator, dataset_name = sort_arg.rpartition("_")
    if separator and metric_name in METRIC_NAMES and dataset_name in dataset_names:
        return sort_arg
    valid_keys = [
        "name",
        *(f"{metric_name}_{dataset_name}" for metric_name in METRIC_NAMES for dataset_name in dataset_names),
    ]
    raise ValueError(f"--sort must be one of: {', '.join(valid_keys)}")


def format_metric(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{value:.4f}"


def flatten_metrics(metrics_by_dataset: dict[str, ad_safe.ClassificationMetrics]) -> dict[str, float | None]:
    row: dict[str, float | None] = {}
    for dataset_name, metrics in metrics_by_dataset.items():
        metric_values = metrics.to_json_dict()
        row[f"acc_{dataset_name}"] = metric_values["accuracy"]
        for metric_name in METRIC_NAMES:
            if metric_name == "acc":
                continue
            row[f"{metric_name}_{dataset_name}"] = metric_values[metric_name]
    return row


def sort_results(
    results: list[tuple[Path, dict[str, ad_safe.ClassificationMetrics]]],
    sort_key: str,
) -> None:
    if sort_key == "name":
        results.sort(key=lambda item: item[0].name)
        return

    metric_name, _, dataset_name = sort_key.rpartition("_")
    def key(item: tuple[Path, dict[str, ad_safe.ClassificationMetrics]]) -> tuple[bool, float]:
        value = flatten_metrics(item[1]).get(sort_key)
        if value is None or not math.isfinite(value):
            return (True, 0.0)
        numeric_value = float(value)
        if metric_name not in LOWER_IS_BETTER_METRICS:
            numeric_value = -numeric_value
        return (False, numeric_value)

    results.sort(key=key)


def make_cell(metrics: ad_safe.ClassificationMetrics) -> str:
    values = {
        "acc": metrics.accuracy,
        "auc": metrics.auc,
        "nll": metrics.nll,
        "avg_conf": metrics.avg_conf,
        "avg_margin": metrics.avg_margin,
        "avg_correct_conf": metrics.avg_correct_conf,
        "avg_wrong_conf": metrics.avg_wrong_conf,
        "safe_recall": metrics.safe_recall,
        "unsafe_recall": metrics.unsafe_recall,
    }
    return " ".join(f"{name}={format_metric(values[name])}" for name in PRINT_METRICS)


def print_results_matrix(
    results: list[tuple[Path, dict[str, ad_safe.ClassificationMetrics]]],
    dataset_names: tuple[str, ...],
) -> None:
    headers = ["model", *dataset_names]
    rows = [
        [model_path.name, *(make_cell(metrics_by_dataset[dataset_name]) for dataset_name in dataset_names)]
        for model_path, metrics_by_dataset in results
    ]
    widths = {
        header: max(len(header), *(len(row[index]) for row in rows))
        for index, header in enumerate(headers)
    }

    print("\nResults")
    print("  ".join(header.ljust(widths[header]) for header in headers))
    print("  ".join("-" * widths[header] for header in headers))
    for row in rows:
        print("  ".join(value.ljust(widths[headers[index]]) for index, value in enumerate(row)))


def write_metrics_csv(
    *,
    results: list[tuple[Path, dict[str, ad_safe.ClassificationMetrics]]],
    dataset_names: tuple[str, ...],
    output_dir: Path,
    timestamp: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{timestamp}-ad-safe-test-metrics.csv"
    metric_columns = [
        f"{metric_name}_{dataset_name}"
        for dataset_name in dataset_names
        for metric_name in METRIC_NAMES
    ]
    fieldnames = ["rank", "model_name", "model_path", *metric_columns]
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for rank, (model_path, metrics_by_dataset) in enumerate(results, start=1):
            flat = flatten_metrics(metrics_by_dataset)
            writer.writerow(
                {
                    "rank": rank,
                    "model_name": model_path.name,
                    "model_path": str(model_path.resolve()),
                    **{
                        column: "" if flat.get(column) is None else f"{float(flat[column]):.8f}"
                        for column in metric_columns
                    },
                }
            )
    return output_path


def main() -> None:
    args = parse_args()
    dataset_names = parse_dataset_names(args.dataset)
    sort_key = resolve_sort_key(args.sort, dataset_names)
    batch_size = (
        args.batch_size
        if args.batch_size is not None
        else ad_safe.get_default_batch_size()
    )
    if batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    datasets_and_loaders = {
        dataset_name: (
            dataset := ad_safe.load_dataset(dataset_name),
            ad_safe.make_data_loader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
            )[0],
        )
        for dataset_name in dataset_names
    }
    model_paths = find_model_paths(args.model_path, args.limit)

    print(f"Using device: {ad_safe.DEVICE}")
    print(f"Datasets: {', '.join(dataset_names)}")
    print(f"Batch size: {batch_size}")
    print(f"Model path: {args.model_path}")
    print(f"Models to evaluate: {len(model_paths)}")
    print(f"Sort: {sort_key}")

    results: list[tuple[Path, dict[str, ad_safe.ClassificationMetrics]]] = []
    for model_path in model_paths:
        print(f"\nEvaluating {model_path.name}...")
        model = ad_safe.load_model(model_path)
        model_results: dict[str, ad_safe.ClassificationMetrics] = {}
        for dataset_name, (_, data_loader) in datasets_and_loaders.items():
            metrics = ad_safe.evaluate_metrics(model, data_loader, dataset_name)
            model_results[dataset_name] = metrics
            print(f"{dataset_name}: {metrics}")
        results.append((model_path, model_results))

    sort_results(results, sort_key)
    print_results_matrix(results, dataset_names)
    csv_path = write_metrics_csv(
        results=results,
        dataset_names=dataset_names,
        output_dir=args.output_dir,
        timestamp=datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),
    )
    print(f"\nMetrics CSV saved to {csv_path}")


if __name__ == "__main__":
    main()
