#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import ad_safe_lib as ad_safe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate saved ad-safety models on one or more dataset splits."
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
        "--dataset-fraction",
        type=float,
        default=1.0,
        help="Optional stratified fraction of each evaluation dataset to use",
    )
    parser.add_argument(
        "--dataset-seed",
        type=int,
        default=None,
        help="Seed for stratified evaluation subsets when --dataset-fraction is below 1",
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
        help="Model path or glob pattern. Bare names/globs default to artefacts/ad_safe_runs.",
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
        default=ad_safe.AD_SAFE_RUNS_DIR,
        help="Directory where the timestamp-prefixed metrics CSV will be written",
    )
    return parser.parse_args()


def validate_fraction(value: object, *, field_name: str) -> float:
    fraction = float(value)
    if fraction <= 0 or fraction > 1:
        raise ValueError(f"{field_name} must be in the range (0, 1]")
    return fraction


def discover_dataset_splits() -> list[str]:
    if not ad_safe.DATA_DIR.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {ad_safe.DATA_DIR}")
    return sorted(
        path.name
        for path in ad_safe.DATA_DIR.iterdir()
        if path.is_dir() and any(sample_path.is_file() for sample_path in path.glob("*/*"))
    )


def expand_model_path_pattern(pattern: str) -> list[Path]:
    path = Path(pattern)
    if path.is_absolute():
        parent = path.parent
        name_pattern = path.name
    elif path.parent == Path("."):
        parent = ad_safe.AD_SAFE_RUNS_DIR
        name_pattern = path.name
    else:
        parent = path.parent
        name_pattern = path.name

    if any(character in name_pattern for character in "*?["):
        return sorted(parent.glob(name_pattern))
    resolved = ad_safe.resolve_existing_path(path)
    return [resolved if resolved is not None else ad_safe.AD_SAFE_RUNS_DIR / path]


def find_model_paths(model_path_arg: str, limit: int | None) -> list[Path]:
    model_paths: list[Path] = []
    for pattern in (part.strip() for part in model_path_arg.split(",") if part.strip()):
        model_paths.extend(expand_model_path_pattern(pattern))

    model_paths = sorted(dict.fromkeys(model_paths))
    if not model_paths:
        raise FileNotFoundError(
            f"No model checkpoints matching '{model_path_arg}' found in {ad_safe.AD_SAFE_RUNS_DIR}"
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

    available = set(discover_dataset_splits())
    invalid_names = [name for name in dataset_names if name not in available]
    if invalid_names:
        raise ValueError(
            f"Unknown dataset split(s): {', '.join(invalid_names)}. "
            f"Available: {', '.join(sorted(available))}"
        )
    return tuple(dict.fromkeys(dataset_names))


def resolve_sort_key(sort_arg: str | None, dataset_names: tuple[str, ...]) -> str:
    if sort_arg is None:
        return f"acc_{dataset_names[0]}"
    if sort_arg == "name":
        return sort_arg
    metric_name, separator, dataset_name = sort_arg.rpartition("_")
    if separator and metric_name in ad_safe.METRIC_NAMES and dataset_name in dataset_names:
        return sort_arg
    valid_keys = [
        "name",
        *(f"{metric_name}_{dataset_name}" for metric_name in ad_safe.METRIC_NAMES for dataset_name in dataset_names),
    ]
    raise ValueError(f"--sort must be one of: {', '.join(valid_keys)}")


def build_evaluation_plan(args: argparse.Namespace) -> ad_safe.EvaluationPlan:
    dataset_names = parse_dataset_names(args.dataset)
    sort_key = resolve_sort_key(args.sort, dataset_names)
    dataset_fraction = validate_fraction(args.dataset_fraction, field_name="--dataset-fraction")
    model_paths = find_model_paths(args.model_path, args.limit)

    print(f"Using device: {ad_safe.DEVICE}")
    print(f"Datasets: {', '.join(dataset_names)}")
    print(f"Dataset fraction: {dataset_fraction}")
    print(f"Batch size: {args.batch_size if args.batch_size is not None else 'auto'}")
    print(f"Model path: {args.model_path}")
    print(f"Models to evaluate: {len(model_paths)}")
    print(f"Sort: {sort_key}")

    return ad_safe.EvaluationPlan(
        models=tuple(
            ad_safe.ModelEvalSpec(path=model_path)
            for model_path in model_paths
        ),
        datasets=tuple(
            ad_safe.DatasetEvalSpec(
                name=dataset_name,
                batch_size=args.batch_size,
                source=ad_safe.DatasetSourceSpec(
                    name=dataset_name,
                    fraction=dataset_fraction,
                    seed=args.dataset_seed,
                ),
            )
            for dataset_name in dataset_names
        ),
        output_dir=args.output_dir,
        sort_key=sort_key,
        write_csv=True,
        print_results=True,
        title="Results",
    )


def main() -> None:
    ad_safe.run_evaluation_plan(build_evaluation_plan(parse_args()))


if __name__ == "__main__":
    main()
