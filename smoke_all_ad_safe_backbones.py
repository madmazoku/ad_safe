#!/usr/bin/env python3

import argparse
import subprocess
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset

import ad_safe


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "smoke_models"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build, minimally train, save, and foreign-load smoke-check every "
            "supported ad_safe backbone."
        )
    )
    parser.add_argument(
        "--backbone",
        action="append",
        choices=sorted(ad_safe.SUPPORTED_BACKBONES),
        help="Backbone to test. May be passed multiple times. Defaults to all.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for saved smoke models. Defaults to challenge/smoke_models/<timestamp>.",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=ad_safe.DEFAULT_LR)
    parser.add_argument(
        "--dataset-split",
        choices=["train", "val", "test"],
        default="train",
        help="Real dataset split used for the smoke train/val subsets.",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=2,
        help="How many real samples per class to use for each smoke subset.",
    )
    parser.add_argument(
        "--use-pretrained",
        action="store_true",
        help="Download/use pretrained weights. Default uses local random-init configs.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue testing remaining backbones after a failure.",
    )
    return parser.parse_args()


def make_contract_model(backbone_name: str, *, use_pretrained: bool) -> nn.Module:
    definition = ad_safe.SUPPORTED_BACKBONES[backbone_name]
    build_fn = getattr(ad_safe, definition.build_fn_name)
    built_model = build_fn(use_pretrained=use_pretrained)
    model = ad_safe.finalize_built_model(built_model, definition)
    ad_safe.configure_trainable_layers(model, unfreeze=())
    return model.to(ad_safe.DEVICE)


def make_subset_by_class(dataset: object, *, samples_per_class: int, offset: int) -> Subset:
    if samples_per_class <= 0:
        raise ValueError("--samples-per-class must be positive")

    targets = getattr(dataset, "targets", None)
    if targets is None:
        raise AttributeError("Dataset does not expose class targets via a 'targets' attribute")

    indices: list[int] = []
    for class_index in range(len(ad_safe.CLASS_NAMES)):
        class_indices = [
            sample_index
            for sample_index, sample_class_index in enumerate(targets)
            if int(sample_class_index) == class_index
        ]
        if len(class_indices) < offset + samples_per_class:
            raise ValueError(
                f"Dataset has only {len(class_indices)} samples for class {class_index}; "
                f"need at least {offset + samples_per_class}"
            )
        indices.extend(class_indices[offset : offset + samples_per_class])

    return Subset(dataset, indices)


def make_loader(dataset: Subset, batch_size: int, *, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=torch.cuda.is_available(),
    )


def run_foreign_contract_check(model_path: Path, batch_size: int) -> None:
    checker_path = SCRIPT_DIR / "check_ad_safe_contract.py"
    subprocess.run(
        [
            sys.executable,
            str(checker_path),
            str(model_path),
            "--batch-size",
            str(batch_size),
        ],
        check=True,
    )


def smoke_backbone(
    backbone_name: str,
    *,
    output_dir: Path,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    train_loader: DataLoader,
    val_loader: DataLoader,
    use_pretrained: bool,
) -> Path:
    print(f"\n=== {backbone_name} ===")
    model = make_contract_model(backbone_name, use_pretrained=use_pretrained)
    config = replace(
        ad_safe.TrainingConfig(base_model=backbone_name),
        epochs=epochs,
        patience=epochs,
        batch_size=batch_size,
        learning_rate=(learning_rate,),
        adversarial=False,
    )
    ad_safe.describe_model(model, config)

    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise ValueError(f"{backbone_name} exposes no trainable parameters")

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"{backbone_name}-model.pt"
    optimizer = optim.Adam(trainable_parameters, lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    ad_safe.save_model(model, model_path)
    model, _ = ad_safe.train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        config=config,
        best_model_path=model_path,
        split_round=1,
        global_epoch_start=0,
        training_start_time=time.time(),
    )
    ad_safe.save_model(model, model_path)
    run_foreign_contract_check(model_path, batch_size=1)
    return model_path


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or DEFAULT_OUTPUT_ROOT / datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    backbone_names = args.backbone or list(ad_safe.SUPPORTED_BACKBONES)
    failures: list[tuple[str, BaseException]] = []
    saved_models: list[Path] = []
    dataset = ad_safe.load_dataset(args.dataset_split)
    train_subset = make_subset_by_class(
        dataset,
        samples_per_class=args.samples_per_class,
        offset=0,
    )
    val_subset = make_subset_by_class(
        dataset,
        samples_per_class=args.samples_per_class,
        offset=args.samples_per_class,
    )
    train_loader = make_loader(train_subset, args.batch_size, shuffle=True)
    val_loader = make_loader(val_subset, args.batch_size, shuffle=False)

    for backbone_name in backbone_names:
        try:
            saved_models.append(
                smoke_backbone(
                    backbone_name,
                    output_dir=output_dir,
                    batch_size=args.batch_size,
                    epochs=args.epochs,
                    learning_rate=args.learning_rate,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    use_pretrained=args.use_pretrained,
                )
            )
        except BaseException as exc:
            failures.append((backbone_name, exc))
            print(f"FAILED {backbone_name}: {exc}", file=sys.stderr)
            if not args.keep_going:
                break

    print("\nSummary:")
    for model_path in saved_models:
        print(f"  ok {model_path}")
    for backbone_name, exc in failures:
        print(f"  failed {backbone_name}: {type(exc).__name__}: {exc}")

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
