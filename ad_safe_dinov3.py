#!/usr/bin/env python3

import argparse
import json
import logging
import operator
import random
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch
from torch import nn, optim
from torch.fx import Graph, GraphModule
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm.auto import tqdm

logging.getLogger("torchao").setLevel(logging.ERROR)
from transformers import DINOv3ViTModel


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "ml_bootcamp_adsafety_dataset"
MODEL_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m"
CLASS_NAMES = ["safe", "unsafe"]
IMAGE_SIZE = 299
DINO_INPUT_SIZE = 224
DEFAULT_DATASET = "train"
DEFAULT_EPOCHS = 20
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_VAL_FRAC = 0.10
DEFAULT_TEST_FRAC = 0.05
DEFAULT_PATIENCE = 5
DEFAULT_SEED = 0
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    loss: float
    acc: float

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, EpochMetrics):
            return NotImplemented
        return (self.acc, -self.loss) > (other.acc, -other.loss)

    def __str__(self) -> str:
        return f"acc:{self.acc:.4f},loss:{self.loss:.4f},epoch:{self.epoch}"


def build_dinov3_features_graph(dinov3: nn.Module) -> GraphModule:
    layers = getattr(dinov3, "layer", None)
    if layers is None and hasattr(dinov3, "model"):
        layers = getattr(dinov3.model, "layer", None)
    if layers is None:
        raise ValueError("DINOv3 layers are expected at 'dinov3.layer' or 'dinov3.model.layer'")

    root = nn.Module()
    root.embeddings = dinov3.embeddings
    root.rope_embeddings = dinov3.rope_embeddings
    root.layers = layers
    root.norm = dinov3.norm

    graph = Graph()
    pixel_values = graph.placeholder("pixel_values")
    hidden_states = graph.call_module("embeddings", args=(pixel_values,))
    position_embeddings = graph.call_module("rope_embeddings", args=(pixel_values,))
    for layer_index in range(len(layers)):
        hidden_states = graph.call_module(
            f"layers.{layer_index}",
            args=(hidden_states,),
            kwargs={"position_embeddings": position_embeddings},
        )
    sequence_output = graph.call_module("norm", args=(hidden_states,))
    pooled_output = graph.call_function(
        operator.getitem,
        args=(sequence_output, (slice(None), 0, slice(None))),
    )
    graph.output(pooled_output)
    graph.lint()

    features = GraphModule(root, graph)
    features.recompile()
    return features


def make_resize_layer() -> nn.Module:
    return nn.Upsample(
        size=(DINO_INPUT_SIZE, DINO_INPUT_SIZE),
        mode="bilinear",
        align_corners=False,
    )


def make_normalize_layer() -> nn.Conv2d:
    normalize = nn.Conv2d(3, 3, kernel_size=1, bias=True)
    with torch.no_grad():
        normalize.weight.zero_()
        for channel_index, channel_std in enumerate(IMAGENET_STD):
            normalize.weight[channel_index, channel_index, 0, 0] = 1.0 / channel_std
            normalize.bias[channel_index] = -IMAGENET_MEAN[channel_index] / channel_std
    for parameter in normalize.parameters():
        parameter.requires_grad = False
    return normalize


def create_model() -> nn.Module:
    dinov3 = DINOv3ViTModel.from_pretrained(MODEL_ID)
    model = nn.Sequential(
        OrderedDict(
            [
                ("resize", make_resize_layer()),
                ("normalize", make_normalize_layer()),
                ("features", build_dinov3_features_graph(dinov3)),
                ("classifier", nn.Linear(int(dinov3.config.hidden_size), len(CLASS_NAMES))),
            ]
        )
    )
    model.backbone_name = "dinov3_vitl16_pretrain_lvd1689m"
    model.native_input_size = DINO_INPUT_SIZE
    return model


def load_model(model_path: Path) -> nn.Module:
    model = torch.load(model_path, map_location=DEVICE, weights_only=False)
    if not isinstance(model, nn.Module):
        raise TypeError(f"{model_path} does not contain a torch.nn.Module")
    return model


def get_features(model: nn.Module) -> nn.Module:
    features = getattr(model, "features", None)
    if not isinstance(features, nn.Module):
        raise ValueError("Model does not expose a 'features' module")
    return features


def get_classifier(model: nn.Module) -> nn.Module:
    classifier = getattr(model, "classifier", None)
    if not isinstance(classifier, nn.Module):
        raise ValueError("Model does not expose a 'classifier' module")
    return classifier


def configure_trainable_layers(model: nn.Module, *, unfreeze_all: bool) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    if unfreeze_all:
        for parameter in get_features(model).parameters():
            parameter.requires_grad = True
    for parameter in get_classifier(model).parameters():
        parameter.requires_grad = True


def count_subset_classes(dataset: datasets.ImageFolder, indices: list[int]) -> dict[str, int]:
    counts = {class_name: 0 for class_name in dataset.classes}
    for index in indices:
        _, class_index = dataset.samples[index]
        counts[dataset.classes[class_index]] += 1
    return counts


def format_counts(counts: dict[str, int]) -> str:
    return ", ".join(f"{class_name}={counts[class_name]}" for class_name in CLASS_NAMES)


def stratified_split_indices(
    dataset: datasets.ImageFolder,
    *,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> tuple[list[int], list[int], list[int]]:
    generator = torch.Generator().manual_seed(seed)
    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []

    for class_index in range(len(dataset.classes)):
        class_indices = [
            sample_index
            for sample_index, (_, sample_class_index) in enumerate(dataset.samples)
            if sample_class_index == class_index
        ]
        permutation = torch.randperm(len(class_indices), generator=generator).tolist()
        class_indices = [class_indices[index] for index in permutation]

        test_size = int(len(class_indices) * test_frac)
        val_size = int(len(class_indices) * val_frac)
        if test_frac > 0 and test_size == 0 and len(class_indices) >= 3:
            test_size = 1
        if val_frac > 0 and val_size == 0 and len(class_indices) - test_size >= 2:
            val_size = 1

        test_indices.extend(class_indices[:test_size])
        val_indices.extend(class_indices[test_size : test_size + val_size])
        train_indices.extend(class_indices[test_size + val_size :])

    return train_indices, val_indices, test_indices


def make_dataloaders(
    dataset_name: str,
    *,
    batch_size: int,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
        ]
    )
    data_dir = DATA_DIR / dataset_name
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    if dataset.classes != CLASS_NAMES:
        raise ValueError(f"Expected classes {CLASS_NAMES}, got {dataset.classes}")

    train_indices, val_indices, test_indices = stratified_split_indices(
        dataset,
        val_frac=val_frac,
        test_frac=test_frac,
        seed=seed,
    )
    if not train_indices or not val_indices:
        raise ValueError("Dataset is too small for the requested train/val/test split")

    print(f"dataset={data_dir}")
    print(f"samples total={len(dataset)} classes={format_counts(count_subset_classes(dataset, list(range(len(dataset)))))}")
    print(f"samples train={len(train_indices)} classes={format_counts(count_subset_classes(dataset, train_indices))}")
    print(f"samples val={len(val_indices)} classes={format_counts(count_subset_classes(dataset, val_indices))}")
    print(f"samples test={len(test_indices)} classes={format_counts(count_subset_classes(dataset, test_indices))}")

    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)
    test_set = Subset(dataset, test_indices)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, test_loader


def make_eval_dataloader(
    dataset_name: str,
    *,
    batch_size: int,
) -> DataLoader:
    transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
        ]
    )
    data_dir = DATA_DIR / dataset_name
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    if dataset.classes != CLASS_NAMES:
        raise ValueError(f"Expected classes {CLASS_NAMES}, got {dataset.classes}")

    print(f"dataset={data_dir}")
    print(f"samples total={len(dataset)} classes={format_counts(count_subset_classes(dataset, list(range(len(dataset)))))}")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )


def evaluate(model: nn.Module, loader: DataLoader, *, desc: str = "eval") -> tuple[float, float]:
    if len(loader.dataset) == 0:
        raise ValueError("Cannot evaluate on an empty dataset")

    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        progress = tqdm(loader, desc=desc, leave=False)
        for images, labels in progress:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item() * images.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += images.size(0)
            progress.set_postfix(loss=total_loss / total, acc=correct / total)
    return total_loss / total, correct / total


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    epochs: int,
    learning_rate: float,
    patience: int,
    output: Path,
) -> nn.Module:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate,
    )
    initial_val_loss, initial_val_acc = evaluate(model, val_loader, desc="initial val")
    best_metrics = EpochMetrics(epoch=0, loss=initial_val_loss, acc=initial_val_acc)
    epochs_without_improvement = 0
    print(f"initial val_loss={best_metrics.loss:.4f} val_acc={best_metrics.acc:.4f}")
    save_model(model, output)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}", leave=False)
        for images, labels in progress:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += images.size(0)
            progress.set_postfix(loss=total_loss / total, acc=correct / total)

        val_loss, val_acc = evaluate(model, val_loader, desc=f"val {epoch}/{epochs}")
        val_metrics = EpochMetrics(epoch=epoch, loss=val_loss, acc=val_acc)
        is_new_best = val_metrics > best_metrics
        if is_new_best:
            best_metrics = val_metrics
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(
            f"epoch={epoch}/{epochs} "
            f"train_loss={total_loss / total:.4f} "
            f"train_acc={correct / total:.4f} "
            f"val_loss={val_metrics.loss:.4f} "
            f"val_acc={val_metrics.acc:.4f} "
            f"best={best_metrics} "
            f"stale_epochs={epochs_without_improvement}"
        )
        if is_new_best:
            save_model(model, output)
        if not is_new_best and patience > 0 and epochs_without_improvement >= patience:
            print(
                f"early stopping at epoch={epoch}: "
                f"validation score did not improve for {epochs_without_improvement} epoch(s)"
            )
            break

    print(f"best={best_metrics}")
    return load_model(output).to(DEVICE)


def save_model(model: nn.Module, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.cpu().eval(), output_path)
    model.to(DEVICE)
    print(f"saved {output_path}")


def write_setup_json(
    output_path: Path,
    *,
    timestamp: str,
    seed: int,
    args: argparse.Namespace,
    output: Path,
) -> None:
    payload = {
        "timestamp": timestamp,
        "seed": seed,
        "device": str(DEVICE),
        "model_id": MODEL_ID,
        "dataset": args.dataset,
        "dataset_dir": str((DATA_DIR / args.dataset).resolve()),
        "model_path": str(args.model_path.resolve()) if args.model_path else None,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "val_frac": args.val_frac,
        "test_frac": args.test_frac,
        "patience": args.patience,
        "unfreeze_all": args.unfreeze_all,
        "output": str(output.resolve()),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"saved {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal DINOv3 ad-safety trainer")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help=f"Dataset folder inside {DATA_DIR}")
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--val-frac", type=float, default=DEFAULT_VAL_FRAC)
    parser.add_argument("--test-frac", type=float, default=DEFAULT_TEST_FRAC)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--unfreeze-all", action="store_true")
    parser.add_argument("--eval", action="store_true", help="Only evaluate --model-path on --dataset")
    return parser.parse_args()


def make_seed() -> int:
    return random.SystemRandom().randrange(2**32)


def resolve_seed(seed: int) -> int:
    return make_seed() if seed == 0 else seed


def validate_args(args: argparse.Namespace) -> None:
    if args.eval and args.model_path is None:
        raise ValueError("--eval requires --model-path")
    if not 0 <= args.val_frac < 1 or not 0 <= args.test_frac < 1:
        raise ValueError("--val-frac and --test-frac must be in [0, 1)")
    if args.val_frac + args.test_frac >= 1:
        raise ValueError("--val-frac + --test-frac must be less than 1")
    if args.patience < 0:
        raise ValueError("--patience must be non-negative")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_eval(args: argparse.Namespace) -> None:
    print(f"device={DEVICE}")
    model = load_model(args.model_path).to(DEVICE)
    data_loader = make_eval_dataloader(
        args.dataset,
        batch_size=args.batch_size,
    )
    loss, accuracy = evaluate(model, data_loader, desc="eval")
    print(f"eval loss={loss:.4f} acc={accuracy:.4f}")


def run_train(args: argparse.Namespace) -> None:
    seed = resolve_seed(args.seed)
    set_seed(seed)

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    output = SCRIPT_DIR / f"{timestamp}-dinov3.pt"
    setup_output = SCRIPT_DIR / f"{timestamp}-dinov3-setup.json"

    print(f"device={DEVICE}")
    print(f"seed={seed}")
    print(f"timestamp={timestamp}")
    print(f"output={output}")
    print(f"setup_output={setup_output}")
    write_setup_json(
        setup_output,
        timestamp=timestamp,
        seed=seed,
        args=args,
        output=output,
    )

    model = load_model(args.model_path) if args.model_path else create_model()
    model = model.to(DEVICE)
    configure_trainable_layers(model, unfreeze_all=args.unfreeze_all)

    train_loader, val_loader, test_loader = make_dataloaders(
        args.dataset,
        batch_size=args.batch_size,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=seed,
    )
    model = train(
        model,
        train_loader,
        val_loader,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        patience=args.patience,
        output=output,
    )

    if len(test_loader.dataset) > 0:
        test_loss, test_acc = evaluate(model, test_loader, desc="test")
        print(f"best test_loss={test_loss:.4f} test_acc={test_acc:.4f}")


def main() -> None:
    args = parse_args()
    validate_args(args)
    if args.eval:
        run_eval(args)
    else:
        run_train(args)


if __name__ == "__main__":
    main()
