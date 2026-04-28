from __future__ import annotations

import math
import random
from collections.abc import MutableSequence
from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from .config import DATA_DIR, DEVICE, IMAGE_SIZE, TEST_RATIO, VALID_RATIO


IMAGE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ]
)


@dataclass(frozen=True)
class DatasetSourceSpec:
    name: str
    fraction: float = 1.0
    seed: int | None = None

    def to_json(self) -> dict[str, object]:
        return {
            "name": self.name,
            "fraction": self.fraction,
            "seed": self.seed,
        }


class RandomSource(Protocol):
    def shuffle(self, x: MutableSequence[int]) -> None:
        raise NotImplementedError


def to_device(*tensors: Tensor, device: torch.device = DEVICE) -> tuple[Tensor, ...]:
    return tuple(tensor.to(device) for tensor in tensors)


def make_data_loader(
    *datasets_list,
    batch_size: int,
    shuffle: bool,
    pin_memory: bool | None = None,
) -> tuple[DataLoader, ...]:
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    return tuple(
        DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=pin_memory,
        )
        for dataset in datasets_list
    )


class PreparedTrainingDataset(Dataset):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.teacher_logits: Tensor | None = None
        self.extra_samples: list[tuple[Tensor, int, Tensor | None]] = []
        self.classes = get_dataset_classes(dataset)
        self.targets = get_dataset_targets(dataset)

    @property
    def base_sample_count(self) -> int:
        return len(self.dataset)

    def set_teacher_logits(self, teacher_logits: Tensor | None) -> None:
        if teacher_logits is not None and teacher_logits.shape[0] != self.base_sample_count:
            raise ValueError(
                "Teacher logits length must match base dataset length, "
                f"got {teacher_logits.shape[0]} logits for {self.base_sample_count} samples"
            )
        self.teacher_logits = teacher_logits

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[Tensor, int] | tuple[Tensor, int, Tensor]:
        if index < self.base_sample_count:
            base_item = self.dataset[index]
            image, label = base_item[0], int(base_item[1])
            teacher_logits: Tensor | None = None
            if self.teacher_logits is not None:
                teacher_logits = self.teacher_logits[index]
            elif len(base_item) >= 3:
                teacher_logits = base_item[2]
        else:
            image, label, teacher_logits = self.extra_samples[index - self.base_sample_count]

        if teacher_logits is None:
            return image, label
        return image, label, teacher_logits

    def get_teacher_logits(self, index: int) -> Tensor | None:
        if index >= self.base_sample_count:
            return self.extra_samples[index - self.base_sample_count][2]
        if self.teacher_logits is not None:
            return self.teacher_logits[index]

        if hasattr(self.dataset, "get_teacher_logits"):
            return self.dataset.get_teacher_logits(index)

        base_item = self.dataset[index]
        if len(base_item) >= 3:
            return base_item[2]
        return None

    def add_sample(
        self,
        image: Tensor,
        label: int,
        teacher_logits: Tensor | None,
    ) -> None:
        if self.teacher_logits is not None:
            if teacher_logits is None:
                raise ValueError("teacher_logits must be provided when dataset has teacher logits")
            teacher_logits = teacher_logits.detach()
        self.extra_samples.append((image.detach().cpu(), int(label), teacher_logits))
        self.targets.append(int(label))


class LabeledDatasetView(Dataset):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        item = self.dataset[index]
        return item[0], item[1]


def get_dataset_classes(dataset: Dataset) -> list[str]:
    current_dataset = dataset
    while isinstance(current_dataset, Subset):
        current_dataset = current_dataset.dataset

    classes = getattr(current_dataset, "classes", None)
    if classes is None:
        raise AttributeError("Dataset does not expose class names via a 'classes' attribute")
    return list(classes)


def get_dataset_targets(dataset: Dataset) -> list[int]:
    if isinstance(dataset, Subset):
        inner_targets = get_dataset_targets(dataset.dataset)
        return [int(inner_targets[int(index)]) for index in dataset.indices]

    dataset_targets = getattr(dataset, "targets", None)
    if dataset_targets is None:
        raise AttributeError("Dataset does not expose class targets via a 'targets' attribute")
    return [int(label) for label in dataset_targets]


def _resolve_random_source(seed: int | None) -> RandomSource:
    return random if seed is None else random.Random(seed)


def _class_index_groups(dataset: Dataset) -> dict[int, list[int]]:
    class_to_indices: dict[int, list[int]] = {}
    for sample_index, class_index in enumerate(get_dataset_targets(dataset)):
        class_to_indices.setdefault(int(class_index), []).append(sample_index)
    return class_to_indices


def _shuffled_class_index_groups(
    dataset: Dataset,
    *,
    rng: RandomSource,
) -> list[list[int]]:
    shuffled_groups: list[list[int]] = []
    for class_indices in _class_index_groups(dataset).values():
        shuffled_indices = list(class_indices)
        rng.shuffle(shuffled_indices)
        shuffled_groups.append(shuffled_indices)
    return shuffled_groups


def _shuffle_if_requested(
    indices: MutableSequence[int],
    *,
    rng: RandomSource,
    shuffle: bool,
) -> None:
    if shuffle:
        rng.shuffle(indices)


def make_stratified_subset(
    dataset: Dataset,
    fraction: float,
    *,
    seed: int = 0,
    min_per_class: int = 1,
    shuffle: bool = True,
) -> Subset:
    if fraction <= 0 or fraction > 1:
        raise ValueError("fraction must be in the range (0, 1]")
    if min_per_class < 0:
        raise ValueError("min_per_class must be non-negative")

    rng = random.Random(seed)
    selected_indices: list[int] = []
    for shuffled_indices in _shuffled_class_index_groups(dataset, rng=rng):
        if fraction >= 1.0:
            keep_count = len(shuffled_indices)
        else:
            keep_count = int(math.ceil(len(shuffled_indices) * fraction))
            if min_per_class > 0 and shuffled_indices:
                keep_count = max(min_per_class, keep_count)
            keep_count = min(len(shuffled_indices), keep_count)
        selected_indices.extend(shuffled_indices[:keep_count])

    _shuffle_if_requested(selected_indices, rng=rng, shuffle=shuffle)
    return Subset(dataset, selected_indices)


def split_train_dataset(
    dataset: datasets.ImageFolder,
    valid_ratio: float = VALID_RATIO,
    test_ratio: float = TEST_RATIO,
    *,
    seed: int | None = None,
) -> tuple[Subset, Subset, Subset]:
    if valid_ratio < 0 or test_ratio < 0:
        raise ValueError("Split ratios must be non-negative")
    if valid_ratio + test_ratio >= 1:
        raise ValueError("valid_ratio + test_ratio must be less than 1")

    rng = _resolve_random_source(seed)
    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []

    for shuffled_indices in _shuffled_class_index_groups(dataset, rng=rng):
        val_count = int(len(shuffled_indices) * valid_ratio)
        test_count = int(len(shuffled_indices) * test_ratio)

        val_indices.extend(shuffled_indices[:val_count])
        test_indices.extend(shuffled_indices[val_count:val_count + test_count])
        train_indices.extend(shuffled_indices[val_count + test_count:])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    rng.shuffle(test_indices)

    return (
        Subset(dataset, train_indices),
        Subset(dataset, val_indices),
        Subset(dataset, test_indices),
    )


def load_dataset(split_name: str) -> datasets.ImageFolder:
    split_dir = DATA_DIR / split_name
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {split_dir}")
    if not any(path.is_file() for path in split_dir.glob("*/*")):
        raise ValueError(f"Dataset directory has no samples: {split_dir}")
    print(f"Loading dataset from {split_dir}...")
    return datasets.ImageFolder(split_dir, transform=IMAGE_TRANSFORM)


def load_dataset_source(spec: DatasetSourceSpec) -> Dataset:
    if spec.fraction <= 0 or spec.fraction > 1:
        raise ValueError("DatasetSourceSpec.fraction must be in the range (0, 1]")

    dataset = load_dataset(spec.name)
    if spec.fraction >= 1.0:
        return dataset

    seed = 0 if spec.seed is None else spec.seed
    subset = make_stratified_subset(dataset, spec.fraction, seed=seed)
    print(
        f"Using stratified dataset source {spec.name}: "
        f"{len(subset)}/{len(dataset)} samples "
        f"(fraction={spec.fraction}, seed={seed})"
    )
    return subset
