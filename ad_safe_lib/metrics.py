from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .backbones import forward_logits
from .config import CLASS_NAMES
from .data import to_device


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
METRIC_NAMES = tuple(csv_name for _, csv_name in METRIC_CSV_FIELDS)
LOWER_IS_BETTER_METRICS = {"nll", "avg_wrong_conf"}
DEFAULT_PRINT_METRICS = ("acc", "auc", "nll", "avg_wrong_conf", "safe_recall", "unsafe_recall")


@dataclass(frozen=True)
class ClassificationMetrics:
    accuracy: float
    auc: float | None
    nll: float
    avg_conf: float
    avg_margin: float
    avg_correct_conf: float
    avg_wrong_conf: float | None
    safe_recall: float | None
    unsafe_recall: float | None

    def __str__(self) -> str:
        parts = [
            f"acc={self.accuracy:.4f}",
            f"nll={self.nll:.4f}",
            f"conf={self.avg_conf:.4f}",
            f"margin={self.avg_margin:.4f}",
        ]
        if self.auc is not None:
            parts.insert(1, f"auc={self.auc:.4f}")
        if self.avg_wrong_conf is not None:
            parts.append(f"wrong_conf={self.avg_wrong_conf:.4f}")
        if self.safe_recall is not None:
            parts.append(f"safe_recall={self.safe_recall:.4f}")
        if self.unsafe_recall is not None:
            parts.append(f"unsafe_recall={self.unsafe_recall:.4f}")
        return " ".join(parts)

    def to_json_dict(self) -> dict[str, float | None]:
        return {
            "accuracy": self.accuracy,
            "auc": self.auc,
            "nll": self.nll,
            "avg_conf": self.avg_conf,
            "avg_margin": self.avg_margin,
            "avg_correct_conf": self.avg_correct_conf,
            "avg_wrong_conf": self.avg_wrong_conf,
            "safe_recall": self.safe_recall,
            "unsafe_recall": self.unsafe_recall,
        }


class ValidationComparison(Enum):
    WORSE = "worse"
    EQUIVALENT = "equivalent"
    BETTER = "better"


@dataclass(frozen=True)
class DefaultValidationMetricComparator:
    min_delta: float = 0.0
    metric_epsilon: float = 1e-9

    def score(self, metrics: ClassificationMetrics) -> float:
        return metrics.accuracy

    def compare(
        self,
        candidate: ClassificationMetrics,
        best: ClassificationMetrics,
    ) -> ValidationComparison:
        if candidate.accuracy > best.accuracy + self.min_delta:
            return ValidationComparison.BETTER
        if candidate.accuracy < best.accuracy - self.min_delta:
            return ValidationComparison.WORSE

        if self._optional_higher(candidate.auc, best.auc):
            return ValidationComparison.BETTER
        if best.auc is not None and candidate.auc is not None and candidate.auc < best.auc - self.metric_epsilon:
            return ValidationComparison.EQUIVALENT
        if self._lower(candidate.nll, best.nll):
            return ValidationComparison.BETTER
        if self._optional_higher(candidate.unsafe_recall, best.unsafe_recall):
            return ValidationComparison.BETTER
        if self._optional_lower(candidate.avg_wrong_conf, best.avg_wrong_conf):
            return ValidationComparison.BETTER
        return ValidationComparison.EQUIVALENT

    def _lower(self, candidate: float, best: float) -> bool:
        return candidate < best - self.metric_epsilon

    def _optional_higher(self, candidate: float | None, best: float | None) -> bool:
        return candidate is not None and best is not None and candidate > best + self.metric_epsilon

    def _optional_lower(self, candidate: float | None, best: float | None) -> bool:
        return candidate is not None and best is not None and candidate < best - self.metric_epsilon


def safe_mean(values: Tensor) -> float | None:
    if values.numel() == 0:
        return None
    return float(values.float().mean().item())


def safe_recall(predictions: Tensor, labels: Tensor, class_index: int) -> float | None:
    class_mask = labels == class_index
    if not bool(class_mask.any()):
        return None
    return float((predictions[class_mask] == class_index).float().mean().item())


def evaluate_classification_metrics(
    model: nn.Module,
    data_loader: DataLoader,
    split_name: str,
) -> ClassificationMetrics:
    model.eval()
    logits_batches: list[Tensor] = []
    label_batches: list[Tensor] = []

    with torch.inference_mode():
        for images, labels in tqdm(
            data_loader,
            desc=f"Evaluating ({split_name})",
            leave=False,
        ):
            images, labels = to_device(images, labels)
            logits = forward_logits(model, images)
            logits_batches.append(logits.detach().cpu())
            label_batches.append(labels.detach().cpu())

    if not logits_batches:
        raise ValueError(f"Cannot evaluate empty data loader for split: {split_name}")

    logits = torch.cat(logits_batches, dim=0)
    labels = torch.cat(label_batches, dim=0)
    probs = torch.softmax(logits, dim=1)
    predictions = logits.argmax(dim=1)
    confidences = probs.gather(1, predictions.unsqueeze(1)).squeeze(1)
    true_confidences = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
    sorted_probs = probs.sort(dim=1, descending=True).values
    margins = sorted_probs[:, 0] - sorted_probs[:, 1]
    correct_mask = predictions == labels

    auc: float | None
    try:
        auc = float(
            roc_auc_score(
                (labels == CLASS_NAMES.index("unsafe")).numpy(),
                probs[:, CLASS_NAMES.index("unsafe")].numpy(),
            )
        )
    except ValueError:
        auc = None

    return ClassificationMetrics(
        accuracy=float(correct_mask.float().mean().item()),
        auc=auc,
        nll=float(F.cross_entropy(logits, labels).item()),
        avg_conf=float(confidences.mean().item()),
        avg_margin=float(margins.mean().item()),
        avg_correct_conf=float(true_confidences.mean().item()),
        avg_wrong_conf=safe_mean(confidences[~correct_mask]),
        safe_recall=safe_recall(predictions, labels, CLASS_NAMES.index("safe")),
        unsafe_recall=safe_recall(predictions, labels, CLASS_NAMES.index("unsafe")),
    )


def evaluate_metrics(
    model: nn.Module,
    data_loader: DataLoader,
    split_name: str,
) -> ClassificationMetrics:
    return evaluate_classification_metrics(model, data_loader, split_name)


def evaluate_accuracy(model: nn.Module, data_loader: DataLoader, split_name: str) -> float:
    return evaluate_metrics(model, data_loader, split_name).accuracy


def evaluate_validation_score(
    model: nn.Module,
    val_loader: DataLoader,
) -> ClassificationMetrics:
    return evaluate_metrics(model, val_loader, "val")

