#!/usr/bin/env python3

import argparse
import builtins
import gc
import json
import operator
import os
import random
import time
import textwrap
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, replace
from datetime import datetime
from enum import Enum
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

def is_notebook_environment() -> bool:
    try:
        from IPython import get_ipython
    except ImportError:
        return False

    shell = get_ipython()
    return shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell"


if not is_notebook_environment():
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from torch import Tensor, nn, optim
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.fx import Graph, GraphModule
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm.auto import tqdm
from torchvision import datasets, models, transforms
from torchvision.models import (
    ConvNeXt_Base_Weights,
    ConvNeXt_Tiny_Weights,
    ConvNeXt_Large_Weights,
    EfficientNet_V2_S_Weights,
    Inception_V3_Weights,
    MaxVit_T_Weights,
    Swin_B_Weights,
    Swin_V2_B_Weights,
    ViT_H_14_Weights,
    ViT_L_16_Weights,
)
from torchvision.transforms import functional as TF
from matplotlib.figure import Figure

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "ml_bootcamp_adsafety_dataset"
CLASS_NAMES = ["safe", "unsafe"]
IMAGE_SIZE = 299
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 3
DEFAULT_RESPLIT_RUNS = 1
DEFAULT_ADV_STEPS = 5
DEFAULT_TARGET_STEPS = 20
DEFAULT_LR = 1e-4
DEFAULT_PATIENCE = 10
DEFAULT_ADV_EPSILON = 0.15
DEFAULT_DISTILLATION_ALPHA = 0.3
DEFAULT_DISTILLATION_TEMPERATURE = 2.0
DEFAULT_COOLDOWN_EVERY_EPOCHS = 0
DEFAULT_COOLDOWN_SECONDS = 0.0
DEFAULT_GPU_MAX_TEMP = 0
DEFAULT_GPU_RESUME_TEMP = 0
DEFAULT_GPU_TEMP_CHECK_SECONDS = 15.0
VALID_RATIO = 0.10
TEST_RATIO = 0.05
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_MODEL_WEIGHTS = Inception_V3_Weights.IMAGENET1K_V1
MODEL_PREPROCESS = DEFAULT_MODEL_WEIGHTS.transforms()
IMAGENET_MEAN = tuple(float(value) for value in MODEL_PREPROCESS.mean)
IMAGENET_STD = tuple(float(value) for value in MODEL_PREPROCESS.std)


@dataclass(frozen=True)
class TrainingConfig:
    base_model: str = "inception_v3"
    epochs: int = DEFAULT_EPOCHS
    patience: int = DEFAULT_PATIENCE
    batch_size: int = DEFAULT_BATCH_SIZE
    learning_rate: tuple[float, ...] = (DEFAULT_LR,)
    learning_rate_multiplier: float = 1.0
    resplit_runs: int = DEFAULT_RESPLIT_RUNS
    unfreeze: tuple[str, ...] = ()
    adversarial: bool = False
    adv_epsilon: float = DEFAULT_ADV_EPSILON
    adv_steps: int = DEFAULT_ADV_STEPS
    teacher_model_path: str | None = None
    distillation_alpha: float = DEFAULT_DISTILLATION_ALPHA
    distillation_temperature: float = DEFAULT_DISTILLATION_TEMPERATURE


@dataclass(frozen=True)
class TrainingHistoryEntry:
    split_round: int
    epoch_in_round: int
    global_epoch: float
    train_loss: float | None
    val_accuracy: float
    val_auc: float | None
    val_nll: float | None
    val_unsafe_recall: float | None
    val_avg_wrong_conf: float | None
    score: float
    elapsed_seconds: float
    is_baseline: bool
    is_new_best: bool


@dataclass(frozen=True)
class CooldownConfig:
    every_epochs: int = DEFAULT_COOLDOWN_EVERY_EPOCHS
    seconds: float = DEFAULT_COOLDOWN_SECONDS
    gpu_max_temp: int = DEFAULT_GPU_MAX_TEMP
    gpu_resume_temp: int = DEFAULT_GPU_RESUME_TEMP
    gpu_temp_check_seconds: float = DEFAULT_GPU_TEMP_CHECK_SECONDS

    @property
    def uses_temperature(self) -> bool:
        return self.gpu_max_temp > 0

    @property
    def enabled(self) -> bool:
        return self.every_epochs > 0 or self.uses_temperature

    def to_json(self) -> dict[str, object]:
        return {
            "every_epochs": self.every_epochs,
            "seconds": self.seconds,
            "gpu_max_temp": self.gpu_max_temp,
            "gpu_resume_temp": self.gpu_resume_temp,
            "gpu_temp_check_seconds": self.gpu_temp_check_seconds,
        }


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


class EpochEndHandler(ABC):
    @abstractmethod
    def on_epoch_end(self, entry: TrainingHistoryEntry) -> None:
        raise NotImplementedError


class CooldownEpochEndHandler(EpochEndHandler):
    def __init__(
        self,
        *,
        config: CooldownConfig,
        backbone_name: str,
        phase_name: str,
    ) -> None:
        self.config = config
        self.backbone_name = backbone_name
        self.phase_name = phase_name
        self._nvml = None
        self._handle = None
        if self.config.uses_temperature:
            if DEVICE.type != "cuda":
                raise RuntimeError("--gpu-max-temp requires CUDA")
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())

    def on_epoch_end(self, entry: TrainingHistoryEntry) -> None:
        if not self.config.enabled:
            return

        current_temp = self._read_gpu_temp()
        reasons = []
        global_epoch = int(entry.global_epoch)
        if self.config.every_epochs > 0 and global_epoch % self.config.every_epochs == 0:
            reasons.append(f"global_epoch={global_epoch}")
        if current_temp is not None and current_temp >= self.config.gpu_max_temp:
            reasons.append(f"gpu_temp={current_temp}C")
        if not reasons:
            return

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        started = time.time()
        deadline = started + self.config.seconds
        print(
            f"Cooldown start | {self.backbone_name} {self.phase_name} | "
            f"reason={','.join(reasons)} | max_wait={self.config.seconds:.1f}s"
        )

        while True:
            current_temp = self._read_gpu_temp()
            if current_temp is not None and current_temp <= self.config.gpu_resume_temp:
                break

            remaining_seconds = deadline - time.time()
            if remaining_seconds <= 0:
                break

            if current_temp is None:
                print(f"Cooldown waiting {remaining_seconds:.1f}s")
            else:
                print(
                    f"Cooldown waiting | gpu_temp={current_temp}C | "
                    f"resume_temp={self.config.gpu_resume_temp}C | "
                    f"remaining={remaining_seconds:.1f}s"
                )
            time.sleep(min(self.config.gpu_temp_check_seconds, remaining_seconds))

        final_temp = self._read_gpu_temp()
        waited_seconds = time.time() - started
        temp_part = "unknown" if final_temp is None else f"{final_temp}C"
        print(f"Cooldown end | waited={waited_seconds:.1f}s | gpu_temp={temp_part}")

    def _read_gpu_temp(self) -> int | None:
        if self._nvml is None or self._handle is None:
            return None
        return int(self._nvml.nvmlDeviceGetTemperature(self._handle, self._nvml.NVML_TEMPERATURE_GPU))


@dataclass(frozen=True)
class BackboneDefinition(ABC):
    name: str
    native_input_size: int
    normalize_mean: tuple[float, float, float] = IMAGENET_MEAN
    normalize_std: tuple[float, float, float] = IMAGENET_STD

    @abstractmethod
    def build(self, *, use_pretrained: bool = True) -> nn.Module:
        raise NotImplementedError


@dataclass(frozen=True)
class SimpleSequentialBackboneDefinition(BackboneDefinition):
    def build(self, *, use_pretrained: bool = True) -> nn.Module:
        return nn.Sequential(self.build_layers())

    @abstractmethod
    def build_layers(self) -> OrderedDict[str, nn.Module]:
        raise NotImplementedError


@dataclass(frozen=True)
class TorchVisionClassifierBackboneDefinition(BackboneDefinition):
    classifier_path = ""

    def build(self, *, use_pretrained: bool = True) -> nn.Module:
        backbone = self.build_torchvision_model(use_pretrained=use_pretrained)
        _replace_linear_classifier(backbone, self.classifier_path)
        return backbone

    @abstractmethod
    def build_torchvision_model(self, *, use_pretrained: bool) -> nn.Module:
        raise NotImplementedError


@dataclass(frozen=True)
class HuggingFaceLogitsBackboneDefinition(BackboneDefinition):
    def build(self, *, use_pretrained: bool = True) -> nn.Module:
        return build_hf_logits_graph(self.build_logits_model(use_pretrained=use_pretrained))

    @abstractmethod
    def build_logits_model(self, *, use_pretrained: bool) -> nn.Module:
        raise NotImplementedError


@dataclass(frozen=True)
class FeatureBackboneDefinition(BackboneDefinition):
    def build(self, *, use_pretrained: bool = True) -> nn.Module:
        features, feature_dim = self.build_features(use_pretrained=use_pretrained)
        return wrap_feature_backbone(
            features,
            feature_dim,
            self.name,
            self.native_input_size,
            mean=self.normalize_mean,
            std=self.normalize_std,
        )

    @abstractmethod
    def build_features(self, *, use_pretrained: bool) -> tuple[nn.Module, int]:
        raise NotImplementedError


@dataclass(frozen=True)
class SimpleCnnBackboneDefinition(SimpleSequentialBackboneDefinition):
    def build_layers(self) -> OrderedDict[str, nn.Module]:
        return OrderedDict(
            [
                (
                    "conv_stage_1",
                    nn.Sequential(
                        nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(kernel_size=2),
                    ),
                ),
                (
                    "conv_stage_2",
                    nn.Sequential(
                        nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(kernel_size=2),
                    ),
                ),
                (
                    "conv_stage_3",
                    nn.Sequential(
                        nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(kernel_size=2),
                    ),
                ),
                (
                    "hidden_1",
                    nn.Sequential(
                        nn.Flatten(),
                        nn.Linear(256 * 16 * 16, 192),
                        nn.ReLU(inplace=True),
                        nn.Dropout(p=0.2),
                    ),
                ),
                (
                    "hidden_2",
                    nn.Sequential(
                        nn.Linear(192, 64),
                        nn.ReLU(inplace=True),
                        nn.Dropout(p=0.2),
                    ),
                ),
                ("classifier", nn.Linear(64, len(CLASS_NAMES))),
            ]
        )


@dataclass(frozen=True)
class SimpleMlpBackboneDefinition(SimpleSequentialBackboneDefinition):
    def build_layers(self) -> OrderedDict[str, nn.Module]:
        flattened_features = 3 * 64 * 64
        return OrderedDict(
            [
                (
                    "hidden_1",
                    nn.Sequential(
                        nn.Flatten(),
                        nn.Linear(flattened_features, 1024),
                        nn.ReLU(inplace=True),
                        nn.Dropout(p=0.2),
                    ),
                ),
                (
                    "hidden_2",
                    nn.Sequential(
                        nn.Linear(1024, 256),
                        nn.ReLU(inplace=True),
                        nn.Dropout(p=0.2),
                    ),
                ),
                ("classifier", nn.Linear(256, len(CLASS_NAMES))),
            ]
        )


@dataclass(frozen=True)
class InceptionV3BackboneDefinition(TorchVisionClassifierBackboneDefinition):
    classifier_path = "fc"

    def build_torchvision_model(self, *, use_pretrained: bool) -> nn.Module:
        return models.inception_v3(
            weights=Inception_V3_Weights.IMAGENET1K_V1 if use_pretrained else None,
            init_weights=False,
        )

    def build(self, *, use_pretrained: bool = True) -> nn.Module:
        backbone = super().build(use_pretrained=use_pretrained)
        if backbone.AuxLogits is not None:
            _replace_linear_classifier(backbone, "AuxLogits.fc")
        return backbone


@dataclass(frozen=True)
class EfficientNetV2SBackboneDefinition(TorchVisionClassifierBackboneDefinition):
    classifier_path = "classifier.-1"

    def build_torchvision_model(self, *, use_pretrained: bool) -> nn.Module:
        return models.efficientnet_v2_s(
            weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1 if use_pretrained else None
        )


@dataclass(frozen=True)
class ConvNeXtTinyBackboneDefinition(TorchVisionClassifierBackboneDefinition):
    classifier_path = "classifier.-1"

    def build_torchvision_model(self, *, use_pretrained: bool) -> nn.Module:
        return models.convnext_tiny(
            weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if use_pretrained else None
        )


@dataclass(frozen=True)
class MaxVitTBackboneDefinition(TorchVisionClassifierBackboneDefinition):
    classifier_path = "classifier.-1"

    def build_torchvision_model(self, *, use_pretrained: bool) -> nn.Module:
        return models.maxvit_t(
            weights=MaxVit_T_Weights.IMAGENET1K_V1 if use_pretrained else None
        )


@dataclass(frozen=True)
class SwinBBackboneDefinition(TorchVisionClassifierBackboneDefinition):
    classifier_path = "head"

    def build_torchvision_model(self, *, use_pretrained: bool) -> nn.Module:
        return models.swin_b(
            weights=Swin_B_Weights.IMAGENET1K_V1 if use_pretrained else None
        )


@dataclass(frozen=True)
class SwinV2BBackboneDefinition(TorchVisionClassifierBackboneDefinition):
    classifier_path = "head"

    def build_torchvision_model(self, *, use_pretrained: bool) -> nn.Module:
        return models.swin_v2_b(
            weights=Swin_V2_B_Weights.IMAGENET1K_V1 if use_pretrained else None
        )


@dataclass(frozen=True)
class ConvNeXtBaseBackboneDefinition(TorchVisionClassifierBackboneDefinition):
    classifier_path = "classifier.-1"

    def build_torchvision_model(self, *, use_pretrained: bool) -> nn.Module:
        return models.convnext_base(
            weights=ConvNeXt_Base_Weights.IMAGENET1K_V1 if use_pretrained else None
        )


@dataclass(frozen=True)
class ConvNeXtLargeBackboneDefinition(TorchVisionClassifierBackboneDefinition):
    classifier_path = "classifier.-1"

    def build_torchvision_model(self, *, use_pretrained: bool) -> nn.Module:
        return models.convnext_large(
            weights=ConvNeXt_Large_Weights.IMAGENET1K_V1 if use_pretrained else None
        )


@dataclass(frozen=True)
class VitL16BackboneDefinition(TorchVisionClassifierBackboneDefinition):
    classifier_path = "heads.head"

    def build_torchvision_model(self, *, use_pretrained: bool) -> nn.Module:
        return models.vit_l_16(
            weights=ViT_L_16_Weights.IMAGENET1K_V1 if use_pretrained else None
        )


@dataclass(frozen=True)
class VitH14BackboneDefinition(TorchVisionClassifierBackboneDefinition):
    classifier_path = "heads.head"

    def build_torchvision_model(self, *, use_pretrained: bool) -> nn.Module:
        return models.vit_h_14(
            weights=ViT_H_14_Weights.IMAGENET1K_SWAG_LINEAR_V1 if use_pretrained else None
        )


@dataclass(frozen=True)
class ClipVitL14BackboneDefinition(FeatureBackboneDefinition):
    def build_features(self, *, use_pretrained: bool) -> tuple[nn.Module, int]:
        try:
            from transformers import CLIPConfig, CLIPModel, CLIPVisionConfig, CLIPVisionModel
        except ImportError as exc:
            raise ImportError(
                "clip_vit_l_14 requires the 'transformers' package. Install it with: pip install transformers"
            ) from exc

        if use_pretrained:
            pretrained_clip = CLIPModel.from_pretrained(HF_CLIP_VIT_L_14_MODEL_ID)
            features = pretrained_clip.vision_model
            hidden_size = int(pretrained_clip.config.vision_config.hidden_size)
            del pretrained_clip
        else:
            try:
                clip_config = CLIPConfig.from_pretrained(HF_CLIP_VIT_L_14_MODEL_ID, local_files_only=True)
                config = clip_config.vision_config
            except Exception:
                config = CLIPVisionConfig()
            features = CLIPVisionModel(config).vision_model
            hidden_size = int(config.hidden_size)

        return build_hf_pooler_graph(features), hidden_size


@dataclass(frozen=True)
class DinoV2ImageClassificationBackboneDefinition(HuggingFaceLogitsBackboneDefinition):
    def build_logits_model(self, *, use_pretrained: bool) -> nn.Module:
        try:
            from transformers import Dinov2Config, Dinov2ForImageClassification
        except ImportError as exc:
            raise ImportError(
                "dinov2_for_image_classification requires the 'transformers' package. "
                "Install it with: pip install transformers"
            ) from exc

        if use_pretrained:
            backbone = Dinov2ForImageClassification.from_pretrained(
                HF_DINOV2_IMAGE_CLASSIFICATION_MODEL_ID,
                num_labels=len(CLASS_NAMES),
                ignore_mismatched_sizes=True,
            )
        else:
            try:
                config = Dinov2Config.from_pretrained(
                    HF_DINOV2_IMAGE_CLASSIFICATION_MODEL_ID,
                    local_files_only=True,
                )
            except Exception:
                config = Dinov2Config()
            config.num_labels = len(CLASS_NAMES)
            backbone = Dinov2ForImageClassification(config)
        return backbone


@dataclass(frozen=True)
class DinoV3VitL16PretrainBackboneDefinition(FeatureBackboneDefinition):
    def build_features(self, *, use_pretrained: bool) -> tuple[nn.Module, int]:
        try:
            from transformers import DINOv3ViTConfig, DINOv3ViTModel
        except ImportError as exc:
            raise ImportError(
                "dinov3_vitl16_pretrain_lvd1689m requires the 'transformers' package. "
                "Install it with: pip install transformers"
            ) from exc

        if use_pretrained:
            backbone = DINOv3ViTModel.from_pretrained(HF_DINOV3_VITL16_PRETRAIN_MODEL_ID)
            hidden_size = int(backbone.config.hidden_size)
        else:
            try:
                config = DINOv3ViTConfig.from_pretrained(
                    HF_DINOV3_VITL16_PRETRAIN_MODEL_ID,
                    local_files_only=True,
                )
            except Exception:
                config = DINOv3ViTConfig(
                    hidden_size=1024,
                    intermediate_size=4096,
                    num_attention_heads=16,
                    num_hidden_layers=24,
                )
            hidden_size = int(config.hidden_size)
            backbone = DINOv3ViTModel(config)

        return build_dinov3_features_graph(backbone), hidden_size


IMAGE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ]
)
NORMALIZE_TRANSFORM = transforms.Normalize(
    mean=MODEL_PREPROCESS.mean,
    std=MODEL_PREPROCESS.std,
)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
HF_CLIP_VIT_L_14_MODEL_ID = "openai/clip-vit-large-patch14"
HF_DINOV2_IMAGE_CLASSIFICATION_MODEL_ID = "facebook/dinov2-base-imagenet1k-1-layer"
HF_DINOV3_VITL16_PRETRAIN_MODEL_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m"

SUPPORTED_BACKBONES: dict[str, BackboneDefinition] = {
    "simple_cnn": SimpleCnnBackboneDefinition(
        name="simple_cnn",
        native_input_size=128,
    ),
    "simple_mlp": SimpleMlpBackboneDefinition(
        name="simple_mlp",
        native_input_size=64,
    ),
    "inception_v3": InceptionV3BackboneDefinition(
        name="inception_v3",
        native_input_size=299,
    ),
    "efficientnet_v2_s": EfficientNetV2SBackboneDefinition(
        name="efficientnet_v2_s",
        native_input_size=384,
    ),
    "convnext_tiny": ConvNeXtTinyBackboneDefinition(
        name="convnext_tiny",
        native_input_size=224,
    ),
    "maxvit_t": MaxVitTBackboneDefinition(
        name="maxvit_t",
        native_input_size=224,
    ),
    "swin_b": SwinBBackboneDefinition(
        name="swin_b",
        native_input_size=224,
    ),
    "swin_v2_b": SwinV2BBackboneDefinition(
        name="swin_v2_b",
        native_input_size=256,
    ),
    "convnext_base": ConvNeXtBaseBackboneDefinition(
        name="convnext_base",
        native_input_size=224,
    ),
    "convnext_large": ConvNeXtLargeBackboneDefinition(
        name="convnext_large",
        native_input_size=224,
    ),
    "vit_l_16": VitL16BackboneDefinition(
        name="vit_l_16",
        native_input_size=224,
    ),
    "vit_h_14": VitH14BackboneDefinition(
        name="vit_h_14",
        native_input_size=224,
    ),
    "clip_vit_l_14": ClipVitL14BackboneDefinition(
        name="clip_vit_l_14",
        native_input_size=224,
        normalize_mean=CLIP_MEAN,
        normalize_std=CLIP_STD,
    ),
    "dinov2_for_image_classification": DinoV2ImageClassificationBackboneDefinition(
        name="dinov2_for_image_classification",
        native_input_size=224,
    ),
    "dinov3_vitl16_pretrain_lvd1689m": DinoV3VitL16PretrainBackboneDefinition(
        name="dinov3_vitl16_pretrain_lvd1689m",
        native_input_size=224,
    ),
}

def make_resize_layer(target_size: int) -> nn.Module:
    if target_size == IMAGE_SIZE:
        return nn.Identity()
    return nn.Upsample(
        size=(target_size, target_size),
        mode="bilinear",
        align_corners=False,
    )


def make_normalize_layer(
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
) -> nn.Conv2d:
    normalize = nn.Conv2d(3, 3, kernel_size=1, bias=True)
    with torch.no_grad():
        normalize.weight.zero_()
        for channel_index, channel_std in enumerate(std):
            normalize.weight[channel_index, channel_index, 0, 0] = 1.0 / channel_std
            normalize.bias[channel_index] = -mean[channel_index] / channel_std
    for parameter in normalize.parameters():
        parameter.requires_grad = False
    return normalize


def wrap_backbone(
    backbone: nn.Module,
    backbone_name: str,
    native_input_size: int,
    *,
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
) -> nn.Sequential:
    model = nn.Sequential(
        OrderedDict(
            [
                ("resize", make_resize_layer(native_input_size)),
                ("normalize", make_normalize_layer(mean=mean, std=std)),
                ("backbone", backbone),
            ]
        )
    )
    model.backbone_name = backbone_name
    model.native_input_size = native_input_size
    model._ad_safe_embedded_preprocess = True
    return model


def wrap_feature_backbone(
    features: nn.Module,
    feature_dim: int,
    backbone_name: str,
    native_input_size: int,
    *,
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
) -> nn.Sequential:
    model = nn.Sequential(
        OrderedDict(
            [
                ("resize", make_resize_layer(native_input_size)),
                ("normalize", make_normalize_layer(mean=mean, std=std)),
                ("features", features),
                ("classifier", nn.Linear(feature_dim, len(CLASS_NAMES))),
            ]
        )
    )
    model.backbone_name = backbone_name
    model.native_input_size = native_input_size
    model._ad_safe_embedded_preprocess = True
    return model


def make_seed() -> int:
    return random.SystemRandom().randrange(2**32)


def parse_learning_rates(learning_rate_arg: str) -> tuple[float, ...]:
    learning_rates = tuple(
        float(part.strip()) for part in learning_rate_arg.split(",") if part.strip()
    )
    if not learning_rates:
        raise ValueError("--learning-rate must contain at least one positive value")
    if any(learning_rate <= 0 for learning_rate in learning_rates):
        raise ValueError("--learning-rate values must be positive")
    return learning_rates


def normalize_learning_rates_value(value: object) -> tuple[float, ...]:
    if isinstance(value, str):
        return parse_learning_rates(value)
    if isinstance(value, (int, float)):
        return parse_learning_rates(str(value))
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("learning_rate must contain at least one positive value")
        return parse_learning_rates(",".join(str(item) for item in value))
    raise ValueError("learning_rate must be a number, string, or list of numbers")


def normalize_unfreeze_value(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(block_name.strip() for block_name in value.split(",") if block_name.strip())
    if isinstance(value, (list, tuple)):
        if not all(isinstance(item, str) for item in value):
            raise ValueError("unfreeze must be a list of strings")
        return tuple(item.strip() for item in value if item.strip())
    raise ValueError("unfreeze must be a comma-separated string or a list of strings")


def get_learning_rate_for_split(config: TrainingConfig, split_round: int) -> float:
    if split_round <= 0:
        raise ValueError("split_round must be positive")

    if len(config.learning_rate) == 1:
        return config.learning_rate[0] * (config.learning_rate_multiplier ** (split_round - 1))

    learning_rate_index = min(split_round - 1, len(config.learning_rate) - 1)
    return config.learning_rate[learning_rate_index]


def get_default_batch_size(device: torch.device = DEVICE) -> int:
    if device.type != "cuda" or not torch.cuda.is_available():
        return DEFAULT_BATCH_SIZE

    total_vram_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
    if total_vram_gb >= 32:
        return 256
    if total_vram_gb >= 16:
        return 128
    if total_vram_gb >= 8:
        return 64
    return DEFAULT_BATCH_SIZE


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logits(outputs: Tensor | object) -> Tensor:
    return outputs.logits if hasattr(outputs, "logits") else outputs


def forward_logits(model: nn.Module, images: Tensor) -> Tensor:
    if getattr(model, "_ad_safe_embedded_preprocess", False):
        return get_logits(model(images))
    normalized_images = NORMALIZE_TRANSFORM(images)
    return get_logits(model(normalized_images))


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
        dataset_targets = getattr(dataset, "targets", None)
        if dataset_targets is None:
            raise AttributeError("Dataset does not expose class targets via a 'targets' attribute")
        self.targets = [int(label) for label in dataset_targets]

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
            image, label = self.dataset[index]
            teacher_logits = self.teacher_logits[index] if self.teacher_logits is not None else None
        else:
            image, label, teacher_logits = self.extra_samples[index - self.base_sample_count]

        if teacher_logits is None:
            return image, label
        return image, label, teacher_logits

    def get_teacher_logits(self, index: int) -> Tensor | None:
        if self.teacher_logits is None:
            return None
        if index >= self.base_sample_count:
            return self.extra_samples[index - self.base_sample_count][2]
        return self.teacher_logits[index]

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


def _get_child_module(module: nn.Module, name: str) -> nn.Module:
    if isinstance(module, (nn.Sequential, nn.ModuleList)) and name.lstrip("-").isdigit():
        return module[int(name)]
    return getattr(module, name)


def _set_child_module(module: nn.Module, name: str, child: nn.Module) -> None:
    if isinstance(module, (nn.Sequential, nn.ModuleList)) and name.lstrip("-").isdigit():
        module[int(name)] = child
    else:
        setattr(module, name, child)


def _replace_linear_classifier(module: nn.Module, path: str) -> None:
    parent = module
    path_parts = path.split(".")
    for part in path_parts[:-1]:
        parent = _get_child_module(parent, part)

    last_part = path_parts[-1]
    classifier = _get_child_module(parent, last_part)
    if not isinstance(classifier, nn.Linear):
        raise TypeError(f"Expected Linear classifier at '{path}', got {type(classifier).__name__}")
    _set_child_module(parent, last_part, nn.Linear(classifier.in_features, len(CLASS_NAMES)))


def build_hf_logits_graph(backbone: nn.Module) -> GraphModule:
    root = nn.Module()
    root.backbone = backbone

    graph = Graph()
    pixel_values = graph.placeholder("pixel_values")
    outputs = graph.call_module("backbone", kwargs={"pixel_values": pixel_values})
    logits = graph.call_function(builtins.getattr, args=(outputs, "logits"))
    graph.output(logits)
    graph.lint()

    logits_graph = GraphModule(root, graph)
    logits_graph.recompile()
    return logits_graph


def build_hf_pooler_graph(vision_model: nn.Module) -> GraphModule:
    root = nn.Module()
    root.vision_model = vision_model

    graph = Graph()
    pixel_values = graph.placeholder("pixel_values")
    outputs = graph.call_module("vision_model", kwargs={"pixel_values": pixel_values})
    pooled_output = graph.call_function(builtins.getattr, args=(outputs, "pooler_output"))
    graph.output(pooled_output)
    graph.lint()

    pooler_graph = GraphModule(root, graph)
    pooler_graph.recompile()
    return pooler_graph


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


def list_supported_backbones() -> list[BackboneDefinition]:
    return [SUPPORTED_BACKBONES[name] for name in SUPPORTED_BACKBONES]


def finalize_built_model(
    built_model: nn.Module,
    definition: BackboneDefinition,
) -> nn.Module:
    if getattr(built_model, "_ad_safe_embedded_preprocess", False):
        built_model.backbone_name = definition.name
        if not hasattr(built_model, "native_input_size"):
            built_model.native_input_size = definition.native_input_size
        return built_model
    return wrap_backbone(
        built_model,
        definition.name,
        definition.native_input_size,
        mean=definition.normalize_mean,
        std=definition.normalize_std,
    )


def count_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def format_parameter_count(parameter_count: int) -> str:
    if parameter_count >= 1_000_000_000:
        return f"{parameter_count / 1_000_000_000:.2f}B"
    if parameter_count >= 1_000_000:
        return f"{parameter_count / 1_000_000:.1f}M"
    if parameter_count >= 1_000:
        return f"{parameter_count / 1_000:.1f}K"
    return str(parameter_count)


def list_supported_backbone_infos() -> list[tuple[BackboneDefinition, int | None, int, tuple[str, ...]]]:
    infos: list[tuple[BackboneDefinition, int | None, int, tuple[str, ...]]] = []
    for definition in list_supported_backbones():
        built_model = definition.build(use_pretrained=False)
        model = finalize_built_model(built_model, definition)
        infos.append(
            (
                definition,
                get_model_native_input_size(model),
                count_parameters(_get_model_core(model)),
                tuple(name for name, _ in discover_model_blocks(model)),
            )
        )
    return infos


def _module_has_parameters(module: nn.Module) -> bool:
    return any(True for _ in module.parameters())


def _resolve_named_submodule(module: nn.Module, path: str) -> nn.Module | None:
    if not path:
        return module

    current_module = module
    for part in path.split("."):
        if hasattr(current_module, part):
            current_module = getattr(current_module, part)
            continue

        if isinstance(current_module, (nn.Sequential, nn.ModuleList)) and part.isdigit():
            index = int(part)
            if 0 <= index < len(current_module):
                current_module = current_module[index]
                continue
        return None

    return current_module


def _get_model_core(model: nn.Module) -> nn.Module:
    if hasattr(model, "_modules") and "backbone" in model._modules:
        return model._modules["backbone"]
    return model


def _discover_head_module_names(model: nn.Module) -> tuple[str, ...]:
    core_model = _get_model_core(model)

    candidate_names: list[str] = []
    for name in (
        "fc",
        "classifier",
        "head",
        "heads",
        "backbone.fc",
        "backbone.classifier",
        "backbone.head",
        "backbone.heads",
    ):
        if _resolve_named_submodule(core_model, name) is not None:
            candidate_names.append(name)

    aux_logits_fc = _resolve_named_submodule(core_model, "AuxLogits.fc")
    if aux_logits_fc is not None:
        candidate_names.append("AuxLogits.fc")

    if candidate_names:
        return tuple(dict.fromkeys(candidate_names))

    if core_model is not model and any(name in model._modules for name in ("fc", "classifier", "head", "heads")):
        return ()

    top_level_children = [
        child_name
        for child_name, child in core_model.named_children()
        if _module_has_parameters(child)
    ]
    if top_level_children:
        return (top_level_children[-1],)
    return ()


def _discover_head_modules(model: nn.Module) -> list[nn.Module]:
    core_model = _get_model_core(model)
    modules: list[nn.Module] = []
    for head_name in _discover_head_module_names(model):
        head_module = _resolve_named_submodule(core_model, head_name)
        if head_module is not None:
            modules.append(head_module)
    return modules


def _should_expand_block(
    child_name: str,
    child_module: nn.Module,
    grandchildren: list[tuple[str, nn.Module]],
) -> bool:
    if not grandchildren:
        return False
    if child_name in {"backbone", "features", "layers", "blocks", "stages"}:
        return True
    return False


def discover_model_blocks(model: nn.Module) -> list[tuple[str, nn.Module]]:
    core_model = _get_model_core(model)
    head_names = set(_discover_head_module_names(model))
    top_level_head_names = {name for name in head_names if "." not in name}
    nested_head_roots = {
        name.split(".")[0]
        for name in head_names
        if "." in name and name.split(".")[0] != "backbone"
    }
    blocks: list[tuple[str, nn.Module]] = []

    for child_name, child_module in core_model.named_children():
        if child_name in {"resize", "normalize", "classifier"}:
            continue
        if (
            child_name in top_level_head_names
            or child_name in nested_head_roots
            or not _module_has_parameters(child_module)
        ):
            continue

        grandchildren = [
            (grandchild_name, grandchild_module)
            for grandchild_name, grandchild_module in child_module.named_children()
            if f"{child_name}.{grandchild_name}" not in head_names
            and _module_has_parameters(grandchild_module)
        ]

        if _should_expand_block(child_name, child_module, grandchildren):
            for grandchild_name, grandchild_module in grandchildren:
                blocks.append((f"{child_name}.{grandchild_name}", grandchild_module))
        else:
            blocks.append((child_name, child_module))

    return blocks


def _find_block_module(model: nn.Module, block_name: str) -> nn.Module | None:
    for current_name, block in discover_model_blocks(model):
        if current_name == block_name:
            return block
    return None


def get_model_display_name(model: nn.Module) -> str:
    if hasattr(model, "backbone_name"):
        return str(model.backbone_name)
    return type(model).__name__


def get_model_input_contract(model: nn.Module) -> str:
    if getattr(model, "_ad_safe_embedded_preprocess", False):
        return "(batch, 3, 299, 299)"
    return "depends on checkpoint architecture"


def get_model_native_input_size(model: nn.Module) -> int | None:
    if hasattr(model, "native_input_size"):
        return int(model.native_input_size)
    resize_module = getattr(model, "resize", None)
    if isinstance(resize_module, nn.Upsample) and resize_module.size is not None:
        if isinstance(resize_module.size, tuple):
            return int(resize_module.size[0])
        return int(resize_module.size)
    if isinstance(resize_module, nn.Identity):
        return IMAGE_SIZE
    return None


def resolve_unfreeze_blocks(
    model: nn.Module,
    *,
    unfreeze_all: bool,
    unfreeze_top: int,
    unfreeze: tuple[str, ...],
) -> tuple[str, ...]:
    available_blocks = [name for name, _ in discover_model_blocks(model)]

    if unfreeze_all:
        selected_blocks = tuple(available_blocks)
    elif unfreeze_top > 0:
        if not available_blocks:
            raise ValueError(
                "This loaded model does not expose configurable backbone blocks; use --unfreeze-all or omit --unfreeze-top"
            )
        if unfreeze_top > len(available_blocks):
            raise ValueError(
                f"Requested {unfreeze_top} unfrozen blocks, but only {len(available_blocks)} are available"
            )
        selected_blocks = tuple(available_blocks[-unfreeze_top:])
    else:
        selected_blocks = unfreeze

    if selected_blocks and not available_blocks:
        raise ValueError(
            "This loaded model does not expose configurable backbone blocks; use --unfreeze-all or omit --unfreeze"
        )

    unknown_blocks = [block_name for block_name in selected_blocks if block_name not in available_blocks]
    if unknown_blocks:
        raise ValueError(
            f"Unknown backbone block(s): {', '.join(unknown_blocks)}. Available blocks: {', '.join(available_blocks)}"
        )

    missing_modules = [
        block_name for block_name in selected_blocks if _find_block_module(model, block_name) is None
    ]
    if missing_modules:
        raise ValueError(
            f"Configured block(s) are not present on the current model instance: {', '.join(missing_modules)}"
        )

    return selected_blocks


def make_model(base_model: str = "inception_v3") -> nn.Module:
    definition = SUPPORTED_BACKBONES[base_model]
    built_model = definition.build(use_pretrained=True)
    model = finalize_built_model(built_model, definition)
    configure_trainable_layers(model, unfreeze=())
    return model.to(DEVICE)


def configure_trainable_layers(
    model: nn.Module,
    unfreeze: tuple[str, ...] = (),
) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False

    for block_name in unfreeze:
        block = _find_block_module(model, block_name)
        if block is None:
            raise ValueError(
                f"Configured block '{block_name}' is not present on model {get_model_display_name(model)}"
            )
        for parameter in block.parameters():
            parameter.requires_grad = True

    for head_module in _discover_head_modules(model):
        for parameter in head_module.parameters():
            parameter.requires_grad = True


def describe_model(model: nn.Module, config: TrainingConfig) -> None:
    available_blocks = [name for name, _ in discover_model_blocks(model)]
    active_blocks = list(config.unfreeze)

    print(f"Base model: {get_model_display_name(model)}")
    native_input_size = get_model_native_input_size(model)
    if native_input_size is not None:
        print(f"Embedded resize target: {native_input_size}")
    print(f"Saved model input contract: {get_model_input_contract(model)}")
    print("Saved model output contract: (batch, 2) logits")
    print(f"Available backbone blocks: {len(available_blocks)}")
    print(f"Backbone blocks: {', '.join(available_blocks) if available_blocks else '<none>'}")
    print(
        "Unfrozen backbone blocks: "
        + (", ".join(active_blocks) if active_blocks else "<head only>")
    )


def get_dataset_classes(dataset: Dataset) -> list[str]:
    current_dataset = dataset
    while isinstance(current_dataset, Subset):
        current_dataset = current_dataset.dataset

    classes = getattr(current_dataset, "classes", None)
    if classes is None:
        raise AttributeError("Dataset does not expose class names via a 'classes' attribute")
    return list(classes)


def split_train_dataset(
    dataset: datasets.ImageFolder,
    valid_ratio: float = VALID_RATIO,
    test_ratio: float = TEST_RATIO,
) -> tuple[Subset, Subset, Subset]:
    if valid_ratio < 0 or test_ratio < 0:
        raise ValueError("Split ratios must be non-negative")
    if valid_ratio + test_ratio >= 1:
        raise ValueError("valid_ratio + test_ratio must be less than 1")

    dataset_targets = getattr(dataset, "targets", None)
    if dataset_targets is None:
        raise AttributeError("Dataset does not expose class targets via a 'targets' attribute")

    class_to_indices: dict[int, list[int]] = {}
    for sample_index, class_index in enumerate(dataset_targets):
        class_to_indices.setdefault(int(class_index), []).append(sample_index)

    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []

    for class_indices in class_to_indices.values():
        random.shuffle(class_indices)
        val_count = int(len(class_indices) * valid_ratio)
        test_count = int(len(class_indices) * test_ratio)

        val_indices.extend(class_indices[:val_count])
        test_indices.extend(class_indices[val_count:val_count + test_count])
        train_indices.extend(class_indices[val_count + test_count:])

    random.shuffle(train_indices)
    random.shuffle(val_indices)
    random.shuffle(test_indices)

    return (
        Subset(dataset, train_indices),
        Subset(dataset, val_indices),
        Subset(dataset, test_indices),
    )


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


def compute_distillation_loss(
    student_logits: Tensor,
    teacher_logits: Tensor,
    *,
    temperature: float,
) -> Tensor:
    student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
    return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature ** 2)


def compute_supervised_training_loss(
    *,
    student_logits: Tensor,
    labels: Tensor,
    criterion: nn.Module,
    config: TrainingConfig,
    teacher_logits: Tensor | None,
) -> tuple[Tensor, Tensor | None]:
    hard_loss = criterion(student_logits, labels)
    if teacher_logits is None:
        return hard_loss, None

    distillation_loss = compute_distillation_loss(
        student_logits,
        teacher_logits,
        temperature=config.distillation_temperature,
    )
    combined_loss = (
        (1.0 - config.distillation_alpha) * hard_loss
        + config.distillation_alpha * distillation_loss
    )
    return combined_loss, distillation_loss


def load_teacher_model(teacher_model_path: str | None) -> nn.Module | None:
    if teacher_model_path is None:
        return None

    teacher_model = load_model(Path(teacher_model_path))
    teacher_model.eval()
    for parameter in teacher_model.parameters():
        parameter.requires_grad_(False)
    return teacher_model


def validate_teacher_logits(logits: Tensor, batch_size: int) -> None:
    if logits.ndim != 2 or logits.shape[0] != batch_size or logits.shape[1] != len(CLASS_NAMES):
        raise ValueError(
            "Teacher model must return logits with shape "
            f"(batch, {len(CLASS_NAMES)}), got {tuple(logits.shape)}"
        )


def precompute_teacher_logits(
    *,
    teacher_model_path: str | None,
    dataset: Dataset,
    batch_size: int,
) -> Tensor | None:
    teacher_model = load_teacher_model(teacher_model_path)
    if teacher_model is None:
        return None

    print(f"Precomputing teacher logits: {teacher_model_path}")
    data_loader, = make_data_loader(dataset, batch_size=batch_size, shuffle=False)
    logits_batches: list[Tensor] = []
    teacher_model.eval()

    with torch.inference_mode():
        for images, _ in tqdm(data_loader, desc="Teacher logits", leave=False):
            images, = to_device(images)
            logits = forward_logits(teacher_model, images)
            validate_teacher_logits(logits, images.shape[0])
            logits_batches.append(logits.detach())

    del teacher_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not logits_batches:
        raise ValueError("Cannot precompute teacher logits for an empty training dataset")
    return torch.cat(logits_batches, dim=0).to(DEVICE)


def extend_dataset_with_adversarial_samples(
    *,
    model: nn.Module,
    dataset: PreparedTrainingDataset,
    criterion: nn.Module,
    config: TrainingConfig,
) -> int:
    if config.adv_steps <= 0:
        raise ValueError("adv_steps must be positive when adversarial training is enabled")

    data_loader, = make_data_loader(
        dataset.dataset,
        batch_size=config.batch_size,
        shuffle=False,
    )
    added_count = 0
    seen_count = 0
    correct_count = 0
    was_training = model.training
    model.eval()

    for images, labels in tqdm(data_loader, desc="Adversarial dataset", leave=False):
        images, labels = to_device(images, labels)
        batch_size = images.shape[0]
        with torch.inference_mode():
            clean_logits = forward_logits(model, images)
            clean_predictions = clean_logits.argmax(dim=1)
        correct_mask = clean_predictions == labels
        correct_count += int(correct_mask.sum().item())

        if correct_mask.any():
            correct_images = images[correct_mask]
            correct_labels = labels[correct_mask]
            source_indices = torch.arange(
                seen_count,
                seen_count + batch_size,
                device=labels.device,
            )[correct_mask]
            adversarial_images = generate_adversarial_perturbation(
                model=model,
                x_original=correct_images,
                criterion=criterion,
                target_labels=correct_labels,
                epsilon=config.adv_epsilon,
                num_steps=config.adv_steps,
            )
            with torch.inference_mode():
                adversarial_logits = forward_logits(model, adversarial_images)
                adversarial_predictions = adversarial_logits.argmax(dim=1)
            successful_mask = adversarial_predictions != correct_labels

            for image, label, source_index in zip(
                adversarial_images[successful_mask],
                correct_labels[successful_mask],
                source_indices[successful_mask],
                strict=True,
            ):
                teacher_logits = (
                    dataset.get_teacher_logits(int(source_index.item()))
                )
                dataset.add_sample(
                    image=image,
                    label=int(label.item()),
                    teacher_logits=teacher_logits,
                )
                added_count += 1

        seen_count += batch_size

    model.train(was_training)
    print(
        "Adversarial dataset extension: "
        f"clean={seen_count}, correct={correct_count}, added={added_count}"
    )
    return added_count


def prepare_training_dataset(
    *,
    model: nn.Module,
    full_train_dataset: Dataset,
    config: TrainingConfig,
    criterion: nn.Module,
) -> PreparedTrainingDataset:
    prepared_dataset = PreparedTrainingDataset(full_train_dataset)
    teacher_logits = precompute_teacher_logits(
        teacher_model_path=config.teacher_model_path,
        dataset=full_train_dataset,
        batch_size=config.batch_size,
    )
    prepared_dataset.set_teacher_logits(teacher_logits)

    if teacher_logits is not None:
        print(f"Using cached teacher logits: {config.teacher_model_path}")
        print(f"Distillation alpha: {config.distillation_alpha}")
        print(f"Distillation temperature: {config.distillation_temperature}")

    if config.adversarial:
        extend_dataset_with_adversarial_samples(
            model=model,
            dataset=prepared_dataset,
            criterion=criterion,
            config=config,
        )
    return prepared_dataset


def train_model_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    config: TrainingConfig,
) -> tuple[float, float | None]:
    model.train()
    total_loss = 0.0
    total_distill_loss = 0.0
    has_teacher_logits = False

    for batch in tqdm(
        train_loader,
        desc="Training",
        leave=False,
    ):
        if config.teacher_model_path is not None:
            images, labels, teacher_logits = batch
            has_teacher_logits = True
            teacher_logits, = to_device(teacher_logits)
        else:
            images, labels = batch
            teacher_logits = None
        images, labels = to_device(images, labels)
        optimizer.zero_grad(set_to_none=True)
        clean_logits = forward_logits(model, images)
        hard_clean_loss = criterion(clean_logits, labels)
        clean_loss, distill_loss = compute_supervised_training_loss(
            student_logits=clean_logits,
            labels=labels,
            criterion=criterion,
            config=config,
            teacher_logits=teacher_logits,
        )
        loss = clean_loss
        if distill_loss is not None:
            total_distill_loss += distill_loss.item()

        loss.backward()
        optimizer.step()
        total_loss += hard_clean_loss.item()

    avg_train_loss = total_loss / len(train_loader)
    avg_distill_train_loss = total_distill_loss / len(train_loader) if has_teacher_logits else None
    return avg_train_loss, avg_distill_train_loss


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    config: TrainingConfig,
    best_model_path: Path,
    split_round: int,
    global_epoch_start: int,
    training_start_time: float,
    epoch_end_handlers: tuple[EpochEndHandler, ...] = (),
) -> tuple[nn.Module, list[TrainingHistoryEntry]]:
    print("Starting training.")
    best_epoch = 0
    patience_counter = 0
    best_model_path = Path(best_model_path)
    comparator = DefaultValidationMetricComparator()
    history: list[TrainingHistoryEntry] = []
    best_metrics = evaluate_validation_score(
        model=model,
        val_loader=val_loader,
    )
    best_score = comparator.score(best_metrics)
    baseline_elapsed = time.time() - training_start_time
    history.append(
        TrainingHistoryEntry(
            split_round=split_round,
            epoch_in_round=0,
            global_epoch=global_epoch_start + 0.5,
            train_loss=None,
            val_accuracy=best_metrics.accuracy,
            val_auc=best_metrics.auc,
            val_nll=best_metrics.nll,
            val_unsafe_recall=best_metrics.unsafe_recall,
            val_avg_wrong_conf=best_metrics.avg_wrong_conf,
            score=best_score,
            elapsed_seconds=baseline_elapsed,
            is_baseline=True,
            is_new_best=False,
        )
    )
    print(f"Initial val metrics: {best_metrics}")

    for epoch in range(1, config.epochs + 1):
        avg_train_loss, avg_distill_train_loss = train_model_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            config=config,
        )
        val_metrics = evaluate_validation_score(
            model=model,
            val_loader=val_loader,
        )
        score = comparator.score(val_metrics)
        comparison = comparator.compare(val_metrics, best_metrics)
        global_epoch = global_epoch_start + epoch
        elapsed = time.time() - training_start_time
        log_parts = [
            f"Split {split_round}/{config.resplit_runs}",
            f"Epoch {epoch}/{config.epochs}",
            f"global_epoch={global_epoch}",
            f"train={avg_train_loss:.4f}",
            f"val={val_metrics.accuracy:.4f}",
        ]
        if val_metrics.auc is not None:
            log_parts.append(f"auc={val_metrics.auc:.4f}")
        log_parts.append(f"nll={val_metrics.nll:.4f}")
        if avg_distill_train_loss is not None:
            log_parts.append(f"distill_train={avg_distill_train_loss:.4f}")
        log_parts.extend(
            [
                f"best={best_score:.4f} @ epoch {best_epoch}",
                f"comparison={comparison.value}",
                f"time={elapsed:.2f}s",
            ]
        )

        print(" | ".join(log_parts))

        is_new_best = comparison == ValidationComparison.BETTER
        entry = TrainingHistoryEntry(
            split_round=split_round,
            epoch_in_round=epoch,
            global_epoch=global_epoch,
            train_loss=avg_train_loss,
            val_accuracy=val_metrics.accuracy,
            val_auc=val_metrics.auc,
            val_nll=val_metrics.nll,
            val_unsafe_recall=val_metrics.unsafe_recall,
            val_avg_wrong_conf=val_metrics.avg_wrong_conf,
            score=score,
            elapsed_seconds=elapsed,
            is_baseline=False,
            is_new_best=is_new_best,
        )
        history.append(entry)

        if is_new_best:
            best_metrics = val_metrics
            best_score = score
            best_epoch = epoch
            patience_counter = 0
            save_model(model, best_model_path)
        elif comparison == ValidationComparison.EQUIVALENT:
            patience_counter = 0
        else:
            patience_counter += 1

        for handler in epoch_end_handlers:
            handler.on_epoch_end(entry)

        if patience_counter >= config.patience:
            print(f"Early stopping at epoch {epoch} (patience={config.patience})")
            break

    if patience_counter > 0:
        model = load_model(best_model_path)
        if best_epoch == 0:
            print(
                f"Loaded best baseline model "
                f"(score: {best_score:.4f})"
            )
        else:
            print(
                f"Loaded best model from epoch {best_epoch} "
                f"(score: {best_score:.4f})"
            )
    else:
        if best_epoch == 0:
            print(
                f"Best baseline model already in memory "
                f"(score: {best_score:.4f})"
            )
        else:
            print(
                f"Best model already in memory from epoch {best_epoch} "
                f"(score: {best_score:.4f})"
            )

    return model, history


def train_model_across_resplits(
    *,
    model: nn.Module,
    full_train_dataset: Dataset,
    config: TrainingConfig,
    best_model_path: Path,
    criterion: nn.Module | None = None,
    training_start_time: float | None = None,
    split_log_prefix: str = "",
    epoch_end_handlers: tuple[EpochEndHandler, ...] = (),
) -> tuple[nn.Module, list[TrainingHistoryEntry]]:
    criterion = nn.CrossEntropyLoss() if criterion is None else criterion
    prepared_train_dataset = prepare_training_dataset(
        model=model,
        full_train_dataset=full_train_dataset,
        config=config,
        criterion=criterion,
    )
    training_start_time = time.time() if training_start_time is None else training_start_time
    history: list[TrainingHistoryEntry] = []
    completed_epochs = 0
    split_prefix = f"{split_log_prefix} " if split_log_prefix else ""

    for split_round in range(1, config.resplit_runs + 1):
        print(f"\n{split_prefix}Split round {split_round}/{config.resplit_runs}")
        train_dataset, val_dataset, test_dataset = split_train_dataset(prepared_train_dataset)
        train_loader, = make_data_loader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            pin_memory=(
                False
                if prepared_train_dataset.teacher_logits is not None
                and prepared_train_dataset.teacher_logits.device.type == "cuda"
                else None
            ),
        )
        val_loader, = make_data_loader(
            LabeledDatasetView(val_dataset),
            batch_size=config.batch_size,
            shuffle=False,
        )
        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples: {len(val_dataset)}")
        print(f"Test samples: {len(test_dataset)}")
        split_learning_rate = get_learning_rate_for_split(config, split_round)
        print(f"Learning rate for split {split_round}: {split_learning_rate}")

        optimizer = optim.Adam(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=split_learning_rate,
        )
        model, round_history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            config=config,
            best_model_path=best_model_path,
            split_round=split_round,
            global_epoch_start=completed_epochs,
            training_start_time=training_start_time,
            epoch_end_handlers=epoch_end_handlers,
        )
        history.extend(round_history)
        completed_epochs += sum(1 for entry in round_history if not entry.is_baseline)

    return model, history


def save_model(model: nn.Module, filename: Path) -> None:
    filename = Path(filename)
    tmp_filename = filename.with_name(f"{filename.name}.tmp")
    print(f"Saving model to {filename}...")

    tmp_filename.unlink(missing_ok=True)
    try:
        torch.save(model, tmp_filename)
        os.replace(tmp_filename, filename)
    except Exception:
        tmp_filename.unlink(missing_ok=True)
        raise


def load_model(filename: Path) -> nn.Module:
    print(f"Loading model from {filename}...")
    loaded_model = torch.load(filename, map_location=DEVICE, weights_only=False)
    if not isinstance(loaded_model, nn.Module):
        raise TypeError(f"Serialized object at {filename} is not a torch.nn.Module")
    loaded_model = loaded_model.to(DEVICE)
    loaded_model.eval()
    return loaded_model


def generate_adversarial_perturbation(
    model: nn.Module,
    x_original: Tensor,
    criterion: nn.Module,
    target_labels: Tensor | None = None,
    epsilon: float = 0.15,
    num_steps: int = DEFAULT_ADV_STEPS,
) -> Tensor:
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")

    x_adv = x_original.detach().clone()
    step_size = epsilon / num_steps

    if target_labels is None:
        raise ValueError("target_labels must be provided for adversarial perturbation")

    was_training = model.training
    model.eval()

    for _ in range(num_steps):
        x_adv.requires_grad_(True)
        model.zero_grad(set_to_none=True)
        logits = forward_logits(model, x_adv)
        loss = criterion(logits, target_labels)
        grad = torch.autograd.grad(loss, x_adv)[0]

        with torch.no_grad():
            x_adv = x_adv + step_size * grad.sign()
            x_adv = torch.clamp(x_adv, x_original - epsilon, x_original + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
            x_adv = x_adv.detach()

    model.train(was_training)
    return x_adv


def evaluate_samples(model: nn.Module, *samples: Tensor) -> tuple[tuple[Tensor, Tensor], ...]:
    model.eval()
    evaluations: list[tuple[Tensor, Tensor]] = []

    with torch.inference_mode():
        for sample in samples:
            logits = forward_logits(model, sample)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            confs = probs.gather(1, preds.unsqueeze(1)).squeeze(1)
            evaluations.append((preds, confs))

    return tuple(evaluations)


def generate_target_class_image(
    model: nn.Module,
    x_initial: Tensor,
    criterion: nn.Module,
    target_label: int,
    step_size: float = 0.01,
    num_steps: int = DEFAULT_TARGET_STEPS,
) -> Tensor:
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")

    x_target = TF.gaussian_blur(x_initial.detach().clone(), kernel_size=9)
    target_labels = torch.tensor([target_label], device=x_initial.device)
    was_training = model.training
    model.eval()

    for _ in range(num_steps):
        x_target.requires_grad_(True)
        model.zero_grad(set_to_none=True)
        logits = forward_logits(model, x_target)
        loss = criterion(logits, target_labels)
        grad = torch.autograd.grad(loss, x_target)[0]

        with torch.no_grad():
            # Gradient descent on the target-class loss nudges the image toward that class.
            x_target = x_target - step_size * grad.sign()
            x_target = torch.clamp(x_target, 0.0, 1.0)
            x_target = TF.gaussian_blur(x_target, kernel_size=9)

            # Expand each channel's current dynamic range so contrast is learned as part of the image.
            channel_min = x_target.amin(dim=(2, 3), keepdim=True)
            channel_max = x_target.amax(dim=(2, 3), keepdim=True)
            channel_range = channel_max - channel_min
            x_target = torch.where(
                channel_range > 1e-6,
                (x_target - channel_min) / channel_range,
                x_target,
            ).detach()

    model.train(was_training)
    return x_target


def make_random_image_tensor() -> Tensor:
    return torch.rand((1, 3, IMAGE_SIZE, IMAGE_SIZE), device=DEVICE)


def save_figure(fig: Figure, output_path: Path) -> None:
    output_path = Path(output_path)
    fig.savefig(output_path, dpi=100, bbox_inches="tight")
    print(f"Figure saved to {output_path}")


def format_info_block_lines(
    items: list[tuple[str, str]],
    *,
    width: int = 38,
    indent: int = 2,
) -> str:
    wrapped_lines: list[str] = []
    subsequent_indent = " " * indent

    for key, value in items:
        prefix = f"{key}="
        wrapped_value = textwrap.wrap(
            value,
            width=max(12, width - len(prefix)),
            break_long_words=False,
            break_on_hyphens=False,
            subsequent_indent=subsequent_indent,
        )
        if not wrapped_value:
            wrapped_lines.append(prefix)
            continue

        wrapped_lines.append(f"{prefix}{wrapped_value[0]}")
        wrapped_lines.extend(wrapped_value[1:])

    return "\n".join(wrapped_lines)


def generate_training_history_figure(
    history: list[TrainingHistoryEntry],
    config: TrainingConfig,
    train_split: str,
    seed: int,
    input_model_path: Path | None,
    output_model_path: Path,
) -> Figure:
    if not history:
        raise ValueError("Training history is empty")

    epoch_history = [entry for entry in history if not entry.is_baseline]
    baseline_history = [entry for entry in history if entry.is_baseline]
    global_epochs = [entry.global_epoch for entry in epoch_history]
    val_scores = [entry.val_accuracy for entry in epoch_history]
    best_epochs = [entry.global_epoch for entry in epoch_history if entry.is_new_best]
    best_scores = [entry.score for entry in epoch_history if entry.is_new_best]

    fig = plt.figure(figsize=(13.5, 7))
    gs = fig.add_gridspec(1, 2, width_ratios=(5.8, 0.9), wspace=0.08)
    score_ax = fig.add_subplot(gs[0, 0])
    info_ax = fig.add_subplot(gs[0, 1])

    score_ax.plot(global_epochs, val_scores, marker="o", linewidth=2, label="val accuracy")

    if best_epochs:
        score_ax.scatter(
            best_epochs,
            best_scores,
            color="crimson",
            s=90,
            zorder=5,
            label="new best checkpoint",
        )

    round_starts: dict[int, int] = {}
    for entry in baseline_history:
        round_starts.setdefault(entry.split_round, int(entry.global_epoch + 0.5))

    baseline_label_drawn = False
    for split_round, start_epoch in sorted(round_starts.items()):
        round_entries = [entry for entry in epoch_history if entry.split_round == split_round]
        if round_entries:
            end_epoch = int(round_entries[-1].global_epoch)
        else:
            end_epoch = start_epoch
        baseline_entry = next(entry for entry in baseline_history if entry.split_round == split_round)
        score_ax.hlines(
            y=baseline_entry.score,
            xmin=start_epoch,
            xmax=end_epoch,
            colors="0.35",
            linestyles=":",
            linewidth=2,
            alpha=0.6,
            label="split baseline" if not baseline_label_drawn else None,
        )
        baseline_label_drawn = True

        if split_round > 1:
            score_ax.axvline(start_epoch - 0.5, color="0.75", linestyle="--", linewidth=1)
        score_ax.text(
            start_epoch,
            1.005,
            f"split {split_round}",
            transform=score_ax.get_xaxis_transform(),
            ha="left",
            va="bottom",
            fontsize=9,
        )

    score_ax.set_ylabel("Accuracy / score")
    score_ax.set_xlabel("Global epoch")
    score_ax.grid(True, alpha=0.3)
    score_ax.legend(loc="lower right")
    info_ax.axis("off")

    params_text = format_info_block_lines(
        [
            ("train_split", train_split),
            ("base_model", config.base_model),
            ("epochs", str(config.epochs)),
            ("resplit_runs", str(config.resplit_runs)),
            ("batch_size", str(config.batch_size)),
            ("learning_rate", ",".join(str(value) for value in config.learning_rate)),
            ("learning_rate_multiplier", str(config.learning_rate_multiplier)),
            ("unfreeze", ",".join(config.unfreeze) if config.unfreeze else "<head only>"),
            ("adversarial", str(config.adversarial)),
            ("adv_epsilon", str(config.adv_epsilon)),
            ("adv_steps", str(config.adv_steps)),
            (
                "teacher_model",
                Path(config.teacher_model_path).name if config.teacher_model_path is not None else "<none>",
            ),
            ("distill_alpha", str(config.distillation_alpha)),
            ("distill_temp", str(config.distillation_temperature)),
            ("seed", str(seed)),
            (
                "input_model",
                Path(input_model_path).name if input_model_path is not None else "<new model>",
            ),
            ("output_model", Path(output_model_path).name),
        ],
        width=34,
        indent=2,
    )
    info_ax.text(
        0.02,
        0.98,
        params_text,
        ha="left",
        va="top",
        fontsize=9,
        family="monospace",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "alpha": 0.9},
        transform=info_ax.transAxes,
    )

    fig.suptitle("Training History", y=0.97)
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.11, top=0.86, wspace=0.1)
    plt.close(fig)
    return fig


def generate_adversarial_example(
    model: nn.Module,
    sample_dataset: Dataset,
    epsilon: float = 0.15,
    num_steps: int = DEFAULT_ADV_STEPS,
) -> Figure:
    criterion = nn.CrossEntropyLoss()
    if len(sample_dataset) == 0:
        raise ValueError("Sample dataset is empty")

    sample_index = random.randrange(len(sample_dataset))
    image_tensor, label_idx = sample_dataset[sample_index]
    original_tensor = image_tensor.unsqueeze(0).to(DEVICE)
    class_names = get_dataset_classes(sample_dataset)
    true_label = class_names[label_idx]
    print(f"Using sample index: {sample_index}")

    adversarial_tensor = generate_adversarial_perturbation(
        model=model,
        x_original=original_tensor,
        criterion=criterion,
        target_labels=torch.tensor([CLASS_NAMES.index(true_label)], device=DEVICE),
        epsilon=epsilon,
        num_steps=num_steps,
    )
    (orig_preds, orig_confs), (adv_preds, adv_confs) = evaluate_samples(
        model, original_tensor, adversarial_tensor
    )

    original_img_np = image_tensor.permute(1, 2, 0).numpy()
    adversarial_img_np = adversarial_tensor[0].detach().cpu().permute(1, 2, 0).numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(original_img_np)
    axes[0].set_title(
        f"Original: {true_label}\nPred: {CLASS_NAMES[orig_preds[0]]} ({orig_confs[0]:.3f})"
    )
    axes[0].axis("off")

    axes[1].imshow(adversarial_img_np)
    axes[1].set_title(
        f"Adversarial\nPred: {CLASS_NAMES[adv_preds[0]]} ({adv_confs[0]:.3f})"
    )
    axes[1].axis("off")

    fig.tight_layout()
    plt.close(fig)
    print(f"  True label: {true_label}")
    print(f"  Original prediction: {CLASS_NAMES[orig_preds[0]]} (confidence: {orig_confs[0]:.3f})")
    print(f"  Adversarial prediction: {CLASS_NAMES[adv_preds[0]]} (confidence: {adv_confs[0]:.3f})")
    print(f"  Attack successful: {orig_preds[0] != adv_preds[0]}")
    return fig


def generate_class_reversal_figure(
    model: nn.Module,
    step_size: float = 0.01,
    num_steps: int = DEFAULT_TARGET_STEPS,
) -> Figure:
    criterion = nn.CrossEntropyLoss()
    print("Generating class reversal figure from random pixel initialization...")

    fig, axes = plt.subplots(1, len(CLASS_NAMES), figsize=(6 * len(CLASS_NAMES), 5))
    if len(CLASS_NAMES) == 1:
        axes = [axes]

    initial_tensor = make_random_image_tensor()

    for idx, class_name in enumerate(CLASS_NAMES):
        generated_tensor = generate_target_class_image(
            model=model,
            x_initial=initial_tensor,
            criterion=criterion,
            target_label=idx,
            step_size=step_size,
            num_steps=num_steps,
        )
        with torch.inference_mode():
            logits = forward_logits(model, generated_tensor)
            probs = torch.softmax(logits, dim=1)
            pred = probs.argmax(dim=1).item()
            confidence = probs[0, pred].item()
        generated_img_np = generated_tensor[0].detach().cpu().permute(1, 2, 0).numpy()

        axes[idx].imshow(generated_img_np)
        axes[idx].set_title(
            f"Target: {class_name}\nPred: {CLASS_NAMES[pred]} ({confidence:.3f})"
        )
        axes[idx].axis("off")

    fig.tight_layout()
    plt.close(fig)
    return fig


def generate_test_figure(
    model: nn.Module,
    sample_dataset: Dataset,
) -> Figure:
    if len(sample_dataset) == 0:
        raise ValueError("Sample dataset is empty")

    sample_index = random.randrange(len(sample_dataset))
    model.eval()
    image_tensor, label_idx = sample_dataset[sample_index]
    input_tensor = image_tensor.unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        logits = forward_logits(model, input_tensor)
        probs = torch.softmax(logits, dim=1)
        pred = probs.argmax(dim=1).item()
        confidence = probs[0, pred].item()

    class_names = get_dataset_classes(sample_dataset)
    true_label = class_names[label_idx]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image_tensor.permute(1, 2, 0).numpy())
    ax.set_title(f"True: {true_label}, Pred: {CLASS_NAMES[pred]} ({confidence:.3f})")
    ax.axis("off")
    fig.tight_layout()
    plt.close(fig)
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train ad safety classifier with optional adversarial training."
    )
    parser.add_argument("--list-base-models", action="store_true", help="List supported base models and exit")
    parser.add_argument("--setup", type=str, default=argparse.SUPPRESS, help="Load run configuration from a setup JSON file")
    parser.add_argument("--model-path", type=str, default=argparse.SUPPRESS, help="Load a pre-trained model")
    parser.add_argument("--model-path-last", action="store_true", default=argparse.SUPPRESS, help="Use the newest '*-model.pt' checkpoint in the script directory as --model-path")
    parser.add_argument("--train-split", choices=["train", "val", "test"], default=argparse.SUPPRESS, help="Dataset split folder used as the source for resplitting during training")
    parser.add_argument("--eval-split", choices=["train", "val", "test"], default=argparse.SUPPRESS, help="Dataset split folder used for final metrics and figures; defaults to --train-split")
    parser.add_argument("--base-model", choices=sorted(SUPPORTED_BACKBONES), default=argparse.SUPPRESS, help="Base backbone to use when creating a new model")
    parser.add_argument(
        "--unfreeze-all",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Train all model layers instead of only the classification head",
    )
    parser.add_argument(
        "--unfreeze-top",
        type=int,
        default=argparse.SUPPRESS,
        help="Number of last backbone blocks to unfreeze in addition to the classification head",
    )
    parser.add_argument(
        "--unfreeze",
        type=str,
        default=argparse.SUPPRESS,
        help="Comma-separated backbone block names to unfreeze in addition to the classification head",
    )
    parser.add_argument("--adversarial", action="store_true", default=argparse.SUPPRESS, help="Enable adversarial training")
    parser.add_argument("--adv-epsilon", type=float, default=argparse.SUPPRESS, help="Adversarial perturbation magnitude")
    parser.add_argument("--adv-steps", type=int, default=argparse.SUPPRESS, help="Attack iterations for adversarial batches")
    parser.add_argument("--teacher-model-path", type=str, default=argparse.SUPPRESS, help="Optional checkpoint to use as a frozen teacher during training")
    parser.add_argument("--distillation-alpha", type=float, default=argparse.SUPPRESS, help="Weight for teacher KL loss when --teacher-model-path is provided")
    parser.add_argument("--distillation-temperature", type=float, default=argparse.SUPPRESS, help="Softmax temperature for teacher distillation")
    parser.add_argument("--cooldown-every-epochs", type=int, default=argparse.SUPPRESS, help="Run cooldown after every N global epochs. Use 0 to disable periodic cooldown.")
    parser.add_argument("--cooldown-seconds", type=float, default=argparse.SUPPRESS, help="Maximum cooldown duration in seconds. Required for enabled cooldown.")
    parser.add_argument("--gpu-max-temp", type=int, default=argparse.SUPPRESS, help="Start cooldown after an epoch when GPU temperature is at least this Celsius value. Use 0 to disable.")
    parser.add_argument("--gpu-resume-temp", type=int, default=argparse.SUPPRESS, help="Resume after GPU temperature is at or below this Celsius value. Defaults to --gpu-max-temp minus 5 when omitted.")
    parser.add_argument("--gpu-temp-check-seconds", type=float, default=argparse.SUPPRESS, help="Seconds between GPU temperature checks during cooldown.")
    parser.add_argument("--epochs", type=int, default=argparse.SUPPRESS, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=argparse.SUPPRESS, help="Training batch size")
    parser.add_argument(
        "--learning-rate",
        type=str,
        default=argparse.SUPPRESS,
        help="Optimizer learning rate, or comma-separated learning rates to use per split round",
    )
    parser.add_argument(
        "--learning-rate-multiplier",
        type=float,
        default=argparse.SUPPRESS,
        help="Multiplier applied to a single learning rate after each split round",
    )
    parser.add_argument("--resplit-runs", type=int, default=argparse.SUPPRESS, help="How many times to resplit train data and continue training")
    parser.add_argument("--patience", type=int, default=argparse.SUPPRESS, help="Early stopping patience (epochs to wait for improvement)")
    parser.add_argument("--seed", type=int, default=argparse.SUPPRESS, help="Random seed. Pass 0 for a fresh random seed")
    return parser.parse_args()


def load_dataset(split_name: str) -> datasets.ImageFolder:
    split_dir = DATA_DIR / split_name
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {split_dir}")
    if not any(path.is_file() for path in split_dir.glob("*/*")):
        raise ValueError(f"Dataset directory has no samples: {split_dir}")
    print(f"Loading dataset from {split_dir}...")
    return datasets.ImageFolder(split_dir, transform=IMAGE_TRANSFORM)


def resolve_model_path(model_path_arg: str | None) -> Path | None:
    if model_path_arg is None:
        return None

    model_path = Path(model_path_arg)
    if not model_path.exists() and not model_path.is_absolute():
        script_relative_model_path = SCRIPT_DIR / model_path
        if script_relative_model_path.exists():
            model_path = script_relative_model_path
    if not model_path.exists():
        raise FileNotFoundError(f"Specified model path does not exist: {model_path}")
    return model_path


def resolve_teacher_model_path(teacher_model_path_arg: str | None) -> str | None:
    teacher_model_path = resolve_model_path(teacher_model_path_arg)
    return str(teacher_model_path.resolve()) if teacher_model_path is not None else None


def resolve_latest_model_path() -> Path:
    candidates = sorted(SCRIPT_DIR.glob("*-model.pt"))
    if not candidates:
        raise FileNotFoundError(
            f"No '*-model.pt' checkpoints found in {SCRIPT_DIR}"
        )
    return candidates[-1]


def resolve_setup_path(setup_path_arg: str | None) -> Path | None:
    if setup_path_arg is None:
        return None

    setup_path = Path(setup_path_arg)
    if not setup_path.exists():
        raise FileNotFoundError(f"Specified setup path does not exist: {setup_path}")
    return setup_path


def load_setup_values(setup_path: Path) -> dict[str, object]:
    try:
        setup_data = json.loads(setup_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Setup JSON is invalid: {setup_path}") from exc

    if not isinstance(setup_data, dict):
        raise ValueError("Setup JSON root must be an object")

    training_config = setup_data.get("training_config", {})
    if training_config is None:
        training_config = {}
    if not isinstance(training_config, dict):
        raise ValueError("training_config must be an object when present")

    cooldown_config = setup_data.get("cooldown", {})
    if cooldown_config is None:
        cooldown_config = {}
    if not isinstance(cooldown_config, dict):
        raise ValueError("cooldown must be an object when present")

    def get_optional_string(container: dict[str, object], field_name: str) -> str | None:
        value = container.get(field_name)
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError(f"{field_name} must be a string or null")
        return value

    def get_optional_int(container: dict[str, object], field_name: str) -> int | None:
        value = container.get(field_name)
        if value is None:
            return None
        if not isinstance(value, int):
            raise ValueError(f"{field_name} must be an integer or null")
        return value

    def get_optional_float(container: dict[str, object], field_name: str) -> float | None:
        value = container.get(field_name)
        if value is None:
            return None
        if not isinstance(value, (int, float)):
            raise ValueError(f"{field_name} must be a number or null")
        return float(value)

    def get_optional_bool(container: dict[str, object], field_name: str) -> bool | None:
        value = container.get(field_name)
        if value is None:
            return None
        if not isinstance(value, bool):
            raise ValueError(f"{field_name} must be a boolean or null")
        return value

    learning_rate_value = training_config.get("learning_rate")
    if learning_rate_value is not None:
        learning_rate_value = normalize_learning_rates_value(learning_rate_value)

    unfreeze_value = training_config.get("unfreeze")
    if unfreeze_value is not None:
        unfreeze_value = normalize_unfreeze_value(unfreeze_value)

    return {
        "seed": get_optional_int(setup_data, "seed"),
        "train_split": get_optional_string(setup_data, "train_split"),
        "eval_split": get_optional_string(setup_data, "eval_split"),
        "base_model": get_optional_string(setup_data, "base_model"),
        "model_path": get_optional_string(setup_data, "original_model_path")
        or get_optional_string(setup_data, "model_path"),
        "epochs": get_optional_int(training_config, "epochs"),
        "patience": get_optional_int(training_config, "patience"),
        "batch_size": get_optional_int(training_config, "batch_size"),
        "learning_rate": learning_rate_value,
        "learning_rate_multiplier": get_optional_float(training_config, "learning_rate_multiplier"),
        "resplit_runs": get_optional_int(training_config, "resplit_runs"),
        "unfreeze": unfreeze_value,
        "adversarial": get_optional_bool(training_config, "adversarial"),
        "adv_epsilon": get_optional_float(training_config, "adv_epsilon"),
        "adv_steps": get_optional_int(training_config, "adv_steps"),
        "teacher_model_path": get_optional_string(training_config, "teacher_model_path"),
        "distillation_alpha": get_optional_float(training_config, "distillation_alpha"),
        "distillation_temperature": get_optional_float(training_config, "distillation_temperature"),
        "cooldown_every_epochs": get_optional_int(cooldown_config, "every_epochs"),
        "cooldown_seconds": get_optional_float(cooldown_config, "seconds"),
        "gpu_max_temp": get_optional_int(cooldown_config, "gpu_max_temp"),
        "gpu_resume_temp": get_optional_int(cooldown_config, "gpu_resume_temp"),
        "gpu_temp_check_seconds": get_optional_float(cooldown_config, "gpu_temp_check_seconds"),
    }


def merge_setup_and_cli_values(args: argparse.Namespace, setup_values: dict[str, object]) -> dict[str, object]:
    def pick_value(name: str, default: object) -> object:
        if hasattr(args, name):
            return getattr(args, name)
        if name in setup_values and setup_values[name] is not None:
            return setup_values[name]
        return default

    return {
        "setup_path": getattr(args, "setup", None),
        "list_base_models": getattr(args, "list_base_models", False),
        "model_path": pick_value("model_path", None),
        "model_path_last": getattr(args, "model_path_last", False),
        "train_split": pick_value("train_split", None),
        "eval_split": pick_value("eval_split", None),
        "base_model": pick_value("base_model", "inception_v3"),
        "unfreeze_all": pick_value("unfreeze_all", False),
        "unfreeze_top": pick_value("unfreeze_top", 0),
        "unfreeze": pick_value("unfreeze", ()),
        "adversarial": pick_value("adversarial", False),
        "adv_epsilon": pick_value("adv_epsilon", DEFAULT_ADV_EPSILON),
        "adv_steps": pick_value("adv_steps", DEFAULT_ADV_STEPS),
        "teacher_model_path": pick_value("teacher_model_path", None),
        "distillation_alpha": pick_value("distillation_alpha", DEFAULT_DISTILLATION_ALPHA),
        "distillation_temperature": pick_value("distillation_temperature", DEFAULT_DISTILLATION_TEMPERATURE),
        "cooldown_every_epochs": pick_value("cooldown_every_epochs", DEFAULT_COOLDOWN_EVERY_EPOCHS),
        "cooldown_seconds": pick_value("cooldown_seconds", DEFAULT_COOLDOWN_SECONDS),
        "gpu_max_temp": pick_value("gpu_max_temp", DEFAULT_GPU_MAX_TEMP),
        "gpu_resume_temp": pick_value("gpu_resume_temp", DEFAULT_GPU_RESUME_TEMP),
        "gpu_temp_check_seconds": pick_value("gpu_temp_check_seconds", DEFAULT_GPU_TEMP_CHECK_SECONDS),
        "epochs": pick_value("epochs", DEFAULT_EPOCHS),
        "batch_size": pick_value("batch_size", None),
        "learning_rate": pick_value("learning_rate", (DEFAULT_LR,)),
        "learning_rate_multiplier": pick_value("learning_rate_multiplier", 1.0),
        "resplit_runs": pick_value("resplit_runs", DEFAULT_RESPLIT_RUNS),
        "patience": pick_value("patience", DEFAULT_PATIENCE),
        "seed": pick_value("seed", None),
    }


def resolve_effective_seed(seed_value: object) -> int:
    if seed_value is None:
        return make_seed()
    if not isinstance(seed_value, int):
        raise ValueError("seed must be an integer or null")
    if seed_value == 0:
        return make_seed()
    return seed_value


def build_training_config(values: dict[str, object]) -> TrainingConfig:
    batch_size_value = values["batch_size"]
    batch_size = batch_size_value if batch_size_value is not None else get_default_batch_size()
    learning_rates = normalize_learning_rates_value(values["learning_rate"])
    learning_rate_multiplier = float(values["learning_rate_multiplier"])
    explicit_unfreeze = normalize_unfreeze_value(values["unfreeze"])
    epochs = int(values["epochs"])
    adv_steps = int(values["adv_steps"])
    adv_epsilon = float(values["adv_epsilon"])
    teacher_model_path = values["teacher_model_path"]
    distillation_alpha = float(values["distillation_alpha"])
    distillation_temperature = float(values["distillation_temperature"])
    resplit_runs = int(values["resplit_runs"])
    patience = int(values["patience"])
    unfreeze_top = int(values["unfreeze_top"])
    unfreeze_all = bool(values["unfreeze_all"])

    if batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if epochs <= 0:
        raise ValueError("--epochs must be positive")
    if adv_steps < 0:
        raise ValueError("--adv-steps must be non-negative")
    if adv_epsilon < 0:
        raise ValueError("--adv-epsilon must be non-negative")
    if teacher_model_path is not None and not isinstance(teacher_model_path, str):
        raise ValueError("--teacher-model-path must be a string or null")
    if distillation_alpha < 0 or distillation_alpha > 1:
        raise ValueError("--distillation-alpha must be between 0 and 1")
    if distillation_temperature <= 0:
        raise ValueError("--distillation-temperature must be positive")
    if resplit_runs <= 0:
        raise ValueError("--resplit-runs must be positive")
    if patience < 0:
        raise ValueError("--patience must be non-negative")
    if unfreeze_top < 0:
        raise ValueError("--unfreeze-top must be non-negative")
    if learning_rate_multiplier <= 0:
        raise ValueError("--learning-rate-multiplier must be positive")
    if len(learning_rates) > 1 and values["learning_rate_multiplier"] != 1.0:
        raise ValueError("--learning-rate-multiplier cannot be used with multiple --learning-rate values")
    unfreeze_mode_count = int(unfreeze_all) + int(unfreeze_top > 0) + int(bool(explicit_unfreeze))
    if unfreeze_mode_count > 1:
        raise ValueError("Use only one of --unfreeze-all, --unfreeze-top, or --unfreeze")

    return TrainingConfig(
        base_model=str(values["base_model"]),
        epochs=epochs,
        patience=patience,
        batch_size=batch_size,
        learning_rate=learning_rates,
        learning_rate_multiplier=learning_rate_multiplier,
        resplit_runs=resplit_runs,
        unfreeze=explicit_unfreeze,
        adversarial=bool(values["adversarial"]),
        adv_epsilon=adv_epsilon,
        adv_steps=adv_steps,
        teacher_model_path=teacher_model_path,
        distillation_alpha=distillation_alpha,
        distillation_temperature=distillation_temperature,
    )


def build_cooldown_config(
    *,
    every_epochs: int = DEFAULT_COOLDOWN_EVERY_EPOCHS,
    seconds: float = DEFAULT_COOLDOWN_SECONDS,
    gpu_max_temp: int = DEFAULT_GPU_MAX_TEMP,
    gpu_resume_temp: int = DEFAULT_GPU_RESUME_TEMP,
    gpu_temp_check_seconds: float = DEFAULT_GPU_TEMP_CHECK_SECONDS,
) -> CooldownConfig:
    if gpu_max_temp > 0 and gpu_resume_temp == 0:
        gpu_resume_temp = gpu_max_temp - 5

    config = CooldownConfig(
        every_epochs=int(every_epochs),
        seconds=float(seconds),
        gpu_max_temp=int(gpu_max_temp),
        gpu_resume_temp=int(gpu_resume_temp),
        gpu_temp_check_seconds=float(gpu_temp_check_seconds),
    )
    if config.every_epochs < 0:
        raise ValueError("--cooldown-every-epochs must be non-negative")
    if config.seconds < 0:
        raise ValueError("--cooldown-seconds must be non-negative")
    if config.gpu_max_temp < 0:
        raise ValueError("--gpu-max-temp must be non-negative")
    if config.gpu_resume_temp < 0:
        raise ValueError("--gpu-resume-temp must be non-negative")
    if config.gpu_temp_check_seconds <= 0:
        raise ValueError("--gpu-temp-check-seconds must be positive")
    if config.enabled and config.seconds <= 0:
        raise ValueError("--cooldown-seconds must be positive when cooldown is enabled")
    if config.uses_temperature and config.gpu_resume_temp >= config.gpu_max_temp:
        raise ValueError("--gpu-resume-temp must be lower than --gpu-max-temp")
    return config


def config_to_json_dict(config: TrainingConfig) -> dict[str, object]:
    return {
        "epochs": config.epochs,
        "patience": config.patience,
        "batch_size": config.batch_size,
        "learning_rate": list(config.learning_rate),
        "learning_rate_multiplier": config.learning_rate_multiplier,
        "resplit_runs": config.resplit_runs,
        "unfreeze": list(config.unfreeze),
        "adversarial": config.adversarial,
        "adv_epsilon": config.adv_epsilon,
        "adv_steps": config.adv_steps,
        "teacher_model_path": config.teacher_model_path,
        "distillation_alpha": config.distillation_alpha,
        "distillation_temperature": config.distillation_temperature,
    }


def path_to_json(path: Path | None) -> str | None:
    if path is None:
        return None
    return str(path.resolve())


def build_setup_payload(
    *,
    timestamp: str,
    seed: int,
    train_split: str,
    eval_split: str,
    config: TrainingConfig,
    cooldown_config: CooldownConfig,
    original_model_path: Path | None,
    training_checkpoint_path: Path,
    training_history_figure_path: Path,
) -> dict[str, object]:
    return {
        "timestamp": timestamp,
        "seed": seed,
        "train_split": train_split,
        "eval_split": eval_split,
        "base_model": config.base_model,
        "original_model_path": path_to_json(original_model_path),
        "training_checkpoint_path": path_to_json(training_checkpoint_path),
        "training_history_figure_path": path_to_json(training_history_figure_path),
        "training_config": config_to_json_dict(config),
        "cooldown": cooldown_config.to_json(),
    }


def write_setup_file(payload: dict[str, object], output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Setup saved to {output_path}")


def finalize_training_config(
    model: nn.Module,
    config: TrainingConfig,
    *,
    unfreeze_all: bool,
    unfreeze_top: int,
) -> TrainingConfig:
    selected_unfreeze_blocks = resolve_unfreeze_blocks(
        model,
        unfreeze_all=unfreeze_all,
        unfreeze_top=unfreeze_top,
        unfreeze=config.unfreeze,
    )
    return TrainingConfig(
        base_model=get_model_display_name(model),
        epochs=config.epochs,
        patience=config.patience,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        learning_rate_multiplier=config.learning_rate_multiplier,
        resplit_runs=config.resplit_runs,
        unfreeze=selected_unfreeze_blocks,
        adversarial=config.adversarial,
        adv_epsilon=config.adv_epsilon,
        adv_steps=config.adv_steps,
        teacher_model_path=config.teacher_model_path,
        distillation_alpha=config.distillation_alpha,
        distillation_temperature=config.distillation_temperature,
    )


def main() -> None:
    args = parse_args()

    if args.list_base_models:
        print("Supported base models:")
        for definition, native_input_size, backbone_parameter_count, block_names in list_supported_backbone_infos():
            print(
                f"- {definition.name}: native_input={native_input_size or definition.native_input_size}, "
                f"backbone_params={format_parameter_count(backbone_parameter_count)}, "
                f"available_blocks={len(block_names)}, "
                f"blocks={', '.join(block_names) if block_names else '<none>'}"
            )
        return

    if hasattr(args, "model_path") and hasattr(args, "model_path_last"):
        raise ValueError("Use only one of --model-path or --model-path-last")

    setup_path = resolve_setup_path(getattr(args, "setup", None))
    setup_values = load_setup_values(setup_path) if setup_path is not None else {}
    merged_values = merge_setup_and_cli_values(args, setup_values)
    config = build_training_config(merged_values)
    cooldown_config = build_cooldown_config(
        every_epochs=int(merged_values["cooldown_every_epochs"]),
        seconds=float(merged_values["cooldown_seconds"]),
        gpu_max_temp=int(merged_values["gpu_max_temp"]),
        gpu_resume_temp=int(merged_values["gpu_resume_temp"]),
        gpu_temp_check_seconds=float(merged_values["gpu_temp_check_seconds"]),
    )
    config = replace(
        config,
        teacher_model_path=resolve_teacher_model_path(config.teacher_model_path),
    )
    unfreeze_all = bool(merged_values["unfreeze_all"])
    unfreeze_top = int(merged_values["unfreeze_top"])
    run_timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    seed = resolve_effective_seed(merged_values["seed"])
    set_seed(seed)
    eval_split = merged_values["eval_split"] or merged_values["train_split"]

    if merged_values["model_path_last"]:
        original_model_path = resolve_latest_model_path()
    else:
        original_model_path = resolve_model_path(merged_values["model_path"])
    print(f"Using device: {DEVICE}")
    print(f"Using seed: {seed}")
    print(f"Using batch size: {config.batch_size}")
    print(f"Using resplit runs: {config.resplit_runs}")
    print(f"Cooldown: {cooldown_config.to_json()}")
    if setup_path is not None:
        print(f"Using setup: {setup_path}")
    print(f"Training source split: {merged_values['train_split']}")
    print(f"Evaluation split: {eval_split}")
    print(f"Original model path: {original_model_path}")

    if merged_values["train_split"] is None and original_model_path is None and eval_split is not None:
        raise ValueError("Provide --train-split to train or --model-path to evaluate an existing model")
    if merged_values["train_split"] is None and original_model_path is None and eval_split is None:
        raise ValueError("Provide --train-split, --eval-split, or --model-path")
    if merged_values["train_split"] is None and original_model_path is not None and eval_split is None:
        info_model = load_model(original_model_path)
        config = finalize_training_config(
            info_model,
            config,
            unfreeze_all=unfreeze_all,
            unfreeze_top=unfreeze_top,
        )
        configure_trainable_layers(
            info_model,
            unfreeze=config.unfreeze,
        )
        describe_model(info_model, config)
        return
    if eval_split is None:
        raise ValueError("--eval-split is required when --train-split is not provided")

    if original_model_path is not None:
        model = load_model(original_model_path)
    else:
        print(f"Creating model from base backbone: {config.base_model}")
        model = make_model(config.base_model)

    config = finalize_training_config(
        model,
        config,
        unfreeze_all=unfreeze_all,
        unfreeze_top=unfreeze_top,
    )

    configure_trainable_layers(
        model,
        unfreeze=config.unfreeze,
    )
    describe_model(model, config)

    training_history: list[TrainingHistoryEntry] = []
    if merged_values["train_split"] is not None:
        full_train_dataset = load_dataset(str(merged_values["train_split"]))
        model_path = SCRIPT_DIR / f"{run_timestamp}-model.pt"
        training_history_path = SCRIPT_DIR / f"{run_timestamp}-training_history.png"
        setup_output_path = SCRIPT_DIR / f"{run_timestamp}-setup.json"
        write_setup_file(
            build_setup_payload(
                timestamp=run_timestamp,
                seed=seed,
                train_split=str(merged_values["train_split"]),
                eval_split=str(eval_split),
                config=config,
                cooldown_config=cooldown_config,
                original_model_path=original_model_path,
                training_checkpoint_path=model_path,
                training_history_figure_path=training_history_path,
            ),
            setup_output_path,
        )
        save_model(model, model_path)
        epoch_end_handlers: tuple[EpochEndHandler, ...] = ()
        if cooldown_config.enabled:
            epoch_end_handlers = (
                CooldownEpochEndHandler(
                    config=cooldown_config,
                    backbone_name=config.base_model,
                    phase_name="main",
                ),
            )
        model, training_history = train_model_across_resplits(
            model=model,
            full_train_dataset=full_train_dataset,
            config=config,
            best_model_path=model_path,
            epoch_end_handlers=epoch_end_handlers,
        )

        save_figure(
            generate_training_history_figure(
                history=training_history,
                config=config,
                train_split=str(merged_values["train_split"]),
                seed=seed,
                input_model_path=original_model_path,
                output_model_path=model_path,
            ),
            training_history_path,
        )

    eval_dataset = load_dataset(eval_split)
    eval_loader, = make_data_loader(eval_dataset, batch_size=config.batch_size, shuffle=False)
    eval_metrics = evaluate_metrics(model, eval_loader, eval_split)
    print(f"Evaluation metrics ({eval_split}): {eval_metrics}")

    print(f"\nGenerating sample figure from {eval_split}...")
    save_figure(
        generate_test_figure(
            model=model,
            sample_dataset=eval_dataset,
        ),
        SCRIPT_DIR / f"{run_timestamp}-test.png",
    )

    print(f"\nGenerating adversarial attack example from {eval_split}...")
    save_figure(
        generate_adversarial_example(
            model=model,
            sample_dataset=eval_dataset,
            epsilon=config.adv_epsilon,
            num_steps=max(config.adv_steps, 1),
        ),
        SCRIPT_DIR / f"{run_timestamp}_adversarial.png",
    )

    print("\nGenerating class reversal figure...")
    save_figure(
        generate_class_reversal_figure(
            model=model,
        ),
        SCRIPT_DIR / f"{run_timestamp}_class_reversal.png",
    )


if __name__ == "__main__":
    main()
