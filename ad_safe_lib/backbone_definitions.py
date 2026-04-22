from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass

from torch import nn
from torchvision import models
from torchvision.models import (
    ConvNeXt_Base_Weights,
    ConvNeXt_Large_Weights,
    ConvNeXt_Tiny_Weights,
    EfficientNet_V2_S_Weights,
    Inception_V3_Weights,
    MaxVit_T_Weights,
    MobileNet_V3_Small_Weights,
    Swin_B_Weights,
    Swin_V2_B_Weights,
    ViT_H_14_Weights,
    ViT_L_16_Weights,
)

from .config import CLASS_NAMES
from .backbone_wrappers import (
    CLIP_MEAN,
    CLIP_STD,
    HF_CLIP_VIT_L_14_MODEL_ID,
    HF_DINOV2_IMAGE_CLASSIFICATION_MODEL_ID,
    HF_DINOV3_VITL16_PRETRAIN_MODEL_ID,
    IMAGENET_MEAN,
    IMAGENET_STD,
    _replace_linear_classifier,
    build_dinov3_features_graph,
    build_hf_logits_graph,
    build_hf_pooler_graph,
    wrap_feature_backbone,
)


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
class MobileNetV3SmallBackboneDefinition(TorchVisionClassifierBackboneDefinition):
    classifier_path = "classifier.-1"

    def build_torchvision_model(self, *, use_pretrained: bool) -> nn.Module:
        return models.mobilenet_v3_small(
            weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1 if use_pretrained else None
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
