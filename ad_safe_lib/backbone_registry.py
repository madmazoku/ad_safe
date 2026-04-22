from __future__ import annotations

from torch import nn

from .backbone_definitions import (
    BackboneDefinition,
    ClipVitL14BackboneDefinition,
    ConvNeXtBaseBackboneDefinition,
    ConvNeXtLargeBackboneDefinition,
    ConvNeXtTinyBackboneDefinition,
    DinoV2ImageClassificationBackboneDefinition,
    DinoV3VitL16PretrainBackboneDefinition,
    EfficientNetV2SBackboneDefinition,
    InceptionV3BackboneDefinition,
    MaxVitTBackboneDefinition,
    MobileNetV3SmallBackboneDefinition,
    SimpleCnnBackboneDefinition,
    SimpleMlpBackboneDefinition,
    SwinBBackboneDefinition,
    SwinV2BBackboneDefinition,
    TorchVisionClassifierBackboneDefinition,
    VitH14BackboneDefinition,
    VitL16BackboneDefinition,
)
from .backbone_wrappers import CLIP_MEAN, CLIP_STD, wrap_backbone


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
    "mobilenet_v3_small": MobileNetV3SmallBackboneDefinition(
        name="mobilenet_v3_small",
        native_input_size=224,
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
