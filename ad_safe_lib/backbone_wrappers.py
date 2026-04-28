from __future__ import annotations

import builtins
import operator
from collections import OrderedDict

import torch
from torch import Tensor, nn
from torch.fx import Graph, GraphModule
from torchvision import transforms
from .config import CLASS_NAMES, IMAGE_SIZE


IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)
NORMALIZE_TRANSFORM = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
HF_CLIP_VIT_L_14_MODEL_ID = "openai/clip-vit-large-patch14"
HF_DINOV2_IMAGE_CLASSIFICATION_MODEL_ID = "facebook/dinov2-base-imagenet1k-1-layer"
HF_DINOV3_VITL16_PRETRAIN_MODEL_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m"


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


def get_logits(outputs: Tensor | object) -> Tensor:
    return outputs.logits if hasattr(outputs, "logits") else outputs


def forward_logits(model: nn.Module, images: Tensor) -> Tensor:
    if getattr(model, "_ad_safe_embedded_preprocess", False):
        return get_logits(model(images))
    normalized_images = NORMALIZE_TRANSFORM(images)
    return get_logits(model(normalized_images))


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
