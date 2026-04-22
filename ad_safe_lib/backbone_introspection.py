from __future__ import annotations

from torch import nn

from .backbone_definitions import BackboneDefinition
from .backbone_registry import SUPPORTED_BACKBONES, finalize_built_model, list_supported_backbones
from .config import DEVICE, IMAGE_SIZE, TrainingConfig


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
