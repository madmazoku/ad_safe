# Backbones API

Backbone support is split by responsibility:

- `backbone_definitions.py`: concrete model definitions.
- `backbone_registry.py`: supported backbone registry.
- `backbone_wrappers.py`: preprocessing and FX graph wrappers.
- `backbone_introspection.py`: trainability, block discovery, model descriptions.

Public helpers:

- `SUPPORTED_BACKBONES`
- `list_supported_backbones()`
- `make_model(base_model)`
- `configure_trainable_layers(model, unfreeze=...)`
- `finalize_training_config(...)`
- `list_supported_backbone_infos()`

Datasets produce `(3, 299, 299)` tensors. Built and saved models embed resize and normalization layers, so the external model contract is `(batch, 3, 299, 299)` logits-inference input. Each backbone still has its own native input size internally; `list_supported_backbone_infos()` reports that native size and available trainable blocks.
