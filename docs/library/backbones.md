# Backbones API

Backbone support is split by responsibility:

- `backbone_definitions.py`: concrete model definitions.
- `backbone_registry.py`: supported backbone registry.
- `backbone_wrappers.py`: preprocessing and FX graph wrappers.
- `backbone_introspection.py`: trainability, block discovery, model descriptions.

Public helpers:

- `SUPPORTED_BACKBONES`
- `make_model(base_model)`
- `configure_trainable_layers(model, unfreeze=...)`
- `finalize_training_config(...)`
- `list_supported_backbone_infos()`

Saved models embed preprocessing and expect input tensors shaped `(batch, 3, 299, 299)`.
