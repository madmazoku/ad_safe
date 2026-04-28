# Architecture

## Core Rule

Scripts prepare work. `ad_safe_lib` executes work.

CLI scripts may parse arguments, validate JSON, discover local paths, and build typed plans such as `RunPlan` or `EvaluationPlan`. They must not contain training loops, model evaluation loops, DataLoader construction, optimizer setup, metrics CSV writing, or terminal table formatting.

## Library Layers

- `data.py`: dataset loading, stratified subsets, prepared training dataset mutation.
- `training.py`: training loop, teacher logits, and enrichment execution for training subsets.
- `training_runner.py`: executes `RunPlan` jobs/phases and writes training artifacts.
- `evaluation_runner.py`: executes `EvaluationPlan` and evaluates checkpoints.
- `reporting.py`: pure metric formatting, table output, and CSV flatten/write/read helpers.
- `metrics.py`: classification metrics and validation comparator.
- `reversal.py`: class-reversal optimization strategies for diagnostic figures.
- `figures.py`: figure construction and image visualization helpers.
- `backbone_*`: model registry, definitions, wrappers, and trainability introspection.
- `workflows.py`: small reusable high-level helpers for model description and figure artifact generation.

## Enrichment

`ad_safe_lib.enrichment` runners own dataset traversal, `DataLoader` construction, progress bars, source-position tracking, label/logit inheritance, and adding derived samples to `PreparedTrainingDataset`. Enrichment strategies do not iterate over whole datasets. One-to-one transforms implement `transform_sample(image)`, while batch/model-aware transforms implement `generate_batch(...)` and yield `(source_position, derived_image)` pairs.

## Dataset Sources

`DatasetSourceSpec(name, fraction, seed)` is source preparation. A fraction below `1.0` creates a stratified subset before normal train/resplit/evaluation flow. Training and evaluation logic should not care whether a dataset came from disk or from a source subset.

## Artifacts

`datasets/` is local dataset storage. `artefacts/` is local runtime storage for script inputs and outputs. Their contents are ignored by git except placeholder files; documentation should describe their role, not catalog local contents.

## Contract Check

`check_ad_safe_contract.py` simulates foreign code. It must remain standalone and must not import `ad_safe` or `ad_safe_lib`.

## No Legacy

Do not keep deprecated compatibility paths, backup scripts, or unused wrappers. When a module is renamed or split, update scripts, docs, and notebooks in the same change.

## Review Discipline

Architecture is reviewed by reading the code, not by substring checks. When changing a script or library module, inspect the neighboring modules and documentation to make sure responsibilities still line up with this page.

Markdown files are the durable knowledge base for the project. Keep them aligned with code behavior and public APIs; do not use them as inventories of local `artefacts/` contents.
