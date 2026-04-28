# Agent Guide

This directory is the ad-safety project root. Keep changes aligned with the clean architecture rules below.

## Core Rules

- Scripts are thin entrypoints. They may parse CLI arguments or JSON, validate user input, resolve paths, and build typed plans/specs.
- Execution lives in `ad_safe_lib`: training, evaluation, reporting, figures, model IO, dataset loading, cooldown, and contract subprocess calls.
- Do not add `argparse` to `ad_safe_lib`.
- Do not add model training loops, `DataLoader` construction, optimizer setup, metrics CSV writing, or terminal table formatting to scripts.
- Do not add legacy compatibility code. When a module is moved or removed, update imports, docs, and notebooks in the same change.
- Keep `check_ad_safe_contract.py` standalone. It simulates foreign code and must not import `ad_safe` or `ad_safe_lib`.
- Enrichment dataset traversal belongs in `ad_safe_lib.enrichment` runners, not in strategies. Strategies should transform a provided sample or batch (`transform_sample` / `generate_batch`) and return derived samples with source positions; progress bars, `DataLoader` iteration, label/logit inheritance, and attaching samples stay in the runner.

## Paths

- `ad_safe_lib.CHALLENGE_DIR` is the canonical project root.
- Datasets live under `datasets/`.
- Runtime script inputs and outputs live under `artefacts/`.
- Sweep JSON files are resolved from the working directory, `artefacts/sweep_configs/`, `artefacts/`, or the challenge root.
- The contents of `datasets/` and `artefacts/` are local runtime data and should stay ignored by git except `.gitkeep` placeholders.

## Notebooks

- Example notebooks live in `notebooks/ad_safe_examples`.
- They should be lightweight demonstrations of library APIs, not production training recipes.
- Run them from `challenge/notebooks/ad_safe_examples`.
- The first code cell should add `../..` to `sys.path`, import `ad_safe_lib`, and use library path constants.
- Do not search parent directories for `ad_safe.py`.

## Before Finishing

When code behavior, CLI options, JSON formats, or public `ad_safe_lib` APIs change, update the markdown knowledge base in the same change. Keep those notes tied to code behavior, not to local generated artifact contents.

When changing enrichment behavior, keep the architecture note in `docs/architecture.md` and the public API note in `docs/library/training.md` aligned with `ad_safe_lib/enrichment.py`.

Run the static checks that match the change:

```bash
../venv/bin/python -m py_compile \
  ad_safe.py \
  train_prod_ad_safe_sweep.py \
  train_prod_ad_safe_backbones.py \
  ad_safe_test.py \
  check_ad_safe_contract.py \
  ad_safe_lib/*.py
```

Architecture rules are intentionally review rules, not substring checks. Read the affected scripts, library modules, docs, and notebooks, then decide whether the responsibility split still matches this guide.

Do not run full training or inference smoke tests unless the user explicitly asks.
