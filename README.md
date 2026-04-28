# Ad Safety Challenge

This folder contains the ad-safety training and evaluation code for the ML bootcamp challenge.

## Layout

- `ad_safe_lib/`: reusable library code for data loading, backbones, training, evaluation, metrics, artifacts, and figures.
- `ad_safe.py`: single-model training/evaluation CLI.
- `train_prod_ad_safe_sweep.py`: JSON-driven phase sweep CLI for one backbone.
- `train_prod_ad_safe_backbones.py`: compare several backbones with one shared two-phase recipe.
- `ad_safe_test.py`: evaluate saved checkpoints and write metrics CSV.
- `check_ad_safe_contract.py`: standalone foreign-code contract check; it intentionally does not import `ad_safe_lib`.
- `notebooks/ad_safe_examples/`: lightweight example notebooks.
- `docs/`: architecture, script, and library notes.
- `datasets/`: local dataset storage.
- `artefacts/`: local runtime storage for script inputs and outputs. Its contents are ignored by git.

## Quick Start

Run commands from this `challenge/` directory:

```bash
../venv/bin/python ad_safe.py --help
../venv/bin/python train_prod_ad_safe_sweep.py --help
../venv/bin/python train_prod_ad_safe_backbones.py --help
../venv/bin/python ad_safe_test.py --help
```

Use a small dataset fraction for quick local checks:

```bash
../venv/bin/python ad_safe.py \
  --train-split train \
  --eval-split val \
  --base-model simple_cnn \
  --epochs 1 \
  --resplit-runs 1 \
  --batch-size 8 \
  --train-fraction 0.02 \
  --patience 1 \
  --seed 1
```

## Data And Artifacts

Expected dataset root:

```text
datasets/ml_bootcamp_adsafety_dataset/{train,val,test}/{safe,unsafe}/...
```

Datasets and artefacts are local runtime files. The repository tracks only placeholder directories.

## More Docs

- [Scripts](docs/scripts.md)
- [Architecture](docs/architecture.md)
- [Library API](docs/library.md)
- [Agent Rules](AGENTS.md)

## Checks

```bash
../venv/bin/python -m py_compile \
  ad_safe.py \
  train_prod_ad_safe_sweep.py \
  train_prod_ad_safe_backbones.py \
  ad_safe_test.py \
  check_ad_safe_contract.py \
  ad_safe_lib/*.py
```
