# Script Usage

Run examples from `challenge/`.

## Single Run

Quick CPU-friendly run:

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

Evaluate an existing checkpoint:

```bash
../venv/bin/python ad_safe.py \
  --model-path artefacts/ad_safe_runs/example-model.pt \
  --eval-split test \
  --base-model simple_cnn \
  --batch-size 8
```

Use the latest single-run checkpoint:

```bash
../venv/bin/python ad_safe.py \
  --model-path-last \
  --eval-split test \
  --batch-size 16
```

Single-run outputs are written under `artefacts/ad_safe_runs/` with a timestamp prefix:

- `<run_id>-model.pt`
- `<run_id>-training_history.png`
- `<run_id>-phase.json`
- `<run_id>-accuracy.csv`
- `<run_id>-setup.json`

## Sweep

Minimal local sweep config:

```bash
mkdir -p artefacts/sweep_configs
cat > artefacts/sweep_configs/simple_cnn_sweep.json <<'JSON'
{
  "backbone": "simple_cnn",
  "train_split": "train",
  "train_fraction": 0.02,
  "eval_splits": ["val", "test"],
  "resume": true,
  "force": false,
  "defaults": {
    "epochs": 1,
    "resplit_runs": 1,
    "batch_size": 8,
    "learning_rate": 0.0001,
    "learning_rate_multiplier": 1.0,
    "patience": 1,
    "seed": 1,
    "unfreeze_all": false
  },
  "jobs": [
    {
      "phases": [
        {},
        {
          "learning_rate": 0.00005,
          "unfreeze_all": true
        }
      ]
    }
  ]
}
JSON

../venv/bin/python train_prod_ad_safe_sweep.py simple_cnn_sweep.json
```

The sweep config path is resolved in this order when you pass a relative path:

1. current working directory
2. `artefacts/sweep_configs/`
3. `artefacts/`
4. `challenge/`

Sweep outputs are written to `<output_root>/<run_id>/` with per-phase checkpoints and JSON files plus `accuracy.csv` and `setup.json`.

Resume an interrupted sweep:

```bash
../venv/bin/python train_prod_ad_safe_sweep.py simple_cnn_sweep.json --run-id 2026-04-23-20-46-17
```

Teacher distillation phase example:

```json
{
  "teacher_model_path": "artefacts/ad_safe_runs/teacher-model.pt",
  "distillation_alpha": 0.3,
  "distillation_temperature": 2.0
}
```

Enrichment job example with adversarial augmentation phase:

```json
{
  "enrichment_jobs": [
    {
      "phases": [
        {
          "strategy": "adversarial",
          "params": {
            "epsilon": 0.03,
            "steps": 1
          }
        }
      ]
    }
  ]
}
```

Other supported enrichment strategies:

- `horizontal_flip`, `vertical_flip` — mirror transformations (no params)
- `rotate` — rotation by specified angles. Params: `angles` (array of degrees, e.g. `[90, 180, 270]`)
- `scale` — resize by factor. Params: `factor_min`, `factor_max` (floats, e.g. `0.9`, `1.1`)
- `gaussian_blur` — blur augmentation. Params: `kernel_size`, `sigma_min`, `sigma_max`
- `perspective` — perspective distortion. Params: `distortion_scale` (float, e.g. `0.2`)
- `grayscale` — convert to grayscale (no params)
- `adversarial` — adversarial perturbation. Params: `epsilon`, `steps`

## Backbone Comparison

```bash
../venv/bin/python train_prod_ad_safe_backbones.py \
  --backbone simple_mlp,simple_cnn \
  --train-split train \
  --train-fraction 0.02 \
  --epochs 1,1 \
  --resplit-runs 1,1 \
  --batch-size 8,8 \
  --learning-rate 0.001,0.0001 \
  --patience 1,1 \
  --seed 1,2
```

## Evaluate Many Models

```bash
../venv/bin/python ad_safe_test.py val test \
  --model-path "*.pt" \
  --limit 5 \
  --batch-size 16 \
  --sort acc_test
```

Use a smaller stratified evaluation source:

```bash
../venv/bin/python ad_safe_test.py test \
  --model-path "*.pt" \
  --dataset-fraction 0.1 \
  --dataset-seed 7
```

## Standalone Contract Check

```bash
../venv/bin/python check_ad_safe_contract.py artefacts/ad_safe_runs/example-model.pt
```
