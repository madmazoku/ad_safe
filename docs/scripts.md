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

Single-run outputs are written under `artefacts/ad_safe_runs/` with a timestamp prefix.

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

Sweep outputs are written to `artefacts/prod_models/<run_id>_<sweep-title>/` by default. That directory contains every phase checkpoint, phase JSON, history figure, `setup.json`, and the shared `accuracy.csv` for comparing all jobs in the sweep.

Resume an interrupted sweep:

```bash
../venv/bin/python train_prod_ad_safe_sweep.py simple_cnn_sweep.json --run-id 2026-04-23-20-46-17
```

### Sweep Config Format

`train_prod_ad_safe_sweep.py` expects a JSON object. Only documented top-level fields affect execution. Unknown phase fields are rejected in `defaults`, `jobs[]`, and `jobs[].phases[]`.

Top-level fields:

- `backbone` (required string): one of the names exposed by `ad_safe_lib.SUPPORTED_BACKBONES`.
- `title` or `sweep_title` (optional string): readable sweep name used in the output folder. If omitted, the config filename is used.
- `output_root` (optional string): parent output directory. Relative paths are resolved from the config file directory. Defaults to `artefacts/prod_models`.
- `run_id` (optional string or null): output run directory name. `null` or omission creates a timestamp. Path separators are not allowed.
- `train_split` (optional string): dataset split used for training. Defaults to `train`.
- `train_fraction` (optional number): stratified fraction of `train_split`, in `(0, 1]`. Defaults to `1.0`.
- `eval_splits` (optional string or string array): evaluation split names. Omission discovers available dataset splits.
- `resume` (optional boolean): reuse completed matching phase artifacts. Defaults to `true`.
- `force` (optional boolean): rerun phases even when resume artifacts exist. Defaults to `false`.
- `cooldown` (optional object): cooldown settings.
- `defaults` (optional object): default phase fields.
- `jobs` (required non-empty array): sweep jobs.

`cooldown` fields:

- `every_epochs` (integer >= 0): run cooldown after every N global epochs. `0` disables periodic cooldown.
- `seconds` (number >= 0): maximum cooldown duration.
- `gpu_max_temp` (integer >= 0): start cooldown when GPU temperature reaches this Celsius value. `0` disables temperature cooldown.
- `gpu_resume_temp` (integer >= 0): resume at or below this Celsius value. If `gpu_max_temp > 0` and this is `0`, it becomes `gpu_max_temp - 5`.
- `gpu_temp_check_seconds` (number > 0): polling interval while cooling down.

Phase fields accepted in `defaults`, `jobs[]`, and `jobs[].phases[]`:

- `epochs` (positive integer)
- `resplit_runs` (positive integer)
- `batch_size` (positive integer)
- `learning_rate` (positive number, numeric string, or non-empty array of positive numbers)
- `learning_rate_multiplier` (positive number). Cannot be used with multiple `learning_rate` values.
- `patience` (integer >= 0)
- `seed` (non-negative integer or null): `0` or `null` generates a fresh random seed.
- `unfreeze_all` (boolean): train all layers instead of only the head.
- `teacher_model_path` (string or null): teacher checkpoint. Relative/bare values are searched from the config directory, `artefacts/ad_safe_runs/`, `artefacts/`, then the challenge root.
- `distillation_alpha` (number in `[0, 1]`)
- `distillation_temperature` (positive number)

Job objects also accept:

- `title` (optional string)
- `phases` (required non-empty array)
- `enrichment_jobs` (optional array)

Phase objects also accept:

- `title` (optional string)
- `enrichment_jobs` (optional array)

Field inheritance is `defaults` -> `jobs[]` -> `jobs[].phases[]`, with nearer values overriding earlier values. `enrichment_jobs` follows the same nearest-scope rule: phase-level enrichment replaces job/default enrichment, and job-level enrichment replaces default enrichment.

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
      "input_replay_fraction": 0.25,
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

`input_replay_fraction` controls how much of a phase input dataset is replayed into the enrichment output:

- `1.0` keeps all input samples plus derived samples
- `0.0` keeps only derived samples
- any intermediate float keeps a deterministic random fraction of the input samples

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
