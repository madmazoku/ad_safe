# Ad Safety Example Notebooks

These notebooks are intentionally tiny. They demonstrate the library APIs and runtime path constants, not final model quality.

- `01_from_scratch_quick_train.ipynb`: scratch `simple_cnn` train, evaluation, figures.
- `02_staged_pretrained_adversarial.ipynb`: lightweight pretrained staged training with adversarial augmentation.
- `03_teacher_student_distillation.ipynb`: train a teacher, then train a student with teacher logits.
- `04_sweep_and_compare.ipynb`: small programmatic sweep and metrics comparison.

All examples use `DatasetSourceSpec(..., fraction=...)` so the library loads a small stratified source before the normal training/evaluation flow. Increase those fractions, epochs, and batch size for real experiments.

Run these notebooks with the current working directory set to this folder:

```bash
cd challenge/notebooks/ad_safe_examples
jupyter lab
```

The first cell adds `../..` to `sys.path`, then all project paths come from `ad_safe_lib` constants such as `CHALLENGE_DIR`, `DATA_DIR`, and `AD_SAFE_RUNS_DIR`.
