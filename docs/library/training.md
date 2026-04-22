# Training API

Use `TrainingConfig` for model/training behavior and `RunPlan` for executable training jobs.

Common types:

- `TrainingConfig`
- `PhaseSpec`
- `JobSpec`
- `RunPlan`
- `TrainingRunResult`

Execution:

```python
result = ad_safe.run_training_plan(plan)
```

Teacher distillation is controlled by `teacher_model_path`, `distillation_alpha`, and `distillation_temperature`. Adversarial mode mutates the training dataset by appending generated adversarial samples before normal train/validation resplits.

`generate_adversarial_perturbation(...)` accepts an attack strategy object:

- `BudgetedPgdStrategy(epsilon=..., num_steps=...)`: use the requested epsilon budget directly. This is the training default.
- `MinimalFlipPgdStrategy(max_epsilon=..., num_steps=...)`: search up to the requested epsilon and return the smallest found perturbation that flips the true class. This is intended for figures and analysis.

Pass `return_result=True` for a single-sample attack when you also need epsilon, confidence, and distance metrics.

Class reversal uses a separate strategy family because it is not an adversarial
perturbation from a real sample. It optimizes a synthetic image toward a target
class:

- `RandomRestartTargetClassStrategy(step_size=..., num_steps=..., num_restarts=...)`
  starts from random images, optimizes each target class, and keeps the best
  restart by target prediction, target confidence, and margin.

Use it through `generate_class_reversal_figure(model, strategy=...)` for visual
diagnostics.
