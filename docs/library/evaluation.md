# Evaluation API

Evaluation uses typed specs and a runner:

- `ModelEvalSpec`
- `DatasetEvalSpec`
- `EvaluationPlan`
- `EvaluationRunResult`
- `run_evaluation_plan`

Example:

```python
plan = ad_safe.EvaluationPlan(
    models=(ad_safe.ModelEvalSpec(path=model_path),),
    datasets=(
        ad_safe.DatasetEvalSpec(
            name="test",
            batch_size=16,
            source=ad_safe.DatasetSourceSpec("test", fraction=1.0),
        ),
    ),
    output_dir=ad_safe.AD_SAFE_RUNS_DIR,
)
result = ad_safe.run_evaluation_plan(plan)
```
