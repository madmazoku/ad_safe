# Library API

`ad_safe_lib` is the public import surface for notebooks, scripts, and manual experiments.

Detailed notes:

- [Data](library/data.md)
- [Training](library/training.md)
- [Evaluation](library/evaluation.md)
- [Metrics](library/metrics.md)
- [Backbones](library/backbones.md)
- [Artifacts](library/artifacts.md)

Additional helpers are also part of the public surface:

- path helpers/constants from `paths.py` (for example `CHALLENGE_DIR`, `AD_SAFE_RUNS_DIR`, `resolve_existing_path`)
- reporting helpers from `reporting.py` (metrics table/CSV formatting)
- high-level workflow helpers from `workflows.py`

Figure helpers live in `figures.py`; class-reversal optimization strategy
objects live separately in `reversal.py`.

Most workflows use these building blocks:

```python
import ad_safe_lib as ad_safe

train_source = ad_safe.DatasetSourceSpec("train", fraction=0.02, seed=1)
config = ad_safe.TrainingConfig(base_model="simple_cnn", epochs=1, batch_size=8)
```

Prefer direct use of typed plans and runners for experiments. Keep CLI parsing and JSON-specific interpretation outside the library.
