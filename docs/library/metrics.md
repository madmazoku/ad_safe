# Metrics API

`ClassificationMetrics` contains:

- accuracy
- AUC
- NLL
- average confidence
- average margin
- average correct/wrong confidence
- safe and unsafe recall

Use `evaluate_metrics(model, loader, split_name)` for direct evaluation and `DefaultValidationMetricComparator` for validation comparison during training.

Reporting helpers in `reporting.py` format metrics into terminal matrix tables and CSV rows. They do not load models or datasets.
