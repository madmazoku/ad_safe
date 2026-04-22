# Data API

Key types and functions:

- `DatasetSourceSpec(name, fraction=1.0, seed=None)`
- `load_dataset(split_name)`
- `load_dataset_source(spec)`
- `make_stratified_subset(dataset, fraction, seed=...)`
- `make_data_loader(...)`

`load_dataset` always loads the full named split from `DATA_DIR`. `load_dataset_source` adds optional stratified subsetting before the dataset enters training or evaluation.

Example:

```python
source = ad_safe.DatasetSourceSpec("train", fraction=0.05, seed=42)
dataset = ad_safe.load_dataset_source(source)
```
