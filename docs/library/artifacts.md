# Artifacts API

Model IO and JSON helpers live in `artifacts.py`:

- `save_model`
- `load_model`
- `write_json_file`
- `load_json_file`
- `write_setup_file`
- `path_to_json`

High-level training artifacts are written by `training_runner.py`. Evaluation CSVs are written by `evaluation_runner.py` through reporting helpers.

Generated files belong under `artefacts/` and are ignored by git.
