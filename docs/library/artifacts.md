# Artifacts API

Model IO and JSON helpers live in `artifacts.py`:

- `save_model`
- `load_model`
- `write_json_file`
- `load_json_file`
- `write_setup_file`
- `path_to_json`

High-level training artifacts are written by `training_runner.py`. Sweep runs keep all phase artifacts for a run in one directory, with one shared `accuracy.csv` for comparison. Evaluation CSVs are written by `evaluation_runner.py` through reporting helpers.

Project scripts use `artefacts/` as local runtime storage. Its contents are ignored by git.
