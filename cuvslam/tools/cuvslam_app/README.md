# cuvslam_app compatibility wrapper

`tools/cuvslam_app/cuvslam_app.py` is kept as a transition entry point for existing CI and scripts.

New workflows should install `tools/python_tools` and call:

- `cuvslam_tracker` for one sequence.
- `cuvslam_reporter` for dataset reports.
- `cuvslam_validator` for multi-dataset validation.

The wrapper forwards old `cuvslam_app.py` calls to the new package entry points and expands legacy reporter configs from `CUVSLAM_DATASETS` when needed.
