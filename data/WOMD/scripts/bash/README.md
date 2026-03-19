# `data/WOMD/scripts/bash/` — Convenience Wrappers (Paths + Batch Execution)

These scripts are thin wrappers around the Python utilities in `data/WOMD/scripts/lib/`
(or around a pre-built Waymo conversion binary).

They're designed for a **container layout where**:
- The repo is mounted at `/workspace`
- Your Python venv is at `/opt/venv`
- The Waymo Motion dataset is under `/workspace/data/WOMD/datasets/...`

> **Important**: Run all scripts from the `/workspace` directory:
> ```bash
> cd /workspace
> bash data/WOMD/scripts/bash/<script_name>.sh
> ```

If you are not using that layout, you can still run the underlying Python scripts
directly and pass custom paths.

---

## Scripts (file-by-file)

### `convert_scenario_tf.sh`
**Purpose:** Convert raw Waymo Scenario TFRecords to processed Motion TF Example TFRecords
**How:** Runs a conversion binary in parallel via GNU `parallel`.

Key details:
- Accepts two optional positional arguments: `[INPUT_DIR]` and `[OUTPUT_DIR]`
- Default input (used when no argument supplied):
  - `/workspace/data/WOMD/datasets/waymo_open_dataset/motion_v_1_3_0/raw/scenario/training_interactive`
- Default output (used when no argument supplied):
  - `/workspace/data/WOMD/datasets/waymo_open_dataset/motion_v_1_3_0/processed/tf/training_interactive`
- Expected converter binary:
  - `/workspace/externals/waymo-open-dataset/src/bazel-bin/waymo_open_dataset/data_conversion/convert_scenario_to_tf_example`

Parallelism:
- Runs `-j6` jobs to avoid memory issues when parsing large TFRecords

What you get:
- One output TFRecord per input TFRecord (same base filename), written to the output directory

---

### `filter_interactive_scenario.sh`
**Purpose:** Create an “interactive-only” dataset split from raw scenario TFRecords.

What “interactive” means here:
- The scenario proto contains `objects_of_interest`, and the script keeps only scenarios where
  `len(objects_of_interest) == 2`.

Default behavior:
- Input defaults to `training_20s`
- Output defaults to a sibling directory `training_20s_interactive`

It calls:
- `data/WOMD/scripts/lib/filter_interactive_scenario.py`

Notable flags used:
- `--num-workers 8` (multiprocessing across TFRecord files)
- Files are copied to the output directory by default (use `--move` to delete originals after filtering)

---

### `generate_movies.sh`
**Purpose:** Generate BEV (bird’s-eye-view) **MP4 movies** from Waymo data.

It runs:
- `data/WOMD/scripts/lib/generate_bev_movie.py`

What it produces:
- Animated movies showing the roadgraph + agents over time.
- Output defaults to `/workspace/data/WOMD/movies/bev/` (as defined in the Python script).

Default calls inside the wrapper:
- Scenario (raw) format:
  - `training_interactive`
  - `validation_interactive`
  - `testing_interactive`
- TF Example (processed) format:
  - `training_interactive`
  - `validation_interactive`
  - `testing_interactive`

You can pass `NUM_SCENARIOS` as the first arg:
```bash
./generate_movies.sh 20
```

---

### `generate_visualizations.sh`
**Purpose:** Generate **static** visualizations (PNG images) from Waymo data.

It runs:
- `data/WOMD/scripts/lib/visualize_scenario.py`

Default calls inside the wrapper:
- Scenario (raw) format:
  - `validation_interactive` and `testing_interactive`
  - Uses `--multi-frame` and `--combined` to generate richer plots.
- TF Example (processed) format:
  - `training_interactive`

Output defaults to `/workspace/data/WOMD/visualizations/` (as defined in the Python script).

---

### `run_m2i_inference.sh`
**Purpose:** A legacy wrapper intended to run upstream M2I inference and visualization.

⚠️ **Current state:** This wrapper references scripts that are **not present** in this repository:
- `models/pretrained/m2i/run_inference.py`
- `models/pretrained/m2i/visualize_predictions.py`

Out of the box, this wrapper won't run unless you add those missing scripts
(or point it at the equivalents in your M2I checkout).

Recommended replacement:
- Use `models/pretrained/m2i/scripts/lib/m2i_live_inference.py` (live pipeline; no precomputed pickles)
- Then use `models/pretrained/m2i/scripts/lib/generate_m2i_movies.py` for movies
