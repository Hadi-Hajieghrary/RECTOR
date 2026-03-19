# RECTOR Scripts

Project-level scripts for the RECTOR (Rule-Enforced Constrained Trajectory Optimization and Reasoning) project.

## Directory Structure

```
scripts/
├── analysis/                         # Offline analysis scripts for paper revision
│   └── val_test_distribution_compare.py  # Val vs test distributional comparison (KS tests)
├── simulation_engine/                # Closed-loop Waymax simulation framework
│   ├── config.py                     # Experiment configuration dataclasses
│   ├── validate_50.py                # Quick 50-scenario validation
│   ├── waymax_bridge/                # Waymax simulator interface
│   ├── selectors/                    # Confidence, weighted-sum, RECTOR lex selectors
│   └── viz/                          # Visualization and metric reporting
├── git-pre-commit-hook.sh            # Pre-commit hook: auto-generates workspace documentation
├── git-pre-commit-generate-movies.sh # Pre-commit hook: generates BEV movies from Waymo scenarios
├── README.md                         # This file
└── WOMD/                             # Waymo Open Motion Dataset management scripts
    ├── download_waymo_sample.sh      # Downloads dataset samples via gsutil
    ├── check_waymo_status.sh         # Reports dataset download status
    ├── clear_waymo_data.sh           # Removes downloaded tfrecord files
    └── README.md                     # WOMD scripts documentation
```

---

## Analysis Scripts

The `analysis/` directory contains offline analysis scripts used during paper revision.

### val_test_distribution_compare.py — Val/Test Distributional Analysis

Compares the validation (12,800 scenarios) and test (14,665 scenarios) splits of WOMD to test the distributional-shift hypothesis for the test-set generalization gap. Uses direct proto parsing from TFRecords (works for both splits, including the test split which lacks ground truth future trajectories).

**Extracts per-scenario metadata:**
- Agent count, moving agents, nearby agents (<20m)
- Ego speed (m/s), ego heading change (rad)
- Lane count (map complexity proxy)

**Statistical tests:** Two-sample Kolmogorov-Smirnov test on each dimension.

```bash
python scripts/analysis/val_test_distribution_compare.py \
    --max_scenarios 1280 \
    --output output/evaluation/val_test_distribution.json
```

**Produces:** `val_test_distribution.json` with per-split statistics, speed/agent-count regime breakdowns, and KS test results.

---

## Simulation Engine

The `simulation_engine/` directory implements a complete closed-loop simulation framework for evaluating RECTOR's trajectory selection strategies in realistic driving scenarios. It bridges RECTOR's trajectory prediction with the Waymax simulator (Google DeepMind's JAX-based driving simulator).

See [simulation_engine/README.md](simulation_engine/README.md) for full documentation.

---

## Git Hook Scripts

These scripts implement Git pre-commit hooks that run automatically on every `git commit`. They are installed into `.git/hooks/pre-commit` by the devcontainer's `setup_git_hooks.sh` script during container creation.

| Script | Shell mode | Shebang |
|--------|-----------|---------|
| `git-pre-commit-hook.sh` | `set -euo pipefail` (strict: exit on error, undefined vars, pipe failures) | `#!/usr/bin/env bash` |
| `git-pre-commit-generate-movies.sh` | `set -e` (exit on error only) | `#!/bin/bash` |

### git-pre-commit-hook.sh

**Purpose:** Automatically generates workspace documentation before each commit, ensuring the repository always contains an up-to-date structural overview.

**Generated Files:**

| Output File | Content | Method |
|-------------|---------|--------|
| `WORKSPACE_STRUCTURE.md` | Complete workspace directory tree | `tree -L 3 -a --dirsfirst -F` |
| `data/DATA_INVENTORY.md` | Data directory tree with statistics | `tree data/ -d -L 4` + `find`/`du` |

**How it works:**

1. **Generates `WORKSPACE_STRUCTURE.md`:**
   - Runs `tree` at 3 levels depth with hidden files included (`-a`), directories listed first (`--dirsfirst`), and type indicators (`-F`).
   - Excludes noise directories and files: `.git`, `__pycache__`, `*.pyc`, `.pytest_cache`, `.ipynb_checkpoints`, `*.egg-info`, `.cache`, `.mypy_cache`, `.ruff_cache`, `node_modules`, `References`, `.specstory`.
   - Wraps the output in a markdown code block with a header and UTC timestamp.

2. **Generates `data/DATA_INVENTORY.md`:**
   - Creates `data/` directory if it doesn't exist.
   - Generates a directory-only tree of `data/` at 4 levels depth (`tree data/ -d -L 4 --dirsfirst -F`).
   - For each subdirectory under `data/`, computes:
     - Total file count (via `find -type f | wc -l`)
     - Total size (via `du -sh`)
   - **Waymo-specific handling** (directories matching `waymo*`): Provides detailed breakdowns by format and split:
     - **Scenario format**: Iterates `data/<waymo_dir>/scenario/*/`, counts `*tfrecord*` files per split, lists up to 20 filenames with byte sizes.
     - **TF format**: Same structure under `data/<waymo_dir>/tf/*/`.
     - **Lidar & Camera format**: Same structure under `data/<waymo_dir>/lidar_and_camera/*/`.
   - **Non-Waymo directories**: Lists all files (excluding `.gitkeep`) with relative paths.
   - Appends a UTC timestamp.

3. **Reports generated files** to the console for visibility (the hook does not auto-stage files).

4. **Calls `git-pre-commit-generate-movies.sh`** if the script exists at `/workspace/scripts/git-pre-commit-generate-movies.sh`.

**Key Variables:**

| Variable | Value | Description |
|----------|-------|-------------|
| `WORKSPACE_TREE_FILE` | `WORKSPACE_STRUCTURE.md` | Output path for workspace tree |
| `DATA_INVENTORY_FILE` | `data/DATA_INVENTORY.md` | Output path for data inventory |

**Installation:** Automatically installed by `.devcontainer/scripts/setup_git_hooks.sh` during container creation, which copies this script to `.git/hooks/pre-commit`. To reinstall manually:

```bash
bash .devcontainer/scripts/setup_git_hooks.sh
```

---

### git-pre-commit-generate-movies.sh

**Purpose:** Generates bird's-eye-view (BEV) MP4 visualization movies from Waymo scenario data. Runs as the final step of the pre-commit hook, producing visual summaries of interactive driving scenarios.

**Configuration:**

| Setting | Value | Description |
|---------|-------|-------------|
| `DATA_DIR` | `/workspace/data/WOMD/datasets/waymo_open_dataset/motion_v_1_3_0` | Waymo dataset root |
| `MOVIES_DIR` | `/workspace/data/WOMD/movies/bev` | Output directory for generated movies |
| `MAX_SCENARIOS` | `5` | Maximum movies to generate per format/split combination |
| `PYTHON` | `/opt/venv/bin/python` | Python interpreter path |
| `VISUALIZE_SCRIPT` | `/workspace/data/WOMD/scripts/lib/generate_bev_movie.py` | Visualization script |

**How it works:**

1. **Pre-flight checks:**
   - Prints a warning and exits (exit 0) if `generate_bev_movie.py` doesn't exist at the expected path.
   - Prints a warning and exits (exit 0) if the Waymo data directory doesn't exist.
   - These checks ensure the hook never blocks commits when data or scripts are absent.

2. **Processes interactive splits only:**
   - `training_interactive`
   - `validation_interactive`
   - `testing_interactive`

3. **For each format/split combination** (both `scenario` and `tf` formats):
   - Resolves the input directory:
     - Scenario format: `<DATA_DIR>/raw/scenario/<split>/`
     - TF format: `<DATA_DIR>/processed/tf/<split>/`
   - Skips if the directory doesn't exist or contains no `*.tfrecord*` files.
   - Skips if `MAX_SCENARIOS` (5) or more `.mp4` files already exist in the output directory.
   - Runs `generate_bev_movie.py` with `--format`, `--split`, `--num`, and `--output-dir` arguments.
   - Suppresses warning output via grep filter (filters: `tensorflow`, `oneDNN`, `GPU`, `CUDA`, `TensorRT`, `NUMA`).

4. **Prints a summary** of total MP4 and GIF file counts across all output directories.

**Output structure:**

```
data/WOMD/movies/bev/
├── scenario/
│   ├── training_interactive/    # Up to 5 MP4 files
│   ├── validation_interactive/
│   └── testing_interactive/
└── tf/
    ├── training_interactive/
    ├── validation_interactive/
    └── testing_interactive/
```

**Called by:** `git-pre-commit-hook.sh` (final step of the pre-commit hook).
