# `data/WOMD/scripts/` — Data Processing Utilities

This directory contains all runnable scripts for processing Waymo Open Motion Dataset data, including format conversion, filtering, and visualization.

---

## Table of Contents

1. [Overview](#overview)
2. [Directory Structure](#directory-structure)
3. [Quick Reference](#quick-reference)
4. [Bash Wrappers](#bash-wrappers)
5. [Python Scripts](#python-scripts)
6. [Usage Examples](#usage-examples)

---

## Overview

The scripts follow a two-layer pattern:

| Layer | Location | Purpose |
|-------|----------|---------|
| **Bash wrappers** | `bash/` | Set paths and call Python scripts with common defaults |
| **Python implementations** | `lib/` | Full-featured scripts with CLI options |

To understand or modify behavior, read the `lib/` Python code. The bash scripts are thin wrappers for convenience.

---

## Directory Structure

```
scripts/
├── README.md                    # This file
│
├── bash/                        # Shell wrappers
│   ├── README.md               # Bash scripts documentation
│   ├── convert_scenario_tf.sh  # Format conversion wrapper
│   ├── filter_interactive_scenario.sh  # Filtering wrapper
│   ├── generate_movies.sh      # BEV movie generation
│   ├── generate_visualizations.sh  # Static image generation
│   └── run_m2i_inference.sh    # M2I inference wrapper
│
└── lib/                         # Python implementations
    ├── README.md               # Python scripts documentation
    ├── scenario_to_example.py  # Scenario → TF Example converter
    ├── filter_interactive_scenario.py  # 2-agent scenario filter
    ├── visualize_scenario.py   # Static PNG visualization
    └── generate_bev_movie.py   # BEV MP4 movie generation
```

---

## Quick Reference

> **Note**: All commands should be run from the `/workspace` directory:
> ```bash
> cd /workspace
> ```

### Data Conversion

```bash
# Convert Scenario TFRecords to TF Example format
python data/WOMD/scripts/lib/scenario_to_example.py \
    --input_dir data/WOMD/datasets/waymo_open_dataset/motion_v_1_3_0/raw/scenario/training \
    --output_dir data/WOMD/datasets/waymo_open_dataset/motion_v_1_3_0/processed/tf/training \
    --num_map_samples 20000
```

### Filtering

```bash
# Filter raw Scenario TFRecords to interactive scenarios only
python data/WOMD/scripts/lib/filter_interactive_scenario.py \
    --input-dir data/WOMD/datasets/waymo_open_dataset/motion_v_1_3_0/raw/scenario/training \
    --output-dir data/WOMD/datasets/waymo_open_dataset/motion_v_1_3_0/raw/scenario/training_interactive \
    --type v2v
```

### Visualization

```bash
# Generate static images
python data/WOMD/scripts/lib/visualize_scenario.py \
    --format scenario \
    --split training_interactive \
    --num 10 \
    --output-dir data/WOMD/visualizations/scenario/training_interactive

# Generate BEV movies
python data/WOMD/scripts/lib/generate_bev_movie.py \
    --format scenario \
    --split training_interactive \
    --num 5 \
    --fps 10 \
    --output-dir data/WOMD/movies/bev/training_interactive
```

---

## Bash Wrappers

### `convert_scenario_tf.sh`

Converts raw Waymo Scenario TFRecords to TF Example format.

```bash
# Default usage (training split)
bash data/WOMD/scripts/bash/convert_scenario_tf.sh

# With explicit input/output directories
bash data/WOMD/scripts/bash/convert_scenario_tf.sh \
    data/WOMD/datasets/waymo_open_dataset/motion_v_1_3_0/raw/scenario/validation_interactive \
    data/WOMD/datasets/waymo_open_dataset/motion_v_1_3_0/processed/tf/validation_interactive
```

**Options:**
- Uses C++ converter binary (build required)
- Supports custom input/output directory paths via positional args

### `filter_interactive_scenario.sh`

Filters raw Scenario TFRecords to keep only interactive scenarios.

```bash
# Default usage
bash data/WOMD/scripts/bash/filter_interactive_scenario.sh

# With explicit input/output directories
bash data/WOMD/scripts/bash/filter_interactive_scenario.sh \
    data/WOMD/datasets/waymo_open_dataset/motion_v_1_3_0/raw/scenario/training_20s \
    data/WOMD/datasets/waymo_open_dataset/motion_v_1_3_0/raw/scenario/training_20s_interactive
```

**Options:**
- Keeps only scenarios with exactly 2 agents of interest
- Supports v2v (vehicle-vehicle), v2p (vehicle-pedestrian), v2c (vehicle-cyclist)

### `generate_visualizations.sh`

Generates static PNG visualizations.

```bash
# Generate 10 visualizations
bash data/WOMD/scripts/bash/generate_visualizations.sh 10

# Default (5 visualizations)
bash data/WOMD/scripts/bash/generate_visualizations.sh
```

### `generate_movies.sh`

Generates BEV MP4 movies.

```bash
# Generate 5 movies
bash data/WOMD/scripts/bash/generate_movies.sh 5

# Default (3 movies)
bash data/WOMD/scripts/bash/generate_movies.sh
```

---

## Python Scripts

### `scenario_to_example.py`

Converts Waymo Scenario proto TFRecords to TF Example TFRecords.

```bash
python data/WOMD/scripts/lib/scenario_to_example.py --help
```

**Arguments:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--input_dir` | str | required | Input directory with Scenario TFRecords |
| `--output_dir` | str | required | Output directory for Example TFRecords |
| `--max_agents` | int | 128 | Max agents per example |
| `--num_map_samples` | int | 20000 | Number of map samples |
| `--pattern` | str | `*.tfrecord*` | Glob pattern for input files |

**Example:**

```bash
python data/WOMD/scripts/lib/scenario_to_example.py \
    --input_dir data/WOMD/datasets/waymo_open_dataset/motion_v_1_3_0/raw/scenario/training \
    --output_dir data/WOMD/datasets/waymo_open_dataset/motion_v_1_3_0/processed/tf/training \
    --num_map_samples 20000 \
    --max_agents 128
```

### `filter_interactive_scenario.py`

Filters raw Scenario TFRecords to keep only interactive scenarios.

```bash
python data/WOMD/scripts/lib/filter_interactive_scenario.py --help
```

**Arguments:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--input-dir` | str | required | Input directory containing raw Scenario TFRecords |
| `--output-dir` | str | required | Output directory for filtered TFRecords |
| `--num-workers` | int | — | Number of parallel workers |
| `--max-files` | int | — | Max files to process |
| `--move` | flag | — | Delete originals after copying |
| `--type` | str | — | Filter by interaction type (v2v, v2p, v2c, others) |
| `--count-type` | flag | — | Print statistics |
| `--skip-filtering` | flag | — | Skip filtering step |
| `--non-interactive-dir` | str | — | Separate directory for non-interactive scenarios |

**Example:**

```bash
python data/WOMD/scripts/lib/filter_interactive_scenario.py \
    --input-dir data/WOMD/datasets/waymo_open_dataset/motion_v_1_3_0/raw/scenario/training \
    --output-dir data/WOMD/datasets/waymo_open_dataset/motion_v_1_3_0/raw/scenario/training_interactive \
    --type v2v \
    --num-workers 4
```

### `visualize_scenario.py`

Generates static PNG visualizations of scenarios.

```bash
python data/WOMD/scripts/lib/visualize_scenario.py --help
```

**Arguments:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--tfrecord` | str | — | Direct path to a TFRecord file |
| `--format` | str | — | Input format (scenario or tf) |
| `--split` | str | — | Dataset split name (e.g., training_interactive) |
| `--all` | flag | — | Visualize all scenarios in input |
| `--num` / `-n` | int | 5 | Number of scenarios to visualize |
| `--scenario-index` | int | 0 | Starting scenario index |
| `--multi-frame` | flag | — | Enable multi-frame 2x3 grid mode |
| `--combined` | flag | — | Enable combined visualization |
| `--output-dir` | str | — | Output directory for PNG images |
| `--dpi` | int | 150 | Image resolution |

**Example:**

```bash
python data/WOMD/scripts/lib/visualize_scenario.py \
    --format tf \
    --split training_interactive \
    --num 20 \
    --output-dir data/WOMD/visualizations/tf/training_interactive \
    --dpi 200
```

### `generate_bev_movie.py`

Generates bird's-eye-view MP4 movies.

```bash
python data/WOMD/scripts/lib/generate_bev_movie.py --help
```

**Arguments:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--tfrecord` | str | — | Direct path to a TFRecord file |
| `--format` | str | — | Input format (scenario or tf) |
| `--split` | str | — | Dataset split name |
| `--all` | flag | — | Process all scenarios |
| `--num` / `-n` | int | 5 | Number of scenarios to process |
| `--scenario-index` | int | 0 | Starting scenario index |
| `--output-dir` | str | — | Output directory for MP4 movies |
| `--fps` | int | 10 | Frames per second |
| `--dpi` | int | 100 | Resolution |
| `--view-range` | float | 50.0 | Camera range in meters |
| `--no-follow` | flag | — | Fixed world view instead of ego-follow |
| `--gif` | flag | — | Also save as GIF |

**Example:**

```bash
python data/WOMD/scripts/lib/generate_bev_movie.py \
    --format tf \
    --split training_interactive \
    --num 10 \
    --fps 10 \
    --output-dir data/WOMD/movies/bev/training_interactive \
    --view-range 50.0
```

---

## Usage Examples

### Complete Data Pipeline

```bash
cd /workspace

# Step 1: Convert raw Waymo data
python data/WOMD/scripts/lib/scenario_to_example.py \
    --input_dir data/WOMD/datasets/waymo_open_dataset/motion_v_1_3_0/raw/scenario/training \
    --output_dir data/WOMD/datasets/waymo_open_dataset/motion_v_1_3_0/processed/tf/training

# Step 2: Filter raw scenarios to interactive scenarios
python data/WOMD/scripts/lib/filter_interactive_scenario.py \
    --input-dir data/WOMD/datasets/waymo_open_dataset/motion_v_1_3_0/raw/scenario/training \
    --output-dir data/WOMD/datasets/waymo_open_dataset/motion_v_1_3_0/raw/scenario/training_interactive

# Step 3: Augment with rule annotations
python -m waymo_rule_eval.augmentation.augment_cli \
    --input data/WOMD/datasets/waymo_open_dataset/motion_v_1_3_0/processed/tf/training_interactive \
    --output data/WOMD/datasets/waymo_open_dataset/motion_v_1_3_0/processed/augmented/scenario/training_interactive

# Step 4: Verify with visualization
python data/WOMD/scripts/lib/visualize_scenario.py \
    --format tf \
    --split training_interactive \
    --output-dir data/WOMD/visualizations/tf/training_interactive \
    --num 5
```

### Quick Visualization Check

```bash
# Fast sanity check on data
bash data/WOMD/scripts/bash/generate_visualizations.sh 3
bash data/WOMD/scripts/bash/generate_movies.sh 3
```

---

## Data Format Support

The visualization scripts (`visualize_scenario.py` and `generate_bev_movie.py`) support two input formats:

| Format | Description | Flag |
|--------|-------------|------|
| `scenario` | Raw Waymo Scenario proto TFRecords | `--format scenario` |
| `tf` | Converted TF Example TFRecords | `--format tf` |

The scripts auto-detect the format if not specified.

---

## Related Documentation

| Document | Description |
|----------|-------------|
| [bash/README.md](bash/README.md) | Bash wrapper details |
| [lib/README.md](lib/README.md) | Python implementation details |
| [../waymo_rule_eval/README.md](../waymo_rule_eval/README.md) | Rule evaluation framework |
| [../src/README.md](../src/README.md) | C++ converter documentation |

---

*Last updated: March 4, 2026*
