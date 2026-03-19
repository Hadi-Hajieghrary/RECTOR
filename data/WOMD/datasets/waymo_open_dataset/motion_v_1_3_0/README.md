# Data Directory

This directory contains data and scripts for the RECTOR trajectory prediction pipeline.

## Directory Structure

```
data/
├── README.md                 # This file
├── DATA_INVENTORY.md         # Detailed data inventory
├── datasets/
│   └── waymo_open_dataset/
│       └── motion_v_1_3_0/
│           ├── raw/
│           │   └── scenario/         # Original Scenario proto format
│           │       ├── training_interactive/    (50 files, 89 GB)
│           │       ├── testing_interactive/     (50 files, 9.7 GB)
│           │       └── validation_interactive/  (50 files, 13 GB)
│           └── processed/
│               └── tf/               # Converted tf.Example FLAT format
│                   ├── training_interactive/    (106 files, 41 GB) ✅
│                   └── validation_interactive/  (50 files, TFRecords) ✅
├── scripts/
│   ├── bash/                 # Shell scripts for batch processing
│   │   ├── convert_scenario_tf.sh    # Parallel Scenario→tf.Example conversion
│   │   └── generate_movies.sh
│   └── lib/                  # Python library modules
│       ├── filter_interactive_scenario.py
│       ├── generate_bev_movie.py     # BEV movie generator (MP4+GIF)
│       └── visualize_scenario.py     # Static visualization generator
├── src/                      # C++ source code
│   ├── build.sh              # Build script for converter
│   ├── convert_scenario_to_tf_example.cc
│   └── BUILD_INSTRUCTIONS.md
├── movies/                   # Generated visualization movies
│   └── bev/                  # BEV movies (MP4+GIF)
└── visualizations/           # Static visualization images
    ├── scenario/             # From Scenario format
    └── tf/                   # From TF format
```

## Data Formats

### Scenario Proto Format (Raw)
The original Waymo Motion Dataset format with nested protocol buffer messages.
- Location: `raw/scenario/`
- Files: `*.tfrecord-*-of-*`

### tf.Example Flat Format (Processed)
The format required by M2I, with flattened tensors.
- Location: `processed/tf/`
- Key format: `state/current/x`, `state/past/x`, `state/future/x`, etc.

**Important:** M2I requires the "flat" format, not the Scenario proto format.

## Quick Start

### 1. Build the Converter (if not already built)

```bash
cd /workspace/data/WOMD/src
./build.sh
```

### 2. Convert Scenario → tf.Example (Flat Format)

```bash
cd /workspace/data/WOMD/scripts/bash
./convert_scenario_tf.sh
```

### 3. Generate Visualizations

```bash
cd /workspace/data/WOMD/scripts/lib

# BEV movies (MP4 + GIF)
python generate_bev_movie.py --format scenario --split validation_interactive --num 5

# Static visualizations (overview, multi-frame, combined)
python visualize_scenario.py --format scenario --split validation_interactive --num 5 --multi-frame --combined
```

### 4. Use with M2I Live Inference

```bash
cd /workspace/models/pretrained/m2i

# Run full pipeline on validation data
./run_m2i_pipeline.sh -n 30 -v 20

# The data directory is automatically detected:
# /workspace/data/WOMD/datasets/waymo_open_dataset/motion_v_1_3_0/processed/tf/validation_interactive
```

## Data Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Waymo Motion Dataset v1.3.0                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              raw/scenario/*_interactive/                         │
│                   (Scenario Proto Format)                        │
│                                                                  │
│  • testing_interactive:     50 files,  9.7 GB                    │
│  • validation_interactive:  50 files, 13.0 GB                    │
│  • training_interactive:    50 files, 89.0 GB                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │  convert_scenario_to_tf_example.cc
                              │  (C++ binary via Bazel)
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              processed/tf/*_interactive/                         │
│                   (tf.Example FLAT Format)                       │
│                                                                  │
│  Keys: state/current/x, state/past/x, state/future/x            │
│        roadgraph_samples/xyz, roadgraph_samples/type, etc.      │
│                                                                  │
│  • training_interactive:   106 files, 41.0 GB  ✅ Converted      │
│  • validation_interactive:  50 files           ✅ Converted      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │  m2i_live_inference.py
                              │  (Python + PyTorch)
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    M2I 3-Stage Predictions                       │
│                                                                  │
│  • Marginal (DenseTNT): 6 trajectories per agent                │
│  • Relation (V2V): Influencer/reactor classification            │
│  • Conditional (V2V): 6 conditioned trajectories per reactor    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │  generate_m2i_movies.py
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Visualization Movies                          │
│                                                                  │
│  • PNG: Static prediction images                                 │
│  • MP4: Animated trajectory evolution                            │
│  Location: /workspace/models/movies/m2i_predictions/            │
└─────────────────────────────────────────────────────────────────┘
```

## TFRecord Format Details

### Flat Format Keys (for M2I)

```python
# Agent state (shape: [128] for current, [128, 10] for past/future)
'state/current/x'        # X position at current timestep
'state/current/y'        # Y position
'state/current/heading'  # Heading angle (radians)
'state/current/valid'    # Validity mask
'state/past/x'           # Past 10 timesteps (1 second @ 10Hz)
'state/past/y'
'state/future/x'         # Future 80 timesteps (8 seconds @ 10Hz)
'state/future/y'

# Agent metadata
'state/id'               # Agent IDs
'state/type'             # 1=vehicle, 2=pedestrian, 3=cyclist

# Road graph
'roadgraph_samples/xyz'  # Road point coordinates
'roadgraph_samples/type' # Road element types

# Track IDs
'scenario/id'            # Unique scenario identifier
'state/tracks_to_predict' # Which agents to predict (2 for interactive)
```

### Data Dimensions

| Dimension | Value | Description |
|-----------|-------|-------------|
| Max agents | 128 | Maximum agents per scenario |
| History | 11 frames | 1.1 seconds @ 10Hz (current + 10 past) |
| Future | 80 frames | 8.0 seconds @ 10Hz |
| Raster | 224×224×60 | Bird's-eye-view image for CNN encoder |

## Rule Evaluation Framework

The `waymo_rule_eval/` directory contains a comprehensive framework for evaluating autonomous driving rule compliance. See [waymo_rule_eval/README.md](./waymo_rule_eval/README.md) for full documentation.

### Quick Overview

- **28 registered rules** across 10 levels (L0-L10)
- **Rule applicability varies** by scenario content (not all rules apply to every scenario)
- Covers: collision avoidance, lane-keeping, speed limits, traffic signs, VRU interactions

### Augmentation Pipeline

Generate rule evaluation vectors for ML training:

```bash
# Process scenarios with rule evaluations
cd /workspace/data
python -m waymo_rule_eval.augmentation.process_scenarios \
    --input "/workspace/data/WOMD/datasets/waymo_open_dataset/motion_v_1_3_0/raw/scenario/validation_interactive/*.tfrecord*" \
    --output augmented/validation_sample.jsonl \
    --max-scenarios 100
```

Output JSONL contains per-window:
- Boolean applicability vector (28 rules)
- Violation detection (which rules were violated)
- Severity metrics for each applicable rule

## Visualization Tools

### BEV Movies (`generate_bev_movie.py`)

Generate bird's-eye view movies with:
- Ego vehicle tracking (camera follows ego)
- All agents with velocity arrows
- Map features (lanes, crosswalks, stop signs)
- History trails with fading
- Dual output: MP4 (10 fps) + GIF (5 fps)

```bash
python generate_bev_movie.py --format scenario --split validation_interactive --num 5
```

### Static Visualizations (`visualize_scenario.py`)

Generate PNG visualizations:
- **Overview**: Full trajectory plot
- **Multi-frame**: 2x3 grid of 6 key frames
- **Combined**: Overview + 4 timeline frames

```bash
python visualize_scenario.py --format scenario --split validation_interactive --num 5 --multi-frame --combined
```

## Interactive Scenario Statistics

From training_interactive (50 files, ~10,000 scenarios):

| Interaction Type | Count | Percentage |
|------------------|-------|------------|
| vehicle ↔ vehicle | ~7,900 | 77.2% |
| vehicle ↔ pedestrian | ~1,600 | 15.2% |
| vehicle ↔ cyclist | ~800 | 7.5% |
| other | ~20 | 0.2% |

## Requirements

- **Python 3.10+** with TensorFlow, matplotlib
- **Bazel 5.4.0**: For building C++ converter
- **ffmpeg**: For video generation

All requirements are pre-installed in the devcontainer.
