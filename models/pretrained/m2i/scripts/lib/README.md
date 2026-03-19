# `models/pretrained/m2i/scripts/lib/` — M2I Python Modules

This folder contains all Python scripts for the M2I (Multi-agent Interaction) inference pipeline.
These scripts handle TFRecord parsing, model inference, and visualization generation.

The primary script is `m2i_receding_horizon_full.py` which implements **receding horizon**
prediction — simulating real-time autonomous vehicle operation where predictions are
updated continuously as new observations become available.

> **Note:** All commands should be run from the `/workspace` directory.

---

## Quick Start

```bash
cd /workspace

# Run receding horizon pipeline (recommended)
bash models/pretrained/m2i/scripts/bash/run_m2i_pipeline.sh -n 10

# Or directly with Python
python models/pretrained/m2i/scripts/lib/m2i_receding_horizon_full.py \
    --num_scenarios 10 --start_t 10 --end_t 90 --step 1 \
    --generate_movies --device cuda
```

---

## Quick Reference

| File | Purpose | Primary Use Case |
|------|---------|------------------|
| `m2i_receding_horizon_full.py` | **PRIMARY**: Receding horizon prediction | Standalone script |
| `waymo_flat_parser.py` | TFRecord parser compatibility | Import as module |
| `generate_m2i_movies.py` | Static prediction visualization | Standalone script |
| `m2i_live_inference.py` | Legacy 3-stage pipeline | Standalone script |
| `m2i_receding_horizon.py` | Simple receding horizon (older) | Standalone script |
| `m2i_gpu_pipeline.py` | All-in-one sandbox | Standalone script |
| `subprocess_relation.py` | Relation inference subprocess | Called by other scripts |
| `subprocess_conditional.py` | Conditional inference subprocess | Called by other scripts |

---

## File Descriptions

### 1. `waymo_flat_parser.py`

**Purpose:** TFRecord parser compatibility layer between RECTOR workspace and M2I.

**The Problem It Solves:**

The original M2I code (in `/workspace/externals/M2I/src/waymo_tutorial.py`) expects Waymo TFRecords
with **shaped tensors**:
```
roadgraph_samples/xyz: shape [20000, 3]
state/future/x: shape [128, 80]
```

The RECTOR workspace generates TFRecords with **flat tensors**:
```
roadgraph_samples/xyz: shape [90000]  (30000 points × 3 coords)
state/future/x: shape [10240]         (128 agents × 80 timesteps)
```

**What It Provides:**

1. `features_description_flat` — TensorFlow feature schema for flat format
2. `_parse_flat(example)` — Parses and reshapes flat tensors to M2I's expected shapes
3. `patch_waymo_tutorial()` — Monkey-patches M2I's parser with the flat parser

**Critical Defensive Fixes:**

- **Lane Type Clipping:** Clips `roadgraph_samples/type` to max 19 (M2I expects 0–19)
- **Lane ID Wrapping:** Wraps `roadgraph_samples/id` modulo 1000 (M2I has range limits)

**Usage:**

```python
# In your script, BEFORE creating any M2I dataset:
import sys
sys.path.insert(0, '/workspace/models/pretrained/m2i/scripts/lib')
from waymo_flat_parser import patch_waymo_tutorial

# Apply the patch
patch_waymo_tutorial()

# Now M2I's dataset will correctly parse RECTOR TFRecords
from dataset_waymo import Dataset
dataset = Dataset(args, ...)
```

**When You Need This:**

- Always, when using M2I with RECTOR's TFRecords
- If you see errors like "lane type out of range" or "lane id too large"

---

### 2. `m2i_live_inference.py`

**Purpose:** Run the complete M2I 3-stage pipeline directly on TFRecords.

**The Three Stages:**

| Stage | Model File | Purpose |
|-------|------------|---------|
| 1. DenseTNT (Marginal) | `model.24.bin` | Predicts 6 trajectory modes for each agent independently |
| 2. Relation V2V | `model.25.bin` | Determines influencer↔reactor relationships |
| 3. Conditional V2V | `model.29.bin` | Predicts reactor trajectory conditioned on influencer |

**Output Files:**

```
<output_dir>/
├── marginal/
│   └── marginal_predictions.pickle    # {scenario_id: {rst, score, ids}}
├── relation/
│   └── relation_predictions.pickle    # {scenario_id: relation_labels}
└── conditional/
    └── conditional_predictions.pickle # {scenario_id: {rst, score, ids}}
```

**Prediction Format:**
- `rst`: Trajectory predictions, shape `[N_agents, 6, 80, 2]` (6 modes × 80 timesteps × XY)
- `score`: Confidence scores, shape `[N_agents, 6]`
- `ids`: Waymo track IDs, shape `[N_agents]`

**Usage:**

```bash
# Run from /workspace directory
cd /workspace

# Basic usage - run all 3 stages
python models/pretrained/m2i/scripts/lib/m2i_live_inference.py \
    --num_scenarios 50 \
    --data_dir data/WOMD/datasets/waymo_open_dataset/motion_v_1_3_0/processed/tf/validation_interactive \
    --output_dir output/m2i_live \
    --device cuda

# Run only marginal stage
python models/pretrained/m2i/scripts/lib/m2i_live_inference.py \
    --stage marginal \
    --num_scenarios 100 \
    --device cuda

# Run specific stages
python models/pretrained/m2i/scripts/lib/m2i_live_inference.py \
    --stage relation \
    --num_scenarios 50
```

**CLI Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--stage` | `all` | Which stage(s): `marginal`, `relation`, `conditional`, or `all` |
| `--num_scenarios` | `10` | Number of scenarios to process |
| `--data_dir` | script default: `/workspace/data/WOMD/datasets/.../validation_interactive` | Path to TF Example TFRecords; in this workspace pass `/workspace/data/WOMD/datasets/.../validation_interactive` |
| `--output_dir` | `/workspace/output/m2i_live` | Where to save prediction pickles |
| `--device` | `cuda` | Device: `cuda` or `cpu` |

**Requirements:**
- Model weights at `models/pretrained/m2i/models/{densetnt,relation_v2v,conditional_v2v}/`
- TF Example format TFRecords (not raw Scenario protos)

---

### 3. `generate_m2i_movies.py`

**Purpose:** Create visualization movies from M2I predictions.

**What It Visualizes:**

- **Road Network:** Lanes, boundaries, crosswalks (gray)
- **Past Trajectories:**
  - Influencer history (blue solid line)
  - Reactor history (orange solid line)
- **Ground Truth Future:**
  - Influencer GT (dark blue dashed)
  - Reactor GT (dark orange dashed)
- **Predictions:**
  - Influencer prediction (green solid)
  - Reactor's 6 modes (plasma colormap, with confidence scores)

**Output:**
- PNG images: Static BEV snapshots
- MP4 movies: Animated 10-second scenarios at 10 FPS

**Usage:**

```bash
# Run from /workspace directory
cd /workspace

# Generate both images and movies
python models/pretrained/m2i/scripts/lib/generate_m2i_movies.py \
    --data_dir data/WOMD/datasets/waymo_open_dataset/motion_v_1_3_0/processed/tf/validation_interactive \
    --predictions_dir output/m2i_live \
    --output_dir models/pretrained/m2i/movies/m2i_predictions \
    --num_scenarios 10 \
    --mode both \
    --fps 10

# Generate only movies (faster)
python models/pretrained/m2i/scripts/lib/generate_m2i_movies.py \
    --predictions_dir output/m2i_live \
    --output_dir models/pretrained/m2i/movies \
    --num_scenarios 5 \
    --mode movie

# Generate only images (for thumbnails)
python models/pretrained/m2i/scripts/lib/generate_m2i_movies.py \
    --predictions_dir output/m2i_live \
    --output_dir models/pretrained/m2i/movies \
    --mode image
```

**CLI Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_dir` | validation_interactive | Path to TFRecords |
| `--predictions_dir` | `/workspace/output/m2i_live` | Directory with prediction pickles |
| `--output_dir` | `movies/m2i_predictions` | Where to save visualizations |
| `--num_scenarios` | `10` | Number of scenarios to visualize |
| `--mode` | `both` | Output mode: `image`, `movie`, or `both` |
| `--fps` | `10` | Frames per second for movies |

**Requirements:**
- Prediction pickle files from `m2i_live_inference.py`
- Corresponding TFRecord files

---

### 4. `m2i_gpu_pipeline.py`

**Purpose:** All-in-one GPU inference and visualization sandbox.

**Features:**
- Loads all 3 M2I models onto GPU simultaneously
- Runs inference in different modes
- Generates visualizations including GIFs
- Supports legacy precomputed prediction loading

**When to Use:**
- Quick experimentation and prototyping
- When you want everything in one script
- For debugging individual components

**When NOT to Use:**
- Production pipelines (prefer `m2i_live_inference.py` + `generate_m2i_movies.py`)
- Memory-constrained environments (loads all models at once)

**Usage:**

```bash
cd /workspace

# Run inference only
python models/pretrained/m2i/scripts/lib/m2i_gpu_pipeline.py \
    --mode inference \
    --num_scenarios 20

# Run visualization only (requires prior inference)
python models/pretrained/m2i/scripts/lib/m2i_gpu_pipeline.py \
    --mode visualize \
    --num_scenarios 20

# Run full pipeline
python models/pretrained/m2i/scripts/lib/m2i_gpu_pipeline.py \
    --mode all \
    --num_scenarios 20
```

**CLI Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | `all` | Mode: `inference`, `visualize`, or `all` |
| `--num_scenarios` | `10` | Number of scenarios |
| `--device` | `cuda` | Device: `cuda` or `cpu` |

---

### 5. `m2i_receding_horizon.py`

**Purpose:** Simple receding horizon prediction demonstration.

**Concept:**

Standard M2I predicts once at t=10 for the full future (t=11–90).
Receding horizon simulates real-time AV operation:

```
At t=10: Observe [t=0:10], Predict [t=11:90]
At t=11: Observe [t=1:11], Predict [t=12:91]
At t=12: Observe [t=2:12], Predict [t=13:92]
...
```

**Why This Matters:**
- Shows how predictions improve with more observations
- Simulates online operation where new data arrives continuously
- Reveals model sensitivity to recent vs. historical observations

**Usage:**

```bash
cd /workspace

# Run receding horizon with 10 prediction steps
python models/pretrained/m2i/scripts/lib/m2i_receding_horizon.py \
    --num_scenarios 5 \
    --prediction_steps 10 \
    --device cuda
```

**CLI Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--num_scenarios` | `5` | Number of scenarios |
| `--prediction_steps` | `10` | How many timesteps to predict at |
| `--device` | `cuda` | Device: `cuda` or `cpu` |

**Note:** This is a simpler variant. For full features, use `m2i_receding_horizon_full.py`.

---

### 6. `m2i_receding_horizon_full.py`

**Purpose:** Comprehensive receding horizon inference with all features.

**Components:**

1. **BEV Raster Rendering** (`_render_bev_raster`)
   - Creates 224×224×60 bird's-eye-view images
   - Channel layout:
     - Ch 0-10: Ego agent trajectory history
     - Ch 20-30: Other agents' trajectories
     - Ch 40-59: Road lane types (20 types)
   - Used by DenseTNT's CNN encoder

2. **Road Network Vectors** (`_create_road_vectors`)
   - Creates 128-dim polyline vectors for VectorNet
   - Groups road points by lane ID
   - Scale factor: 0.03 (matching M2I's `stride_10_2` config)

3. **Receding Horizon Loop** (`run_receding_horizon`)
   - Runs from `start_t` to `end_t` with configurable step
   - Computes ADE/FDE metrics at each timestep

4. **Movie Visualization** (`RecedingHorizonVisualizer`)
   - MP4 movies showing predictions evolving over time
   - 10 FPS matching Waymo temporal resolution

**Usage:**

```bash
cd /workspace

# Basic receding horizon with movies
python models/pretrained/m2i/scripts/lib/m2i_receding_horizon_full.py \
    --num_scenarios 10 \
    --start_t 10 \
    --end_t 90 \
    --step 1 \
    --generate_movies \
    --device cuda

# With custom TFRecord
python models/pretrained/m2i/scripts/lib/m2i_receding_horizon_full.py \
    --tfrecord /path/to/validation.tfrecord \
    --num_scenarios 5 \
    --generate_movies

# With subprocess-isolated 3-stage pipeline (RECOMMENDED)
python models/pretrained/m2i/scripts/lib/m2i_receding_horizon_full.py \
    --num_scenarios 10 \
    --subprocess-pipeline \
    --generate_movies
```

**CLI Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--tfrecord` | validation_interactive | Path to TFRecord file |
| `--num_scenarios` | `3` | Number of scenarios |
| `--start_t` | `10` | First prediction timestep (≥10) |
| `--end_t` | `30` | Last prediction timestep (≤90) |
| `--step` | `5` | Timestep increment |
| `--output` | auto | Output pickle path |
| `--movies_dir` | `movies/receding_horizon` | Movie output directory |
| `--generate_movies` | `False` | Enable movie generation |
| `--device` | `cuda` | Device: `cuda` or `cpu` |
| `--subprocess-pipeline` | `False` | Use subprocess isolation (recommended) |

**Subprocess Mode:**

The `--subprocess-pipeline` flag runs Relation and Conditional stages in separate Python
processes. This is recommended because M2I uses global variables that can conflict when
running multiple models sequentially.

---

### 7. `subprocess_relation.py`

**Purpose:** Isolated subprocess for Relation V2V inference.

**Why Subprocesses?**

M2I's `VectorNet` and `utils.py` use global state:
- `utils.args` is a global object modified by each model
- Running DenseTNT → Relation → Conditional in one process causes conflicts
- Subprocess isolation gives each stage a clean Python environment

**How It Works:**

1. Called by `m2i_receding_horizon_full.py` with `--subprocess-pipeline`
2. Receives input data as a pickle file
3. Loads Relation V2V model (model.25.bin)
4. Runs inference
5. Writes output to specified pickle file

**Direct Usage (rarely needed):**

```bash
cd /workspace

python models/pretrained/m2i/scripts/lib/subprocess_relation.py \
    --input_pickle /tmp/input_data.pickle \
    --output_pickle /tmp/relation_output.pickle \
    --device cuda
```

**Note:** This script is typically not called directly. It's invoked by `m2i_receding_horizon_full.py`.

---

### 8. `subprocess_conditional.py`

**Purpose:** Isolated subprocess for Conditional V2V inference.

**Similar to `subprocess_relation.py` but:**
- Loads Conditional V2V model (model.29.bin)
- Expects influencer trajectories as input
- Outputs reactor trajectories conditioned on influencer

**Direct Usage (rarely needed):**

```bash
cd /workspace

python models/pretrained/m2i/scripts/lib/subprocess_conditional.py \
    --input_pickle /tmp/input_data.pickle \
    --output_pickle /tmp/conditional_output.pickle \
    --device cuda
```

---

## Common Workflows

### Workflow 1: Receding Horizon with Movies (Recommended)

```bash
cd /workspace

# Using bash wrapper (simplest)
bash models/pretrained/m2i/scripts/bash/run_m2i_pipeline.sh -n 10

# Or directly with Python
python models/pretrained/m2i/scripts/lib/m2i_receding_horizon_full.py \
    --num_scenarios 10 \
    --start_t 10 \
    --end_t 90 \
    --step 1 \
    --generate_movies \
    --device cuda
```

### Workflow 2: Full 3-Stage Pipeline with Subprocess Isolation

```bash
cd /workspace

# Using bash wrapper
bash models/pretrained/m2i/scripts/bash/run_m2i_pipeline.sh -n 10 --subprocess-pipeline

# Or directly with Python
python models/pretrained/m2i/scripts/lib/m2i_receding_horizon_full.py \
    --num_scenarios 10 \
    --subprocess-pipeline \
    --generate_movies \
    --device cuda
```

### Workflow 3: Fast Preview (Larger Timestep Steps)

```bash
cd /workspace

# Process every 5th timestep instead of every timestep
bash models/pretrained/m2i/scripts/bash/run_m2i_pipeline.sh -n 5 --step 5
```

### Workflow 4: Legacy Static Inference + Visualization

```bash
cd /workspace

# Step 1: Run inference
python models/pretrained/m2i/scripts/lib/m2i_live_inference.py \
    --num_scenarios 50 \
    --output_dir output/m2i_live \
    --device cuda

# Step 2: Generate movies
python models/pretrained/m2i/scripts/lib/generate_m2i_movies.py \
    --predictions_dir output/m2i_live \
    --output_dir models/pretrained/m2i/movies/m2i_predictions \
    --num_scenarios 20 \
    --mode both
```

---

## Troubleshooting

### "Model file not found: model.24.bin"

You need to place pretrained weights at:
```
models/pretrained/m2i/models/densetnt/model.24.bin
models/pretrained/m2i/models/relation_v2v/model.25.bin
models/pretrained/m2i/models/conditional_v2v/model.29.bin
```

### "Cannot import dataset_waymo / modeling.vectornet"

M2I source is not on PYTHONPATH. The scripts add it automatically, but ensure:
```
/workspace/externals/M2I/src/
```
exists and contains `dataset_waymo.py`, `modeling/`, etc.

### "Lane type out of range" / "Lane id too large"

The flat parser patch wasn't applied. Ensure `patch_waymo_tutorial()` is called
before any dataset creation.

### Slow inference on CPU

DenseTNT and VectorNet are GPU-intensive. Use `--device cuda` if available.
Expect ~10x slowdown on CPU.

### Memory errors with large batches

Reduce `--num_scenarios` or process in smaller batches.

---

## Dependencies

All scripts require:
- PyTorch (with CUDA for GPU inference)
- TensorFlow (for TFRecord parsing)
- NumPy
- Matplotlib (for visualization)
- FFmpeg (for MP4 generation)

The M2I models require the upstream M2I repository at `/workspace/externals/M2I/`.
