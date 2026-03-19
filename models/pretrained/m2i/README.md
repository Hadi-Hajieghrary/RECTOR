# `models/pretrained/m2i/` — M2I Receding Horizon Prediction and Visualization

This folder contains scripts that run the **M2I (Marginal-to-Interactive)** prediction
pipeline on Waymo Motion TFRecords using a **receding horizon** approach that simulates
real-time autonomous vehicle operation.

## Quick Start

```bash
# Run from /workspace directory
cd /workspace

# Run receding horizon pipeline with movie generation (recommended)
bash models/pretrained/m2i/scripts/bash/run_m2i_pipeline.sh -n 10

# Full 3-stage pipeline with subprocess isolation
bash models/pretrained/m2i/scripts/bash/run_m2i_pipeline.sh -n 10 --subprocess-pipeline

# Fast preview with larger timestep increments
bash models/pretrained/m2i/scripts/bash/run_m2i_pipeline.sh -n 5 --step 5
```

---

## Receding Horizon Concept

Traditional prediction runs once at t=10 and predicts the full future (t=11–90).
Receding horizon simulates real-time AV operation:

```
At t=10: Observe [t=0:10], Predict [t=11:90] → 8 seconds ahead
At t=20: Observe [t=10:20], Predict [t=21:90] → 7 seconds ahead
At t=50: Observe [t=40:50], Predict [t=51:90] → 4 seconds ahead
...
```

Benefits:
- Shows how predictions improve with more observations
- Simulates online operation where new data arrives continuously
- Generates ~8 second movies matching BEV movie duration

---

## Folder Structure

```
models/pretrained/m2i/
├── scripts/                 # All runnable scripts
│   ├── bash/                # Bash wrapper scripts
│   │   └── run_m2i_pipeline.sh  # Receding horizon pipeline runner
│   └── lib/                 # Python scripts and modules
│       ├── m2i_receding_horizon_full.py  # PRIMARY: Receding horizon inference
│       ├── generate_m2i_movies.py        # Static prediction visualization
│       ├── m2i_live_inference.py         # Legacy 3-stage pipeline
│       ├── m2i_receding_horizon.py       # Simple receding horizon (older)
│       ├── m2i_gpu_pipeline.py           # All-in-one sandbox
│       ├── subprocess_conditional.py     # Subprocess helper
│       ├── subprocess_relation.py        # Subprocess helper
│       └── waymo_flat_parser.py          # TFRecord parser patch
├── models/                  # Model weight files (not included)
│   ├── densetnt/
│   ├── relation_v2v/
│   └── conditional_v2v/
├── movies/                  # Generated visualization movies
│   ├── receding_horizon/    # Receding horizon movies (primary)
│   └── m2i_predictions/     # Static prediction movies
└── relations/               # Relation prediction outputs
```

---

## The Core Problem This Folder Solves

M2I was originally built around a specific Waymo Motion TF Example schema and parser
(`waymo_tutorial.py` in the upstream M2I repo).

In this workspace, the Waymo to TF Example conversion is configured slightly differently
(e.g., different roadgraph sample count), and many tensors appear in a **flat** form when
parsed by TensorFlow.

This folder provides:

### 1. A parser patch (`scripts/lib/waymo_flat_parser.py`)
- Defines a flat `features_description`
- Reshapes flat arrays into the shapes M2I expects
- Performs critical sanity fixes so M2I doesn't crash:
  - Clamps lane types to the 0–19 range
  - Wraps lane IDs to the 0–999 range

### 2. A live inference runner (`scripts/lib/m2i_live_inference.py`)
- Runs DenseTNT marginal predictions directly from TFRecords (no precomputed files)
- Runs Relation prediction (influencer ↔ reactor)
- Runs Conditional prediction (reactor conditioned on influencer)

### 3. Visualization tools
- Overlay predictions onto a BEV view of the scene
- Generate static images and MP4 movies

---

## Expected Pretrained Weights

The pretrained M2I weight files are present in this workspace at these locations:

```
models/pretrained/m2i/models/
  densetnt/model.24.bin
  relation_v2v/model.25.bin
  conditional_v2v/model.29.bin
```

All three `.bin` files are currently available under `models/pretrained/m2i/models/`.

---

## Scripts in this folder (file-by-file)

### `scripts/bash/run_m2i_pipeline.sh`
The primary pipeline runner using receding horizon prediction:

```bash
# Basic usage
bash models/pretrained/m2i/scripts/bash/run_m2i_pipeline.sh -n 10

# With 3-stage pipeline (DenseTNT + Relation + Conditional)
bash models/pretrained/m2i/scripts/bash/run_m2i_pipeline.sh -n 10 --subprocess-pipeline

# Fast preview with larger timestep steps
bash models/pretrained/m2i/scripts/bash/run_m2i_pipeline.sh -n 5 --step 5
```

Key options:
- `-n, --num_scenarios` : how many scenarios to process
- `--start_t`           : first prediction timestep (default: 10)
- `--end_t`             : last prediction timestep (default: 90)
- `--step`              : timestep increment (default: 1)
- `--subprocess-pipeline` : enable 3-stage M2I with subprocess isolation
- `--no-movies`         : skip movie generation
- `--device`            : `cuda` or `cpu`

Outputs:
- Prediction pickles: `output/m2i_live/receding_horizon/predictions.pickle`
- Movies: `models/pretrained/m2i/movies/receding_horizon/*.mp4`

---

### `scripts/lib/m2i_live_inference.py`
**Purpose:** Run the canonical M2I *three-stage* pipeline **live**.

Stages:
1. **DenseTNT (marginal)**
   Predicts 6 future trajectory modes for each agent independently.
2. **Relation V2V**
   Predicts which agent is the “influencer” and which is the “reactor”
   (relation direction between the two objects-of-interest).
3. **Conditional V2V**
   Predicts the reactor trajectory conditioned on the influencer trajectory hypothesis.

Important details you’ll miss if you only skim:
- The script intentionally resets random seeds before dataset creation to improve reproducibility.
- It stores DenseTNT predictions in world coordinates (M2I decoder already reverses normalization).
- It uses a “batch over mappings” internal representation from M2I’s `Dataset`.

Output files (pickles):
- `marginal/marginal_predictions.pickle`
  A dict: `scenario_id -> {rst, score, ids}` where:
  - `rst`   has shape `[N_agents, 6, 80, 2]` (modes × timesteps × XY)
  - `score` has shape `[N_agents, 6]`
  - `ids`   has shape `[N_agents]` (Waymo track IDs)

- `relation/relation_predictions.pickle`
  A dict stored by M2I’s `globals.sun_1_pred_relations`.
  (This is how upstream M2I exposes relation outputs.)

- `conditional/conditional_predictions.pickle`
  Same `{rst, score, ids}` structure as marginal, but produced by the conditional model.

CLI highlights:
- `--stage {marginal,relation,conditional,all}`
- `--num_scenarios`
- `--data_dir`
- `--output_dir`
- `--device`
- `--seed`

---

### `scripts/lib/generate_m2i_movies.py`
**Purpose:** Turn TFRecords + prediction pickles into human-viewable media.

It reads:
- TF Example TFRecord files (for geometry + ground truth)
- the marginal + conditional prediction pickles
- (optionally) relation predictions to determine influencer/reactor roles

It generates:
- static PNGs
- MP4 movies

Notable design choices:
- It assumes an **interactive scenario**: two key agents where:
  - “reactor” is usually taken from the conditional predictions
  - “influencer” is chosen from the marginal predictions list
- It overlays:
  - past trajectories
  - ground truth future trajectories
  - predicted trajectories (modes)
  - endpoints (stars) for easy FDE intuition

CLI highlights:
- `--data_dir`
- `--predictions_dir`
- `--output_dir`
- `--num_scenarios`
- `--mode {image,movie,both}`
- `--fps`

---

### `scripts/lib/waymo_flat_parser.py`
**Purpose:** Make M2I compatible with this workspace's TFRecord feature shapes.

What it provides:
- `features_description_flat`: a TensorFlow `FixedLenFeature` schema where certain tensors
  are declared as **flat vectors** (e.g., roadgraph xyz is `[90000]` not `[30000,3]`).
- `_parse_flat(example)`: parses a TF Example and reshapes arrays into M2I’s expected shapes.
- `patch_waymo_tutorial()`: monkey-patches the upstream `waymo_tutorial._parse` with `_parse_flat`.

Two critical “defensive” fixes:
1) `roadgraph_samples/type` is clipped to **max 19**
   because M2I assumes 20 lane/road types (0–19).
2) `roadgraph_samples/id` is wrapped modulo **1000**
   because M2I asserts lane IDs are within a fixed range.

If you see M2I crashes like "lane type out of range" or "lane id too large",
this file is the reason they stop happening.

---

### `scripts/lib/m2i_receding_horizon.py`
A receding-horizon experiment script (a lighter/older variant) that re-runs prediction
at multiple timesteps within a single scenario, mimicking online prediction updates.

Use this if you want a simpler baseline than `scripts/lib/m2i_receding_horizon_full.py`.

---

### `scripts/lib/m2i_receding_horizon_full.py`
A **fully documented** receding-horizon runner that can:
- render BEV rasters matching M2I’s raster encoder expectations
- construct road polyline vectors for VectorNet
- run prediction repeatedly from `start_t` to `end_t`
- compute ADE/FDE style metrics
- optionally generate MP4 movies showing predictions evolving over time

It also supports a **subprocess-isolated 3-stage pipeline**, which is the recommended way
to run the full M2I pipeline without global-arg interference inside M2I's code.

---

### `scripts/lib/subprocess_relation.py` and `scripts/lib/subprocess_conditional.py`
These are helper scripts used by `scripts/lib/m2i_receding_horizon_full.py`
to run stages 2 and 3 in a clean Python interpreter.

Why subprocesses?
- Upstream M2I uses global variables and a shared `utils.args` object.
- Running different models sequentially in one process can cause subtle collisions.
- Spawning a subprocess gives each stage a clean global state and reduces “mystery bugs”.

Each subprocess script:
- loads a stage-specific model
- loads the dataset
- runs inference
- writes a pickle file to the requested output path

---

### `scripts/lib/m2i_gpu_pipeline.py`
A large "all-in-one" script that combines:
- live inference
- optional legacy precomputed prediction loading
- visualization utilities (including GIF generation)

This is useful as a sandbox, but if you want the cleanest mental model:
- prefer `scripts/lib/m2i_live_inference.py` + `scripts/lib/generate_m2i_movies.py`
- prefer `scripts/lib/m2i_receding_horizon_full.py` for receding-horizon analysis

---

## Common Failure Modes

### "Model file not found: model.24.bin"
You haven't placed the pretrained `.bin` weights in `models/pretrained/m2i/models/...`.

### "Cannot import dataset_waymo / modeling.vectornet"
Your upstream M2I repo is not on `PYTHONPATH` (or not at `/workspace/externals/M2I/src`).

### Lane type or lane ID assertions in M2I
If you bypass `scripts/lib/waymo_flat_parser.patch_waymo_tutorial()`, M2I may crash when parsing.
Always ensure the patch is applied before dataset creation.

### CPU-only performance is very slow
DenseTNT and VectorNet are compute-intensive. Use CUDA if available.


---

## Notes about `.gitkeep`

This folder contains a `.gitkeep` file.

- Its only purpose is to keep this directory tracked in git even when weight files, movies,
  or other generated artifacts are not present.
- You can ignore it for functional purposes.

