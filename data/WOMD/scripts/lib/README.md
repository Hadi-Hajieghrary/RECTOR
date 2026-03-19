# `data/WOMD/scripts/lib/` — Python Implementations (Filter / Convert / Visualize)

This folder contains the core logic of the dataset pipeline.

All scripts here are designed to be:
- Runnable as standalone CLIs (each has an `argparse` main)
- Importable as libraries (their functions and classes can be reused)

> **Note**: Run all Python scripts from the `/workspace` directory:
> ```bash
> cd /workspace
> python data/WOMD/scripts/lib/<script_name>.py --help
> ```

Two distinct TFRecord formats appear throughout:

1. **Scenario format** (`--format scenario`)
   TFRecord where each record is a serialized `scenario_pb2.Scenario`.

2. **TF Example format** (`--format tf`)
   TFRecord where each record is a serialized `tf.train.Example` with keys like
   `state/past/x`, `roadgraph_samples/xyz`, etc.

---

## Scripts (file-by-file)

### `filter_interactive_scenario.py`
**Goal:** Create a subset of a raw Waymo scenario split containing only "interactive" scenarios.

Definition used in this repository:
- A scenario is **interactive** if it contains `objects_of_interest`
  and `len(objects_of_interest) == 2`.

What the script does:
- Reads every TFRecord file in an input directory.
- For each record:
  - parses the `Scenario` proto
  - checks the interactive condition above
  - optionally counts the pair type (vehicle/ped/cyclist)
  - optionally filters by interaction type (e.g., only `v2v`)
  - writes matching records to an output TFRecord file
  - optionally writes the non-interactive records to a separate directory

Key implementation details (small but important):
- **Object typing** is derived from the two `objects_of_interest` track IDs:
  - vehicle = `1`, pedestrian = `2`, cyclist = `3`
- Pair type keys:
  - `v2v`, `v2p`, `v2c`, `p2p`, `p2c`, `c2c`, `other`
- Uses multiprocessing **across TFRecord files** (`mp.Pool`) for speed.

CLI highlights:
- `--input-dir`, `--output-dir` (required)
- `--type {v2v,v2p,v2c,others}` (optional)
- `--count-type` (print breakdown)
- `--skip-filtering` (only compute stats; don’t write output)
- `--non-interactive-dir` (optional output for non-interactive scenarios)
- `--num-workers` (parallelism)
- `--move` (delete originals after writing)

---

### `scenario_to_example.py`
**Goal:** Convert raw Waymo Scenario TFRecords to Motion TF Example TFRecords in pure Python.

What it produces:
- A TFRecord of `tf.train.Example` records containing the Motion feature layout:
  - `scenario/id`
  - `state/past/*`, `state/current/*`, `state/future/*` for up to 128 agents
  - `roadgraph_samples/*` for up to 20,000 map samples
  - Traffic light state tensors
  - Tracks-to-predict masks and objects-of-interest masks

Key “shape” assumptions:
- max agents: **128**
- past steps: **10** (1 second at 10Hz)
- future steps: **80** (8 seconds at 10Hz)
- map samples: default **20,000** (configurable)
- traffic lights: **16** max

Important implementation detail:
- TF Example “shape” is implicit (it’s just a long list), so correctness depends on
  using the same “flattening and ordering” contract when parsing later.

CLI:
- `--input_dir` and `--output_dir` (required directories)
- `--max_agents` (default 128)
- `--num_map_samples` (default 20000)
- `--pattern` (glob pattern, default `*.tfrecord*`)

---

### `generate_bev_movie.py`
**Goal:** Generate BEV (bird’s-eye-view) **movies** from Waymo data.

Supported inputs:
- raw scenario TFRecords (`--format scenario`)
- processed TF Example TFRecords (`--format tf`)
- single-file mode (`--tfrecord <path> --scenario-index <i>`)

What the movie shows:
- road network (roadgraph points / map features)
- agents (boxes + headings)
- historical and future motion traces

Key visualization choices:
- “Follow ego” mode:
  - by default, the camera centers on the SDC/ego vehicle (or the first valid track)
  - `--no-follow` disables this and uses a fixed world-centric view
- `--view-range` sets the visible window in meters
- can optionally also save GIFs (`--gif`)

Outputs:
- MP4 files written under an output directory rooted at `/workspace/data/WOMD/movies/bev/`
  (unless overridden by `--output-dir`)

CLI highlights:
- `--tfrecord`, `--format`, `--split`, `--all`
- `--num` (how many scenarios per split)
- `--fps`, `--dpi`
- `--view-range`, `--no-follow`, `--gif`

---

### `visualize_scenario.py`
**Goal:** Generate **static plots** (PNG) from Waymo data.

Supported inputs:
- raw scenario TFRecords (`--format scenario`)
- processed TF Example TFRecords (`--format tf`)
- single-file mode (`--tfrecord <path> --scenario-index <i>`)

Visualization “modes” it can generate:
- **single-frame**: a compact plot around a key timestep
- **multi-frame** (`--multi-frame`): a grid of multiple timesteps to show evolution
- **combined** (`--combined`): richer composite images

Outputs:
- PNG images written under `/workspace/data/WOMD/visualizations/` by default.

CLI highlights:
- `--format {scenario,tf}`, `--split <name>`, `--num <N>`
- `--multi-frame`, `--combined`
- `--output-dir` to override destination

---

## When to prefer which tool

- Want a quick sanity-check plot?
  → `visualize_scenario.py`

- Want an animation you can scrub through?
  → `generate_bev_movie.py`

- Need TF Example TFRecords but don’t want Bazel?
  → `scenario_to_example.py`

- Need interactive-only subset from the raw dataset?
  → `filter_interactive_scenario.py`
