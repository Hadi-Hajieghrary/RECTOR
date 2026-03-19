# `models/pretrained/m2i/scripts/bash/` — M2I Bash Scripts

This folder contains bash wrapper scripts for running the M2I pipeline.

---

## `run_m2i_pipeline.sh`

The primary pipeline runner using **receding horizon** prediction.

### Quick Start

```bash
cd /workspace

# Basic usage: 10 scenarios with movie generation
bash models/pretrained/m2i/scripts/bash/run_m2i_pipeline.sh -n 10

# Full 3-stage pipeline with subprocess isolation (recommended for accuracy)
bash models/pretrained/m2i/scripts/bash/run_m2i_pipeline.sh -n 10 --subprocess-pipeline

# Fast preview with larger timestep steps
bash models/pretrained/m2i/scripts/bash/run_m2i_pipeline.sh -n 5 --step 5

# Inference only, no movies
bash models/pretrained/m2i/scripts/bash/run_m2i_pipeline.sh -n 20 --no-movies
```

### Receding Horizon Concept

Traditional prediction runs once at t=10 and predicts the full future.
Receding horizon simulates real-time AV operation:

```
At t=10: Observe [t=0:10], Predict [t=11:90] → 8 seconds ahead
At t=20: Observe [t=10:20], Predict [t=21:90] → 7 seconds ahead
At t=50: Observe [t=40:50], Predict [t=51:90] → 4 seconds ahead
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `-n, --num_scenarios` | `10` | Number of scenarios to process |
| `-d, --data_dir` | validation_interactive | Input TFRecord directory |
| `-o, --output_dir` | `output/m2i_live/receding_horizon` | Prediction output directory |
| `-m, --movies_dir` | `movies/receding_horizon` | Movie output directory |
| `--start_t` | `10` | First prediction timestep |
| `--end_t` | `90` | Last prediction timestep |
| `--step` | `1` | Timestep increment |
| `--device` | `cuda` | Device: `cuda` or `cpu` |
| `--subprocess-pipeline` | off | Enable 3-stage pipeline with subprocess isolation |
| `--no-movies` | off | Skip movie generation |
| `-h, --help` | - | Show help message |

### Output

**Predictions:**
- `output/m2i_live/receding_horizon/predictions.pickle`

**Movies:**
- `models/pretrained/m2i/movies/receding_horizon/*.mp4` (10 FPS, ~8 seconds)
- `models/pretrained/m2i/movies/receding_horizon/*.gif` (shareable format)

### Examples

```bash
# Quick test
bash models/pretrained/m2i/scripts/bash/run_m2i_pipeline.sh -n 2

# Full run with 50 scenarios
bash models/pretrained/m2i/scripts/bash/run_m2i_pipeline.sh -n 50 --start_t 10 --end_t 90 --step 1

# CPU only (slower)
bash models/pretrained/m2i/scripts/bash/run_m2i_pipeline.sh -n 5 --device cpu

# Custom data directory
bash models/pretrained/m2i/scripts/bash/run_m2i_pipeline.sh -n 10 \
    -d /path/to/testing_interactive
```

---

## Pipeline Modes

### DenseTNT Only (Default)

Runs marginal trajectory prediction independently for each agent.
Fast and reliable.

```bash
bash models/pretrained/m2i/scripts/bash/run_m2i_pipeline.sh -n 10
```

### 3-Stage Subprocess Pipeline (Recommended for Accuracy)

Runs the full M2I pipeline:
1. **DenseTNT (marginal)** — Independent trajectory prediction
2. **Relation V2V** — Determines influencer/reactor relationships
3. **Conditional V2V** — Reactor prediction conditioned on influencer

Uses subprocess isolation to avoid global variable conflicts in M2I's code.

```bash
bash models/pretrained/m2i/scripts/bash/run_m2i_pipeline.sh -n 10 --subprocess-pipeline
```
