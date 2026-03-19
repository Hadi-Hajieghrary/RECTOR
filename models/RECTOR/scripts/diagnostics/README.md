# RECTOR Trajectory Diagnostics

This module provides diagnostic tools for inspecting trajectory quality, debugging data pipeline discrepancies, and validating kinematic properties of model outputs. These scripts are not used for paper results — they are development tools for identifying and resolving issues during model development.

---

## Scripts

### diagnostic_compare.py — Pipeline Parity Check

**Purpose:** Verifies that the training data pipeline (`WaymoDataset`) and the receding-horizon visualization pipeline (`ScenarioLoader`) produce identical model inputs for the same scenario at the same timestep.

**Why this matters:** RECTOR's training and visualization pipelines parse WOMD data through different code paths. If these paths produce different tensor values, the model receives different inputs during evaluation vs. visualization, leading to inconsistent behavior. This script catches such discrepancies.

**What it checks:**
- Per-tensor maximum absolute difference for ego_history, agent_states, lane_centers
- Per-column (x, y, heading, speed) difference statistics
- Model output comparison: trajectory endpoints and confidence scores from both input paths

**Note:** All paths are hardcoded. Uses `random.seed(42)`, selects one file from the first 10 validation files, and processes 1 scenario at t=10. No CLI arguments.

### diagnostic_quality.py — Trajectory Smoothness Inspection

**Purpose:** Analyzes whether model-generated trajectories exhibit artifacts like zig-zag oscillations, sudden lateral jumps, or physically implausible motion patterns.

**Metrics computed per trajectory mode:**
- **Lateral sign changes** (zig-zag count): Number of times the trajectory crosses the forward axis. Smooth trajectories have 0-1 crossings; oscillating trajectories have many.
- **Max lateral displacement**: Maximum perpendicular offset from the initial heading direction. Large values may indicate lane-departure behavior.
- **RMS lateral velocity**: Root-mean-square of lateral velocity component. High values indicate lateral instability.
- **ADE/FDE**: Standard accuracy metrics for context.

Also reports ground-truth trajectory smoothness for comparison baseline.

**Note:** All paths are hardcoded. Uses `random.seed(42)`, samples up to 5 validation files, and processes up to 10 scenarios. No CLI arguments.

### diagnostic_smooth.py — Smoothing Effectiveness Analysis

**Purpose:** Quantifies the effect of post-hoc Gaussian smoothing on trajectory quality metrics. Used to decide whether post-processing is needed before rule evaluation (smooth trajectories produce more reliable kinematic derivatives for comfort-tier rules).

**Comparison:**
- **Raw trajectory**: Direct model output
- **Smoothed trajectory**: Gaussian filter (sigma=3.0) applied to positions

For each scenario, reports before/after values for zig-zag count, max lateral displacement, and RMS lateral velocity. If smoothing dramatically improves metrics, it suggests the model's raw output has high-frequency noise that should be addressed in training rather than patched in post-processing.

**Note:** All paths are hardcoded. Uses `random.seed(42)`, samples up to 3 validation files, and processes up to 5 scenarios. No CLI arguments.

### diagnostic_methods.py — Trajectory Reconstruction Method Comparison

**Purpose:** Compares multiple trajectory reconstruction approaches to evaluate alternatives to Gaussian smoothing. Implements cubic spline reconstruction from anchor points as an alternative smoothing strategy.

**Methods compared:**
- **Raw**: Direct model output (no post-processing)
- **Gauss sigma=3**: Gaussian filter smoothing (current default)
- **Anchor-8**: Cubic spline fit through 8 evenly-spaced anchor points (clamped boundary)
- **Anchor-5**: Cubic spline fit through 5 evenly-spaced anchor points (clamped boundary)
- **GT**: Ground-truth trajectory for reference

For each method, reports zig-zag count (lateral sign changes), max lateral displacement, forward distance, and endpoint coordinates.

**Note:** All paths are hardcoded. Uses `random.seed(42)`, samples up to 5 validation files, and processes up to 8 scenarios. No CLI arguments.

---

## When to Use These Tools

| Symptom | Diagnostic | Expected Finding |
|---------|-----------|-----------------|
| Visualization shows different trajectories than training | `diagnostic_compare.py` | Tensor mismatch in ego_history or lane_centers |
| Rule evaluator flags comfort violations on seemingly smooth paths | `diagnostic_quality.py` | High zig-zag count or lateral velocity |
| Kinematic analysis shows unrealistic jerk values | `diagnostic_smooth.py` | Raw trajectories have high-frequency noise |
| Training loss converges but qualitative results look wrong | `diagnostic_compare.py` | Coordinate system or normalization mismatch |
| Need to evaluate alternative smoothing strategies | `diagnostic_methods.py` | Anchor-based spline vs. Gaussian quality tradeoffs |

---

## Related Documentation

- [../models/README.md](../models/README.md) — Neural architecture (trajectory output format)
- [../experiments/analyze_kinematics.py](../experiments/analyze_kinematics.py) — Systematic kinematic analysis (more comprehensive than diagnostics)
- [../training/README.md](../training/README.md) — Training pipeline (where trajectory issues originate)
