# Closed-Loop Simulation Engine

This module implements a complete closed-loop simulation framework for evaluating RECTOR's trajectory selection strategies in realistic driving scenarios. It bridges RECTOR's trajectory prediction with the Waymax simulator (Google DeepMind's JAX-based driving simulator) to measure how trajectory selection decisions affect long-horizon driving outcomes.

The central question this module answers: **Does RECTOR's lexicographic rule-aware selection produce safer, more compliant driving behavior than confidence-based or weighted-sum selection — not just in open-loop prediction, but when decisions compound over time in a closed-loop setting?**

---

## Architecture

The simulation engine follows a modular pipeline design:

```
WOMD TFRecords
       │
       ▼
scenario_loader ─── Parse Scenario protos → SimulatorState
       │
       ▼
env_factory ─────── Create Waymax PlanningAgentEnvironment
       │
       ▼
WaymaxRECTORLoop ── Closed-loop simulation:
  │                   ┌─────────────────────────────────┐
  │                   │  For each replanning instant:    │
  │                   │  1. observation_extractor()      │
  │                   │  2. generator.generate()         │
  │                   │  3. selector.select()            │
  │                   │  4. action_converter()           │
  │                   │  5. env.step()                   │
  │                   │  6. metric_collector.step()      │
  │                   └─────────────────────────────────┘
       │
       ▼
ScenarioResult ──── Aggregated metrics per scenario
       │
       ▼
viz/* ──────────── Figures, tables, and videos
```

### Core Design Decisions

1. **Waymax as simulation backend**: Waymax provides physics-based vehicle dynamics (DeltaGlobal or InvertibleBicycle), JAX-accelerated metric computation, and built-in overlap/offroad/kinematic-infeasibility detection. Using an established simulator avoids reimplementing collision physics.

2. **DeltaGlobal dynamics**: RECTOR outputs position-based trajectories in ego-local coordinates. The `DeltaGlobal` dynamics model interprets actions as (dx, dy, dyaw) deltas, which is a natural fit. The `InvertibleBicycleModel` was found to cause yaw spirals because it reinterprets position deltas through bicycle kinematics.

3. **Log-replay for non-ego agents**: Other vehicles follow their ground-truth logged trajectories. This isolates the evaluation to ego behavior — RECTOR's selection strategy is the only variable. The IDM (Intelligent Driver Model) option exists for reactive agent experiments.

4. **Re-plan every 20 steps (2 seconds)**: At each replanning instant, the trajectory generator produces K=6 fresh candidates over a 5-second horizon. The selector picks the best mode, and the ego follows it for 20 steps before replanning. This mimics real-time operation at 0.5 Hz planning frequency.

5. **Three selector strategies for A/B comparison**: Confidence (baseline), Weighted Sum (tuned baseline), and RECTOR Lexicographic (proposed method) all implement the same `BaseSelector` interface, enabling controlled comparisons with identical data flow.

---

## Directory Structure

```text
simulation_engine/
├── config.py                    Hydra-compatible dataclass configuration
├── validate_50.py               End-to-end validation (50 scenarios × 3 selectors)
├── __init__.py
├── waymax_bridge/               Waymax simulator integration layer
│   ├── env_factory.py           Create PlanningAgentEnvironment
│   ├── scenario_loader.py       WOMD TFRecords → SimulatorState
│   ├── simulation_loop.py       Main closed-loop orchestration
│   ├── action_converter.py      Ego-local trajectory → Waymax Action
│   ├── observation_extractor.py SimulatorState → RECTOR input tensors
│   ├── metric_collector.py      Step-wise and aggregate metrics
│   └── __init__.py
├── selectors/                   Trajectory selection strategies
│   ├── base.py                  Abstract BaseSelector + DecisionTrace
│   ├── confidence.py            Baseline: argmax(confidence)
│   ├── weighted_sum.py          Tuned weighted aggregation
│   ├── rector_lex.py            RECTOR lexicographic selection
│   └── __init__.py
└── viz/                         Visualization and analysis
    ├── generate_all.py          Master orchestrator
    ├── bev_rollout.py           Simple BEV animation
    ├── enhanced_bev_rollout.py  Dual-panel BEV + violations
    ├── generate_bev_batch.py    Batch BEV movie generator
    ├── generate_bev_challenging.py  BEV movies for complex infrastructure scenarios
    ├── metric_distributions.py  Violin and box plots
    ├── scenario_heatmap.py      Scenario × metric heatmap
    ├── scenario_safety_profile.py  Per-scenario safety bars
    ├── summary_table.py         LaTeX and Markdown tables
    └── __init__.py
```

---

## Modules

### config.py — Experiment Configuration

Centralized dataclass-based configuration for all simulation parameters:

| Config Class | Key Parameters | Purpose |
|-------------|----------------|---------|
| `WaymaxBridgeConfig` | dt_replan=2.0s, horizon=5.0s, hz=10, dynamics="delta" | Simulation timing and dynamics |
| `AgentConfig` | agent_model="log_playback" or "idm" | Non-ego agent behavior |
| `SelectorConfig` | methods=["confidence","weighted_sum","rector_lex"], rector_epsilon=[1e-3]*4 | Selection strategy parameters |
| `MiningConfig` | max_scenarios=12800, per-class targets | Scenario dataset configuration |
| `MetricConfig` | bootstrap_resamples=10000, TTC_threshold=1.5s | Analysis thresholds |
| `ExperimentConfig` | Top-level aggregator | Output directories |

### waymax_bridge/ — Simulator Integration

See [waymax_bridge/README.md](waymax_bridge/README.md) for detailed documentation of each component.

**Data flow through the bridge:**

1. `scenario_loader.load_scenarios()` — Parses TFRecord shards, extracts trajectories (x, y, z, vel_x, vel_y, yaw), roadgraph points grouped by segment ID (lanes, road lines, edges, stop signs, crosswalks, speed bumps), and traffic light states. Ensures the SDC is in slot 0 if its index would exceed the max_objects limit.

2. `env_factory.make_env()` — Creates a `PlanningAgentEnvironment` with DeltaGlobal dynamics and log-replay non-ego agents. Initializes with 11 warm-up steps (WOMD convention).

3. `observation_extractor.extract_observation()` — Converts the Waymax state to RECTOR input tensors: ego history [1,11,4], agent states [1,32,11,4], lane centers [1,64,20,2], all in ego-centric coordinates normalized by TRAJECTORY_SCALE=50.

4. `action_converter.trajectory_to_action()` — Converts the selected trajectory from ego-local normalized coordinates to Waymax DeltaGlobal actions (dx, dy, dyaw).

5. `metric_collector.step()` — Accumulates per-step metrics (overlap, log divergence, kinematic infeasibility, jerk, TTC, clearance).

### selectors/ — Trajectory Selection Strategies

See [selectors/README.md](selectors/README.md) for detailed documentation.

All selectors implement the same interface:

```python
def select(candidates, probs, tier_scores, rule_scores, rule_applicability)
    → (selected_idx: int, DecisionTrace)
```

| Selector | Method | Complexity | Key Property |
|----------|--------|------------|--------------|
| `ConfidenceSelector` | argmax(confidence) | O(K) | Ignores rules entirely |
| `WeightedSumSelector` | argmin(w·violations) | O(K·R) | Can trade safety for comfort |
| `RECTORLexSelector` | Tier-by-tier elimination | O(K·T) | **Safety violations can never be traded for comfort improvements** |

The `DecisionTrace` dataclass records the full decision audit trail: which candidates survived each tier, which tier first eliminated candidates, per-candidate scores, and a human-readable reason string.

### viz/ — Visualization and Analysis

See [viz/README.md](viz/README.md) for detailed documentation.

---

## Running Simulations

### Quick Validation (50 scenarios, ~5 minutes)

```bash
cd /workspace
python scripts/simulation_engine/validate_50.py
```

Runs 50 scenarios × 3 selectors with reduced parameters (max_objects=32, max_rg_points=5000) for fast verification. Results saved to `/workspace/output/closedloop/validate_50_results.json`.

### Full Artifact Generation

```bash
python scripts/simulation_engine/viz/generate_all.py \
    --results /workspace/output/closedloop/validate_50_results.json \
    --outdir /workspace/output/closedloop/artifacts/
```

Generates all visualization artifacts: metric distributions, safety profiles, summary tables, scenario heatmaps, and optionally BEV rollout videos.

### Enhanced BEV Movies (50 diverse scenarios, ~1 hour)

```bash
python scripts/simulation_engine/viz/enhanced_bev_rollout.py \
    --target-movies 50 --shards-to-scan 40 --scenarios-per-shard 40
```

Mines diverse scenarios (turns, lane changes, exit ramps, complex maneuvers), runs closed-loop simulation, and renders dual-panel videos with trajectory candidates and violation metrics.

---

## Metrics Collected

| Metric | Source | Description |
|--------|--------|-------------|
| Overlap rate | Waymax built-in | Fraction of steps with ego-agent bounding box overlap |
| Log divergence | Waymax built-in | Distance between ego position and logged human trajectory |
| Kinematic infeasibility | Waymax built-in | Fraction of steps violating kinematic constraints |
| Offroad rate | Waymax built-in | Fraction of steps where ego center is off drivable surface |
| Jerk (mean, max) | Custom | Acceleration derivatives — smoothness indicator |
| Min clearance | Custom | Minimum distance to any non-ego object at any timestep |
| TTC (min, mean) | Custom | Time-to-collision under constant-velocity assumption |

---

## Related Documentation

- [waymax_bridge/README.md](waymax_bridge/README.md) — Detailed bridge module documentation
- [selectors/README.md](selectors/README.md) — Selection strategy documentation
- [viz/README.md](viz/README.md) — Visualization module documentation
- [../../output/closedloop/videos/README.md](../../output/closedloop/videos/README.md) — Generated video documentation
- [../../models/RECTOR/README.md](../../models/RECTOR/README.md) — RECTOR scientific overview
