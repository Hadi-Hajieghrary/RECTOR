# RECTOR Core Pipeline Library

This directory contains the reusable components that power the M2I+RECTOR trajectory selection pipeline and the receding-horizon planning system. These modules implement the key claim that **rule reasoning can act as a layer on top of an existing prediction model** -- taking pretrained M2I candidates and selecting among them using explicit traffic rule evaluation.

Unlike the task-specific scripts in sibling directories, the modules here are designed as importable libraries used across experiments, visualizations, and closed-loop simulation.

---

## Architecture Overview

```
WOMD Scenario Proto
       |
       v
real_data_loader.py ---- Load and cache scenario data
       |
       v
scene_feature_extractor.py ---- Extract proxy-compatible features
       |
       |---> m2i_trajectory_generator.py ---- DenseTNT -> 6 trajectory modes
       |              |
       |              v
       |---> rector_pipeline.py ---- RECTOR model inference (learned)
       |         OR
       |---> m2i_rule_selector.py ---- Kinematic rule evaluation (explicit)
       |              |
       |              v
       |---> safety_layer.py ---- Multi-layer safety validation
       |              |
       |              v
       |---> batched_adapter.py ---- Thread-safe batched M2I inference
       |              |
       |              v
       +---> planning_loop.py ---- Receding-horizon orchestration
       |              |
       |              v
       |      data_contracts.py ---- Type-safe result structures
       |
       +---> rector_receding_horizon.py ---- Closed-loop simulation driver
       |
       +---> visualize.py ---- BEV rendering and movie generation
       |
       +---> generate_rector_movies.py ---- TFRecord-based movie generation
       |
       +---> benchmark.py ---- Latency and throughput benchmarks
       |
       +---> tune_parameters.py ---- Grid search and sensitivity analysis
```

---

## Modules in Detail

### planning_loop.py -- Receding-Horizon Planning

The `RECTORPlanner` orchestrates the full planning pipeline at each timestep:

1. **CandidateGenerator**: Samples M=16 ego trajectory candidates using lane-following at multiple speed levels, lane changes (left/right), emergency stops, goal-directed paths, and perturbations of existing candidates
2. **ReactorSelector**: Identifies K=3 most relevant surrounding agents based on distance, corridor overlap, and closing speed
3. **Batched M2I prediction**: For each (candidate, reactor) pair, predicts conditional futures
4. **Scoring**: Weighted linear combination with `ScoringWeights` (w_collision=100, w_comfort=5, w_progress=10, w_lane_deviation=2, w_hysteresis=3). Lower total score is better.
5. **Selection**: `argmin` over safe candidates with hysteresis -- sticks with previous selection if its score is within 1.0 of the new best. Falls back to lowest collision probability if all candidates are unsafe.

**Key method:** `plan_tick(ego_state, agent_states, current_lane, left_lane, right_lane, goal)` -- returns a `PlanningResult` (the local one defined in this file, not the one in `data_contracts.py`).

**Note:** There are two distinct `PlanningResult` classes. The one in `planning_loop.py` holds `selected_candidate`, `selected_idx`, `all_scores`, `safety_scores`, `timing_ms`, and `iteration`. The one in `data_contracts.py` holds `selected_candidate_id`, `selected_trajectory`, `candidate_scores`, `safety_scores`, `reactor_predictions`, `planning_time_ms`, `num_candidates_evaluated`, and `num_reactors`.

### rector_pipeline.py -- Full RECTOR Inference

The `RECTORPipeline` combines all learned components:

```
Scenario -> M2I Trajectory Generator (6 modes, frozen)
              |
           M2ISceneEncoder (trainable by default; freeze_m2i=False) -> RuleApplicabilityHead (learned)
              |
           DifferentiableRuleProxies (28 rules)
              |
           TieredRuleScorer -> best trajectory + audit
```

Contains `RECTORApplicabilityModel` -- the trainable subset (SceneEncoder + ApplicabilityHead) that can be loaded from a checkpoint for inference. The constructor accepts `freeze_m2i` (default `False`); set it to `True` to freeze encoder weights during fine-tuning.

### m2i_trajectory_generator.py -- DenseTNT Wrapper

Wraps the pretrained M2I DenseTNT model for multi-modal trajectory prediction:

- Converts WOMD Scenario protos to M2I's internal mapping format (polyline vectors + 224x224 BEV raster)
- Runs VectorNet forward pass with NMS-based diversity enforcement
- Returns [6, 80, 2] trajectory modes with confidence scores

**Constructor:** `M2ITrajectoryGenerator(model_path, device, future_frames, mode_num, hidden_size)`.

**Key challenge solved:** M2I uses global-state `other_params` for configuration. The `OtherParams` class provides a hybrid list/dict interface to handle M2I's inconsistent parameter access patterns.

### m2i_rule_selector.py -- Explicit Rule-Based Selection

Combines M2I trajectory generation with a `KinematicRuleEvaluator` for numpy-based rule violation scoring:

1. Generate 6 trajectories with DenseTNT
2. Evaluate kinematic violations per trajectory (comfort, safety, road rules)
3. Compute lexicographic composite scores across 4 tiers
4. Select the best trajectory by rule compliance; break ties by M2I confidence

**Key method:** `evaluate_scenario()` returns a `GenerationResult` with trajectories, violations, timing, and the selected index.

### safety_layer.py -- Multi-Layer Safety Validation

A comprehensive safety checking system with four specialized components:

| Component | Class | Purpose |
|-----------|-------|---------|
| Kinematic feasibility | `RealityChecker` | Validates acceleration, jerk, yaw rate, speed bounds |
| Collision detection | `OBBCollisionChecker` | Separating Axis Theorem for oriented bounding box overlap |
| Non-cooperative envelopes | `NonCoopEnvelopeGenerator` | 5 worst-case agent behaviors: constant velocity, max braking, mild braking, left drift, right drift |
| Risk scoring | `DeterministicCVaRScorer` | Conditional Value-at-Risk with Common Random Numbers |

The `IntegratedSafetyChecker` combines all components into a single `SafetyCheckResult` containing feasibility, collisions, clearance, CVaR score, and detailed violation flags.

**25/25 tests passing** (see `tests/test_safety_layer.py`).

### batched_adapter.py -- Efficient M2I Batching

The `BatchedM2IAdapter` optimizes M2I inference for the planning loop by avoiding redundant computation:

- Scene encoding runs **once per tick** (cached in `SceneEmbeddingCache`)
- Reactor tensor packs computed **once per reactor** (cached)
- All MxK prediction pairs processed in a **single batched forward pass**

Also provides `M2IArgsContext`, a thread-safe context manager that saves and restores M2I's module-level global `args` state using a global lock, preventing contamination across concurrent model calls.

Target latency: <50ms for M=16 candidates x K=3 reactors (batch size 48).

### real_data_loader.py -- Scenario Loading

Loads real WOMD scenarios from raw TFRecords (via Waymo Scenario protobufs) with in-memory caching:
- Extracts ego states, agent trajectories, lane centerlines
- Supports random sampling, iteration, and batch access
- Caches up to N scenarios for repeated evaluation

**Note:** This module defines its own local `AgentState` and `LaneInfo` dataclasses that are structurally similar but **not identical** to the ones in `data_contracts.py` (e.g., `agent_type` is a `str` in `real_data_loader.py` but an `int` in `data_contracts.py`; `LaneInfo` here has `lane_type` instead of `width`). Consumers should be aware of these type differences when passing data between modules.

### scene_feature_extractor.py -- Feature Extraction for Proxies

Converts WOMD Scenario protos into the feature dictionaries expected by differentiable rule proxies:
- Agent positions, sizes, types in ego-local frame
- Lane geometry, headings, speed limits
- Road edges, stoplines, signal states, crosswalk polygons

### data_contracts.py -- Type-Safe Data Structures

Immutable (frozen) dataclasses defining the contracts between pipeline stages:

| Contract | Key Fields | Purpose |
|----------|-----------|---------|
| `PlanningConfig` | 16 candidates, 3 reactors, CVaR config | Planning parameters |
| `ModelConfig` | hidden_size, future_frame_num, mode_num | M2I model parameters |
| `AgentState` | x, y, heading, velocity_x, velocity_y, agent_id | Single agent state |
| `AgentHistory` | agent_id, states (Tuple[AgentState, ...]) | Historical trajectory |
| `EgoCandidate` | candidate_id, trajectory [H,2], velocities, generation_method, parent_id | Single candidate |
| `EgoCandidateBatch` | trajectories [M,H,2] as tensor | GPU-ready batch |
| `ReactorTensorPack` | Pre-computed reactor features | Reusable across candidates |
| `SceneEmbeddingCache` | Encoder output + optional GPU cache | Avoid re-encoding |
| `SinglePrediction` | ego_candidate_id, reactor_id, trajectories, scores | One (candidate, reactor) prediction |
| `PredictionResult` | trajectories [M,K,N,H,2], scores [M,K,N] | All predictions batched |
| `CollisionCheckResult` | has_collision, collision_time, min_distance | Single collision check |
| `SafetyScore` | feasibility, collision, clearance, CVaR | Safety evaluation |
| `PlanningResult` | selected_candidate_id, selected_trajectory, candidate_scores, safety_scores, reactor_predictions, planning_time_ms, num_candidates_evaluated, num_reactors | Final output |

All contracts are frozen (immutable) to prevent mutation during planning and enable safe concurrent access.

### visualize.py -- BEV Rendering and Movie Generation

Bird's-Eye-View visualization utilities for RECTOR planning results (~720 lines):

- **`BEVConfig`**: Rendering parameters (image size, meters/pixel, ego-centered, grid, legend, trail length)
- **`BEVRenderer`**: Core renderer producing ego-centric BEV frames with road graphs, lane centerlines, oriented agent boxes, velocity arrows, trajectory candidates with score-based coloring, reactor predictions, safety envelopes, and collision points
- **`RECTORMovieGenerator`**: Accumulates rendered frames and writes MP4 animations (via imageio) or PNG sequences

Also includes `render_candidate_comparison()` for side-by-side candidate grid views and a `--demo` CLI mode that loads a real Waymo scenario and runs the planner.

### rector_receding_horizon.py -- Closed-Loop Simulation Driver

Runs RECTOR planning in a receding-horizon loop across Waymo scenarios (~560 lines):

- **`RecedingHorizonConfig`**: Start/end timestep, step size, planning config, movie generation flag
- **`WaymoScenarioParser`**: Parses TFRecord flat features into trajectory arrays, extracts agents at any timestep, and approximates nearest lanes from roadgraph points
- **`RECTORRecedingHorizon`**: Drives the `RECTORPlanner` through time, collects `TimestepResult`/`ScenarioResult` objects, computes metrics (mean/max/p95 planning time, selection switch rate, score statistics), and optionally generates visualization movies

CLI: `python rector_receding_horizon.py --tfrecord <path> --num_scenarios 5`

### benchmark.py -- Performance Benchmarking

Latency and throughput measurement suite for the RECTOR planning pipeline (~545 lines):

- **`BenchmarkResult`**: Statistics dataclass (mean, std, min, max, p50, p95, p99, pass/fail)
- **`run_benchmark(func, warmup, runs, threshold_ms, name)`**: Generic benchmark runner with warm-up, GC control, and threshold checking
- Individual benchmarks for candidate generation, reactor selection, scoring, and the full planning loop
- Scaling benchmarks varying M across [4, 8, 16, 32]
- Optional M2I inference and GPU memory profiling

Targets: batched forward pass < 50ms (Gate #2), full planning loop < 100ms (Gate #4).

### tune_parameters.py -- Parameter Tuning

Grid search and sensitivity analysis utilities for RECTOR planning parameters (~480 lines):

- **`TuningConfig`**: Ranges for M, K, CVaR alpha, hysteresis bonus
- **`EvaluationResult`**: Metrics per configuration (time, score, collision rate, switch rate, hysteresis effectiveness)
- **`grid_search()`**: Exhaustive search over parameter combinations
- **`sensitivity_analysis()`**: Single-parameter sweeps with fixed baseline
- **`find_pareto_frontier()`**: Pareto-optimal configurations on time-vs-score tradeoff
- **`RECOMMENDED_CONFIGS`**: Four presets -- "fast" (M=4, K=2), "balanced" (M=8, K=3), "thorough" (M=16, K=5), "conservative" (M=8, K=3, alpha=0.05)

### generate_rector_movies.py -- TFRecord-Based Movie Generation

A standalone movie generator that works directly from Waymo TFRecord flat features (~795 lines):

- **`RECTORMovieGenerator`** (separate class from the one in `visualize.py`): Parses TFRecords into scenario dicts with road polylines grouped by type, renders oriented vehicle patches with type-based coloring, draws history trails, ground-truth futures, candidate trajectories with safety scores, and a detailed legend
- Supports MP4 (via FFMpeg) and GIF (via Pillow) output
- Uses `matplotlib.animation.FuncAnimation` for frame generation
- Includes full roadgraph type mapping (freeway lanes, surface streets, bike lanes, road lines, edges, crosswalks, stop signs, speed bumps)

CLI: `python generate_rector_movies.py --tfrecord <path> -n 3 --num_candidates 8`

---

## Reading Order

For understanding the M2I+RECTOR method:
1. `m2i_trajectory_generator.py` -- How candidates are generated
2. `m2i_rule_selector.py` -- How rules are applied to candidates
3. `planning_loop.py` -- How selection is orchestrated over time

For understanding the learned RECTOR pipeline:
1. `rector_pipeline.py` -- End-to-end learned inference
2. `safety_layer.py` -- Post-selection safety validation

For understanding closed-loop evaluation:
1. `batched_adapter.py` -- Efficient batched M2I inference
2. `rector_receding_horizon.py` -- Simulation driver
3. `benchmark.py` -- Performance measurement

---

## Related Documentation

- [../models/README.md](../models/README.md) -- Neural architecture components
- [../proxies/README.md](../proxies/README.md) -- Differentiable rule proxies
- [../../README.md](../../README.md) -- RECTOR scientific overview
- [../../docs/](../../docs/) -- Design notes on batched adapter, planning loop, safety layer
