# Waymax Bridge

This module provides the integration layer between RECTOR's trajectory prediction pipeline and Google DeepMind's Waymax driving simulator. It handles the full data lifecycle: loading WOMD scenarios into Waymax-compatible formats, creating simulation environments, converting RECTOR outputs to simulator actions, extracting observations for the next prediction step, and collecting safety and performance metrics throughout the rollout.

The bridge is designed so that upstream components (trajectory generators, selectors) never interact with Waymax directly — they work with standard NumPy/PyTorch tensors and Python dataclasses. This separation enables testing generators and selectors without a running simulator.

---

## Module Architecture

```
WOMD TFRecords
       │
       ▼
┌─────────────────────────────┐
│  scenario_loader.py         │  Parse protobuf → SimulatorState
│  ┌─────────────────────┐    │  (trajectories, roadgraph, traffic lights)
│  │ load_scenarios()    │    │
│  │ scenario_proto_to_  │    │
│  │ simulator_state()   │    │
│  └─────────────────────┘    │
└─────────────────────────────┘
       │
       ▼
┌─────────────────────────────┐
│  env_factory.py             │  SimulatorState → PlanningAgentEnvironment
│  ┌─────────────────────┐    │  (dynamics model, agent policies, init)
│  │ make_env()          │    │
│  └─────────────────────┘    │
└─────────────────────────────┘
       │
       ▼
┌───────────────────────────────────────────────────────────────┐
│  simulation_loop.py — WaymaxRECTORLoop                       │
│                                                               │
│  For each step:                                               │
│  ┌────────────────┐  ┌───────────────┐  ┌─────────────────┐  │
│  │ observation_    │→ │ generator     │→ │ selector        │  │
│  │ extractor.py    │  │ .generate()   │  │ .select()       │  │
│  └────────────────┘  └───────────────┘  └─────────────────┘  │
│                                                  │            │
│  ┌────────────────┐  ┌───────────────┐           │            │
│  │ metric_        │← │ action_       │←──────────┘            │
│  │ collector.py   │  │ converter.py  │                        │
│  └────────────────┘  └───────────────┘                        │
└───────────────────────────────────────────────────────────────┘
       │
       ▼
ScenarioResult (metrics, decision traces, wall time)
```

---

## Modules

### scenario_loader.py — WOMD TFRecord Parser

**Core function:** `scenario_proto_to_simulator_state(scenario_proto)` → `SimulatorState`

Converts a Waymo Scenario protobuf into a Waymax-compatible `SimulatorState` containing:

| Output Field | Structure | Description |
|-------------|-----------|-------------|
| Object trajectories | Trajectory dataclass with 11 fields per agent per timestep: x, y, z, vel_x, vel_y, yaw, valid, length, width, height, timestamp_micros | Each field is a [N, T] array. Together they fully describe object motion and geometry over time. |
| Object metadata | [N] | Object types, IDs, is_sdc, is_valid, objects_of_interest |
| Roadgraph points | RoadgraphPoints dataclass with 9 fields: x, y, z, dir_x, dir_y, dir_z, types, ids, valid | Each field is a [P] array. Points are grouped by segment ID and typed (1=lane center, 2=road line, 3=road edge, 4=stop sign, 5=crosswalk, 6=speed bump). |
| Traffic lights | TrafficLights dataclass with 6 fields: x, y, z, state, lane_ids, valid | Each field is a [T, S] array (timesteps x signal slots). |

**Key design choices:**
- SDC (self-driving car) is swapped to slot 0 only if `sdc_idx >= max_objects` (to ensure the SDC is not truncated). If the SDC index is already within the max_objects range, it remains at its original slot.
- Roadgraph points are grouped by segment ID and typed (1=lane center, 2=road line, 3=road edge, 4=stop sign, 5=crosswalk, 6=speed bump)
- Arrays are padded or truncated to configurable limits (max_objects=128, max_rg_points=30000)
- Temporal window: 10 past + 1 current + 80 future = 91 timesteps

**Generator function:** `load_scenarios(config)` yields `(scenario_id, SimulatorState)` tuples by iterating through TFRecord shards.

### env_factory.py — Environment Construction

**Core function:** `make_env(state, dynamics_model, agent_model, ...)` → `(PlanningAgentEnvironment, initial_state)`

Creates a Waymax environment configured for RECTOR evaluation:

| Parameter | Options | Default | Rationale |
|-----------|---------|---------|-----------|
| dynamics_model | "delta", "bicycle" | "delta" | DeltaGlobal matches RECTOR's position-based outputs |
| agent_model | "log_playback", "idm" | "idm" | IDM provides reactive non-ego agent behavior; "log_playback" replays logged trajectories |
| init_steps | int | 11 | WOMD warm-up convention (10 past + 1 current) |

**Why DeltaGlobal:** RECTOR produces trajectories as (x, y, vx, vy) in ego-local coordinates. DeltaGlobal interprets actions as position/heading deltas, which is a direct match. The InvertibleBicycleModel would reinterpret these deltas through bicycle kinematics, causing compounding yaw errors.

### simulation_loop.py — Closed-Loop Orchestration

**Core class:** `WaymaxRECTORLoop`

Runs a complete closed-loop simulation for one scenario:

1. Creates environment via `make_env()`
2. Loops for `max_sim_steps` (default 80 = 8 seconds at 10 Hz)
3. At each replanning instant (every `steps_per_replan` = 20 steps):
   - Extracts observation from current state
   - Calls `generator.generate()` for K trajectory candidates
   - Computes tier scores by aggregating per-rule costs into 4 tiers using `TIER_INDICES`
   - Calls `selector.select()` to get the best candidate and a `DecisionTrace`
   - Stores selected trajectory for execution
4. At each simulation step: converts trajectory to action, steps environment, collects metrics
5. Returns `ScenarioResult` with aggregated metrics and full decision trace history

**Batch function:** `run_batch(loader_cfg, selectors, generator)` processes all scenario x selector combinations with progress tracking. It iterates through scenarios from the loader, and for each scenario runs every selector, returning a flat list of `ScenarioResult` objects.

**MockLogReplayGenerator**: Testing utility that creates 6 candidate trajectories from the logged ego trajectory with added noise. Each candidate receives all-zero rule costs (`np.zeros`), making it useful for validating the simulation pipeline end-to-end without requiring the real RECTOR model. Note: `DiverseMockGenerator` (in `enhanced_bev_rollout.py`) is the variant that produces synthetic non-zero rule violations for visualization demos.

### observation_extractor.py — State-to-Tensor Conversion

**Core function:** `extract_observation(state, config)` → `Dict[str, Tensor]`

Converts Waymax `SimulatorState` into the tensor format expected by RECTOR models:

| Output Key | Shape | Coordinate Frame | Description |
|-----------|-------|------------------|-------------|
| `ego_history` | [1, 11, 4] | Ego-local, normalized | (x, y, heading, speed) / TRAJECTORY_SCALE |
| `agent_states` | [1, 32, 11, 4] | Ego-local, normalized | Nearest 32 agents, same format |
| `lane_centers` | [1, 64, 20, 2] | Ego-local, normalized | 64 nearest lanes, 20 points each |
| `agent_mask` | [1, 32] | — | Binary validity mask |
| `lane_mask` | [1, 64] | — | Binary validity mask |
| `ego_pos` | [2] | World frame | Reference position (meters) |
| `ego_yaw` | scalar | World frame | Reference heading (radians) |

**TRAJECTORY_SCALE = 50.0** — All spatial coordinates are divided by 50 for numerical stability during model inference, then multiplied back when converting to actions.

### action_converter.py — Trajectory-to-Action Conversion

**Core function:** `trajectory_to_action(trajectory, ego_pos, ego_yaw, state, steps)` → `Action`

Converts the selected trajectory from RECTOR's ego-local normalized coordinate system to Waymax's global-frame DeltaGlobal actions:

1. **Denormalize**: Multiply positions by TRAJECTORY_SCALE (50.0)
2. **Rotate to world frame**: Apply 2D rotation by ego_yaw
3. **Translate to world frame**: Add ego_pos
4. **Compute heading**: arctan2(vy, vx) from velocity components
5. **Compute deltas**: (dx, dy, dyaw) relative to current SDC state in the simulator
6. **Wrap angles**: Ensure dyaw is in [-pi, pi]

### metric_collector.py — Metric Accumulation

**Core class:** `MetricCollector`

Accumulates per-step metrics and computes aggregate statistics:

| Metric Category | Per-Step | Aggregated |
|----------------|----------|------------|
| Overlap | Binary collision flag | Rate (%), max overlap |
| Log divergence | Distance to logged trajectory | Mean, final, max |
| Kinematic infeasibility | Constraint violation flag | Rate (%) |
| Jerk | Acceleration derivative magnitude | Mean, max |
| Min clearance | Distance to nearest object | Min, mean |
| TTC | Time-to-collision estimate | Min, mean |

**TTC computation** uses constant-velocity assumption: solves the quadratic equation `|rel_pos + t * rel_vel|^2 = combined_radius^2` for the smallest positive t.

---

## Related Documentation

- [../README.md](../README.md) — Simulation engine overview
- [../selectors/README.md](../selectors/README.md) — Selection strategy details
- [../viz/README.md](../viz/README.md) — Visualization module
