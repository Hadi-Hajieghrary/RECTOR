# Phase 4: Planning Loop

## Summary

**Status**: ✅ COMPLETE (15/15 tests passing)
**Date**: December 2024
**Location**: `/workspace/models/RECTOR/scripts/lib/planning_loop.py`

## Overview

The Planning Loop integrates all RECTOR components into a complete planning tick:

1. **Candidate Generation** - Generate M ego trajectory candidates
2. **Reactor Selection** - Select K most relevant reactors
3. **Prediction** - Get reactor responses (from M2I or dummy)
4. **Scoring** - Score candidates based on safety and progress
5. **Selection** - Pick best candidate with hysteresis

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         RECTORPlanner                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌───────────────────┐    ┌───────────────────┐   ┌─────────────────┐  │
│  │ CandidateGenerator│    │  ReactorSelector  │   │ CandidateScorer │  │
│  │                   │    │                   │   │                 │  │
│  │ - Lane following  │    │ - Distance-based  │   │ - Collision risk│  │
│  │ - Lane changes    │    │ - Corridor-based  │   │ - Comfort       │  │
│  │ - Emergency stop  │    │ - Approaching     │   │ - Progress      │  │
│  │ - Goal-directed   │    │                   │   │ - Hysteresis    │  │
│  └───────────────────┘    └───────────────────┘   └─────────────────┘  │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │                       plan_tick() Flow                             │ │
│  ├───────────────────────────────────────────────────────────────────┤ │
│  │  1. generate()  → EgoCandidateBatch (M candidates)                │ │
│  │  2. select()    → reactor_ids (K reactors)                        │ │
│  │  3. predict()   → Dict[reactor_id, List[SinglePrediction]]        │ │
│  │  4. score_all() → scores[M], safety[M]                            │ │
│  │  5. select_with_hysteresis() → best_idx                           │ │
│  └───────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

## Key Components

### CandidateGenerator

Generates M ego trajectory candidates using multiple strategies:

| Strategy | Description |
|----------|-------------|
| Lane Following | Follow lane at 0%, 50%, 75%, 100%, 110% of speed limit |
| Lane Change Left | Smooth lane change to left lane |
| Lane Change Right | Smooth lane change to right lane |
| Emergency Stop | Decelerate to stop (5 m/s²) |
| Goal-Directed | Steer toward goal position |
| Perturbations | Small variations of other candidates |

```python
generator = CandidateGenerator(m_candidates=16, horizon_steps=80, dt=0.1)
candidates = generator.generate(ego_state, current_lane, goal=goal)
```

### ReactorSelector

Selects K most relevant reactors based on:
- Distance to ego
- Whether in ego's corridor (envelope of all candidates)
- Closing speed (approaching vs receding)

```python
selector = ReactorSelector(k_reactors=3, corridor_width=10.0)
reactor_ids = selector.select(ego_candidates, agent_states, ego_state)
```

### CandidateScorer

Scores candidates based on multiple factors:

| Factor | Weight | Description |
|--------|--------|-------------|
| Collision Risk | 100.0 | CVaR-based collision probability |
| Comfort | 5.0 | Acceleration, jerk, lateral accel |
| Progress | 10.0 | Distance reduction toward goal |
| Lane Deviation | 2.0 | Average distance from lane center |
| Hysteresis | 3.0 | Similarity to previous selection |

```python
scorer = CandidateScorer(weights=ScoringWeights(w_collision=100.0))
scores, safety_scores = scorer.score_all(candidates, predictions, goal=goal)
```

### RECTORPlanner

Main planning loop integrating all components:

```python
planner = create_planner()

result = planner.plan_tick(
    ego_state=ego,
    agent_states=agents,
    current_lane=lane,
    goal=goal,
)

print(f"Selected: {result.selected_idx}")
print(f"Timing: {result.timing_ms}")
```

## Planning Result

```python
@dataclass
class PlanningResult:
    selected_candidate: EgoCandidate
    selected_idx: int
    all_scores: np.ndarray
    safety_scores: List[SimpleSafetyScore]
    timing_ms: Dict[str, float]
    iteration: int
```

## Test Results

```
$ pytest models/RECTOR/tests/test_planning_loop.py -v

test_generate_default_candidates      PASSED
test_trajectory_starts_at_ego_position PASSED
test_lane_following_candidates        PASSED
test_with_goal                        PASSED
test_select_closest_agents            PASSED
test_empty_agents                     PASSED
test_score_all_basic                  PASSED
test_safe_candidates_flagged          PASSED
test_goal_progress_affects_score      PASSED
test_plan_tick_basic                  PASSED
test_plan_tick_with_goal              PASSED
test_timing_reported                  PASSED
test_iteration_counter                PASSED
test_hysteresis_prefers_previous      PASSED
test_planning_tick_latency            PASSED

15 passed in 4.77s
```

## Performance

| Metric | Value | Notes |
|--------|-------|-------|
| Candidate Generation | ~30ms | 16 candidates |
| Reactor Selection | ~1ms | K=3 reactors |
| Dummy Prediction | ~8ms | Without M2I |
| Scoring | ~196ms | M×K collision checks |
| Selection | <1ms | With hysteresis |
| **Total** | ~250ms | Without M2I inference |

Target with M2I: <100ms (after optimization)

## Configuration

Planning parameters from `PlanningConfig`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_candidates` | 16 | Number of ego candidates |
| `candidate_horizon` | 80 | Timesteps (8 seconds) |
| `candidate_dt` | 0.1 | Timestep duration |
| `max_reactors` | 3 | Maximum reactors to consider |

## Files Created

| File | Purpose |
|------|---------|
| `scripts/lib/planning_loop.py` | Planning loop implementation (1,259 lines) |
| `tests/test_planning_loop.py` | Unit tests (15 tests) |
| `docs/PHASE4_PLANNING_LOOP.md` | This documentation |

## Integration with M2I

Currently using dummy predictions. Full M2I integration via:

```python
from batched_adapter import BatchedM2IAdapter

adapter = BatchedM2IAdapter(config)
adapter.load_models()

planner = RECTORPlanner(config, adapter=adapter)
```

## Next Steps (Phase 5)

Phase 5 will implement the Safety Layer:

1. **Collision Detection** - OBB-based collision checking
2. **Non-Cooperative Envelope** - Conservative worst-case trajectories
3. **Reality Checks** - Kinematic feasibility validation
4. **CVaR Risk Scoring** - Proper risk quantification
