# Phase 5: Safety Layer

## Summary

**Status**: ✅ COMPLETE (25/25 tests passing)
**Date**: December 2024
**Location**: `/workspace/models/RECTOR/scripts/lib/safety_layer.py`

## Overview

The Safety Layer provides comprehensive safety checking for trajectory candidates:

1. **Reality Check** - Kinematic feasibility validation
2. **Non-Cooperative Envelope** - Conservative worst-case trajectories
3. **OBB Collision Detection** - Oriented bounding box collision checking
4. **CVaR Risk Scoring** - Conditional Value at Risk computation

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      IntegratedSafetyChecker                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌───────────────────┐    ┌───────────────────┐   ┌─────────────────┐  │
│  │  RealityChecker   │    │  NonCoopEnvelope  │   │ OBBCollision    │  │
│  │                   │    │    Generator      │   │    Checker      │  │
│  │ - a_lon bounds    │    │ - Const velocity  │   │ - SAT algorithm │  │
│  │ - a_lat bounds    │    │ - Braking         │   │ - Trajectory    │  │
│  │ - jerk bounds     │    │ - Drifting        │   │   checking      │  │
│  │ - yaw rate        │    │ - Worst case      │   │                 │  │
│  │ - speed limits    │    │   extraction      │   │                 │  │
│  └───────────────────┘    └───────────────────┘   └─────────────────┘  │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │             DeterministicCVaRScorer                                │ │
│  │                                                                    │ │
│  │  - Common Random Numbers (CRN) for fair comparison                │ │
│  │  - Time-weighted collision costs                                  │ │
│  │  - Mode probability weighting                                     │ │
│  │  - CVaR = average of worst α-quantile                            │ │
│  └───────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. RealityChecker

Validates kinematic feasibility of trajectories:

```python
from safety_layer import RealityChecker, KinematicLimits

limits = KinematicLimits(
    a_lon_min=-8.0,   # Max braking
    a_lon_max=4.0,    # Max acceleration
    a_lat_max=6.0,    # Max lateral accel
    jerk_max=12.0,    # Max jerk
    yaw_rate_max=1.0, # Max yaw rate
    v_max=40.0,       # Max speed
)

checker = RealityChecker(limits)
result = checker.check_trajectory(trajectory)  # [H, 2]

if not result.is_feasible:
    print(f"Violations: {result.violations}")
```

### 2. NonCoopEnvelopeGenerator

Generates conservative worst-case trajectory envelopes:

```python
from safety_layer import NonCoopEnvelopeGenerator, NonCoopParams

params = NonCoopParams(
    a_brake_min=-4.0,     # Conservative braking
    reaction_time=0.5,     # Reaction delay
    heading_bound=0.1,     # Max heading change rate
)

generator = NonCoopEnvelopeGenerator(params)
envelope = generator.generate_envelope(agent_state, horizon=80)
# Returns [5, H, 2] with 5 envelope samples

worst_case = generator.get_worst_case(envelope, ego_trajectory)
```

**Envelope samples include:**
- Constant velocity (baseline)
- Max braking after reaction time
- Mild braking
- Slight left drift
- Slight right drift

### 3. OBBCollisionChecker

Oriented Bounding Box collision detection using Separating Axis Theorem:

```python
from safety_layer import OBBCollisionChecker

checker = OBBCollisionChecker(
    ego_length=4.5,
    ego_width=2.0,
    safety_margin=0.5,
)

# Point check
collides = checker.check_collision(ego_pos, ego_heading, other_pos, other_heading)

# Trajectory check
has_collision, collision_time, min_dist = checker.check_trajectory_collision(
    ego_traj,    # [H, 2]
    other_traj,  # [H, 2]
)
```

### 4. DeterministicCVaRScorer

CVaR (Conditional Value at Risk) scoring with deterministic sampling:

```python
from safety_layer import DeterministicCVaRScorer, CVaRConfig

config = CVaRConfig(
    alpha=0.1,              # Risk level (10% worst cases)
    num_samples=100,        # Number of samples
    seed=42,                # For reproducibility
    time_weight_decay=0.95, # Earlier collisions weighted more
)

scorer = DeterministicCVaRScorer(config)

# MUST call before scoring each tick (for CRN)
scorer.prepare_tick(num_reactors=3, num_modes=6)

risk = scorer.score_candidate(ego_traj, predictions)
```

**CVaR Properties:**
- Uses Common Random Numbers (CRN) for fair candidate comparison
- Same seed → same results (reproducible)
- Time-weighted: earlier collisions penalized more heavily
- Mode probability weighted: high-probability modes weighted more

### 5. IntegratedSafetyChecker

Combines all components for complete safety assessment:

```python
from safety_layer import IntegratedSafetyChecker

checker = IntegratedSafetyChecker()
result = checker.check_candidate(ego_candidate, predictions, agent_states)

if not result.is_safe:
    print(f"Violations: {result.violations}")
    print(f"Has collision: {result.has_collision}")
    print(f"CVaR risk: {result.cvar_risk}")
```

## Test Results

```
$ pytest models/RECTOR/tests/test_safety_layer.py -v

TestRealityChecker
  test_feasible_straight_trajectory    PASSED
  test_infeasible_teleport             PASSED
  test_infeasible_excessive_speed      PASSED
  test_infeasible_harsh_braking        PASSED
  test_short_trajectory                PASSED
  test_batch_check                     PASSED

TestNonCoopEnvelope
  test_envelope_shape                  PASSED
  test_constant_velocity_sample        PASSED
  test_braking_sample                  PASSED
  test_worst_case_extraction           PASSED

TestOBBCollision
  test_no_collision_far_apart          PASSED
  test_collision_overlapping           PASSED
  test_collision_at_angle              PASSED
  test_no_collision_perpendicular      PASSED
  test_trajectory_collision            PASSED
  test_trajectory_no_collision         PASSED

TestCVaRScorer
  test_low_risk_distant_reactor        PASSED
  test_high_risk_collision_course      PASSED
  test_deterministic_with_seed         PASSED

TestIntegratedSafetyChecker
  test_safe_candidate                  PASSED
  test_unsafe_collision                PASSED

TestBoundingBox
  test_get_corners_axis_aligned        PASSED
  test_get_corners_rotated             PASSED

TestPerformance
  test_reality_check_speed             PASSED
  test_collision_check_speed           PASSED

25 passed in 1.62s
```

## Performance

| Component | Time | Notes |
|-----------|------|-------|
| Reality Check | <1ms | Per trajectory |
| OBB Collision | <10ms | Per trajectory pair |
| Non-Coop Envelope | ~1ms | 5 samples |
| CVaR Scoring | ~5ms | 100 samples, 3 reactors |

## Kinematic Limits

| Parameter | Default | Description |
|-----------|---------|-------------|
| `a_lon_min` | -8.0 m/s² | Maximum braking |
| `a_lon_max` | 4.0 m/s² | Maximum acceleration |
| `a_lat_max` | 6.0 m/s² | Maximum lateral acceleration |
| `jerk_max` | 12.0 m/s³ | Maximum jerk |
| `yaw_rate_max` | 1.0 rad/s | Maximum yaw rate |
| `v_max` | 40.0 m/s | Maximum speed (~144 km/h) |

## Files Created

| File | Purpose |
|------|---------|
| `scripts/lib/safety_layer.py` | Safety layer implementation (950 lines) |
| `tests/test_safety_layer.py` | Unit tests (25 tests) |
| `docs/PHASE5_SAFETY_LAYER.md` | This documentation |

## Integration with Planning Loop

```python
from safety_layer import IntegratedSafetyChecker
from planning_loop import RECTORPlanner

# Create safety checker
safety_checker = IntegratedSafetyChecker()

# Use in planning
for candidate in candidates:
    result = safety_checker.check_candidate(
        candidate, predictions, agent_states
    )
    if result.is_safe:
        # Include in scoring
        pass
```

## Safety Guarantees

1. **Kinematic Feasibility**: All selected trajectories are physically realizable
2. **Collision Detection**: OBB-based detection with safety margin
3. **Worst-Case Analysis**: Non-cooperative envelopes for unmodeled behaviors
4. **Risk Quantification**: CVaR provides interpretable risk metric

## Next Steps

With all phases complete, the RECTOR planner can be integrated end-to-end:

1. Connect batched adapter to planning loop
2. Add receding horizon replanning
3. Integration testing with real scenarios
4. Performance optimization
