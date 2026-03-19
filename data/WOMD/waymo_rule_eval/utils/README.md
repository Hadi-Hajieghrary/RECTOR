# Utils Module - Constants, Logging, and Version Management

## Overview

The `utils` module provides foundational infrastructure for the Waymo Rule Evaluation package. It centralizes physical constants, tuning thresholds, version tracking, and structured logging—ensuring consistency across all rule implementations and enabling reproducible evaluations.

## Philosophy

### Design Principles

**1. Centralized constants and thresholds**
All magic numbers, thresholds, and physical constants are defined in one place (`constants.py`). This eliminates scattered literals and ensures:
- Consistent behavior across rules using the same thresholds
- Easy tuning and experimentation
- Clear documentation of parameter choices

**2. Semantic Versioning for Rules**
Each rule implementation is versioned independently (`version.py`), enabling:
- Tracking of rule behavior changes over time
- Reproducibility of historical evaluations
- A/B testing of rule modifications

**3. Contextual Logging**
The logging system (`wre_logging.py`) supports structured, context-aware logging using Python's `contextvars`. This enables:
- Correlation of log messages with specific runs and scenarios
- Clean separation of concerns (rules don't manage logging context)
- Easy integration with log aggregation systems

## Module Structure

```
utils/
├── __init__.py         # Re-exports constants
├── constants.py        # Physical constants and thresholds
├── version.py          # Version tracking for rules and engine
├── wre_logging.py      # Structured logging utilities
└── README.md           # This documentation
```

---

## Constants (`constants.py`)

### Waymo Object Types

```python
# Object type codes from Waymo Open Motion Dataset
WAYMO_TYPE_UNSET = 0
WAYMO_TYPE_VEHICLE = 1
WAYMO_TYPE_PEDESTRIAN = 2
WAYMO_TYPE_CYCLIST = 3

# Bidirectional mappings for convenience
AGENT_TYPE_MAP = {
    WAYMO_TYPE_VEHICLE: "vehicle",
    WAYMO_TYPE_PEDESTRIAN: "pedestrian",
    WAYMO_TYPE_CYCLIST: "cyclist",
}

TYPE_TO_WAYMO_MAP = {
    "vehicle": WAYMO_TYPE_VEHICLE,
    "pedestrian": WAYMO_TYPE_PEDESTRIAN,
    "cyclist": WAYMO_TYPE_CYCLIST,
}
```

### Default Dimensions

Standard bounding box dimensions when explicit sizes are unavailable:

```python
DEFAULT_DIMENSIONS = {
    "vehicle": (4.5, 1.8),      # (length, width) in meters
    "pedestrian": (0.5, 0.5),
    "cyclist": (1.8, 0.6),
}

DEFAULT_EGO_LENGTH = 4.8        # Typical autonomous vehicle
DEFAULT_EGO_WIDTH = 1.9
```

### Time Constants

```python
DEFAULT_DT_S = 0.1              # Waymo uses 10 Hz sampling
```

### Traffic Signal States

```python
# Signal state codes from Waymo
SIGNAL_STATE_UNKNOWN = 0
SIGNAL_STATE_ARROW_STOP = 1
SIGNAL_STATE_ARROW_CAUTION = 2
SIGNAL_STATE_ARROW_GO = 3
SIGNAL_STATE_STOP = 4
SIGNAL_STATE_CAUTION = 5
SIGNAL_STATE_GO = 6
SIGNAL_STATE_FLASHING_STOP = 7
SIGNAL_STATE_FLASHING_CAUTION = 8

# Semantic groupings for rule logic
RED_SIGNAL_STATES = frozenset({
    SIGNAL_STATE_ARROW_STOP,
    SIGNAL_STATE_STOP,
    SIGNAL_STATE_FLASHING_STOP,
})

YELLOW_SIGNAL_STATES = frozenset({
    SIGNAL_STATE_ARROW_CAUTION,
    SIGNAL_STATE_CAUTION,
    SIGNAL_STATE_FLASHING_CAUTION,
})

GREEN_SIGNAL_STATES = frozenset({
    SIGNAL_STATE_ARROW_GO,
    SIGNAL_STATE_GO,
})
```

### Rule-Specific Thresholds

#### L0: Critical Safety Rules

```python
# L0.R2: Safe Longitudinal Distance
SAFE_FOLLOWING_TIME_GAP_S = 2.0           # 2-second rule
SAFE_FOLLOWING_DETECTION_RANGE_M = 60.0   # Detection range

# L0.R3: Safe Lateral Clearance
MIN_LATERAL_CLEARANCE_VEHICLE_M = 0.5
MIN_LATERAL_CLEARANCE_CYCLIST_M = 1.0
MIN_LATERAL_CLEARANCE_PEDESTRIAN_M = 1.5
LATERAL_DETECTION_RANGE_M = 30.0

# L0.R4: Crosswalk Occupancy
CROSSWALK_DETECTION_BUFFER_M = 5.0
MIN_PEDESTRIAN_SPEED_MPS = 0.3
CROSSWALK_MAX_DISTANCE_M = 50.0
```

#### L1: Comfort/Smoothness Rules

```python
# L1.R1-R2: Acceleration/Braking
L1_COMFORTABLE_ACCEL_MPS2 = 2.0           # Comfort threshold
L1_CRITICAL_ACCEL_MPS2 = 3.0              # Critical threshold
COMFORT_DECELERATION_LIMIT = 2.5          # m/s²
EMERGENCY_DECELERATION_LIMIT = 4.0        # Hard braking

# L1.R3: Steering Smoothness
L1_COMFORTABLE_HEADING_RATE_DEG_S = 15.0
L1_CRITICAL_HEADING_RATE_DEG_S = 30.0
SMOOTH_LATERAL_ACCEL_LIMIT = 2.0          # m/s²

# L1.R4: Speed Consistency
L1_COMFORTABLE_SPEED_VARIANCE_MPS = 2.0
L1_SPEED_WINDOW_DURATION_S = 2.0
L1_OSCILLATION_THRESHOLD = 3             # Sign changes
```

#### L3-L4: Surface and Maneuver Rules

```python
# L3.R3: Drivable Surface
L3_LANE_WIDTH_M = 3.7
L3_DRIVABLE_BUFFER_M = 0.5
OFFROAD_THRESHOLD_M = 1.0

# L4.R3: Left Turn Gap
L4_SAFE_TTC_S = 4.0                       # Safe time-to-collision
L4_CRITICAL_TTC_S = 2.0                   # Critical threshold
LEFT_TURN_MIN_GAP_S = 4.0
LEFT_TURN_DETECTION_RANGE_M = 50.0
```

#### L5: Zone Compliance Rules

```python
SCHOOL_ZONE_SPEED_LIMIT_MPS = 11.2        # ~25 mph
CONSTRUCTION_ZONE_SPEED_LIMIT_MPS = 13.4  # ~30 mph
PARKING_ZONE_SPEED_LIMIT_MPS = 4.5        # ~10 mph
```

#### L6: Interaction Rules

```python
CYCLIST_PASSING_CLEARANCE_M = 1.5
CYCLIST_DETECTION_RANGE_M = 30.0
PEDESTRIAN_YIELD_DISTANCE_M = 5.0
INTERSECTION_YIELD_GAP_S = 3.0
LANE_CHANGE_GAP_FRONT_M = 10.0
LANE_CHANGE_GAP_REAR_M = 8.0
```

#### L7-L8: Traffic Control Rules

```python
# L7: Lane/Speed
L7_LANE_HALF_WIDTH_M = 1.8
L7_MIN_DEPARTURE_DURATION_S = 0.5
L7_SPEED_TOLERANCE_MPS = 0.5

# L8: Stop Signs/Traffic Lights
L8_STOP_SPEED_MPS = 0.3                   # Speed to consider stopped
L8_STOPLINE_EPSILON_M = 0.2
L8_WRONGWAY_ANGLE_DEG = 90.0
STOP_SIGN_STOP_DURATION_S = 1.0
```

### Normalization Factors

Used to convert raw severity values to a [0, 1] scale:

```python
RULE_NORMALIZATION = {
    "L0.R2": 20.0,    # 20 meter-seconds
    "L0.R3": 10.0,    # 10 meter-seconds
    "L0.R4": 20.0,    # 20 m²·s
    "L1.R1": 10.0,    # 10 m/s² total
    "L1.R2": 10.0,    # 10 m/s² total
    "L1.R3": 5.0,     # 5 rad total
    "L6.R2": 20.0,    # 20 meter-seconds
    "L8.R1": 1.0,     # Per violation
    "L10.R1": 2.0,    # 2m penetration
    "L10.R2": 5.0,    # 5m total deficit
}
```

---

## Version Tracking (`version.py`)

### Engine Version

```python
ENGINE_VERSION = "1.0.0"
```

### Rule Versions

Each rule is versioned independently to track implementation changes:

```python
RULE_VERSIONS = {
    # L0: Critical Safety
    "L0.R2": "1.0.0",   # SafeLongitudinalDistance
    "L0.R3": "1.0.0",   # SafeLateralClearance
    "L0.R4": "1.0.0",   # CrosswalkOccupancy

    # L1: Comfort
    "L1.R1": "1.0.0",   # SmoothAcceleration
    "L1.R2": "1.0.0",   # SmoothBraking
    "L1.R3": "1.0.0",   # SmoothSteering
    "L1.R4": "1.0.0",   # SpeedConsistency
    "L1.R5": "1.0.0",   # LaneChangeSmoothness

    # ... additional rules

    # L10: Collision
    "L10.R1": "1.0.0",  # Collision
    "L10.R2": "1.0.0",  # VRUClearance
}
```

### Version API

```python
from waymo_rule_eval.utils.version import (
    get_engine_version,
    get_rule_version,
    get_all_versions,
    VersionInfo,
)

# Get engine version
print(get_engine_version())  # "1.0.0"

# Get specific rule version
print(get_rule_version("L10.R1"))  # "1.0.0"

# Get all versions
versions = get_all_versions()

# Create version info for output records
info = VersionInfo.current()
record["versions"] = info.to_dict()
```

### Use Cases

**Reproducibility**: Include version info in output records:
```python
result = {
    "scenario_id": "...",
    "rule_id": "L10.R1",
    "rule_version": get_rule_version("L10.R1"),
    "engine_version": get_engine_version(),
    # ...
}
```

**A/B Testing**: Compare results from different rule versions:
```python
# Filter results by rule version
v1_results = [r for r in results if r["rule_version"] == "1.0.0"]
v2_results = [r for r in results if r["rule_version"] == "1.1.0"]
```

---

## Structured Logging (`wre_logging.py`)

### Logger Factory

```python
from waymo_rule_eval.utils.wre_logging import get_logger

log = get_logger(__name__)
log.info("Processing scenario")
log.error("Failed to load file", exc_info=True)
```

### Context Variables

The logging module uses Python's `contextvars` for request-scoped context:

```python
from waymo_rule_eval.utils.wre_logging import (
    set_ctx,
    reset_ctx,
    get_run_id,
    get_scenario_id,
)

# Set context for current execution scope
tokens = set_ctx(run_id="run-2024-01", scenario_id="scn_12345")
try:
    # All log messages in this scope have access to context
    log.info("Processing", extra={
        "run_id": get_run_id(),
        "scenario_id": get_scenario_id(),
    })
finally:
    # Clean up context
    reset_ctx(tokens)
```

### Why Context Variables?

Traditional approaches (thread-local storage) fail in async contexts. Context variables:
- Work correctly with asyncio and threading
- Automatically propagate to child tasks
- Support proper reset (token-based)

### Logging Format

Default format includes timestamp, logger name, level, and message:

```
2024-01-15 10:30:45,123 - waymo_rule_eval.pipeline.rule_executor - INFO - Processing 100 scenarios
```

### Integration Example

```python
def process_scenario(scenario_id: str):
    tokens = set_ctx(scenario_id=scenario_id)
    try:
        log.info(f"Starting scenario {scenario_id}")
        # ... processing ...
        log.info(f"Completed scenario {scenario_id}")
    except Exception as e:
        log.error(f"Failed: {e}", exc_info=True)
        raise
    finally:
        reset_ctx(tokens)
```

---

## Usage Patterns

### Importing Constants

```python
# Import all constants
from waymo_rule_eval.utils.constants import *

# Or import specific constants
from waymo_rule_eval.utils.constants import (
    SAFE_FOLLOWING_TIME_GAP_S,
    MIN_LATERAL_CLEARANCE_PEDESTRIAN_M,
    RED_SIGNAL_STATES,
)
```

### Using in Rules

```python
from waymo_rule_eval.rules.base import RuleBase
from waymo_rule_eval.utils.constants import (
    SAFE_FOLLOWING_TIME_GAP_S,
    SAFE_FOLLOWING_DETECTION_RANGE_M,
)

class SafeLongitudinalDistance(RuleBase):
    def evaluate(self, ctx):
        # Use centralized constants
        time_gap = SAFE_FOLLOWING_TIME_GAP_S  # 2.0 seconds
        detection_range = SAFE_FOLLOWING_DETECTION_RANGE_M  # 60.0 meters
        # ...
```

### Adding New Constants

When adding a new rule or modifying thresholds:

1. Add constants to `constants.py` in the appropriate section
2. Add rule version to `version.py`
3. Document the rationale in code comments

```python
# In constants.py
# L99.R1: New Custom Rule
L99_THRESHOLD_VALUE = 5.0  # Based on [citation/justification]

# In version.py
RULE_VERSIONS["L99.R1"] = "1.0.0"
```

---

## Design Rationale

### Why Centralize Constants?

1. **Consistency**: All rules using "2-second rule" use the same value
2. **Tuning**: Change once, apply everywhere
3. **Documentation**: Constants serve as self-documenting configuration
4. **Testing**: Easy to create test fixtures with known thresholds

### Why Separate Versions?

1. **Granularity**: Track individual rule changes without engine version bump
2. **Audit Trail**: Know exactly which logic produced historical results
3. **Rollback**: Identify when behavior changed

### Why Context Variables for Logging?

1. **Cleaner APIs**: Rules don't need to pass run_id/scenario_id everywhere
2. **Async Safety**: Works correctly with concurrent processing
3. **Scope Safety**: Token-based reset prevents context leaks

---

## Constant Categories Reference

| Category | Purpose | Examples |
|----------|---------|----------|
| Object Types | Waymo dataset codes | `WAYMO_TYPE_VEHICLE` |
| Dimensions | Default bounding boxes | `DEFAULT_EGO_LENGTH` |
| Time | Temporal constants | `DEFAULT_DT_S` |
| Signals | Traffic light states | `RED_SIGNAL_STATES` |
| Safety | Critical thresholds | `SAFE_FOLLOWING_TIME_GAP_S` |
| Comfort | Smoothness thresholds | `L1_COMFORTABLE_ACCEL_MPS2` |
| Zone | Speed zone limits | `SCHOOL_ZONE_SPEED_LIMIT_MPS` |
| Normalization | Severity scaling | `RULE_NORMALIZATION` |

---

## See Also

- **[Rules README](../rules/README.md)** - How constants are used in rules
- **[Pipeline README](../pipeline/README.md)** - How version info is tracked
- **[Main README](../README.md)** - Package overview
