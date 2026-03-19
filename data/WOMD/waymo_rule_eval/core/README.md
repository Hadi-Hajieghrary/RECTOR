# Core Module

The `core/` module contains the fundamental data structures and utilities that form the backbone of the Waymo Rule Evaluation Framework.

---

## Table of Contents

1. [Overview](#overview)
2. [Philosophy](#philosophy)
3. [Data Classes](#data-classes)
4. [Geometry Utilities](#geometry-utilities)
5. [Temporal-Spatial Indexing](#temporal-spatial-indexing)
6. [Usage Examples](#usage-examples)
7. [Implementation Details](#implementation-details)

---

## Overview

The core module provides three main components:

| File | Purpose |
|------|---------|
| `context.py` | Data classes for representing scenarios (ego, agents, map, signals) |
| `geometry.py` | Geometric operations (transforms, collision detection, distances) |
| `temporal_spatial.py` | Efficient spatial indexing for O(log N) agent lookups |

---

## Philosophy

### Immutability

All data classes in this module are designed to be **effectively immutable** after construction. While Python dataclasses don't enforce immutability, the framework treats these objects as read-only. This enables:

- **Thread Safety**: Multiple rules can evaluate the same scenario concurrently
- **Reproducibility**: Re-running evaluation produces identical results
- **Debugging**: State doesn't change unexpectedly during evaluation

### Validity Tracking

Real-world sensor data often has gaps. The Waymo dataset marks certain timesteps as "invalid" when:
- An agent is occluded and cannot be tracked
- Sensor readings are unreliable
- An agent enters or exits the scene

Every trajectory in the core module carries a `valid` array indicating which timesteps have reliable data. Rules must respect these validity masks to avoid analyzing garbage data.

### Coordinate System

All positions use a **global ENU (East-North-Up) coordinate system**:
- **X-axis**: Points East
- **Y-axis**: Points North
- **Heading**: 0 = East, π/2 = North (counter-clockwise positive)

This matches the Waymo dataset's native coordinate system.

---

## Data Classes

### EgoState

Represents the ego (self-driving) vehicle's trajectory over time.

```python
@dataclass
class EgoState:
    x: np.ndarray           # X positions (meters), shape (T,)
    y: np.ndarray           # Y positions (meters), shape (T,)
    yaw: np.ndarray         # Heading angles (radians), shape (T,)
    speed: np.ndarray       # Speed (m/s), shape (T,)
    length: float           # Vehicle length (meters), default 4.8
    width: float            # Vehicle width (meters), default 1.9
    valid: np.ndarray       # Validity mask (bool), shape (T,)
```

**Key Methods:**

| Method | Description |
|--------|-------------|
| `is_valid_at(t)` | Check if ego has valid state at timestep t |
| `position_at(t)` | Get (x, y) tuple at timestep t, or None if invalid |
| `state_at(t)` | Get full state dict at timestep t |
| `get_acceleration(dt)` | Compute acceleration via finite differences |
| `get_jerk(dt)` | Compute jerk (acceleration derivative) |
| `get_yaw_rate(dt)` | Compute yaw rate (heading derivative) |
| `get_lateral_acceleration(dt)` | Compute lateral acceleration |

**Computed Properties (Cached):**

The kinematic derivatives (acceleration, jerk, yaw rate) are computed on first access and cached for efficiency:

```python
ego = EgoState(x, y, yaw, speed)

# First call computes and caches
accel = ego.get_acceleration(dt=0.1)

# Subsequent calls return cached value
accel_again = ego.get_acceleration(dt=0.1)  # No recomputation
```

### Agent

Represents other traffic participants (vehicles, pedestrians, cyclists).

```python
@dataclass
class Agent:
    id: int                 # Unique identifier
    type: str               # "vehicle", "pedestrian", or "cyclist"
    x: np.ndarray           # X positions (meters), shape (T,)
    y: np.ndarray           # Y positions (meters), shape (T,)
    yaw: np.ndarray         # Heading angles (radians), shape (T,)
    speed: np.ndarray       # Speed (m/s), shape (T,)
    length: float           # Length (meters)
    width: float            # Width (meters)
    valid: np.ndarray       # Validity mask (bool), shape (T,)
```

**Default Dimensions by Type:**

| Type | Length (m) | Width (m) |
|------|------------|-----------|
| vehicle | 4.5 | 1.8 |
| pedestrian | 0.5 | 0.5 |
| cyclist | 1.8 | 0.6 |

### MapContext

Contains static map features.

```python
@dataclass
class MapContext:
    lane_center_xy: np.ndarray                    # Lane centerline points, shape (N, 2) — required
    lane_id: Optional[int] = None                 # Ego's current lane ID
    stopline_xy: Optional[np.ndarray] = None      # Stop line positions, shape (M, 2)
    crosswalk_polys: List[np.ndarray] = []        # Crosswalk polygons (each (K, 2))
    road_edges: List[Dict[str, Any]] = []         # Road edge polylines as dicts
    stop_signs: List[Dict[str, Any]] = []         # Stop sign data as dicts
    speed_limit: Optional[np.ndarray] = None      # Per-point speed limits (m/s)
    speed_limit_mask: Optional[np.ndarray] = None # Per-point speed limit validity mask
    all_lanes: List[Dict[str, Any]] = []          # All lane data for multi-lane scenarios
    construction_zone_xy: Optional[np.ndarray] = None  # Construction zone polygon, shape (K, 2)
    school_zone_xy: Optional[np.ndarray] = None   # School zone polygon, shape (K, 2)
    boundary_types: Optional[np.ndarray] = None   # Per-point boundary type codes
```

**Key Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `has_lane_geometry` | `bool` | True if centerline has ≥ 2 points |
| `has_stoplines` | `bool` | True if stop line data present |
| `has_crosswalks` | `bool` | True if crosswalk polygons present |
| `has_road_edges` | `bool` | True if road edge data present |
| `has_stop_signs` | `bool` | True if stop sign data present |
| `has_speed_limits` | `bool` | True if speed limit array present |

**Lane Geometry:**

The lane centerline (`lane_center_xy`) is a required numpy array of shape (N, 2) representing the ego lane's polyline. Road edges and stop signs are stored as lists of dicts (not raw arrays) to support rich map metadata from the Waymo proto.

### MapSignals

Contains dynamic traffic signal states.

```python
@dataclass
class MapSignals:
    signal_state: np.ndarray              # Signal state per timestep, shape (T,)
    ego_lane_id: Optional[int] = None     # Lane ID used for signal association
    confidence: Optional[np.ndarray] = None  # Per-timestep confidence, shape (T,)
```

**Key Methods:**

| Method | Description |
|--------|-------------|
| `is_red_at(t)` | True if signal is red (stop) at timestep t |
| `is_yellow_at(t)` | True if signal is yellow (caution) at timestep t |
| `is_green_at(t)` | True if signal is green (go) at timestep t |
| `state_at(t)` | Get integer signal state at timestep t |

**Signal State Constants:**

| Value | State |
|-------|-------|
| 0 | Unknown |
| 1 | Arrow Stop |
| 2 | Arrow Caution |
| 3 | Arrow Go |
| 4 | Stop (Red) |
| 5 | Caution (Yellow) |
| 6 | Go (Green) |
| 7 | Flashing Stop |
| 8 | Flashing Caution |

### ScenarioContext

The top-level container bundling all scenario data.

```python
@dataclass
class ScenarioContext:
    scenario_id: str                        # Unique scenario identifier
    ego: EgoState                           # Ego vehicle state
    agents: List[Agent]                     # All other agents
    map_context: MapContext                 # Map features (required, not Optional)
    signals: Optional[MapSignals] = None    # Traffic signals
    dt: float = 0.1                         # Time step (seconds)
    window_start_ts: Optional[float] = None # Window start timestamp (set when slicing)
    window_size: Optional[int] = None       # Window size in frames (set when slicing)
    dataset_kind: str = "motion_scenario"   # Dataset type identifier
    trajectory_injection: Optional[Any] = None  # Injected trajectory (for planning)
```

> **Note**: `map_context` is the primary field; a `map` property alias is provided for compatibility: `ctx.map` returns `ctx.map_context`.

**Convenience Properties:**

```python
ctx = ScenarioContext(...)

# Filter agents by type
ctx.vehicles      # List of vehicle agents only
ctx.pedestrians   # List of pedestrian agents only
ctx.cyclists      # List of cyclist agents only
ctx.vrus          # List of vulnerable road users (pedestrians + cyclists)

# Metadata
ctx.n_frames      # Number of timesteps
ctx.has_signals   # True if traffic signal data is present

# Map shorthand
ctx.map           # Alias for ctx.map_context
```

---

## Geometry Utilities

The `geometry.py` module provides heading-aware geometric operations essential for rule evaluation.

### Coordinate Transforms

**Ego-Centric Coordinates:**

Transform world coordinates to ego's local frame where +X is forward and +Y is left.

```python
from waymo_rule_eval.core.geometry import compute_relative_position

# Target position in ego's frame
longitudinal, lateral = compute_relative_position(
    ego_x=10.0, ego_y=20.0, ego_yaw=1.57,
    target_x=15.0, target_y=20.0
)
# longitudinal > 0 means target is ahead
# lateral > 0 means target is to the left
```

**Ahead/Behind Detection:**

```python
from waymo_rule_eval.core.geometry import is_ahead_of_ego

ahead = is_ahead_of_ego(
    ego_x=10.0, ego_y=20.0, ego_yaw=1.57,
    target_x=15.0, target_y=20.0,
    min_longitudinal=0.0
)
```

### Oriented Bounding Boxes

For collision detection, vehicles are modeled as oriented bounding boxes (OBBs).

```python
from waymo_rule_eval.core.geometry import oriented_box_corners

# Get 4 corner points of an oriented box
corners = oriented_box_corners(
    cx=10.0, cy=20.0,    # Center position
    yaw=1.57,            # Heading angle
    length=4.5,          # Box length
    width=1.8            # Box width
)
# Returns: numpy array of shape (4, 2)
```

### SAT Collision Detection

The Separating Axis Theorem (SAT) provides exact collision detection between convex polygons.

```python
from waymo_rule_eval.core.geometry import (
    oriented_box_corners,
    get_box_separating_axes,
    sat_collision_check
)

# Get corners for two vehicles
corners_ego = oriented_box_corners(ego_x, ego_y, ego_yaw, ego_len, ego_wid)
corners_agent = oriented_box_corners(agent_x, agent_y, agent_yaw, agent_len, agent_wid)

# Get separating axes (perpendicular to box edges)
axes = get_box_separating_axes(corners_ego, corners_agent)

# Check for collision
collides, penetration_depth = sat_collision_check(corners_ego, corners_agent, axes)
```

**SAT Algorithm:**

1. For each edge of both boxes, compute the perpendicular axis
2. Project both boxes onto each axis
3. If projections don't overlap on any axis, boxes don't collide
4. If projections overlap on all axes, boxes collide
5. Penetration depth is the minimum overlap across all axes

### Bumper-to-Bumper Distance

Calculate the actual gap between vehicles, accounting for their lengths.

```python
from waymo_rule_eval.core.geometry import compute_bumper_distance

# Distance from ego's front bumper to leading vehicle's rear bumper
gap = compute_bumper_distance(
    ego_x=10.0, ego_y=20.0, ego_yaw=0.0, ego_length=4.8,
    lead_x=20.0, lead_y=20.0, lead_yaw=0.0, lead_length=4.5
)
```

### Angle Utilities

```python
from waymo_rule_eval.core.geometry import normalize_angle, angle_diff

# Normalize to [-π, π]
angle = normalize_angle(5.0)  # Returns equivalent angle in [-π, π]

# Signed difference between angles
diff = angle_diff(3.14, 0.0)  # ~π radians
```

---

## Temporal-Spatial Indexing

The `temporal_spatial.py` module provides efficient spatial queries across time.

### Problem

Naive collision checking is O(N × T) where N is agent count and T is timesteps. For scenarios with 100+ agents and 91 timesteps, this becomes expensive.

### Solution: TemporalSpatialIndex

The `TemporalSpatialIndex` pre-builds a spatial index for each timestep, enabling O(log N) range queries.

```python
from waymo_rule_eval.core.temporal_spatial import TemporalSpatialIndex

# Build index once
spatial_idx = TemporalSpatialIndex(agents=ctx.agents, n_frames=91)

# Fast range query: agents within 50m of ego at timestep t
nearby = spatial_idx.query_radius(
    t=10,
    x=ego.x[10],
    y=ego.y[10],
    radius=50.0
)
# Returns: List[Agent] within radius
```

### Implementation Details

The index uses a grid-based approach:
1. Divide space into cells (default: 50m × 50m)
2. Assign each agent to cells based on their positions at each timestep
3. For range queries, check only agents in relevant cells

**Performance:**

| Scenario | Naive | With Index |
|----------|-------|------------|
| 50 agents, 91 frames, 1 query/frame | ~4,500 distance checks | ~200 distance checks |
| 100 agents, 91 frames, 1 query/frame | ~9,100 distance checks | ~400 distance checks |

---

## Usage Examples

### Complete Rule Evaluation Flow

```python
from waymo_rule_eval.core.context import ScenarioContext, EgoState, Agent
from waymo_rule_eval.core.geometry import compute_relative_position
from waymo_rule_eval.core.temporal_spatial import TemporalSpatialIndex
import numpy as np

# 1. Create scenario context (normally from data adapter)
T = 91
ego = EgoState(
    x=np.linspace(0, 100, T),
    y=np.zeros(T),
    yaw=np.zeros(T),
    speed=np.full(T, 10.0),
    length=4.8,
    width=1.9
)

agents = [
    Agent(
        id=1,
        type="vehicle",
        x=np.linspace(20, 120, T),
        y=np.zeros(T),
        yaw=np.zeros(T),
        speed=np.full(T, 10.0),
        length=4.5,
        width=1.8,
        valid=np.ones(T, dtype=bool)
    )
]

ctx = ScenarioContext(
    scenario_id="example_001",
    ego=ego,
    agents=agents
)

# 2. Build spatial index for efficient queries
spatial_idx = TemporalSpatialIndex(agents, T)

# 3. Find leading vehicle at each timestep
for t in range(T):
    if not ego.is_valid_at(t):
        continue

    # Find nearby agents
    nearby = spatial_idx.query_radius(t, ego.x[t], ego.y[t], radius=100.0)

    # Filter to vehicles ahead
    for agent in nearby:
        if not agent.valid[t]:
            continue

        long, lat = compute_relative_position(
            ego.x[t], ego.y[t], ego.yaw[t],
            agent.x[t], agent.y[t]
        )

        if long > 0 and abs(lat) < 2.0:  # Ahead and in-lane
            print(f"t={t}: Leading vehicle {agent.id} at {long:.1f}m ahead")
            break
```

### Kinematic Analysis

```python
from waymo_rule_eval.core.context import EgoState
import numpy as np

# Create ego with varying speed (braking scenario)
T = 50
speed = np.linspace(20.0, 5.0, T)  # Braking from 20 to 5 m/s

ego = EgoState(
    x=np.cumsum(speed * 0.1),  # Integrate speed for position
    y=np.zeros(T),
    yaw=np.zeros(T),
    speed=speed
)

# Analyze comfort metrics
dt = 0.1
accel = ego.get_acceleration(dt)
jerk = ego.get_jerk(dt)

print(f"Max deceleration: {np.min(accel):.2f} m/s²")
print(f"Max jerk: {np.max(np.abs(jerk)):.2f} m/s³")

# Check comfort thresholds
if np.min(accel) < -3.0:
    print("⚠ Harsh braking detected!")
if np.max(np.abs(jerk)) > 2.0:
    print("⚠ Uncomfortable jerk detected!")
```

---

## Implementation Details

### Memory Layout

Arrays are stored in row-major (C) order for cache-friendly access:

```python
# Time is the first axis for per-timestep iteration
ego.x  # Shape: (T,) - iterating over time is fast

# Positions can also be viewed as (T, 2) for vectorized geometry
positions = np.column_stack([ego.x, ego.y])  # Shape: (T, 2)
```

### NaN Handling

Invalid timesteps are marked with NaN in position arrays and False in validity masks:

```python
# Check validity before using data
if ego.is_valid_at(t):
    x, y = ego.position_at(t)
else:
    # Skip this timestep
    pass

# Vectorized validity filtering
valid_speeds = ego.speed[ego.valid]
```

### Finite Difference Accuracy

Kinematic derivatives use forward differences:

```
acceleration[t] = (speed[t] - speed[t-1]) / dt
```

This introduces a one-timestep delay and some noise. For smoother derivatives, consider applying a Savitzky-Golay filter before differentiation.

---

## See Also

- [`rules/base.py`](../rules/base.py) - How rules use these data structures
- [`data_access/adapter_motion_scenario.py`](../data_access/adapter_motion_scenario.py) - How contexts are created from Waymo data
