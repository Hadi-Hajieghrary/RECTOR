# Data Access Module

The `data_access/` module provides adapters for loading scenario data from various formats into the framework's standardized `ScenarioContext` representation.

---

## Table of Contents

1. [Overview](#overview)
2. [Philosophy](#philosophy)
3. [Waymo Motion Dataset Adapter](#waymo-motion-dataset-adapter)
4. [Data Extraction Details](#data-extraction-details)
5. [Handling Edge Cases](#handling-edge-cases)
6. [Usage Examples](#usage-examples)
7. [Creating Custom Adapters](#creating-custom-adapters)

---

## Overview

The data access layer bridges external data formats and the framework's internal representation. Currently supported:

| Format | Adapter | Source |
|--------|---------|--------|
| Waymo Motion TFRecord | `MotionScenarioReader` | Waymo Open Motion Dataset v1.3.0 |

The adapter pattern allows the framework to support new data sources without modifying rule logic.

---

## Philosophy

### Adapter Pattern

Each data source gets its own adapter class that implements:
1. **Loading**: Read raw data files
2. **Parsing**: Extract relevant fields
3. **Converting**: Transform to `ScenarioContext`

This separation means:
- Rules work identically regardless of data source
- New data formats require only a new adapter
- Format-specific quirks are isolated in adapters

### Validity Preservation

Real sensor data has gaps. The adapter must:
- Identify invalid/missing timesteps
- Mark them in validity arrays
- Never fabricate data for invalid regions

### Fail-Safe Loading

When loading fails:
- Log detailed error with file path
- Skip corrupted records
- Continue processing remaining data
- Report skipped items at the end

---

## Waymo Motion Dataset Adapter

The `MotionScenarioReader` class reads Waymo Open Motion Dataset v1.3.0 TFRecord files.

### Class Definition

```python
class MotionScenarioReader:
    """
    Reads Waymo Motion Dataset scenarios with full agent type support.

    Usage:
        reader = MotionScenarioReader()
        for context in reader.read_tfrecord("path/to/file.tfrecord"):
            print(f"Scenario {context.scenario_id}")
    """
```

### Initialization

```python
from waymo_rule_eval.data_access.adapter_motion_scenario import MotionScenarioReader

# Default: 10 Hz (Waymo standard)
reader = MotionScenarioReader(dt=0.1)
```

### Methods

#### `read_tfrecord(path: str) -> Iterator[ScenarioContext]`

Read all scenarios from a TFRecord file.

Supported record encodings:
- Raw Waymo records (serialized `scenario_pb2.Scenario`)
- Augmented records (serialized `tf.train.Example` containing `scenario/proto`)

```python
reader = MotionScenarioReader()

for scenario in reader.read_tfrecord("/path/to/validation.tfrecord"):
    print(f"Scenario: {scenario.scenario_id}")
    print(f"  Frames: {len(scenario.ego.x)}")
    print(f"  Agents: {len(scenario.agents)}")
```

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| path | str | Path to TFRecord file |

**Returns:** Iterator of `ScenarioContext` objects

#### `parse_scenario(scenario_proto) -> ScenarioContext`

Parse a single Waymo Scenario protobuf into ScenarioContext.

```python
import tensorflow as tf
from waymo_open_dataset.protos import scenario_pb2

# Manual parsing for custom workflows
dataset = tf.data.TFRecordDataset("file.tfrecord")
for record in dataset:
    proto = scenario_pb2.Scenario()
    proto.ParseFromString(record.numpy())

    context = reader.parse_scenario(proto)
```

---

## Data Extraction Details

### Ego Vehicle

The ego vehicle is identified by `scenario.sdc_track_index` (Self-Driving Car track).

**Extracted Fields:**

| Field | Source | Notes |
|-------|--------|-------|
| x, y | `track.states[t].center_x/y` | Global coordinates (meters) |
| yaw | `track.states[t].heading` | Radians, 0 = East |
| speed | `track.states[t].velocity_x/y` | Computed magnitude |
| length | `track.states[t].length` | Actual from proto |
| width | `track.states[t].width` | Actual from proto |
| valid | `track.states[t].valid` | Boolean per timestep |

### Other Agents

All non-ego tracks are extracted as agents.

**Agent Types:**

| Waymo Type ID | Internal Type | Default Dimensions |
|---------------|---------------|-------------------|
| 1 | vehicle | 4.5m × 1.8m |
| 2 | pedestrian | 0.5m × 0.5m |
| 3 | cyclist | 1.8m × 0.6m |

**Notes:**
- Actual dimensions from proto are used when available
- Falls back to defaults if proto dimensions are zero/missing

### Map Features

The adapter extracts static map features from `scenario.map_features`.

**Lane Information:**

```python
# Ego's current lane (if determinable)
map_context.lane_id

# Lane centerline as numpy array of (x, y) points
map_context.lane_center_xy  # Shape: (N, 2)

# Stop line positions
map_context.stopline_xy  # Shape: (M, 2) or empty

# All lane data (for multi-lane scenarios)
map_context.all_lanes  # List of dicts
```

**Road Infrastructure:**

```python
# Road edges (drivable area boundaries)
map_context.road_edges  # List of dicts

# Crosswalk polygons
map_context.crosswalk_polys  # List of (K, 2) arrays

# Stop sign data
map_context.stop_signs  # List of dicts

# Speed limit array (per centerline point, if available)
map_context.speed_limit  # np.ndarray or None
map_context.speed_limit_mask  # per-point validity mask or None

# Optional zone polygons
map_context.construction_zone_xy  # (K, 2) array or None
map_context.school_zone_xy        # (K, 2) array or None
```

### Traffic Signals

Dynamic traffic signal states are extracted per timestep.

```python
# Signal state at each timestep (91 values for Waymo)
signals.signal_state  # numpy array, shape (T,)

# Lane ID used for signal association
signals.ego_lane_id  # int or None

# Per-timestep confidence values
signals.confidence  # numpy array, shape (T,)
```

**Signal State Values:**

| Value | Meaning |
|-------|---------|
| 0 | Unknown |
| 1 | Arrow Stop (red arrow) |
| 2 | Arrow Caution (yellow arrow) |
| 3 | Arrow Go (green arrow) |
| 4 | Stop (solid red) |
| 5 | Caution (solid yellow) |
| 6 | Go (solid green) |
| 7 | Flashing Stop |
| 8 | Flashing Caution |

---

## Handling Edge Cases

### Missing Validity Data

If proto doesn't have validity flags:

```python
# Default: all valid if no explicit validity
if not hasattr(state, 'valid'):
    valid = True
```

### Zero-Length Agents

Some agents may have zero dimensions in proto:

```python
# Use type-specific defaults if dimensions are zero
if length <= 0 or width <= 0:
    length, width = DEFAULT_DIMENSIONS[agent_type]
```

### Corrupted Records

TFRecord files may have corrupted entries:

```python
try:
    for record in dataset:
        # Process record
except tf.errors.DataLossError as e:
    log.warning(f"Skipping corrupted record: {e}")
    continue
```

### NaN in Kinematics

Waymo data may have NaN for velocity/acceleration at certain frames:

```python
# Compute speed, handling NaN
vx = state.velocity_x
vy = state.velocity_y
speed = np.sqrt(vx**2 + vy**2) if not np.isnan(vx) else np.nan
```

### No Ego Track

If sdc_track_index points to missing track:

```python
if ego_track is None:
    log.warning(f"No ego track found, using first track as fallback")
    ego_track = scenario.tracks[0]
```

---

## Usage Examples

### Basic Reading

```python
from waymo_rule_eval.data_access.adapter_motion_scenario import MotionScenarioReader

reader = MotionScenarioReader()

for ctx in reader.read_tfrecord("validation.tfrecord"):
    print(f"Scenario: {ctx.scenario_id}")
    print(f"  Duration: {len(ctx.ego.x) * ctx.dt:.1f}s")
    print(f"  Vehicles: {len(ctx.vehicles)}")
    print(f"  Pedestrians: {len(ctx.pedestrians)}")
    print(f"  Cyclists: {len(ctx.cyclists)}")
```

### Processing Multiple Files

```python
import glob
from waymo_rule_eval.data_access.adapter_motion_scenario import MotionScenarioReader

reader = MotionScenarioReader()

files = glob.glob("/data/waymo/motion_v_1_3_0/scenario/*.tfrecord*")

for filepath in files:
    print(f"Processing: {filepath}")

    for ctx in reader.read_tfrecord(filepath):
        # Process each scenario
        pass
```

### Accessing Agent Data

```python
reader = MotionScenarioReader()

for ctx in reader.read_tfrecord("file.tfrecord"):
    # Ego vehicle
    ego = ctx.ego
    print(f"Ego position at t=0: ({ego.x[0]:.1f}, {ego.y[0]:.1f})")
    print(f"Ego speed at t=0: {ego.speed[0]:.1f} m/s")

    # Other vehicles
    for vehicle in ctx.vehicles:
        print(f"Vehicle {vehicle.id}: {vehicle.length:.1f}m × {vehicle.width:.1f}m")

    # Pedestrians
    for ped in ctx.pedestrians:
        valid_frames = np.sum(ped.valid)
        print(f"Pedestrian {ped.id}: visible for {valid_frames} frames")
```

### Accessing Map Data

```python
reader = MotionScenarioReader()

for ctx in reader.read_tfrecord("file.tfrecord"):
    map_ctx = ctx.map

    if map_ctx is None:
        print("No map data available")
        continue

    print(f"Lane ID: {map_ctx.lane_id}")
    print(f"Speed limit data: {map_ctx.speed_limit}")
    print(f"Road edges: {len(map_ctx.road_edges)} entries")
    print(f"Crosswalks: {len(map_ctx.crosswalk_polys)}")
    print(f"Stop signs: {len(map_ctx.stop_signs)}")
```

### Accessing Signal Data

```python
reader = MotionScenarioReader()

for ctx in reader.read_tfrecord("file.tfrecord"):
    signals = ctx.signals

    if signals is None:
        print("No signal data available")
        continue

    # Check signal state at each frame
    for t in range(len(signals.signal_state)):
        state = signals.signal_state[t]
        if state == 4:  # Red
            print(f"Frame {t}: Red light")
        elif state == 6:  # Green
            print(f"Frame {t}: Green light")
```

---

## Creating Custom Adapters

To support a new data format, create an adapter class that produces `ScenarioContext` objects.

### Adapter Interface

```python
from typing import Iterator
from waymo_rule_eval.core.context import (
    ScenarioContext, EgoState, Agent, MapContext, MapSignals
)

class MyCustomAdapter:
    """Adapter for MyCustom data format."""

    def __init__(self, config: dict = None):
        """Initialize with optional configuration."""
        self.config = config or {}

    def read_file(self, path: str) -> Iterator[ScenarioContext]:
        """
        Read scenarios from a custom format file.

        Args:
            path: Path to data file

        Yields:
            ScenarioContext for each scenario in file
        """
        # Your loading logic here
        data = self._load_file(path)

        for scenario_data in data:
            yield self._convert_scenario(scenario_data)

    def _convert_scenario(self, data) -> ScenarioContext:
        """Convert raw data to ScenarioContext."""

        # Create EgoState
        ego = EgoState(
            x=data['ego_x'],  # numpy array
            y=data['ego_y'],
            yaw=data['ego_heading'],
            speed=data['ego_speed'],
            length=data.get('ego_length', 4.8),
            width=data.get('ego_width', 1.9),
            valid=data.get('ego_valid', None)
        )

        # Create Agents
        agents = []
        for agent_data in data.get('agents', []):
            agent = Agent(
                id=agent_data['id'],
                type=agent_data['type'],  # "vehicle", "pedestrian", "cyclist"
                x=agent_data['x'],
                y=agent_data['y'],
                yaw=agent_data['heading'],
                speed=agent_data['speed'],
                length=agent_data['length'],
                width=agent_data['width'],
                valid=agent_data.get('valid', None)
            )
            agents.append(agent)

        # Create MapContext (required; pass empty arrays if no map data)
        lane_center = data['map'].get('centerline', np.empty((0, 2))) if 'map' in data else np.empty((0, 2))
        map_context = MapContext(
            lane_center_xy=lane_center,
            lane_id=data['map'].get('lane_id') if 'map' in data else None,
            road_edges=data['map'].get('road_edges', []) if 'map' in data else [],
            crosswalk_polys=data['map'].get('crosswalks', []) if 'map' in data else [],
            stop_signs=data['map'].get('stop_signs', []) if 'map' in data else [],
            speed_limit=data['map'].get('speed_limit') if 'map' in data else None,
        )

        # Create MapSignals (optional)
        signals = None
        if 'signals' in data:
            signals = MapSignals(
                signal_state=data['signals']['states'],
                ego_lane_id=data['signals'].get('lane_id')
            )

        return ScenarioContext(
            scenario_id=data['id'],
            ego=ego,
            agents=agents,
            map_context=map_context,
            signals=signals,
            dt=data.get('dt', 0.1)
        )
```

### Required Fields

| Field | Required | Default |
|-------|----------|---------|
| scenario_id | Yes | - |
| ego.x, ego.y, ego.yaw, ego.speed | Yes | - |
| ego.length, ego.width | No | 4.8m, 1.9m |
| ego.valid | No | All True |
| agents | No | Empty list |
| map_context | Yes | — (pass `MapContext(lane_center_xy=np.empty((0,2)))` if no map) |
| signals | No | None |

### Using Custom Adapter

```python
from my_custom_adapter import MyCustomAdapter
from waymo_rule_eval.pipeline.rule_executor import RuleExecutor

adapter = MyCustomAdapter()
executor = RuleExecutor()
executor.register_all_rules()

for scenario in adapter.read_file("my_data.custom"):
    result = executor.evaluate(scenario)
    print(f"{scenario.scenario_id}: {result.n_violations} violations")
```

---

## Performance Tips

### Streaming vs. Loading All

For large files, use streaming:

```python
# Good: Stream scenarios one at a time
for scenario in reader.read_tfrecord("large_file.tfrecord"):
    result = executor.evaluate(scenario)
    # Process immediately, don't accumulate

# Bad: Load all into memory
scenarios = list(reader.read_tfrecord("large_file.tfrecord"))  # Memory hog!
```

### Parallel File Processing

Different files can be read in parallel:

```python
from concurrent.futures import ProcessPoolExecutor

def process_file(filepath):
    reader = MotionScenarioReader()
    executor = RuleExecutor()
    executor.register_all_rules()

    results = []
    for scenario in reader.read_tfrecord(filepath):
        result = executor.evaluate(scenario)
        results.append(result.to_dict())
    return results

files = glob.glob("*.tfrecord")
with ProcessPoolExecutor(max_workers=4) as pool:
    all_results = list(pool.map(process_file, files))
```

---

## See Also

- [`adapter_motion_scenario.py`](./adapter_motion_scenario.py) - Implementation
- [`../core/context.py`](../core/context.py) - Output data classes
- [Waymo Open Dataset Documentation](https://waymo.com/open/data/motion/)
