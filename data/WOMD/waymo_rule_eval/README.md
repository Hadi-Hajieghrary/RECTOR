# Waymo Rule Evaluation Framework

A Python framework for evaluating autonomous driving rule compliance on the Waymo Open Motion Dataset v1.3.0.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Philosophy & Design Principles](#philosophy--design-principles)
3. [Architecture Overview](#architecture-overview)
4. [Installation](#installation)
5. [Quick Start](#quick-start)
6. [Module Reference](#module-reference)
7. [Rule Taxonomy](#rule-taxonomy)
8. [Data Flow & Pipeline](#data-flow--pipeline)
9. [Extending the Framework](#extending-the-framework)
10. [API Reference](#api-reference)
11. [Testing](#testing)
12. [Troubleshooting](#troubleshooting)

---

## Introduction

The **Waymo Rule Evaluation Framework** (`waymo_rule_eval`) is a modular system for assessing autonomous driving behavior against a 28-rule traffic taxonomy. It processes scenarios from the Waymo Open Motion Dataset and produces structured outputs that can be used for:

- **Safety analysis**: identify collision risks, near-misses, and unsafe behaviors
- **Comfort assessment**: measure braking, acceleration, and steering smoothness
- **Regulatory compliance checks**: evaluate traffic-signal, stop-sign, and speed-limit adherence
- **ML training data generation**: produce rule-aligned supervision vectors for downstream models

### Key Features

- **28 rules** registered through `register_all_rules()` across 10 hierarchical levels (L0-L10)
- **Two-Phase Evaluation Pattern**: Applicability detection followed by violation assessment
- **Spatial Indexing**: O(log N) agent lookups using temporal-spatial indexing
- **Waymo Integration**: Native support for Waymo Motion Dataset TFRecord format
- **ML-Ready Output**: Generate augmented TFRecords with rule evaluation features
- **Comprehensive Testing**: More than 1,400 lines of fixtures and regression coverage for edge cases

---

## Philosophy & Design Principles

### 1. Separation of Concerns

The framework strictly separates three concerns:

| Concern | Module | Responsibility |
|---------|--------|----------------|
| **Data Access** | `data_access/` | Parse Waymo TFRecords into standardized `ScenarioContext` |
| **Rule Logic** | `rules/` | Implement rule-specific detection and evaluation |
| **Orchestration** | `pipeline/` | Coordinate rule execution and aggregate results |

This separation allows each component to evolve independently. For example, you can add a new rule without changing the Waymo parsing layer.

### 2. Two-Phase Evaluation Pattern

Every rule follows the same two-phase pattern:

```
┌─────────────────────────────────────────────────────────────────┐
│                    PHASE 1: Applicability                        │
│  "Does this rule apply to this scenario?"                        │
│                                                                  │
│  Examples:                                                       │
│  • Red light rule → Only applies if traffic signal present      │
│  • Following distance → Only applies if leading vehicle exists  │
│  • Pedestrian interaction → Only applies if pedestrians present │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ If applicable
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PHASE 2: Violation Evaluation                 │
│  "How severely was the rule violated?"                          │
│                                                                  │
│  Returns:                                                        │
│  • Severity score (0.0 = no violation, higher = worse)          │
│  • Detailed measurements (speeds, distances, times)             │
│  • Frame-level violation information                            │
└─────────────────────────────────────────────────────────────────┘
```

This pattern provides several benefits:
- **Efficiency**: skip expensive violation checks when a rule does not apply
- **Interpretability**: preserve the distinction between "not applicable" and "no violation"
- **Metrics clarity**: track applicability rates separately from violation rates

### 3. Immutable Context

The `ScenarioContext` object passed to rules is treated as immutable. Rules never modify the scenario data; they only read and analyze it. This ensures:
- Thread safety for parallel evaluation
- Reproducible results
- Clear separation between scenario data and rule logic

### 4. Fail-Safe Defaults

When data is missing or invalid:
- Missing map data → return "not applicable"
- Invalid kinematics → degrade gracefully with partial evaluation
- Parse errors → log a warning, skip the scenario, and continue processing

---

## Architecture Overview

```
waymo_rule_eval/
│
├── core/                    # Core data structures
│   ├── context.py           # ScenarioContext, EgoState, Agent, MapContext
│   ├── geometry.py          # Geometric operations (SAT, transforms)
│   └── temporal_spatial.py  # Spatial indexing for efficient lookups
│
├── data_access/             # Data loading
│   └── adapter_motion_scenario.py  # Waymo TFRecord → ScenarioContext
│
├── rules/                   # Rule implementations
│   ├── base.py              # ApplicabilityDetector, ViolationEvaluator base classes
│   ├── registry.py          # Central rule registration
│   └── l*_*.py              # Individual rule implementations
│
├── pipeline/                # Execution pipeline
│   ├── rule_executor.py     # RuleExecutor, ScenarioResult
│   └── window_scheduler.py  # Temporal windowing (optional)
│
├── augmentation/            # Data augmentation
│   ├── tfrecord_augmenter.py  # Add rule features to TFRecords
│   └── augment_cli.py       # Command-line augmentation tool
│
├── cli/                     # Command-line interface
│   └── wrun.py              # Main CLI entry point
│
├── utils/                   # Utilities
│   ├── constants.py         # Thresholds, Waymo constants
│   ├── wre_logging.py       # Logging configuration
│   └── version.py           # Package version
│
└── tests/                   # Test suite
    ├── conftest.py          # Comprehensive test fixtures
    └── test_*.py            # Test modules
```

### Data Flow Diagram

```
┌─────────────────────┐
│   TFRecord File     │
│  (Waymo Scenario)   │
└─────────────────────┘
          │
          │  MotionScenarioReader.read_tfrecord()
          ▼
┌─────────────────────┐
│  ScenarioContext    │
│  • ego: EgoState    │
│  • agents: [Agent]  │
│  • map: MapContext  │
│  • signals: MapSig  │
└─────────────────────┘
          │
          │  RuleExecutor.evaluate()
          ▼
┌─────────────────────┐     ┌─────────────────────┐
│   For each rule:    │────▶│ ApplicabilityDetect │
│                     │     │ detector.detect()   │
└─────────────────────┘     └─────────────────────┘
          │                           │
          │                           │ If applies=True
          │                           ▼
          │                 ┌─────────────────────┐
          │                 │ ViolationEvaluator  │
          │                 │ evaluator.evaluate()│
          │                 └─────────────────────┘
          │                           │
          ▼                           ▼
┌─────────────────────────────────────────────────┐
│                  ScenarioResult                  │
│  • scenario_id: str                              │
│  • rule_results: [PipelineRuleResult]            │
│    - rule_id, applies, severity, measurements   │
└─────────────────────────────────────────────────┘
```

---

## Installation

### Prerequisites

- Python 3.10 or higher
- TensorFlow 2.11+ (for TFRecord parsing)
- Waymo Open Dataset package

### Install Dependencies

```bash
cd /workspace/data/WOMD
pip install -r waymo_rule_eval/requirements.txt
```

> **Execution context:** Run `python -m waymo_rule_eval...` commands from `/workspace/data/WOMD`, or set `PYTHONPATH=/workspace/data/WOMD`, so the `waymo_rule_eval` package is importable.

### Verify Installation

```python
from waymo_rule_eval.pipeline.rule_executor import RuleExecutor

executor = RuleExecutor()
executor.register_all_rules()
print(f"Loaded {len(executor.rules)} rules")
# Output: Loaded 28 rules
```

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | ≥1.21 | Array operations |
| scipy | ≥1.7 | Spatial algorithms |
| shapely | ≥1.8 | Geometric operations |
| tensorflow-cpu | ≥2.11 | TFRecord parsing |
| waymo-open-dataset-tf-2-11-0 | latest | Waymo proto definitions |
| tqdm | ≥4.0 | Progress bars |

---

## Quick Start

### Basic Rule Evaluation

```python
from waymo_rule_eval.pipeline.rule_executor import RuleExecutor
from waymo_rule_eval.data_access.adapter_motion_scenario import MotionScenarioReader

# Initialize components
reader = MotionScenarioReader()
executor = RuleExecutor()
executor.register_all_rules()

# Process a TFRecord file
for scenario in reader.read_tfrecord("/path/to/scenario.tfrecord"):
    result = executor.evaluate(scenario)

    print(f"Scenario: {result.scenario_id}")
    print(f"  Applicable rules: {result.n_applicable}")
    print(f"  Violations: {result.n_violations}")

    for rule_result in result.rule_results:
        if rule_result.has_violation:
            print(f"  ⚠ {rule_result.rule_id}: severity={rule_result.severity:.2f}")
```

### Command-Line Evaluation

```bash
cd /workspace/data/WOMD

# Evaluate scenarios and output to JSONL
python -m waymo_rule_eval.cli.wrun \
    --scenario "/path/to/scenarios/*.tfrecord*" \
    --out results.jsonl \
    --window-size 8.0 \
    --stride 2.0

# Evaluate specific rules only
python -m waymo_rule_eval.cli.wrun \
    --scenario "/path/to/*.tfrecord*" \
    --out results.jsonl \
    --rules L10.R1,L0.R2,L8.R1
```

### Augment TFRecords with Rule Data

```bash
cd /workspace/data/WOMD

# Add rule evaluation features to TFRecords
python -m waymo_rule_eval.augmentation.augment_cli \
    --input /path/to/raw/scenarios/ \
    --output /path/to/augmented/scenarios/
```

---

## Module Reference

### `core/` - Core Data Structures

Contains the fundamental data classes used throughout the framework.

| Class | File | Description |
|-------|------|-------------|
| `EgoState` | context.py | Ego vehicle trajectory (x, y, yaw, speed) with validity masks |
| `Agent` | context.py | Other traffic participants (vehicles, pedestrians, cyclists) |
| `MapContext` | context.py | Static map features (lane boundaries, road edges) |
| `MapSignals` | context.py | Dynamic traffic signal states per timestep |
| `ScenarioContext` | context.py | Complete scenario container bundling all above |

**Key Design**: All trajectory arrays use the same indexing convention:
- Index 0 = first timestep (t=0)
- Index i = timestep at time i * dt (dt = 0.1s for Waymo)

See [`core/README.md`](./core/README.md) for detailed documentation.

### `rules/` - Rule Implementations

Contains **28 rules** (all registered via `register_all_rules()`), organized by level:

| Level | Category | Rules | Description |
|-------|----------|-------|-------------|
| L0 | Safety Critical | 3 | Safe longitudinal distance, lateral clearance, crosswalk occupancy |
| L1 | Comfort | 5 | Smooth braking, acceleration, steering, speed consistency, lane change smoothness |
| L3 | Surface | 1 | Drivable surface adherence |
| L4 | Maneuver | 1 | Left turn gap acceptance |
| L5 | Regulatory | 5 | Traffic signal compliance, priority, parking, school zone, construction zone |
| L6 | Interaction | 5 | Following distance, pedestrian interaction, cyclist interaction, cooperative lane change, intersection negotiation |
| L7 | Lane/Speed | 2 | Lane departure, speed limit compliance |
| L8 | Traffic Control | 4 | Red light, stop sign, crosswalk yield, wrong-way |
| L10 | Collision | 2 | Collision detection, VRU clearance |
| | **Total** | **28** | |

See [`rules/README.md`](./rules/README.md) for the complete rule catalog.

### `pipeline/` - Execution Pipeline

Orchestrates rule evaluation with support for:
- **Batch Processing**: Evaluate multiple scenarios efficiently
- **Windowed Evaluation**: Divide long scenarios into overlapping windows
- **Selective Rules**: Run only specific rules via filtering

See [`pipeline/README.md`](./pipeline/README.md) for pipeline details.

### `data_access/` - Data Loading

Adapts Waymo Motion Dataset format to the framework's internal representation.

**Key Class**: `MotionScenarioReader`
- Reads Waymo TFRecord files
- Extracts all agent types (vehicles, pedestrians, cyclists)
- Handles validity masks for missing data
- Associates traffic signals with ego's lane

See [`data_access/README.md`](./data_access/README.md) for adapter details.

### `augmentation/` - Data Augmentation

Adds rule evaluation features to TFRecords for ML training pipelines.

**Output Features** (per scenario):
- `rule/applicability`: (28,) int64 - which rules apply (1=applicable, 0=not applicable)
- `rule/violations`: (28,) int64 - which rules were violated (1=violation, 0=no violation)
- `rule/severity`: (28,) float32 - violation severity scores [0.0, 1.0+]
- `rule/ids`: (28,) bytes - rule identifier labels (fixed slots for consistent indexing)

> **Note**: All 28 rule slots are registered and active. The schema uses a fixed 28-element vector for consistent indexing.

See [`augmentation/README.md`](./augmentation/README.md) for augmentation details.

---

## Rule Taxonomy

The framework defines 28 rules organized into a hierarchical taxonomy, all registered and actively evaluated:

### Level 0: Safety Critical

These rules address fundamental safety requirements.

| Rule ID | Name | Description | Threshold |
|---------|------|-------------|-----------|
| L0.R2 | Safe Following Distance | Maintain 2-second gap to leading vehicle | 2.0s time gap |
| L0.R3 | Safe Lateral Clearance | Maintain safe lateral distance to adjacent vehicles | 1.0m clearance |
| L0.R4 | Crosswalk Occupancy | Do not block crosswalks when stopped | N/A |

### Level 1: Comfort

These rules ensure smooth, comfortable driving.

| Rule ID | Name | Description | Threshold |
|---------|------|-------------|-----------|
| L1.R1 | Smooth Acceleration | Avoid harsh acceleration | 3.0 m/s² |
| L1.R2 | Smooth Braking | Avoid harsh deceleration | -3.0 m/s² |
| L1.R3 | Smooth Steering | Avoid abrupt steering inputs | 0.5 rad/s |
| L1.R4 | Speed Consistency | Maintain consistent speed | 2.0 m/s variance |
| L1.R5 | Lane Change Smoothness | Execute lane changes smoothly | Lateral accel |

### Level 3: Surface Compliance

| Rule ID | Name | Description |
|---------|------|-------------|
| L3.R3 | Drivable Surface | Stay within drivable road boundaries |

### Level 4: Maneuver Execution

| Rule ID | Name | Description |
|---------|------|-------------|
| L4.R3 | Left Turn Gap | Accept appropriate gaps when turning left |

### Level 5: Regulatory Compliance

| Rule ID | Name | Description |
|---------|------|-------------|
| L5.R1 | Traffic Signal Compliance | Obey traffic signals |
| L5.R2 | Priority Violation | Respect right-of-way rules |
| L5.R3 | Parking Violation | Proper parking behavior |
| L5.R4 | School Zone | Comply with school zone restrictions |
| L5.R5 | Construction Zone | Comply with construction zone restrictions |

### Level 6: Interaction

| Rule ID | Name | Description |
|---------|------|-------------|
| L6.R1 | Cooperative Lane Change | Smooth lane changes with gap acceptance |
| L6.R2 | Following Distance | Maintain safe following distance |
| L6.R3 | Intersection Negotiation | Proper intersection negotiation |
| L6.R4 | Pedestrian Interaction | Safe behavior around pedestrians |
| L6.R5 | Cyclist Interaction | Safe behavior around cyclists |

### Level 7: Lane and Speed

| Rule ID | Name | Description |
|---------|------|-------------|
| L7.R3 | Lane Departure | Stay within lane boundaries |
| L7.R4 | Speed Limit | Obey posted speed limits |

### Level 8: Traffic Control

| Rule ID | Name | Description |
|---------|------|-------------|
| L8.R1 | Red Light | Stop at red lights |
| L8.R2 | Stop Sign | Complete stop at stop signs |
| L8.R3 | Crosswalk Yield | Yield to pedestrians in crosswalks |
| L8.R5 | Wrong-Way | No wrong-way driving |

### Level 10: Collision

| Rule ID | Name | Description |
|---------|------|-------------|
| L10.R1 | Collision | No collision with any agent |
| L10.R2 | VRU Clearance | Maintain clearance from vulnerable road users |

---

## Data Flow & Pipeline

### Processing Pipeline

```
1. LOAD: Read TFRecord file
   └── MotionScenarioReader.read_tfrecord(path)

2. PARSE: Convert each record to ScenarioContext
   └── reader.parse_scenario(proto)

3. EVALUATE: Run all rules on scenario
   ├── For each rule:
   │   ├── Check applicability
   │   └── If applicable, evaluate violation
   └── Aggregate into ScenarioResult

4. OUTPUT: Write results
   ├── JSONL: Structured evaluation data
   ├── CSV: Tabular summary
   └── TFRecord: Augmented scenario with features
```

### Windowed Evaluation (Optional)

For long scenarios, the framework supports sliding window evaluation:

```
Scenario: |-------- 91 frames (9.1 seconds) --------|
Window 1: |--- 80 frames ---|
Window 2:     |--- 80 frames ---|
Window 3:         |--- 80 frames ---|
          (stride = 20 frames = 2 seconds)
```

Configure via CLI:
```bash
--window-size 8.0  # 8 seconds per window
--stride 2.0       # 2 seconds between windows
```

---

## Extending the Framework

### Adding a New Rule

1. **Create rule file** in `rules/`:

```python
# rules/l99_my_rule.py
from .base import ApplicabilityDetector, ViolationEvaluator, ApplicabilityResult, ViolationResult
from ..core.context import ScenarioContext

_RULE_ID = "L99.R1"
_LEVEL = 99
_NAME = "My Custom Rule"


class MyRuleApplicability(ApplicabilityDetector):
    rule_id = _RULE_ID
    level = _LEVEL
    name = _NAME

    def detect(self, ctx: ScenarioContext) -> ApplicabilityResult:
        # Check if rule applies
        applies = len(ctx.agents) > 0
        return ApplicabilityResult(
            applies=applies,
            confidence=1.0,
            reasons=["Agents present" if applies else "No agents"],
            features={"n_agents": len(ctx.agents)}
        )


class MyRuleViolation(ViolationEvaluator):
    rule_id = _RULE_ID
    level = _LEVEL
    name = _NAME

    def evaluate(self, ctx: ScenarioContext, applicability: ApplicabilityResult) -> ViolationResult:
        # Evaluate violation severity
        severity = 0.0
        measurements = {}

        # ... your evaluation logic ...

        return ViolationResult(
            severity=severity,
            severity_normalized=min(severity / 10.0, 1.0),
            measurements=measurements,
            explanation="Rule evaluation complete"
        )
```

2. **Register in registry.py**:

```python
# In rules/registry.py, add to _rules() function:
from .l99_my_rule import MyRuleApplicability, MyRuleViolation

# Add to the returned list:
RuleEntry("L99.R1", MyRuleApplicability(), MyRuleViolation()),
```

3. **Add tests** in `tests/test_rules.py`

### Custom Data Adapter

To support non-Waymo data formats:

```python
from waymo_rule_eval.core.context import ScenarioContext, EgoState, Agent

class MyDataAdapter:
    def load(self, path: str) -> ScenarioContext:
        # Parse your data format
        # Return ScenarioContext with:
        # - ego: EgoState with x, y, yaw, speed arrays
        # - agents: List[Agent] for other participants
        # - map: MapContext (optional)
        # - signals: MapSignals (optional)
        pass
```

---

## API Reference

### RuleExecutor

```python
class RuleExecutor:
    def __init__(self):
        """Initialize empty executor."""

    def register_rule(self, detector, evaluator, description=""):
        """Register a single rule."""

    def register_all_rules(self):
        """Register all rules from registry."""

    def evaluate(self, ctx: ScenarioContext) -> ScenarioResult:
        """Evaluate all registered rules on a scenario."""

    @property
    def rules(self) -> List[RuleDefinition]:
        """List of registered rules."""
```

### ScenarioResult

```python
@dataclass
class ScenarioResult:
    scenario_id: str
    n_frames: int
    n_agents: int
    rule_results: List[PipelineRuleResult]

    @property
    def n_applicable(self) -> int: ...

    @property
    def n_violations(self) -> int: ...

    @property
    def total_severity(self) -> float: ...

    def to_dict(self) -> Dict[str, Any]: ...
```

### PipelineRuleResult

```python
@dataclass
class PipelineRuleResult:
    rule_id: str
    level: int
    name: str
    applies: bool
    applicability: Optional[ApplicabilityResult]
    violation: Optional[ViolationResult]
    error: Optional[str]

    @property
    def has_violation(self) -> bool: ...

    @property
    def severity(self) -> float: ...
```

---

## Testing

### Run All Tests

```bash
cd /workspace/data/WOMD/waymo_rule_eval
python -m pytest tests/ -v
```

### Run Specific Test Categories

```bash
# Unit tests only
python -m pytest tests/test_rules.py -v

# Integration tests
python -m pytest tests/test_integration.py -v

# Real scenario tests (requires data)
python -m pytest tests/test_real_scenarios.py -v
```

### Test Coverage

```bash
python -m pytest tests/ --cov=waymo_rule_eval --cov-report=html
```

---

## Troubleshooting

### Common Issues

**1. Waymo Dataset Not Installed**

```
ImportError: No module named 'waymo_open_dataset'
```

Solution:
```bash
pip install waymo-open-dataset-tf-2-11-0
```

**2. TensorFlow Version Mismatch**

```
AttributeError: module 'tensorflow' has no attribute 'data'
```

Solution: Ensure TensorFlow 2.x is installed:
```bash
pip install tensorflow-cpu>=2.11.0
```

**3. DataLossError on Corrupted TFRecords**

```
tensorflow.python.framework.errors_impl.DataLossError: truncated record
```

The augmentation CLI handles this gracefully by skipping corrupted files and continuing. Check the log for skipped files.

**4. Memory Issues with Large Datasets**

For processing large datasets, use streaming:
```python
for scenario in reader.read_tfrecord(path):
    result = executor.evaluate(scenario)
    # Process result immediately, don't accumulate
```

---

## Related Documentation

## Design notes (reference vs current)

This package was developed from earlier internal/reference implementations (see `ref/` in this repository).
The current `waymo_rule_eval` codebase consolidates those designs into a smaller, more maintainable surface area.

### Key architectural differences

- **Unified rule execution**: the current pipeline evaluates applicability and (when applicable) violations in a single pass.
- **Windowed evaluation support**: scenarios can be evaluated over sliding time windows (for temporal rules and aggregation).
- **Rule bundling**: each rule is represented as a `(detector, evaluator)` pair attached to a `rule_id`.

### Rule Vector: Canonical Ordering and the 28-Slot Schema

The augmentation system uses a **fixed 28-slot rule vector** defined in `augmentation/augment_cli.py`:

```python
RULE_ORDER = [
    "L0.R2", "L0.R3", "L0.R4",           # Safety (3)
    "L1.R1", "L1.R2", "L1.R3", "L1.R4", "L1.R5",  # Comfort (5)
    "L3.R3",                              # Road (1)
    "L4.R3",                              # Maneuver (1)
    "L5.R1", "L5.R2", "L5.R3", "L5.R4", "L5.R5",  # Regulatory (5)
    "L6.R1", "L6.R2", "L6.R3", "L6.R4", "L6.R5",  # Interaction (5)
    "L7.R3", "L7.R4",                     # Lane/Speed (2)
    "L8.R1", "L8.R2", "L8.R3", "L8.R5",   # Traffic Control (4)
    "L10.R1", "L10.R2",                   # Collision (2)
]  # Total: 28 rules
```

**Important**:
- The `rule/ids` feature in TFRecords is the **source of truth** for index ordering
- The canonical tier assignments are in `rules/rule_constants.py`
- Model heads and proxies should use `RULE_INDEX_MAP` from `rule_constants.py`

> **Historical Note**: Earlier dataset versions used a 22-rule subset. The current registry
> and augmentation pipeline use the full 28-rule schema. Always check `rule/ids` in your TFRecords to confirm.

### Relationship to `ref/`

- `ref/` includes earlier prototypes and reports.
- `waymo_rule_eval/` is the maintained implementation intended for use.

---

## License

This project is licensed under the MIT License. See the [LICENSE](../../LICENSE) file for details.

---

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

---

## References

- [Waymo Open Motion Dataset](https://waymo.com/open/data/motion/)
- [Waymo Motion Prediction Challenge](https://waymo.com/open/challenges/2023/motion-prediction/)
- [TensorFlow TFRecord Documentation](https://www.tensorflow.org/tutorials/load_data/tfrecord)
