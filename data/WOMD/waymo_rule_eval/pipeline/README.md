# Pipeline Module

The `pipeline/` module orchestrates rule evaluation, coordinating the execution of all registered rules on scenario data.

---

## Table of Contents

1. [Overview](#overview)
2. [Philosophy](#philosophy)
3. [Core Components](#core-components)
4. [Rule Executor](#rule-executor)
5. [Window Scheduler](#window-scheduler)
6. [Result Data Classes](#result-data-classes)
7. [Usage Examples](#usage-examples)
8. [Advanced Configuration](#advanced-configuration)

---

## Overview

The pipeline module serves as the orchestration layer between data sources and rule implementations. It:

1. **Loads Rules**: Registers all rules from the registry
2. **Executes Evaluation**: Runs each rule's two-phase evaluation
3. **Aggregates Results**: Combines individual rule results into scenario-level summaries
4. **Handles Errors**: Gracefully manages rule failures without stopping the pipeline

---

## Philosophy

### Single Responsibility

The pipeline has one job: orchestrate rule execution. It does not:
- Parse data files (that's `data_access/`)
- Implement rule logic (that's `rules/`)
- Write output files (that's the caller's responsibility)

### Fail-Soft Behavior

When a rule throws an exception:
1. Log the error
2. Mark that rule as "errored" in results
3. Continue with remaining rules
4. Return partial results

This ensures one buggy rule doesn't crash the entire evaluation.

### Streaming Design

The executor processes one scenario at a time, returning results immediately. This enables:
- Memory-efficient processing of large datasets
- Real-time feedback on progress
- Easy parallelization at the scenario level

---

## Core Components

```
pipeline/
├── rule_executor.py      # Main orchestration logic
└── window_scheduler.py   # Temporal windowing (optional)
```

### Data Flow

```
┌─────────────────────┐
│  ScenarioContext    │
└─────────────────────┘
          │
          │  RuleExecutor.evaluate()
          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    For Each Registered Rule                      │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ 1. Applicability Detection                               │    │
│  │    detector.detect(ctx) → ApplicabilityResult            │    │
│  │                                                          │    │
│  │ 2. If applicable:                                        │    │
│  │    Violation Evaluation                                  │    │
│  │    evaluator.evaluate(ctx, applicability)                │    │
│  │    → ViolationResult                                     │    │
│  │                                                          │    │
│  │ 3. Combine into PipelineRuleResult                       │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────┐
│   ScenarioResult    │
│   (all rule results)│
└─────────────────────┘
```

---

## Rule Executor

The `RuleExecutor` class is the main entry point for rule evaluation.

### Class Definition

```python
class RuleExecutor:
    """
    Executes a set of rules on scenarios.

    Usage:
        executor = RuleExecutor()
        executor.register_all_rules()
        result = executor.evaluate(scenario_context)
    """
```

### Initialization

```python
from waymo_rule_eval.pipeline.rule_executor import RuleExecutor

# Create executor
executor = RuleExecutor()

# Register all rules from the registry
executor.register_all_rules()

# Or register specific rules
from waymo_rule_eval.rules.l10_collision import (
    CollisionApplicability, CollisionViolation
)
executor.register_rule(
    CollisionApplicability(),
    CollisionViolation(),
    description="Detect collisions"
)
```

### Methods

#### `register_rule(detector, evaluator, description="")`

Register a single rule with its detector and evaluator.

```python
executor.register_rule(
    detector=MyRuleApplicability(),
    evaluator=MyRuleViolation(),
    description="My custom rule for XYZ detection"
)
```

#### `register_all_rules()`

Register all rules from the centralized registry.

```python
executor.register_all_rules()
print(f"Registered {len(executor.rules)} rules")
```

#### `evaluate(ctx: ScenarioContext) -> ScenarioResult`

Evaluate all registered rules on a scenario.

```python
from waymo_rule_eval.data_access.adapter_motion_scenario import MotionScenarioReader

reader = MotionScenarioReader()
executor = RuleExecutor()
executor.register_all_rules()

for scenario in reader.read_tfrecord("path/to/file.tfrecord"):
    result = executor.evaluate(scenario)
    print(f"Scenario {result.scenario_id}: {result.n_violations} violations")
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `rules` | List[RuleDefinition] | List of registered rule definitions |

---

## Window Scheduler

The `WindowScheduler` divides long scenarios into overlapping evaluation windows.

### Why Windowing?

Waymo scenarios are 9.1 seconds (91 frames). Some use cases require:
- Sliding-window analysis for temporal patterns
- Shorter evaluation segments for real-time simulation
- Overlap to catch violations spanning window boundaries

### Usage

```python
from waymo_rule_eval.pipeline.window_scheduler import WindowScheduler

# Create scheduler with 8-second windows, 2-second stride
scheduler = WindowScheduler(
    window_size_s=8.0,
    stride_s=2.0,
    dt=0.1  # Waymo timestep
)

# Get window boundaries
windows = scheduler.get_windows(n_frames=91)
# Returns: [(0, 80), (20, 100), ...] as (start_frame, end_frame) tuples

for start, end in windows:
    # Create windowed context
    windowed_ctx = scheduler.slice_context(scenario, start, end)

    # Evaluate on window
    result = executor.evaluate(windowed_ctx)
```

### Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `window_size_s` | 8.0 | Window duration in seconds |
| `stride_s` | 2.0 | Stride between windows in seconds |
| `dt` | 0.1 | Time step (seconds per frame) |

### Window Diagram

```
Full Scenario: |-------- 91 frames (9.1 seconds) --------|

Window 1:      |======== 80 frames (8.0 sec) ========|
Window 2:          |======== 80 frames ========|
Window 3:              |======== 80 frames ========|

Stride:        |==20==|
               (2.0 seconds)
```

---

## Result Data Classes

### RuleDefinition

Internal representation of a registered rule.

```python
@dataclass
class RuleDefinition:
    rule_id: str                      # e.g., "L10.R1"
    level: int                        # e.g., 10
    name: str                         # e.g., "Collision"
    description: str                  # Optional description
    detector: ApplicabilityDetector   # Applicability logic
    evaluator: ViolationEvaluator     # Violation logic
```

### PipelineRuleResult

Result from evaluating a single rule.

```python
@dataclass
class PipelineRuleResult:
    rule_id: str                              # Rule identifier
    level: int                                # Rule level
    name: str                                 # Rule name
    applies: bool                             # Did rule apply?
    applicability: Optional[ApplicabilityResult]  # Applicability details
    violation: Optional[ViolationResult]      # Violation details
    error: Optional[str]                      # Error message if failed
```

**Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `has_violation` | bool | True if violation occurred |
| `severity` | float | Violation severity (0 if none) |

**Methods:**

```python
# Convert to dictionary for serialization
rule_result.to_dict()
# Returns:
# {
#     "rule_id": "L10.R1",
#     "level": 10,
#     "name": "Collision",
#     "applies": True,
#     "has_violation": True,
#     "severity": 0.85,
#     "applicability": {...},
#     "violation": {...}
# }
```

### ScenarioResult

Complete result from evaluating all rules on a scenario.

```python
@dataclass
class ScenarioResult:
    scenario_id: str                     # Scenario identifier
    n_frames: int                        # Number of frames
    n_agents: int                        # Number of agents
    rule_results: List[PipelineRuleResult]  # Results per rule
```

**Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `n_applicable` | int | Count of applicable rules |
| `n_violations` | int | Count of violated rules |
| `total_severity` | float | Sum of all severities |
| `max_severity_rule` | PipelineRuleResult | Rule with highest severity |

**Methods:**

```python
# Convert to dictionary
scenario_result.to_dict()
# Returns:
# {
#     "scenario_id": "abc123",
#     "n_frames": 91,
#     "n_agents": 25,
#     "n_applicable": 18,
#     "n_violations": 3,
#     "total_severity": 1.45,
#     "rules": [...]
# }
```

---

## Usage Examples

### Basic Evaluation

```python
from waymo_rule_eval.pipeline.rule_executor import RuleExecutor
from waymo_rule_eval.data_access.adapter_motion_scenario import MotionScenarioReader

# Initialize
reader = MotionScenarioReader()
executor = RuleExecutor()
executor.register_all_rules()

# Process file
results = []
for scenario in reader.read_tfrecord("validation.tfrecord"):
    result = executor.evaluate(scenario)
    results.append(result.to_dict())

# Analyze results
total_violations = sum(r["n_violations"] for r in results)
print(f"Total violations: {total_violations}")
```

### Selective Rule Evaluation

```python
from waymo_rule_eval.pipeline.rule_executor import RuleExecutor
from waymo_rule_eval.rules.l10_collision import (
    CollisionApplicability, CollisionViolation
)
from waymo_rule_eval.rules.l0_safe_longitudinal_distance import (
    SafeLongitudinalDistanceApplicability, SafeLongitudinalDistanceViolation
)

# Register only specific rules
executor = RuleExecutor()
executor.register_rule(
    CollisionApplicability(),
    CollisionViolation(),
    description="Collision detection"
)
executor.register_rule(
    SafeLongitudinalDistanceApplicability(),
    SafeLongitudinalDistanceViolation(),
    description="Following distance"
)

# Evaluate with reduced rule set
result = executor.evaluate(scenario)
```

### Windowed Evaluation

```python
from waymo_rule_eval.pipeline.rule_executor import RuleExecutor
from waymo_rule_eval.pipeline.window_scheduler import WindowScheduler

executor = RuleExecutor()
executor.register_all_rules()

scheduler = WindowScheduler(window_size_s=5.0, stride_s=1.0)

# Get windows
windows = scheduler.get_windows(n_frames=91)

for i, (start, end) in enumerate(windows):
    # Create windowed context
    windowed = scheduler.slice_context(scenario, start, end)

    # Evaluate
    result = executor.evaluate(windowed)

    print(f"Window {i} (frames {start}-{end}): {result.n_violations} violations")
```

### Error Handling

```python
from waymo_rule_eval.pipeline.rule_executor import RuleExecutor

executor = RuleExecutor()
executor.register_all_rules()

result = executor.evaluate(scenario)

# Check for rule errors
for rule_result in result.rule_results:
    if rule_result.error:
        print(f"Rule {rule_result.rule_id} failed: {rule_result.error}")
    elif rule_result.has_violation:
        print(f"Rule {rule_result.rule_id} violated (severity: {rule_result.severity})")
```

### Batch Processing with Progress

```python
from waymo_rule_eval.pipeline.rule_executor import RuleExecutor
from waymo_rule_eval.data_access.adapter_motion_scenario import MotionScenarioReader
from tqdm import tqdm
import glob

executor = RuleExecutor()
executor.register_all_rules()

reader = MotionScenarioReader()

# Find all TFRecord files
files = glob.glob("/data/scenarios/*.tfrecord*")

# Process with progress bar
all_results = []
for filepath in tqdm(files, desc="Processing files"):
    for scenario in reader.read_tfrecord(filepath):
        result = executor.evaluate(scenario)
        all_results.append(result.to_dict())

print(f"Processed {len(all_results)} scenarios")
```

---

## Advanced Configuration

### Custom Rule Ordering

By default, rules execute in registration order. To customize:

```python
from waymo_rule_eval.rules.registry import all_rules

executor = RuleExecutor()

# Get rules and sort by level (lowest first)
rules = sorted(all_rules(), key=lambda r: r.detector.level)

for rule_entry in rules:
    executor.register_rule(
        rule_entry.detector,
        rule_entry.evaluator,
        description=""
    )
```

### Parallel Scenario Processing

The executor is stateless, so scenarios can be processed in parallel:

```python
from concurrent.futures import ProcessPoolExecutor
from waymo_rule_eval.pipeline.rule_executor import RuleExecutor
from waymo_rule_eval.data_access.adapter_motion_scenario import MotionScenarioReader

def process_scenario(scenario_data):
    """Process a single scenario (runs in worker process)."""
    executor = RuleExecutor()
    executor.register_all_rules()

    reader = MotionScenarioReader()
    scenario = reader.parse_scenario(scenario_data)

    return executor.evaluate(scenario).to_dict()

# Load scenarios
reader = MotionScenarioReader()
scenarios = list(reader.read_tfrecord("validation.tfrecord"))

# Process in parallel
with ProcessPoolExecutor(max_workers=4) as pool:
    results = list(pool.map(process_scenario, scenarios))
```

### Logging Configuration

The pipeline uses Python's logging module:

```python
import logging

# Enable debug logging for pipeline
logging.getLogger("waymo_rule_eval.pipeline").setLevel(logging.DEBUG)

# This will show:
# - Which rules are being evaluated
# - Applicability decisions
# - Any errors encountered
```

---

## Performance Considerations

### Memory

- Each `ScenarioResult` is independent
- Process and discard results to avoid accumulation
- Use generators/iterators for large datasets

### Speed

- Spatial indexing reduces collision checks from O(N²) to O(N log N)
- Applicability detection is cheap; expensive work only for applicable rules
- Rule execution is single-threaded; parallelize at scenario level

### Benchmarks

| Metric | Typical Value |
|--------|--------------|
| Rules per scenario | ~28 rules evaluated |
| Time per scenario | ~50-100ms |
| Memory per scenario | ~10MB peak |
| Throughput | ~10-20 scenarios/second |

---

## See Also

- [`rule_executor.py`](./rule_executor.py) - Implementation details
- [`window_scheduler.py`](./window_scheduler.py) - Windowing logic
- [`../rules/README.md`](../rules/README.md) - Rule implementations
- [`../cli/README.md`](../cli/README.md) - Command-line interface
