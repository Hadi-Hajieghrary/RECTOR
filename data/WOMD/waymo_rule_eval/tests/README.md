# Tests Module - Testing Framework and Guidelines

## Overview

The `tests` module provides a comprehensive test suite for the Waymo Rule Evaluation package. It includes unit tests, integration tests, and realistic scenario fixtures designed to validate rule correctness across diverse driving situations.

## Philosophy

### Testing Principles

**1. Scenario-Driven Testing**
Tests are built around realistic driving scenarios rather than isolated function calls. Each test fixture represents a complete driving situation (collision, lane change, pedestrian crossing) that can trigger specific rules:

```python
def test_collision_detection(self, fixture_factory):
    scenario = fixture_factory.create_rear_end_collision_scenario()
    result = executor.evaluate(scenario)
    assert any(r.rule_id == "L10.R1" and r.severity > 0 for r in result.rule_results)
```

**2. Two-Phase Validation**
Every test validates both the applicability phase (does the rule apply?) and the evaluation phase (what is the severity?):

```python
# Phase 1: Applicability
applicable_rules = [r for r in results if r.applies]

# Phase 2: Severity
violations = [r for r in results if r.severity > 0]
```

**3. Parameterized Fixtures**
Test fixtures are parameterized to cover boundary conditions, edge cases, and normal operation:

```python
@pytest.fixture(params=[
    ("normal_driving", 0.0),      # No violation expected
    ("tailgating", 0.5),          # Moderate violation
    ("collision", 1.0),           # Maximum severity
])
def driving_scenario(request, fixture_factory):
    scenario_type, expected_severity = request.param
    return fixture_factory.create(scenario_type), expected_severity
```

**4. Physics-Based Realism**
Trajectory fixtures use realistic kinematic constraints (acceleration limits, steering rates) to ensure rules behave correctly with real-world driving data.

## Module Structure

```
tests/
├── __init__.py             # Test package initialization
├── conftest.py             # Pytest fixtures (scenario builders)
├── test_context.py         # Core context dataclass tests
├── test_geometry.py        # Geometry utility tests
├── test_pipeline.py        # Pipeline execution tests
├── test_rules.py           # Individual rule tests
├── test_integration.py     # End-to-end integration tests
├── test_real_scenarios.py  # Tests with real Waymo data
└── README.md               # This documentation
```

## Running Tests

### Basic Execution

```bash
# Run all tests
cd /workspace/data
pytest waymo_rule_eval/tests/ -v

# Run with coverage
pytest waymo_rule_eval/tests/ --cov=waymo_rule_eval --cov-report=html

# Run specific test file
pytest waymo_rule_eval/tests/test_rules.py -v

# Run specific test class
pytest waymo_rule_eval/tests/test_rules.py::TestRuleExecutorIntegration -v

# Run specific test
pytest waymo_rule_eval/tests/test_rules.py::TestRuleExecutorIntegration::test_executor_initialization -v
```

### Filtering Tests

```bash
# Run only integration tests
pytest waymo_rule_eval/tests/ -k "integration" -v

# Run only collision-related tests
pytest waymo_rule_eval/tests/ -k "collision" -v

# Skip slow tests
pytest waymo_rule_eval/tests/ -m "not slow" -v

# Run only tests for specific rule level
pytest waymo_rule_eval/tests/ -k "L10" -v
```

### Parallel Execution

```bash
# Run tests in parallel (requires pytest-xdist)
pytest waymo_rule_eval/tests/ -n auto -v
```

## Test Files

### `conftest.py` - Fixtures and Builders

The fixture module provides a comprehensive `TrajectoryBuilder` class and scenario factory for creating test data:

```python
@dataclass
class TrajectoryBuilder:
    """Builder for creating realistic vehicle trajectories."""
    n_frames: int = 91  # 9.1 seconds at 10 Hz
    dt: float = 0.1

    def straight_line(self, start_x, start_y, heading, speed, acceleration=0.0):
        """Create straight-line trajectory with optional acceleration."""
        ...

    def lane_change(self, start_x, start_y, heading, speed, lateral_offset=3.5):
        """Create lane change trajectory with smooth lateral motion."""
        ...

    def stop_at_position(self, start_x, start_y, heading, initial_speed, stop_x, stop_y):
        """Create trajectory that stops at a specific position."""
        ...
```

**Available Scenario Fixtures**:

| Fixture Method | Description | Expected Rules |
|----------------|-------------|----------------|
| `create_normal_driving_scenario()` | Steady-state highway driving | No violations |
| `create_rear_end_collision_scenario()` | Ego collides with lead vehicle | L10.R1 |
| `create_pedestrian_crossing_scenario()` | Pedestrian in crosswalk | L0.R4, L6.R4 |
| `create_harsh_braking_scenario()` | Emergency stop maneuver | L1.R1, L1.R2 |
| `create_speeding_scenario()` | Exceeding speed limit | L7.R4 |
| `create_lane_departure_scenario()` | Drifting out of lane | L7.R3 |
| `create_red_light_violation_scenario()` | Running a red light | L8.R1 |
| `create_stop_sign_run_through_scenario()` | Rolling through stop sign | L8.R2 |
| `create_tailgating_scenario()` | Following too closely | L0.R2, L6.R2 |
| `create_cyclist_close_pass_scenario()` | Passing cyclist unsafely | L6.R5, L10.R2 |

### `test_context.py` - Core Data Structure Tests

Tests for the core context dataclasses:

```python
class TestEgoState:
    """Tests for EgoState dataclass."""

    def test_ego_state_creation(self):
        """EgoState can be created with numpy arrays."""
        ...

    def test_speed_computation(self):
        """Speed is correctly computed from velocity components."""
        ...

class TestScenarioContext:
    """Tests for ScenarioContext dataclass."""

    def test_agent_lookup(self):
        """Agents can be looked up by ID."""
        ...

    def test_frame_slicing(self):
        """Context can be sliced to specific frame range."""
        ...
```

### `test_geometry.py` - Geometry Utility Tests

Tests for geometric computation utilities:

```python
class TestPolygonIntersection:
    """Tests for polygon overlap detection."""

    def test_overlapping_rectangles(self):
        """Correctly detects overlapping rectangles."""
        ...

    def test_non_overlapping_rectangles(self):
        """Correctly identifies non-overlapping rectangles."""
        ...

class TestDistanceCalculations:
    """Tests for distance computation."""

    def test_euclidean_distance(self):
        """Euclidean distance computed correctly."""
        ...

    def test_oriented_bounding_box_distance(self):
        """OBB distance computed correctly."""
        ...
```

### `test_pipeline.py` - Pipeline Execution Tests

Tests for the rule execution pipeline:

```python
class TestRuleExecutor:
    """Tests for RuleExecutor."""

    def test_rule_registration(self):
        """Rules can be registered and executed."""
        ...

    def test_rule_filtering(self):
        """Specific rules can be selected for execution."""
        ...

class TestWindowedExecutor:
    """Tests for windowed evaluation."""

    def test_window_generation(self):
        """Windows are correctly generated from scenarios."""
        ...

    def test_overlapping_windows(self):
        """Overlapping windows share common frames."""
        ...
```

### `test_rules.py` - Individual Rule Tests

Comprehensive tests for each rule implementation:

```python
class TestCollisionRule:
    """Tests for L10.R1 Collision detection."""

    def test_no_collision_normal_driving(self, fixture_factory):
        """No collision detected in normal driving."""
        scenario = fixture_factory.create_normal_driving_scenario()
        result = executor.evaluate_rule("L10.R1", scenario)
        assert result.severity == 0.0

    def test_collision_detected(self, fixture_factory):
        """Collision correctly detected."""
        scenario = fixture_factory.create_rear_end_collision_scenario()
        result = executor.evaluate_rule("L10.R1", scenario)
        assert result.severity > 0.0

class TestSmoothBrakingRule:
    """Tests for L1.R1/L1.R2 Smooth braking/acceleration."""

    def test_comfortable_braking(self, fixture_factory):
        """Comfortable braking within limits."""
        ...

    def test_harsh_braking_violation(self, fixture_factory):
        """Harsh braking correctly flagged."""
        ...
```

### `test_integration.py` - End-to-End Tests

Full pipeline integration tests:

```python
class TestEndToEndPipeline:
    """End-to-end pipeline integration tests."""

    def test_full_pipeline_normal_scenario(self, fixture_factory):
        """Complete pipeline execution on normal driving scenario."""
        ...

    def test_full_pipeline_collision_scenario(self, fixture_factory):
        """Complete pipeline execution on collision scenario."""
        ...

class TestScenarioBatching:
    """Tests for batch scenario processing."""

    def test_process_multiple_scenarios(self, fixture_factory):
        """Process multiple scenarios using batch evaluation."""
        ...

class TestResultSerialization:
    """Tests for result output."""

    def test_jsonl_serialization(self):
        """Results can be serialized to JSONL."""
        ...
```

### `test_real_scenarios.py` - Real Data Tests

Tests using actual Waymo Open Motion Dataset data:

```python
class TestRealScenarios:
    """Tests with real Waymo Motion scenarios."""

    @pytest.mark.skipif(not WAYMO_DATA_AVAILABLE, reason="Waymo data not available")
    def test_evaluate_real_scenario(self):
        """Evaluate a real Waymo scenario."""
        ...

    @pytest.mark.slow
    def test_batch_evaluation(self):
        """Batch evaluate multiple real scenarios."""
        ...
```

## Fixture Factory

The `fixture_factory` pytest fixture provides access to scenario builders:

```python
@pytest.fixture
def fixture_factory():
    """Provide access to scenario factory."""
    return ScenarioFactory()
```

### Creating Custom Scenarios

```python
def test_custom_scenario(self, fixture_factory):
    # Use trajectory builder for custom trajectories
    builder = TrajectoryBuilder(n_frames=100, dt=0.1)

    # Create ego trajectory
    ego_traj = builder.straight_line(
        start_x=0, start_y=0,
        heading=0.0, speed=20.0,
        acceleration=-2.0  # Braking
    )

    # Build scenario context
    scenario = fixture_factory.build_scenario(
        ego=ego_traj,
        agents=[...],
        map_elements=[...],
    )

    # Evaluate
    result = executor.evaluate(scenario)
```

## Test Categories

### Unit Tests

Test individual functions and methods in isolation:

```python
def test_polygon_area():
    """Test polygon area calculation."""
    polygon = [(0, 0), (1, 0), (1, 1), (0, 1)]
    assert compute_area(polygon) == pytest.approx(1.0)
```

### Component Tests

Test integrated components:

```python
class TestRuleExecutor:
    def test_rule_registration(self):
        """Rules can be registered."""
        executor = RuleExecutor()
        executor.register_all_rules()
        assert len(executor.rules) > 0
```

### Integration Tests

Test complete workflows:

```python
class TestEndToEndPipeline:
    def test_full_pipeline(self, fixture_factory):
        """Complete evaluation pipeline."""
        scenario = fixture_factory.create_normal_driving_scenario()
        executor = RuleExecutor()
        executor.register_all_rules()
        result = executor.evaluate(scenario)
        assert isinstance(result, ScenarioResult)
```

### Regression Tests

Ensure fixed bugs don't recur:

```python
def test_edge_case_zero_speed():
    """Regression: division by zero when speed is 0."""
    # This previously caused a crash
    scenario = fixture_factory.create_stationary_scenario()
    result = executor.evaluate(scenario)
    # Should complete without error
```

## Writing New Tests

### Test Structure

```python
class TestNewRule:
    """Tests for new rule implementation."""

    def test_rule_applicability(self, fixture_factory):
        """Test when rule applies."""
        scenario = fixture_factory.create_applicable_scenario()
        executor = RuleExecutor()
        executor.register_rule(NewRule())

        result = executor.evaluate(scenario)
        rule_result = next(r for r in result.rule_results if r.rule_id == "LX.RY")

        assert rule_result.applies

    def test_rule_violation_severity(self, fixture_factory):
        """Test severity calculation."""
        scenario = fixture_factory.create_violation_scenario()
        executor = RuleExecutor()
        executor.register_rule(NewRule())

        result = executor.evaluate(scenario)
        rule_result = next(r for r in result.rule_results if r.rule_id == "LX.RY")

        assert rule_result.severity > 0.0
        assert rule_result.severity <= 1.0  # Normalized

    def test_rule_no_violation(self, fixture_factory):
        """Test no violation in normal case."""
        scenario = fixture_factory.create_normal_driving_scenario()
        executor = RuleExecutor()
        executor.register_rule(NewRule())

        result = executor.evaluate(scenario)
        rule_result = next(r for r in result.rule_results if r.rule_id == "LX.RY")

        assert rule_result.severity == 0.0
```

### Creating New Fixtures

```python
# In conftest.py
@pytest.fixture
def new_scenario_type(fixture_factory):
    """Fixture for new scenario type."""
    builder = TrajectoryBuilder()

    # Build ego trajectory
    ego = builder.straight_line(0, 0, 0, 15.0)

    # Build agent trajectories
    agent = builder.straight_line(30, 3.5, 0, 10.0)

    return fixture_factory.build_scenario(
        ego=ego,
        agents=[agent],
        scenario_id="new_scenario_001"
    )
```

## Best Practices

### 1. Use Descriptive Test Names

```python
# Good
def test_collision_detected_when_vehicles_overlap():
    ...

# Bad
def test_collision():
    ...
```

### 2. One Assertion Per Concept

```python
# Good
def test_severity_within_bounds():
    assert result.severity >= 0.0
    assert result.severity <= 1.0

# Avoid
def test_everything():
    assert result.severity >= 0.0
    assert result.severity <= 1.0
    assert result.applies == True
    assert len(result.explanation) > 0
    # ... many more
```

### 3. Test Boundary Conditions

```python
def test_following_distance_at_threshold():
    """Test exact threshold value."""
    scenario = fixture_factory.create_scenario_at_threshold()
    ...

def test_following_distance_just_below_threshold():
    """Test just below threshold (no violation)."""
    ...

def test_following_distance_just_above_threshold():
    """Test just above threshold (violation)."""
    ...
```

### 4. Document Expected Outcomes

```python
def test_harsh_braking_scenario(self, fixture_factory):
    """
    Evaluate harsh braking scenario.

    Expected outcomes:
    - L1.R1 (SmoothAcceleration): applies=True, severity > 0.5
    - L1.R2 (SmoothBraking): applies=True, severity > 0.5
    - L10.R1 (Collision): applies=True, severity = 0.0
    """
    ...
```

## Troubleshooting

### Import Errors

```
ModuleNotFoundError: No module named 'waymo_rule_eval'
```

**Solution**: Ensure the workspace root is in `sys.path`:
```python
import sys
sys.path.insert(0, "/workspace/data")
```

### Fixture Not Found

```
fixture 'fixture_factory' not found
```

**Solution**: Ensure `conftest.py` is in the tests directory and defines the fixture.

### Slow Tests

**Solution**: Mark slow tests and skip in CI:
```python
@pytest.mark.slow
def test_large_batch_processing():
    ...
```

Run without slow tests:
```bash
pytest -m "not slow"
```

### Flaky Tests

**Cause**: Tests depending on floating-point precision or timing.

**Solution**: Use `pytest.approx()` for floats:
```python
assert result.severity == pytest.approx(0.5, abs=0.01)
```

## Continuous Integration

### CI Configuration Example

```yaml
# .github/workflows/tests.yml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - run: pip install -r requirements.txt
      - run: pip install pytest pytest-cov
      - run: pytest waymo_rule_eval/tests/ -v --cov=waymo_rule_eval
```

## See Also

- **[Rules README](../rules/README.md)** - Rule implementations being tested
- **[Pipeline README](../pipeline/README.md)** - Executor implementation
- **[Main README](../README.md)** - Package overview
