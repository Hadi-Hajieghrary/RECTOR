# Rules Module

The `rules/` module contains implementations of traffic rules used by the Waymo Rule Evaluation Framework. The framework defines **28 rules** for the augmented TFRecord schema, all of which are registered and active in the `RuleExecutor`.

---

## Table of Contents

1. [Overview](#overview)
2. [Philosophy](#philosophy)
3. [Two-Phase Evaluation Pattern](#two-phase-evaluation-pattern)
4. [Base Classes](#base-classes)
5. [Rule Registry](#rule-registry)
6. [Complete Rule Catalog](#complete-rule-catalog)
7. [Implementation Guide](#implementation-guide)
8. [Rule Design Patterns](#rule-design-patterns)

---

## Overview

This module implements a comprehensive taxonomy of traffic rules organized into hierarchical levels:

| Level | Category | Rules | Description |
|-------|----------|-------|-------------|
| L0 | Safety Critical | 3 | Fundamental safety requirements |
| L1 | Comfort | 5 | Smooth driving behavior |
| L3 | Surface | 1 | Road boundary compliance |
| L4 | Maneuver | 1 | Maneuver execution quality |
| L5 | Regulatory | 5 | Traffic law compliance |
| L6 | Interaction | 5 | Multi-agent interactions |
| L7 | Lane/Speed | 2 | Lane keeping and speed limits |
| L8 | Traffic Control | 4 | Traffic signals and signs |
| L10 | Collision | 2 | Collision detection and prevention |
| **Total** | | **28** | |

> **Note**: All 28 rules are registered and active in `RuleExecutor.register_all_rules()`. Some rules (e.g., L5.R3 Parking, L5.R4 School Zone, L5.R5 Construction Zone) may have limited applicability in certain Waymo scenarios due to missing map/scenario data, in which case they return "not applicable".

---

## Philosophy

### Hierarchical Organization

Rules are organized by severity and type:

- **L0-L1**: Immediate safety and comfort (most critical)
- **L3-L4**: Road and maneuver compliance
- **L5-L8**: Regulatory and traffic control
- **L10**: Final collision assessment

Lower levels (L0) represent more fundamental requirements. A vehicle should not be evaluated on higher-level rules if it fails critical L0 rules (e.g., collision).

### Applicability First

Every rule must first determine if it's applicable before evaluating for violations. This prevents:

- False positives on irrelevant scenarios
- Wasted computation on non-applicable rules
- Confusion between "not applicable" and "no violation"

### Severity Scoring

Rules don't just return pass/fail. They return a **severity score**:

```
severity = 0.0         → No violation
severity > 0.0         → Violation occurred
severity approaching 1.0 → Severe violation
severity > 1.0         → Critical violation
```

Severity enables:
- Ranking of violations by importance
- Weighted aggregation across rules
- Threshold tuning for different applications

---

## Two-Phase Evaluation Pattern

Every rule follows a strict two-phase pattern:

```
┌─────────────────────────────────────────────────────────────────┐
│                    PHASE 1: Applicability                        │
│                    (ApplicabilityDetector)                       │
├─────────────────────────────────────────────────────────────────┤
│ Input:  ScenarioContext                                          │
│ Output: ApplicabilityResult                                      │
│         - applies: bool                                          │
│         - confidence: float (0-1)                                │
│         - reasons: List[str]                                     │
│         - features: Dict (for debugging)                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ If applies == True
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PHASE 2: Violation                            │
│                    (ViolationEvaluator)                          │
├─────────────────────────────────────────────────────────────────┤
│ Input:  ScenarioContext, ApplicabilityResult                     │
│ Output: ViolationResult                                          │
│         - severity: float (0 = no violation)                     │
│         - severity_normalized: float (0-1)                       │
│         - measurements: Dict                                     │
│         - explanation: str                                       │
│         - frame_violations: List[int]                            │
└─────────────────────────────────────────────────────────────────┘
```

### Why Two Phases?

1. **Efficiency**: Skip expensive violation computation for non-applicable rules
2. **Clarity**: Separate "does this apply?" from "is it violated?"
3. **Debugging**: Applicability reasons explain why rules were skipped
4. **Feature Reuse**: Applicability features can inform violation evaluation

---

## Base Classes

Located in `base.py`, these abstract classes define the interface for all rules.

### ApplicabilityResult

```python
@dataclass
class ApplicabilityResult:
    applies: bool                    # Does the rule apply?
    confidence: float = 1.0          # Confidence in decision (0-1)
    reasons: List[str] = []          # Human-readable explanations
    features: Dict[str, Any] = {}    # Computed features for debugging
    rule_id: Optional[str] = None
    rule_level: Optional[int] = None
    name: Optional[str] = None
```

**Example:**

```python
ApplicabilityResult(
    applies=True,
    confidence=0.95,
    reasons=["Traffic signal detected at intersection"],
    features={"signal_distance_m": 45.2, "signal_state": "red"}
)
```

### ViolationResult

```python
@dataclass
class ViolationResult:
    severity: float = 0.0            # Raw severity (0 = no violation)
    severity_normalized: float = 0.0 # Normalized to [0, 1]
    measurements: Dict[str, Any] = {} # Quantitative measurements
    explanation: str = ""            # Human-readable description
    frame_violations: List[int] = [] # Timesteps with violations
    confidence: float = 1.0          # Confidence in evaluation
    timeseries: Dict[str, Any] = {}  # Per-frame data
```

**Properties:**

| Property | Description |
|----------|-------------|
| `has_violation` | True if severity > 0 |
| `violation_count` | Number of frames with violations |

### ApplicabilityDetector

Abstract base class for applicability detection.

```python
class ApplicabilityDetector(ABC):
    rule_id: str = "UNKNOWN"
    level: int = 0
    name: str = "Unknown Rule"

    @abstractmethod
    def detect(self, ctx: ScenarioContext) -> ApplicabilityResult:
        """Determine if this rule applies to the scenario."""
        pass
```

### ViolationEvaluator

Abstract base class for violation evaluation.

```python
class ViolationEvaluator(ABC):
    rule_id: str = "UNKNOWN"
    level: int = 0
    name: str = "Unknown Rule"

    @abstractmethod
    def evaluate(
        self,
        ctx: ScenarioContext,
        applicability: ApplicabilityResult
    ) -> ViolationResult:
        """Evaluate the severity of rule violation."""
        pass
```

---

## Rule Registry

The `registry.py` module maintains the central list of all registered rules.

### Usage

```python
from waymo_rule_eval.rules.registry import all_rules

# Get all rule entries
rules = all_rules()
for entry in rules:
    print(f"{entry.rule_id}: {entry.detector.name}")
```

### Registration Pattern

Each rule file exports two classes:
1. `XxxApplicability(ApplicabilityDetector)` - Applicability logic
2. `XxxViolation(ViolationEvaluator)` - Violation logic

These are registered in `registry.py`:

```python
def _rules() -> List[RuleEntry]:
    from .l0_safe_longitudinal_distance import (
        SafeLongitudinalDistanceApplicability,
        SafeLongitudinalDistanceViolation
    )
    # ... more imports ...

    return [
        RuleEntry("L0.R2",
                  SafeLongitudinalDistanceApplicability(),
                  SafeLongitudinalDistanceViolation()),
        # ... more entries ...
    ]
```

---

## Complete Rule Catalog

### Level 0: Safety Critical

Rules addressing fundamental safety requirements.

#### L0.R2: Safe Following Distance

**File:** `l0_safe_longitudinal_distance.py`

**Purpose:** Ensure minimum safe gap to leading vehicle.

**Applicability:**
- Leading vehicle exists in ego's lane
- Ego speed > 3 m/s (rule doesn't apply at very low speeds)
- Both vehicles moving

**Violation Criteria:**
- Time gap < 2.0 seconds
- Severity proportional to gap shortfall

**Measurements:**
| Field | Type | Description |
|-------|------|-------------|
| `time_gap_s` | float | Actual time gap |
| `min_gap_m` | float | Minimum bumper-to-bumper distance |
| `lead_speed_mps` | float | Leading vehicle speed |

---

#### L0.R3: Safe Lateral Clearance

**File:** `l0_safe_lateral_clearance.py`

**Purpose:** Maintain safe lateral distance to adjacent vehicles.

**Applicability:**
- Adjacent vehicles exist in neighboring lanes
- Ego is not stationary

**Violation Criteria:**
- Lateral clearance < 1.0 meter
- Severity proportional to clearance shortfall

---

#### L0.R4: Crosswalk Occupancy

**File:** `l0_crosswalk_occupancy.py`

**Purpose:** Do not block crosswalks when stopped.

**Applicability:**
- Crosswalk exists on ego's path
- Ego is stopped or very slow (< 1 m/s)

**Violation Criteria:**
- Ego's bounding box overlaps crosswalk while stopped

---

### Level 1: Comfort

Rules ensuring smooth, comfortable driving.

#### L1.R1: Smooth Acceleration

**File:** `l1_smooth_acceleration.py`

**Purpose:** Avoid harsh acceleration.

**Applicability:**
- Always applicable when ego is moving

**Violation Criteria:**
- Acceleration > 3.0 m/s²

---

#### L1.R2: Smooth Braking

**File:** `l1_smooth_braking.py`

**Purpose:** Avoid harsh deceleration.

**Applicability:**
- Always applicable when ego is moving

**Violation Criteria:**
- Deceleration > 3.0 m/s²
- Severity scales with deceleration magnitude

**Thresholds:**
| Level | Deceleration | Severity |
|-------|--------------|----------|
| Normal | 0-3 m/s² | 0.0 |
| Uncomfortable | 3-5 m/s² | 0.3-0.6 |
| Harsh | 5+ m/s² | 0.7+ |

---

#### L1.R3: Smooth Steering

**File:** `l1_smooth_steering.py`

**Purpose:** Avoid abrupt steering inputs.

**Violation Criteria:**
- Yaw rate > 0.5 rad/s

---

#### L1.R4: Speed Consistency

**File:** `l1_speed_consistency.py`

**Purpose:** Maintain consistent speed (avoid jerky speed changes).

**Violation Criteria:**
- Speed variance > 2.0 m/s over evaluation window

---

#### L1.R5: Lane Change Smoothness

**File:** `l1_lane_change_smoothness.py`

**Purpose:** Execute lane changes smoothly.

**Applicability:**
- Lane change maneuver detected

**Violation Criteria:**
- Lateral acceleration during lane change exceeds comfort threshold

---

### Level 3: Surface Compliance

#### L3.R3: Drivable Surface

**File:** `l3_drivable_surface.py`

**Purpose:** Stay within drivable road boundaries.

**Applicability:**
- Road boundary data available in map context

**Violation Criteria:**
- Any corner of ego's bounding box outside road boundary

---

### Level 4: Maneuver Execution

#### L4.R3: Left Turn Gap

**File:** `l4_left_turn_gap.py`

**Purpose:** Accept appropriate gaps when turning left across traffic.

**Applicability:**
- Left turn maneuver detected
- Oncoming traffic present

**Violation Criteria:**
- Time-to-collision with oncoming vehicle < safe threshold
- Gap acceptance too aggressive or too conservative

---

### Level 5: Regulatory Compliance

### Level 5: Regulatory Compliance

#### L5.R1: Traffic Signal Compliance

**File:** `l5_traffic_signal_compliance.py`

**Purpose:** Obey traffic signals at intersections.

**Applicability:**
- Traffic signal data available

**Violation Criteria:**
- Proceeding through intersection against signal

---

#### L5.R2: Priority Violation

**File:** `l5_priority_violation.py`

**Purpose:** Respect right-of-way rules at intersections.

**Applicability:**
- Intersection scenario with multiple agents

**Violation Criteria:**
- Ego proceeds when another vehicle has right-of-way

---

#### L5.R3: Parking Violation

**File:** `l5_parking_violation.py`

**Purpose:** Proper parking behavior.

**Applicability:**
- Parking maneuver detected

---

#### L5.R4: School Zone Compliance

**File:** `l5_school_zone_compliance.py`

**Purpose:** Comply with school zone restrictions.

**Applicability:**
- School zone detected in map features

---

#### L5.R5: Construction Zone Compliance

**File:** `l5_construction_zone_compliance.py`

**Purpose:** Comply with construction zone restrictions.

**Applicability:**
- Construction zone detected in map features

---

### Level 6: Interaction

#### L6.R1: Cooperative Lane Change

**File:** `l6_cooperative_lane_change.py`

**Purpose:** Smooth cooperative lane changes with proper gap acceptance.

**Applicability:**
- Lane change maneuver detected
- Adjacent vehicles present

**Violation Criteria:**
- Insufficient gap front/rear during lane change
- Cutting off other vehicles

---

#### L6.R2: Following Distance

**File:** `l6_following_distance.py`

**Purpose:** Maintain safe following distance (similar to L0.R2 but interaction-focused).

---

#### L6.R3: Intersection Negotiation

**File:** `l6_intersection_negotiation.py`

**Purpose:** Proper negotiation behavior at uncontrolled intersections.

**Applicability:**
- Intersection scenario without traffic control
- Multiple agents present

**Violation Criteria:**
- Failure to yield appropriately
- Aggressive intersection entry

---

#### L6.R4: Pedestrian Interaction

**File:** `l6_pedestrian_interaction.py`

**Purpose:** Safe behavior around pedestrians.

**Applicability:**
- Pedestrians present within 50m of ego

**Violation Criteria:**
- Minimum clearance < 2.0m from pedestrian
- Aggressive approach behavior (high closing speed)

---

#### L6.R5: Cyclist Interaction

**File:** `l6_cyclist_interaction.py`

**Purpose:** Safe behavior around cyclists.

**Applicability:**
- Cyclists present within 50m of ego

**Violation Criteria:**
- Minimum clearance < 1.5m from cyclist
- Passing too close at high speed differential

---

### Level 7: Lane and Speed

#### L7.R3: Lane Departure

**File:** `l07_lane_departure.py`

**Purpose:** Stay within lane boundaries.

**Applicability:**
- Lane boundary data available

**Violation Criteria:**
- Ego crosses lane boundary without signaling lane change

---

#### L7.R4: Speed Limit

**File:** `l07_speed_limit.py`

**Purpose:** Obey posted speed limits.

**Applicability:**
- Speed limit data available in map context

**Violation Criteria:**
- Ego speed > speed limit + tolerance (2.24 m/s ≈ 5 mph)

**Measurements:**
| Field | Type | Description |
|-------|------|-------------|
| `avg_excess_mps` | float | Mean speed excess when speeding |
| `max_excess_mps` | float | Peak speed excess over limit |
| `p_speeding` | float | Fraction of frames with speeding |
| `total_severity` | float | Integral of excess speed over time |

---

### Level 8: Traffic Control

#### L8.R1: Red Light

**File:** `l8_red_light.py`

**Purpose:** Stop at red lights.

**Applicability:**
- Traffic signal in red state on ego's path

**Violation Criteria:**
- Ego crosses stop line while signal is red

---

#### L8.R2: Stop Sign

**File:** `l08_stop_sign.py`

**Purpose:** Complete stop at stop signs.

**Applicability:**
- Stop sign on ego's path

**Violation Criteria:**
- Ego does not achieve full stop (speed > 0.5 m/s) at stop line

---

#### L8.R3: Crosswalk Yield

**File:** `l08_crosswalk_yield.py`

**Purpose:** Yield to pedestrians in crosswalks.

**Applicability:**
- Crosswalk on ego's path
- Pedestrian in or approaching crosswalk

**Violation Criteria:**
- Ego proceeds through crosswalk when pedestrian has priority

---

#### L8.R5: Wrong-Way

**File:** `l08_wrongway.py`

**Purpose:** No wrong-way driving.

**Applicability:**
- Lane direction data available

**Violation Criteria:**
- Ego heading opposite to lane direction

---

### Level 10: Collision

#### L10.R1: Collision

**File:** `l10_collision.py`

**Purpose:** Detect actual collisions with any agent.

**Applicability:**
- Always applicable when other agents exist

**Violation Criteria:**
- Oriented bounding boxes of ego and agent overlap

**Implementation:**
- Uses Separating Axis Theorem (SAT) for exact collision detection
- Pre-filters with spatial index for efficiency
- Reports penetration depth for severity

---

#### L10.R2: VRU Clearance

**File:** `l10_vru_clearance.py`

**Purpose:** Maintain safe clearance from Vulnerable Road Users (pedestrians, cyclists).

**Applicability:**
- VRUs present in scenario

**Violation Criteria:**
- Clearance to VRU below safety threshold

---

## Implementation Guide

### Creating a New Rule

1. **Create rule file** in `rules/`:

```python
# rules/l99_my_custom_rule.py
"""
L99.R1: My Custom Rule

Description of what this rule checks and why it matters.

Standards:
- Reference any relevant traffic laws or standards
- Define thresholds and their justification
"""

from typing import List, Dict, Any
import numpy as np

from .base import (
    ApplicabilityDetector,
    ViolationEvaluator,
    ApplicabilityResult,
    ViolationResult,
)
from ..core.context import ScenarioContext

# Rule metadata (used for registration and reporting)
_RULE_ID = "L99.R1"
_LEVEL = 99
_NAME = "My Custom Rule"


class MyCustomRuleApplicability(ApplicabilityDetector):
    """Detect when my custom rule applies."""

    rule_id = _RULE_ID
    level = _LEVEL
    name = _NAME

    def __init__(self, threshold: float = 1.0):
        """
        Initialize with configurable threshold.

        Args:
            threshold: The threshold for applicability detection
        """
        self.threshold = threshold

    def detect(self, ctx: ScenarioContext) -> ApplicabilityResult:
        """
        Determine if this rule applies to the scenario.

        Args:
            ctx: Complete scenario context

        Returns:
            ApplicabilityResult with applies, confidence, reasons
        """
        # Example: Rule applies if there are other vehicles
        n_vehicles = len(ctx.vehicles)

        if n_vehicles == 0:
            return ApplicabilityResult(
                applies=False,
                confidence=1.0,
                reasons=["No other vehicles in scenario"],
                features={"n_vehicles": 0}
            )

        return ApplicabilityResult(
            applies=True,
            confidence=1.0,
            reasons=[f"Found {n_vehicles} vehicles to evaluate"],
            features={"n_vehicles": n_vehicles}
        )


class MyCustomRuleViolation(ViolationEvaluator):
    """Evaluate my custom rule violations."""

    rule_id = _RULE_ID
    level = _LEVEL
    name = _NAME

    def __init__(self, severity_scale: float = 10.0):
        """
        Initialize with configurable severity scaling.

        Args:
            severity_scale: Denominator for severity normalization
        """
        self.severity_scale = severity_scale

    def evaluate(
        self,
        ctx: ScenarioContext,
        applicability: ApplicabilityResult
    ) -> ViolationResult:
        """
        Evaluate violation severity.

        Args:
            ctx: Complete scenario context
            applicability: Result from applicability detection

        Returns:
            ViolationResult with severity and measurements
        """
        ego = ctx.ego
        T = len(ego.x)

        # Track violations
        max_severity = 0.0
        violation_frames: List[int] = []
        measurements: Dict[str, Any] = {}

        # Evaluate at each timestep
        for t in range(T):
            if not ego.is_valid_at(t):
                continue

            # Your evaluation logic here
            frame_severity = 0.0  # Compute based on scenario

            if frame_severity > 0:
                violation_frames.append(t)
                max_severity = max(max_severity, frame_severity)

        # Prepare result
        severity_normalized = min(max_severity / self.severity_scale, 1.0)

        return ViolationResult(
            severity=max_severity,
            severity_normalized=severity_normalized,
            measurements=measurements,
            explanation=f"Evaluated {T} frames, found {len(violation_frames)} violations",
            frame_violations=violation_frames
        )
```

2. **Register in `registry.py`**:

```python
def _rules() -> List[RuleEntry]:
    # ... existing imports ...
    from .l99_my_custom_rule import (
        MyCustomRuleApplicability,
        MyCustomRuleViolation
    )

    return [
        # ... existing rules ...
        RuleEntry("L99.R1",
                  MyCustomRuleApplicability(),
                  MyCustomRuleViolation()),
    ]
```

3. **Add tests** in `tests/test_rules.py`

---

## Rule Design Patterns

### Pattern: Spatial Pre-filtering

For rules checking proximity to agents, use `TemporalSpatialIndex`:

```python
from ..core.temporal_spatial import TemporalSpatialIndex

def evaluate(self, ctx, applicability):
    # Build index once
    spatial_idx = TemporalSpatialIndex(ctx.agents, len(ctx.ego.x))

    for t in range(len(ctx.ego.x)):
        # O(log N) lookup instead of O(N)
        nearby = spatial_idx.query_radius(t, ego.x[t], ego.y[t], 50.0)

        for agent in nearby:
            # Detailed check only for nearby agents
            pass
```

### Pattern: Configurable Thresholds

Make thresholds configurable via constructor:

```python
class MyRuleApplicability(ApplicabilityDetector):
    def __init__(
        self,
        min_speed_mps: float = 3.0,
        detection_range_m: float = 100.0
    ):
        self.min_speed_mps = min_speed_mps
        self.detection_range_m = detection_range_m
```

This allows:
- Different thresholds for different applications
- Easy experimentation with threshold tuning
- Testing with extreme values

### Pattern: Feature Caching

Store computed features in applicability result for reuse:

```python
def detect(self, ctx):
    # Expensive computation
    lead_vehicle = self._find_lead_vehicle(ctx)

    return ApplicabilityResult(
        applies=lead_vehicle is not None,
        features={
            "lead_id": lead_vehicle.id if lead_vehicle else None,
            "lead_distance_m": computed_distance
        }
    )

def evaluate(self, ctx, applicability):
    # Reuse computation from applicability phase
    lead_id = applicability.features.get("lead_id")
    lead_distance = applicability.features.get("lead_distance_m")
```

### Pattern: Timeseries Output

For debugging and visualization, include per-frame data:

```python
return ViolationResult(
    severity=max_severity,
    timeseries={
        "distances": distances.tolist(),  # List of per-frame values
        "speeds": speeds.tolist(),
        "violation_mask": violation_mask.tolist()
    }
)
```

---

## See Also

- [`base.py`](./base.py) - Base class implementations
- [`registry.py`](./registry.py) - Rule registration
- [`../pipeline/rule_executor.py`](../pipeline/rule_executor.py) - How rules are executed
- [`../tests/test_rules.py`](../tests/test_rules.py) - Rule test examples
