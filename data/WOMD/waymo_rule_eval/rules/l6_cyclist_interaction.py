"""
L6.R5: Cyclist Interaction Rule

Ensures safe interactions with cyclists through appropriate passing distances,
speed adjustments, and yielding behavior when required.

Safe cyclist interaction requires:
- Minimum lateral clearance: 1.5 meters (5 feet)
- Safe overtaking with adequate gap
- Speed reduction when passing
- Yielding when cyclist has right-of-way
"""

import numpy as np

from ..core.context import ScenarioContext
from .base import (
    ApplicabilityDetector,
    ApplicabilityResult,
    ViolationEvaluator,
    ViolationResult,
)

_RULE_ID = "L6.R5"
_RULE_LEVEL = 6
_RULE_NAME = "Cyclist interaction"

# Normalization for severity
RULE_NORM = 12.0

# Default thresholds
DEFAULT_PROXIMITY_THRESHOLD_M = 10.0
DEFAULT_MIN_LATERAL_CLEARANCE_M = 1.5
DEFAULT_MAX_SPEED_NEAR_CYCLIST_MPS = 8.9  # ~20 mph


class CyclistInteractionApplicability(ApplicabilityDetector):
    """
    Detects cyclist encounter scenarios for ego vehicle.

    Identifies when ego is in proximity to cyclists.
    """

    rule_id = _RULE_ID
    level = _RULE_LEVEL
    name = _RULE_NAME

    def __init__(
        self,
        proximity_threshold_m: float = DEFAULT_PROXIMITY_THRESHOLD_M,
        min_encounter_duration_s: float = 0.0,  # Reduced for sparse data
    ):
        """
        Initialize cyclist interaction applicability detector.

        Args:
            proximity_threshold_m: Maximum distance to consider encounter
            min_encounter_duration_s: Minimum duration for valid encounter
        """
        self.proximity_threshold_m = proximity_threshold_m
        self.min_encounter_duration_s = min_encounter_duration_s

    def detect(self, ctx: ScenarioContext) -> ApplicabilityResult:
        """
        Detect cyclist encounter scenarios.

        Args:
            ctx: Scenario context with ego state and other agents

        Returns:
            ApplicabilityResult with cyclist encounter information
        """
        # Validate ego state
        if ctx.ego is None or not hasattr(ctx.ego, "x"):
            return ApplicabilityResult(
                applies=False,
                confidence=0.0,
                reasons=["No ego state data available"],
                features={},
                rule_id=self.rule_id,
                rule_level=self.level,
                name=self.name,
            )

        ego_x = np.atleast_1d(ctx.ego.x)
        ego_y = np.atleast_1d(ctx.ego.y)

        if len(ego_x) < 2:
            return ApplicabilityResult(
                applies=False,
                confidence=0.0,
                reasons=["Insufficient ego trajectory data"],
                features={},
                rule_id=self.rule_id,
                rule_level=self.level,
                name=self.name,
            )

        # Find cyclists
        cyclists = ctx.cyclists if hasattr(ctx, "cyclists") else []

        if len(cyclists) == 0:
            return ApplicabilityResult(
                applies=False,
                confidence=1.0,
                reasons=["No cyclists in scenario"],
                features={},
                rule_id=self.rule_id,
                rule_level=self.level,
                name=self.name,
            )

        # Check for proximity encounters
        n_frames = len(ego_x)
        encounter_frames = []

        for t in range(n_frames):
            ego_pos = np.array([ego_x[t], ego_y[t]])

            for cyc in cyclists:
                if t >= len(cyc.x) or not cyc.is_valid_at(t):
                    continue

                cyc_pos = np.array([cyc.x[t], cyc.y[t]])
                distance = np.linalg.norm(cyc_pos - ego_pos)

                if distance < self.proximity_threshold_m:
                    encounter_frames.append(t)
                    break

        if len(encounter_frames) == 0:
            return ApplicabilityResult(
                applies=False,
                confidence=1.0,
                reasons=["No cyclist proximity encounters"],
                features={},
                rule_id=self.rule_id,
                rule_level=self.level,
                name=self.name,
            )

        # Check minimum duration
        dt = ctx.dt
        encounter_duration = len(encounter_frames) * dt

        if encounter_duration < self.min_encounter_duration_s:
            return ApplicabilityResult(
                applies=False,
                confidence=0.8,
                reasons=[f"Encounter duration too short"],
                features={},
                rule_id=self.rule_id,
                rule_level=self.level,
                name=self.name,
            )

        features = {
            "cyclists": cyclists,
            "encounter_frames": encounter_frames,
            "encounter_duration_s": encounter_duration,
        }

        return ApplicabilityResult(
            applies=True,
            confidence=1.0,
            reasons=[
                f"{len(cyclists)} cyclist(s), {encounter_duration:.1f}s encounter"
            ],
            features=features,
            rule_id=self.rule_id,
            rule_level=self.level,
            name=self.name,
        )


class CyclistInteractionViolation(ViolationEvaluator):
    """
    Evaluates cyclist interaction safety compliance.

    Checks for:
    - Safe lateral clearance during passing
    - Appropriate speed when near cyclists
    - Safe overtaking behavior
    """

    rule_id = _RULE_ID
    level = 6
    name = _RULE_NAME

    def __init__(
        self,
        min_lateral_clearance_m: float = DEFAULT_MIN_LATERAL_CLEARANCE_M,
        max_speed_near_cyclist_mps: float = DEFAULT_MAX_SPEED_NEAR_CYCLIST_MPS,
        max_severity: float = RULE_NORM,
    ):
        """
        Initialize cyclist interaction evaluator.

        Args:
            min_lateral_clearance_m: Minimum safe passing distance
            max_speed_near_cyclist_mps: Max speed near cyclists
            max_severity: Maximum severity score
        """
        self.min_lateral_clearance_m = min_lateral_clearance_m
        self.max_speed_near_cyclist_mps = max_speed_near_cyclist_mps
        self.max_severity = max_severity

    def evaluate(
        self, ctx: ScenarioContext, applicability: ApplicabilityResult
    ) -> ViolationResult:
        """
        Evaluate cyclist interaction compliance.

        Args:
            ctx: Scenario context
            applicability: Applicability result with cyclist encounters

        Returns:
            ViolationResult with severity and measurements
        """
        if not applicability.applies:
            return ViolationResult(
                severity=0.0,
                severity_normalized=0.0,
                measurements={},
                explanation="Rule not applicable",
                frame_violations=[],
                rule_id=self.rule_id,
                name=self.name,
            )

        features = applicability.features
        cyclists = features.get("cyclists", [])
        encounter_frames = features.get("encounter_frames", [])

        ego_x = np.atleast_1d(ctx.ego.x)
        ego_y = np.atleast_1d(ctx.ego.y)
        ego_speed = np.atleast_1d(ctx.ego.speed)
        dt = ctx.dt

        n_frames = len(ego_x)

        # Track violations
        violation_frames = []
        clearance_violations = 0
        speed_violations = 0
        min_clearance = float("inf")
        max_speed_near_cyc = 0.0
        total_severity = 0.0

        # Evaluate each encounter frame
        for t in encounter_frames:
            if t >= n_frames:
                continue

            ego_pos = np.array([ego_x[t], ego_y[t]])
            current_speed = ego_speed[t]

            # Find minimum distance to any cyclist
            frame_min_clearance = float("inf")

            for cyc in cyclists:
                if t >= len(cyc.x) or not cyc.is_valid_at(t):
                    continue

                cyc_pos = np.array([cyc.x[t], cyc.y[t]])
                distance = np.linalg.norm(cyc_pos - ego_pos)

                # Account for vehicle size
                clearance = distance - (ctx.ego.length / 2 + 0.5)
                frame_min_clearance = min(frame_min_clearance, clearance)

            # Update tracking
            if frame_min_clearance < min_clearance:
                min_clearance = frame_min_clearance

            if current_speed > max_speed_near_cyc:
                max_speed_near_cyc = current_speed

            frame_violation = False

            # Check clearance violation
            if frame_min_clearance < self.min_lateral_clearance_m:
                clearance_violations += 1
                clearance_deficit = self.min_lateral_clearance_m - frame_min_clearance
                severity_contrib = clearance_deficit / self.min_lateral_clearance_m
                total_severity += severity_contrib
                frame_violation = True

            # Check speed violation when close
            if frame_min_clearance < self.min_lateral_clearance_m * 2:
                if current_speed > self.max_speed_near_cyclist_mps:
                    speed_violations += 1
                    speed_excess = current_speed - self.max_speed_near_cyclist_mps
                    severity_contrib = (
                        speed_excess / self.max_speed_near_cyclist_mps
                    ) * 0.3
                    total_severity += severity_contrib * dt
                    frame_violation = True

            if frame_violation:
                violation_frames.append(t)

        # Cap severity
        total_severity = min(total_severity, self.max_severity)
        normalized = min(1.0, total_severity / self.max_severity)

        # Build measurements
        measurements = {
            "min_lateral_clearance_m": (
                min_clearance if min_clearance < float("inf") else None
            ),
            "max_speed_near_cyclist_mps": max_speed_near_cyc,
            "clearance_violation_frames": clearance_violations,
            "speed_violation_frames": speed_violations,
            "total_encounter_frames": len(encounter_frames),
            "cyclist_count": len(cyclists),
        }

        # Build explanation
        if total_severity > 0:
            explanation = [
                f"Cyclist interaction violations detected",
                f"Minimum clearance: {min_clearance:.2f}m (required: {self.min_lateral_clearance_m:.1f}m)",
                f"Max speed near cyclist: {max_speed_near_cyc:.1f} m/s",
                f"Clearance violations: {clearance_violations} frames",
            ]
        else:
            explanation = [
                f"Safe cyclist interaction",
                f"Minimum clearance: {min_clearance:.2f}m (OK)",
            ]

        return ViolationResult(
            severity=total_severity,
            severity_normalized=normalized,
            measurements=measurements,
            explanation=explanation,
            frame_violations=list(set(violation_frames)),
            rule_id=self.rule_id,
            name=self.name,
        )
