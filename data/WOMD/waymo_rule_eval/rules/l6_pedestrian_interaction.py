"""
L6.R4: Pedestrian Interaction Rule

Ensures safe and courteous interactions with pedestrians through proper yielding,
safe passing distances, and appropriate speed reductions.

Safe pedestrian interaction requires:
- Minimum lateral clearance: 1.5 meters (5 feet)
- Speed reduction near pedestrians (≤ 15 mph / 6.7 m/s)
- Yielding when pedestrians have right-of-way
"""

import numpy as np

from ..core.context import ScenarioContext
from ..utils.constants import MIN_MOVING_SPEED_MPS
from .base import (
    ApplicabilityDetector,
    ApplicabilityResult,
    ViolationEvaluator,
    ViolationResult,
)

_RULE_ID = "L6.R4"
_RULE_LEVEL = 6
_RULE_NAME = "Pedestrian interaction"

# Normalization for severity
RULE_NORM = 12.0

# Default thresholds
DEFAULT_PROXIMITY_THRESHOLD_M = 10.0
DEFAULT_MIN_LATERAL_CLEARANCE_M = 1.5
DEFAULT_MAX_SPEED_NEAR_PED_MPS = 6.7  # ~15 mph
DEFAULT_CLOSE_PROXIMITY_M = 3.0


class PedestrianInteractionApplicability(ApplicabilityDetector):
    """
    Detects pedestrian encounter scenarios for ego vehicle.

    Identifies when ego is in proximity to pedestrians.
    """

    rule_id = _RULE_ID
    level = _RULE_LEVEL
    name = _RULE_NAME

    def __init__(
        self,
        proximity_threshold_m: float = DEFAULT_PROXIMITY_THRESHOLD_M,
        min_encounter_duration_s: float = 0.0,  # Reduced to support sparse data
    ):
        """
        Initialize pedestrian interaction applicability detector.

        Args:
            proximity_threshold_m: Maximum distance to consider encounter
            min_encounter_duration_s: Minimum duration for valid encounter
        """
        self.proximity_threshold_m = proximity_threshold_m
        self.min_encounter_duration_s = min_encounter_duration_s

    def detect(self, ctx: ScenarioContext) -> ApplicabilityResult:
        """
        Detect pedestrian encounter scenarios.

        Args:
            ctx: Scenario context with ego state and other agents

        Returns:
            ApplicabilityResult with pedestrian encounter information
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

        # Find pedestrians
        pedestrians = ctx.pedestrians if hasattr(ctx, "pedestrians") else []

        if len(pedestrians) == 0:
            return ApplicabilityResult(
                applies=False,
                confidence=1.0,
                reasons=["No pedestrians in scenario"],
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

            for ped in pedestrians:
                if t >= len(ped.x) or not ped.is_valid_at(t):
                    continue

                ped_pos = np.array([ped.x[t], ped.y[t]])
                distance = np.linalg.norm(ped_pos - ego_pos)

                if distance < self.proximity_threshold_m:
                    encounter_frames.append(t)
                    break

        if len(encounter_frames) == 0:
            return ApplicabilityResult(
                applies=False,
                confidence=1.0,
                reasons=["No pedestrian proximity encounters"],
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
                reasons=[
                    f"Encounter duration {encounter_duration:.1f}s < minimum {self.min_encounter_duration_s:.1f}s"
                ],
                features={},
                rule_id=self.rule_id,
                rule_level=self.level,
                name=self.name,
            )

        features = {
            "pedestrians": pedestrians,
            "encounter_frames": encounter_frames,
            "encounter_duration_s": encounter_duration,
        }

        return ApplicabilityResult(
            applies=True,
            confidence=1.0,
            reasons=[
                f"{len(pedestrians)} pedestrian(s), {encounter_duration:.1f}s encounter time"
            ],
            features=features,
            rule_id=self.rule_id,
            rule_level=self.level,
            name=self.name,
        )


class PedestrianInteractionViolation(ViolationEvaluator):
    """
    Evaluates pedestrian interaction safety compliance.

    Checks for:
    - Safe lateral clearance
    - Speed reduction near pedestrians
    - Yielding behavior
    """

    rule_id = _RULE_ID
    level = 6
    name = _RULE_NAME

    def __init__(
        self,
        min_lateral_clearance_m: float = DEFAULT_MIN_LATERAL_CLEARANCE_M,
        max_speed_near_ped_mps: float = DEFAULT_MAX_SPEED_NEAR_PED_MPS,
        close_proximity_m: float = DEFAULT_CLOSE_PROXIMITY_M,
        max_severity: float = RULE_NORM,
    ):
        """
        Initialize pedestrian interaction evaluator.

        Args:
            min_lateral_clearance_m: Minimum safe passing distance
            max_speed_near_ped_mps: Max speed near pedestrians
            close_proximity_m: Very close proximity threshold
            max_severity: Maximum severity score
        """
        self.min_lateral_clearance_m = min_lateral_clearance_m
        self.max_speed_near_ped_mps = max_speed_near_ped_mps
        self.close_proximity_m = close_proximity_m
        self.max_severity = max_severity

    def evaluate(
        self, ctx: ScenarioContext, applicability: ApplicabilityResult
    ) -> ViolationResult:
        """
        Evaluate pedestrian interaction compliance.

        Args:
            ctx: Scenario context
            applicability: Applicability result with pedestrian encounters

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
        pedestrians = features.get("pedestrians", [])
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
        max_speed_near_ped = 0.0
        total_severity = 0.0

        # Evaluate each encounter frame
        for t in encounter_frames:
            if t >= n_frames:
                continue

            ego_pos = np.array([ego_x[t], ego_y[t]])
            current_speed = ego_speed[t]

            # Find minimum distance to any pedestrian
            frame_min_clearance = float("inf")

            for ped in pedestrians:
                if t >= len(ped.x) or not ped.is_valid_at(t):
                    continue

                ped_pos = np.array([ped.x[t], ped.y[t]])
                distance = np.linalg.norm(ped_pos - ego_pos)

                # Account for vehicle size
                clearance = distance - (ctx.ego.length / 2 + 0.5)
                frame_min_clearance = min(frame_min_clearance, clearance)

            # Update tracking
            if frame_min_clearance < min_clearance:
                min_clearance = frame_min_clearance

            if current_speed > max_speed_near_ped:
                max_speed_near_ped = current_speed

            frame_violation = False

            # Check clearance violation
            if frame_min_clearance < self.min_lateral_clearance_m:
                clearance_violations += 1
                clearance_deficit = self.min_lateral_clearance_m - frame_min_clearance
                severity_contrib = clearance_deficit / self.min_lateral_clearance_m
                total_severity += severity_contrib
                frame_violation = True

            # Check speed violation (only if close to pedestrian)
            if frame_min_clearance < self.close_proximity_m * 2:
                if current_speed > self.max_speed_near_ped_mps:
                    speed_violations += 1
                    speed_excess = current_speed - self.max_speed_near_ped_mps
                    severity_contrib = (
                        speed_excess / self.max_speed_near_ped_mps
                    ) * 0.5
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
            "max_speed_near_pedestrian_mps": max_speed_near_ped,
            "clearance_violation_frames": clearance_violations,
            "speed_violation_frames": speed_violations,
            "total_encounter_frames": len(encounter_frames),
            "pedestrian_count": len(pedestrians),
        }

        # Build explanation
        if total_severity > 0:
            explanation = [
                f"Pedestrian interaction violations detected",
                f"Minimum clearance: {min_clearance:.2f}m (required: {self.min_lateral_clearance_m:.1f}m)",
                f"Max speed near pedestrian: {max_speed_near_ped:.1f} m/s",
                f"Clearance violations: {clearance_violations} frames",
                f"Speed violations: {speed_violations} frames",
            ]
        else:
            explanation = [
                f"Safe pedestrian interaction",
                f"Minimum clearance: {min_clearance:.2f}m (OK)",
                f"Max speed near pedestrian: {max_speed_near_ped:.1f} m/s (OK)",
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
