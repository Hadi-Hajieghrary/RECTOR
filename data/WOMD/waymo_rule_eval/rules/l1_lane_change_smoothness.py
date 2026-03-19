"""
L1.R5 Lane Change Smoothness Rule

Evaluates whether the ego vehicle executes lane changes smoothly without
harsh lateral movements that could cause passenger discomfort.

Standards:
    - Comfortable lateral acceleration: ±1.5 m/s²
    - Critical lateral acceleration: ±2.5 m/s²
    - Lane change detection: Sustained lateral movement > 0.5 m
    - Severity normalization: 15.0
"""

import logging
from typing import Any, Dict

import numpy as np

from ..core.context import ScenarioContext
from ..utils.constants import (
    L1_COMFORTABLE_LATERAL_ACCEL_MPS2,
    L1_CRITICAL_LATERAL_ACCEL_MPS2,
    MIN_MOVING_SPEED_MPS,
)
from .base import (
    ApplicabilityDetector,
    ApplicabilityResult,
    ViolationEvaluator,
    ViolationResult,
)

logger = logging.getLogger(__name__)

# Rule-specific normalization factor
RULE_NORM = {"L1.R5": 15.0}


class LaneChangeSmoothnessApplicability(ApplicabilityDetector):
    """
    Detects whether the lane change smoothness rule applies to the scenario.

    The rule applies when:
    - Vehicle is moving (speed > threshold)
    - Lateral movement is detected (lateral velocity > threshold)
    - Sufficient data for lateral dynamics analysis
    """

    rule_id = "L1.R5"
    level = 1
    name = "Lane change smoothness"

    def __init__(
        self,
        min_speed_mps: float = MIN_MOVING_SPEED_MPS,
        min_lateral_velocity_mps: float = 0.1,
        min_frames: int = 3,
    ):
        """
        Initialize the lane change smoothness applicability detector.

        Args:
            min_speed_mps: Minimum speed to consider vehicle as moving
            min_lateral_velocity_mps: Minimum lateral velocity for detection
            min_frames: Minimum frames required for applicability
        """
        self.min_speed_mps = min_speed_mps
        self.min_lateral_velocity_mps = min_lateral_velocity_mps
        self.min_frames = min_frames

    def detect(self, ctx: ScenarioContext) -> ApplicabilityResult:
        """
        Detect if lane change smoothness evaluation applies.

        Args:
            ctx: ScenarioContext with ego state information

        Returns:
            ApplicabilityResult indicating if rule applies
        """
        try:
            # Check for sufficient data
            n_frames = len(ctx.ego.speed)
            if n_frames < self.min_frames:
                return ApplicabilityResult(
                    rule_id=self.rule_id,
                    rule_level=self.level,
                    name=self.name,
                    applies=False,
                    confidence=1.0,
                    reasons=["Insufficient frames for lane change analysis"],
                    features={"total_frames": n_frames},
                )

            dt = ctx.dt

            # Identify moving frames
            moving_mask = ctx.ego.speed >= self.min_speed_mps
            moving_frames = int(np.sum(moving_mask))

            if moving_frames < self.min_frames:
                return ApplicabilityResult(
                    rule_id=self.rule_id,
                    rule_level=self.level,
                    name=self.name,
                    applies=False,
                    confidence=1.0,
                    reasons=["Vehicle is mostly stationary"],
                    features={
                        "moving_frames": moving_frames,
                        "total_frames": n_frames,
                        "min_speed_mps": self.min_speed_mps,
                    },
                )

            # Calculate lateral velocity (perpendicular to heading)
            # Handle NaN values in position/yaw data
            lateral_velocity = np.full(n_frames, np.nan)
            if n_frames > 1:
                # Calculate velocity components
                dx = np.diff(ctx.ego.x)
                dy = np.diff(ctx.ego.y)

                # For each frame (except first), calculate lateral component
                for i in range(1, n_frames):
                    yaw = ctx.ego.yaw[i]
                    if np.isnan(yaw) or np.isnan(dx[i - 1]) or np.isnan(dy[i - 1]):
                        continue  # Skip NaN frames
                    # Lateral direction (perpendicular, left is positive)
                    lateral_x = -np.sin(yaw)
                    lateral_y = np.cos(yaw)

                    # Project velocity onto lateral direction
                    vx = dx[i - 1] / dt
                    vy = dy[i - 1] / dt
                    lateral_velocity[i] = vx * lateral_x + vy * lateral_y

            # Detect lateral movement (absolute lateral velocity)
            # Only consider valid (non-NaN) lateral velocity values
            valid_lateral = ~np.isnan(lateral_velocity)
            lateral_movement_mask = (
                valid_lateral
                & moving_mask
                & (np.abs(lateral_velocity) >= self.min_lateral_velocity_mps)
            )
            lateral_movement_frames = int(np.nansum(lateral_movement_mask))

            if lateral_movement_frames < 2:
                return ApplicabilityResult(
                    rule_id=self.rule_id,
                    rule_level=self.level,
                    name=self.name,
                    applies=False,
                    confidence=0.9,
                    reasons=["No significant lateral movement detected"],
                    features={
                        "lateral_movement_frames": lateral_movement_frames,
                        "moving_frames": moving_frames,
                        "total_frames": n_frames,
                        "max_lateral_velocity_mps": float(
                            np.max(np.abs(lateral_velocity))
                        ),
                    },
                )

            # Rule applies - vehicle has lateral movement
            confidence = min(1.0, lateral_movement_frames / moving_frames)

            return ApplicabilityResult(
                rule_id=self.rule_id,
                rule_level=self.level,
                name=self.name,
                applies=True,
                confidence=confidence,
                reasons=["Vehicle exhibits lateral movement for lane change analysis"],
                features={
                    "lateral_movement_frames": lateral_movement_frames,
                    "moving_frames": moving_frames,
                    "total_frames": n_frames,
                    "max_lateral_velocity_mps": float(np.max(np.abs(lateral_velocity))),
                    "min_lateral_velocity_threshold_mps": self.min_lateral_velocity_mps,
                },
            )

        except Exception as e:
            logger.error(f"Error in LaneChangeSmoothnessApplicability.detect: {e}")
            return ApplicabilityResult(
                rule_id=self.rule_id,
                rule_level=self.level,
                name=self.name,
                applies=False,
                confidence=0.0,
                reasons=[f"Error during detection: {str(e)}"],
                features={},
            )


class LaneChangeSmoothnessViolation(ViolationEvaluator):
    """
    Evaluates lane change smoothness violations.

    Detects harsh lateral accelerations during lane changes that indicate
    uncomfortable lateral movements for passengers.
    """

    rule_id = "L1.R5"
    level = 1
    name = "Lane change smoothness"

    def __init__(
        self,
        comfortable_lateral_accel_mps2: float = L1_COMFORTABLE_LATERAL_ACCEL_MPS2,
        critical_lateral_accel_mps2: float = L1_CRITICAL_LATERAL_ACCEL_MPS2,
        min_speed_mps: float = MIN_MOVING_SPEED_MPS,
        min_lateral_velocity_mps: float = 0.1,
    ):
        """
        Initialize the lane change smoothness violation evaluator.

        Args:
            comfortable_lateral_accel_mps2: Comfortable lateral acceleration threshold
            critical_lateral_accel_mps2: Critical lateral acceleration threshold
            min_speed_mps: Minimum speed to evaluate (filter stationary periods)
            min_lateral_velocity_mps: Minimum lateral velocity for lane change detection
        """
        self.comfortable_lateral_accel_mps2 = comfortable_lateral_accel_mps2
        self.critical_lateral_accel_mps2 = critical_lateral_accel_mps2
        self.min_speed_mps = min_speed_mps
        self.min_lateral_velocity_mps = min_lateral_velocity_mps

    def evaluate(
        self, ctx: ScenarioContext, app: ApplicabilityResult
    ) -> ViolationResult:
        """
        Evaluate lane change smoothness violations.

        Args:
            ctx: ScenarioContext with ego state
            app: ApplicabilityResult from detector

        Returns:
            ViolationResult with severity and measurements
        """
        try:
            if not app.applies:
                return ViolationResult(
                    rule_id=self.rule_id,
                    name=self.name,
                    severity=0.0,
                    severity_normalized=0.0,
                    timeseries={},
                    measurements={},
                    explanation="Rule does not apply to this scenario",
                    confidence=1.0,
                )

            n_frames = len(ctx.ego.speed)
            dt = ctx.dt

            # Identify moving frames
            moving_mask = ctx.ego.speed >= self.min_speed_mps
            moving_count = int(np.sum(moving_mask))

            # Calculate lateral velocity (perpendicular to heading)
            lateral_velocity = np.zeros(n_frames)
            if n_frames > 1:
                dx = np.diff(ctx.ego.x)
                dy = np.diff(ctx.ego.y)

                for i in range(1, n_frames):
                    yaw = ctx.ego.yaw[i]
                    # Lateral direction (perpendicular to heading, left is positive)
                    lateral_x = -np.sin(yaw)
                    lateral_y = np.cos(yaw)

                    # Project velocity onto lateral direction
                    vx = dx[i - 1] / dt
                    vy = dy[i - 1] / dt
                    lateral_velocity[i] = vx * lateral_x + vy * lateral_y

            # Calculate lateral acceleration
            lateral_acceleration = np.zeros(n_frames)
            if n_frames > 2:
                lateral_acceleration[1:] = np.diff(lateral_velocity) / dt

            # Identify lane change frames (significant lateral velocity while moving)
            lane_change_mask = (
                np.abs(lateral_velocity) >= self.min_lateral_velocity_mps
            ) & moving_mask
            lane_change_count = int(np.sum(lane_change_mask))

            # Detect violations: high lateral acceleration during lane changes
            lateral_accel_magnitude = np.abs(lateral_acceleration)
            violations_mask = (
                lateral_accel_magnitude > self.comfortable_lateral_accel_mps2
            ) & lane_change_mask

            # Calculate excess lateral acceleration (for severity)
            excess_lateral_accel = np.maximum(
                0, lateral_accel_magnitude - self.comfortable_lateral_accel_mps2
            )
            excess_lateral_accel_during_lc = excess_lateral_accel * lane_change_mask

            # Check for violations
            has_violation = bool(np.any(violations_mask))

            # Calculate severity: integral of excess lateral acceleration over time
            severity = float(np.sum(excess_lateral_accel_during_lc) * dt)

            # Normalize severity (capped at 1.0)
            normalized = (
                min(1.0, severity / RULE_NORM[self.rule_id]) if severity > 0 else 0.0
            )

            # Prepare measurements
            violation_frames = int(np.sum(violations_mask))
            p_violation = (
                violation_frames / lane_change_count if lane_change_count > 0 else 0.0
            )

            measurements = {
                "max_lateral_accel_mps2": float(np.max(lateral_accel_magnitude)),
                "max_lateral_accel_during_lc_mps2": (
                    float(np.max(lateral_accel_magnitude[lane_change_mask]))
                    if lane_change_count > 0
                    else 0.0
                ),
                "avg_lateral_accel_during_lc_mps2": (
                    float(np.mean(lateral_accel_magnitude[lane_change_mask]))
                    if lane_change_count > 0
                    else 0.0
                ),
                "max_lateral_velocity_mps": float(np.max(np.abs(lateral_velocity))),
                "avg_lateral_velocity_mps": (
                    float(np.mean(np.abs(lateral_velocity[moving_mask])))
                    if moving_count > 0
                    else 0.0
                ),
                "violation_frames": violation_frames,
                "lane_change_frames": lane_change_count,
                "moving_frames": moving_count,
                "total_frames": n_frames,
                "p_violation": p_violation,
                "total_severity": severity,
                "comfortable_lateral_accel_threshold_mps2": self.comfortable_lateral_accel_mps2,
                "critical_lateral_accel_threshold_mps2": self.critical_lateral_accel_mps2,
            }

            # Prepare timeseries
            timeseries = {
                "lateral_velocity_mps": lateral_velocity.tolist(),
                "lateral_acceleration_mps2": lateral_acceleration.tolist(),
                "excess_lateral_accel_mps2": excess_lateral_accel_during_lc.tolist(),
                "moving_mask": moving_mask.astype(int).tolist(),
                "lane_change_mask": lane_change_mask.astype(int).tolist(),
                "violation_mask": violations_mask.astype(int).tolist(),
            }

            # Prepare explanation
            if not has_violation:
                explanation = (
                    f"Smooth lane changes maintained. "
                    f"Max lateral acceleration: {measurements['max_lateral_accel_during_lc_mps2']:.2f} m/s² "
                    f"(threshold: {self.comfortable_lateral_accel_mps2:.1f} m/s²) "
                    f"during {lane_change_count} lane change frames."
                )
            else:
                explanation = (
                    f"Lane change smoothness violation detected: "
                    f"harsh lateral acceleration ({measurements['max_lateral_accel_during_lc_mps2']:.2f} m/s² > "
                    f"{self.comfortable_lateral_accel_mps2:.1f} m/s²). "
                    f"Violation in {violation_frames}/{lane_change_count} lane change frames "
                    f"({p_violation:.1%}). "
                    f"Severity: {severity:.2f} (normalized: {normalized:.3f})"
                )

            confidence = 0.95 if has_violation else 0.85

            return ViolationResult(
                rule_id=self.rule_id,
                name=self.name,
                severity=severity,
                severity_normalized=normalized,
                timeseries=timeseries,
                measurements=measurements,
                explanation=explanation,
                confidence=confidence,
            )

        except Exception as e:
            logger.error(f"Error in LaneChangeSmoothnessViolation.evaluate: {e}")
            return ViolationResult(
                rule_id=self.rule_id,
                name=self.name,
                severity=0.0,
                severity_normalized=0.0,
                timeseries={},
                measurements={},
                explanation=f"Error during evaluation: {str(e)}",
                confidence=0.0,
            )
