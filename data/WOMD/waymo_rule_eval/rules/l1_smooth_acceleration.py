"""
L1.R1: Smooth Acceleration Rule

Evaluates passenger comfort by detecting harsh acceleration and jerk.
Vehicles should accelerate smoothly without excessive acceleration or jerk
that causes passenger discomfort.

Standards:
- Maximum comfortable acceleration: 2.0 m/s²
- Maximum comfortable jerk: 2.0 m/s³
- Critical acceleration threshold: 3.0 m/s²
- Critical jerk threshold: 4.0 m/s³

Severity Calculation:
- severity = ∫(excess_accel + excess_jerk) dt
- Normalized by RULE_NORM for consistent scoring
"""

import logging
from typing import Optional

import numpy as np

from ..core.context import ScenarioContext
from ..utils.constants import (
    L1_COMFORTABLE_ACCEL_MPS2,
    L1_COMFORTABLE_JERK_MPS3,
    L1_CRITICAL_ACCEL_MPS2,
    L1_CRITICAL_JERK_MPS3,
    MIN_MOVING_SPEED_MPS,
)
from .base import (
    ApplicabilityDetector,
    ApplicabilityResult,
    ViolationEvaluator,
    ViolationResult,
)

log = logging.getLogger(__name__)

# Normalization constant for severity
RULE_NORM = {"L1.R1": 10.0}


class SmoothAccelerationApplicability(ApplicabilityDetector):
    """
    Detects scenarios where smooth acceleration rule applies.

    Applies when:
    - Vehicle has valid speed data for derivative calculations
    - Vehicle is moving (speed > minimum threshold)
    - Sufficient data points for acceleration and jerk calculation
    """

    rule_id = "L1.R1"
    level = 1
    name = "Smooth acceleration"

    def __init__(
        self, min_speed_mps: float = MIN_MOVING_SPEED_MPS, min_frames: int = 3
    ):
        """
        Initialize smooth acceleration applicability detector.

        Args:
            min_speed_mps: Minimum speed to consider (m/s). Below this,
                          vehicle is considered stationary.
            min_frames: Minimum number of frames needed for derivative calculation.
        """
        self.min_speed_mps = min_speed_mps
        self.min_frames = min_frames

    def detect(self, ctx: ScenarioContext) -> ApplicabilityResult:
        """
        Detect if smooth acceleration rule applies to this scenario.

        Args:
            ctx: Scenario context with ego vehicle state

        Returns:
            ApplicabilityResult indicating if rule applies
        """
        reasons = []
        features = {}

        # Check if we have enough frames
        n_frames = len(ctx.ego.speed)
        if n_frames < self.min_frames:
            reasons.append(f"Insufficient frames ({n_frames} < {self.min_frames})")
            return ApplicabilityResult(
                rule_id=self.rule_id,
                rule_level=self.level,
                name=self.name,
                applies=False,
                confidence=0.0,
                reasons=reasons,
                features=features,
            )

        # Check if vehicle is moving in any frame
        moving_mask = ctx.ego.speed > self.min_speed_mps
        moving_frames = int(np.sum(moving_mask))

        if moving_frames == 0:
            reasons.append("Vehicle not moving (all speeds below threshold)")
            return ApplicabilityResult(
                rule_id=self.rule_id,
                rule_level=self.level,
                name=self.name,
                applies=False,
                confidence=0.0,
                reasons=reasons,
                features=features,
            )

        # Calculate confidence based on proportion of moving frames
        confidence = float(moving_frames) / n_frames

        features["moving_frames"] = moving_frames
        features["total_frames"] = n_frames
        features["max_speed_mps"] = float(np.max(ctx.ego.speed))

        reasons.append(f"Vehicle moving in {moving_frames}/{n_frames} frames")

        return ApplicabilityResult(
            rule_id=self.rule_id,
            rule_level=self.level,
            name=self.name,
            applies=True,
            confidence=confidence,
            reasons=reasons,
            features=features,
        )


class SmoothAccelerationViolation(ViolationEvaluator):
    """
    Evaluates smooth acceleration violations.

    Detects:
    - Harsh acceleration (magnitude > comfortable threshold)
    - Excessive jerk (magnitude > comfortable threshold)

    Calculates severity as integral of excess acceleration and jerk over time.
    """

    rule_id = "L1.R1"
    level = 1
    name = "Smooth acceleration"

    def __init__(
        self,
        comfortable_accel_mps2: float = L1_COMFORTABLE_ACCEL_MPS2,
        critical_accel_mps2: float = L1_CRITICAL_ACCEL_MPS2,
        comfortable_jerk_mps3: float = L1_COMFORTABLE_JERK_MPS3,
        critical_jerk_mps3: float = L1_CRITICAL_JERK_MPS3,
        min_speed_mps: float = MIN_MOVING_SPEED_MPS,
    ):
        """
        Initialize smooth acceleration violation evaluator.

        Args:
            comfortable_accel_mps2: Maximum comfortable acceleration (m/s²)
            critical_accel_mps2: Critical acceleration threshold (m/s²)
            comfortable_jerk_mps3: Maximum comfortable jerk (m/s³)
            critical_jerk_mps3: Critical jerk threshold (m/s³)
            min_speed_mps: Minimum speed to evaluate (m/s)
        """
        self.comfortable_accel = comfortable_accel_mps2
        self.critical_accel = critical_accel_mps2
        self.comfortable_jerk = comfortable_jerk_mps3
        self.critical_jerk = critical_jerk_mps3
        self.min_speed = min_speed_mps

    def evaluate(
        self, ctx: ScenarioContext, app: ApplicabilityResult
    ) -> ViolationResult:
        """
        Evaluate smooth acceleration violations.

        Args:
            ctx: Scenario context with ego vehicle state
            app: Applicability result from detector

        Returns:
            ViolationResult with severity and measurements
        """
        n_frames = len(ctx.ego.speed)

        # Calculate acceleration (derivative of speed)
        # accel[i] = (speed[i+1] - speed[i]) / dt
        acceleration = np.zeros(n_frames)
        if n_frames >= 2:
            acceleration[:-1] = np.diff(ctx.ego.speed) / ctx.dt
            acceleration[-1] = acceleration[-2] if n_frames >= 2 else 0.0

        # Calculate jerk (derivative of acceleration)
        # jerk[i] = (accel[i+1] - accel[i]) / dt
        jerk = np.zeros(n_frames)
        if n_frames >= 3:
            jerk[:-1] = np.diff(acceleration) / ctx.dt
            jerk[-1] = jerk[-2]

        # Get absolute values for threshold comparison
        accel_mag = np.abs(acceleration)
        jerk_mag = np.abs(jerk)

        # Only evaluate frames where vehicle is moving
        moving_mask = ctx.ego.speed > self.min_speed
        moving_count = int(np.sum(moving_mask))

        # Detect violations
        accel_violation = (accel_mag > self.comfortable_accel) & moving_mask
        jerk_violation = (jerk_mag > self.comfortable_jerk) & moving_mask

        # Combined violation mask
        violation_mask = accel_violation | jerk_violation
        violation_frames = int(np.sum(violation_mask))

        # Calculate excess acceleration and jerk
        excess_accel = np.maximum(0.0, accel_mag - self.comfortable_accel)
        excess_jerk = np.maximum(0.0, jerk_mag - self.comfortable_jerk)

        # Apply moving mask
        excess_accel = excess_accel * moving_mask
        excess_jerk = excess_jerk * moving_mask

        # Calculate severity as integral of excess over time
        # severity = ∫(excess_accel + excess_jerk) dt
        severity_per_frame = (excess_accel + excess_jerk) * ctx.dt
        severity = float(np.sum(severity_per_frame))

        # Normalize severity (capped at 1.0)
        norm_factor = RULE_NORM.get(self.rule_id, 1.0)
        normalized = min(1.0, severity / norm_factor) if norm_factor > 0 else 0.0

        # Create timeseries for violation tracking
        timeseries = {"violation": violation_mask.astype(float).tolist()}

        # Measurements
        measurements = {
            "max_acceleration_mps2": (
                float(np.max(accel_mag[moving_mask])) if moving_count > 0 else 0.0
            ),
            "max_jerk_mps3": (
                float(np.max(jerk_mag[moving_mask])) if moving_count > 0 else 0.0
            ),
            "avg_acceleration_mps2": (
                float(np.mean(accel_mag[moving_mask])) if moving_count > 0 else 0.0
            ),
            "avg_jerk_mps3": (
                float(np.mean(jerk_mag[moving_mask])) if moving_count > 0 else 0.0
            ),
            "accel_violation_frames": int(np.sum(accel_violation)),
            "jerk_violation_frames": int(np.sum(jerk_violation)),
            "total_violation_frames": violation_frames,
            "total_moving_frames": moving_count,
            "p_violation": (
                float(violation_frames / moving_count) if moving_count > 0 else 0.0
            ),
            "total_severity": severity,
            "comfortable_accel_threshold": self.comfortable_accel,
            "comfortable_jerk_threshold": self.comfortable_jerk,
        }

        # Explanation
        explanation = []
        if violation_frames > 0:
            explanation.append(
                f"Harsh acceleration detected in {violation_frames}/{moving_count} moving frames. "
                f"Max acceleration: {measurements['max_acceleration_mps2']:.2f} m/s² "
                f"(comfortable: {self.comfortable_accel:.2f} m/s²). "
                f"Max jerk: {measurements['max_jerk_mps3']:.2f} m/s³ "
                f"(comfortable: {self.comfortable_jerk:.2f} m/s³). "
                f"Total severity: {severity:.2f}."
            )
        else:
            explanation.append(
                f"Smooth acceleration in all {moving_count} moving frames. "
                f"Max acceleration: {measurements['max_acceleration_mps2']:.2f} m/s² "
                f"(comfortable: {self.comfortable_accel:.2f} m/s²). "
                f"Max jerk: {measurements['max_jerk_mps3']:.2f} m/s³ "
                f"(comfortable: {self.comfortable_jerk:.2f} m/s³)."
            )

        # Confidence based on data quality
        confidence = 1.0 if moving_count >= 3 else 0.5

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
