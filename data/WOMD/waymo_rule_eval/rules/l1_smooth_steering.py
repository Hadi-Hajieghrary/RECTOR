"""
L1.R3: Smooth Steering Rule

Evaluates passenger comfort by detecting harsh steering and excessive
heading rate changes. Vehicles should steer smoothly without excessive heading
rate (yaw rate) or angular jerk that causes passenger discomfort.

Standards:
- Maximum comfortable heading rate: 15°/s (0.262 rad/s)
- Maximum comfortable angular jerk: 15°/s² (0.262 rad/s²)
- Critical heading rate threshold: 30°/s (0.524 rad/s)
- Critical angular jerk threshold: 30°/s² (0.524 rad/s²)

Severity Calculation:
- severity = ∫(excess_rate + excess_jerk) dt
- Normalized by RULE_NORM for consistent scoring
"""

import logging
from typing import Optional

import numpy as np

from ..core.context import ScenarioContext
from ..utils.constants import (
    L1_COMFORTABLE_ANGULAR_JERK_DEG_S2,
    L1_COMFORTABLE_HEADING_RATE_DEG_S,
    L1_CRITICAL_ANGULAR_JERK_DEG_S2,
    L1_CRITICAL_HEADING_RATE_DEG_S,
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
RULE_NORM = {"L1.R3": 5.0}


class SmoothSteeringApplicability(ApplicabilityDetector):
    """
    Detects scenarios where smooth steering rule applies.

    Applies when:
    - Vehicle has valid yaw data for derivative calculations
    - Vehicle is moving and turning (heading changing)
    - Sufficient data points for heading rate and angular jerk calculation
    """

    rule_id = "L1.R3"
    level = 1
    name = "Smooth steering"

    def __init__(
        self,
        min_speed_mps: float = MIN_MOVING_SPEED_MPS,
        min_frames: int = 3,
        heading_rate_threshold_rad_s: float = 0.01,
    ):
        """
        Initialize smooth steering applicability detector.

        Args:
            min_speed_mps: Minimum speed to consider (m/s). Below this,
                          vehicle is considered stationary.
            min_frames: Minimum number of frames needed for derivative calculation.
            heading_rate_threshold_rad_s: Minimum heading rate magnitude to count as turning (rad/s).
        """
        self.min_speed_mps = min_speed_mps
        self.min_frames = min_frames
        self.heading_rate_threshold = heading_rate_threshold_rad_s

    def detect(self, ctx: ScenarioContext) -> ApplicabilityResult:
        """
        Detect if smooth steering rule applies to this scenario.

        Args:
            ctx: Scenario context with ego vehicle state

        Returns:
            ApplicabilityResult indicating if rule applies
        """
        reasons = []
        features = {}

        # Check if we have enough frames with valid yaw data
        n_frames = len(ctx.ego.yaw)
        valid_yaw_mask = ~np.isnan(ctx.ego.yaw)
        valid_yaw_count = int(np.sum(valid_yaw_mask))

        if valid_yaw_count < self.min_frames:
            reasons.append(
                f"Insufficient valid yaw frames ({valid_yaw_count} < {self.min_frames})"
            )
            return ApplicabilityResult(
                rule_id=self.rule_id,
                rule_level=self.level,
                name=self.name,
                applies=False,
                confidence=0.0,
                reasons=reasons,
                features=features,
            )

        # Check if vehicle is moving in any frame (use nansum for speed)
        valid_speed_mask = ~np.isnan(ctx.ego.speed)
        moving_mask = (ctx.ego.speed > self.min_speed_mps) & valid_speed_mask
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

        # Calculate heading rate (yaw rate) to detect turning
        # Handle NaN values - use valid yaw transitions only
        heading_rate = np.full(n_frames, np.nan)
        if n_frames >= 2:
            yaw_diff = np.diff(ctx.ego.yaw)
            # Normalize angle differences to [-pi, pi] to handle wrap-around
            yaw_diff = (yaw_diff + np.pi) % (2 * np.pi) - np.pi
            heading_rate[:-1] = yaw_diff / ctx.dt

        # Only consider valid heading rate values
        valid_heading = ~np.isnan(heading_rate)

        # Detect turning (heading rate above threshold) - only valid frames
        turning_mask = (
            valid_heading
            & moving_mask
            & (np.abs(heading_rate) > self.heading_rate_threshold)
        )
        turning_frames = int(np.nansum(turning_mask))

        if turning_frames == 0:
            reasons.append("No turning detected (heading rate below threshold)")
            return ApplicabilityResult(
                rule_id=self.rule_id,
                rule_level=self.level,
                name=self.name,
                applies=False,
                confidence=0.0,
                reasons=reasons,
                features=features,
            )

        # Calculate confidence based on proportion of turning frames
        confidence = float(turning_frames) / moving_frames if moving_frames > 0 else 0.0

        features["turning_frames"] = turning_frames
        features["moving_frames"] = moving_frames
        features["total_frames"] = n_frames
        features["max_heading_rate_deg_s"] = float(
            np.max(np.abs(heading_rate[turning_mask])) * 180.0 / np.pi
        )

        reasons.append(
            f"Vehicle turning in {turning_frames}/{moving_frames} moving frames"
        )

        return ApplicabilityResult(
            rule_id=self.rule_id,
            rule_level=self.level,
            name=self.name,
            applies=True,
            confidence=confidence,
            reasons=reasons,
            features=features,
        )


class SmoothSteeringViolation(ViolationEvaluator):
    """
    Evaluates smooth steering violations.

    Detects:
    - Harsh steering (heading rate > comfortable threshold)
    - Excessive angular jerk (magnitude > comfortable threshold)

    Calculates severity as integral of excess heading rate and angular jerk over time.
    """

    rule_id = "L1.R3"
    level = 1
    name = "Smooth steering"

    def __init__(
        self,
        comfortable_heading_rate_deg_s: float = L1_COMFORTABLE_HEADING_RATE_DEG_S,
        critical_heading_rate_deg_s: float = L1_CRITICAL_HEADING_RATE_DEG_S,
        comfortable_angular_jerk_deg_s2: float = L1_COMFORTABLE_ANGULAR_JERK_DEG_S2,
        critical_angular_jerk_deg_s2: float = L1_CRITICAL_ANGULAR_JERK_DEG_S2,
        min_speed_mps: float = MIN_MOVING_SPEED_MPS,
        heading_rate_threshold_rad_s: float = 0.01,
    ):
        """
        Initialize smooth steering violation evaluator.

        Args:
            comfortable_heading_rate_deg_s: Maximum comfortable heading rate (°/s)
            critical_heading_rate_deg_s: Critical heading rate threshold (°/s)
            comfortable_angular_jerk_deg_s2: Maximum comfortable angular jerk (°/s²)
            critical_angular_jerk_deg_s2: Critical angular jerk threshold (°/s²)
            min_speed_mps: Minimum speed to evaluate (m/s)
            heading_rate_threshold_rad_s: Minimum heading rate to count as turning (rad/s)
        """
        # Convert degrees to radians for internal calculations
        self.comfortable_heading_rate = comfortable_heading_rate_deg_s * np.pi / 180.0
        self.critical_heading_rate = critical_heading_rate_deg_s * np.pi / 180.0
        self.comfortable_angular_jerk = comfortable_angular_jerk_deg_s2 * np.pi / 180.0
        self.critical_angular_jerk = critical_angular_jerk_deg_s2 * np.pi / 180.0

        # Store original degree values for measurements
        self.comfortable_heading_rate_deg = comfortable_heading_rate_deg_s
        self.comfortable_angular_jerk_deg = comfortable_angular_jerk_deg_s2

        self.min_speed = min_speed_mps
        self.heading_rate_threshold = heading_rate_threshold_rad_s

    def evaluate(
        self, ctx: ScenarioContext, app: ApplicabilityResult
    ) -> ViolationResult:
        """
        Evaluate smooth steering violations.

        Args:
            ctx: Scenario context with ego vehicle state
            app: Applicability result from detector

        Returns:
            ViolationResult with severity and measurements
        """
        n_frames = len(ctx.ego.yaw)

        # Calculate heading rate (yaw rate)
        heading_rate = np.zeros(n_frames)
        if n_frames >= 2:
            yaw_diff = np.diff(ctx.ego.yaw)
            # Normalize angle differences to [-pi, pi] to handle wrap-around
            yaw_diff = (yaw_diff + np.pi) % (2 * np.pi) - np.pi
            heading_rate[:-1] = yaw_diff / ctx.dt
            heading_rate[-1] = heading_rate[-2]

        # Calculate angular jerk (derivative of heading rate)
        angular_jerk = np.zeros(n_frames)
        if n_frames >= 3:
            angular_jerk[:-1] = np.diff(heading_rate) / ctx.dt
            angular_jerk[-1] = angular_jerk[-2]

        # Identify moving and turning frames
        moving_mask = ctx.ego.speed > self.min_speed
        turning_mask = (
            np.abs(heading_rate) > self.heading_rate_threshold
        ) & moving_mask
        turning_count = int(np.sum(turning_mask))

        # Get magnitudes for threshold comparison
        heading_rate_mag = np.abs(heading_rate)
        angular_jerk_mag = np.abs(angular_jerk)

        # Detect violations (only during turning)
        heading_rate_violation = (
            heading_rate_mag > self.comfortable_heading_rate
        ) & turning_mask
        angular_jerk_violation = (
            angular_jerk_mag > self.comfortable_angular_jerk
        ) & turning_mask

        # Combined violation mask
        violation_mask = heading_rate_violation | angular_jerk_violation
        violation_frames = int(np.sum(violation_mask))

        # Calculate excess heading rate and angular jerk
        excess_rate = np.maximum(0.0, heading_rate_mag - self.comfortable_heading_rate)
        excess_jerk = np.maximum(0.0, angular_jerk_mag - self.comfortable_angular_jerk)

        # Apply turning mask (only count during turning)
        excess_rate = excess_rate * turning_mask
        excess_jerk = excess_jerk * turning_mask

        # Calculate severity as integral of excess over time
        # Convert to degrees for more intuitive severity values
        severity_per_frame = (
            (excess_rate * 180.0 / np.pi) + (excess_jerk * 180.0 / np.pi)
        ) * ctx.dt
        severity = float(np.sum(severity_per_frame))

        # Normalize severity (capped at 1.0)
        norm_factor = RULE_NORM.get(self.rule_id, 1.0)
        normalized = min(1.0, severity / norm_factor) if norm_factor > 0 else 0.0

        # Create timeseries for violation tracking
        timeseries = violation_mask.astype(float)

        # Measurements
        measurements = {
            "max_heading_rate_deg_s": (
                float(np.max(heading_rate_mag[turning_mask]) * 180.0 / np.pi)
                if turning_count > 0
                else 0.0
            ),
            "max_angular_jerk_deg_s2": (
                float(np.max(angular_jerk_mag[turning_mask]) * 180.0 / np.pi)
                if turning_count > 0
                else 0.0
            ),
            "avg_heading_rate_deg_s": (
                float(np.mean(heading_rate_mag[turning_mask]) * 180.0 / np.pi)
                if turning_count > 0
                else 0.0
            ),
            "avg_angular_jerk_deg_s2": (
                float(np.mean(angular_jerk_mag[turning_mask]) * 180.0 / np.pi)
                if turning_count > 0
                else 0.0
            ),
            "heading_rate_violation_frames": int(np.sum(heading_rate_violation)),
            "angular_jerk_violation_frames": int(np.sum(angular_jerk_violation)),
            "total_violation_frames": violation_frames,
            "total_turning_frames": turning_count,
            "p_violation": (
                float(violation_frames / turning_count) if turning_count > 0 else 0.0
            ),
            "total_severity": severity,
            "comfortable_heading_rate_threshold_deg_s": self.comfortable_heading_rate_deg,
            "comfortable_angular_jerk_threshold_deg_s2": self.comfortable_angular_jerk_deg,
        }

        # Explanation
        explanation = []
        if violation_frames > 0:
            explanation.append(
                f"Harsh steering detected in {violation_frames}/{turning_count} turning frames. "
                f"Max heading rate: {measurements['max_heading_rate_deg_s']:.2f} °/s "
                f"(comfortable: {self.comfortable_heading_rate_deg:.2f} °/s). "
                f"Max angular jerk: {measurements['max_angular_jerk_deg_s2']:.2f} °/s² "
                f"(comfortable: {self.comfortable_angular_jerk_deg:.2f} °/s²). "
                f"Total severity: {severity:.2f}."
            )
        else:
            explanation.append(
                f"Smooth steering in all {turning_count} turning frames. "
                f"Max heading rate: {measurements['max_heading_rate_deg_s']:.2f} °/s "
                f"(comfortable: {self.comfortable_heading_rate_deg:.2f} °/s). "
                f"Max angular jerk: {measurements['max_angular_jerk_deg_s2']:.2f} °/s² "
                f"(comfortable: {self.comfortable_angular_jerk_deg:.2f} °/s²)."
            )

        # Confidence based on data quality
        confidence = 1.0 if turning_count >= 3 else 0.5

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
