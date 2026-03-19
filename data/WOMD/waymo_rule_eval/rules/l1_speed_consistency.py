"""
L1.R4 Speed Consistency Rule

Evaluates whether the ego vehicle maintains consistent speed without excessive
variations or oscillations that could cause passenger discomfort.

Standards:
    - Comfortable speed variance: ±2.0 m/s over 2-second rolling window
    - Critical speed variance: ±4.0 m/s over 2-second rolling window
    - Oscillation detection: Multiple acceleration sign changes
    - Severity normalization: 20.0 (variance accumulates over time)
"""

import logging
from typing import Any, Dict

import numpy as np

from ..core.context import ScenarioContext
from ..utils.constants import (
    L1_COMFORTABLE_SPEED_VARIANCE_MPS,
    L1_CRITICAL_SPEED_VARIANCE_MPS,
    L1_OSCILLATION_THRESHOLD,
    L1_SPEED_WINDOW_DURATION_S,
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
RULE_NORM = {"L1.R4": 20.0}


class SpeedConsistencyApplicability(ApplicabilityDetector):
    """
    Detects whether the speed consistency rule applies to the scenario.

    The rule applies when:
    - Vehicle is moving (speed > threshold)
    - Sufficient data for rolling window analysis (>= 2 seconds)
    """

    rule_id = "L1.R4"
    level = 1
    name = "Speed consistency"

    def __init__(
        self,
        min_speed_mps: float = MIN_MOVING_SPEED_MPS,
        min_frames: int = 3,
        window_duration_s: float = L1_SPEED_WINDOW_DURATION_S,
    ):
        """
        Initialize the speed consistency applicability detector.

        Args:
            min_speed_mps: Minimum speed to consider vehicle as moving
            min_frames: Minimum frames required for applicability
            window_duration_s: Duration of rolling window for analysis
        """
        self.min_speed_mps = min_speed_mps
        self.min_frames = min_frames
        self.window_duration_s = window_duration_s

    def detect(self, ctx: ScenarioContext) -> ApplicabilityResult:
        """
        Detect if speed consistency evaluation applies.

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
                    reasons=["Insufficient frames for speed consistency analysis"],
                    features={"total_frames": n_frames},
                )

            # Identify moving frames (handle NaN speed values)
            valid_speed = ~np.isnan(ctx.ego.speed)
            valid_speed_count = int(np.sum(valid_speed))

            if valid_speed_count < self.min_frames:
                return ApplicabilityResult(
                    rule_id=self.rule_id,
                    rule_level=self.level,
                    name=self.name,
                    applies=False,
                    confidence=1.0,
                    reasons=[f"Insufficient valid speed data ({valid_speed_count})"],
                    features={"valid_speed_frames": valid_speed_count},
                )

            moving_mask = valid_speed & (ctx.ego.speed >= self.min_speed_mps)
            moving_frames = int(np.sum(moving_mask))

            # Check for sufficient moving data
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

            # Check for sufficient window size
            window_frames = int(self.window_duration_s / ctx.dt)
            if n_frames < window_frames:
                return ApplicabilityResult(
                    rule_id=self.rule_id,
                    rule_level=self.level,
                    name=self.name,
                    applies=False,
                    confidence=0.8,
                    reasons=[
                        f"Insufficient data for {self.window_duration_s}s rolling window"
                    ],
                    features={
                        "total_frames": n_frames,
                        "required_frames": window_frames,
                        "window_duration_s": self.window_duration_s,
                    },
                )

            # Rule applies - vehicle is moving with sufficient data
            confidence = min(1.0, moving_frames / n_frames)

            return ApplicabilityResult(
                rule_id=self.rule_id,
                rule_level=self.level,
                name=self.name,
                applies=True,
                confidence=confidence,
                reasons=[
                    "Vehicle is moving with sufficient data for consistency analysis"
                ],
                features={
                    "moving_frames": moving_frames,
                    "total_frames": n_frames,
                    "window_duration_s": self.window_duration_s,
                    "window_frames": window_frames,
                    "min_speed_mps": self.min_speed_mps,
                },
            )

        except Exception as e:
            logger.error(f"Error in SpeedConsistencyApplicability.detect: {e}")
            return ApplicabilityResult(
                rule_id=self.rule_id,
                rule_level=self.level,
                name=self.name,
                applies=False,
                confidence=0.0,
                reasons=[f"Error during detection: {str(e)}"],
                features={},
            )


class SpeedConsistencyViolation(ViolationEvaluator):
    """
    Evaluates speed consistency violations.

    Detects excessive speed variations over rolling windows that indicate
    inconsistent or oscillatory driving behavior causing passenger discomfort.
    """

    rule_id = "L1.R4"
    level = 1
    name = "Speed consistency"

    def __init__(
        self,
        comfortable_variance_mps: float = L1_COMFORTABLE_SPEED_VARIANCE_MPS,
        critical_variance_mps: float = L1_CRITICAL_SPEED_VARIANCE_MPS,
        window_duration_s: float = L1_SPEED_WINDOW_DURATION_S,
        oscillation_threshold: int = L1_OSCILLATION_THRESHOLD,
        min_speed_mps: float = MIN_MOVING_SPEED_MPS,
    ):
        """
        Initialize the speed consistency violation evaluator.

        Args:
            comfortable_variance_mps: Comfortable speed variance threshold
            critical_variance_mps: Critical speed variance threshold
            window_duration_s: Rolling window duration for variance calculation
            oscillation_threshold: Number of sign changes indicating oscillation
            min_speed_mps: Minimum speed to evaluate (filter stationary periods)
        """
        self.comfortable_variance_mps = comfortable_variance_mps
        self.critical_variance_mps = critical_variance_mps
        self.window_duration_s = window_duration_s
        self.oscillation_threshold = oscillation_threshold
        self.min_speed_mps = min_speed_mps

    def evaluate(
        self, ctx: ScenarioContext, app: ApplicabilityResult
    ) -> ViolationResult:
        """
        Evaluate speed consistency violations.

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

            speeds = ctx.ego.speed
            n_frames = len(speeds)
            dt = ctx.dt

            # Calculate rolling window size
            window_frames = int(self.window_duration_s / dt)

            # Identify moving frames
            moving_mask = speeds >= self.min_speed_mps
            moving_count = int(np.sum(moving_mask))

            # Calculate speed variance over rolling windows
            variances = np.zeros(n_frames)
            for i in range(n_frames):
                start = max(0, i - window_frames + 1)
                end = i + 1
                window_speeds = speeds[start:end]
                if len(window_speeds) >= 2:
                    variances[i] = np.std(window_speeds)

            # Calculate acceleration for oscillation detection
            acceleration = np.zeros(n_frames)
            if n_frames > 1:
                acceleration[1:] = np.diff(speeds) / dt

            # Detect oscillations (sign changes in acceleration)
            oscillation_count = 0
            if n_frames > 2:
                accel_signs = np.sign(
                    acceleration[1:]
                )  # Skip first frame (no accel yet)
                sign_changes = np.diff(accel_signs) != 0
                oscillation_count = int(np.sum(sign_changes))

            # Detect violations: high variance during moving periods
            variance_violations = (
                variances > self.comfortable_variance_mps
            ) & moving_mask

            # Calculate excess variance (for severity)
            excess_variance = np.maximum(0, variances - self.comfortable_variance_mps)
            excess_variance_moving = excess_variance * moving_mask

            # Check for violations
            has_variance_violation = bool(np.any(variance_violations))
            has_oscillation_violation = oscillation_count >= self.oscillation_threshold
            has_violation = has_variance_violation or has_oscillation_violation

            # Calculate severity: integral of excess variance over time
            severity = float(np.sum(excess_variance_moving) * dt)

            # Add penalty for oscillations
            if has_oscillation_violation:
                oscillation_penalty = (
                    oscillation_count - self.oscillation_threshold
                ) * 0.5
                severity += oscillation_penalty

            # Normalize severity (capped at 1.0)
            normalized = (
                min(1.0, severity / RULE_NORM[self.rule_id]) if severity > 0 else 0.0
            )

            # Prepare measurements
            violation_frames = int(np.sum(variance_violations))
            p_violation = violation_frames / moving_count if moving_count > 0 else 0.0

            measurements = {
                "max_variance_mps": float(np.max(variances)),
                "avg_variance_mps": (
                    float(np.mean(variances[moving_mask])) if moving_count > 0 else 0.0
                ),
                "max_variance_moving_mps": (
                    float(np.max(variances[moving_mask])) if moving_count > 0 else 0.0
                ),
                "variance_violation_frames": violation_frames,
                "oscillation_count": oscillation_count,
                "total_moving_frames": moving_count,
                "total_frames": n_frames,
                "p_violation": p_violation,
                "total_severity": severity,
                "comfortable_variance_threshold_mps": self.comfortable_variance_mps,
                "critical_variance_threshold_mps": self.critical_variance_mps,
                "window_duration_s": self.window_duration_s,
            }

            # Prepare timeseries
            timeseries = {
                "variance_mps": variances.tolist(),
                "excess_variance_mps": excess_variance_moving.tolist(),
                "moving_mask": moving_mask.astype(int).tolist(),
                "violation_mask": variance_violations.astype(int).tolist(),
            }

            # Prepare explanation
            if not has_violation:
                explanation = (
                    f"Speed consistency maintained. "
                    f"Max variance: {measurements['max_variance_mps']:.2f} m/s "
                    f"(threshold: {self.comfortable_variance_mps:.1f} m/s), "
                    f"Oscillations: {oscillation_count} "
                    f"(threshold: {self.oscillation_threshold})"
                )
            else:
                violation_parts = []
                if has_variance_violation:
                    violation_parts.append(
                        f"high variance ({measurements['max_variance_mps']:.2f} m/s > "
                        f"{self.comfortable_variance_mps:.1f} m/s)"
                    )
                if has_oscillation_violation:
                    violation_parts.append(
                        f"excessive oscillations ({oscillation_count} > "
                        f"{self.oscillation_threshold})"
                    )

                explanation = (
                    f"Speed consistency violation detected: {' and '.join(violation_parts)}. "
                    f"Violation in {violation_frames}/{moving_count} moving frames "
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
            logger.error(f"Error in SpeedConsistencyViolation.evaluate: {e}")
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
