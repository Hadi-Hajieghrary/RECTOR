"""
L7.R4: Speed Limit Rule

Detects when a vehicle exceeds the posted speed limit.
Severity is based on the magnitude and duration of speeding.

Severity Calculation:
- severity = ∫(excess_speed) dt (m/s · s = m)
- Normalized by RULE_NORM for consistent scoring
"""

from __future__ import annotations

import logging
import os

import numpy as np

from ..core.context import ScenarioContext
from ..utils.constants import L7_SPEED_TOLERANCE_MPS
from .base import (
    ApplicabilityDetector,
    ApplicabilityResult,
    ViolationEvaluator,
    ViolationResult,
)

log = logging.getLogger(__name__)
RULE_NORM = {"L7.R4": 50.0}


def _default_speed_limit_mps() -> float:
    """Get default speed limit from environment if set, else use urban default."""
    val = os.getenv("WAYMO_DEFAULT_SPEED_LIMIT_MPH", "").strip()
    if val:
        try:
            return float(val) * 0.44704  # mph to m/s
        except (ValueError, TypeError):
            pass
    # Default to 35 mph (15.6 m/s) for urban driving - common Waymo scenario
    # This allows the rule to apply and detect obvious speeding (e.g., >50 mph)
    return 35.0 * 0.44704  # 15.6 m/s


class SpeedLimitApplicability(ApplicabilityDetector):
    """
    Checks if speed limit rule is applicable.

    Applicable when:
    - Speed limit data is available (from map or environment default)
    - Vehicle is moving
    """

    rule_id = "L7.R4"
    level = 7
    name = "Speed limit"

    def detect(self, ctx: ScenarioContext) -> ApplicabilityResult:
        """
        Check if speed limit evaluation applies.
        """
        # Get speed limit from map
        speed_limit = None
        speed_limit_mask = None

        if hasattr(ctx, "map") and ctx.map is not None:
            speed_limit = getattr(ctx.map, "speed_limit", None)
            speed_limit_mask = getattr(ctx.map, "speed_limit_mask", None)

        # Calculate coverage
        coverage = 0.0
        if speed_limit_mask is not None and len(speed_limit_mask) > 0:
            coverage = float(np.mean(speed_limit_mask))

        # If no speed limit or low coverage, try default
        if speed_limit is None or coverage <= 0.5:
            default_limit = _default_speed_limit_mps()
            if default_limit is not None:
                speed_limit = np.full_like(ctx.ego.speed, default_limit)
                speed_limit_mask = np.ones_like(ctx.ego.speed, dtype=bool)
                coverage = 1.0
                log.debug("Using default speed limit from environment")

        applies = coverage > 0.5

        features = {
            "v": ctx.ego.speed,
            "sl": speed_limit,
            "mask": speed_limit_mask,
            "dt": ctx.dt,
        }

        reason = f"Speed limit known for {coverage:.2%} of frames"

        return ApplicabilityResult(
            rule_id=self.rule_id,
            rule_level=self.level,
            name=self.name,
            applies=applies,
            confidence=coverage,
            reasons=[reason],
            features=features,
        )


class SpeedLimitViolation(ViolationEvaluator):
    """
    Evaluates speed limit violations.

    A violation occurs when:
    - Vehicle speed exceeds speed limit + tolerance

    Severity is based on:
    - Integral of excess speed over time
    """

    rule_id = "L7.R4"
    level = 7
    name = "Speed limit"

    def __init__(self, tolerance_mps: float = L7_SPEED_TOLERANCE_MPS):
        """
        Initialize the evaluator.

        Args:
            tolerance_mps: Speed tolerance before violation (m/s)
        """
        self.tolerance = tolerance_mps

    def evaluate(
        self, ctx: ScenarioContext, app: ApplicabilityResult
    ) -> ViolationResult:
        """
        Evaluate speed limit violations.
        """
        if not app.applies:
            return ViolationResult(
                rule_id=self.rule_id,
                name=self.name,
                severity=0.0,
                severity_normalized=0.0,
                timeseries=None,
                measurements={},
                explanation=["Speed limit not applicable"],
                confidence=0.0,
            )

        v = app.features["v"]
        sl = app.features["sl"]
        mask = app.features["mask"]
        dt = app.features["dt"]

        # Calculate excess speed
        excess = np.zeros_like(v)
        known = mask if mask is not None else np.ones_like(v, dtype=bool)

        if sl is not None:
            excess[known] = np.maximum(0.0, v[known] - sl[known] - self.tolerance)

        # Calculate severity: integral of excess speed over time
        severity = float(np.sum(excess) * dt)

        # Normalize
        norm_factor = RULE_NORM.get(self.rule_id, 50.0)
        normalized = min(severity / norm_factor, 1.0)

        # Measurements
        speeding_mask = excess > 0
        p_speeding = float(np.mean(speeding_mask))

        measurements = {
            "avg_excess_mps": (
                float(np.mean(excess[excess > 0])) if np.any(speeding_mask) else 0.0
            ),
            "max_excess_mps": float(np.max(excess)) if len(excess) > 0 else 0.0,
            "p_speeding": p_speeding,
            "total_severity": severity,
        }

        explanation = ["Speeding" if severity > 0 else "Compliant"]
        if severity > 0:
            explanation.append(
                f"Exceeded speed limit in {p_speeding:.1%} of frames, "
                f"max excess: {measurements['max_excess_mps']:.1f} m/s"
            )

        return ViolationResult(
            rule_id=self.rule_id,
            name=self.name,
            severity=severity,
            severity_normalized=normalized,
            timeseries=excess,
            measurements=measurements,
            explanation=explanation,
            confidence=app.confidence,
        )
