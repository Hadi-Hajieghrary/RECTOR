"""
L1.R2: Smooth Braking Rule

Checks for harsh braking (excessive deceleration).

Features:
- Computes acceleration via finite differences
- Flags comfort and emergency braking thresholds
- Tracks duration of harsh braking
"""

from dataclasses import dataclass
from typing import List

from ..core.context import ScenarioContext
from ..utils.constants import (
    COMFORT_DECELERATION_LIMIT,
    DEFAULT_DT_S,
    EMERGENCY_DECELERATION_LIMIT,
    HARSH_BRAKING_THRESHOLD,
)
from .base import (
    ApplicabilityDetector,
    ApplicabilityResult,
    ViolationEvaluator,
    ViolationResult,
)

_RULE_ID = "L1.R2"
_LEVEL = 1
_NAME = "Smooth Braking"


@dataclass
class BrakingEvent:
    """Details of a harsh braking event."""

    frame_idx: int
    deceleration_ms2: float
    ego_speed_ms: float
    is_emergency: bool


class SmoothBrakingApplicability(ApplicabilityDetector):
    """Smooth braking always applies when ego is moving."""

    rule_id = _RULE_ID
    level = _LEVEL
    name = _NAME

    def detect(self, ctx: ScenarioContext) -> ApplicabilityResult:
        """Check if ego is moving in any frame."""
        import numpy as np

        ego = ctx.ego
        max_speed = float(np.nanmax(ego.speed))

        if max_speed < 1.0:  # ~2 mph
            return ApplicabilityResult(
                applies=False,
                confidence=1.0,
                reasons=["Ego barely moves in scenario"],
                features={"max_speed_ms": max_speed},
            )

        return ApplicabilityResult(
            applies=True,
            confidence=1.0,
            reasons=[f"Ego reaches {max_speed:.1f} m/s"],
            features={"max_speed_ms": max_speed},
        )


class SmoothBrakingViolation(ViolationEvaluator):
    """Evaluate braking smoothness."""

    rule_id = _RULE_ID
    level = _LEVEL
    name = _NAME

    def __init__(
        self,
        comfort_limit: float = COMFORT_DECELERATION_LIMIT,
        harsh_threshold: float = HARSH_BRAKING_THRESHOLD,
        emergency_threshold: float = EMERGENCY_DECELERATION_LIMIT,
    ):
        self.comfort_limit = comfort_limit
        self.harsh_threshold = harsh_threshold
        self.emergency_threshold = emergency_threshold

    def evaluate(
        self, ctx: ScenarioContext, applicability: ApplicabilityResult
    ) -> ViolationResult:
        """Evaluate braking smoothness."""

        ego = ctx.ego
        dt = ctx.dt
        T = len(ego.x)

        # Get acceleration
        acceleration = ego.get_acceleration(dt)

        violation_frames: List[int] = []
        events: List[BrakingEvent] = []
        total_harsh = 0.0
        max_decel = 0.0
        n_emergency = 0

        for t in range(T):
            if not ego.is_valid_at(t):
                continue

            acc = acceleration[t]
            if np.isnan(acc):
                continue
            espeed = ego.speed[t]

            # Deceleration is negative acceleration
            decel = -acc

            if decel > self.harsh_threshold:
                if t not in violation_frames:
                    violation_frames.append(t)

                total_harsh += (decel - self.comfort_limit) * dt
                max_decel = max(max_decel, decel)

                is_emergency = decel > self.emergency_threshold
                if is_emergency:
                    n_emergency += 1

                events.append(
                    BrakingEvent(
                        frame_idx=t,
                        deceleration_ms2=decel,
                        ego_speed_ms=espeed,
                        is_emergency=is_emergency,
                    )
                )

        # Compute severity
        severity = total_harsh
        severity_normalized = min(1.0, max_decel / self.emergency_threshold)

        return ViolationResult(
            severity=severity,
            severity_normalized=severity_normalized,
            measurements={
                "n_violation_frames": len(violation_frames),
                "n_harsh_events": len(events),
                "n_emergency_brakes": n_emergency,
                "max_deceleration_ms2": max_decel,
                "total_harsh_ms2": total_harsh,
            },
            explanation=(
                (f"{len(events)} harsh braking events, " f"max {max_decel:.2f} m/s²")
                if len(events) > 0
                else "Smooth braking"
            ),
            frame_violations=violation_frames,
        )
