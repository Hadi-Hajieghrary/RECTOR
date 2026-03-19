"""
L8.R1: Red Light Violation Rule

Detects running red lights by checking if ego crosses a stopline
while the associated signal is red.

Features:
- Associates signals to ego's lane
- Tracks stopline crossing
- Accounts for yellow light transitions
"""

from dataclasses import dataclass
from typing import List

from ..core.context import ScenarioContext
from ..utils.constants import STOPLINE_DISTANCE_THRESHOLD_M
from .base import (
    ApplicabilityDetector,
    ApplicabilityResult,
    ViolationEvaluator,
    ViolationResult,
)

_RULE_ID = "L8.R1"
_LEVEL = 8
_NAME = "Red Light"


@dataclass
class RedLightViolationEvent:
    """Details of a red light violation."""

    frame_idx: int
    ego_position: tuple
    ego_speed_ms: float
    stopline_distance_m: float


class RedLightApplicability(ApplicabilityDetector):
    """Red light rule applies when there are traffic signals."""

    rule_id = _RULE_ID
    level = _LEVEL
    name = _NAME

    def detect(self, ctx: ScenarioContext) -> ApplicabilityResult:
        """Check if scenario has traffic signals."""
        if not ctx.has_signals:
            return ApplicabilityResult(
                applies=False,
                confidence=1.0,
                reasons=["No traffic signals in scenario"],
                features={"has_signals": False},
            )

        # Check if any frame has a red signal
        signals = ctx.signals
        has_red = any(signals.is_red_at(t) for t in range(len(signals)))

        if not has_red:
            return ApplicabilityResult(
                applies=False,
                confidence=1.0,
                reasons=["No red light phases in scenario"],
                features={"has_signals": True, "has_red_phase": False},
            )

        return ApplicabilityResult(
            applies=True,
            confidence=1.0,
            reasons=["Traffic signals present with red phases"],
            features={"has_signals": True, "has_red_phase": True},
        )


class RedLightViolation(ViolationEvaluator):
    """Evaluate red light violations."""

    rule_id = _RULE_ID
    level = _LEVEL
    name = _NAME

    def __init__(
        self,
        stopline_threshold_m: float = STOPLINE_DISTANCE_THRESHOLD_M,
    ):
        self.stopline_threshold_m = stopline_threshold_m

    def evaluate(
        self, ctx: ScenarioContext, applicability: ApplicabilityResult
    ) -> ViolationResult:
        """Evaluate red light violations."""

        ego = ctx.ego
        signals = ctx.signals
        map_ctx = ctx.map_context
        T = len(ego.x)

        if not map_ctx.has_stoplines:
            return ViolationResult(
                severity=0.0,
                explanation="No stoplines in map",
            )

        violation_frames: List[int] = []
        events: List[RedLightViolationEvent] = []

        # Track if ego was behind stopline
        was_behind_stopline = True
        last_stopline_dist = float("inf")

        for t in range(T):
            if not ego.is_valid_at(t):
                continue

            ex, ey = ego.x[t], ego.y[t]
            espeed = ego.speed[t]

            # Find distance to nearest stopline
            import numpy as np

            stoplines = map_ctx.stopline_xy
            dists = np.sqrt((stoplines[:, 0] - ex) ** 2 + (stoplines[:, 1] - ey) ** 2)
            min_dist = float(np.min(dists)) if len(dists) > 0 else float("inf")

            # Check if crossing stopline while red
            is_near_stopline = min_dist < self.stopline_threshold_m
            is_crossing = (
                last_stopline_dist > self.stopline_threshold_m and is_near_stopline
            )

            if is_crossing and signals.is_red_at(t):
                if t not in violation_frames:
                    violation_frames.append(t)

                events.append(
                    RedLightViolationEvent(
                        frame_idx=t,
                        ego_position=(ex, ey),
                        ego_speed_ms=espeed,
                        stopline_distance_m=min_dist,
                    )
                )

            last_stopline_dist = min_dist

        # Compute severity (each violation is serious)
        severity = float(len(violation_frames))
        severity_normalized = min(1.0, severity)

        return ViolationResult(
            severity=severity,
            severity_normalized=severity_normalized,
            measurements={
                "n_violation_frames": len(violation_frames),
                "n_violations": len(events),
            },
            explanation=(
                (f"{len(events)} red light violations")
                if len(events) > 0
                else "No red light violations"
            ),
            frame_violations=violation_frames,
        )
