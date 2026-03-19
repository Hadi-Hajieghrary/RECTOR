# waymo_rule_eval/rules/l5_parking_violation.py
# -*- coding: utf-8 -*-
"""
L5.R3 - Parking Violation Rule

Detects illegal parking or stopping in restricted zones.

Violation Types:
  1. Illegal stationary periods (stopping where prohibited)
  2. Roadway obstruction (stopping in travel lanes)
  3. Extended stationary duration in restricted areas

Standards:
  - Stationary threshold: speed < 0.5 m/s for >= 3.0 seconds
  - Roadway parking: lateral offset > 2.5m from lane center
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from ..core.context import ScenarioContext
from .base import (
    ApplicabilityDetector,
    ApplicabilityResult,
    FrameViolation,
    ViolationEvaluator,
    ViolationResult,
)

RULE_NORM = 30.0


def contiguous_spans(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Find contiguous True spans in a boolean mask."""
    spans = []
    in_span = False
    start = 0
    for i, val in enumerate(mask):
        if val and not in_span:
            start = i
            in_span = True
        elif not val and in_span:
            spans.append((start, i))
            in_span = False
    if in_span:
        spans.append((start, len(mask)))
    return spans


class ParkingViolationApplicability(ApplicabilityDetector):
    """
    Detect if parking violation checking is applicable.

    Applies when:
    - Vehicle is stationary or moving very slowly
    - Sustained stationary periods detected
    """

    rule_id = "L5.R3"
    level = 5
    name = "Parking violation"

    def __init__(
        self,
        stationary_speed_threshold_mps: float = 0.5,
        min_stationary_duration_s: float = 3.0,
    ):
        self.stationary_speed_threshold = stationary_speed_threshold_mps
        self.min_stationary_duration = min_stationary_duration_s

    def detect(self, ctx: ScenarioContext) -> ApplicabilityResult:
        ego_speed = ctx.ego.speed
        dt = ctx.dt or 0.1

        # Detect stationary periods
        stationary_mask = ego_speed < self.stationary_speed_threshold
        stationary_spans = contiguous_spans(stationary_mask)

        # Check if any stationary period exceeds minimum duration
        has_parking_period = False
        max_stationary_duration = 0.0
        total_stationary_frames = 0

        for start, end in stationary_spans:
            duration = (end - start) * dt
            if duration >= self.min_stationary_duration:
                has_parking_period = True
                total_stationary_frames += end - start
                max_stationary_duration = max(max_stationary_duration, duration)

        if not has_parking_period:
            return ApplicabilityResult(
                rule_id=self.rule_id,
                rule_level=self.level,
                name=self.name,
                applies=False,
                confidence=0.0,
                reasons=["No sustained stationary periods detected"],
                features={},
            )

        # Calculate confidence based on duration
        duration_factor = min(1.0, max_stationary_duration / 10.0)
        confidence = 0.5 + 0.5 * duration_factor

        return ApplicabilityResult(
            rule_id=self.rule_id,
            rule_level=self.level,
            name=self.name,
            applies=True,
            confidence=confidence,
            reasons=[f"Stationary for {max_stationary_duration:.1f}s"],
            features={
                "stationary_spans": stationary_spans,
                "max_stationary_duration": max_stationary_duration,
                "total_stationary_frames": total_stationary_frames,
            },
        )


class ParkingViolationEvaluator(ViolationEvaluator):
    """
    Evaluates parking violations based on location and duration.

    Notes:
    - In standard Waymo dataset, no parking zone annotations are available
    - This evaluator uses heuristics based on lateral position and speed
    """

    rule_id = "L5.R3"
    name = "Parking violation"

    def __init__(
        self,
        lane_center_tolerance_m: float = 2.5,
        duration_severity_factor: float = 0.1,
    ):
        self.lane_center_tolerance = lane_center_tolerance_m
        self.duration_severity_factor = duration_severity_factor

    def evaluate(
        self, ctx: ScenarioContext, app: ApplicabilityResult
    ) -> ViolationResult:
        if not app.applies:
            return ViolationResult(
                rule_id=self.rule_id,
                name=self.name,
                severity=0.0,
                severity_normalized=0.0,
                measurements={},
                explanation="Rule not applicable",
                frame_violations=[],
                confidence=0.0,
            )

        stationary_spans = app.features.get("stationary_spans", [])
        max_duration = app.features.get("max_stationary_duration", 0.0)
        dt = ctx.dt or 0.1

        frame_violations = []
        total_violation_duration = 0.0
        in_lane_stops = 0

        ego_x, ego_y = ctx.ego.x, ctx.ego.y

        # Check if stopped in lane (near lane center)
        for start, end in stationary_spans:
            duration = (end - start) * dt

            # Check proximity to lane center if available
            in_lane = False
            if (
                hasattr(ctx, "map_context")
                and ctx.map_context is not None
                and hasattr(ctx.map_context, "lane_center_xy")
                and ctx.map_context.lane_center_xy is not None
                and len(ctx.map_context.lane_center_xy) > 0
            ):

                lane_pts = ctx.map_context.lane_center_xy
                mid_frame = (start + end) // 2
                ego_pos = np.array([ego_x[mid_frame], ego_y[mid_frame]])

                dists = np.linalg.norm(lane_pts - ego_pos, axis=1)
                min_dist = np.min(dists)

                if min_dist < self.lane_center_tolerance:
                    in_lane = True
                    in_lane_stops += 1

            # Calculate severity based on duration and location
            severity = duration * self.duration_severity_factor
            if in_lane:
                severity *= 1.5  # Higher severity for blocking lane

            if severity > 0.1:
                total_violation_duration += duration
                for t in range(start, end):
                    frame_violations.append(
                        FrameViolation(
                            frame_idx=t,
                            severity=severity / (end - start),
                            details={
                                "in_lane": in_lane,
                                "duration": duration,
                            },
                        )
                    )

        total_severity = total_violation_duration * self.duration_severity_factor
        normalized = min(1.0, total_severity / RULE_NORM)

        explanation = []
        if in_lane_stops > 0:
            explanation.append(f"Stopped in lane {in_lane_stops} time(s)")
        explanation.append(f"Max stationary duration: {max_duration:.1f}s")

        return ViolationResult(
            rule_id=self.rule_id,
            name=self.name,
            severity=total_severity,
            severity_normalized=normalized,
            measurements={
                "total_violation_duration": total_violation_duration,
                "in_lane_stops": in_lane_stops,
                "max_stationary_duration": max_duration,
            },
            explanation="; ".join(explanation),
            frame_violations=frame_violations,
            confidence=app.confidence,
        )
