# waymo_rule_eval/rules/l5_school_zone_compliance.py
# -*- coding: utf-8 -*-
"""
L5.R4 - School Zone Compliance Rule

Detects speed limit violations in school zones during active hours.

Standards:
- School zone speed limit: 15-25 mph (6.7-11.2 m/s), default 20 mph (8.9 m/s)
- Active hours: 7:00 AM - 4:00 PM on weekdays
- Violation threshold: > 5 mph (2.2 m/s) over limit

Note: School zone data is not available in standard Waymo dataset.
This rule requires custom annotations to be applicable.
"""
from __future__ import annotations

import numpy as np

from ..core.context import ScenarioContext
from .base import (
    ApplicabilityDetector,
    ApplicabilityResult,
    FrameViolation,
    ViolationEvaluator,
    ViolationResult,
)

RULE_NORM = 20.0
DEFAULT_SCHOOL_ZONE_SPEED_LIMIT_MPS = 8.9  # 20 mph


class SchoolZoneComplianceApplicability(ApplicabilityDetector):
    """
    Detects when the ego vehicle is in or near a school zone.

    Note: School zone data not available in standard Waymo dataset.
    This rule will not be applicable unless custom annotations are added.
    """

    rule_id = "L5.R4"
    level = 5
    name = "School zone compliance"

    def __init__(
        self, school_zone_proximity_m: float = 100.0, min_time_in_zone_s: float = 1.0
    ):
        self.school_zone_proximity_m = school_zone_proximity_m
        self.min_time_in_zone_s = min_time_in_zone_s

    def detect(self, ctx: ScenarioContext) -> ApplicabilityResult:
        # Check for school zone data in map
        has_school_zones = (
            hasattr(ctx, "map_context")
            and ctx.map_context is not None
            and hasattr(ctx.map_context, "school_zone_xy")
            and ctx.map_context.school_zone_xy is not None
            and len(ctx.map_context.school_zone_xy) > 0
        )

        if not has_school_zones:
            return ApplicabilityResult(
                rule_id=self.rule_id,
                rule_level=self.level,
                name=self.name,
                applies=False,
                confidence=0.0,
                reasons=["No school zone data available"],
                features={"has_school_zone_data": False},
            )

        # Calculate proximity to school zones
        school_zones = ctx.map_context.school_zone_xy
        ego_x, ego_y = ctx.ego.x, ctx.ego.y
        n_frames = len(ego_x)
        dt = ctx.dt or 0.1

        school_zone_mask = np.zeros(n_frames, dtype=bool)
        for i in range(n_frames):
            if np.isnan(ego_x[i]) or np.isnan(ego_y[i]):
                continue
            ego_pos = np.array([ego_x[i], ego_y[i]])
            distances = np.linalg.norm(school_zones - ego_pos, axis=1)
            if np.min(distances) <= self.school_zone_proximity_m:
                school_zone_mask[i] = True

        # Check minimum time in zone
        time_in_zone = np.sum(school_zone_mask) * dt
        applies = time_in_zone >= self.min_time_in_zone_s

        return ApplicabilityResult(
            rule_id=self.rule_id,
            rule_level=self.level,
            name=self.name,
            applies=applies,
            confidence=0.9 if applies else 0.0,
            reasons=[f"Time in school zone: {time_in_zone:.1f}s"],
            features={
                "has_school_zone_data": True,
                "school_zone_mask": school_zone_mask,
                "time_in_zone": time_in_zone,
            },
        )


class SchoolZoneComplianceEvaluator(ViolationEvaluator):
    """
    Evaluates speed violations in school zones.
    """

    rule_id = "L5.R4"
    name = "School zone compliance"

    def __init__(
        self,
        speed_limit_mps: float = DEFAULT_SCHOOL_ZONE_SPEED_LIMIT_MPS,
        violation_threshold_mps: float = 2.2,  # 5 mph over limit
    ):
        self.speed_limit = speed_limit_mps
        self.violation_threshold = violation_threshold_mps

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

        school_zone_mask = app.features.get("school_zone_mask", np.array([]))
        ego_speed = ctx.ego.speed

        frame_violations = []
        speed_violations = 0
        max_excess_speed = 0.0

        for t in range(len(ego_speed)):
            if t >= len(school_zone_mask):
                break
            if not school_zone_mask[t]:
                continue

            speed = ego_speed[t]
            excess = speed - self.speed_limit

            if excess > self.violation_threshold:
                severity = min(1.0, excess / 10.0)
                speed_violations += 1
                max_excess_speed = max(max_excess_speed, excess)

                frame_violations.append(
                    FrameViolation(
                        frame_idx=t,
                        severity=severity,
                        details={
                            "speed": float(speed),
                            "excess_speed": float(excess),
                            "limit": float(self.speed_limit),
                        },
                    )
                )

        total_severity = speed_violations * 0.5
        normalized = min(1.0, total_severity / RULE_NORM)

        explanation = []
        if speed_violations > 0:
            explanation.append(f"Speed violations: {speed_violations}")
            explanation.append(f"Max excess: {max_excess_speed:.1f} m/s")
        else:
            explanation.append("No violations in school zone")

        return ViolationResult(
            rule_id=self.rule_id,
            name=self.name,
            severity=total_severity,
            severity_normalized=normalized,
            measurements={
                "speed_violations": speed_violations,
                "max_excess_speed": max_excess_speed,
            },
            explanation="; ".join(explanation),
            frame_violations=frame_violations,
            confidence=app.confidence,
        )
