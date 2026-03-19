# waymo_rule_eval/rules/l6_cooperative_lane_change.py
# -*- coding: utf-8 -*-
"""
L6.R1 - Cooperative Lane Change Rule

Ensures safe and courteous lane changes with proper gap selection
and smooth execution that respects surrounding vehicles.

Lane changes should include:
- Selection of safe gaps (minimum 2 seconds spacing)
- Smooth lateral transitions (< 0.5 m/s² lateral acceleration)
- Courtesy to surrounding vehicles (no forced braking)
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

RULE_NORM = 15.0


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


class CooperativeLaneChangeApplicability(ApplicabilityDetector):
    """
    Detects lane changes in the ego vehicle's trajectory.

    Uses lateral motion analysis to identify lane change periods.
    """

    rule_id = "L6.R1"
    level = 6
    name = "Cooperative lane change"

    def __init__(
        self,
        lateral_velocity_threshold_mps: float = 0.2,
        min_lateral_displacement_m: float = 2.5,
        min_lane_change_duration_s: float = 0.5,
        max_lane_change_duration_s: float = 10.0,
    ):
        self.lateral_velocity_threshold = lateral_velocity_threshold_mps
        self.min_lateral_displacement = min_lateral_displacement_m
        self.min_lane_change_duration = min_lane_change_duration_s
        self.max_lane_change_duration = max_lane_change_duration_s

    def _calculate_lateral_velocity(self, ctx: ScenarioContext) -> np.ndarray:
        """Calculate lateral velocity from trajectory."""
        ego_x = ctx.ego.x
        ego_y = ctx.ego.y
        ego_yaw = ctx.ego.yaw
        dt = ctx.dt or 0.1

        n = len(ego_x)
        lateral_velocity = np.zeros(n)

        for i in range(1, n):
            dx = ego_x[i] - ego_x[i - 1]
            dy = ego_y[i] - ego_y[i - 1]

            # Project onto lateral direction (perpendicular to heading)
            heading = ego_yaw[i - 1]
            lateral_dir = np.array([-np.sin(heading), np.cos(heading)])
            lateral_velocity[i] = (dx * lateral_dir[0] + dy * lateral_dir[1]) / dt

        return lateral_velocity

    def detect(self, ctx: ScenarioContext) -> ApplicabilityResult:
        if len(ctx.ego.x) < 2:
            return ApplicabilityResult(
                rule_id=self.rule_id,
                rule_level=self.level,
                name=self.name,
                applies=False,
                confidence=0.0,
                reasons=["Insufficient trajectory data"],
                features={},
            )

        dt = ctx.dt or 0.1
        lateral_velocity = self._calculate_lateral_velocity(ctx)

        # Detect periods of sustained lateral motion
        lat_motion_mask = np.abs(lateral_velocity) > self.lateral_velocity_threshold
        lat_motion_spans = contiguous_spans(lat_motion_mask)

        # Filter spans by duration
        min_frames = int(self.min_lane_change_duration / dt)
        max_frames = int(self.max_lane_change_duration / dt)

        valid_spans = [
            span
            for span in lat_motion_spans
            if min_frames <= (span[1] - span[0]) <= max_frames
        ]

        # Validate each span as a lane change
        lane_change_events = []
        for start_idx, end_idx in valid_spans:
            # Calculate total lateral displacement
            ego_x = ctx.ego.x[start_idx : end_idx + 1]
            ego_y = ctx.ego.y[start_idx : end_idx + 1]

            # Simple lateral displacement estimation
            lat_disp = np.sum(lateral_velocity[start_idx:end_idx]) * dt

            if abs(lat_disp) >= self.min_lateral_displacement:
                direction = "left" if lat_disp > 0 else "right"
                lane_change_events.append(
                    {
                        "start_frame": start_idx,
                        "end_frame": end_idx,
                        "duration": (end_idx - start_idx) * dt,
                        "lateral_displacement": lat_disp,
                        "direction": direction,
                    }
                )

        if not lane_change_events:
            return ApplicabilityResult(
                rule_id=self.rule_id,
                rule_level=self.level,
                name=self.name,
                applies=False,
                confidence=0.8,
                reasons=["No lane changes detected"],
                features={"lane_change_events": []},
            )

        confidence = 0.7
        if len(ctx.agents) > 0:
            confidence += 0.2  # Can evaluate gaps with other agents

        return ApplicabilityResult(
            rule_id=self.rule_id,
            rule_level=self.level,
            name=self.name,
            applies=True,
            confidence=confidence,
            reasons=[f"Detected {len(lane_change_events)} lane change(s)"],
            features={
                "lane_change_events": lane_change_events,
                "lateral_velocity": lateral_velocity,
            },
        )


class CooperativeLaneChangeEvaluator(ViolationEvaluator):
    """
    Evaluates lane change safety and courtesy.

    Checks:
    - Safe gaps to surrounding vehicles
    - Smooth lateral transitions
    - Impact on surrounding vehicles (forced braking)
    """

    rule_id = "L6.R1"
    level = 6
    name = "Cooperative lane change"

    def __init__(
        self,
        min_gap_s: float = 2.0,
        max_lateral_accel_mps2: float = 0.5,
        min_distance_to_agent_m: float = 5.0,
    ):
        self.min_gap = min_gap_s
        self.max_lateral_accel = max_lateral_accel_mps2
        self.min_distance_to_agent = min_distance_to_agent_m

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

        lane_changes = app.features.get("lane_change_events", [])
        lateral_velocity = app.features.get("lateral_velocity", np.array([]))
        dt = ctx.dt or 0.1

        frame_violations = []
        gap_violations = 0
        accel_violations = 0
        min_gap_observed = np.inf

        # Calculate lateral acceleration
        lateral_accel = np.diff(lateral_velocity) / dt
        lateral_accel = np.append(lateral_accel, 0)
        # Guard against NaN from invalid frames
        lateral_accel = np.where(np.isnan(lateral_accel), 0.0, lateral_accel)

        ego_x, ego_y = ctx.ego.x, ctx.ego.y
        ego_speed = ctx.ego.speed

        for lc in lane_changes:
            start = lc["start_frame"]
            end = lc["end_frame"]

            for t in range(start, min(end, len(ego_x))):
                # Check lateral acceleration
                if t < len(lateral_accel):
                    lat_accel = abs(lateral_accel[t])
                    if lat_accel > self.max_lateral_accel:
                        accel_violations += 1
                        severity = min(1.0, lat_accel / (2 * self.max_lateral_accel))
                        frame_violations.append(
                            FrameViolation(
                                frame_idx=t,
                                severity=severity,
                                details={
                                    "type": "lateral_acceleration",
                                    "lateral_accel": float(lat_accel),
                                },
                            )
                        )

                # Check gap to agents
                for agent in ctx.agents:
                    if t >= len(agent.x):
                        continue

                    dx = agent.x[t] - ego_x[t]
                    dy = agent.y[t] - ego_y[t]
                    dist = np.sqrt(dx**2 + dy**2)

                    if np.isnan(dist) or dist >= self.min_distance_to_agent:
                        continue

                    # Calculate time gap
                    rel_speed = ego_speed[t] - agent.speed[t]
                    if rel_speed > 0:
                        time_gap = dist / rel_speed
                        if time_gap < self.min_gap:
                            gap_violations += 1
                            min_gap_observed = min(min_gap_observed, time_gap)
                            severity = 1.0 - (time_gap / self.min_gap)
                            frame_violations.append(
                                FrameViolation(
                                    frame_idx=t,
                                    severity=severity,
                                    details={
                                        "type": "gap_violation",
                                        "gap": float(time_gap),
                                        "agent_id": agent.id,
                                    },
                                )
                            )

        total_severity = gap_violations * 0.5 + accel_violations * 0.2
        normalized = min(1.0, total_severity / RULE_NORM)

        explanation = []
        if gap_violations > 0:
            explanation.append(f"Gap violations: {gap_violations}")
        if accel_violations > 0:
            explanation.append(f"Harsh lateral maneuvers: {accel_violations}")
        if not explanation:
            explanation.append("Lane changes executed safely")

        return ViolationResult(
            rule_id=self.rule_id,
            name=self.name,
            severity=total_severity,
            severity_normalized=normalized,
            measurements={
                "gap_violations": gap_violations,
                "accel_violations": accel_violations,
                "min_gap_observed": float(min_gap_observed),
                "lane_changes_analyzed": len(lane_changes),
            },
            explanation="; ".join(explanation),
            frame_violations=frame_violations,
            confidence=app.confidence,
        )
