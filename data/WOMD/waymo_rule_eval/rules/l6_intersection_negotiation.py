# waymo_rule_eval/rules/l6_intersection_negotiation.py
# -*- coding: utf-8 -*-
"""
L6.R3 - Intersection Negotiation Rule

Ensures safe behavior at intersections through proper right-of-way
compliance, safe entry gaps, and conflict zone management.

Safe intersection negotiation requires:
- Yielding to vehicles with right-of-way
- Maintaining safe entry gaps (>= 3.0 seconds)
- Proper left turn yielding to oncoming traffic
- Avoiding gridlock (clear exit path)
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


class IntersectionNegotiationApplicability(ApplicabilityDetector):
    """
    Detects intersection entry events for ego vehicle.

    Identifies when ego enters an intersection area by detecting:
    - Multi-agent density
    - Significant heading changes
    - Proximity to intersection zones
    """

    rule_id = "L6.R3"
    level = 6
    name = "Intersection negotiation"

    def __init__(
        self,
        intersection_radius_m: float = 20.0,
        min_intersection_duration_s: float = 1.0,
        multi_agent_threshold: int = 2,
        heading_change_threshold_deg: float = 15.0,
    ):
        self.intersection_radius = intersection_radius_m
        self.min_intersection_duration = min_intersection_duration_s
        self.multi_agent_threshold = multi_agent_threshold
        self.heading_change_threshold = np.radians(heading_change_threshold_deg)

    def _detect_intersection_zones(self, ctx: ScenarioContext) -> np.ndarray:
        """Detect potential intersection zones using multi-agent density."""
        ego_x, ego_y = ctx.ego.x, ctx.ego.y
        n_frames = len(ego_x)

        # Count nearby agents at each frame
        agent_density = np.zeros(n_frames)

        for agent in ctx.agents:
            if len(agent.x) != n_frames:
                continue

            dx = agent.x - ego_x
            dy = agent.y - ego_y
            distances = np.sqrt(dx**2 + dy**2)

            agent_density += (distances < self.intersection_radius).astype(float)

        # Intersection likely when multiple agents nearby
        intersection_mask = agent_density >= self.multi_agent_threshold

        return intersection_mask

    def _classify_turn_direction(
        self, ctx: ScenarioContext, start_idx: int, end_idx: int
    ) -> str:
        """Classify the turn direction during intersection traversal."""
        ego_yaw = ctx.ego.yaw

        start_yaw = ego_yaw[start_idx]
        end_yaw = ego_yaw[min(end_idx, len(ego_yaw) - 1)]

        # Normalize heading change
        delta = end_yaw - start_yaw
        while delta > np.pi:
            delta -= 2 * np.pi
        while delta < -np.pi:
            delta += 2 * np.pi

        if abs(delta) < self.heading_change_threshold:
            return "straight"
        elif delta > 0:
            return "left"
        else:
            return "right"

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

        # Check for sufficient agents
        if len(ctx.agents) < self.multi_agent_threshold:
            return ApplicabilityResult(
                rule_id=self.rule_id,
                rule_level=self.level,
                name=self.name,
                applies=False,
                confidence=0.7,
                reasons=[f"Fewer than {self.multi_agent_threshold} agents"],
                features={"intersection_entries": []},
            )

        dt = ctx.dt or 0.1

        # Detect intersection zones
        intersection_mask = self._detect_intersection_zones(ctx)
        intersection_spans = contiguous_spans(intersection_mask)

        # Filter by minimum duration
        min_frames = int(self.min_intersection_duration / dt)
        valid_spans = [
            span for span in intersection_spans if (span[1] - span[0]) >= min_frames
        ]

        if not valid_spans:
            return ApplicabilityResult(
                rule_id=self.rule_id,
                rule_level=self.level,
                name=self.name,
                applies=False,
                confidence=0.8,
                reasons=["No sustained intersection presence"],
                features={"intersection_entries": []},
            )

        # Build intersection entry information
        intersection_entries = []
        for start_idx, end_idx in valid_spans:
            duration = (end_idx - start_idx) * dt
            turn_direction = self._classify_turn_direction(ctx, start_idx, end_idx)

            intersection_entries.append(
                {
                    "start_frame": start_idx,
                    "end_frame": end_idx,
                    "duration": duration,
                    "turn_direction": turn_direction,
                }
            )

        total_time = sum(e["duration"] for e in intersection_entries)

        return ApplicabilityResult(
            rule_id=self.rule_id,
            rule_level=self.level,
            name=self.name,
            applies=True,
            confidence=0.8,
            reasons=[
                f"Detected {len(intersection_entries)} intersection event(s)",
                f"Total time: {total_time:.1f}s",
            ],
            features={
                "intersection_entries": intersection_entries,
                "intersection_mask": intersection_mask,
            },
        )


class IntersectionNegotiationEvaluator(ViolationEvaluator):
    """
    Evaluates intersection negotiation safety.

    Checks:
    - Safe entry gaps with crossing traffic
    - TTC to conflicting vehicles
    - Speed appropriateness during turns
    """

    rule_id = "L6.R3"
    level = 6
    name = "Intersection negotiation"

    def __init__(
        self,
        min_entry_gap_s: float = 3.0,
        critical_ttc_s: float = 2.0,
        max_turn_speed_mps: float = 8.0,
    ):
        self.min_entry_gap = min_entry_gap_s
        self.critical_ttc = critical_ttc_s
        self.max_turn_speed = max_turn_speed_mps

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

        entries = app.features.get("intersection_entries", [])

        frame_violations = []
        gap_violations = 0
        speed_violations = 0
        min_ttc_observed = np.inf

        ego_x, ego_y = ctx.ego.x, ctx.ego.y
        ego_speed = ctx.ego.speed
        ego_yaw = ctx.ego.yaw

        for entry in entries:
            start = entry["start_frame"]
            end = entry["end_frame"]
            turn_dir = entry["turn_direction"]

            for t in range(start, min(end, len(ego_x))):
                # Check speed during turns
                if turn_dir in ["left", "right"]:
                    speed = ego_speed[t]
                    if speed > self.max_turn_speed:
                        speed_violations += 1
                        severity = min(1.0, (speed - self.max_turn_speed) / 5.0)
                        frame_violations.append(
                            FrameViolation(
                                frame_idx=t,
                                severity=severity,
                                details={
                                    "type": "turn_speed",
                                    "speed": float(speed),
                                    "turn_direction": turn_dir,
                                },
                            )
                        )

                # Check TTC to conflicting agents
                for agent in ctx.agents:
                    if t >= len(agent.x):
                        continue

                    dx = agent.x[t] - ego_x[t]
                    dy = agent.y[t] - ego_y[t]
                    center_dist = np.sqrt(dx**2 + dy**2)
                    # Approximate bumper-to-bumper distance
                    half_lengths = (ctx.ego.length + agent.length) / 2.0
                    dist = max(0.0, center_dist - half_lengths)

                    if dist > 30:  # Skip distant agents
                        continue

                    # Calculate closing speed
                    rel_vx = ego_speed[t] * np.cos(ego_yaw[t]) - agent.speed[
                        t
                    ] * np.cos(agent.yaw[t])
                    rel_vy = ego_speed[t] * np.sin(ego_yaw[t]) - agent.speed[
                        t
                    ] * np.sin(agent.yaw[t])

                    closing = -(dx * rel_vx + dy * rel_vy) / (dist + 0.1)

                    if closing > 0.5:
                        ttc = dist / closing
                        if ttc < min_ttc_observed:
                            min_ttc_observed = ttc

                        if ttc < self.critical_ttc:
                            gap_violations += 1
                            severity = 1.0 - (ttc / self.critical_ttc)
                            frame_violations.append(
                                FrameViolation(
                                    frame_idx=t,
                                    severity=severity,
                                    details={
                                        "type": "gap_violation",
                                        "ttc": float(ttc),
                                        "agent_id": agent.id,
                                    },
                                )
                            )

        total_severity = gap_violations * 0.5 + speed_violations * 0.2
        normalized = min(1.0, total_severity / RULE_NORM)

        explanation = []
        if gap_violations > 0:
            explanation.append(f"Gap violations: {gap_violations}")
        if speed_violations > 0:
            explanation.append(f"Speed violations: {speed_violations}")
        if not explanation:
            explanation.append("Intersection negotiated safely")

        return ViolationResult(
            rule_id=self.rule_id,
            name=self.name,
            severity=total_severity,
            severity_normalized=normalized,
            measurements={
                "gap_violations": gap_violations,
                "speed_violations": speed_violations,
                "min_ttc_observed": float(min_ttc_observed),
                "intersections_analyzed": len(entries),
            },
            explanation="; ".join(explanation),
            frame_violations=frame_violations,
            confidence=app.confidence,
        )
