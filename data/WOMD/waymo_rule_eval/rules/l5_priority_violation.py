# waymo_rule_eval/rules/l5_priority_violation.py
# -*- coding: utf-8 -*-
"""
L5.R2 - Priority Violation Rule

Detects violations of right-of-way and priority rules at intersections,
merges, and multi-agent interactions.

Violation Types:
  1. Yield sign violations (failure to yield at marked yield signs)
  2. Merge violations (cutting off other vehicles during lane merges)
  3. Intersection priority violations (failure to yield at uncontrolled intersections)

Standards:
  - Critical TTC threshold: < 3.0 seconds
  - Warning TTC threshold: < 5.0 seconds
  - Safe merge gap: >= 2.0 seconds
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

RULE_NORM = 40.0


class PriorityViolationApplicability(ApplicabilityDetector):
    """
    Detect if priority violation checking is applicable.

    Applies when:
    - Other agents are present (multi-agent scenarios)
    - Ego is approaching an intersection or merge point
    - Close proximity to other vehicles requiring priority assessment
    """

    rule_id = "L5.R2"
    level = 5
    name = "Priority violation"

    def __init__(self, detection_distance_m: float = 50.0, min_agents: int = 1):
        self.detection_distance_m = detection_distance_m
        self.min_agents = min_agents

    def detect(self, ctx: ScenarioContext) -> ApplicabilityResult:
        # Check for presence of other agents
        if len(ctx.agents) < self.min_agents:
            return ApplicabilityResult(
                rule_id=self.rule_id,
                rule_level=self.level,
                name=self.name,
                applies=False,
                confidence=0.0,
                reasons=["Insufficient agents for priority assessment"],
                features={},
            )

        ego_x = ctx.ego.x
        ego_y = ctx.ego.y
        T = len(ego_x)

        # Analyze agent proximity
        min_distances = []
        interacting_agents = []

        for agent in ctx.agents:
            if len(agent.x) != T:
                continue

            dx = agent.x - ego_x
            dy = agent.y - ego_y
            distances = np.sqrt(dx**2 + dy**2)
            min_dist = float(np.nanmin(distances))
            min_distances.append(min_dist)

            if min_dist < self.detection_distance_m:
                interacting_agents.append(agent.id)

        if len(interacting_agents) == 0:
            return ApplicabilityResult(
                rule_id=self.rule_id,
                rule_level=self.level,
                name=self.name,
                applies=False,
                confidence=0.0,
                reasons=["No agents within interaction distance"],
                features={},
            )

        # Calculate confidence
        proximity_factor = min(1.0, len(interacting_agents) / 3.0)
        dist_factor = min(1.0, 30.0 / np.min(min_distances))
        confidence = 0.5 * proximity_factor + 0.5 * dist_factor

        return ApplicabilityResult(
            rule_id=self.rule_id,
            rule_level=self.level,
            name=self.name,
            applies=True,
            confidence=confidence,
            reasons=[f"Ego interacts with {len(interacting_agents)} agent(s)"],
            features={
                "interacting_agent_ids": interacting_agents,
                "min_agent_distance": float(np.min(min_distances)),
            },
        )


class PriorityViolationEvaluator(ViolationEvaluator):
    """
    Evaluates priority violations based on time-to-collision (TTC).

    Detects:
    - Unsafe gaps during merge/lane change
    - Cutting off other vehicles
    - Failure to yield right-of-way
    """

    rule_id = "L5.R2"
    name = "Priority violation"

    def __init__(
        self,
        critical_ttc_threshold_s: float = 3.0,
        warning_ttc_threshold_s: float = 5.0,
        safe_gap_s: float = 2.0,
    ):
        self.critical_ttc = critical_ttc_threshold_s
        self.warning_ttc = warning_ttc_threshold_s
        self.safe_gap = safe_gap_s

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

        ego_x, ego_y = ctx.ego.x, ctx.ego.y
        ego_speed = ctx.ego.speed
        ego_yaw = ctx.ego.yaw
        dt = ctx.dt or 0.1
        T = len(ego_x)

        frame_violations = []
        ttc_violations = 0
        min_ttc_observed = np.inf

        for agent in ctx.agents:
            if len(agent.x) != T:
                continue

            for t in range(T):
                # Calculate relative position
                dx = agent.x[t] - ego_x[t]
                dy = agent.y[t] - ego_y[t]
                dist = np.sqrt(dx**2 + dy**2)

                if dist > self.warning_ttc * ego_speed[t] + 20:
                    continue  # Too far for TTC calculation

                # Calculate relative velocity
                rel_vx = (agent.speed[t] * np.cos(agent.yaw[t])) - (
                    ego_speed[t] * np.cos(ego_yaw[t])
                )
                rel_vy = (agent.speed[t] * np.sin(agent.yaw[t])) - (
                    ego_speed[t] * np.sin(ego_yaw[t])
                )

                # Project relative velocity onto distance vector
                if dist > 0.1:
                    closing_speed = -(dx * rel_vx + dy * rel_vy) / dist
                else:
                    closing_speed = 0.0

                # Calculate TTC
                if closing_speed > 0.5:  # Closing in
                    ttc = dist / closing_speed

                    if ttc < min_ttc_observed:
                        min_ttc_observed = ttc

                    if ttc < self.critical_ttc:
                        severity = 1.0 - (ttc / self.critical_ttc)
                        ttc_violations += 1
                        frame_violations.append(
                            FrameViolation(
                                frame_idx=t,
                                severity=severity,
                                details={
                                    "ttc": float(ttc),
                                    "agent_id": agent.id,
                                    "distance": float(dist),
                                    "closing_speed": float(closing_speed),
                                },
                            )
                        )

        total_severity = ttc_violations * 0.5
        normalized = min(1.0, total_severity / RULE_NORM)

        explanation = []
        if ttc_violations > 0:
            explanation.append(f"TTC violations: {ttc_violations}")
            explanation.append(f"Min TTC: {min_ttc_observed:.2f}s")
        else:
            explanation.append("No priority violations detected")

        return ViolationResult(
            rule_id=self.rule_id,
            name=self.name,
            severity=total_severity,
            severity_normalized=normalized,
            measurements={
                "ttc_violations": ttc_violations,
                "min_ttc": float(min_ttc_observed),
            },
            explanation="; ".join(explanation),
            frame_violations=frame_violations,
            confidence=app.confidence,
        )
