# waymo_rule_eval/rules/l5_traffic_signal_compliance.py
# -*- coding: utf-8 -*-
"""
L5.R1 - Traffic Signal Compliance Rule

Evaluates compliance with traffic signals (red, yellow, green lights)
at controlled intersections according to traffic laws.

Standards:
    - RED: Must stop before stop line and remain stopped
    - YELLOW: Should prepare to stop if safe to do so
    - GREEN: May proceed through intersection
    - Detection distance: Within 50m of traffic signal
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from ..core.context import ScenarioContext
from ..utils.constants import (
    RED_SIGNAL_STATES,
    SIGNAL_STATE_UNKNOWN,
    YELLOW_SIGNAL_STATES,
)
from .base import (
    ApplicabilityDetector,
    ApplicabilityResult,
    FrameViolation,
    ViolationEvaluator,
    ViolationResult,
)

# Signal state constants - USE canonical values from constants.py
# These match the Waymo encoding used in adapter_motion_scenario.py
SIGNAL_UNKNOWN = SIGNAL_STATE_UNKNOWN  # 0

RULE_NORM = 50.0  # High severity for traffic signal violations


class TrafficSignalComplianceApplicability(ApplicabilityDetector):
    """
    Detects whether the traffic signal compliance rule applies.

    Applies when:
    - Stop line geometry is available
    - Traffic signal state data is available
    - Vehicle is within detection distance of signal
    """

    rule_id = "L5.R1"
    level = 5
    name = "Traffic signal compliance"

    def __init__(self, detection_distance_m: float = 50.0):
        self.detection_distance_m = detection_distance_m

    def detect(self, ctx: ScenarioContext) -> ApplicabilityResult:
        # Check for stop line geometry
        has_stopline = (
            hasattr(ctx, "map_context")
            and ctx.map_context is not None
            and hasattr(ctx.map_context, "stopline_xy")
            and ctx.map_context.stopline_xy is not None
            and len(ctx.map_context.stopline_xy) >= 2
        )

        if not has_stopline:
            return ApplicabilityResult(
                rule_id=self.rule_id,
                rule_level=self.level,
                name=self.name,
                applies=False,
                confidence=0.0,
                reasons=["No stop line geometry available"],
                features={},
            )

        # Check for traffic signal data
        has_signals = (
            hasattr(ctx, "signals")
            and ctx.signals is not None
            and hasattr(ctx.signals, "signal_state")
            and ctx.signals.signal_state is not None
        )

        if not has_signals:
            return ApplicabilityResult(
                rule_id=self.rule_id,
                rule_level=self.level,
                name=self.name,
                applies=False,
                confidence=0.0,
                reasons=["No traffic signal data available"],
                features={},
            )

        # Calculate distance to stop line
        stopline = ctx.map_context.stopline_xy
        ego_x, ego_y = ctx.ego.x, ctx.ego.y

        # Find closest point on stop line to ego at each frame
        distances = np.full(len(ego_x), np.inf)
        for i in range(len(ego_x)):
            if np.isnan(ego_x[i]) or np.isnan(ego_y[i]):
                continue
            dists = np.sqrt(
                (stopline[:, 0] - ego_x[i]) ** 2 + (stopline[:, 1] - ego_y[i]) ** 2
            )
            distances[i] = np.min(dists)

        min_distance = np.nanmin(distances)
        close_enough = min_distance < self.detection_distance_m

        # Check for non-unknown signal states
        signal_states = ctx.signals.signal_state
        has_valid_signals = np.any(signal_states > SIGNAL_UNKNOWN)

        applies = close_enough and has_valid_signals

        if applies:
            proximity_factor = max(0.0, 1.0 - min_distance / self.detection_distance_m)
            signal_quality = float(np.sum(signal_states > SIGNAL_UNKNOWN)) / len(
                signal_states
            )
            confidence = 0.5 * proximity_factor + 0.5 * signal_quality
        else:
            confidence = 0.0

        reasons = []
        if not close_enough:
            reasons.append(f"Too far from stop line ({min_distance:.1f}m)")
        if not has_valid_signals:
            reasons.append("No valid signal states")
        if applies:
            reasons.append(
                f"Within {min_distance:.1f}m of stop line with active signals"
            )

        return ApplicabilityResult(
            rule_id=self.rule_id,
            rule_level=self.level,
            name=self.name,
            applies=applies,
            confidence=confidence,
            reasons=reasons,
            features={
                "min_distance_to_stopline": min_distance,
                "distances": distances,
                "signal_states": signal_states,
            },
        )


class TrafficSignalComplianceViolation(ViolationEvaluator):
    """
    Evaluates traffic signal compliance violations.

    Violations:
    - Running red light (crossing stop line while signal is red)
    - Excessive speed on yellow (not slowing down on yellow)
    """

    rule_id = "L5.R1"
    name = "Traffic signal compliance"

    def __init__(
        self,
        red_light_speed_threshold_mps: float = 1.0,
        yellow_decel_threshold_mps2: float = -1.0,
    ):
        self.red_light_speed_threshold = red_light_speed_threshold_mps
        self.yellow_decel_threshold = yellow_decel_threshold_mps2

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

        signal_states = app.features.get("signal_states", np.array([]))
        distances = app.features.get("distances", np.array([]))
        ego_speed = ctx.ego.speed
        dt = ctx.dt or 0.1

        frame_violations = []
        red_light_violations = 0
        yellow_violations = 0
        max_severity = 0.0

        # Calculate acceleration
        accel = np.diff(ego_speed) / dt
        accel = np.append(accel, 0)  # Pad to same length

        for t in range(len(signal_states)):
            state = signal_states[t]
            speed = ego_speed[t] if t < len(ego_speed) else 0.0
            dist = distances[t] if t < len(distances) else np.inf

            violation_severity = 0.0
            violation_type = None

            # Red light violation: crossing stop line while red
            if state in RED_SIGNAL_STATES:
                # Check if crossing stop line (small distance) with speed
                if dist < 5.0 and speed > self.red_light_speed_threshold:
                    violation_severity = min(1.0, speed / 10.0)  # Scale by speed
                    violation_type = "red_light_running"
                    red_light_violations += 1

            # Yellow light: should be decelerating
            elif state in YELLOW_SIGNAL_STATES:
                if dist < 30.0 and t < len(accel):
                    # Should be slowing down, not accelerating
                    if accel[t] > 0.5:  # Accelerating through yellow
                        violation_severity = 0.3 * min(1.0, accel[t] / 2.0)
                        violation_type = "yellow_acceleration"
                        yellow_violations += 1

            if violation_severity > 0:
                max_severity = max(max_severity, violation_severity)
                frame_violations.append(
                    FrameViolation(
                        frame_idx=t,
                        severity=violation_severity,
                        details={
                            "type": violation_type,
                            "signal_state": int(state),
                            "speed": float(speed),
                            "distance": float(dist),
                        },
                    )
                )

        # Calculate total severity
        total_severity = red_light_violations * 1.0 + yellow_violations * 0.3
        normalized = min(1.0, total_severity / RULE_NORM)

        explanation = []
        if red_light_violations > 0:
            explanation.append(f"Red light violations: {red_light_violations}")
        if yellow_violations > 0:
            explanation.append(f"Yellow light accelerations: {yellow_violations}")
        if not explanation:
            explanation.append("No violations detected")

        return ViolationResult(
            rule_id=self.rule_id,
            name=self.name,
            severity=total_severity,
            severity_normalized=normalized,
            measurements={
                "red_light_violations": red_light_violations,
                "yellow_violations": yellow_violations,
                "max_frame_severity": max_severity,
            },
            explanation="; ".join(explanation),
            frame_violations=frame_violations,
            confidence=app.confidence,
        )
