"""
L4.R3: Left Turn Gap Acceptance Rule

When making a left turn across oncoming traffic, the SDC must accept only
safe gaps that provide sufficient time and distance for the turn maneuver
without forcing oncoming vehicles to brake or take evasive action.

Standards:
- Minimum TTC: 4.0 seconds (safe gap threshold)
- Critical TTC: 2.0 seconds (unsafe gap)
- Turn detection: Heading change > 15° left
- Severity: Inverse TTC × time (1/s·s)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from ..core.context import ScenarioContext
from ..core.temporal_spatial import TemporalSpatialIndex
from ..utils.constants import (
    L4_CRITICAL_TTC_S,
    L4_ONCOMING_DETECTION_RANGE_M,
    L4_SAFE_TTC_S,
    L4_TURN_THRESHOLD_DEG,
    MIN_MOVING_SPEED_MPS,
)
from .base import (
    ApplicabilityDetector,
    ApplicabilityResult,
    ViolationEvaluator,
    ViolationResult,
)

log = logging.getLogger(__name__)
RULE_NORM = {"L4.R3": 5.0}  # Normalization: 5.0 (1/s·s)


class LeftTurnGapApplicability(ApplicabilityDetector):
    """
    Detects if the SDC is making a left turn across oncoming traffic.

    Applicability Conditions:
    - Vehicle is performing a left turn (heading change > threshold)
    - Oncoming vehicles are detected in opposite lane
    - Vehicle is in motion during turn
    """

    rule_id = "L4.R3"
    level = 4
    name = "Left turn gap acceptance"

    def __init__(
        self,
        turn_threshold_deg: float = L4_TURN_THRESHOLD_DEG,
        min_speed_mps: float = MIN_MOVING_SPEED_MPS,
        oncoming_detection_range_m: float = L4_ONCOMING_DETECTION_RANGE_M,
        opposite_heading_tolerance_deg: float = 45.0,
    ):
        """
        Initialize the detector.

        Args:
            turn_threshold_deg: Min heading change to detect left turn (°)
            min_speed_mps: Min speed to consider vehicle in motion
            oncoming_detection_range_m: Range to detect oncoming vehicles
            opposite_heading_tolerance_deg: Tolerance for opposite heading
        """
        self.turn_threshold_rad = np.deg2rad(turn_threshold_deg)
        self.min_speed_mps = min_speed_mps
        self.oncoming_detection_range_m = oncoming_detection_range_m
        self.opposite_heading_tolerance_rad = np.deg2rad(opposite_heading_tolerance_deg)

    def detect(self, ctx: ScenarioContext) -> ApplicabilityResult:
        """
        Check if left turn gap acceptance rule applies to this scenario.
        """
        total_frames = len(ctx.ego.x)

        # Detect left turn frames
        turn_frames = []
        for frame_idx in range(1, total_frames):
            prev_heading = ctx.ego.yaw[frame_idx - 1]
            curr_heading = ctx.ego.yaw[frame_idx]

            # Normalize heading change to [-pi, pi]
            heading_change = self._normalize_angle(curr_heading - prev_heading)

            # Left turn: positive heading change (counterclockwise)
            is_turning = heading_change > self.turn_threshold_rad
            is_moving = ctx.ego.speed[frame_idx] >= self.min_speed_mps

            if is_turning and is_moving:
                turn_frames.append(frame_idx)

        if not turn_frames:
            return ApplicabilityResult(
                rule_id=self.rule_id,
                rule_level=self.level,
                name=self.name,
                applies=False,
                confidence=0.0,
                reasons=["No left turn maneuver detected"],
                features={},
            )

        # Check for oncoming vehicles during turn
        oncoming_detected = False
        ts_index = TemporalSpatialIndex(ctx.agents, total_frames)

        for frame_idx in turn_frames:
            if self._has_oncoming_vehicles(ctx, ts_index, frame_idx):
                oncoming_detected = True
                break

        if not oncoming_detected:
            return ApplicabilityResult(
                rule_id=self.rule_id,
                rule_level=self.level,
                name=self.name,
                applies=False,
                confidence=0.0,
                reasons=["Left turn detected but no oncoming vehicles present"],
                features={"turn_frames": len(turn_frames)},
            )

        confidence = len(turn_frames) / total_frames

        explanation = (
            f"Left turn detected in {len(turn_frames)}/{total_frames} frames "
            f"with oncoming vehicles present."
        )

        return ApplicabilityResult(
            rule_id=self.rule_id,
            rule_level=self.level,
            name=self.name,
            applies=True,
            confidence=confidence,
            reasons=[explanation],
            features={"turn_frames": len(turn_frames), "total_frames": total_frames},
        )

    def _has_oncoming_vehicles(
        self, ctx: ScenarioContext, ts_index: TemporalSpatialIndex, frame_idx: int
    ) -> bool:
        """
        Check if oncoming vehicles are present.
        """
        ego_x = ctx.ego.x[frame_idx]
        ego_y = ctx.ego.y[frame_idx]
        ego_heading = ctx.ego.yaw[frame_idx]

        # Find nearby vehicles
        nearby = ts_index.query_radius(
            frame_idx, ego_x, ego_y, self.oncoming_detection_range_m
        )

        # Check if any are oncoming (opposite direction)
        for agent, _dist in nearby:
            if agent.is_vru:  # Skip pedestrians/cyclists
                continue
            if frame_idx >= len(agent.x):
                continue

            agent_heading = agent.yaw[frame_idx]
            heading_diff = self._normalize_angle(agent_heading - ego_heading)

            # Oncoming: heading difference close to ±180°
            if abs(abs(heading_diff) - np.pi) < self.opposite_heading_tolerance_rad:
                return True

        return False

    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-pi, pi] range."""
        return float((angle + np.pi) % (2 * np.pi) - np.pi)


class LeftTurnGapViolation(ViolationEvaluator):
    """
    Evaluates left turn gap acceptance violations.

    Violation occurs when:
    - SDC makes left turn with insufficient gap
    - Time-to-collision (TTC) with oncoming vehicle < safe threshold

    Severity is based on:
    - Inverse TTC (1/TTC) - higher when gap is smaller
    - Duration of unsafe gap acceptance
    - Integral: ∫(1/TTC) dt when TTC < threshold
    """

    rule_id = "L4.R3"
    level = 4
    name = "Left turn gap acceptance"

    def __init__(
        self,
        safe_ttc_s: float = L4_SAFE_TTC_S,
        critical_ttc_s: float = L4_CRITICAL_TTC_S,
        turn_threshold_deg: float = L4_TURN_THRESHOLD_DEG,
        min_speed_mps: float = MIN_MOVING_SPEED_MPS,
        oncoming_detection_range_m: float = L4_ONCOMING_DETECTION_RANGE_M,
    ):
        """
        Initialize the evaluator.
        """
        self.safe_ttc_s = safe_ttc_s
        self.critical_ttc_s = critical_ttc_s
        self.turn_threshold_rad = np.deg2rad(turn_threshold_deg)
        self.min_speed_mps = min_speed_mps
        self.oncoming_detection_range_m = oncoming_detection_range_m

    def evaluate(
        self, ctx: ScenarioContext, app: ApplicabilityResult
    ) -> ViolationResult:
        """
        Evaluate left turn gap acceptance violations.
        """
        total_frames = len(ctx.ego.x)
        dt = ctx.dt

        ts_index = TemporalSpatialIndex(ctx.agents, total_frames)

        # Track violations per frame
        inverse_ttcs = []
        violation_frames = []
        min_ttc_overall = float("inf")

        for frame_idx in range(1, total_frames):
            # Check if making a left turn
            prev_heading = ctx.ego.yaw[frame_idx - 1]
            curr_heading = ctx.ego.yaw[frame_idx]
            heading_change = self._normalize_angle(curr_heading - prev_heading)

            is_turning = heading_change > self.turn_threshold_rad
            is_moving = ctx.ego.speed[frame_idx] >= self.min_speed_mps

            if not (is_turning and is_moving):
                inverse_ttcs.append(0.0)
                continue

            # Find oncoming vehicles and calculate TTC
            min_ttc = self._calculate_min_ttc(ctx, ts_index, frame_idx)
            min_ttc_overall = min(min_ttc_overall, min_ttc)

            # Calculate inverse TTC for severity (only if unsafe)
            if min_ttc < self.safe_ttc_s:
                effective_ttc = max(min_ttc, self.critical_ttc_s * 0.5)
                inverse_ttc = 1.0 / effective_ttc
                violation_frames.append(frame_idx)
            else:
                inverse_ttc = 0.0

            inverse_ttcs.append(inverse_ttc)

        # Calculate severity: integral of inverse TTC over time
        severity_raw = float(np.sum(inverse_ttcs) * dt)

        # Normalize severity
        norm_factor = RULE_NORM.get(self.rule_id, 5.0)
        normalized_severity = min(1.0, severity_raw / norm_factor)

        # Calculate measurements
        measurements = {
            "total_inverse_ttc_time": severity_raw,
            "min_ttc_s": min_ttc_overall if np.isfinite(min_ttc_overall) else -1.0,
            "violation_frames": len(violation_frames),
            "total_frames": total_frames,
            "p_violation": (
                len(violation_frames) / total_frames if total_frames > 0 else 0.0
            ),
        }

        explanation = (
            f"Left turn with unsafe gap in "
            f"{measurements['violation_frames']}/{total_frames} frames. "
            f"Min TTC: {measurements['min_ttc_s']:.2f}s. "
            f"Total severity: {severity_raw:.2f}."
        )

        timeseries_arr = np.array(inverse_ttcs)

        return ViolationResult(
            rule_id=self.rule_id,
            name=self.name,
            severity=severity_raw,
            severity_normalized=normalized_severity,
            timeseries=timeseries_arr,
            measurements=measurements,
            explanation=[explanation],
            confidence=1.0 if severity_raw > 0 else 0.0,
        )

    def _calculate_min_ttc(
        self, ctx: ScenarioContext, ts_index: TemporalSpatialIndex, frame_idx: int
    ) -> float:
        """
        Calculate minimum TTC with oncoming vehicles.
        """
        ego_x = ctx.ego.x[frame_idx]
        ego_y = ctx.ego.y[frame_idx]
        ego_heading = ctx.ego.yaw[frame_idx]
        ego_speed = ctx.ego.speed[frame_idx]

        # Find nearby vehicles
        nearby = ts_index.query_radius(
            frame_idx, ego_x, ego_y, self.oncoming_detection_range_m
        )

        min_ttc = float("inf")

        for agent, _dist in nearby:
            if agent.is_vru:
                continue
            if frame_idx >= len(agent.x):
                continue

            # Check if oncoming (opposite direction)
            agent_heading = agent.yaw[frame_idx]
            heading_diff = self._normalize_angle(agent_heading - ego_heading)

            # Oncoming: heading difference close to ±180°
            if abs(abs(heading_diff) - np.pi) > np.deg2rad(45.0):
                continue

            # Calculate distance (approximate bumper-to-bumper by subtracting half-lengths)
            agent_x = agent.x[frame_idx]
            agent_y = agent.y[frame_idx]
            center_distance = np.sqrt((agent_x - ego_x) ** 2 + (agent_y - ego_y) ** 2)
            half_lengths = (ctx.ego.length + agent.length) / 2.0
            distance = max(0.0, center_distance - half_lengths)

            # Calculate relative velocity (closing speed)
            agent_speed = agent.speed[frame_idx]
            relative_speed = ego_speed + agent_speed

            # Calculate TTC
            if relative_speed > 0.1:
                ttc = distance / relative_speed
                min_ttc = min(min_ttc, ttc)

        return min_ttc

    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-pi, pi] range."""
        return float((angle + np.pi) % (2 * np.pi) - np.pi)
