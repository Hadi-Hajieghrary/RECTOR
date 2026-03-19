"""
L8.R5: Wrong-way Driving Rule

Detects when a vehicle is traveling in the opposite direction to the lane flow.
The rule compares the vehicle's heading with the lane tangent direction.

Violation severity is proportional to:
- Magnitude of heading misalignment
- Duration of wrong-way driving
- Speed while driving wrong-way
"""

from __future__ import annotations

import logging

import numpy as np

from ..core.context import ScenarioContext
from ..utils.constants import (
    L8_WRONGWAY_ANGLE_DEG,
    L8_WRONGWAY_MIN_DURATION_S,
    MIN_MOVING_SPEED_MPS,
)
from .base import (
    ApplicabilityDetector,
    ApplicabilityResult,
    ViolationEvaluator,
    ViolationResult,
)

log = logging.getLogger(__name__)
RULE_NORM = {"L8.R5": 10.0}


class WrongwayApplicability(ApplicabilityDetector):
    """
    Checks if wrong-way detection is applicable.

    Applicable when:
    - Vehicle is on a valid lane with defined centerline
    - Vehicle has valid heading information
    - Vehicle is moving
    """

    rule_id = "L8.R5"
    level = 8
    name = "Wrong-way driving"

    def __init__(self, min_speed_mps: float = MIN_MOVING_SPEED_MPS):
        """
        Initialize the detector.

        Args:
            min_speed_mps: Minimum speed to consider vehicle moving
        """
        self.min_speed_mps = min_speed_mps

    def detect(self, ctx: ScenarioContext) -> ApplicabilityResult:
        """
        Check if wrong-way detection applies.
        """
        # Need valid map context with lane centerline
        if not hasattr(ctx, "map") or ctx.map is None:
            return ApplicabilityResult(
                rule_id=self.rule_id,
                rule_level=self.level,
                name=self.name,
                applies=False,
                confidence=1.0,
                reasons=["No map context"],
                features={},
            )

        # Check for lane centerline
        if not hasattr(ctx.map, "lane_center_xy"):
            return ApplicabilityResult(
                rule_id=self.rule_id,
                rule_level=self.level,
                name=self.name,
                applies=False,
                confidence=1.0,
                reasons=["No lane centerline"],
                features={},
            )

        if ctx.map.lane_center_xy is None or len(ctx.map.lane_center_xy) == 0:
            return ApplicabilityResult(
                rule_id=self.rule_id,
                rule_level=self.level,
                name=self.name,
                applies=False,
                confidence=1.0,
                reasons=["Empty lane centerline"],
                features={},
            )

        # Check if we have ego state
        if ctx.ego is None:
            return ApplicabilityResult(
                rule_id=self.rule_id,
                rule_level=self.level,
                name=self.name,
                applies=False,
                confidence=1.0,
                reasons=["No ego state"],
                features={},
            )

        # Check if vehicle is moving
        if np.max(ctx.ego.speed) < self.min_speed_mps:
            return ApplicabilityResult(
                rule_id=self.rule_id,
                rule_level=self.level,
                name=self.name,
                applies=False,
                confidence=1.0,
                reasons=[
                    f"Vehicle not moving (max speed {np.max(ctx.ego.speed):.2f} m/s)"
                ],
                features={},
            )

        return ApplicabilityResult(
            rule_id=self.rule_id,
            rule_level=self.level,
            name=self.name,
            applies=True,
            confidence=1.0,
            reasons=["Lane centerline and ego heading available"],
            features={},
        )


class WrongwayViolation(ViolationEvaluator):
    """
    Evaluates wrong-way driving violations.

    A violation occurs when:
    1. Vehicle's heading differs from lane tangent by > angle_threshold (90°)
    2. The misalignment persists for at least min_duration_s seconds
    3. Vehicle is moving
    """

    rule_id = "L8.R5"
    level = 8
    name = "Wrong-way driving"

    def __init__(
        self,
        angle_threshold_deg: float = L8_WRONGWAY_ANGLE_DEG,
        min_duration_s: float = L8_WRONGWAY_MIN_DURATION_S,
        min_speed_mps: float = MIN_MOVING_SPEED_MPS,
        max_severity: float = 10.0,
    ):
        """
        Initialize the evaluator.
        """
        self.angle_threshold_deg = angle_threshold_deg
        self.angle_threshold_rad = np.deg2rad(angle_threshold_deg)
        self.min_duration_s = min_duration_s
        self.min_speed_mps = min_speed_mps
        self.max_severity = max_severity

    def evaluate(
        self, ctx: ScenarioContext, app: ApplicabilityResult
    ) -> ViolationResult:
        """
        Evaluate wrong-way driving violations.
        """
        if not app.applies:
            return ViolationResult(
                rule_id=self.rule_id,
                name=self.name,
                severity=0.0,
                severity_normalized=0.0,
                timeseries=None,
                measurements={},
                explanation=["Not applicable"],
                confidence=0.0,
            )

        ego_x = ctx.ego.x
        ego_y = ctx.ego.y
        ego_yaw = ctx.ego.yaw
        ego_speed = ctx.ego.speed
        centerline = np.array(ctx.map.lane_center_xy)
        dt = ctx.dt

        n_frames = len(ego_x)

        if n_frames == 0 or len(centerline) < 2:
            return ViolationResult(
                rule_id=self.rule_id,
                name=self.name,
                severity=0.0,
                severity_normalized=0.0,
                timeseries=None,
                measurements={},
                explanation=["Insufficient data"],
                confidence=0.0,
            )

        # Calculate heading mismatch for each frame
        angle_diffs = np.zeros(n_frames)
        wrongway_mask = np.zeros(n_frames, dtype=bool)

        for i in range(n_frames):
            if ego_speed[i] < self.min_speed_mps:
                continue

            ego_pos = np.array([ego_x[i], ego_y[i]])

            # Find nearest point on centerline
            dists = np.linalg.norm(centerline - ego_pos, axis=1)
            nearest_idx = np.argmin(dists)

            # Get local tangent direction
            if nearest_idx == 0:
                tangent = centerline[1] - centerline[0]
            elif nearest_idx == len(centerline) - 1:
                tangent = centerline[-1] - centerline[-2]
            else:
                tangent = centerline[nearest_idx + 1] - centerline[nearest_idx - 1]

            tangent = tangent / (np.linalg.norm(tangent) + 1e-8)

            # Expected heading (lane direction)
            expected_heading = np.arctan2(tangent[1], tangent[0])

            # Actual heading
            actual_heading = ego_yaw[i]

            # Calculate angle difference (absolute)
            diff = np.abs(self._normalize_angle(actual_heading - expected_heading))
            angle_diffs[i] = np.rad2deg(diff)

            # Wrong-way if diff > 90 degrees (i.e., going opposite direction)
            if diff > self.angle_threshold_rad:
                wrongway_mask[i] = True

        if not np.any(wrongway_mask):
            return ViolationResult(
                rule_id=self.rule_id,
                name=self.name,
                severity=0.0,
                severity_normalized=0.0,
                timeseries=angle_diffs,
                measurements={
                    "max_angle_diff_deg": float(np.max(angle_diffs)),
                    "wrongway_duration_s": 0.0,
                },
                explanation=["No wrong-way driving detected"],
                confidence=1.0,
            )

        # Find continuous wrong-way segments
        wrongway_indices = np.where(wrongway_mask)[0]
        segments = self._find_continuous_segments(wrongway_indices)

        # Find longest segment
        if not segments:
            duration_s = 0.0
        else:
            max_segment = max(segments, key=lambda s: len(s))
            duration_s = len(max_segment) * dt

        if duration_s < self.min_duration_s:
            return ViolationResult(
                rule_id=self.rule_id,
                name=self.name,
                severity=0.0,
                severity_normalized=0.0,
                timeseries=angle_diffs,
                measurements={
                    "max_angle_diff_deg": float(np.max(angle_diffs[wrongway_mask])),
                    "wrongway_duration_s": duration_s,
                },
                explanation=[f"Wrong-way duration {duration_s:.2f}s below threshold"],
                confidence=1.0,
            )

        # Compute severity
        max_angle = float(np.max(angle_diffs[max_segment]))
        avg_speed = float(np.mean(ego_speed[max_segment]))
        first_wrongway_time = float(max_segment[0] * dt)

        # Severity: combination of angle, duration, and speed
        angle_factor = (max_angle - self.angle_threshold_deg) / 90.0
        duration_factor = min(1.0, duration_s / 5.0)
        speed_factor = min(1.0, avg_speed / 10.0)

        severity = self.max_severity * (
            0.4 * angle_factor + 0.4 * duration_factor + 0.2 * speed_factor
        )
        severity = max(0.0, min(severity, self.max_severity))

        normalized = severity / self.max_severity

        measurements = {
            "max_angle_diff_deg": max_angle,
            "wrongway_duration_s": duration_s,
            "avg_speed_mps": avg_speed,
            "first_wrongway_time_s": first_wrongway_time,
        }

        return ViolationResult(
            rule_id=self.rule_id,
            name=self.name,
            severity=severity,
            severity_normalized=normalized,
            timeseries=angle_diffs,
            measurements=measurements,
            explanation=[
                f"Wrong-way driving for {duration_s:.2f}s",
                f"Max heading deviation: {max_angle:.1f}°",
            ],
            confidence=1.0,
        )

    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-pi, pi] range."""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle

    def _find_continuous_segments(self, indices: np.ndarray) -> list:
        """Find continuous segments in an array of indices."""
        if len(indices) == 0:
            return []  # No segments to report

        segments = []
        current_segment = [indices[0]]

        for i in range(1, len(indices)):
            if indices[i] == indices[i - 1] + 1:
                current_segment.append(indices[i])
            else:
                segments.append(current_segment)
                current_segment = [indices[i]]

        segments.append(current_segment)
        return segments
