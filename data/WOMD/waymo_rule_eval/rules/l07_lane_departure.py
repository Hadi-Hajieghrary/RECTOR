"""
L7.R3: Lane Departure Rule

Detects when a vehicle crosses a solid lane boundary.
The rule checks if the vehicle's lateral offset from the lane centerline exceeds
the lane half-width, indicating the vehicle has crossed into an adjacent lane.

Violation severity is proportional to:
- Duration of the departure
- Magnitude of the offset beyond the boundary
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from ..core.context import ScenarioContext
from ..utils.constants import (
    L7_LANE_HALF_WIDTH_M,
    L7_MIN_DEPARTURE_DURATION_S,
    MIN_MOVING_SPEED_MPS,
)
from .base import (
    ApplicabilityDetector,
    ApplicabilityResult,
    ViolationEvaluator,
    ViolationResult,
)

log = logging.getLogger(__name__)
RULE_NORM = {"L7.R3": 10.0}


class LaneDepartureApplicability(ApplicabilityDetector):
    """
    Checks if lane departure detection is applicable.

    Applicable when:
    - Vehicle is on a valid lane with defined centerline
    - Lane has boundary information
    """

    rule_id = "L7.R3"
    level = 7
    name = "Lane Departure"

    def detect(self, ctx: ScenarioContext) -> ApplicabilityResult:
        """
        Check if scenario has required map information for lane departure.
        """
        # Need valid map context with boundary information
        if not hasattr(ctx, "map") or ctx.map is None:
            return ApplicabilityResult(
                rule_id=self.rule_id,
                rule_level=self.level,
                name=self.name,
                applies=False,
                confidence=1.0,
                reasons=["No map context available"],
                features={},
            )

        # Check if we have lane centerline information
        if not hasattr(ctx.map, "lane_center_xy"):
            return ApplicabilityResult(
                rule_id=self.rule_id,
                rule_level=self.level,
                name=self.name,
                applies=False,
                confidence=1.0,
                reasons=["No lane centerline data"],
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

        # Check if we have ego trajectory
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

        # Check for boundary types (optional - if not present, assume solid)
        has_boundary_types = (
            hasattr(ctx.map, "boundary_types") and ctx.map.boundary_types is not None
        )

        return ApplicabilityResult(
            rule_id=self.rule_id,
            rule_level=self.level,
            name=self.name,
            applies=True,
            confidence=1.0 if has_boundary_types else 0.8,
            reasons=["Lane centerline available for offset calculation"],
            features={
                "has_boundary_types": has_boundary_types,
                "n_frames": len(ctx.ego.x),
            },
        )


class LaneDepartureViolation(ViolationEvaluator):
    """
    Evaluates lane departure violations.

    A violation occurs when:
    1. Vehicle's lateral offset exceeds lane half-width
    2. The crossed boundary is solid (not dashed)
    3. The departure persists for at least min_duration_s seconds
    """

    rule_id = "L7.R3"
    level = 7
    name = "Lane Departure"

    def __init__(
        self,
        lane_half_width_m: float = L7_LANE_HALF_WIDTH_M,
        min_duration_s: float = L7_MIN_DEPARTURE_DURATION_S,
        max_severity: float = 10.0,
    ):
        """
        Initialize the evaluator.

        Args:
            lane_half_width_m: Half-width of lane for boundary detection
            min_duration_s: Minimum duration for violation
            max_severity: Maximum severity value
        """
        self.lane_half_width_m = lane_half_width_m
        self.min_duration_s = min_duration_s
        self.max_severity = max_severity

    def evaluate(
        self, ctx: ScenarioContext, app: ApplicabilityResult
    ) -> ViolationResult:
        """
        Evaluate lane departure violation.
        """
        if not app.applies:
            return ViolationResult(
                rule_id=self.rule_id,
                name=self.name,
                severity=0.0,
                severity_normalized=0.0,
                timeseries=None,
                measurements={},
                explanation=["Rule not applicable"],
                confidence=0.0,
            )

        ego_x = ctx.ego.x
        ego_y = ctx.ego.y
        centerline = ctx.map.lane_center_xy
        dt = ctx.dt

        ego_traj = np.column_stack([ego_x, ego_y])

        if ego_traj.shape[0] == 0 or len(centerline) < 2:
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

        # Get boundary types (default to solid if not available)
        boundary_types = getattr(ctx.map, "boundary_types", None)
        left_solid = True
        right_solid = True

        if boundary_types is not None:
            if isinstance(boundary_types, (list, tuple, np.ndarray)):
                if len(boundary_types) >= 2:
                    left_solid = boundary_types[0] == 2
                    right_solid = boundary_types[1] == 2
                elif len(boundary_types) == 1:
                    left_solid = boundary_types[0] == 2
                    right_solid = boundary_types[0] == 2

        # If neither boundary is solid, no violation possible
        if not left_solid and not right_solid:
            return ViolationResult(
                rule_id=self.rule_id,
                name=self.name,
                severity=0.0,
                severity_normalized=0.0,
                timeseries=None,
                measurements={"note": "No solid boundaries"},
                explanation=["No solid lane boundaries - departure allowed"],
                confidence=1.0,
            )

        # Compute lateral offsets for each trajectory point
        offsets = self._compute_lateral_offsets(ego_traj, centerline)

        # Detect departures (left: negative offset, right: positive offset)
        left_departures = (offsets < -self.lane_half_width_m) & left_solid
        right_departures = (offsets > self.lane_half_width_m) & right_solid
        violations = left_departures | right_departures

        if not np.any(violations):
            return ViolationResult(
                rule_id=self.rule_id,
                name=self.name,
                severity=0.0,
                severity_normalized=0.0,
                timeseries=np.abs(offsets),
                measurements={
                    "max_offset_m": float(np.max(np.abs(offsets))),
                    "departure_duration_s": 0.0,
                },
                explanation=["No lane departure detected"],
                confidence=1.0,
            )

        # Find continuous violation segments
        violation_indices = np.where(violations)[0]
        segments = self._find_continuous_segments(violation_indices)

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
                timeseries=np.abs(offsets),
                measurements={
                    "max_offset_m": float(np.max(np.abs(offsets[violations]))),
                    "departure_duration_s": duration_s,
                },
                explanation=[f"Departure duration {duration_s:.2f}s below threshold"],
                confidence=1.0,
            )

        # Compute measurements
        max_offset_m = float(np.max(np.abs(offsets[max_segment])))
        first_departure_time_s = float(max_segment[0] * dt)

        # Determine which side was crossed
        avg_offset = np.mean(offsets[max_segment])
        boundary_side = "left" if avg_offset < 0 else "right"

        # Compute severity
        offset_severity = min(5.0, (max_offset_m - self.lane_half_width_m) * 2.0)
        duration_severity = min(5.0, duration_s * 2.0)
        severity = offset_severity + duration_severity
        severity = min(severity, self.max_severity)

        normalized = severity / self.max_severity

        measurements = {
            "max_offset_m": max_offset_m,
            "departure_duration_s": duration_s,
            "boundary_side": boundary_side,
            "first_departure_time_s": first_departure_time_s,
        }

        return ViolationResult(
            rule_id=self.rule_id,
            name=self.name,
            severity=severity,
            severity_normalized=normalized,
            timeseries=np.abs(offsets),
            measurements=measurements,
            explanation=[
                f"Vehicle departed {boundary_side} for {duration_s:.2f}s",
                f"Max offset: {max_offset_m:.2f}m (threshold: {self.lane_half_width_m:.2f}m)",
            ],
            confidence=1.0,
        )

    def _compute_lateral_offsets(
        self, trajectory: np.ndarray, centerline: np.ndarray
    ) -> np.ndarray:
        """
        Compute lateral offset from trajectory to lane centerline.

        Returns:
            (T,) array of lateral offsets (positive = right, negative = left)
        """
        centerline = np.array(centerline)
        offsets = np.zeros(trajectory.shape[0])

        for i, pos in enumerate(trajectory):
            # Find nearest point on centerline
            dists = np.linalg.norm(centerline - pos, axis=1)
            nearest_idx = np.argmin(dists)

            # Get local tangent direction at nearest point
            if nearest_idx == 0:
                tangent = centerline[1] - centerline[0]
            elif nearest_idx == len(centerline) - 1:
                tangent = centerline[-1] - centerline[-2]
            else:
                tangent = centerline[nearest_idx + 1] - centerline[nearest_idx - 1]

            tangent = tangent / (np.linalg.norm(tangent) + 1e-8)

            # Vector from centerline to trajectory point
            to_point = pos - centerline[nearest_idx]

            # Lateral offset is cross product (positive = right)
            offsets[i] = np.cross(tangent, to_point)

        return offsets

    def _find_continuous_segments(self, indices: np.ndarray) -> list:
        """
        Find continuous segments in an array of indices.
        """
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
