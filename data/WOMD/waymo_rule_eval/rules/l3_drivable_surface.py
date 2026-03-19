"""
L3.R3: Drivable Surface Constraint Rule

The SDC must remain within designated drivable surfaces at all times.
Violations occur when the vehicle's footprint extends beyond legal
driving areas onto non-drivable surfaces.

Standards:
- Zero tolerance for off-road driving
- Drivable areas: Lanes, shoulders (if permitted)
- Severity: Off-road distance × time (m·s)
"""

from __future__ import annotations

from typing import Optional

import numpy as np

try:
    from shapely.geometry import LineString, MultiPolygon, Point, Polygon
    from shapely.validation import make_valid
except ImportError:
    Polygon = None
    LineString = None
    MultiPolygon = None
    Point = None
    make_valid = None

import logging

from ..core.context import ScenarioContext
from ..utils.constants import (
    L3_DRIVABLE_BUFFER_M,
    L3_LANE_WIDTH_M,
    MIN_MOVING_SPEED_MPS,
    VEHICLE_LENGTH_M,
    VEHICLE_WIDTH_M,
)
from .base import (
    ApplicabilityDetector,
    ApplicabilityResult,
    ViolationEvaluator,
    ViolationResult,
)

log = logging.getLogger(__name__)
RULE_NORM = {"L3.R3": 15.0}  # Normalization: 15 m·s


class DrivableSurfaceApplicability(ApplicabilityDetector):
    """
    Detects if the SDC is in scenarios where drivable surface constraint applies.

    Applicability Conditions:
    - Drivable area map features are available
    - Vehicle is in motion (speed > min_speed_mps)
    - Map data quality is sufficient
    """

    rule_id = "L3.R3"
    level = 3
    name = "Drivable surface constraint"

    def __init__(
        self,
        min_speed_mps: float = MIN_MOVING_SPEED_MPS,
        require_lane_data: bool = True,
    ):
        """
        Initialize the detector.

        Args:
            min_speed_mps: Minimum speed to consider vehicle in motion
            require_lane_data: Whether to require lane data for applicability
        """
        self.min_speed_mps = min_speed_mps
        self.require_lane_data = require_lane_data

    def detect(self, ctx: ScenarioContext) -> ApplicabilityResult:
        """
        Check if drivable surface constraint rule applies to this scenario.

        Args:
            ctx: Scenario context with ego state, agents, and map

        Returns:
            ApplicabilityResult with confidence based on data availability
        """
        # Check if drivable area data is available
        has_drivable_areas = self._has_drivable_areas(ctx)

        if not has_drivable_areas:
            return ApplicabilityResult(
                rule_id=self.rule_id,
                rule_level=self.level,
                name=self.name,
                applies=False,
                confidence=0.0,
                reasons=["No drivable area map features available"],
                features={},
            )

        # Check if vehicle is in motion
        total_frames = len(ctx.ego.x)
        moving_frames = int(np.sum(ctx.ego.speed >= self.min_speed_mps))

        # Calculate confidence based on moving frames
        confidence = moving_frames / total_frames if total_frames > 0 else 0.0
        applies = confidence > 0.0

        explanation = (
            f"Drivable area data available. "
            f"Vehicle moving in {moving_frames}/{total_frames} frames."
        )

        return ApplicabilityResult(
            rule_id=self.rule_id,
            rule_level=self.level,
            name=self.name,
            applies=applies,
            confidence=confidence,
            reasons=[explanation],
            features={"moving_frames": moving_frames, "total_frames": total_frames},
        )

    def _has_drivable_areas(self, ctx: ScenarioContext) -> bool:
        """
        Check if drivable area map features are available.
        """
        if hasattr(ctx, "map") and ctx.map is not None:
            # Check for lane center data
            if hasattr(ctx.map, "lane_center_xy"):
                if ctx.map.lane_center_xy is not None:
                    if len(ctx.map.lane_center_xy) > 0:
                        return True
        return False


class DrivableSurfaceViolation(ViolationEvaluator):
    """
    Evaluates drivable surface constraint violations when SDC goes off-road.

    Violation occurs when:
    - Vehicle's bounding box extends beyond drivable areas
    - Any part of the vehicle is on non-drivable surface

    Severity is based on:
    - Distance off-road (meters)
    - Duration off-road (seconds)
    - Integral: ∫(off_road_distance) dt
    """

    rule_id = "L3.R3"
    level = 3
    name = "Drivable surface constraint"

    def __init__(
        self,
        lane_width_m: float = L3_LANE_WIDTH_M,
        buffer_m: float = L3_DRIVABLE_BUFFER_M,
        vehicle_length_m: float = VEHICLE_LENGTH_M,
        vehicle_width_m: float = VEHICLE_WIDTH_M,
    ):
        """
        Initialize the evaluator.

        Args:
            lane_width_m: Typical lane width for creating drivable zones
            buffer_m: Buffer around lane center for drivable area
            vehicle_length_m: SDC length
            vehicle_width_m: SDC width
        """
        self.lane_width_m = lane_width_m
        self.buffer_m = buffer_m
        self.vehicle_length_m = vehicle_length_m
        self.vehicle_width_m = vehicle_width_m

    def evaluate(
        self, ctx: ScenarioContext, app: ApplicabilityResult
    ) -> ViolationResult:
        """
        Evaluate drivable surface constraint violations.
        """
        if Polygon is None:
            return ViolationResult(
                rule_id=self.rule_id,
                name=self.name,
                severity=0.0,
                severity_normalized=0.0,
                timeseries=None,
                measurements={},
                explanation=["Shapely not installed"],
                confidence=0.0,
            )

        # Create drivable area polygon from map data
        drivable_area = self._create_drivable_area(ctx)

        if drivable_area is None or drivable_area.is_empty:
            return ViolationResult(
                rule_id=self.rule_id,
                name=self.name,
                severity=0.0,
                severity_normalized=0.0,
                timeseries=None,
                measurements={},
                explanation=["No drivable area available for evaluation"],
                confidence=0.0,
            )

        # Track violations per frame
        off_road_distances = []
        total_frames = len(ctx.ego.x)
        dt = ctx.dt

        for frame_idx in range(total_frames):
            try:
                # Get ego state
                ego_x = ctx.ego.x[frame_idx]
                ego_y = ctx.ego.y[frame_idx]
                ego_heading = ctx.ego.yaw[frame_idx]

                # Create ego bounding box
                v_len = getattr(ctx.ego, "length", self.vehicle_length_m)
                v_wid = getattr(ctx.ego, "width", self.vehicle_width_m)

                ego_bbox = self._create_oriented_box(
                    ego_x, ego_y, ego_heading, v_len, v_wid
                )

                # Skip frame if bbox is invalid
                if ego_bbox is None:
                    off_road_distances.append(0.0)
                    continue

                # Check if vehicle is within drivable area
                if drivable_area.contains(ego_bbox):
                    off_road_distance = 0.0
                elif drivable_area.intersects(ego_bbox):
                    # Partially off drivable surface
                    # Handle MultiPolygon - use boundary instead of exterior
                    try:
                        if isinstance(drivable_area, MultiPolygon):
                            off_road_distance = ego_bbox.centroid.distance(
                                drivable_area.boundary
                            )
                        else:
                            off_road_distance = ego_bbox.centroid.distance(
                                drivable_area.exterior
                            )
                    except Exception:
                        off_road_distance = ego_bbox.centroid.distance(drivable_area)
                else:
                    # Completely off drivable surface
                    off_road_distance = ego_bbox.centroid.distance(drivable_area)

                off_road_distances.append(off_road_distance)
            except Exception as e:
                # Skip this frame on any geometry error
                log.debug(f"Geometry error at frame {frame_idx}: {e}")
                off_road_distances.append(0.0)

        # Calculate severity: integral of off-road distance over time
        severity_raw = float(np.sum(off_road_distances) * dt)  # m·s

        # Normalize severity
        norm_factor = RULE_NORM.get(self.rule_id, 15.0)
        normalized_severity = min(1.0, severity_raw / norm_factor)

        # Calculate measurements
        violation_count = sum(1 for d in off_road_distances if d > 0.0)
        measurements = {
            "total_off_road_distance_time_ms": severity_raw,
            "max_off_road_distance_m": float(max(off_road_distances)),
            "avg_off_road_distance_m": float(np.mean(off_road_distances)),
            "violation_frames": violation_count,
            "total_frames": total_frames,
            "p_violation": violation_count / total_frames if total_frames > 0 else 0.0,
        }

        explanation = (
            f"Vehicle off drivable surface for "
            f"{measurements['violation_frames']}/{total_frames} frames. "
            f"Max distance: {measurements['max_off_road_distance_m']:.2f} m. "
            f"Total severity: {severity_raw:.2f} m·s."
        )

        timeseries_arr = np.array(off_road_distances)

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

    def _create_drivable_area(self, ctx: ScenarioContext) -> Optional[Polygon]:
        """
        Create a drivable area polygon from map data.
        """
        if not hasattr(ctx, "map") or ctx.map is None:
            return None

        if not hasattr(ctx.map, "lane_center_xy"):
            return None

        lane_center = ctx.map.lane_center_xy

        if lane_center is None or len(lane_center) < 2:
            return None

        try:
            # Filter out NaN values from lane center points
            if hasattr(lane_center, "shape") and len(lane_center.shape) == 2:
                valid_mask = ~np.any(np.isnan(lane_center), axis=1)
                lane_center = lane_center[valid_mask]

            if len(lane_center) < 2:
                return None

            # Remove duplicate consecutive points
            if hasattr(lane_center, "shape"):
                diffs = np.diff(lane_center, axis=0)
                non_dup = np.any(diffs != 0, axis=1)
                non_dup = np.concatenate([[True], non_dup])
                lane_center = lane_center[non_dup]

            if len(lane_center) < 2:
                return None

            lane_line = LineString(lane_center)
            if not lane_line.is_valid:
                return None

            buffer_distance = (self.lane_width_m / 2.0) + self.buffer_m
            drivable_polygon = lane_line.buffer(buffer_distance, cap_style=2)

            if not drivable_polygon.is_valid:
                # Try to fix with buffer(0)
                drivable_polygon = drivable_polygon.buffer(0)

            return drivable_polygon if drivable_polygon.is_valid else None
        except Exception as e:
            log.debug(f"Failed to create drivable area: {e}")
            return None

    def _create_oriented_box(
        self, x: float, y: float, heading: float, length: float, width: float
    ) -> Optional[Polygon]:
        """
        Create an oriented bounding box for the vehicle.

        Returns None if inputs are invalid (NaN, etc.)
        """
        # Validate inputs - check for NaN/Inf
        if np.isnan(x) or np.isnan(y) or np.isnan(heading):
            return None
        if np.isinf(x) or np.isinf(y) or np.isinf(heading):
            return None
        if length <= 0 or width <= 0:
            return None

        half_length = length / 2
        half_width = width / 2

        corners = np.array(
            [
                [-half_length, -half_width],
                [half_length, -half_width],
                [half_length, half_width],
                [-half_length, half_width],
            ]
        )

        cos_h = np.cos(heading)
        sin_h = np.sin(heading)
        rotation = np.array([[cos_h, -sin_h], [sin_h, cos_h]])

        rotated_corners = corners @ rotation.T
        translated_corners = rotated_corners + np.array([x, y])

        try:
            poly = Polygon(translated_corners)
            if not poly.is_valid:
                poly = poly.buffer(0)
            return poly if poly.is_valid else None
        except Exception:
            return None
