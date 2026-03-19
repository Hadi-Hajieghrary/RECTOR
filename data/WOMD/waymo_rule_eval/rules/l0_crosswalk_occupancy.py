"""
L0.R4: Crosswalk Occupancy Rule

The ego must NOT occupy a crosswalk area when pedestrians are present or
approaching the crosswalk, ensuring safe passage for vulnerable road users.

Standards:
- Detection zone: Crosswalk area + buffer (default 5m)
- Minimum pedestrian speed: 0.3 m/s (walking)
- Occupancy check: Polygon intersection test
- Severity: Overlap area × time (m²·s)
"""

from typing import List, Optional

import numpy as np

from ..core.context import ScenarioContext
from ..core.geometry import oriented_box_corners
from ..utils.constants import (
    CROSSWALK_DETECTION_BUFFER_M,
    CROSSWALK_MAX_DISTANCE_M,
    DEFAULT_EGO_LENGTH,
    DEFAULT_EGO_WIDTH,
    MIN_PEDESTRIAN_SPEED_MPS,
    RULE_NORMALIZATION,
)
from .base import (
    ApplicabilityDetector,
    ApplicabilityResult,
    ViolationEvaluator,
    ViolationResult,
)

# Optional shapely import
try:
    from shapely.geometry import Point, Polygon

    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False


_RULE_ID = "L0.R4"
_LEVEL = 0
_NAME = "Crosswalk Occupancy"


def _corners_to_shapely_polygon(corners: np.ndarray):
    """Convert oriented box corners to Shapely polygon."""
    if not SHAPELY_AVAILABLE:
        return None
    return Polygon(corners)


class CrosswalkOccupancyApplicability(ApplicabilityDetector):
    """
    Detects if ego is near crosswalks with pedestrians present.

    Applicability Conditions:
    - Crosswalk features exist in map
    - Pedestrians are detected within detection zone
    - Pedestrians are moving (speed > min_speed)
    """

    rule_id = _RULE_ID
    level = _LEVEL
    name = _NAME

    def __init__(
        self,
        detection_buffer_m: float = CROSSWALK_DETECTION_BUFFER_M,
        min_ped_speed_mps: float = MIN_PEDESTRIAN_SPEED_MPS,
        max_distance_m: float = CROSSWALK_MAX_DISTANCE_M,
    ):
        self.detection_buffer_m = detection_buffer_m
        self.min_ped_speed_mps = min_ped_speed_mps
        self.max_distance_m = max_distance_m

    def detect(self, ctx: ScenarioContext) -> ApplicabilityResult:
        """Check if crosswalk occupancy rule applies."""
        if not SHAPELY_AVAILABLE:
            return ApplicabilityResult(
                applies=False,
                confidence=0.0,
                reasons=["Shapely not installed for polygon operations"],
                features={},
            )

        # Extract crosswalks from map
        crosswalks = self._extract_crosswalks(ctx)

        if not crosswalks:
            return ApplicabilityResult(
                applies=False,
                confidence=0.0,
                reasons=["No crosswalk features found in map"],
                features={"n_crosswalks": 0},
            )

        # Check each frame for applicability
        applicable_frames = 0
        T = len(ctx.ego.x)
        pedestrians = ctx.pedestrians

        for t in range(T):
            if not ctx.ego.is_valid_at(t):
                continue

            ego_x, ego_y = ctx.ego.x[t], ctx.ego.y[t]
            ego_pos = Point(ego_x, ego_y)

            # Find nearby crosswalks
            nearby_crosswalks = []
            for cw_poly in crosswalks:
                if cw_poly.distance(ego_pos) <= self.max_distance_m:
                    nearby_crosswalks.append(cw_poly)

            if not nearby_crosswalks:
                continue

            # Check for active pedestrians near crosswalks
            has_active_peds = False
            for ped in pedestrians:
                if not ped.is_valid_at(t):
                    continue

                ped_x, ped_y = ped.x[t], ped.y[t]
                ped_speed = ped.speed[t]

                if ped_speed < self.min_ped_speed_mps:
                    continue

                ped_pos = Point(ped_x, ped_y)

                for cw_poly in nearby_crosswalks:
                    buffered = cw_poly.buffer(self.detection_buffer_m)
                    if buffered.contains(ped_pos):
                        has_active_peds = True
                        break

                if has_active_peds:
                    break

            if has_active_peds:
                applicable_frames += 1

        confidence = applicable_frames / T if T > 0 else 0.0
        applies = applicable_frames > 0

        return ApplicabilityResult(
            applies=applies,
            confidence=confidence,
            reasons=[
                f"Found {len(crosswalks)} crosswalks, "
                f"applies in {applicable_frames}/{T} frames"
            ],
            features={
                "n_crosswalks": len(crosswalks),
                "applicable_frames": applicable_frames,
            },
        )

    def _extract_crosswalks(self, ctx: ScenarioContext) -> List:
        """Extract crosswalk polygons from map context."""
        crosswalks = []
        map_ctx = ctx.map_context

        if not map_ctx.has_crosswalks:
            return crosswalks

        for cw_coords in map_ctx.crosswalk_polys:
            if len(cw_coords) >= 3:
                try:
                    poly = Polygon(cw_coords)
                    if poly.is_valid:
                        crosswalks.append(poly)
                except Exception:
                    continue

        return crosswalks


class CrosswalkOccupancyViolation(ViolationEvaluator):
    """
    Evaluates crosswalk occupancy violations.

    Violation occurs when:
    - Ego's bounding box intersects with crosswalk polygon
    - Pedestrians are detected in or near the crosswalk

    Severity: Integral of overlap area over time (m²·s)
    """

    rule_id = _RULE_ID
    level = _LEVEL
    name = _NAME

    def __init__(
        self,
        detection_buffer_m: float = CROSSWALK_DETECTION_BUFFER_M,
        min_ped_speed_mps: float = MIN_PEDESTRIAN_SPEED_MPS,
    ):
        self.detection_buffer_m = detection_buffer_m
        self.min_ped_speed_mps = min_ped_speed_mps

    def evaluate(
        self, ctx: ScenarioContext, applicability: ApplicabilityResult
    ) -> ViolationResult:
        """Evaluate crosswalk occupancy violations."""
        if not SHAPELY_AVAILABLE:
            return ViolationResult(
                severity=0.0,
                explanation="Shapely not available",
            )

        crosswalks = self._extract_crosswalks(ctx)
        if not crosswalks:
            return ViolationResult(
                severity=0.0,
                explanation="No crosswalks in scenario",
            )

        ego = ctx.ego
        pedestrians = ctx.pedestrians
        T = len(ego.x)
        dt = ctx.dt

        ego_length = getattr(ego, "length", DEFAULT_EGO_LENGTH)
        ego_width = getattr(ego, "width", DEFAULT_EGO_WIDTH)

        overlap_areas = []
        violation_frames: List[int] = []

        for t in range(T):
            if not ego.is_valid_at(t):
                overlap_areas.append(0.0)
                continue

            ex, ey, eyaw = ego.x[t], ego.y[t], ego.yaw[t]

            # Create ego bounding box
            corners = oriented_box_corners(ex, ey, eyaw, ego_length, ego_width)
            ego_poly = _corners_to_shapely_polygon(corners)

            if ego_poly is None:
                overlap_areas.append(0.0)
                continue

            max_overlap = 0.0

            for cw_poly in crosswalks:
                if not ego_poly.intersects(cw_poly):
                    continue

                # Check if pedestrians are present
                has_peds = self._check_pedestrians_present(ctx, t, cw_poly, pedestrians)

                if has_peds:
                    overlap = ego_poly.intersection(cw_poly)
                    overlap_area = overlap.area
                    max_overlap = max(max_overlap, overlap_area)

            overlap_areas.append(max_overlap)
            if max_overlap > 0:
                violation_frames.append(t)

        # Calculate severity: integral of overlap area over time
        severity = sum(overlap_areas) * dt
        norm_factor = RULE_NORMALIZATION.get(_RULE_ID, 20.0)
        severity_normalized = min(1.0, severity / norm_factor)

        return ViolationResult(
            severity=severity,
            severity_normalized=severity_normalized,
            measurements={
                "total_overlap_area_time_m2s": severity,
                "max_overlap_area_m2": max(overlap_areas) if overlap_areas else 0,
                "n_violation_frames": len(violation_frames),
            },
            explanation=(
                (
                    f"Occupied crosswalk in {len(violation_frames)} frames, "
                    f"total severity {severity:.2f} m²·s"
                )
                if violation_frames
                else "No crosswalk occupancy violations"
            ),
            frame_violations=violation_frames,
        )

    def _extract_crosswalks(self, ctx: ScenarioContext) -> List:
        """Extract crosswalk polygons from map context."""
        crosswalks = []
        map_ctx = ctx.map_context

        if not map_ctx.has_crosswalks:
            return crosswalks

        for cw_coords in map_ctx.crosswalk_polys:
            if len(cw_coords) >= 3:
                try:
                    poly = Polygon(cw_coords)
                    if poly.is_valid:
                        crosswalks.append(poly)
                except Exception:
                    continue

        return crosswalks

    def _check_pedestrians_present(
        self,
        ctx: ScenarioContext,
        t: int,
        crosswalk_poly,
        pedestrians: List,
    ) -> bool:
        """Check if active pedestrians are near the crosswalk."""
        buffered = crosswalk_poly.buffer(self.detection_buffer_m)

        for ped in pedestrians:
            if not ped.is_valid_at(t):
                continue

            ped_speed = ped.speed[t]
            if ped_speed < self.min_ped_speed_mps:
                continue

            ped_pos = Point(ped.x[t], ped.y[t])
            if buffered.contains(ped_pos):
                return True

        return False
