"""
Geometry utilities for heading-aware spatial reasoning.

Contains functions for:
- Heading-aware coordinate transforms
- Oriented bounding box operations
- SAT (Separating Axis Theorem) collision detection
- Bumper-to-bumper distance calculation
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


def normalize_angle(angle):
    """Normalize angle to [-pi, pi]. Accepts scalar or array."""
    return (np.asarray(angle) + np.pi) % (2 * np.pi) - np.pi


def angle_diff(a: float, b: float) -> float:
    """Signed angular difference (a - b), normalized to [-pi, pi]."""
    return normalize_angle(a - b)


def compute_relative_position(
    ego_x: float, ego_y: float, ego_yaw: float, target_x: float, target_y: float
) -> Tuple[float, float]:
    """Target position in ego frame. Returns (longitudinal, lateral)."""
    dx = target_x - ego_x
    dy = target_y - ego_y

    cos_yaw = np.cos(ego_yaw)
    sin_yaw = np.sin(ego_yaw)

    # Rotate to ego frame
    longitudinal = dx * cos_yaw + dy * sin_yaw
    lateral = -dx * sin_yaw + dy * cos_yaw

    return longitudinal, lateral


def is_ahead_of_ego(
    ego_x: float,
    ego_y: float,
    ego_yaw: float,
    target_x: float,
    target_y: float,
    min_longitudinal: float = 0.0,
) -> bool:
    """Check if target is ahead of ego by at least min_longitudinal."""
    lon, _ = compute_relative_position(ego_x, ego_y, ego_yaw, target_x, target_y)
    return lon > min_longitudinal


@dataclass
class BumperToBumperResult:
    """Result of bumper-to-bumper distance calculation."""

    center_distance: float  # Center-to-center distance
    bumper_distance: float  # Gap between bumpers
    longitudinal: float  # Longitudinal separation (+ = ahead)
    lateral: float  # Lateral separation (+ = left)
    ego_half_length: float  # Ego half-length
    target_half_length: float  # Target half-length

    @property
    def is_ahead(self) -> bool:
        """Target is ahead of ego."""
        return self.longitudinal > 0

    @property
    def is_in_lane(self) -> bool:
        """Target is roughly in same lane (within typical lane width)."""
        return abs(self.lateral) < 2.0


def compute_bumper_distance(
    ego_x: float,
    ego_y: float,
    ego_yaw: float,
    ego_length: float,
    target_x: float,
    target_y: float,
    target_length: float,
) -> BumperToBumperResult:
    """
    Compute bumper-to-bumper distance accounting for vehicle lengths.

    Calculates the gap between the front bumper of ego and
    the rear bumper of a target vehicle ahead.

    Args:
        ego_x, ego_y, ego_yaw: Ego position and heading
        ego_length: Length of ego vehicle
        target_x, target_y: Target center position
        target_length: Length of target vehicle

    Returns:
        BumperToBumperResult with all distance components
    """
    long_dist, lat_dist = compute_relative_position(
        ego_x, ego_y, ego_yaw, target_x, target_y
    )

    ego_half = ego_length / 2.0
    target_half = target_length / 2.0
    center_dist = np.sqrt(long_dist**2 + lat_dist**2)

    # Bumper distance: subtract half-lengths if target is ahead
    if long_dist > 0:
        # Target is ahead: gap = longitudinal - ego_front - target_rear
        bumper_dist = long_dist - ego_half - target_half
    else:
        # Target is behind: gap = |longitudinal| - target_front - ego_rear
        bumper_dist = abs(long_dist) - ego_half - target_half

    return BumperToBumperResult(
        center_distance=center_dist,
        bumper_distance=bumper_dist,
        longitudinal=long_dist,
        lateral=lat_dist,
        ego_half_length=ego_half,
        target_half_length=target_half,
    )


def oriented_box_corners(
    cx: float, cy: float, yaw: float, length: float, width: float
) -> np.ndarray:
    """Compute 4 corners of an oriented bounding box → (4, 2) array."""
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)

    half_l = length / 2.0
    half_w = width / 2.0

    # Corners in local frame: [front-right, front-left, rear-left, rear-right]
    local_corners = np.array(
        [
            [half_l, -half_w],  # front-right
            [half_l, half_w],  # front-left
            [-half_l, half_w],  # rear-left
            [-half_l, -half_w],  # rear-right
        ]
    )

    # Rotation matrix
    rot = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])

    # Transform to global frame
    global_corners = local_corners @ rot.T + np.array([cx, cy])
    return global_corners


def get_box_separating_axes(yaw: float) -> np.ndarray:
    """Two edge-normal axes for SAT collision detection → (2, 2) array."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, s], [-s, c]])  # Along heading  # Perpendicular to heading


def project_polygon_onto_axis(
    corners: np.ndarray, axis: np.ndarray
) -> Tuple[float, float]:
    """Project polygon corners onto axis → (min_proj, max_proj)."""
    projections = corners @ axis
    return float(np.min(projections)), float(np.max(projections))


def intervals_overlap(
    min1: float, max1: float, min2: float, max2: float
) -> Tuple[bool, float]:
    """Check if two 1D intervals overlap → (overlaps, overlap_depth)."""
    if max1 < min2 or max2 < min1:
        return False, 0.0

    overlap = min(max1, max2) - max(min1, min2)
    return True, overlap


def sat_collision_check(
    box1_corners: np.ndarray,
    box1_axes: np.ndarray,
    box2_corners: np.ndarray,
    box2_axes: np.ndarray,
) -> Tuple[bool, float]:
    """SAT collision check for two oriented boxes → (collides, min_penetration)."""
    min_penetration = float("inf")

    # Check all 4 axes (2 from each box)
    all_axes = np.vstack([box1_axes, box2_axes])

    for axis in all_axes:
        min1, max1 = project_polygon_onto_axis(box1_corners, axis)
        min2, max2 = project_polygon_onto_axis(box2_corners, axis)

        overlaps, overlap_amount = intervals_overlap(min1, max1, min2, max2)

        if not overlaps:
            # Found separating axis - no collision
            return False, 0.0

        min_penetration = min(min_penetration, overlap_amount)

    return True, min_penetration


def point_in_polygon(point: Tuple[float, float], polygon: np.ndarray) -> bool:
    """Check if point is inside polygon (ray casting)."""
    x, y = point
    n = len(polygon)
    inside = False

    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]

        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i

    return inside


def polygon_area(polygon: np.ndarray) -> float:
    """Polygon area via the shoelace formula."""
    n = len(polygon)
    area = 0.0

    for i in range(n):
        j = (i + 1) % n
        area += polygon[i, 0] * polygon[j, 1]
        area -= polygon[j, 0] * polygon[i, 1]

    return abs(area) / 2.0


def polygon_centroid(polygon: np.ndarray) -> Tuple[float, float]:
    """Centroid of a polygon → (cx, cy)."""
    return float(np.mean(polygon[:, 0])), float(np.mean(polygon[:, 1]))
