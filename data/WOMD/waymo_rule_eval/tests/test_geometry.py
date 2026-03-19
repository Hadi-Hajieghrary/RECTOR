"""
Comprehensive Unit Tests for Geometry Utilities.

Tests collision detection, coordinate transforms, and spatial calculations
with various edge cases and realistic scenarios.
"""

import os
import sys

import numpy as np
import pytest

# Add the workspace root to path so waymo_rule_eval package can be imported
_workspace_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _workspace_root not in sys.path:
    sys.path.insert(0, _workspace_root)

from waymo_rule_eval.core.geometry import (
    compute_relative_position,
    get_box_separating_axes,
    normalize_angle,
    oriented_box_corners,
    point_in_polygon,
    project_polygon_onto_axis,
    sat_collision_check,
)


class TestRelativePosition:
    """Tests for heading-aware coordinate transform."""

    def test_target_ahead_no_rotation(self):
        """Target directly ahead, ego heading +X."""
        lon, lat = compute_relative_position(
            ego_x=0, ego_y=0, ego_yaw=0, target_x=10, target_y=0
        )
        assert lon == pytest.approx(10.0, abs=0.01)
        assert lat == pytest.approx(0.0, abs=0.01)

    def test_target_left_no_rotation(self):
        """Target to the left, ego heading +X."""
        lon, lat = compute_relative_position(
            ego_x=0, ego_y=0, ego_yaw=0, target_x=0, target_y=10
        )
        assert lon == pytest.approx(0.0, abs=0.01)
        assert lat == pytest.approx(10.0, abs=0.01)

    def test_target_right_no_rotation(self):
        """Target to the right, ego heading +X."""
        lon, lat = compute_relative_position(
            ego_x=0, ego_y=0, ego_yaw=0, target_x=0, target_y=-10
        )
        assert lon == pytest.approx(0.0, abs=0.01)
        assert lat == pytest.approx(-10.0, abs=0.01)

    def test_target_behind_no_rotation(self):
        """Target behind, ego heading +X."""
        lon, lat = compute_relative_position(
            ego_x=0, ego_y=0, ego_yaw=0, target_x=-10, target_y=0
        )
        assert lon == pytest.approx(-10.0, abs=0.01)
        assert lat == pytest.approx(0.0, abs=0.01)

    def test_ego_heading_plus_y(self):
        """Ego heading +Y (pi/2), target ahead."""
        lon, lat = compute_relative_position(
            ego_x=0, ego_y=0, ego_yaw=np.pi / 2, target_x=0, target_y=10
        )
        assert lon == pytest.approx(10.0, abs=0.01)
        assert lat == pytest.approx(0.0, abs=0.01)

    def test_ego_heading_minus_x(self):
        """Ego heading -X (pi), target ahead."""
        lon, lat = compute_relative_position(
            ego_x=0, ego_y=0, ego_yaw=np.pi, target_x=-10, target_y=0
        )
        assert lon == pytest.approx(10.0, abs=0.01)
        assert lat == pytest.approx(0.0, abs=0.01)

    def test_diagonal_target(self):
        """Target at 45 degrees, ego heading +X."""
        lon, lat = compute_relative_position(
            ego_x=0, ego_y=0, ego_yaw=0, target_x=10, target_y=10
        )
        assert lon == pytest.approx(10.0, abs=0.01)
        assert lat == pytest.approx(10.0, abs=0.01)

    def test_offset_ego_position(self):
        """Ego not at origin."""
        lon, lat = compute_relative_position(
            ego_x=5, ego_y=5, ego_yaw=0, target_x=15, target_y=5
        )
        assert lon == pytest.approx(10.0, abs=0.01)
        assert lat == pytest.approx(0.0, abs=0.01)

    def test_negative_heading(self):
        """Negative heading angle."""
        lon, lat = compute_relative_position(
            ego_x=0, ego_y=0, ego_yaw=-np.pi / 2, target_x=0, target_y=-10
        )
        assert lon == pytest.approx(10.0, abs=0.01)
        assert lat == pytest.approx(0.0, abs=0.01)


class TestOrientedBoxCorners:
    """Tests for oriented bounding box corner computation."""

    def test_box_at_origin_no_rotation(self):
        """Box at origin with no rotation."""
        corners = oriented_box_corners(cx=0, cy=0, yaw=0, length=4.0, width=2.0)

        assert corners.shape == (4, 2)

        # Front-right: (length/2, -width/2) = (2, -1)
        assert corners[0, 0] == pytest.approx(2.0, abs=0.01)
        assert corners[0, 1] == pytest.approx(-1.0, abs=0.01)

        # Front-left: (2, 1)
        assert corners[1, 0] == pytest.approx(2.0, abs=0.01)
        assert corners[1, 1] == pytest.approx(1.0, abs=0.01)

        # Rear-left: (-2, 1)
        assert corners[2, 0] == pytest.approx(-2.0, abs=0.01)
        assert corners[2, 1] == pytest.approx(1.0, abs=0.01)

        # Rear-right: (-2, -1)
        assert corners[3, 0] == pytest.approx(-2.0, abs=0.01)
        assert corners[3, 1] == pytest.approx(-1.0, abs=0.01)

    def test_box_rotated_90_degrees(self):
        """Box rotated 90 degrees (heading +Y)."""
        corners = oriented_box_corners(cx=0, cy=0, yaw=np.pi / 2, length=4.0, width=2.0)

        # Front-right (2, -1) rotated 90deg -> (1, 2)
        assert corners[0, 0] == pytest.approx(1.0, abs=0.01)
        assert corners[0, 1] == pytest.approx(2.0, abs=0.01)

    def test_box_offset_position(self):
        """Box at offset position."""
        corners = oriented_box_corners(cx=10, cy=5, yaw=0, length=4.0, width=2.0)

        # Front-right: (10+2, 5-1) = (12, 4)
        assert corners[0, 0] == pytest.approx(12.0, abs=0.01)
        assert corners[0, 1] == pytest.approx(4.0, abs=0.01)

    def test_box_rotated_45_degrees(self):
        """Box rotated 45 degrees."""
        corners = oriented_box_corners(cx=0, cy=0, yaw=np.pi / 4, length=4.0, width=2.0)

        # Front-right before rotation: (2, -1)
        expected_x = 2 * np.cos(np.pi / 4) - (-1) * np.sin(np.pi / 4)
        expected_y = 2 * np.sin(np.pi / 4) + (-1) * np.cos(np.pi / 4)

        assert corners[0, 0] == pytest.approx(expected_x, abs=0.01)
        assert corners[0, 1] == pytest.approx(expected_y, abs=0.01)

    def test_square_box(self):
        """Square box (length == width)."""
        corners = oriented_box_corners(cx=0, cy=0, yaw=0, length=2.0, width=2.0)

        # All corners at distance sqrt(2) from center
        for corner in corners:
            dist = np.sqrt(corner[0] ** 2 + corner[1] ** 2)
            assert dist == pytest.approx(np.sqrt(2), abs=0.01)


class TestSATCollisionCheck:
    """Tests for Separating Axis Theorem collision detection."""

    def test_clearly_separated_boxes(self):
        """Two boxes clearly separated."""
        box1 = oriented_box_corners(0, 0, 0, 4.0, 2.0)
        box2 = oriented_box_corners(10, 0, 0, 4.0, 2.0)

        axes1 = get_box_separating_axes(0)
        axes2 = get_box_separating_axes(0)

        collides, penetration = sat_collision_check(box1, axes1, box2, axes2)

        assert collides is False
        assert penetration == 0.0

    def test_overlapping_boxes(self):
        """Two overlapping boxes."""
        box1 = oriented_box_corners(0, 0, 0, 4.0, 2.0)
        box2 = oriented_box_corners(3, 0, 0, 4.0, 2.0)  # 1m overlap

        axes1 = get_box_separating_axes(0)
        axes2 = get_box_separating_axes(0)

        collides, penetration = sat_collision_check(box1, axes1, box2, axes2)

        assert collides is True
        assert penetration == pytest.approx(1.0, abs=0.1)

    def test_touching_boxes(self):
        """Two boxes just touching (edge contact)."""
        box1 = oriented_box_corners(0, 0, 0, 4.0, 2.0)
        box2 = oriented_box_corners(4, 0, 0, 4.0, 2.0)  # Exactly touching

        axes1 = get_box_separating_axes(0)
        axes2 = get_box_separating_axes(0)

        collides, penetration = sat_collision_check(box1, axes1, box2, axes2)

        # Edge contact should be considered collision
        assert collides is True
        assert penetration == pytest.approx(0.0, abs=0.1)

    def test_rotated_overlapping_boxes(self):
        """Two rotated boxes that overlap."""
        box1 = oriented_box_corners(0, 0, np.pi / 4, 4.0, 2.0)
        box2 = oriented_box_corners(2, 0, -np.pi / 4, 4.0, 2.0)

        axes1 = get_box_separating_axes(np.pi / 4)
        axes2 = get_box_separating_axes(-np.pi / 4)

        collides, penetration = sat_collision_check(box1, axes1, box2, axes2)

        assert collides is True
        assert penetration > 0

    def test_laterally_separated_boxes(self):
        """Two boxes separated laterally."""
        box1 = oriented_box_corners(0, 0, 0, 4.0, 2.0)
        box2 = oriented_box_corners(0, 5, 0, 4.0, 2.0)  # 5m lateral offset

        axes1 = get_box_separating_axes(0)
        axes2 = get_box_separating_axes(0)

        collides, penetration = sat_collision_check(box1, axes1, box2, axes2)

        assert collides is False

    def test_contained_box(self):
        """Small box completely inside large box."""
        box1 = oriented_box_corners(0, 0, 0, 10.0, 10.0)  # Large box
        box2 = oriented_box_corners(0, 0, 0, 2.0, 2.0)  # Small box inside

        axes1 = get_box_separating_axes(0)
        axes2 = get_box_separating_axes(0)

        collides, penetration = sat_collision_check(box1, axes1, box2, axes2)

        assert collides is True
        assert penetration > 0


class TestProjectPolygon:
    """Tests for polygon projection onto axis."""

    def test_axis_aligned_projection(self):
        """Project square onto x-axis."""
        polygon = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1]])
        axis = np.array([1, 0])  # X-axis

        proj_min, proj_max = project_polygon_onto_axis(polygon, axis)

        assert proj_min == pytest.approx(-1.0, abs=0.01)
        assert proj_max == pytest.approx(1.0, abs=0.01)

    def test_diagonal_projection(self):
        """Project square onto diagonal axis."""
        polygon = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1]])
        axis = np.array([1, 1]) / np.sqrt(2)  # Normalized diagonal

        proj_min, proj_max = project_polygon_onto_axis(polygon, axis)

        # Corners at (1,1) and (-1,-1) project to sqrt(2) and -sqrt(2)
        assert proj_min == pytest.approx(-np.sqrt(2), abs=0.01)
        assert proj_max == pytest.approx(np.sqrt(2), abs=0.01)


class TestPointInPolygon:
    """Tests for point-in-polygon check."""

    def test_point_inside_square(self):
        """Point clearly inside square."""
        polygon = np.array([[0, 0], [4, 0], [4, 4], [0, 4]])

        assert point_in_polygon((2, 2), polygon) is True

    def test_point_outside_square(self):
        """Point clearly outside square."""
        polygon = np.array([[0, 0], [4, 0], [4, 4], [0, 4]])

        assert point_in_polygon((10, 10), polygon) is False

    def test_point_on_edge(self):
        """Point on polygon edge."""
        polygon = np.array([[0, 0], [4, 0], [4, 4], [0, 4]])

        # Edge case - on boundary
        result = point_in_polygon((2, 0), polygon)
        # Implementation may vary for boundary points
        assert isinstance(result, bool)

    def test_point_at_vertex(self):
        """Point at polygon vertex."""
        polygon = np.array([[0, 0], [4, 0], [4, 4], [0, 4]])

        result = point_in_polygon((0, 0), polygon)
        assert isinstance(result, bool)

    def test_complex_polygon(self):
        """Point inside complex polygon."""
        # L-shaped polygon
        polygon = np.array([[0, 0], [6, 0], [6, 2], [2, 2], [2, 4], [0, 4]])

        assert point_in_polygon((1, 1), polygon) is True  # Bottom part
        assert point_in_polygon((1, 3), polygon) is True  # Left part
        assert point_in_polygon((4, 3), polygon) is False  # Cutout area


class TestNormalizeAngle:
    """Tests for angle normalization to [-pi, pi]."""

    def test_already_normalized(self):
        """Angle already in range."""
        assert normalize_angle(0.5) == pytest.approx(0.5, abs=0.01)
        assert normalize_angle(-0.5) == pytest.approx(-0.5, abs=0.01)

    def test_wrap_positive(self):
        """Positive angle wrapping."""
        assert normalize_angle(2 * np.pi) == pytest.approx(0.0, abs=0.01)
        # 3*pi wraps to pi (or -pi depending on implementation)
        result = normalize_angle(3 * np.pi)
        assert abs(result) == pytest.approx(np.pi, abs=0.01)
        result = normalize_angle(2.5 * np.pi)
        assert result == pytest.approx(0.5 * np.pi, abs=0.01)

    def test_wrap_negative(self):
        """Negative angle wrapping."""
        assert normalize_angle(-2 * np.pi) == pytest.approx(0.0, abs=0.01)
        # -3*pi wraps to pi (or -pi)
        result = normalize_angle(-3 * np.pi)
        assert abs(result) == pytest.approx(np.pi, abs=0.01)

    def test_large_angles(self):
        """Very large angles."""
        assert normalize_angle(10 * np.pi) == pytest.approx(0.0, abs=0.01)
        assert normalize_angle(-10 * np.pi) == pytest.approx(0.0, abs=0.01)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
