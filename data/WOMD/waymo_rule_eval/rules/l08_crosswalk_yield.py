"""
L8.R3: Crosswalk Yield Rule

Detects when a vehicle fails to yield to pedestrians in a crosswalk.
The rule checks for pedestrians in or approaching crosswalks and evaluates
whether the vehicle maintains safe time-to-collision (TTC) and appropriate
deceleration behavior.

Violation severity is proportional to:
- Time-to-collision with pedestrians (lower TTC = more severe)
- Distance to crosswalk when pedestrian is present
- Vehicle speed when approaching occupied crosswalk
"""

from typing import Any, Dict, List, Optional

import numpy as np

from ..core.context import Agent, ScenarioContext
from ..utils.constants import MIN_MOVING_SPEED_MPS
from .base import (
    ApplicabilityDetector,
    ApplicabilityResult,
    ViolationEvaluator,
    ViolationResult,
)

_RULE_ID = "L8.R3"
_RULE_LEVEL = 8
_RULE_NAME = "Crosswalk Yield"

# Normalization for severity
RULE_NORM = 10.0

# Default thresholds
DEFAULT_MIN_TTC_THRESHOLD_S = 3.0
DEFAULT_CROSSWALK_PROXIMITY_M = 15.0
DEFAULT_VRU_PROXIMITY_TO_CROSSWALK_M = 5.0


class CrosswalkYieldApplicability(ApplicabilityDetector):
    """
    Checks if crosswalk yield detection is applicable.

    Applicable when:
    - Map contains crosswalk polygons
    - Pedestrians or cyclists (VRUs) are present in the scenario
    - Vehicle is moving (speed > minimum threshold)
    """

    rule_id = _RULE_ID
    level = _RULE_LEVEL
    name = _RULE_NAME

    def __init__(
        self,
        min_ego_speed_mps: float = MIN_MOVING_SPEED_MPS,
    ):
        """
        Initialize crosswalk yield applicability detector.

        Args:
            min_ego_speed_mps: Minimum speed to consider vehicle moving
        """
        self.min_ego_speed_mps = min_ego_speed_mps

    def detect(self, ctx: ScenarioContext) -> ApplicabilityResult:
        """
        Check if the scenario has the required information for crosswalk yield detection.

        Args:
            ctx: The scenario context containing map and agent data

        Returns:
            ApplicabilityResult with applies=True if detection can be performed
        """
        # Validate ego
        if ctx.ego is None or not hasattr(ctx.ego, "x"):
            return ApplicabilityResult(
                applies=False,
                confidence=0.0,
                reasons=["No ego state data available"],
                features={},
                rule_id=self.rule_id,
                rule_level=self.level,
                name=self.name,
            )

        # Need valid map context with crosswalk polygons
        if ctx.map is None:
            return ApplicabilityResult(
                applies=False,
                confidence=1.0,
                reasons=["No map context available"],
                features={},
                rule_id=self.rule_id,
                rule_level=self.level,
                name=self.name,
            )

        # Check if we have crosswalk information
        if not hasattr(ctx.map, "crosswalk_polys") or ctx.map.crosswalk_polys is None:
            return ApplicabilityResult(
                applies=False,
                confidence=1.0,
                reasons=["No crosswalk data in map"],
                features={},
                rule_id=self.rule_id,
                rule_level=self.level,
                name=self.name,
            )

        if len(ctx.map.crosswalk_polys) == 0:
            return ApplicabilityResult(
                applies=False,
                confidence=1.0,
                reasons=["No crosswalks in scene"],
                features={},
                rule_id=self.rule_id,
                rule_level=self.level,
                name=self.name,
            )

        # Check if we have vulnerable road users (pedestrians, cyclists)
        vrus = [agent for agent in ctx.agents if agent.is_vru]

        if len(vrus) == 0:
            return ApplicabilityResult(
                applies=False,
                confidence=1.0,
                reasons=["No pedestrians or cyclists present"],
                features={},
                rule_id=self.rule_id,
                rule_level=self.level,
                name=self.name,
            )

        # Check if vehicle is moving
        if np.max(ctx.ego.speed) < self.min_ego_speed_mps:
            return ApplicabilityResult(
                applies=False,
                confidence=1.0,
                reasons=[
                    f"Vehicle not moving (max speed {np.max(ctx.ego.speed):.2f} < {self.min_ego_speed_mps:.2f} m/s)"
                ],
                features={},
                rule_id=self.rule_id,
                rule_level=self.level,
                name=self.name,
            )

        # Prepare features for evaluator
        features = {
            "vrus": vrus,
            "crosswalk_polys": ctx.map.crosswalk_polys,
        }

        return ApplicabilityResult(
            applies=True,
            confidence=1.0,
            reasons=["Crosswalks and VRUs present"],
            features=features,
            rule_id=self.rule_id,
            rule_level=self.level,
            name=self.name,
        )


class CrosswalkYieldViolation(ViolationEvaluator):
    """
    Evaluates crosswalk yield violations.

    A violation occurs when:
    1. A pedestrian/cyclist is in or near a crosswalk
    2. Vehicle is approaching the crosswalk
    3. Time-to-collision (TTC) falls below safety threshold
    4. Vehicle does not decelerate appropriately

    Measurements:
    - min_ttc_s: Minimum time-to-collision with VRU
    - min_distance_m: Closest approach distance to crosswalk with VRU
    - vru_in_crosswalk: Whether VRU was inside crosswalk bounds
    - ego_speed_at_min_ttc_mps: Vehicle speed at minimum TTC
    - vru_id: ID of the VRU involved
    """

    rule_id = _RULE_ID
    level = 8
    name = _RULE_NAME

    def __init__(
        self,
        min_ttc_threshold_s: float = DEFAULT_MIN_TTC_THRESHOLD_S,
        crosswalk_proximity_m: float = DEFAULT_CROSSWALK_PROXIMITY_M,
        vru_crosswalk_proximity_m: float = DEFAULT_VRU_PROXIMITY_TO_CROSSWALK_M,
        min_ego_speed_mps: float = MIN_MOVING_SPEED_MPS,
        max_severity: float = RULE_NORM,
    ):
        """
        Initialize crosswalk yield violation evaluator.

        Args:
            min_ttc_threshold_s: TTC threshold for violation (default: 3.0s)
            crosswalk_proximity_m: Max ego distance to crosswalk (default: 15m)
            vru_crosswalk_proximity_m: Max VRU distance to crosswalk (default: 5m)
            min_ego_speed_mps: Minimum ego speed for evaluation
            max_severity: Maximum severity score
        """
        self.min_ttc_threshold_s = min_ttc_threshold_s
        self.crosswalk_proximity_m = crosswalk_proximity_m
        self.vru_crosswalk_proximity_m = vru_crosswalk_proximity_m
        self.min_ego_speed_mps = min_ego_speed_mps
        self.max_severity = max_severity

    def evaluate(
        self, ctx: ScenarioContext, applicability: ApplicabilityResult
    ) -> ViolationResult:
        """
        Evaluate crosswalk yield violation.

        Args:
            ctx: The scenario context
            applicability: Result from applicability detector

        Returns:
            ViolationResult with severity and measurements
        """
        if not applicability.applies:
            return ViolationResult(
                severity=0.0,
                severity_normalized=0.0,
                measurements={},
                explanation="Rule not applicable",
                frame_violations=[],
                rule_id=self.rule_id,
                name=self.name,
            )

        features = applicability.features
        vrus = features.get("vrus", [])
        crosswalk_polys = features.get("crosswalk_polys", [])

        ego_x = np.atleast_1d(ctx.ego.x)
        ego_y = np.atleast_1d(ctx.ego.y)
        ego_speed = np.atleast_1d(ctx.ego.speed)
        dt = ctx.dt

        n_frames = len(ego_x)

        if n_frames == 0 or len(vrus) == 0:
            return ViolationResult(
                severity=0.0,
                severity_normalized=0.0,
                measurements={},
                explanation="No data to evaluate",
                frame_violations=[],
                rule_id=self.rule_id,
                name=self.name,
            )

        # Track minimum TTC across all frames and VRUs
        min_ttc = float("inf")
        min_ttc_frame = -1
        vru_involved = None
        vru_in_crosswalk_flag = False
        min_distance_to_crosswalk = float("inf")
        violation_frames = []

        # For each frame, check if any VRU is in/near a crosswalk and compute TTC
        for t in range(n_frames):
            ego_pos = np.array([ego_x[t], ego_y[t]])
            ego_speed_val = ego_speed[t]

            # Skip if ego is not moving
            if ego_speed_val < self.min_ego_speed_mps:
                continue

            # Check each VRU
            for vru in vrus:
                # Get VRU position at this frame
                if t >= len(vru.x) or not vru.is_valid_at(t):
                    continue

                vru_pos = np.array([vru.x[t], vru.y[t]])

                # Check if VRU is in or near any crosswalk
                for crosswalk_poly in crosswalk_polys:
                    crosswalk_arr = np.array(crosswalk_poly)
                    if len(crosswalk_arr) < 3:
                        continue

                    # Check if VRU is inside crosswalk
                    vru_in_crosswalk = self._point_in_polygon(vru_pos, crosswalk_arr)

                    # Check distance from VRU to crosswalk
                    vru_dist_to_crosswalk = self._point_to_polygon_distance(
                        vru_pos, crosswalk_arr
                    )

                    # Check if VRU is in or approaching crosswalk
                    if (
                        vru_in_crosswalk
                        or vru_dist_to_crosswalk < self.vru_crosswalk_proximity_m
                    ):
                        # Compute ego distance to crosswalk
                        ego_dist_to_crosswalk = self._point_to_polygon_distance(
                            ego_pos, crosswalk_arr
                        )

                        # Only consider if ego is approaching (within proximity threshold)
                        if ego_dist_to_crosswalk < self.crosswalk_proximity_m:
                            # Compute TTC (distance / speed)
                            ego_to_vru = np.linalg.norm(vru_pos - ego_pos)

                            if ego_speed_val > 0.1:
                                ttc = ego_to_vru / ego_speed_val

                                if ttc < self.min_ttc_threshold_s:
                                    violation_frames.append(t)

                                if ttc < min_ttc:
                                    min_ttc = ttc
                                    min_ttc_frame = t
                                    vru_involved = vru
                                    vru_in_crosswalk_flag = vru_in_crosswalk
                                    min_distance_to_crosswalk = ego_dist_to_crosswalk

        # Check if violation occurred
        if min_ttc < self.min_ttc_threshold_s and min_ttc_frame >= 0:
            ego_speed_at_min_ttc = float(ego_speed[min_ttc_frame])
            first_violation_time_s = float(min_ttc_frame * dt)

            # Compute severity based on TTC, speed, and proximity
            # TTC contribution (0-5): lower TTC = worse
            ttc_severity = min(5.0, (self.min_ttc_threshold_s - min_ttc) * 2.0)

            # Speed contribution (0-3): faster = worse
            speed_severity = min(3.0, ego_speed_at_min_ttc / 10.0)

            # Proximity contribution (0-2): closer to crosswalk = worse
            proximity_severity = min(
                2.0, (self.crosswalk_proximity_m - min_distance_to_crosswalk) / 7.5
            )

            severity = ttc_severity + speed_severity + proximity_severity
            severity = min(severity, self.max_severity)

            # Measurements
            measurements = {
                "min_ttc_s": float(min_ttc),
                "min_distance_m": min_distance_to_crosswalk,
                "vru_in_crosswalk": vru_in_crosswalk_flag,
                "ego_speed_at_min_ttc_mps": ego_speed_at_min_ttc,
                "vru_id": vru_involved.id if vru_involved else -1,
                "vru_type": vru_involved.type if vru_involved else "unknown",
                "first_violation_time_s": first_violation_time_s,
                "violation_frame_count": len(set(violation_frames)),
            }

            explanation = [
                f"Failed to yield to {measurements['vru_type']} (ID {measurements['vru_id']})",
                f"Minimum TTC: {min_ttc:.2f}s (threshold: {self.min_ttc_threshold_s:.1f}s)",
                f"Distance to crosswalk: {min_distance_to_crosswalk:.1f}m",
                f"Vehicle speed: {ego_speed_at_min_ttc:.1f} m/s",
            ]

            if vru_in_crosswalk_flag:
                explanation.append("VRU was inside crosswalk bounds")

            return ViolationResult(
                severity=severity,
                severity_normalized=min(1.0, severity / self.max_severity),
                measurements=measurements,
                explanation=explanation,
                frame_violations=list(set(violation_frames)),
                rule_id=self.rule_id,
                name=self.name,
            )

        # No violation
        return ViolationResult(
            severity=0.0,
            severity_normalized=0.0,
            measurements={
                "min_ttc_s": float(min_ttc) if min_ttc < float("inf") else None,
                "vru_count": len(vrus),
                "crosswalk_count": len(crosswalk_polys),
            },
            explanation="Safe yielding to VRUs at crosswalks",
            frame_violations=[],
            rule_id=self.rule_id,
            name=self.name,
        )

    def _point_in_polygon(self, point: np.ndarray, polygon: np.ndarray) -> bool:
        """
        Check if a point is inside a polygon using ray casting algorithm.

        Args:
            point: (2,) array [x, y]
            polygon: (N, 2) array of polygon vertices

        Returns:
            True if point is inside polygon
        """
        x, y = point[0], point[1]
        n = len(polygon)
        inside = False

        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    def _point_to_polygon_distance(
        self, point: np.ndarray, polygon: np.ndarray
    ) -> float:
        """
        Compute minimum distance from point to polygon boundary.

        Args:
            point: (2,) array [x, y]
            polygon: (N, 2) array of polygon vertices

        Returns:
            Minimum distance to any edge of the polygon
        """
        min_dist = float("inf")

        # Check distance to each edge
        for i in range(len(polygon)):
            p1 = polygon[i]
            p2 = polygon[(i + 1) % len(polygon)]

            dist = self._point_to_segment_distance(point, p1, p2)
            min_dist = min(min_dist, dist)

        return min_dist

    def _point_to_segment_distance(
        self, point: np.ndarray, seg_start: np.ndarray, seg_end: np.ndarray
    ) -> float:
        """
        Compute distance from point to line segment.

        Args:
            point: (2,) array
            seg_start: (2,) array, segment start point
            seg_end: (2,) array, segment end point

        Returns:
            Distance to segment
        """
        seg_vec = seg_end - seg_start
        seg_len_sq = np.dot(seg_vec, seg_vec)

        if seg_len_sq < 1e-8:
            return np.linalg.norm(point - seg_start)

        # Project point onto line
        t = np.dot(point - seg_start, seg_vec) / seg_len_sq
        t = np.clip(t, 0.0, 1.0)

        projection = seg_start + t * seg_vec
        return np.linalg.norm(point - projection)
