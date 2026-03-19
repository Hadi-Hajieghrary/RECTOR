"""
L0.R2: Safe Longitudinal Distance Rule

Maintains safe following distance behind leading vehicles.
Requires minimum 2-second time gap based on current speed.

Standards:
- Minimum 2-second time gap at current speed
- Bumper-to-bumper distance calculation
- Only applies when following another vehicle in same lane
"""

from typing import List, Optional, Tuple

import numpy as np

from ..core.context import ScenarioContext
from ..core.geometry import compute_bumper_distance
from ..core.temporal_spatial import TemporalSpatialIndex
from ..utils.constants import (
    DEFAULT_EGO_LENGTH,
    FOLLOWING_DISTANCE_MIN_SPEED_MS,
    LANE_WIDTH_DEFAULT_M,
    RULE_NORMALIZATION,
    SAFE_FOLLOWING_DETECTION_RANGE_M,
    SAFE_FOLLOWING_TIME_GAP_S,
)
from .base import (
    ApplicabilityDetector,
    ApplicabilityResult,
    ViolationEvaluator,
    ViolationResult,
)

_RULE_ID = "L0.R2"
_LEVEL = 0
_NAME = "Safe Longitudinal Distance"


class SafeLongitudinalDistanceApplicability(ApplicabilityDetector):
    """
    Detects when safe following distance rule applies.

    Rule applies when:
    - Ego vehicle has a leading vehicle in same lane
    - Both vehicles are moving (speed > threshold)
    - Within detection range
    """

    rule_id = _RULE_ID
    level = _LEVEL
    name = _NAME

    def __init__(
        self,
        detection_range_m: float = SAFE_FOLLOWING_DETECTION_RANGE_M,
        min_speed_mps: float = FOLLOWING_DISTANCE_MIN_SPEED_MS,
        lane_tolerance_m: float = LANE_WIDTH_DEFAULT_M / 2.0,
    ):
        self.detection_range_m = detection_range_m
        self.min_speed_mps = min_speed_mps
        self.lane_tolerance_m = lane_tolerance_m

    def detect(self, ctx: ScenarioContext) -> ApplicabilityResult:
        """Detect if rule is applicable."""
        applicable_frames = 0
        T = len(ctx.ego.x)
        ego = ctx.ego
        vehicles = ctx.vehicles

        if not vehicles:
            return ApplicabilityResult(
                applies=False,
                confidence=0.0,
                reasons=["No other vehicles in scenario"],
                features={"n_vehicles": 0},
            )

        spatial_idx = TemporalSpatialIndex(vehicles, T)

        for t in range(T):
            if not ego.is_valid_at(t):
                continue

            ego_speed = ego.speed[t]
            if ego_speed < self.min_speed_mps:
                continue

            ex, ey, eyaw = ego.x[t], ego.y[t], ego.yaw[t]
            ego_length = getattr(ego, "length", DEFAULT_EGO_LENGTH)

            # Find leading vehicle
            lead = self._find_lead_vehicle(spatial_idx, t, ex, ey, eyaw, ego_length)

            if lead is not None:
                applicable_frames += 1

        applies = applicable_frames > 0
        confidence = applicable_frames / T if T > 0 else 0.0

        return ApplicabilityResult(
            applies=applies,
            confidence=confidence,
            reasons=[f"Leading vehicle detected in {applicable_frames}/{T} frames"],
            features={
                "applicable_frames": applicable_frames,
                "n_vehicles": len(vehicles),
            },
        )

    def _find_lead_vehicle(
        self,
        spatial_idx: TemporalSpatialIndex,
        t: int,
        ex: float,
        ey: float,
        eyaw: float,
        ego_length: float,
    ) -> Optional[Tuple]:
        """Find the closest vehicle ahead in ego's lane."""
        nearby = spatial_idx.at_frame(t).query_radius(ex, ey, self.detection_range_m)

        for vehicle, _ in nearby:
            if not vehicle.is_valid_at(t):
                continue

            vx, vy = vehicle.x[t], vehicle.y[t]
            v_length = getattr(vehicle, "length", 4.5)

            result = compute_bumper_distance(ex, ey, eyaw, ego_length, vx, vy, v_length)

            # Must be ahead and in same lane
            if result.is_ahead and abs(result.lateral) < self.lane_tolerance_m:
                return (vehicle, result.bumper_distance)

        return None


class SafeLongitudinalDistanceViolation(ViolationEvaluator):
    """
    Evaluate safe following distance violations.

    Violation occurs when:
    - Time headway < min_time_gap_s (default 2 seconds)
    - Uses bumper-to-bumper distance
    """

    rule_id = _RULE_ID
    level = _LEVEL
    name = _NAME

    def __init__(
        self,
        min_time_gap_s: float = SAFE_FOLLOWING_TIME_GAP_S,
        detection_range_m: float = SAFE_FOLLOWING_DETECTION_RANGE_M,
        lane_tolerance_m: float = LANE_WIDTH_DEFAULT_M / 2.0,
    ):
        self.min_time_gap_s = min_time_gap_s
        self.detection_range_m = detection_range_m
        self.lane_tolerance_m = lane_tolerance_m

    def evaluate(
        self, ctx: ScenarioContext, applicability: ApplicabilityResult
    ) -> ViolationResult:
        """Compute violation severity."""
        ego = ctx.ego
        vehicles = ctx.vehicles
        T = len(ego.x)
        dt = ctx.dt

        spatial_idx = TemporalSpatialIndex(vehicles, T)
        ego_length = getattr(ego, "length", DEFAULT_EGO_LENGTH)

        gap_deficits: List[float] = []
        violation_frames: List[int] = []
        worst_time_gap = float("inf")

        for t in range(T):
            if not ego.is_valid_at(t):
                gap_deficits.append(0.0)
                continue

            ego_speed = ego.speed[t]
            if ego_speed < FOLLOWING_DISTANCE_MIN_SPEED_MS:
                gap_deficits.append(0.0)
                continue

            ex, ey, eyaw = ego.x[t], ego.y[t], ego.yaw[t]

            # Required distance based on 2-second rule
            required_dist = ego_speed * self.min_time_gap_s

            # Find closest lead vehicle
            nearby = spatial_idx.at_frame(t).query_radius(
                ex, ey, self.detection_range_m
            )

            min_gap = float("inf")
            for vehicle, _ in nearby:
                if not vehicle.is_valid_at(t):
                    continue

                vx, vy = vehicle.x[t], vehicle.y[t]
                v_length = getattr(vehicle, "length", 4.5)

                result = compute_bumper_distance(
                    ex, ey, eyaw, ego_length, vx, vy, v_length
                )

                if result.is_ahead and abs(result.lateral) < self.lane_tolerance_m:
                    if result.bumper_distance < min_gap:
                        min_gap = result.bumper_distance

            if min_gap < float("inf"):
                time_gap = min_gap / ego_speed if ego_speed > 0 else float("inf")
                worst_time_gap = min(worst_time_gap, time_gap)

                if min_gap < required_dist:
                    deficit = required_dist - min_gap
                    gap_deficits.append(deficit)
                    violation_frames.append(t)
                else:
                    gap_deficits.append(0.0)
            else:
                gap_deficits.append(0.0)

        # Severity: integral of gap deficit over time
        severity = sum(gap_deficits) * dt
        norm_factor = RULE_NORMALIZATION.get(_RULE_ID, 20.0)
        severity_normalized = min(1.0, severity / norm_factor)

        if worst_time_gap == float("inf"):
            worst_time_gap = self.min_time_gap_s  # No leader found

        return ViolationResult(
            severity=severity,
            severity_normalized=severity_normalized,
            measurements={
                "n_violation_frames": len(violation_frames),
                "worst_time_gap_s": worst_time_gap,
                "total_deficit_m": sum(gap_deficits),
            },
            explanation=(
                (
                    f"Following distance violations in {len(violation_frames)} "
                    f"frames, worst time gap {worst_time_gap:.2f}s"
                )
                if violation_frames
                else "Safe following distance maintained"
            ),
            frame_violations=violation_frames,
        )
