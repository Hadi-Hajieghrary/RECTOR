"""
L6.R2: Following Distance Rule (Two-Second Rule)

Checks if ego maintains safe following distance from lead vehicle.

Features:
- Implements the "two-second rule" for safe following
- Heading-aware: only checks vehicles ahead in ego's lane
- Uses bumper-to-bumper distance, not center-to-center
"""

from dataclasses import dataclass
from typing import List, Optional

from ..core.context import ScenarioContext
from ..core.geometry import compute_bumper_distance
from ..core.temporal_spatial import TemporalSpatialIndex
from ..utils.constants import (
    DEFAULT_EGO_LENGTH,
    DEFAULT_SPATIAL_RADIUS_M,
    FOLLOWING_DISTANCE_MIN_RATIO,
    FOLLOWING_DISTANCE_MIN_SPEED_MS,
    LANE_WIDTH_DEFAULT_M,
    TWO_SECOND_RULE_TIME_S,
)
from .base import (
    ApplicabilityDetector,
    ApplicabilityResult,
    ViolationEvaluator,
    ViolationResult,
)

_RULE_ID = "L6.R2"
_LEVEL = 6
_NAME = "Following Distance"


@dataclass
class FollowingViolationEvent:
    """Details of a following distance violation."""

    frame_idx: int
    lead_vehicle_id: int
    actual_distance_m: float
    required_distance_m: float
    ego_speed_ms: float
    time_gap_s: float


class FollowingDistanceApplicability(ApplicabilityDetector):
    """Following distance applies when there are vehicles."""

    rule_id = _RULE_ID
    level = _LEVEL
    name = _NAME

    def detect(self, ctx: ScenarioContext) -> ApplicabilityResult:
        """Check if there are any vehicles to follow."""
        vehicles = ctx.vehicles
        n_vehicles = len(vehicles)

        if n_vehicles == 0:
            return ApplicabilityResult(
                applies=False,
                confidence=1.0,
                reasons=["No other vehicles in scenario"],
                features={"n_vehicles": 0},
            )

        return ApplicabilityResult(
            applies=True,
            confidence=1.0,
            reasons=[f"Following distance check for {n_vehicles} vehicles"],
            features={"n_vehicles": n_vehicles},
        )


class FollowingDistanceViolation(ViolationEvaluator):
    """Evaluate following distance violations."""

    rule_id = _RULE_ID
    level = _LEVEL
    name = _NAME

    def __init__(
        self,
        time_gap_s: float = TWO_SECOND_RULE_TIME_S,
        min_speed_ms: float = FOLLOWING_DISTANCE_MIN_SPEED_MS,
        min_ratio: float = FOLLOWING_DISTANCE_MIN_RATIO,
        lane_width_m: float = LANE_WIDTH_DEFAULT_M,
    ):
        self.time_gap_s = time_gap_s
        self.min_speed_ms = min_speed_ms
        self.min_ratio = min_ratio
        self.half_lane = lane_width_m / 2.0

    def evaluate(
        self, ctx: ScenarioContext, applicability: ApplicabilityResult
    ) -> ViolationResult:
        """Evaluate following distance violations."""

        ego = ctx.ego
        vehicles = ctx.vehicles
        T = len(ego.x)

        ego_length = getattr(ego, "length", DEFAULT_EGO_LENGTH)

        # Build spatial index for vehicles
        spatial_idx = TemporalSpatialIndex(vehicles, T)

        violation_frames: List[int] = []
        events: List[FollowingViolationEvent] = []
        total_violation_m = 0.0
        worst_time_gap = float("inf")

        for t in range(T):
            if not ego.is_valid_at(t):
                continue

            ex, ey = ego.x[t], ego.y[t]
            eyaw = ego.yaw[t]
            espeed = ego.speed[t]

            # Only apply at higher speeds
            if espeed < self.min_speed_ms:
                continue

            # Required following distance (two-second rule)
            required_dist = espeed * self.time_gap_s

            # Find lead vehicle
            lead = self._find_lead_vehicle(spatial_idx, t, ex, ey, eyaw, ego_length)

            if lead is None:
                continue

            lead_vehicle, actual_dist = lead

            # Check violation
            if actual_dist < required_dist * self.min_ratio:
                violation_amount = required_dist - actual_dist
                time_gap = actual_dist / espeed if espeed > 0 else float("inf")

                if t not in violation_frames:
                    violation_frames.append(t)

                total_violation_m += violation_amount
                worst_time_gap = min(worst_time_gap, time_gap)

                events.append(
                    FollowingViolationEvent(
                        frame_idx=t,
                        lead_vehicle_id=lead_vehicle.id,
                        actual_distance_m=actual_dist,
                        required_distance_m=required_dist,
                        ego_speed_ms=espeed,
                        time_gap_s=time_gap,
                    )
                )

        # Compute severity
        severity = total_violation_m
        severity_normalized = min(1.0, 1.0 - worst_time_gap / self.time_gap_s)
        if worst_time_gap == float("inf"):
            severity_normalized = 0.0

        return ViolationResult(
            severity=severity,
            severity_normalized=severity_normalized,
            measurements={
                "n_violation_frames": len(violation_frames),
                "n_violation_events": len(events),
                "total_violation_m": total_violation_m,
                "worst_time_gap_s": worst_time_gap,
            },
            explanation=(
                (
                    f"{len(violation_frames)} frames with tailgating, "
                    f"worst gap {worst_time_gap:.2f}s"
                )
                if len(violation_frames) > 0
                else "Safe following distance"
            ),
            frame_violations=violation_frames,
        )

    def _find_lead_vehicle(
        self,
        spatial_idx: TemporalSpatialIndex,
        t: int,
        ex: float,
        ey: float,
        eyaw: float,
        ego_length: float,
    ) -> Optional[tuple]:
        """Find the closest vehicle ahead in ego's lane."""
        nearby = spatial_idx.at_frame(t).query_radius(
            ex, ey, DEFAULT_SPATIAL_RADIUS_M, agent_type="vehicle"
        )

        best_lead = None
        best_dist = float("inf")

        for vehicle, center_dist in nearby:
            if not vehicle.is_valid_at(t):
                continue

            vx, vy = vehicle.x[t], vehicle.y[t]
            v_length = getattr(vehicle, "length", 4.5)

            result = compute_bumper_distance(ex, ey, eyaw, ego_length, vx, vy, v_length)

            # Must be ahead
            if not result.is_ahead:
                continue

            # Must be in same lane (roughly)
            if abs(result.lateral) > self.half_lane:
                continue

            # Track closest ahead
            if result.bumper_distance < best_dist:
                best_dist = result.bumper_distance
                best_lead = (vehicle, result.bumper_distance)

        return best_lead
