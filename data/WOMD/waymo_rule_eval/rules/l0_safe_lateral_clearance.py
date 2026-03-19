"""
L0.R3: Safe Lateral Clearance Rule

Maintains safe lateral distance from other road users to prevent side-swipe
collisions and ensure vulnerable road user safety.

Standards:
- Vehicles: 0.5m minimum lateral clearance
- Cyclists: 1.0m minimum lateral clearance
- Pedestrians: 1.5m minimum lateral clearance
"""

from typing import Dict, List

import numpy as np

from ..core.context import ScenarioContext
from ..core.geometry import compute_relative_position
from ..core.temporal_spatial import TemporalSpatialIndex
from ..utils.constants import (
    FOLLOWING_DISTANCE_MIN_SPEED_MS,
    LATERAL_DETECTION_RANGE_M,
    MIN_LATERAL_CLEARANCE_CYCLIST_M,
    MIN_LATERAL_CLEARANCE_PEDESTRIAN_M,
    MIN_LATERAL_CLEARANCE_VEHICLE_M,
    RULE_NORMALIZATION,
)
from .base import (
    ApplicabilityDetector,
    ApplicabilityResult,
    ViolationEvaluator,
    ViolationResult,
)

_RULE_ID = "L0.R3"
_LEVEL = 0
_NAME = "Safe Lateral Clearance"


class SafeLateralClearanceApplicability(ApplicabilityDetector):
    """
    Detects when safe lateral clearance rule applies.

    Rule applies when:
    - Ego vehicle has nearby road users within detection range
    - Ego is moving (speed > threshold)
    - Other agents are within lateral proximity
    """

    rule_id = _RULE_ID
    level = _LEVEL
    name = _NAME

    def __init__(
        self,
        detection_range_m: float = LATERAL_DETECTION_RANGE_M,
        min_speed_mps: float = FOLLOWING_DISTANCE_MIN_SPEED_MS,
        max_lateral_distance_m: float = 5.0,
    ):
        self.detection_range_m = detection_range_m
        self.min_speed_mps = min_speed_mps
        self.max_lateral_distance_m = max_lateral_distance_m

    def detect(self, ctx: ScenarioContext) -> ApplicabilityResult:
        """Detect if rule is applicable."""
        applicable_frames = 0
        T = len(ctx.ego.x)
        ego = ctx.ego

        # Build spatial index once
        spatial_idx = TemporalSpatialIndex(ctx.agents, T)

        for t in range(T):
            if not ego.is_valid_at(t):
                continue

            ego_speed = ego.speed[t]
            if ego_speed < self.min_speed_mps:
                continue

            ex, ey, eyaw = ego.x[t], ego.y[t], ego.yaw[t]

            # Find nearby agents
            nearby = spatial_idx.at_frame(t).query_radius(
                ex, ey, self.detection_range_m
            )

            for agent, center_dist in nearby:
                if not agent.is_valid_at(t):
                    continue

                ax, ay = agent.x[t], agent.y[t]
                lon, lat = compute_relative_position(ex, ey, eyaw, ax, ay)

                if abs(lat) < self.max_lateral_distance_m:
                    applicable_frames += 1
                    break

        applies = applicable_frames > 0
        confidence = applicable_frames / T if T > 0 else 0.0

        return ApplicabilityResult(
            applies=applies,
            confidence=confidence,
            reasons=[
                f"Nearby agents within lateral range in "
                f"{applicable_frames}/{T} frames"
            ],
            features={
                "applicable_frames": applicable_frames,
                "detection_range_m": self.detection_range_m,
            },
        )


class SafeLateralClearanceViolation(ViolationEvaluator):
    """
    Evaluate safe lateral clearance violations.

    Violation occurs when lateral distance < minimum clearance for agent type.
    Different clearance requirements for different agent types.
    """

    rule_id = _RULE_ID
    level = _LEVEL
    name = _NAME

    def __init__(
        self,
        min_clearance_vehicle_m: float = MIN_LATERAL_CLEARANCE_VEHICLE_M,
        min_clearance_cyclist_m: float = MIN_LATERAL_CLEARANCE_CYCLIST_M,
        min_clearance_ped_m: float = MIN_LATERAL_CLEARANCE_PEDESTRIAN_M,
        detection_range_m: float = LATERAL_DETECTION_RANGE_M,
    ):
        self.min_clearance: Dict[str, float] = {
            "vehicle": min_clearance_vehicle_m,
            "cyclist": min_clearance_cyclist_m,
            "pedestrian": min_clearance_ped_m,
        }
        self.detection_range_m = detection_range_m

    def evaluate(
        self, ctx: ScenarioContext, applicability: ApplicabilityResult
    ) -> ViolationResult:
        """Compute violation severity."""
        ego = ctx.ego
        T = len(ego.x)
        dt = ctx.dt

        spatial_idx = TemporalSpatialIndex(ctx.agents, T)

        clearance_deficits: List[float] = []
        violation_frames: List[int] = []
        total_deficit = 0.0
        worst_deficit = 0.0

        for t in range(T):
            if not ego.is_valid_at(t):
                clearance_deficits.append(0.0)
                continue

            ego_speed = ego.speed[t]
            if ego_speed < FOLLOWING_DISTANCE_MIN_SPEED_MS:
                clearance_deficits.append(0.0)
                continue

            ex, ey, eyaw = ego.x[t], ego.y[t], ego.yaw[t]

            nearby = spatial_idx.at_frame(t).query_radius(
                ex, ey, self.detection_range_m
            )

            max_deficit = 0.0

            for agent, center_dist in nearby:
                if not agent.is_valid_at(t):
                    continue

                ax, ay = agent.x[t], agent.y[t]
                lon, lat = compute_relative_position(ex, ey, eyaw, ax, ay)

                abs_lat = abs(lat)

                # Determine minimum clearance based on agent type
                min_clearance = self.min_clearance.get(
                    agent.type.lower(), MIN_LATERAL_CLEARANCE_VEHICLE_M
                )

                # Account for widths
                ego_half_width = getattr(ego, "width", 1.9) / 2.0
                agent_half_width = getattr(agent, "width", 1.8) / 2.0

                # Actual clearance (edge to edge)
                actual_clearance = abs_lat - ego_half_width - agent_half_width

                if actual_clearance < min_clearance:
                    deficit = min_clearance - actual_clearance
                    max_deficit = max(max_deficit, deficit)

            clearance_deficits.append(max_deficit)
            if max_deficit > 0:
                violation_frames.append(t)
                total_deficit += max_deficit

            worst_deficit = max(worst_deficit, max_deficit)

        # Severity: integral of clearance deficit over time
        severity = total_deficit * dt
        norm_factor = RULE_NORMALIZATION.get(_RULE_ID, 10.0)
        severity_normalized = min(1.0, severity / norm_factor)

        return ViolationResult(
            severity=severity,
            severity_normalized=severity_normalized,
            measurements={
                "n_violation_frames": len(violation_frames),
                "total_deficit_m": total_deficit,
                "worst_deficit_m": worst_deficit,
                "avg_deficit_m": (
                    np.mean([d for d in clearance_deficits if d > 0])
                    if violation_frames
                    else 0.0
                ),
            },
            explanation=(
                (
                    f"Lateral clearance violations in {len(violation_frames)} "
                    f"frames, worst deficit {worst_deficit:.2f}m"
                )
                if violation_frames
                else "Safe lateral clearance maintained"
            ),
            frame_violations=violation_frames,
        )
