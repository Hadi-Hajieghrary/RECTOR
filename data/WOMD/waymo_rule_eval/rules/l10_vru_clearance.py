"""
L10.R2: VRU Clearance Rule

Checks minimum clearance to Vulnerable Road Users (pedestrians, cyclists).

Features:
- Filters for VRUs specifically
- Accounts for ego speed (higher speed = more clearance needed)
- Uses spatial indexing for efficiency
"""

from dataclasses import dataclass
from typing import List, Tuple

from ..core.context import ScenarioContext
from ..core.geometry import compute_bumper_distance
from ..core.temporal_spatial import TemporalSpatialIndex
from ..utils.constants import (
    DEFAULT_EGO_LENGTH,
    DEFAULT_SPATIAL_RADIUS_M,
    VRU_CLEARANCE_SPEED_THRESHOLD_MS,
    VRU_CYCLIST_CLEARANCE_M,
    VRU_PEDESTRIAN_CLEARANCE_M,
)
from .base import (
    ApplicabilityDetector,
    ApplicabilityResult,
    ViolationEvaluator,
    ViolationResult,
)

_RULE_ID = "L10.R2"
_LEVEL = 10
_NAME = "VRU Clearance"


@dataclass
class ClearanceViolationEvent:
    """Details of a clearance violation."""

    frame_idx: int
    agent_id: int
    agent_type: str
    clearance_m: float
    required_clearance_m: float
    ego_speed_ms: float


class VRUClearanceApplicability(ApplicabilityDetector):
    """VRU clearance applies when there are pedestrians or cyclists."""

    rule_id = _RULE_ID
    level = _LEVEL
    name = _NAME

    def detect(self, ctx: ScenarioContext) -> ApplicabilityResult:
        """Check if there are any VRUs in the scenario."""
        vrus = ctx.vrus
        n_vrus = len(vrus)

        if n_vrus == 0:
            return ApplicabilityResult(
                applies=False,
                confidence=1.0,
                reasons=["No VRUs (pedestrians/cyclists) in scenario"],
                features={"n_vrus": 0},
            )

        n_peds = len(ctx.pedestrians)
        n_cyclists = len(ctx.cyclists)

        return ApplicabilityResult(
            applies=True,
            confidence=1.0,
            reasons=[f"VRU clearance for {n_peds} peds, {n_cyclists} cyclists"],
            features={
                "n_vrus": n_vrus,
                "n_pedestrians": n_peds,
                "n_cyclists": n_cyclists,
            },
        )


class VRUClearanceViolation(ViolationEvaluator):
    """Evaluate clearance to VRUs."""

    rule_id = _RULE_ID
    level = _LEVEL
    name = _NAME

    def __init__(
        self,
        ped_clearance_m: float = VRU_PEDESTRIAN_CLEARANCE_M,
        cyclist_clearance_m: float = VRU_CYCLIST_CLEARANCE_M,
        speed_threshold_ms: float = VRU_CLEARANCE_SPEED_THRESHOLD_MS,
    ):
        self.ped_clearance_m = ped_clearance_m
        self.cyclist_clearance_m = cyclist_clearance_m
        self.speed_threshold_ms = speed_threshold_ms

    def evaluate(
        self, ctx: ScenarioContext, applicability: ApplicabilityResult
    ) -> ViolationResult:
        """Evaluate VRU clearance violations."""

        ego = ctx.ego
        vrus = ctx.vrus
        T = len(ego.x)

        ego_length = getattr(ego, "length", DEFAULT_EGO_LENGTH)

        # Build spatial index for VRUs only
        spatial_idx = TemporalSpatialIndex(vrus, T)

        violation_frames: List[int] = []
        events: List[ClearanceViolationEvent] = []
        total_violation_m = 0.0
        worst_violation_m = 0.0

        for t in range(T):
            if not ego.is_valid_at(t):
                continue

            ex, ey = ego.x[t], ego.y[t]
            eyaw = ego.yaw[t]
            espeed = ego.speed[t]

            # Only check clearance at higher speeds
            if espeed < self.speed_threshold_ms:
                continue

            # Query nearby VRUs
            nearby = spatial_idx.at_frame(t).query_radius(
                ex, ey, DEFAULT_SPATIAL_RADIUS_M
            )

            for vru, center_dist in nearby:
                if not vru.is_valid_at(t):
                    continue

                # Compute bumper-to-bumper distance
                vru_length = getattr(vru, "length", 0.5)
                result = compute_bumper_distance(
                    ex, ey, eyaw, ego_length, vru.x[t], vru.y[t], vru_length
                )

                clearance = result.bumper_distance

                # Determine required clearance based on VRU type
                if vru.is_pedestrian:
                    required = self.ped_clearance_m
                else:
                    required = self.cyclist_clearance_m

                # Check violation
                if clearance < required:
                    violation_amount = required - clearance

                    if t not in violation_frames:
                        violation_frames.append(t)

                    total_violation_m += violation_amount
                    worst_violation_m = max(worst_violation_m, violation_amount)

                    events.append(
                        ClearanceViolationEvent(
                            frame_idx=t,
                            agent_id=vru.id,
                            agent_type=vru.type,
                            clearance_m=clearance,
                            required_clearance_m=required,
                            ego_speed_ms=espeed,
                        )
                    )

        # Compute severity
        severity = total_violation_m
        severity_normalized = min(1.0, worst_violation_m / 2.0)

        return ViolationResult(
            severity=severity,
            severity_normalized=severity_normalized,
            measurements={
                "n_violation_frames": len(violation_frames),
                "n_violation_events": len(events),
                "total_violation_m": total_violation_m,
                "worst_violation_m": worst_violation_m,
            },
            explanation=(
                (f"{len(violation_frames)} frames with VRU clearance violations")
                if len(violation_frames) > 0
                else "No VRU clearance violations"
            ),
            frame_violations=violation_frames,
        )
