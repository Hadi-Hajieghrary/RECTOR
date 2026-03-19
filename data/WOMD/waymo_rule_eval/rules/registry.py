# waymo_rule_eval/rules/registry.py
# -*- coding: utf-8 -*-
"""Centralized rule registry for all detectors and evaluators."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class RuleEntry:
    """Registry entry for a rule with its detector and evaluator."""

    rule_id: str
    detector: Any
    evaluator: Any


def _rules() -> List[RuleEntry]:
    """Build and return the full list of registered rules."""
    # L0: Critical Safety Rules
    from .l0_crosswalk_occupancy import (
        CrosswalkOccupancyApplicability,
        CrosswalkOccupancyViolation,
    )
    from .l0_safe_lateral_clearance import (
        SafeLateralClearanceApplicability,
        SafeLateralClearanceViolation,
    )
    from .l0_safe_longitudinal_distance import (
        SafeLongitudinalDistanceApplicability,
        SafeLongitudinalDistanceViolation,
    )

    # L7: Lane/Speed Rules
    from .l07_lane_departure import LaneDepartureApplicability, LaneDepartureViolation
    from .l07_speed_limit import SpeedLimitApplicability, SpeedLimitViolation
    from .l08_crosswalk_yield import (
        CrosswalkYieldApplicability,
        CrosswalkYieldViolation,
    )
    from .l08_stop_sign import StopSignApplicability, StopSignViolation
    from .l08_wrongway import WrongwayApplicability, WrongwayViolation
    from .l1_lane_change_smoothness import (
        LaneChangeSmoothnessApplicability,
        LaneChangeSmoothnessViolation,
    )
    from .l1_smooth_acceleration import (
        SmoothAccelerationApplicability,
        SmoothAccelerationViolation,
    )

    # L1: Comfort Rules
    from .l1_smooth_braking import SmoothBrakingApplicability, SmoothBrakingViolation
    from .l1_smooth_steering import SmoothSteeringApplicability, SmoothSteeringViolation
    from .l1_speed_consistency import (
        SpeedConsistencyApplicability,
        SpeedConsistencyViolation,
    )

    # L3: Surface Rules
    from .l3_drivable_surface import (
        DrivableSurfaceApplicability,
        DrivableSurfaceViolation,
    )

    # L4: Maneuver Rules
    from .l4_left_turn_gap import LeftTurnGapApplicability, LeftTurnGapViolation
    from .l5_construction_zone_compliance import (
        ConstructionZoneComplianceApplicability,
        ConstructionZoneComplianceEvaluator,
    )
    from .l5_parking_violation import (
        ParkingViolationApplicability,
        ParkingViolationEvaluator,
    )
    from .l5_priority_violation import (
        PriorityViolationApplicability,
        PriorityViolationEvaluator,
    )
    from .l5_school_zone_compliance import (
        SchoolZoneComplianceApplicability,
        SchoolZoneComplianceEvaluator,
    )

    # L5: Regulatory Rules
    from .l5_traffic_signal_compliance import (
        TrafficSignalComplianceApplicability,
        TrafficSignalComplianceViolation,
    )
    from .l6_cooperative_lane_change import (
        CooperativeLaneChangeApplicability,
        CooperativeLaneChangeEvaluator,
    )
    from .l6_cyclist_interaction import (
        CyclistInteractionApplicability,
        CyclistInteractionViolation,
    )

    # L6: Interaction Rules
    from .l6_following_distance import (
        FollowingDistanceApplicability,
        FollowingDistanceViolation,
    )
    from .l6_intersection_negotiation import (
        IntersectionNegotiationApplicability,
        IntersectionNegotiationEvaluator,
    )
    from .l6_pedestrian_interaction import (
        PedestrianInteractionApplicability,
        PedestrianInteractionViolation,
    )

    # L8: Traffic Control Rules
    from .l8_red_light import RedLightApplicability, RedLightViolation

    # L10: Collision Rules
    from .l10_collision import CollisionApplicability, CollisionViolation
    from .l10_vru_clearance import VRUClearanceApplicability, VRUClearanceViolation

    rules = [
        # L0: Critical Safety
        RuleEntry(
            "L0.R2",
            SafeLongitudinalDistanceApplicability(),
            SafeLongitudinalDistanceViolation(),
        ),
        RuleEntry(
            "L0.R3",
            SafeLateralClearanceApplicability(),
            SafeLateralClearanceViolation(),
        ),
        RuleEntry(
            "L0.R4", CrosswalkOccupancyApplicability(), CrosswalkOccupancyViolation()
        ),
        # L1: Comfort
        RuleEntry(
            "L1.R1", SmoothAccelerationApplicability(), SmoothAccelerationViolation()
        ),
        RuleEntry("L1.R2", SmoothBrakingApplicability(), SmoothBrakingViolation()),
        RuleEntry("L1.R3", SmoothSteeringApplicability(), SmoothSteeringViolation()),
        RuleEntry(
            "L1.R4", SpeedConsistencyApplicability(), SpeedConsistencyViolation()
        ),
        RuleEntry(
            "L1.R5",
            LaneChangeSmoothnessApplicability(),
            LaneChangeSmoothnessViolation(),
        ),
        # L3: Surface
        RuleEntry("L3.R3", DrivableSurfaceApplicability(), DrivableSurfaceViolation()),
        # L4: Maneuver
        RuleEntry("L4.R3", LeftTurnGapApplicability(), LeftTurnGapViolation()),
        # L6: Interaction
        RuleEntry(
            "L6.R2", FollowingDistanceApplicability(), FollowingDistanceViolation()
        ),
        RuleEntry(
            "L6.R4",
            PedestrianInteractionApplicability(),
            PedestrianInteractionViolation(),
        ),
        RuleEntry(
            "L6.R5", CyclistInteractionApplicability(), CyclistInteractionViolation()
        ),
        RuleEntry(
            "L6.R1",
            CooperativeLaneChangeApplicability(),
            CooperativeLaneChangeEvaluator(),
        ),
        RuleEntry(
            "L6.R3",
            IntersectionNegotiationApplicability(),
            IntersectionNegotiationEvaluator(),
        ),
        # L5: Regulatory
        RuleEntry(
            "L5.R1",
            TrafficSignalComplianceApplicability(),
            TrafficSignalComplianceViolation(),
        ),
        RuleEntry(
            "L5.R2", PriorityViolationApplicability(), PriorityViolationEvaluator()
        ),
        RuleEntry(
            "L5.R3", ParkingViolationApplicability(), ParkingViolationEvaluator()
        ),
        RuleEntry(
            "L5.R4",
            SchoolZoneComplianceApplicability(),
            SchoolZoneComplianceEvaluator(),
        ),
        RuleEntry(
            "L5.R5",
            ConstructionZoneComplianceApplicability(),
            ConstructionZoneComplianceEvaluator(),
        ),
        # L7: Lane/Speed
        RuleEntry("L7.R3", LaneDepartureApplicability(), LaneDepartureViolation()),
        RuleEntry("L7.R4", SpeedLimitApplicability(), SpeedLimitViolation()),
        # L8: Traffic Control
        RuleEntry("L8.R1", RedLightApplicability(), RedLightViolation()),
        RuleEntry("L8.R2", StopSignApplicability(), StopSignViolation()),
        RuleEntry("L8.R3", CrosswalkYieldApplicability(), CrosswalkYieldViolation()),
        RuleEntry("L8.R5", WrongwayApplicability(), WrongwayViolation()),
        # L10: Collision
        RuleEntry("L10.R1", CollisionApplicability(), CollisionViolation()),
        RuleEntry("L10.R2", VRUClearanceApplicability(), VRUClearanceViolation()),
    ]

    return rules


def all_rules() -> List[RuleEntry]:
    """Get all registered rules."""
    return _rules()


def detectors() -> List[Any]:
    """Get all applicability detectors."""
    return [r.detector for r in _rules()]


def evaluators() -> List[Any]:
    """Get all violation evaluators."""
    return [r.evaluator for r in _rules()]


def detectors_by_rule() -> Dict[str, Any]:
    """Get detectors indexed by rule ID."""
    return {r.rule_id: r.detector for r in _rules()}


def evaluators_by_rule() -> Dict[str, Any]:
    """Get evaluators indexed by rule ID."""
    return {r.rule_id: r.evaluator for r in _rules()}


def get_rule(rule_id: str) -> RuleEntry:
    """Get a specific rule by ID."""
    for r in _rules():
        if r.rule_id == rule_id:
            return r
    raise KeyError(f"Unknown rule: {rule_id}")


def get_rules_by_level(level: int) -> List[RuleEntry]:
    """Get all rules for a specific level (e.g., 0, 1, 3, 4, 6, 7, 8, 10)."""
    prefix = f"L{level}."
    return [r for r in _rules() if r.rule_id.startswith(prefix)]


def rule_ids() -> List[str]:
    """Get list of all rule IDs."""
    return [r.rule_id for r in _rules()]
