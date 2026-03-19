"""Rules modules for Waymo Rule Evaluation."""

from .base import (
    ApplicabilityDetector,
    ApplicabilityResult,
    RuleResult,
    ViolationEvaluator,
    ViolationResult,
)

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
from .l08_crosswalk_yield import CrosswalkYieldApplicability, CrosswalkYieldViolation
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
from .l3_drivable_surface import DrivableSurfaceApplicability, DrivableSurfaceViolation

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
from .registry import (
    RuleEntry,
    all_rules,
    detectors,
    detectors_by_rule,
    evaluators,
    evaluators_by_rule,
    get_rule,
    get_rules_by_level,
    rule_ids,
)

__all__ = [
    # Base classes
    "ApplicabilityResult",
    "ViolationResult",
    "ApplicabilityDetector",
    "ViolationEvaluator",
    "RuleResult",
    # Registry
    "RuleEntry",
    "all_rules",
    "detectors",
    "evaluators",
    "detectors_by_rule",
    "evaluators_by_rule",
    "get_rule",
    "get_rules_by_level",
    "rule_ids",
    # L0 Rules
    "CrosswalkOccupancyApplicability",
    "CrosswalkOccupancyViolation",
    "SafeLateralClearanceApplicability",
    "SafeLateralClearanceViolation",
    "SafeLongitudinalDistanceApplicability",
    "SafeLongitudinalDistanceViolation",
    # L1 Rules
    "SmoothBrakingApplicability",
    "SmoothBrakingViolation",
    "SmoothAccelerationApplicability",
    "SmoothAccelerationViolation",
    "SmoothSteeringApplicability",
    "SmoothSteeringViolation",
    "LaneChangeSmoothnessApplicability",
    "LaneChangeSmoothnessViolation",
    "SpeedConsistencyApplicability",
    "SpeedConsistencyViolation",
    # L3 Rules
    "DrivableSurfaceApplicability",
    "DrivableSurfaceViolation",
    # L4 Rules
    "LeftTurnGapApplicability",
    "LeftTurnGapViolation",
    # L5 Rules
    "TrafficSignalComplianceApplicability",
    "TrafficSignalComplianceViolation",
    "ConstructionZoneComplianceApplicability",
    "ConstructionZoneComplianceEvaluator",
    "ParkingViolationApplicability",
    "ParkingViolationEvaluator",
    "PriorityViolationApplicability",
    "PriorityViolationEvaluator",
    "SchoolZoneComplianceApplicability",
    "SchoolZoneComplianceEvaluator",
    # L6 Rules
    "FollowingDistanceApplicability",
    "FollowingDistanceViolation",
    "PedestrianInteractionApplicability",
    "PedestrianInteractionViolation",
    "CyclistInteractionApplicability",
    "CyclistInteractionViolation",
    "CooperativeLaneChangeApplicability",
    "CooperativeLaneChangeEvaluator",
    "IntersectionNegotiationApplicability",
    "IntersectionNegotiationEvaluator",
    # L7 Rules
    "LaneDepartureApplicability",
    "LaneDepartureViolation",
    "SpeedLimitApplicability",
    "SpeedLimitViolation",
    # L8 Rules
    "RedLightApplicability",
    "RedLightViolation",
    "StopSignApplicability",
    "StopSignViolation",
    "WrongwayApplicability",
    "WrongwayViolation",
    "CrosswalkYieldApplicability",
    "CrosswalkYieldViolation",
    # L10 Rules
    "CollisionApplicability",
    "CollisionViolation",
    "VRUClearanceApplicability",
    "VRUClearanceViolation",
]
