"""
Differentiable Rule Proxies for RECTOR.

This module provides differentiable approximations of rule violations
that enable gradient-based training for rule-aware trajectory generation.

Main components:
- DifferentiableProxy: Abstract base class for all proxies
- CollisionProxy: Collision and clearance rules (L0, L10)
- SmoothnessProxy: Kinematic comfort rules (L1)
- LaneProxy: Lane and road structure rules (L3, L7, L8)
- SignalProxy: Traffic signal rules (L5, L8)
- InteractionProxy: Agent interaction rules (L4, L6)
- DifferentiableRuleProxies: Aggregator combining all proxies
"""

from .base import (
    DifferentiableProxy,
    exponential_cost,
    soft_threshold,
    smooth_max,
    smooth_min,
    obb_corners,
    compute_time_to_collision,
)

from .collision_proxy import CollisionProxy, VRUClearanceProxy
from .smoothness_proxy import SmoothnessProxy, LateralAccelerationProxy
from .lane_proxy import LaneProxy, SpeedLimitProxy
from .signal_proxy import SignalProxy
from .interaction_proxy import InteractionProxy
from .aggregator import DifferentiableRuleProxies, ProxyRegistry


__all__ = [
    # Base
    "DifferentiableProxy",
    "exponential_cost",
    "soft_threshold",
    "smooth_max",
    "smooth_min",
    "obb_corners",
    "compute_time_to_collision",
    # Individual proxies
    "CollisionProxy",
    "VRUClearanceProxy",
    "SmoothnessProxy",
    "LateralAccelerationProxy",
    "LaneProxy",
    "SpeedLimitProxy",
    "SignalProxy",
    "InteractionProxy",
    # Aggregator
    "DifferentiableRuleProxies",
    "ProxyRegistry",
]
