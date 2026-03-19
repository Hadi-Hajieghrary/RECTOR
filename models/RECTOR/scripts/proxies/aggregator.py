"""
Aggregates all proxy modules into single [B, M, NUM_RULES] output.

This module combines outputs from all individual proxies into a unified
violation cost tensor, properly indexed according to the canonical rule ordering.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Set, List
import sys
import os

# Add parent paths for imports
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "data")
)

from .base import DifferentiableProxy
from .collision_proxy import CollisionProxy, VRUClearanceProxy
from .smoothness_proxy import SmoothnessProxy
from .lane_proxy import LaneProxy, SpeedLimitProxy
from .signal_proxy import SignalProxy
from .interaction_proxy import InteractionProxy


# Import rule constants - handle import path issues gracefully
try:
    from waymo_rule_eval.rules.rule_constants import (
        RULE_IDS,
        NUM_RULES,
        RULE_INDEX_MAP,
        SAFETY_CRITICAL_RULES,
        get_tier_weight_vector,
    )
except ImportError:
    # Fallback definitions if waymo_rule_eval not in path
    RULE_IDS = []
    NUM_RULES = 28
    RULE_INDEX_MAP = {}
    SAFETY_CRITICAL_RULES = set()

    def get_tier_weight_vector():
        return None


class DifferentiableRuleProxies(nn.Module):
    """
    Aggregator for all differentiable rule proxies.

    Combines outputs from individual proxies (collision, smoothness, lane,
    signal, interaction) into a single [B, M, NUM_RULES] cost tensor.

    Each position in the output corresponds to a rule in the canonical
    RULE_IDS ordering.
    """

    def __init__(
        self,
        dt: float = 0.1,
        ego_length: float = 4.5,
        ego_width: float = 2.0,
        softness: float = 5.0,
        require_safety_proxies: bool = True,
    ):
        """
        Initialize aggregated proxies.

        Args:
            dt: Time step in seconds
            ego_length: Ego vehicle length
            ego_width: Ego vehicle width
            softness: Controls gradient sharpness of cost functions
            require_safety_proxies: If True, raise error if safety-critical
                                   rules don't have proxies
        """
        super().__init__()

        self.dt = dt
        self.softness = softness
        self.require_safety_proxies = require_safety_proxies

        # Initialize individual proxies with shared softness
        self.collision_proxy = CollisionProxy(
            ego_length=ego_length,
            ego_width=ego_width,
            softness=softness,
        )

        self.vru_clearance_proxy = VRUClearanceProxy(softness=softness)

        self.smoothness_proxy = SmoothnessProxy(dt=dt, softness=softness)

        self.lane_proxy = LaneProxy(softness=softness)

        self.speed_limit_proxy = SpeedLimitProxy(dt=dt, softness=softness)

        self.signal_proxy = SignalProxy(dt=dt, softness=softness)

        self.interaction_proxy = InteractionProxy(dt=dt, softness=softness)

        # Collect all proxies
        self.proxies: List[DifferentiableProxy] = [
            self.collision_proxy,
            self.vru_clearance_proxy,
            self.smoothness_proxy,
            self.lane_proxy,
            self.speed_limit_proxy,
            self.signal_proxy,
            self.interaction_proxy,
        ]

        # Build rule ID to proxy mapping
        self.rule_to_proxy: Dict[str, DifferentiableProxy] = {}
        self.covered_rules: Set[str] = set()

        for proxy in self.proxies:
            for rule_id in proxy.rule_ids:
                self.rule_to_proxy[rule_id] = proxy
                self.covered_rules.add(rule_id)

        # Validate safety proxy coverage
        if self.require_safety_proxies:
            self._validate_safety_coverage()

        # Log coverage info
        self._log_coverage()

    def _validate_safety_coverage(self):
        """Validate that all safety-critical rules have proxies."""
        missing_safety = SAFETY_CRITICAL_RULES - self.covered_rules
        if missing_safety:
            raise ValueError(
                f"Safety-critical rules missing proxies: {missing_safety}. "
                f"All Tier-0 rules MUST have differentiable proxies."
            )

    def _log_coverage(self):
        """Log proxy coverage information."""
        if NUM_RULES > 0:
            covered_count = len(self.covered_rules & set(RULE_IDS))
            print(f"Proxy coverage: {covered_count}/{NUM_RULES} rules")

            uncovered = set(RULE_IDS) - self.covered_rules
            if uncovered:
                print(f"  Rules without proxies: {sorted(uncovered)}")
        else:
            print(
                f"Proxy coverage: {len(self.covered_rules)} rules (constants not loaded)"
            )

    @property
    def rule_ids(self) -> List[str]:
        """All rule IDs covered by this aggregator."""
        return list(self.covered_rules)

    def forward(
        self,
        trajectories: torch.Tensor,
        scene_features: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute violation costs for all rules.

        Args:
            trajectories: [B, M, H, 2+] candidate trajectories
            scene_features: Dict of scene feature tensors

        Returns:
            [B, M, NUM_RULES] violation costs, where each position
            corresponds to a rule in RULE_IDS ordering.
            Cost is 0 for no violation, approaching 1 for severe violation.
            Rules without proxies get 0 cost (no gradient).
        """
        B, M, H, _ = trajectories.shape
        device = trajectories.device

        # Determine number of rules
        n_rules = NUM_RULES if NUM_RULES > 0 else 28

        # Initialize output tensor
        costs = torch.zeros(B, M, n_rules, device=device)

        # Run each proxy and collect results
        all_costs: Dict[str, torch.Tensor] = {}

        for proxy in self.proxies:
            try:
                proxy_costs = proxy(trajectories, scene_features)
                all_costs.update(proxy_costs)
            except Exception as e:
                # Log error but continue
                print(f"Warning: Proxy {type(proxy).__name__} failed: {e}")

        # Map costs to output tensor using canonical ordering
        for rule_id, cost in all_costs.items():
            if rule_id in RULE_INDEX_MAP:
                idx = RULE_INDEX_MAP[rule_id]
                costs[:, :, idx] = cost
            elif RULE_INDEX_MAP:
                # Rule not in canonical ordering
                print(f"Warning: Rule {rule_id} not in RULE_INDEX_MAP")

        return costs

    def forward_with_names(
        self,
        trajectories: torch.Tensor,
        scene_features: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute violation costs and return as named dict.

        Returns:
            Dict mapping rule_id to [B, M] cost tensor
        """
        all_costs: Dict[str, torch.Tensor] = {}

        for proxy in self.proxies:
            try:
                proxy_costs = proxy(trajectories, scene_features)
                all_costs.update(proxy_costs)
            except Exception as e:
                print(f"Warning: Proxy {type(proxy).__name__} failed: {e}")

        return all_costs

    def get_tier_weighted_costs(
        self,
        trajectories: torch.Tensor,
        scene_features: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute tier-weighted violation costs.

        Returns:
            [B, M, NUM_RULES] costs weighted by tier importance
        """
        costs = self.forward(trajectories, scene_features)

        # Get tier weights
        weights = get_tier_weight_vector()
        if weights is not None:
            weights = torch.tensor(weights, device=costs.device)
            costs = costs * weights.unsqueeze(0).unsqueeze(0)

        return costs


class ProxyRegistry:
    """
    Registry for proxy classes and validation.

    Maps rule IDs to their proxy classes and provides
    validation utilities.
    """

    _registry: Dict[str, type] = {}

    @classmethod
    def register(cls, rule_id: str, proxy_class: type):
        """Register a proxy class for a rule ID."""
        cls._registry[rule_id] = proxy_class

    @classmethod
    def get_proxy(cls, rule_id: str) -> Optional[type]:
        """Get proxy class for a rule ID."""
        return cls._registry.get(rule_id)

    @classmethod
    def get_covered_rules(cls) -> Set[str]:
        """Get set of rule IDs with registered proxies."""
        return set(cls._registry.keys())

    @classmethod
    def validate_safety_coverage(cls):
        """Validate that all safety-critical rules have proxies."""
        covered = cls.get_covered_rules()
        missing = SAFETY_CRITICAL_RULES - covered

        if missing:
            raise ValueError(
                f"Safety-critical rules without proxies: {missing}. "
                f"Covered rules: {covered}"
            )

        print(f"✓ Safety proxy coverage validated: {len(SAFETY_CRITICAL_RULES)} rules")

    @classmethod
    def build_combined_proxy(cls, **kwargs) -> DifferentiableRuleProxies:
        """Build combined proxy aggregator."""
        return DifferentiableRuleProxies(**kwargs)


# Auto-register proxies
def _auto_register_proxies():
    """Register all proxy classes with the registry."""
    proxies = [
        CollisionProxy(),
        VRUClearanceProxy(),
        SmoothnessProxy(),
        LaneProxy(),
        SpeedLimitProxy(),
        SignalProxy(),
        InteractionProxy(),
    ]

    for proxy in proxies:
        for rule_id in proxy.rule_ids:
            ProxyRegistry.register(rule_id, type(proxy))


# Run auto-registration
try:
    _auto_register_proxies()
except Exception as e:
    print(f"Warning: Proxy auto-registration failed: {e}")
