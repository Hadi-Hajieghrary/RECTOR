"""
Tiered Rule Scorer.

Implements lexicographic (tiered) scoring for trajectory selection.
Safety > Legal > Road > Comfort

Key insight: A trajectory with ANY safety violation is worse than
a trajectory with ALL other violations but no safety violation.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

import os
import sys

_WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.insert(0, os.path.join(_WORKSPACE, "data/WOMD"))
sys.path.insert(0, os.path.join(_WORKSPACE, "data"))
from waymo_rule_eval.rules.rule_constants import (
    NUM_RULES,
    get_tier_mask,
    TIERS,
    TIER_WEIGHTS,
)


class TieredRuleScorer(nn.Module):
    """
    Score trajectories using tiered rule violations.

    Tier ordering (most to least important):
    1. Safety (L0, L10): Collision avoidance
    2. Legal (L5, L7, L8): Traffic laws
    3. Road (L3, L7 road rules): Lane keeping
    4. Comfort (L1, L4, L6): Kinematic comfort

    Within each tier, violations are aggregated with optional weighting.
    Final score is lexicographic: compare tier 1 first, then tier 2, etc.
    """

    def __init__(
        self,
        use_learned_weights: bool = False,
        tier_weights: Optional[Dict[str, float]] = None,
        epsilon: float = 0.01,
    ):
        """
        Initialize scorer.

        Args:
            use_learned_weights: Learn per-rule weights within tiers
            tier_weights: Override tier importance weights
            epsilon: Threshold for considering violations equal
        """
        super().__init__()

        self.epsilon = epsilon

        # Default tier weights (for soft aggregation mode)
        default_weights = {
            "safety": 1000.0,
            "legal": 100.0,
            "road": 10.0,
            "comfort": 1.0,
        }
        self.tier_weights = tier_weights or default_weights

        # Register tier masks as buffers
        for tier_name in TIERS:
            mask = get_tier_mask(tier_name)
            self.register_buffer(
                f"{tier_name}_mask", torch.tensor(mask, dtype=torch.float32)
            )

        # Optional learned per-rule weights
        if use_learned_weights:
            self.rule_weights = nn.Parameter(torch.ones(NUM_RULES))
        else:
            self.register_buffer("rule_weights", torch.ones(NUM_RULES))

    def get_tier_violations(
        self,
        violations: torch.Tensor,
        applicability: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Get aggregated violation per tier.

        Args:
            violations: [B, M, R] violation values per rule
            applicability: [B, R] which rules are applicable (optional)

        Returns:
            Dict mapping tier name to [B, M] violation scores
        """
        B = violations.shape[0]
        tier_violations = {}

        for tier_name in TIERS:
            mask = getattr(self, f"{tier_name}_mask").to(violations.device)
            # [R]

            # Combine tier mask with per-rule applicability
            if applicability is not None:
                # [B, R] = [B, R] * [R] — only rules in this tier AND applicable
                effective_mask = applicability * mask.unsqueeze(0)
            else:
                effective_mask = mask.unsqueeze(0).expand(B, -1)

            # Per-rule weighted violations, masked by applicability
            # [B, R] = rule_weights [R] * effective_mask [B, R]
            weights = self.rule_weights * effective_mask
            num_applicable = effective_mask.sum(dim=-1, keepdim=True).clamp(
                min=1
            )  # [B, 1]

            # [B, M] = sum over R of ([B, M, R] * [B, 1, R]) / [B, 1]
            tier_viol = (violations * weights.unsqueeze(1)).sum(dim=-1) / num_applicable

            tier_violations[tier_name] = tier_viol

        return tier_violations

    def lexicographic_compare(
        self,
        tier_violations: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Lexicographic comparison for trajectory ranking.

        Returns indices of best trajectory per batch.

        Args:
            tier_violations: Dict[tier_name -> [B, M]]

        Returns:
            best_indices: [B] index of best trajectory per batch
        """
        B, M = tier_violations["safety"].shape
        device = tier_violations["safety"].device

        # Start with all trajectories as candidates
        candidates = torch.ones(B, M, dtype=torch.bool, device=device)

        for tier_name in TIERS:
            viol = tier_violations[tier_name]  # [B, M]

            # For each batch, find minimum among candidates
            masked_viol = viol.clone()
            masked_viol[~candidates] = float("inf")

            min_viol = masked_viol.min(dim=1, keepdim=True)[0]  # [B, 1]

            # Keep only trajectories within epsilon of minimum
            within_epsilon = (viol <= min_viol + self.epsilon) & candidates

            # Update candidates
            candidates = within_epsilon

        # If no candidates survived filtering (edge case), restore all so
        # argmax doesn't silently return 0 for an all-False row.
        empty_mask = candidates.sum(dim=1) == 0
        candidates[empty_mask] = True

        # Pick first remaining candidate (arbitrary tiebreaker)
        indices = candidates.float().argmax(dim=1)

        return indices

    def score_trajectories(
        self,
        violations: torch.Tensor,
        applicability: Optional[torch.Tensor] = None,
        mode: str = "lexicographic",
    ) -> torch.Tensor:
        """
        Score trajectories for ranking.

        Args:
            violations: [B, M, R] violation values
            applicability: [B, R] rule applicability
            mode: 'lexicographic' or 'weighted'

        Returns:
            scores: [B, M] lower is better
        """
        tier_violations = self.get_tier_violations(violations, applicability)

        if mode == "weighted":
            # Soft weighted aggregation
            score = torch.zeros_like(tier_violations["safety"])
            for tier_name, weight in self.tier_weights.items():
                score = score + weight * tier_violations[tier_name]
            return score

        elif mode == "lexicographic":
            # Create composite score that preserves lexicographic ordering
            # Multiply each tier by decreasing powers of large number
            score = torch.zeros_like(tier_violations["safety"])

            multiplier = 1e12
            for tier_name in TIERS:
                score = score + multiplier * tier_violations[tier_name]
                multiplier = multiplier / 1000

            return score

        else:
            raise ValueError(f"Unknown mode: {mode}")

    def select_best(
        self,
        violations: torch.Tensor,
        applicability: Optional[torch.Tensor] = None,
        mode: str = "lexicographic",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select best trajectory per batch.

        Args:
            violations: [B, M, R] violation values
            applicability: [B, R] rule applicability
            mode: 'lexicographic' or 'weighted'

        Returns:
            Tuple of:
            - best_indices: [B] index of best trajectory
            - best_scores: [B] score of best trajectory
        """
        scores = self.score_trajectories(violations, applicability, mode)
        best_scores, best_indices = scores.min(dim=1)
        return best_indices, best_scores

    def forward(
        self,
        violations: torch.Tensor,
        applicability: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass.

        Returns:
            Dict with:
            - scores: [B, M] trajectory scores
            - tier_violations: Dict[tier -> [B, M]]
            - best_indices: [B]
            - best_scores: [B]
        """
        tier_violations = self.get_tier_violations(violations, applicability)
        scores = self.score_trajectories(violations, applicability, "lexicographic")
        best_indices, best_scores = self.select_best(violations, applicability)

        return {
            "scores": scores,
            "tier_violations": tier_violations,
            "best_indices": best_indices,
            "best_scores": best_scores,
        }


class SoftLexicographicLoss(nn.Module):
    """
    Differentiable approximation to lexicographic selection.

    Uses softmin over tiers to create gradient signal that
    respects tier priorities.
    """

    def __init__(
        self,
        tier_temperatures: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize loss.

        Args:
            tier_temperatures: Temperature per tier (lower = harder selection)
        """
        super().__init__()

        # Default temperatures (higher tiers have lower temp = stricter)
        default_temps = {
            "safety": 0.01,
            "legal": 0.1,
            "road": 1.0,
            "comfort": 10.0,
        }
        self.temperatures = tier_temperatures or default_temps

        # Register tier masks
        for tier_name in TIERS:
            mask = get_tier_mask(tier_name)
            self.register_buffer(
                f"{tier_name}_mask", torch.tensor(mask, dtype=torch.float32)
            )

    def forward(
        self,
        violations: torch.Tensor,
        applicability: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute tiered compliance loss.

        Lower violations in higher-priority tiers get stronger gradients.

        Args:
            violations: [B, M, R] or [B, R]
            applicability: [B, R]

        Returns:
            Dict with per-tier losses and total loss
        """
        if violations.dim() == 2:
            violations = violations.unsqueeze(1)  # [B, 1, R]

        losses = {}
        cumulative_weight = 1.0

        for tier_name in TIERS:
            mask = getattr(self, f"{tier_name}_mask").to(violations.device)
            temp = self.temperatures[tier_name]

            # Get violations for this tier
            tier_viol = violations * mask  # [B, M, R]

            # Apply applicability
            if applicability is not None:
                tier_viol = tier_viol * applicability.unsqueeze(1)

            # Aggregate within tier (mean over applicable rules)
            if applicability is not None:
                num_applicable = (
                    (applicability * mask).sum(dim=-1, keepdim=True).clamp(min=1)
                )
            else:
                num_applicable = (mask > 0).sum().clamp(min=1)
            tier_loss = tier_viol.sum(dim=-1) / num_applicable
            # [B, M]

            # Apply softmin across modes (encourage best mode)
            weights = torch.softmax(-tier_loss / temp, dim=1)
            tier_loss = (weights * tier_loss).sum(dim=1)  # [B]

            losses[f"{tier_name}_loss"] = tier_loss.mean()

            # Cumulative weighting (higher tiers get exponentially more weight)
            cumulative_weight *= 10

        # Weighted sum
        total = torch.zeros(1, device=violations.device)
        weight = 1.0
        for tier_name in reversed(TIERS):  # Comfort first, Safety last (highest weight)
            total = total + weight * losses[f"{tier_name}_loss"]
            weight *= 10

        losses["total"] = total

        return losses


class DifferentiableTieredSelection(nn.Module):
    """
    Differentiable trajectory selection using tiered soft attention.

    For training: uses soft attention to select trajectories
    For inference: uses hard argmin selection
    """

    def __init__(
        self,
        temperature: float = 0.1,
        tier_scale: float = 10.0,
    ):
        super().__init__()
        self.temperature = temperature
        self.tier_scale = tier_scale

        for tier_name in TIERS:
            mask = get_tier_mask(tier_name)
            self.register_buffer(
                f"{tier_name}_mask", torch.tensor(mask, dtype=torch.float32)
            )

    def forward(
        self,
        trajectories: torch.Tensor,
        violations: torch.Tensor,
        applicability: Optional[torch.Tensor] = None,
        hard: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select best trajectory with differentiable attention.

        Args:
            trajectories: [B, M, T, D]
            violations: [B, M, R]
            applicability: [B, R]
            hard: Use hard selection (non-differentiable)

        Returns:
            selected_trajectory: [B, T, D]
            selection_weights: [B, M]
        """
        B, M, T, D = trajectories.shape
        device = trajectories.device

        # Compute scores (lexicographic style) with per-rule applicability
        score = torch.zeros(B, M, device=device)
        scale = self.tier_scale ** (len(TIERS) - 1)

        for tier_name in TIERS:
            mask = getattr(self, f"{tier_name}_mask").to(device)
            if applicability is not None:
                effective = applicability * mask.unsqueeze(0)  # [B, R]
                num_app = effective.sum(dim=-1, keepdim=True).clamp(min=1)  # [B, 1]
                tier_viol = (violations * effective.unsqueeze(1)).sum(dim=-1) / num_app
            else:
                num_rules = mask.sum().clamp(min=1)
                tier_viol = (violations * mask).sum(dim=-1) / num_rules  # [B, M]
            score = score + scale * tier_viol
            scale = scale / self.tier_scale

        # Softmax selection
        weights = torch.softmax(-score / self.temperature, dim=1)  # [B, M]

        if hard:
            # Hard selection (inference)
            idx = score.argmin(dim=1)  # [B]
            selected = trajectories[torch.arange(B, device=device), idx]
            weights = torch.zeros(B, M, device=device)
            weights[torch.arange(B, device=device), idx] = 1.0
        else:
            # Soft selection (training)
            selected = torch.einsum("bm,bmtd->btd", weights, trajectories)

        return selected, weights
