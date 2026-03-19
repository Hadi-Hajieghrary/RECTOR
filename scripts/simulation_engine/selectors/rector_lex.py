"""Method C: RECTOR lexicographic selector (§6.4)."""

from __future__ import annotations

import time
from typing import List, Tuple

import numpy as np

from .base import BaseSelector, DecisionTrace

# Tier names for readable decision traces
TIER_NAMES = ["Safety", "Legal", "Road", "Comfort"]


class RECTORLexSelector(BaseSelector):
    """
    Lexicographic tier-ordered selection.

    For each tier (Safety → Legal → Road → Comfort), candidates whose
    tier score exceeds the best surviving score + epsilon are eliminated.
    Among final survivors, the highest-confidence candidate wins.

    This implements the structural guarantee: Safety is *never* traded
    for Comfort, regardless of relative magnitudes.
    """

    def __init__(self, epsilon: np.ndarray | List[float]):
        # epsilon: [4] — per-tier tolerances
        self.epsilon = np.asarray(epsilon, dtype=np.float64)
        assert self.epsilon.shape == (
            4,
        ), f"Expected 4 epsilons, got {self.epsilon.shape}"

    def select(
        self,
        candidates: np.ndarray,
        probs: np.ndarray,
        tier_scores: np.ndarray,
        rule_scores: np.ndarray,
        rule_applicability: np.ndarray,
    ) -> Tuple[int, DecisionTrace]:
        t0 = time.perf_counter()

        K = candidates.shape[0]
        surviving = np.ones(K, dtype=bool)
        per_tier_survivors: List[np.ndarray] = []
        first_diff_tier = None

        for tier in range(4):  # Safety → Legal → Road → Comfort
            tier_vals = tier_scores[:, tier]
            min_val = tier_vals[surviving].min()

            within_tol = tier_vals <= min_val + self.epsilon[tier]
            new_surviving = surviving & within_tol

            per_tier_survivors.append(new_surviving.copy())

            if new_surviving.sum() < surviving.sum() and first_diff_tier is None:
                first_diff_tier = tier

            surviving = new_surviving
            if surviving.sum() == 1:
                break

        # Among survivors, pick highest confidence
        survivor_probs = np.where(surviving, probs, -np.inf)
        idx = int(np.argmax(survivor_probs))

        dt_ms = (time.perf_counter() - t0) * 1e3
        tier_name = (
            TIER_NAMES[first_diff_tier] if first_diff_tier is not None else "None"
        )
        trace = DecisionTrace(
            method="rector_lex",
            selected_idx=idx,
            reason=f"Lex selection, first diff tier: {tier_name} ({first_diff_tier})",
            active_rules=rule_applicability.copy(),
            first_diff_tier=first_diff_tier,
            per_tier_survivors=per_tier_survivors,
            per_candidate_scores=tier_scores.copy(),
            timestamp_ms=dt_ms,
        )
        return idx, trace
