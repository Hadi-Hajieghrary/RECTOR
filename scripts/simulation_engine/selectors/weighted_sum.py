"""Method B: Independently-tuned weighted-sum selector (§6.3)."""

from __future__ import annotations

import time
from typing import Tuple

import numpy as np

from .base import BaseSelector, DecisionTrace


class WeightedSumSelector(BaseSelector):
    """
    Select the candidate minimizing ∑(w_i · rule_score_i · applicability_i).

    Weights are passed at construction time. The closed-loop validation
    (validate_50.py) uses uniform per-rule weights np.ones(28); the
    weight_grid_search.py script can tune tier-level weights separately.
    """

    def __init__(self, weights: np.ndarray):
        # weights: [28] — per-rule weights
        self.weights = np.asarray(weights, dtype=np.float64)

    def select(
        self,
        candidates: np.ndarray,
        probs: np.ndarray,
        tier_scores: np.ndarray,
        rule_scores: np.ndarray,
        rule_applicability: np.ndarray,
    ) -> Tuple[int, DecisionTrace]:
        t0 = time.perf_counter()

        # Mask by applicability, then weighted sum
        masked = rule_scores * rule_applicability[None, :]  # [K, 28]
        cost = (masked * self.weights[None, :]).sum(axis=1)  # [K]
        idx = int(np.argmin(cost))

        dt_ms = (time.perf_counter() - t0) * 1e3
        trace = DecisionTrace(
            method="weighted_sum",
            selected_idx=idx,
            reason=f"Min weighted cost: {cost[idx]:.4f}",
            active_rules=rule_applicability.copy(),
            first_diff_tier=None,
            per_candidate_scores=tier_scores.copy(),
            timestamp_ms=dt_ms,
        )
        return idx, trace
