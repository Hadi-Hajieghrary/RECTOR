"""Method A: Confidence-only selector (§6.2)."""

from __future__ import annotations

import time
from typing import Tuple

import numpy as np

from .base import BaseSelector, DecisionTrace


class ConfidenceSelector(BaseSelector):
    """Select the candidate with the highest generator confidence score."""

    def select(
        self,
        candidates: np.ndarray,
        probs: np.ndarray,
        tier_scores: np.ndarray,
        rule_scores: np.ndarray,
        rule_applicability: np.ndarray,
    ) -> Tuple[int, DecisionTrace]:
        t0 = time.perf_counter()
        idx = int(np.argmax(probs))
        dt_ms = (time.perf_counter() - t0) * 1e3

        trace = DecisionTrace(
            method="confidence",
            selected_idx=idx,
            reason=f"Highest confidence: {probs[idx]:.4f}",
            active_rules=None,
            first_diff_tier=None,
            timestamp_ms=dt_ms,
        )
        return idx, trace
