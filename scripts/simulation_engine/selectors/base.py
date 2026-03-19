"""
Base selector interface and decision trace dataclass (§6.1, §6.5).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class DecisionTrace:
    """Auditable record of a single selection decision (§6.5)."""

    method: str  # 'confidence', 'weighted_sum', 'rector_lex'
    selected_idx: int  # Index of selected candidate (0..K-1)
    reason: str  # Human-readable explanation
    active_rules: Optional[np.ndarray] = None  # [28] binary applicability mask
    first_diff_tier: Optional[int] = (
        None  # Tier that first eliminated a candidate (0-3)
    )
    per_tier_survivors: Optional[List[np.ndarray]] = field(
        default=None, repr=False
    )  # Surviving mask at each tier
    per_candidate_scores: Optional[np.ndarray] = field(
        default=None, repr=False
    )  # [K, 4] tier scores
    timestamp_ms: float = 0.0  # Wall-clock time (ms)


class BaseSelector(ABC):
    """
    Contract: all selectors receive identical inputs and return the same
    output type.  Generator output is frozen before selection (§6.1).
    """

    @abstractmethod
    def select(
        self,
        candidates: np.ndarray,  # [K, T, 4]  x, y, vx, vy
        probs: np.ndarray,  # [K]        confidence scores
        tier_scores: np.ndarray,  # [K, 4]     per-tier aggregate
        rule_scores: np.ndarray,  # [K, 28]    per-rule scores
        rule_applicability: np.ndarray,  # [28]        binary mask
    ) -> Tuple[int, DecisionTrace]:
        """Return (selected_index, decision_trace)."""
        ...
