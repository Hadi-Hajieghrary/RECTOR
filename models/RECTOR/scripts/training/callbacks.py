"""
Training Callbacks for RECTOR.

Provides:
- ProxyRefinementCallback: Validates proxy vs exact rule correlation
- ApplicabilityHysteresis: Stable applicability predictions at inference
- EarlyStopping: Stop training when validation loss plateaus
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
import logging

import sys

sys.path.insert(0, "/workspace/data/WOMD")
sys.path.insert(0, "/workspace/data")
from waymo_rule_eval.rules.rule_constants import RULE_IDS, NUM_RULES


class ProxyRefinementCallback:
    """
    Callback to validate proxy costs against exact rule evaluation.

    Every N batches:
    1. Run exact rules on a sample batch
    2. Compare to proxy costs
    3. Log correlation per rule
    4. Warn if correlation < threshold

    This helps identify when proxies drift from exact rules.
    """

    def __init__(
        self,
        exact_rule_evaluator: Optional[Callable] = None,
        check_every: int = 1000,
        correlation_threshold: float = 0.5,
        log_level: str = "WARNING",
    ):
        """
        Initialize callback.

        Args:
            exact_rule_evaluator: Function (trajectories, context) -> violations
            check_every: Check every N batches
            correlation_threshold: Warn if correlation below this
            log_level: Logging level for warnings
        """
        self.exact_evaluator = exact_rule_evaluator
        self.check_every = check_every
        self.correlation_threshold = correlation_threshold

        self.logger = logging.getLogger("ProxyRefinement")
        self.logger.setLevel(getattr(logging, log_level))

        # Track correlations over time
        self.correlation_history: Dict[str, List[float]] = {
            rule_id: [] for rule_id in RULE_IDS
        }

        self.batch_count = 0

    def on_batch_end(
        self,
        proxy_violations: torch.Tensor,
        trajectories: torch.Tensor,
        scene_context: Any,
    ) -> Dict[str, float]:
        """
        Called at end of each batch.

        Args:
            proxy_violations: [B, M, R] proxy-computed violations
            trajectories: [B, M, T, D] generated trajectories
            scene_context: Context for exact evaluation

        Returns:
            Dict with correlation metrics (if checked)
        """
        self.batch_count += 1

        if self.batch_count % self.check_every != 0:
            return {}

        if self.exact_evaluator is None:
            self.logger.debug("No exact evaluator configured, skipping check")
            return {}

        return self._check_correlations(proxy_violations, trajectories, scene_context)

    def _check_correlations(
        self,
        proxy_violations: torch.Tensor,
        trajectories: torch.Tensor,
        scene_context: Any,
    ) -> Dict[str, float]:
        """Check proxy vs exact correlations."""
        with torch.no_grad():
            # Get exact violations
            try:
                exact_violations = self.exact_evaluator(trajectories, scene_context)
            except Exception as e:
                self.logger.warning(f"Exact evaluation failed: {e}")
                return {}

            # Flatten for correlation
            proxy_flat = proxy_violations.cpu().numpy().reshape(-1, NUM_RULES)
            exact_flat = exact_violations.cpu().numpy().reshape(-1, NUM_RULES)

            correlations = {}
            warnings = []

            for i, rule_id in enumerate(RULE_IDS):
                proxy_col = proxy_flat[:, i]
                exact_col = exact_flat[:, i]

                # Skip if no variation
                if proxy_col.std() < 1e-8 or exact_col.std() < 1e-8:
                    continue

                corr = np.corrcoef(proxy_col, exact_col)[0, 1]

                if np.isnan(corr):
                    continue

                correlations[f"{rule_id}_correlation"] = corr
                self.correlation_history[rule_id].append(corr)

                if corr < self.correlation_threshold:
                    warnings.append(f"{rule_id}: correlation={corr:.3f}")

            # Log warnings
            if warnings:
                self.logger.warning(
                    f"Low proxy-exact correlations at batch {self.batch_count}:\n"
                    + "\n".join(warnings)
                )

            return correlations

    def get_summary(self) -> Dict[str, float]:
        """Get summary of correlation history."""
        summary = {}
        for rule_id, history in self.correlation_history.items():
            if history:
                summary[f"{rule_id}_mean_corr"] = np.mean(history)
                summary[f"{rule_id}_min_corr"] = np.min(history)
        return summary


class ApplicabilityHysteresis(nn.Module):
    """
    Hysteresis for stable applicability predictions at inference.

    Prevents flickering by using different thresholds for turning
    a rule "on" vs "off", and requiring a rule to be active for
    a minimum number of frames.

    on_threshold > off_threshold creates hysteresis band.
    """

    def __init__(
        self,
        on_threshold: float = 0.6,
        off_threshold: float = 0.4,
        min_on_frames: int = 3,
        min_off_frames: int = 2,
    ):
        """
        Initialize hysteresis.

        Args:
            on_threshold: Probability threshold to turn rule ON
            off_threshold: Probability threshold to turn rule OFF
            min_on_frames: Minimum frames before rule can turn off
            min_off_frames: Minimum frames before rule can turn on
        """
        super().__init__()

        assert (
            on_threshold > off_threshold
        ), "on_threshold must be > off_threshold for hysteresis"

        self.on_threshold = on_threshold
        self.off_threshold = off_threshold
        self.min_on_frames = min_on_frames
        self.min_off_frames = min_off_frames

        # State tracking per batch element
        self._state: Optional[torch.Tensor] = None  # [B, R] current binary state
        self._on_count: Optional[torch.Tensor] = None  # [B, R] frames since turned on
        self._off_count: Optional[torch.Tensor] = None  # [B, R] frames since turned off

    def reset(self, batch_size: Optional[int] = None):
        """Reset state for new sequence."""
        self._state = None
        self._on_count = None
        self._off_count = None

    def forward(
        self,
        prob: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply hysteresis to applicability probabilities.

        Args:
            prob: [B, R] applicability probabilities

        Returns:
            [B, R] stable binary applicability
        """
        B, R = prob.shape
        device = prob.device

        # Initialize state if needed
        if self._state is None:
            # Use simple threshold for initial state
            self._state = (prob > 0.5).float()
            self._on_count = torch.zeros(B, R, device=device)
            self._off_count = torch.zeros(B, R, device=device)
            return self._state

        # Check batch size consistency
        if self._state.shape[0] != B:
            self.reset()
            return (prob > 0.5).float()

        # Update counters
        self._on_count = self._on_count + self._state
        self._off_count = self._off_count + (1 - self._state)

        # Compute transitions
        # Turn ON: prob > on_threshold AND been off for min_off_frames
        can_turn_on = (self._state == 0) & (self._off_count >= self.min_off_frames)
        should_turn_on = prob > self.on_threshold
        turn_on = can_turn_on & should_turn_on

        # Turn OFF: prob < off_threshold AND been on for min_on_frames
        can_turn_off = (self._state == 1) & (self._on_count >= self.min_on_frames)
        should_turn_off = prob < self.off_threshold
        turn_off = can_turn_off & should_turn_off

        # Update state
        new_state = self._state.clone()
        new_state[turn_on] = 1.0
        new_state[turn_off] = 0.0

        # Reset counters on transitions
        self._on_count[turn_on] = 0
        self._off_count[turn_off] = 0

        self._state = new_state

        return new_state

    def get_stable_applicability(
        self,
        prob: torch.Tensor,
        return_raw: bool = False,
    ) -> torch.Tensor:
        """
        Convenience method combining forward with optional raw output.

        Args:
            prob: [B, R] applicability probabilities
            return_raw: Also return raw probabilities

        Returns:
            Stable binary applicability (and optionally raw prob)
        """
        stable = self.forward(prob)
        if return_raw:
            return stable, prob
        return stable


class EarlyStopping:
    """
    Early stopping callback.

    Stops training when validation loss doesn't improve for patience epochs.
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        mode: str = "min",
    ):
        """
        Initialize early stopping.

        Args:
            patience: Epochs to wait before stopping
            min_delta: Minimum improvement to reset patience
            mode: 'min' or 'max' for metric
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode

        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.counter = 0
        self.should_stop = False

    def __call__(self, value: float) -> bool:
        """
        Check if should stop.

        Args:
            value: Current metric value

        Returns:
            True if should stop training
        """
        if self.mode == "min":
            improved = value < self.best_value - self.min_delta
        else:
            improved = value > self.best_value + self.min_delta

        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop

    def reset(self):
        """Reset state."""
        self.best_value = float("inf") if self.mode == "min" else float("-inf")
        self.counter = 0
        self.should_stop = False


@dataclass
class TrainingMetrics:
    """Container for training metrics."""

    epoch: int
    step: int
    train_loss: float
    val_loss: Optional[float] = None
    learning_rate: float = 0.0
    proxy_correlations: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        d = {
            "epoch": self.epoch,
            "step": self.step,
            "train_loss": self.train_loss,
            "learning_rate": self.learning_rate,
        }
        if self.val_loss is not None:
            d["val_loss"] = self.val_loss
        if self.proxy_correlations:
            d.update(self.proxy_correlations)
        return d
