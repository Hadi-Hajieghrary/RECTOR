"""
Generate rule labels for training samples.

CRITICAL: Uses canonical RULE_IDS ordering and trajectory-conditioned evaluation.

This module provides functions to:
1. Evaluate rule applicability (scene-only)
2. Evaluate violations given a specific trajectory
3. Generate training labels for augmented samples
"""

import numpy as np
from typing import Tuple, Dict, List, Optional, Any
import sys
import os

# Add paths for imports
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "data")
)

try:
    from waymo_rule_eval.core.context import ScenarioContext
    from waymo_rule_eval.core.context_extension import (
        ScenarioContextExtension,
        inject_ego_trajectory,
    )
    from waymo_rule_eval.rules.rule_constants import (
        RULE_IDS,
        NUM_RULES,
        RULE_INDEX_MAP,
    )
    from waymo_rule_eval.rules.adapter import (
        evaluate_rule,
        evaluate_applicability_only,
        get_rule_id,
    )
    from waymo_rule_eval.rules.registry import all_rules

    WAYMO_AVAILABLE = True
except ImportError as e:
    print(f"Warning: waymo_rule_eval not available: {e}")
    WAYMO_AVAILABLE = False
    RULE_IDS = []
    NUM_RULES = 28
    RULE_INDEX_MAP = {}


def generate_labels(
    ctx: ScenarioContext,
    ego_trajectory: np.ndarray,
    other_futures: Optional[np.ndarray] = None,
    future_start_idx: int = 11,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate rule labels for a given trajectory.

    CRITICAL: This function evaluates rules in a trajectory-conditioned manner.
    Applicability depends only on the scene, but violations depend on the
    specific trajectory being evaluated.

    Args:
        ctx: ScenarioContext with scene information
        ego_trajectory: [H, 2] or [H, 4] candidate trajectory to evaluate
        other_futures: [N_agents, H, 2] predicted other agent trajectories (optional)
        future_start_idx: Timestep where trajectory prediction begins

    Returns:
        Tuple of:
        - applicability: [NUM_RULES] bool array
        - violations: [NUM_RULES] bool array
        - severity: [NUM_RULES] float array in [0, 1]
    """
    if not WAYMO_AVAILABLE:
        # Return dummy labels
        return (
            np.zeros(NUM_RULES, dtype=bool),
            np.zeros(NUM_RULES, dtype=bool),
            np.zeros(NUM_RULES, dtype=float),
        )

    n_rules = len(RULE_IDS) if RULE_IDS else NUM_RULES
    applicability = np.zeros(n_rules, dtype=bool)
    violations = np.zeros(n_rules, dtype=bool)
    severity = np.zeros(n_rules, dtype=float)

    # Create trajectory-injected context
    ext = ScenarioContextExtension(ctx)
    ctx_with_traj = ext.with_ego_trajectory(
        trajectory=ego_trajectory,
        source="candidate",
        future_start_idx=future_start_idx,
    )

    # Optionally inject other agent futures
    if other_futures is not None:
        ctx_with_traj = ScenarioContextExtension(ctx_with_traj).with_other_futures(
            futures=other_futures,
            future_start_idx=future_start_idx,
        )

    # Evaluate each rule
    rules = all_rules()

    for rule in rules:
        rule_id = get_rule_id(rule)

        if rule_id not in RULE_INDEX_MAP:
            continue

        idx = RULE_INDEX_MAP[rule_id]

        try:
            # Evaluate rule (applicability uses scene-only, violation uses trajectory)
            app_result, vio_result = evaluate_rule(
                rule=rule,
                ctx_scene=ctx,  # Scene-only for applicability
                ctx_with_traj=ctx_with_traj,  # With trajectory for violation
            )

            applicability[idx] = app_result.applies

            if app_result.applies and vio_result is not None:
                violations[idx] = vio_result.severity > 0
                severity[idx] = vio_result.severity_normalized

        except Exception as e:
            print(f"Warning: Error evaluating rule {rule_id}: {e}")

    return applicability, violations, severity


def generate_applicability_only(ctx: ScenarioContext) -> np.ndarray:
    """
    Generate only applicability labels (scene-only, no trajectory needed).

    This is used when we only need to know which rules apply to a scene,
    without evaluating any specific trajectory.

    Args:
        ctx: ScenarioContext with scene information

    Returns:
        applicability: [NUM_RULES] bool array
    """
    if not WAYMO_AVAILABLE:
        return np.zeros(NUM_RULES, dtype=bool)

    n_rules = len(RULE_IDS) if RULE_IDS else NUM_RULES
    applicability = np.zeros(n_rules, dtype=bool)

    rules = all_rules()

    for rule in rules:
        rule_id = get_rule_id(rule)

        if rule_id not in RULE_INDEX_MAP:
            continue

        idx = RULE_INDEX_MAP[rule_id]

        try:
            app_result = evaluate_applicability_only(rule, ctx)
            applicability[idx] = app_result.applies
        except Exception as e:
            print(f"Warning: Error evaluating applicability for {rule_id}: {e}")

    return applicability


def generate_contrastive_labels(
    ctx: ScenarioContext,
    gt_trajectory: np.ndarray,
    perturbed_trajectory: np.ndarray,
    other_futures: Optional[np.ndarray] = None,
    future_start_idx: int = 11,
) -> Dict[str, np.ndarray]:
    """
    Generate labels for both ground truth and perturbed trajectories.

    Used for contrastive learning where we want the model to prefer
    the GT trajectory over the perturbed one.

    Args:
        ctx: ScenarioContext
        gt_trajectory: Ground truth trajectory
        perturbed_trajectory: Perturbed trajectory (with violations)
        other_futures: Other agent future predictions
        future_start_idx: Prediction start index

    Returns:
        Dict with keys:
        - applicability: [NUM_RULES]
        - gt_violations: [NUM_RULES]
        - gt_severity: [NUM_RULES]
        - perturbed_violations: [NUM_RULES]
        - perturbed_severity: [NUM_RULES]
    """
    # Applicability is the same for both (scene-only)
    applicability = generate_applicability_only(ctx)

    # Evaluate GT trajectory
    _, gt_violations, gt_severity = generate_labels(
        ctx, gt_trajectory, other_futures, future_start_idx
    )

    # Evaluate perturbed trajectory
    _, pert_violations, pert_severity = generate_labels(
        ctx, perturbed_trajectory, other_futures, future_start_idx
    )

    return {
        "applicability": applicability,
        "gt_violations": gt_violations,
        "gt_severity": gt_severity,
        "perturbed_violations": pert_violations,
        "perturbed_severity": pert_severity,
    }


class OtherFuturesAugmenter:
    """
    Adds prediction-like noise to ground truth other-agent trajectories.

    This bridges the train/test gap: during training we have GT futures,
    but at test time we only have predictions with noise.

    The noise model:
    1. Position noise grows linearly with horizon (prediction uncertainty)
    2. Noise is temporally correlated (random walk, not IID)
    3. Noise magnitude can be calibrated from real predictor statistics
    """

    def __init__(
        self,
        position_noise_per_second: float = 0.3,  # meters/second of horizon
        dt: float = 0.1,
        correlation: float = 0.95,  # Temporal correlation coefficient
        seed: Optional[int] = None,
    ):
        """
        Initialize augmenter.

        Args:
            position_noise_per_second: Position std grows by this per second
            dt: Time step in seconds
            correlation: Temporal correlation (0 = IID, 1 = full correlation)
            seed: Random seed for reproducibility
        """
        self.position_noise_per_second = position_noise_per_second
        self.dt = dt
        self.correlation = correlation
        self.rng = np.random.default_rng(seed)

    def augment(
        self,
        gt_futures: np.ndarray,
        validity: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Add prediction-like noise to GT futures.

        Args:
            gt_futures: [N_agents, H, 2] ground truth positions
            validity: [N_agents, H] validity mask (optional)

        Returns:
            [N_agents, H, 2] noisy futures
        """
        N, H, D = gt_futures.shape

        # Compute noise std that grows with horizon
        time = np.arange(H) * self.dt
        noise_std = self.position_noise_per_second * time  # [H]
        noise_std = noise_std[np.newaxis, :, np.newaxis]  # [1, H, 1]

        # Generate correlated noise (random walk)
        innovation = self.rng.standard_normal((N, H, D))

        noise = np.zeros((N, H, D))
        noise[:, 0, :] = innovation[:, 0, :]

        for t in range(1, H):
            noise[:, t, :] = (
                self.correlation * noise[:, t - 1, :]
                + np.sqrt(1 - self.correlation**2) * innovation[:, t, :]
            )

        # Scale by growing std
        noise = noise * noise_std

        # Apply noise
        noisy_futures = gt_futures + noise

        # Apply validity mask
        if validity is not None:
            noisy_futures[~validity] = np.nan

        return noisy_futures

    @classmethod
    def from_predictor_stats(
        cls,
        mean_ade: float,
        final_de: float,
        horizon_seconds: float = 8.0,
    ) -> "OtherFuturesAugmenter":
        """
        Create augmenter calibrated from predictor statistics.

        Args:
            mean_ade: Average displacement error across horizon
            final_de: Final displacement error at end of horizon
            horizon_seconds: Prediction horizon in seconds

        Returns:
            Calibrated augmenter
        """
        # Estimate position_noise_per_second from final_de
        # Assuming linear growth, final_de ≈ noise_per_second * horizon
        noise_per_second = final_de / horizon_seconds

        return cls(position_noise_per_second=noise_per_second)


class PredictorErrorStats:
    """
    Container for predictor error statistics.

    Used to calibrate OtherFuturesAugmenter.
    """

    def __init__(
        self,
        mean_ade: float = 1.0,
        std_ade: float = 0.5,
        final_de_mean: float = 2.5,
        final_de_std: float = 1.0,
        miss_rate: float = 0.1,
        horizon_seconds: float = 8.0,
    ):
        self.mean_ade = mean_ade
        self.std_ade = std_ade
        self.final_de_mean = final_de_mean
        self.final_de_std = final_de_std
        self.miss_rate = miss_rate
        self.horizon_seconds = horizon_seconds

    def get_augmenter(self) -> OtherFuturesAugmenter:
        """Create calibrated augmenter."""
        return OtherFuturesAugmenter.from_predictor_stats(
            mean_ade=self.mean_ade,
            final_de=self.final_de_mean,
            horizon_seconds=self.horizon_seconds,
        )
