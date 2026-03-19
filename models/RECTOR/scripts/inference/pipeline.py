"""
RECTOR Inference Pipeline.

Provides inference with optional exact rule evaluation.

Inference flow:
1. Generate M trajectory candidates via CVAE
2. Fast shortlisting via proxy violations
3. (Optional) Exact rule evaluation on top-K
4. Tiered selection for final trajectory

Features:
- Batch inference
- Temperature-controlled sampling
- Shortlisting for efficiency
- Integration with exact Waymo rules
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import numpy as np

import sys

sys.path.insert(0, "/workspace/models/RECTOR/scripts")
sys.path.insert(0, "/workspace/data/WOMD")
sys.path.insert(0, "/workspace/data")

from models import RuleAwareGenerator
from waymo_rule_eval.rules.rule_constants import (
    NUM_RULES,
    RULE_IDS,
    get_tier_mask,
    TIERS,
    RULE_INDEX_MAP,
)


@dataclass
class InferenceConfig:
    """Inference configuration."""

    # Sampling
    num_samples: int = 64
    temperature: float = 1.0

    # Shortlisting
    use_shortlisting: bool = True
    shortlist_k: int = 10

    # Exact evaluation
    use_exact_rules: bool = False

    # Selection
    selection_mode: str = "lexicographic"  # or 'weighted'

    # Device
    device: str = "cuda"

    # Batching
    batch_size: int = 32


class RECTORInference:
    """
    Inference engine for RECTOR.

    Handles trajectory generation, rule evaluation, and selection.
    """

    def __init__(
        self,
        model: RuleAwareGenerator,
        config: Optional[InferenceConfig] = None,
    ):
        """
        Initialize inference engine.

        Args:
            model: Trained RuleAwareGenerator
            config: Inference configuration
        """
        self.model = model
        self.config = config or InferenceConfig()

        self.device = torch.device(self.config.device)
        self.model = self.model.to(self.device)
        self.model.eval()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        config: Optional[InferenceConfig] = None,
    ) -> "RECTORInference":
        """Load from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Reconstruct model
        model_config = checkpoint.get("config", {})
        model = RuleAwareGenerator(
            embed_dim=model_config.get("embed_dim", 256),
            num_heads=model_config.get("num_heads", 8),
            num_encoder_layers=model_config.get("num_encoder_layers", 3),
            trajectory_length=model_config.get("trajectory_length", 80),
            num_modes=model_config.get("num_modes", 6),
            latent_dim=model_config.get("latent_dim", 32),
        )

        model.load_state_dict(checkpoint["model_state_dict"])

        return cls(model, config)

    @torch.no_grad()
    def encode_scene(
        self,
        ego_history: torch.Tensor,
        agent_states: torch.Tensor,
        lane_centers: torch.Tensor,
        agent_mask: Optional[torch.Tensor] = None,
        lane_mask: Optional[torch.Tensor] = None,
        agent_types: Optional[torch.Tensor] = None,
    ):
        """Encode scene features. Returns (scene_embed, element_features, element_pad_mask)."""
        encoder_out = self.model.encode_scene(
            ego_history.to(self.device),
            agent_states.to(self.device),
            lane_centers.to(self.device),
            agent_mask.to(self.device) if agent_mask is not None else None,
            lane_mask.to(self.device) if lane_mask is not None else None,
        )
        if isinstance(encoder_out, tuple):
            return encoder_out  # (scene_embed, element_features, element_pad_mask)
        return encoder_out, None, None

    @torch.no_grad()
    def predict_applicability(
        self,
        scene_embedding: torch.Tensor,
        element_features: Optional[torch.Tensor] = None,
        element_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict rule applicability.

        Returns:
            binary_applicability: [B, R] 0/1
            applicability_prob: [B, R] probabilities
        """
        _, prob = self.model.predict_applicability(
            scene_embedding, element_features, element_mask
        )
        binary = (prob > 0.5).float()
        return binary, prob

    @torch.no_grad()
    def generate_candidates(
        self,
        scene_embedding: torch.Tensor,
        num_samples: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate trajectory candidates.

        Args:
            scene_embedding: [B, D]
            num_samples: Number of samples (default: config value)
            temperature: Sampling temperature

        Returns:
            trajectories: [B, M, T, D]
            confidences: [B, M]
        """
        num_samples = num_samples or self.config.num_samples
        temperature = temperature or self.config.temperature

        trajectories, confidences = self.model.trajectory_head.sample(
            scene_embedding, num_samples, temperature
        )

        return trajectories, confidences

    @torch.no_grad()
    def evaluate_proxy_violations(
        self,
        trajectories: torch.Tensor,
        scene_features: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Evaluate violations using differentiable proxies.

        Args:
            trajectories: [B, M, T, D]
            scene_features: Dict with lane_centers, etc.

        Returns:
            violations: [B, M, R]
        """
        return self.model.evaluate_violations(trajectories, scene_features)

    def shortlist_candidates(
        self,
        trajectories: torch.Tensor,
        violations: torch.Tensor,
        applicability: Optional[torch.Tensor] = None,
        k: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Shortlist top-K candidates based on proxy violations.

        Args:
            trajectories: [B, M, T, D]
            violations: [B, M, R]
            applicability: [B, R]
            k: Number to keep

        Returns:
            shortlisted_trajectories: [B, K, T, D]
            shortlisted_violations: [B, K, R]
            shortlist_indices: [B, K]
        """
        k = k or self.config.shortlist_k
        B, M, T, D = trajectories.shape

        # Compute proxy scores
        scores = self.model.tiered_scorer.score_trajectories(
            violations, applicability, mode="lexicographic"
        )  # [B, M]

        # Get top-K (lowest scores)
        _, indices = torch.topk(scores, k, dim=1, largest=False)  # [B, K]

        # Gather shortlisted
        indices_traj = indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, T, D)
        shortlisted_traj = torch.gather(trajectories, 1, indices_traj)

        indices_viol = indices.unsqueeze(-1).expand(-1, -1, violations.shape[-1])
        shortlisted_viol = torch.gather(violations, 1, indices_viol)

        return shortlisted_traj, shortlisted_viol, indices

    def evaluate_exact_violations(
        self,
        trajectories: torch.Tensor,
        scene_context: Any,
    ) -> torch.Tensor:
        """
        Evaluate violations using exact Waymo rules.

        Args:
            trajectories: [B, M, T, D] - GPU tensor of candidate trajectories
            scene_context: ScenarioContext or list of ScenarioContext for evaluation

        Returns:
            violations: [B, M, R] - binary violations from exact rules
        """
        try:
            from waymo_rule_eval.core.context_extension import ScenarioContextExtension
            from waymo_rule_eval.rules.adapter import evaluate_rule, get_rule_id
            from waymo_rule_eval.rules.registry import all_rules
        except ImportError:
            raise NotImplementedError(
                "Exact rule evaluation requires waymo_rule_eval package"
            )

        B, M, T, D = trajectories.shape
        violations = torch.zeros(B, M, NUM_RULES, device=trajectories.device)

        # Move trajectories to CPU for rule evaluation
        trajectories_np = trajectories.cpu().numpy()

        # Handle single context or list of contexts
        if not isinstance(scene_context, list):
            scene_contexts = [scene_context] * B
        else:
            scene_contexts = scene_context

        # Get all rules once
        rules = all_rules()

        for b in range(B):
            ctx = scene_contexts[b]

            for m in range(M):
                # Extract trajectory for this candidate
                traj = trajectories_np[b, m]  # [T, D]

                # Only use x, y positions (first 2 dims)
                if D >= 2:
                    traj_xy = traj[:, :2]
                else:
                    traj_xy = traj

                # Create trajectory-injected context
                ext = ScenarioContextExtension(ctx)
                ctx_with_traj = ext.with_ego_trajectory(
                    trajectory=traj_xy,
                    source="candidate",
                )

                # Evaluate each rule
                for rule in rules:
                    rule_id = get_rule_id(rule)

                    if rule_id not in RULE_INDEX_MAP:
                        continue

                    idx = RULE_INDEX_MAP[rule_id]

                    try:
                        app_result, vio_result = evaluate_rule(
                            rule=rule,
                            ctx_scene=ctx,
                            ctx_with_traj=ctx_with_traj,
                        )

                        if app_result.applies and vio_result is not None:
                            if vio_result.severity > 0:
                                violations[b, m, idx] = 1.0
                    except Exception:
                        # Rule evaluation failed, leave as 0
                        pass

        return violations

    def select_best(
        self,
        trajectories: torch.Tensor,
        violations: torch.Tensor,
        applicability: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select best trajectory using tiered scoring.

        Args:
            trajectories: [B, M, T, D]
            violations: [B, M, R]
            applicability: [B, R]

        Returns:
            best_trajectory: [B, T, D]
            best_index: [B]
        """
        best_idx, _ = self.model.tiered_scorer.select_best(
            violations, applicability, mode=self.config.selection_mode
        )

        B = trajectories.shape[0]
        best_traj = trajectories[torch.arange(B, device=trajectories.device), best_idx]

        return best_traj, best_idx

    @torch.no_grad()
    def infer(
        self,
        ego_history: torch.Tensor,
        agent_states: torch.Tensor,
        lane_centers: torch.Tensor,
        agent_mask: Optional[torch.Tensor] = None,
        lane_mask: Optional[torch.Tensor] = None,
        agent_types: Optional[torch.Tensor] = None,
        scene_features: Optional[Dict[str, torch.Tensor]] = None,
        scene_context: Optional[Any] = None,
        return_all: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Full inference pipeline.

        Args:
            ego_history: [B, T_hist, 4]
            agent_states: [B, N, T, 4]
            lane_centers: [B, L, P, 2]
            agent_mask: [B, N]
            lane_mask: [B, L]
            agent_types: [B, N]
            scene_features: Dict for proxy evaluation
            scene_context: For exact evaluation
            return_all: Return all candidates and scores

        Returns:
            Dict with:
            - trajectory: [B, T, D] best trajectory
            - applicability: [B, R] predicted applicability
            - violations: [B, R] best trajectory violations
            - confidence: [B] best trajectory confidence
            - (if return_all) all_trajectories, all_violations, etc.
        """
        # Move to device
        ego_history = ego_history.to(self.device)
        agent_states = agent_states.to(self.device)
        lane_centers = lane_centers.to(self.device)

        if agent_mask is not None:
            agent_mask = agent_mask.to(self.device)
        if lane_mask is not None:
            lane_mask = lane_mask.to(self.device)
        if agent_types is not None:
            agent_types = agent_types.to(self.device)

        if scene_features is None:
            scene_features = {"lane_centers": lane_centers, "lane_mask": lane_mask}
        else:
            scene_features = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in scene_features.items()
            }

        B = ego_history.shape[0]

        # 1. Encode scene
        scene_embed, element_features, element_pad_mask = self.encode_scene(
            ego_history, agent_states, lane_centers, agent_mask, lane_mask, agent_types
        )

        # 2. Predict applicability
        binary_app, app_prob = self.predict_applicability(
            scene_embed, element_features, element_pad_mask
        )

        # 3. Generate candidates
        trajectories, confidences = self.generate_candidates(scene_embed)

        # 4. Evaluate proxy violations
        violations = self.evaluate_proxy_violations(trajectories, scene_features)

        # 5. Shortlisting (if enabled)
        if self.config.use_shortlisting:
            trajectories, violations, shortlist_idx = self.shortlist_candidates(
                trajectories, violations, app_prob
            )
            # Reindex confidences
            confidences = torch.gather(confidences, 1, shortlist_idx)

        # 6. Exact evaluation (if enabled)
        if self.config.use_exact_rules and scene_context is not None:
            violations = self.evaluate_exact_violations(trajectories, scene_context)

        # 7. Select best
        best_traj, best_idx = self.select_best(trajectories, violations, app_prob)

        # Get best trajectory's violations and confidence
        best_viol = violations[torch.arange(B, device=self.device), best_idx]
        best_conf = confidences[torch.arange(B, device=self.device), best_idx]

        result = {
            "trajectory": best_traj,
            "applicability": app_prob,
            "violations": best_viol,
            "confidence": best_conf,
            "best_index": best_idx,
        }

        if return_all:
            result["all_trajectories"] = trajectories
            result["all_violations"] = violations
            result["all_confidences"] = confidences

        return result

    def infer_batch(
        self,
        batch: Dict[str, torch.Tensor],
        return_all: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Inference on a batch dict.

        Args:
            batch: Dict with ego_history, agent_states, lane_centers, etc.
            return_all: Return all candidates

        Returns:
            Inference results dict
        """
        return self.infer(
            ego_history=batch["ego_history"],
            agent_states=batch["agent_states"],
            lane_centers=batch["lane_centers"],
            agent_mask=batch.get("agent_mask"),
            lane_mask=batch.get("lane_mask"),
            agent_types=batch.get("agent_types"),
            scene_features=batch.get("scene_features"),
            return_all=return_all,
        )


class StreamingInference:
    """
    Streaming inference for real-time applications.

    Maintains scene encoding cache for efficiency.
    """

    def __init__(
        self,
        model: RuleAwareGenerator,
        config: Optional[InferenceConfig] = None,
    ):
        self.engine = RECTORInference(model, config)
        self._scene_cache = {}

    def update_scene(
        self,
        scene_id: str,
        ego_history: torch.Tensor,
        agent_states: torch.Tensor,
        lane_centers: torch.Tensor,
        **kwargs,
    ):
        """Update scene encoding cache."""
        scene_embed, element_features, element_pad_mask = self.engine.encode_scene(
            ego_history.unsqueeze(0),
            agent_states.unsqueeze(0),
            lane_centers.unsqueeze(0),
            **{
                k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v
                for k, v in kwargs.items()
            },
        )
        self._scene_cache[scene_id] = {
            "embedding": scene_embed,
            "element_features": element_features,
            "element_pad_mask": element_pad_mask,
            "lane_centers": lane_centers,
        }

    def infer_from_cache(
        self,
        scene_id: str,
    ) -> Dict[str, torch.Tensor]:
        """Infer using cached scene encoding."""
        if scene_id not in self._scene_cache:
            raise ValueError(f"Scene {scene_id} not in cache")

        cache = self._scene_cache[scene_id]
        scene_embed = cache["embedding"]

        # Generate candidates
        trajectories, confidences = self.engine.generate_candidates(scene_embed)

        # Evaluate
        scene_features = {"lane_centers": cache["lane_centers"].unsqueeze(0)}
        violations = self.engine.evaluate_proxy_violations(trajectories, scene_features)

        # Select
        binary_app, app_prob = self.engine.predict_applicability(
            scene_embed, cache.get("element_features"), cache.get("element_pad_mask")
        )
        best_traj, best_idx = self.engine.select_best(
            trajectories, violations, app_prob
        )

        return {
            "trajectory": best_traj.squeeze(0),
            "applicability": app_prob.squeeze(0),
        }

    def clear_cache(self):
        """Clear scene cache."""
        self._scene_cache.clear()


def compute_metrics(
    predictions: Dict[str, torch.Tensor],
    ground_truth: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    """
    Compute inference metrics.

    Args:
        predictions: Dict with trajectory, applicability, violations
        ground_truth: Dict with traj_gt, applicability_gt, violations_gt

    Returns:
        Dict with metrics
    """
    metrics = {}

    # Trajectory metrics
    if "trajectory" in predictions and "traj_gt" in ground_truth:
        pred = predictions["trajectory"]
        gt = ground_truth["traj_gt"]

        # ADE (average displacement error)
        displacement = torch.sqrt(((pred[:, :, :2] - gt[:, :, :2]) ** 2).sum(dim=-1))
        ade = displacement.mean().item()
        metrics["ade"] = ade

        # FDE (final displacement error)
        fde = displacement[:, -1].mean().item()
        metrics["fde"] = fde

    # Applicability metrics
    if "applicability" in predictions and "applicability_gt" in ground_truth:
        pred = (predictions["applicability"] > 0.5).float()
        gt = ground_truth["applicability_gt"]

        # Accuracy
        correct = (pred == gt).float()
        accuracy = correct.mean().item()
        metrics["applicability_accuracy"] = accuracy

        # Per-tier accuracy
        for tier_name in TIERS:
            mask = torch.tensor(get_tier_mask(tier_name), device=pred.device)
            tier_correct = (correct * mask).sum() / (mask.sum() + 1e-8)
            metrics[f"{tier_name}_accuracy"] = tier_correct.item()

    # Violation metrics
    if "violations" in predictions and "violations_gt" in ground_truth:
        pred = predictions["violations"]
        gt = ground_truth["violations_gt"]

        # Mean violation difference
        diff = torch.abs(pred - gt)
        metrics["violation_mae"] = diff.mean().item()

        # Compliance rate (% of rules with violation < threshold)
        threshold = 0.1
        compliant = (pred < threshold).float()
        metrics["compliance_rate"] = compliant.mean().item()

    return metrics
