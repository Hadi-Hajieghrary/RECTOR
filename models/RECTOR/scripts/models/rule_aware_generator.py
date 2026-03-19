"""
Rule-Aware Trajectory Generator.

Features:
1. Transformer-based CVAE decoder (no error accumulation)
2. Goal-conditioned prediction
3. 5-second horizon (50 steps)
4. Unfrozen M2I encoder (fine-tuned with low LR)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from .scene_encoder import SceneEncoder
from .applicability_head import RuleApplicabilityHead
from .cvae_head import CVAETrajectoryHead, TRAJECTORY_LENGTH
from .tiered_scorer import TieredRuleScorer, DifferentiableTieredSelection

# Optional M2I encoder
try:
    from .m2i_encoder import M2ISceneEncoder

    M2I_AVAILABLE = True
except ImportError:
    M2I_AVAILABLE = False

import sys

sys.path.insert(0, "/workspace/data/WOMD")
sys.path.insert(0, "/workspace/data")
sys.path.insert(0, "/workspace/models/RECTOR/scripts")
from waymo_rule_eval.rules.rule_constants import NUM_RULES
from proxies.aggregator import DifferentiableRuleProxies


class RuleAwareGenerator(nn.Module):
    """
    Improved Rule-Aware Generator with Transformer decoder.

    Key differences from V1:
    1. Transformer decoder - parallel generation, no error accumulation
    2. Goal-conditioned prediction - anchors trajectories
    3. 5-second horizon - reduces error compounding
    4. Larger model capacity
    """

    def __init__(
        self,
        # Scene encoder params
        embed_dim: int = 256,
        num_heads: int = 8,
        num_encoder_layers: int = 4,
        history_length: int = 11,
        max_agents: int = 32,
        max_lanes: int = 64,
        # Trajectory params
        trajectory_length: int = TRAJECTORY_LENGTH,  # 50 steps = 5 sec
        num_modes: int = 6,
        latent_dim: int = 64,
        decoder_hidden_dim: int = 256,
        decoder_num_layers: int = 4,
        # Rule params
        num_rules: int = NUM_RULES,
        # Regularization
        dropout: float = 0.1,
        # M2I encoder option
        use_m2i_encoder: bool = True,
        m2i_checkpoint: str = None,
        freeze_m2i: bool = False,  # DON'T freeze - fine-tune with low LR
    ):
        super().__init__()

        self.num_modes = num_modes
        self.num_rules = num_rules
        self.trajectory_length = trajectory_length
        self.use_m2i_encoder = use_m2i_encoder

        # Scene encoder
        if use_m2i_encoder and M2I_AVAILABLE:
            print("Using M2I encoder (will fine-tune with low LR)")
            self.scene_encoder = M2ISceneEncoder(
                m2i_checkpoint=m2i_checkpoint,
                embed_dim=embed_dim,
                max_agents=max_agents,
                max_lanes=max_lanes,
                freeze_m2i=freeze_m2i,  # False - allow fine-tuning
            )
        else:
            self.scene_encoder = SceneEncoder(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_layers=num_encoder_layers,
                history_length=history_length,
                max_agents=max_agents,
                max_lanes=max_lanes,
                dropout=dropout,
            )

        # Rule applicability head
        applicability_hidden = int(embed_dim * 1.5)
        self.applicability_head = RuleApplicabilityHead(
            embed_dim=embed_dim,
            hidden_dim=applicability_hidden,
            num_layers=3,
            num_rules=num_rules,
            num_heads=num_heads,
            dropout=dropout,
        )

        # V2 CVAE with Transformer decoder
        self.trajectory_head = CVAETrajectoryHead(
            scene_dim=embed_dim,
            latent_dim=latent_dim,
            hidden_dim=decoder_hidden_dim,
            trajectory_length=trajectory_length,
            output_dim=4,
            num_modes=num_modes,
            num_layers=decoder_num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Differentiable rule proxies
        self.rule_proxies = DifferentiableRuleProxies(softness=5.0)

        # Tiered scorer
        self.tiered_scorer = TieredRuleScorer(use_learned_weights=False)

        # Differentiable selection
        self.differentiable_selector = DifferentiableTieredSelection(temperature=0.1)

    def encode_scene(
        self,
        ego_history: torch.Tensor,
        agent_states: torch.Tensor,
        lane_centers: torch.Tensor,
        agent_mask: Optional[torch.Tensor] = None,
        lane_mask: Optional[torch.Tensor] = None,
    ):
        """Encode scene to embedding."""
        return self.scene_encoder(
            ego_history, agent_states, lane_centers, agent_mask, lane_mask
        )

    def predict_applicability(
        self,
        scene_embedding: torch.Tensor,
        scene_elements: Optional[torch.Tensor] = None,
        element_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict rule applicability."""
        logits, probs = self.applicability_head(
            scene_embedding, scene_elements, element_mask
        )
        return logits, probs

    def generate_trajectories(
        self,
        scene_embedding: torch.Tensor,
        traj_gt: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Generate trajectory modes."""
        return self.trajectory_head(scene_embedding, traj_gt)

    def forward(
        self,
        ego_history: torch.Tensor,
        agent_states: torch.Tensor,
        lane_centers: torch.Tensor,
        traj_gt: Optional[torch.Tensor] = None,
        agent_mask: Optional[torch.Tensor] = None,
        lane_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass.

        Returns:
            Dictionary with all outputs for training/inference
        """
        # 1. Encode scene
        encoder_out = self.encode_scene(
            ego_history, agent_states, lane_centers, agent_mask, lane_mask
        )
        # M2ISceneEncoder returns (scene_embed, element_features, element_pad_mask)
        if isinstance(encoder_out, tuple):
            scene_embed, element_features, element_pad_mask = encoder_out
        else:
            scene_embed = encoder_out
            element_features, element_pad_mask = None, None

        # 2. Predict applicability
        applicability_logits, applicability_probs = self.predict_applicability(
            scene_embed, element_features, element_pad_mask
        )

        # 3. Generate trajectories
        cvae_outputs = self.generate_trajectories(scene_embed, traj_gt)

        # 4. Get best trajectory (for inference metrics)
        trajectory = cvae_outputs["trajectory"]
        trajectories = cvae_outputs["trajectories"]
        confidence = cvae_outputs["confidence"]

        return {
            # Scene
            "scene_embedding": scene_embed,
            # Applicability
            "applicability": applicability_logits,
            "applicability_prob": applicability_probs,
            # Trajectories
            "trajectory": trajectory,
            "trajectories": trajectories,
            "confidence": confidence,
            # Goals
            "goal_positions": cvae_outputs.get("goal_positions"),
            "goal_confidence": cvae_outputs.get("goal_confidence"),
            # CVAE
            "mu": cvae_outputs["mu"],
            "logvar": cvae_outputs["logvar"],
            "kl_loss": cvae_outputs["kl_loss"],
        }

    @torch.no_grad()
    def inference(
        self,
        ego_history: torch.Tensor,
        agent_states: torch.Tensor,
        lane_centers: torch.Tensor,
        num_samples: int = 6,
        agent_mask: Optional[torch.Tensor] = None,
        lane_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Inference mode (no GT)."""
        self.eval()
        return self.forward(
            ego_history,
            agent_states,
            lane_centers,
            traj_gt=None,
            agent_mask=agent_mask,
            lane_mask=lane_mask,
        )

    def get_parameter_groups(self, base_lr: float) -> List[Dict]:
        """
        Get parameter groups with different learning rates.

        - M2I encoder: 0.1x base_lr (fine-tune slowly)
        - Rest: base_lr
        """
        encoder_params = []
        other_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            if "scene_encoder" in name:
                encoder_params.append(param)
            else:
                other_params.append(param)

        return [
            {"params": encoder_params, "lr": base_lr * 0.1, "name": "encoder"},
            {"params": other_params, "lr": base_lr, "name": "other"},
        ]
