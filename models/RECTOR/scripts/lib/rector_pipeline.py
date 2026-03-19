"""
RECTOR Pipeline: M2I Trajectory Generation + Learned Applicability + Proxy Evaluation
                 + Lexicographic Trajectory Selection.

Architecture:
    Waymo Scenario Proto
      ├→ M2ITrajectoryGenerator (frozen)  → 6 trajectories [6, 80, 2]
      ├→ SceneFeatureExtractor            → scene_features dict (for proxies)
      ├→ M2ISceneEncoder (trainable)      → scene_embed [B, 256]
      │    └→ RuleApplicabilityHead       → applicability [B, 28]
      └→ DifferentiableRuleProxies        → violations [B, 6, 28]
           └→ TieredRuleScorer            → best trajectory + full audit

Training mode:
    Only the SceneEncoder + ApplicabilityHead are trained.
    Input: WaymoDataset batches (ego_history, agent_states, lane_centers)
    Loss: BCE on applicability logits vs ground truth labels.
    M2I trajectories are NOT needed during training.

Inference mode:
    Full pipeline: M2I generates → encoder predicts applicability →
    proxies evaluate → tiered scorer selects best trajectory.
    Returns: best trajectory, all trajectories, applicability, violations, tier scores.
"""

import sys
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn

# RECTOR imports
RECTOR_SCRIPTS = Path(__file__).parent.parent
sys.path.insert(0, str(RECTOR_SCRIPTS))
sys.path.insert(0, str(RECTOR_SCRIPTS / "lib"))
sys.path.insert(0, "/workspace/data/WOMD")
sys.path.insert(0, "/workspace/data")

from models.m2i_encoder import M2ISceneEncoder
from models.applicability_head import RuleApplicabilityHead
from models.tiered_scorer import TieredRuleScorer
from proxies.aggregator import DifferentiableRuleProxies
from lib.scene_feature_extractor import (
    extract_scene_features,
    extract_encoder_inputs,
    transform_trajectories_to_local,
)
from lib.m2i_trajectory_generator import M2ITrajectoryGenerator

try:
    from waymo_rule_eval.rules.rule_constants import (
        NUM_RULES,
        RULE_IDS,
        TIERS,
        get_tier_mask,
    )
except ImportError:
    NUM_RULES = 28
    RULE_IDS = []
    TIERS = ["safety", "legal", "road", "comfort"]


TRAJECTORY_SCALE = 50.0  # Training data normalization factor


@dataclass
class PipelineOutput:
    """Structured output from the RECTOR pipeline."""

    # Selected trajectory
    best_trajectory: np.ndarray  # [80, 2] in world coordinates
    best_index: int  # Index of selected trajectory (0-5)

    # All trajectory candidates
    trajectories_world: np.ndarray  # [M, 80, 2] world coordinates
    trajectories_local: np.ndarray  # [M, 80, 2] ego-local coordinates
    m2i_scores: np.ndarray  # [M] DenseTNT confidence scores

    # Applicability prediction
    applicability_logits: np.ndarray  # [28] raw logits
    applicability_probs: np.ndarray  # [28] sigmoid probabilities
    applicable_rules: List[str]  # Names of applicable rules (prob > 0.5)

    # Per-trajectory per-rule violation metrics
    violations: np.ndarray  # [M, 28] violation costs per trajectory per rule
    tier_violations: Dict[str, np.ndarray]  # tier_name → [M] aggregated per tier

    # Scoring
    scores: np.ndarray  # [M] composite lexicographic scores

    # Metadata
    scenario_id: str = ""
    agent_id: int = 0


class RECTORApplicabilityModel(nn.Module):
    """
    Trainable component: SceneEncoder + ApplicabilityHead.

    This is the learned part of the pipeline, predicting which of 28 rules
    apply to the current driving scenario based on scene context.

    Input: ego_history [B,11,4], agent_states [B,32,11,4], lane_centers [B,64,20,2]
    Output: applicability logits [B, 28], probs [B, 28]
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_rules: int = NUM_RULES,
        m2i_checkpoint: str = None,
        freeze_m2i: bool = False,
    ):
        super().__init__()

        self.scene_encoder = M2ISceneEncoder(
            m2i_checkpoint=m2i_checkpoint,
            embed_dim=embed_dim,
            freeze_m2i=freeze_m2i,
        )

        self.applicability_head = RuleApplicabilityHead(
            embed_dim=embed_dim,
            hidden_dim=int(embed_dim * 1.5),
            num_rules=num_rules,
        )

    def forward(
        self,
        ego_history: torch.Tensor,  # [B, 11, 4]
        agent_states: torch.Tensor,  # [B, 32, 11, 4]
        lane_centers: torch.Tensor,  # [B, 64, 20, 2]
        agent_mask: Optional[torch.Tensor] = None,
        lane_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training / inference.

        Returns:
            Dict with 'applicability' (logits [B,28]) and
            'applicability_prob' (probs [B,28]).
        """
        scene_embed, element_features, element_pad_mask = self.scene_encoder(
            ego_history,
            agent_states,
            lane_centers,
            agent_mask,
            lane_mask,
        )
        logits, probs = self.applicability_head(
            scene_embed,
            element_features,
            element_pad_mask,
        )
        return {
            "applicability": logits,
            "applicability_prob": probs,
            "scene_embedding": scene_embed,
        }

    def get_parameter_groups(self, base_lr: float) -> List[Dict]:
        """Encoder at 0.1x LR, applicability head at full LR."""
        encoder_params = []
        head_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "scene_encoder" in name:
                encoder_params.append(param)
            else:
                head_params.append(param)
        return [
            {"params": encoder_params, "lr": base_lr * 0.1, "name": "encoder"},
            {"params": head_params, "lr": base_lr, "name": "applicability_head"},
        ]


class RECTORPipeline:
    """
    Full RECTOR inference pipeline.

    Orchestrates:
      1. M2I trajectory generation (pretrained, frozen)
      2. Scene encoding + applicability prediction (trained)
      3. Differentiable rule proxy evaluation
      4. Lexicographic tiered trajectory selection

    Usage:
        pipeline = RECTORPipeline(device='cuda')
        pipeline.load(applicability_checkpoint='path/to/best.pt')
        result = pipeline.evaluate_scenario(scenario_proto)
        print(result.best_trajectory)        # [80, 2] world coords
        print(result.applicable_rules)       # ['speed_limit', 'keep_lane', ...]
        print(result.violations)             # [6, 28] per-traj per-rule costs
    """

    def __init__(
        self,
        device: str = "cuda",
        m2i_config_dir: str = "/workspace/externals/M2I/configs",
        m2i_checkpoint: str = "/workspace/models/pretrained/m2i/models/densetnt/model.24.bin",
        applicability_threshold: float = 0.5,
        embed_dim: int = 256,
    ):
        self.device = device
        self.m2i_config_dir = m2i_config_dir
        self.m2i_checkpoint = m2i_checkpoint
        self.applicability_threshold = applicability_threshold
        self.embed_dim = embed_dim

        # Components (initialized in load())
        self.m2i_generator: Optional[M2ITrajectoryGenerator] = None
        self.applicability_model: Optional[RECTORApplicabilityModel] = None
        self.rule_proxies: Optional[DifferentiableRuleProxies] = None
        self.tiered_scorer: Optional[TieredRuleScorer] = None

    def load(
        self,
        applicability_checkpoint: Optional[str] = None,
        m2i_encoder_checkpoint: Optional[str] = None,
    ):
        """
        Load all pipeline components.

        Args:
            applicability_checkpoint: Path to trained RECTORApplicabilityModel .pt file
            m2i_encoder_checkpoint: Optional separate M2I encoder checkpoint
        """
        # 1. M2I trajectory generator (pretrained, CPU/numpy)
        self.m2i_generator = M2ITrajectoryGenerator(
            config_dir=self.m2i_config_dir,
            checkpoint=self.m2i_checkpoint,
        )
        self.m2i_generator.load()

        # 2. Applicability model (trained encoder + head)
        self.applicability_model = RECTORApplicabilityModel(
            embed_dim=self.embed_dim,
            m2i_checkpoint=m2i_encoder_checkpoint or self.m2i_checkpoint,
        ).to(self.device)

        if applicability_checkpoint and os.path.exists(applicability_checkpoint):
            state = torch.load(applicability_checkpoint, map_location=self.device)
            if "model_state_dict" in state:
                self.applicability_model.load_state_dict(state["model_state_dict"])
            else:
                self.applicability_model.load_state_dict(state)
            print(f"Loaded applicability model from {applicability_checkpoint}")

        self.applicability_model.eval()

        # 3. Rule proxies (non-learned, physics-based)
        self.rule_proxies = DifferentiableRuleProxies(softness=5.0)
        self.rule_proxies.to(self.device)
        self.rule_proxies.eval()

        # 4. Tiered scorer (deterministic)
        self.tiered_scorer = TieredRuleScorer(use_learned_weights=False)

        print("RECTOR Pipeline loaded successfully")

    @torch.no_grad()
    def evaluate_scenario(
        self,
        scenario,
        target_agent_idx: int = 0,
    ) -> Optional[PipelineOutput]:
        """
        Run full pipeline on a single Waymo Scenario proto.

        Args:
            scenario: waymo_open_dataset.protos.scenario_pb2.Scenario
            target_agent_idx: 0 = SDC (default)

        Returns:
            PipelineOutput with all results, or None on failure.
        """
        # --- Step 1: Generate trajectories with M2I ---
        traj_world, m2i_scores, mapping = self.m2i_generator.predict_from_scenario(
            scenario,
            target_agent_idx=target_agent_idx,
        )
        if traj_world is None:
            return None

        M, T, _ = traj_world.shape  # [6, 80, 2]

        # Transform to ego-local frame
        sdc_idx = scenario.sdc_track_index
        current_ts = scenario.current_time_index
        ref_state = scenario.tracks[sdc_idx].states[current_ts]

        traj_local = transform_trajectories_to_local(
            traj_world,
            ref_state.center_x,
            ref_state.center_y,
            ref_state.heading,
        )

        # --- Step 2: Predict applicability ---
        encoder_inputs = extract_encoder_inputs(scenario)
        if encoder_inputs is None:
            return None

        # Scale by 1/50 to match training data format
        ego_h = torch.tensor(
            encoder_inputs["ego_history"], device=self.device
        ).unsqueeze(0)
        agent_s = torch.tensor(
            encoder_inputs["agent_states"], device=self.device
        ).unsqueeze(0)
        lane_c = torch.tensor(
            encoder_inputs["lane_centers"], device=self.device
        ).unsqueeze(0)

        ego_h[:, :, :2] /= TRAJECTORY_SCALE
        agent_s[:, :, :, :2] /= TRAJECTORY_SCALE
        lane_c /= TRAJECTORY_SCALE

        app_out = self.applicability_model(ego_h, agent_s, lane_c)
        app_logits = app_out["applicability"]  # [1, 28]
        app_probs = app_out["applicability_prob"]  # [1, 28]

        # --- Step 3: Evaluate trajectories with rule proxies ---
        scene_features = extract_scene_features(scenario, device=self.device)

        # Trajectories tensor: [1, M, 80, 2] in ego-local meters
        traj_tensor = torch.tensor(
            traj_local,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(
            0
        )  # [1, 6, 80, 2]

        violations = self.rule_proxies(traj_tensor, scene_features)  # [1, 6, 28]

        # --- Step 4: Lexicographic trajectory selection ---
        # Binary applicability mask from thresholded probabilities
        app_mask = (app_probs > self.applicability_threshold).float()  # [1, 28]

        scorer_out = self.tiered_scorer(violations, app_mask)
        best_idx = scorer_out["best_indices"][0].item()

        # --- Build output ---
        applicable_rule_names = []
        if RULE_IDS:
            for i, rid in enumerate(RULE_IDS):
                if app_probs[0, i].item() > self.applicability_threshold:
                    applicable_rule_names.append(rid)

        return PipelineOutput(
            best_trajectory=traj_world[best_idx],
            best_index=best_idx,
            trajectories_world=traj_world,
            trajectories_local=traj_local,
            m2i_scores=m2i_scores,
            applicability_logits=app_logits[0].cpu().numpy(),
            applicability_probs=app_probs[0].cpu().numpy(),
            applicable_rules=applicable_rule_names,
            violations=violations[0].cpu().numpy(),
            tier_violations={
                k: v[0].cpu().numpy() for k, v in scorer_out["tier_violations"].items()
            },
            scores=scorer_out["scores"][0].cpu().numpy(),
            scenario_id=(
                scenario.scenario_id if hasattr(scenario, "scenario_id") else ""
            ),
            agent_id=target_agent_idx,
        )

    @torch.no_grad()
    def evaluate_batch(
        self,
        ego_history: torch.Tensor,  # [B, 11, 4]
        agent_states: torch.Tensor,  # [B, 32, 11, 4]
        lane_centers: torch.Tensor,  # [B, 64, 20, 2]
    ) -> Dict[str, torch.Tensor]:
        """
        Predict applicability only (for batch evaluation during training/val).

        Input is already in training format (scaled by 1/50).

        Returns:
            Dict with applicability logits and probs.
        """
        return self.applicability_model(
            ego_history.to(self.device),
            agent_states.to(self.device),
            lane_centers.to(self.device),
        )
