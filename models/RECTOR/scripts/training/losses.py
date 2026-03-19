"""
RECTOR Loss Functions.

Components:
1. Winner-Takes-All (WTA) - Only train best mode
2. minADE/minFDE optimization
3. Temporal weighting (endpoint emphasized)
4. Huber loss for robustness
5. Goal prediction loss
6. Smoothness regularization
7. Rule applicability prediction (BCE)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


# Trajectory scale for converting to meters
TRAJECTORY_SCALE = 50.0


class RECTORLoss(nn.Module):
    """
    Loss function for RECTOR trajectory prediction.

    Components:
    1. WTA Reconstruction Loss - Only on best mode
    2. Endpoint Loss - Extra weight on final position
    3. KL Divergence - Regularize latent space
    4. Goal Loss - Train goal prediction head
    5. Smoothness Loss - Penalize jerky trajectories
    6. Confidence Loss - Train mode confidence with oracle
    """

    def __init__(
        self,
        reconstruction_weight: float = 20.0,
        endpoint_weight: float = 5.0,
        kl_weight: float = 0.05,
        goal_weight: float = 2.0,
        smoothness_weight: float = 1.0,
        confidence_weight: float = 1.0,
        applicability_weight: float = 1.0,
        trajectory_scale: float = TRAJECTORY_SCALE,
    ):
        super().__init__()

        self.reconstruction_weight = reconstruction_weight
        self.endpoint_weight = endpoint_weight
        self.kl_weight = kl_weight
        self.goal_weight = goal_weight
        self.smoothness_weight = smoothness_weight
        self.confidence_weight = confidence_weight
        self.applicability_weight = applicability_weight
        self.trajectory_scale = trajectory_scale

        # Huber loss for robustness to outliers
        self.huber = nn.SmoothL1Loss(reduction="none", beta=0.5)

    def compute_wta_loss(
        self,
        pred_trajectories: torch.Tensor,
        gt_trajectory: torch.Tensor,
    ) -> tuple:
        """
        Winner-Takes-All reconstruction loss.

        Only backprop through the best mode (lowest error).

        Args:
            pred_trajectories: [B, M, T, 4]
            gt_trajectory: [B, T, 4]

        Returns:
            wta_loss: scalar
            best_mode_idx: [B] indices of best modes
            minADE: [B] in meters
            minFDE: [B] in meters
        """
        B, M, T, D = pred_trajectories.shape
        device = pred_trajectories.device

        # Expand GT for comparison
        gt_expanded = gt_trajectory.unsqueeze(1).expand(-1, M, -1, -1)  # [B, M, T, 4]

        # Compute position error for each mode (x, y only)
        pos_error = pred_trajectories[..., :2] - gt_expanded[..., :2]  # [B, M, T, 2]
        pos_dist = torch.norm(pos_error, dim=-1)  # [B, M, T]

        # ADE per mode (in normalized space)
        ade_per_mode = pos_dist.mean(dim=-1)  # [B, M]

        # Find best mode per sample
        best_mode_idx = ade_per_mode.argmin(dim=1)  # [B]

        # Get best trajectories
        batch_idx = torch.arange(B, device=device)
        best_pred = pred_trajectories[batch_idx, best_mode_idx]  # [B, T, 4]

        # Temporal weighting - emphasize later timesteps
        # Weight increases from 1.0 to 2.0 over trajectory
        temporal_weights = torch.linspace(1.0, 2.0, T, device=device)  # [T]
        temporal_weights = temporal_weights / temporal_weights.mean()  # Normalize

        # Huber loss on best mode
        error = self.huber(best_pred, gt_trajectory)  # [B, T, 4]

        # Weight position dims more than heading/speed
        dim_weights = torch.tensor([2.0, 2.0, 0.5, 0.5], device=device)  # [4]

        weighted_error = error * temporal_weights.unsqueeze(-1) * dim_weights
        wta_loss = weighted_error.mean()

        # Compute metrics in meters
        minADE = (ade_per_mode.min(dim=1)[0] * self.trajectory_scale).mean()

        fde_per_mode = pos_dist[:, :, -1]  # [B, M]
        minFDE = (fde_per_mode[batch_idx, best_mode_idx] * self.trajectory_scale).mean()

        return wta_loss, best_mode_idx, minADE, minFDE

    def compute_endpoint_loss(
        self,
        pred_trajectories: torch.Tensor,
        gt_trajectory: torch.Tensor,
        best_mode_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Extra loss on final position (FDE-focused)."""
        B = pred_trajectories.shape[0]
        device = pred_trajectories.device

        batch_idx = torch.arange(B, device=device)
        best_pred = pred_trajectories[batch_idx, best_mode_idx]  # [B, T, 4]

        # Final position error (x, y)
        endpoint_error = self.huber(best_pred[:, -1, :2], gt_trajectory[:, -1, :2])

        return endpoint_error.mean()

    def compute_goal_loss(
        self,
        goal_positions: torch.Tensor,
        gt_trajectory: torch.Tensor,
    ) -> torch.Tensor:
        """
        Train goal head to predict trajectory endpoint.

        Args:
            goal_positions: [B, M, 2] predicted goals
            gt_trajectory: [B, T, 4]
        """
        # GT endpoint
        gt_endpoint = gt_trajectory[:, -1, :2]  # [B, 2]

        # Find closest goal
        goal_dist = torch.norm(
            goal_positions - gt_endpoint.unsqueeze(1), dim=-1
        )  # [B, M]
        best_goal_idx = goal_dist.argmin(dim=1)  # [B]

        B = goal_positions.shape[0]
        batch_idx = torch.arange(B, device=goal_positions.device)
        best_goal = goal_positions[batch_idx, best_goal_idx]  # [B, 2]

        goal_loss = self.huber(best_goal, gt_endpoint).mean()

        return goal_loss

    def compute_smoothness_loss(
        self,
        pred_trajectories: torch.Tensor,
        best_mode_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Penalize jerky trajectories (high jerk)."""
        B = pred_trajectories.shape[0]
        device = pred_trajectories.device

        batch_idx = torch.arange(B, device=device)
        best_pred = pred_trajectories[batch_idx, best_mode_idx]  # [B, T, 4]

        # Compute velocities (first derivative)
        velocity = best_pred[:, 1:, :2] - best_pred[:, :-1, :2]  # [B, T-1, 2]

        # Compute accelerations (second derivative)
        acceleration = velocity[:, 1:] - velocity[:, :-1]  # [B, T-2, 2]

        # Compute jerk (third derivative)
        jerk = acceleration[:, 1:] - acceleration[:, :-1]  # [B, T-3, 2]

        # Penalize high jerk
        jerk_penalty = torch.norm(jerk, dim=-1).mean()

        return jerk_penalty

    def compute_confidence_loss(
        self,
        confidence: torch.Tensor,
        best_mode_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Train confidence head to predict best mode."""
        B = confidence.shape[0]
        device = confidence.device

        # Cross-entropy loss with oracle mode as target
        target = F.one_hot(best_mode_idx, num_classes=confidence.shape[1]).float()

        # Use KL divergence between predicted and oracle distribution
        log_conf = F.log_softmax(confidence, dim=1)
        conf_loss = F.kl_div(log_conf, target, reduction="batchmean")

        return conf_loss

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all losses.

        Args:
            outputs: Model outputs with 'trajectories', 'confidence', 'goal_positions', 'kl_loss'
            targets: Must include 'traj_gt'

        Returns:
            Dictionary with 'total_loss' and component losses
        """
        pred_trajectories = outputs["trajectories"]  # [B, M, T, 4]
        gt_trajectory = targets["traj_gt"]  # [B, T, 4]

        # Handle trajectory length mismatch
        if gt_trajectory.shape[1] > pred_trajectories.shape[2]:
            gt_trajectory = gt_trajectory[:, : pred_trajectories.shape[2], :]

        # 1. WTA Reconstruction Loss
        wta_loss, best_mode_idx, minADE, minFDE = self.compute_wta_loss(
            pred_trajectories, gt_trajectory
        )

        # 2. Endpoint Loss (FDE-focused)
        endpoint_loss = self.compute_endpoint_loss(
            pred_trajectories, gt_trajectory, best_mode_idx
        )

        # 3. KL Divergence
        kl_loss = outputs.get(
            "kl_loss", torch.tensor(0.0, device=pred_trajectories.device)
        )

        # 4. Goal Loss
        goal_loss = torch.tensor(0.0, device=pred_trajectories.device)
        if "goal_positions" in outputs:
            goal_loss = self.compute_goal_loss(outputs["goal_positions"], gt_trajectory)

        # 5. Smoothness Loss
        smoothness_loss = self.compute_smoothness_loss(pred_trajectories, best_mode_idx)

        # 6. Confidence Loss
        confidence_loss = torch.tensor(0.0, device=pred_trajectories.device)
        if "confidence" in outputs:
            confidence_loss = self.compute_confidence_loss(
                outputs["confidence"], best_mode_idx
            )

        # 7. Applicability Loss (BCE on rule applicability logits)
        applicability_loss = torch.tensor(0.0, device=pred_trajectories.device)
        if "applicability" in outputs and "applicability_gt" in targets:
            app_logits = outputs["applicability"]  # [B, R]
            app_gt = targets["applicability_gt"]  # [B, R]
            applicability_loss = F.binary_cross_entropy_with_logits(app_logits, app_gt)

        # Total weighted loss
        total_loss = (
            self.reconstruction_weight * wta_loss
            + self.endpoint_weight * endpoint_loss
            + self.kl_weight * kl_loss
            + self.goal_weight * goal_loss
            + self.smoothness_weight * smoothness_loss
            + self.confidence_weight * confidence_loss
            + self.applicability_weight * applicability_loss
        )

        return {
            "total_loss": total_loss,
            "reconstruction_loss": wta_loss,
            "endpoint_loss": endpoint_loss,
            "kl_loss": kl_loss,
            "goal_loss": goal_loss,
            "smoothness_loss": smoothness_loss,
            "confidence_loss": confidence_loss,
            "applicability_loss": applicability_loss,
            "minADE": minADE,
            "minFDE": minFDE,
        }
