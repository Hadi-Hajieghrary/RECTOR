"""
CVAE Trajectory Head.

Key Improvements:
1. Transformer Decoder (parallel, no error accumulation)
2. Goal-Conditioned Prediction
3. Lane Cross-Attention
4. 5-Second Horizon (50 steps instead of 80)
5. Winner-Takes-All Training
6. Trajectory Refinement Network

Architecture:
- Prior network: p(z|scene) - generates z from scene alone
- Posterior network: q(z|scene, traj_gt) - uses GT during training
- Transformer Decoder: parallel trajectory generation
- Goal Head: predicts likely endpoints from lanes
- Refiner: post-process for smoothness
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import math


# New constants for 5-second prediction
TRAJECTORY_LENGTH = 50  # 5 seconds at 10Hz
HISTORY_LENGTH = 11  # 1.1 seconds


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""

    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TrajectoryEncoder(nn.Module):
    """Encode a trajectory into a fixed-dimensional vector using 1D convolutions."""

    def __init__(
        self,
        trajectory_length: int = TRAJECTORY_LENGTH,
        state_dim: int = 4,
        hidden_dim: int = 128,
        output_dim: int = 128,
    ):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(state_dim, hidden_dim // 2, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, trajectory: torch.Tensor) -> torch.Tensor:
        """Encode trajectory. [B, T, D] -> [B, output_dim]"""
        x = trajectory.transpose(1, 2)  # [B, D, T]
        x = self.conv(x).squeeze(-1)  # [B, hidden]
        return self.proj(x)


class GaussianParams(nn.Module):
    """Output mean and log variance for Gaussian distribution."""

    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.mean = nn.Linear(input_dim, latent_dim)
        self.logvar = nn.Linear(input_dim, latent_dim)

        nn.init.zeros_(self.mean.weight)
        nn.init.zeros_(self.mean.bias)
        nn.init.zeros_(self.logvar.weight)
        nn.init.zeros_(self.logvar.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = self.mean(x)
        logvar = self.logvar(x).clamp(min=-10, max=2)
        return mean, logvar


class GoalPredictionHead(nn.Module):
    """Predict likely goal points from lane information."""

    def __init__(self, scene_dim: int = 256, num_goals: int = 6, goal_dim: int = 64):
        super().__init__()
        self.num_goals = num_goals

        # Goal queries - learnable
        self.goal_queries = nn.Parameter(torch.randn(num_goals, goal_dim))

        # Cross-attention to scene
        self.cross_attn = nn.MultiheadAttention(goal_dim, num_heads=4, batch_first=True)

        # Project scene to goal dim
        self.scene_proj = nn.Linear(scene_dim, goal_dim)

        # Output goal positions
        self.goal_proj = nn.Sequential(
            nn.Linear(goal_dim, goal_dim),
            nn.ReLU(),
            nn.Linear(goal_dim, 2),  # (x, y) position
        )

        # Goal confidence
        self.confidence = nn.Sequential(
            nn.Linear(goal_dim, goal_dim // 2),
            nn.ReLU(),
            nn.Linear(goal_dim // 2, 1),
        )

    def forward(
        self, scene_embedding: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict goal positions.

        Args:
            scene_embedding: [B, scene_dim]

        Returns:
            goal_positions: [B, num_goals, 2]
            goal_features: [B, num_goals, goal_dim]
            goal_confidence: [B, num_goals]
        """
        B = scene_embedding.shape[0]

        # Expand queries for batch
        queries = self.goal_queries.unsqueeze(0).expand(
            B, -1, -1
        )  # [B, num_goals, goal_dim]

        # Project scene and use as key/value
        scene_proj = self.scene_proj(scene_embedding).unsqueeze(1)  # [B, 1, goal_dim]

        # Cross-attention
        goal_features, _ = self.cross_attn(
            queries, scene_proj, scene_proj
        )  # [B, num_goals, goal_dim]
        goal_features = goal_features + queries  # Residual

        # Predict positions and confidence
        goal_positions = self.goal_proj(goal_features)  # [B, num_goals, 2]
        goal_confidence = self.confidence(goal_features).squeeze(-1)  # [B, num_goals]

        return goal_positions, goal_features, goal_confidence


class TransformerTrajectoryDecoder(nn.Module):
    """
    Transformer-based trajectory decoder.

    Key advantages over GRU:
    1. Parallel decoding - no sequential error accumulation
    2. Self-attention captures long-range dependencies
    3. Cross-attention to scene and goals
    """

    def __init__(
        self,
        scene_dim: int = 256,
        latent_dim: int = 64,
        goal_dim: int = 64,
        hidden_dim: int = 256,
        trajectory_length: int = TRAJECTORY_LENGTH,
        output_dim: int = 4,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.trajectory_length = trajectory_length
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        # Input projection
        self.input_proj = nn.Linear(scene_dim + latent_dim + goal_dim, hidden_dim)

        # Learnable trajectory queries (one per timestep)
        self.trajectory_queries = nn.Parameter(
            torch.randn(trajectory_length, hidden_dim) * 0.02
        )

        # Positional encoding
        self.pos_encoder = PositionalEncoding(
            hidden_dim, max_len=trajectory_length, dropout=dropout
        )

        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers
        )

        # Output projection - predicts (dx, dy, heading, speed) per timestep
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

        # Learnable output scale per dimension - LARGER for better coverage
        # At 10Hz, typical displacement per step: 0.5-3m = 0.01-0.06 normalized
        # Scale should allow up to ~0.1 per step for diversity
        self.output_scale = nn.Parameter(torch.tensor([0.15, 0.15, 0.04, 0.10]))

        self._init_weights()

    def _init_weights(self):
        """Initialize for stable training - outputs should start near zero."""
        # Initialize final layer with very small weights for near-zero initial outputs
        nn.init.zeros_(self.output_proj[-1].weight)
        nn.init.zeros_(self.output_proj[-1].bias)

        # Initialize trajectory queries small
        nn.init.normal_(self.trajectory_queries, std=0.02)

    def forward(
        self,
        scene_embedding: torch.Tensor,
        z: torch.Tensor,
        goal_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate trajectory using transformer decoder.

        Args:
            scene_embedding: [B, scene_dim]
            z: [B, latent_dim]
            goal_features: [B, goal_dim]

        Returns:
            trajectory: [B, T, output_dim]
        """
        B = scene_embedding.shape[0]
        device = scene_embedding.device

        # Combine inputs
        combined = torch.cat(
            [scene_embedding, z, goal_features], dim=-1
        )  # [B, combined_dim]
        memory = self.input_proj(combined).unsqueeze(1)  # [B, 1, hidden]

        # Trajectory queries with positional encoding
        queries = self.trajectory_queries.unsqueeze(0).expand(
            B, -1, -1
        )  # [B, T, hidden]
        queries = self.pos_encoder(queries)

        # Transformer decode - parallel generation!
        decoded = self.transformer_decoder(queries, memory)  # [B, T, hidden]

        # Project to trajectory space (deltas)
        raw_output = self.output_proj(decoded)  # [B, T, output_dim]

        # Scale output
        deltas = raw_output * self.output_scale.abs()

        # Cumulative sum to get absolute positions
        trajectory = torch.cumsum(deltas, dim=1)

        return trajectory


class TrajectoryRefiner(nn.Module):
    """
    Refine trajectories for global smoothness and consistency.

    Takes the raw trajectory and applies learned corrections
    based on global trajectory + scene context.
    """

    def __init__(
        self,
        trajectory_length: int = TRAJECTORY_LENGTH,
        state_dim: int = 4,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.trajectory_length = trajectory_length

        # Encode full trajectory
        self.traj_encoder = nn.Sequential(
            nn.Linear(trajectory_length * state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Predict residual correction
        self.residual_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, trajectory_length * state_dim),
        )

        # Small scale for residuals
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(
        self, trajectory: torch.Tensor, scene_embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        Refine trajectory.

        Args:
            trajectory: [B, T, 4]
            scene_embedding: [B, scene_dim] (unused for now, can add cross-attention)

        Returns:
            refined_trajectory: [B, T, 4]
        """
        B = trajectory.shape[0]

        # Encode trajectory
        traj_flat = trajectory.reshape(B, -1)  # [B, T*4]
        encoded = self.traj_encoder(traj_flat)  # [B, hidden]

        # Predict residual
        residual = self.residual_head(encoded)  # [B, T*4]
        residual = residual.reshape(B, self.trajectory_length, -1)  # [B, T, 4]

        # Apply small residual correction
        refined = trajectory + residual * self.residual_scale.abs()

        return refined


class CVAETrajectoryHead(nn.Module):
    """
    Improved CVAE for trajectory generation.

    Key improvements:
    1. Transformer decoder (parallel, no error accumulation)
    2. Goal-conditioned prediction
    3. Trajectory refinement
    4. 5-second horizon
    5. Winner-takes-all compatible
    """

    def __init__(
        self,
        scene_dim: int = 256,
        latent_dim: int = 64,
        hidden_dim: int = 256,
        trajectory_length: int = TRAJECTORY_LENGTH,
        output_dim: int = 4,
        num_modes: int = 6,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.scene_dim = scene_dim
        self.latent_dim = latent_dim
        self.num_modes = num_modes
        self.trajectory_length = trajectory_length
        self.output_dim = output_dim

        goal_dim = 64

        # Trajectory encoder (for posterior)
        self.trajectory_encoder = TrajectoryEncoder(
            trajectory_length=trajectory_length,
            output_dim=latent_dim * 2,
        )

        # Prior network: p(z|scene)
        self.prior_net = nn.Sequential(
            nn.Linear(scene_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.prior_params = GaussianParams(hidden_dim, latent_dim)

        # Posterior network: q(z|scene, traj)
        self.posterior_net = nn.Sequential(
            nn.Linear(scene_dim + latent_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.posterior_params = GaussianParams(hidden_dim, latent_dim)

        # Goal prediction
        self.goal_head = GoalPredictionHead(
            scene_dim, num_goals=num_modes, goal_dim=goal_dim
        )

        # Transformer decoder
        self.decoder = TransformerTrajectoryDecoder(
            scene_dim=scene_dim,
            latent_dim=latent_dim,
            goal_dim=goal_dim,
            hidden_dim=hidden_dim,
            trajectory_length=trajectory_length,
            output_dim=output_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Trajectory refiner
        self.refiner = TrajectoryRefiner(
            trajectory_length=trajectory_length,
            hidden_dim=hidden_dim,
        )

        # Mode confidence head (trained with oracle selection)
        self.confidence_head = nn.Sequential(
            nn.Linear(scene_dim + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def encode_prior(
        self, scene_embedding: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get prior distribution p(z|scene)."""
        h = self.prior_net(scene_embedding)
        return self.prior_params(h)

    def encode_posterior(
        self,
        scene_embedding: torch.Tensor,
        trajectory_gt: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get posterior distribution q(z|scene, traj)."""
        traj_encoded = self.trajectory_encoder(trajectory_gt)
        combined = torch.cat([scene_embedding, traj_encoded], dim=-1)
        h = self.posterior_net(combined)
        return self.posterior_params(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample z using reparameterization trick.

        Always samples (training and inference) so that each of the K modes
        draws a distinct latent vector and produces a diverse set of
        trajectories.  Returning the deterministic mean at inference caused
        all modes to receive the same z, collapsing mode diversity.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self,
        scene_embedding: torch.Tensor,
        traj_gt: Optional[torch.Tensor] = None,
        num_samples: int = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            scene_embedding: [B, scene_dim]
            traj_gt: [B, T, 4] ground truth (for training)
            num_samples: Number of modes to generate (default: self.num_modes)

        Returns:
            Dictionary with trajectories, confidences, and CVAE outputs
        """
        B = scene_embedding.shape[0]
        device = scene_embedding.device
        num_modes = num_samples or self.num_modes

        # Get goal predictions
        goal_positions, goal_features, goal_confidence = self.goal_head(scene_embedding)
        # goal_features: [B, num_modes, goal_dim]

        # Get prior
        prior_mu, prior_logvar = self.encode_prior(scene_embedding)

        # Get posterior (training only)
        if traj_gt is not None:
            post_mu, post_logvar = self.encode_posterior(scene_embedding, traj_gt)
            mu, logvar = post_mu, post_logvar
        else:
            mu, logvar = prior_mu, prior_logvar

        # Generate multiple trajectory modes
        trajectories = []
        confidences = []

        for m in range(num_modes):
            # Sample different z for each mode
            z = self.reparameterize(mu, logvar)

            # Use goal features for this mode
            goal_feat = goal_features[:, m % goal_features.shape[1], :]  # [B, goal_dim]

            # Decode trajectory
            traj = self.decoder(scene_embedding, z, goal_feat)  # [B, T, 4]

            # Refine trajectory
            traj = self.refiner(traj, scene_embedding)

            trajectories.append(traj)

            # Compute confidence for this mode
            conf_input = torch.cat([scene_embedding, z], dim=-1)
            conf = self.confidence_head(conf_input).squeeze(-1)  # [B]
            confidences.append(conf)

        # Stack modes
        trajectories = torch.stack(trajectories, dim=1)  # [B, M, T, 4]
        confidences = torch.stack(confidences, dim=1)  # [B, M]
        confidences = F.softmax(confidences, dim=1)  # Normalize

        # Select best trajectory (highest confidence)
        best_idx = confidences.argmax(dim=1)  # [B]
        best_trajectory = trajectories[
            torch.arange(B, device=device), best_idx
        ]  # [B, T, 4]

        # Compute KL divergence
        kl_loss = torch.tensor(0.0, device=device)
        if traj_gt is not None:
            kl_loss = -0.5 * torch.mean(
                1
                + post_logvar
                - prior_logvar
                - (post_logvar.exp() + (post_mu - prior_mu).pow(2)) / prior_logvar.exp()
            )

        return {
            "trajectory": best_trajectory,  # [B, T, 4]
            "trajectories": trajectories,  # [B, M, T, 4]
            "confidence": confidences,  # [B, M]
            "goal_positions": goal_positions,  # [B, M, 2]
            "goal_confidence": goal_confidence,  # [B, M]
            "mu": mu,  # [B, latent_dim]
            "logvar": logvar,  # [B, latent_dim]
            "prior_mu": prior_mu,  # [B, latent_dim]
            "prior_logvar": prior_logvar,  # [B, latent_dim]
            "kl_loss": kl_loss,  # scalar
        }


def compute_minADE_minFDE(
    pred_trajectories: torch.Tensor,
    gt_trajectory: torch.Tensor,
    scale: float = 50.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute minADE and minFDE over modes.

    Args:
        pred_trajectories: [B, M, T, 4] predicted modes
        gt_trajectory: [B, T, 4] ground truth
        scale: Trajectory normalization scale

    Returns:
        minADE: [B] minimum ADE over modes (in meters)
        minFDE: [B] minimum FDE over modes (in meters)
        best_mode_idx: [B] index of best mode for each sample
    """
    B, M, T, _ = pred_trajectories.shape

    # Expand GT for comparison with all modes
    gt_expanded = gt_trajectory.unsqueeze(1).expand(-1, M, -1, -1)  # [B, M, T, 4]

    # Position error (x, y only)
    pos_error = (
        pred_trajectories[..., :2] - gt_expanded[..., :2]
    ) * scale  # [B, M, T, 2]

    # ADE per mode: mean over timesteps
    ade_per_mode = torch.norm(pos_error, dim=-1).mean(dim=-1)  # [B, M]

    # FDE per mode: final timestep
    fde_per_mode = torch.norm(pos_error[:, :, -1, :], dim=-1)  # [B, M]

    # Min over modes
    minADE, best_mode_idx = ade_per_mode.min(dim=1)  # [B]
    minFDE = fde_per_mode[
        torch.arange(B, device=gt_trajectory.device), best_mode_idx
    ]  # [B]

    return minADE, minFDE, best_mode_idx
