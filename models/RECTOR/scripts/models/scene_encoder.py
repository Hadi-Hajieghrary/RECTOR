"""
Scene Encoder for RECTOR.

Encodes scene features (ego history, other agents, lanes, signals)
into a fixed-dimensional scene embedding for downstream heads.

Architecture:
- PolylineEncoder: MLP over each polyline → single vector
- GlobalAttention: Cross-attention between all elements
- EgoQueryAttention: Cross-attention from ego query to scene
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class PolylineEncoder(nn.Module):
    """
    Encodes a polyline (sequence of points) into a single feature vector.

    Uses MLP over each point followed by max pooling.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 128,
        num_layers: int = 3,
    ):
        super().__init__()

        layers = []
        in_d = input_dim

        for i in range(num_layers - 1):
            out_d = hidden_dim
            layers.extend(
                [
                    nn.Linear(in_d, out_d),
                    nn.LayerNorm(out_d),
                    nn.ReLU(),
                ]
            )
            in_d = out_d

        layers.append(nn.Linear(in_d, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        points: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode polylines.

        Args:
            points: [..., P, D] polyline points
            mask: [..., P] valid mask (optional)

        Returns:
            [...] single feature vector per polyline
        """
        # Apply MLP to each point
        features = self.mlp(points)  # [..., P, output_dim]

        if mask is not None:
            # Mask out invalid points
            mask = mask.unsqueeze(-1).float()
            features = features * mask
            # Max pool with masking
            features = features.masked_fill(mask == 0, -1e9)

        # Max pool over points dimension
        pooled = features.max(dim=-2)[0]  # [..., output_dim]

        return pooled


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention or cross-attention."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply attention with residual connection.

        Args:
            query: [B, Q, D]
            key: [B, K, D]
            value: [B, K, D]
            key_padding_mask: [B, K] True = ignore

        Returns:
            [B, Q, D]
        """
        attn_out, _ = self.attn(
            query,
            key,
            value,
            key_padding_mask=key_padding_mask,
        )
        return self.norm(query + self.dropout(attn_out))


class FeedForward(nn.Module):
    """Feed-forward network with residual connection."""

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.dropout(self.ff(x)))


class TransformerEncoderLayer(nn.Module):
    """Single transformer encoder layer."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        ff_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ff = FeedForward(embed_dim, ff_dim, dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, N, D]
            mask: [B, N] True = ignore

        Returns:
            [B, N, D]
        """
        x = self.self_attn(x, x, x, mask)
        x = self.ff(x)
        return x


class SceneEncoder(nn.Module):
    """
    Encodes complete scene into a fixed-dimensional embedding.

    Processes:
    - Ego history trajectory
    - Other agent trajectories
    - Lane centerlines
    - Additional map features

    Outputs scene embedding for downstream heads.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 3,
        history_length: int = 11,
        ego_state_dim: int = 4,
        agent_state_dim: int = 4,
        lane_point_dim: int = 2,
        max_agents: int = 32,
        max_lanes: int = 64,
        dropout: float = 0.1,
    ):
        """
        Initialize scene encoder.

        Args:
            embed_dim: Dimension of scene embedding
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            history_length: Number of history timesteps
            ego_state_dim: Dimension of ego state (x, y, yaw, speed)
            agent_state_dim: Dimension of agent state
            lane_point_dim: Dimension of lane points
            max_agents: Maximum number of agents
            max_lanes: Maximum number of lanes
            dropout: Dropout rate
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.max_agents = max_agents
        self.max_lanes = max_lanes

        # Polyline encoder hidden dim scales with embed_dim for better context
        polyline_hidden = max(64, embed_dim // 2)

        # Ego history encoder
        self.ego_encoder = PolylineEncoder(
            input_dim=ego_state_dim,
            hidden_dim=polyline_hidden,
            output_dim=embed_dim,
        )

        # Agent trajectory encoder
        self.agent_encoder = PolylineEncoder(
            input_dim=agent_state_dim,
            hidden_dim=polyline_hidden,
            output_dim=embed_dim,
        )

        # Lane encoder
        self.lane_encoder = PolylineEncoder(
            input_dim=lane_point_dim,
            hidden_dim=polyline_hidden,
            output_dim=embed_dim,
        )

        # Type embeddings
        self.type_embedding = nn.Embedding(4, embed_dim)  # ego, vehicle, ped, cyclist

        # Positional encoding for sequence elements
        self.position_embedding = nn.Parameter(
            torch.randn(1, max_agents + max_lanes + 1, embed_dim) * 0.02
        )

        # Transformer layers
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    ff_dim=embed_dim * 4,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        # Ego query for final embedding
        self.ego_query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.query_attn = MultiHeadAttention(embed_dim, num_heads, dropout)

        # Final projection
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(
        self,
        ego_history: torch.Tensor,
        agent_states: torch.Tensor,
        lane_centers: torch.Tensor,
        agent_mask: Optional[torch.Tensor] = None,
        lane_mask: Optional[torch.Tensor] = None,
        agent_types: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode scene features.

        Args:
            ego_history: [B, T_hist, 4] ego trajectory history
            agent_states: [B, N, T, 4] other agent states (use history portion)
            lane_centers: [B, L, P, 2] lane centerline points
            agent_mask: [B, N] True = valid agent
            lane_mask: [B, L] True = valid lane
            agent_types: [B, N] agent type indices

        Returns:
            scene_embedding: [B, embed_dim]
        """
        B = ego_history.shape[0]
        device = ego_history.device

        # Encode ego history
        ego_embed = self.ego_encoder(ego_history)  # [B, D]
        ego_embed = ego_embed.unsqueeze(1)  # [B, 1, D]

        # Add ego type embedding
        ego_type = torch.zeros(B, 1, dtype=torch.long, device=device)
        ego_embed = ego_embed + self.type_embedding(ego_type)

        # Encode agent histories (first T_hist timesteps)
        T_hist = ego_history.shape[1]
        agent_history = agent_states[:, :, :T_hist, :]  # [B, N, T_hist, 4]

        # Reshape for encoding
        N = agent_history.shape[1]
        agent_history_flat = agent_history.reshape(B * N, T_hist, -1)
        agent_embed = self.agent_encoder(agent_history_flat)  # [B*N, D]
        agent_embed = agent_embed.reshape(B, N, -1)  # [B, N, D]

        # Add agent type embeddings
        if agent_types is not None:
            # Shift by 1 (0 is ego)
            agent_type_embed = self.type_embedding(agent_types.long() + 1)
            agent_embed = agent_embed + agent_type_embed

        # Encode lanes
        L = lane_centers.shape[1]
        lane_flat = lane_centers.reshape(B * L, -1, 2)  # [B*L, P, 2]
        lane_embed = self.lane_encoder(lane_flat)  # [B*L, D]
        lane_embed = lane_embed.reshape(B, L, -1)  # [B, L, D]

        # Concatenate all elements
        all_embeds = torch.cat([ego_embed, agent_embed, lane_embed], dim=1)
        # [B, 1 + N + L, D]

        # Add positional encoding
        seq_len = all_embeds.shape[1]
        all_embeds = all_embeds + self.position_embedding[:, :seq_len, :]

        # Create attention mask
        # True = ignore in attention
        attn_mask = torch.zeros(B, 1 + N + L, dtype=torch.bool, device=device)
        attn_mask[:, 0] = False  # Ego always valid

        if agent_mask is not None:
            attn_mask[:, 1 : 1 + N] = ~agent_mask
        if lane_mask is not None:
            attn_mask[:, 1 + N : 1 + N + L] = ~lane_mask

        # Apply transformer layers
        x = all_embeds
        for layer in self.layers:
            x = layer(x, attn_mask)

        # Query attention to get final embedding
        query = self.ego_query.expand(B, -1, -1)  # [B, 1, D]
        scene_embed = self.query_attn(query, x, x, attn_mask)  # [B, 1, D]
        scene_embed = scene_embed.squeeze(1)  # [B, D]

        # Final projection
        scene_embed = self.output_proj(scene_embed)  # [B, D]

        return scene_embed

    def get_all_embeddings(
        self,
        ego_history: torch.Tensor,
        agent_states: torch.Tensor,
        lane_centers: torch.Tensor,
        agent_mask: Optional[torch.Tensor] = None,
        lane_mask: Optional[torch.Tensor] = None,
        agent_types: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get both scene embedding and per-element embeddings.

        Returns:
            Tuple of:
            - scene_embedding: [B, D]
            - element_embeddings: [B, 1+N+L, D]
        """
        # Similar to forward but also return element embeddings
        B = ego_history.shape[0]
        device = ego_history.device

        # Encode components (same as forward)
        ego_embed = self.ego_encoder(ego_history).unsqueeze(1)
        ego_type = torch.zeros(B, 1, dtype=torch.long, device=device)
        ego_embed = ego_embed + self.type_embedding(ego_type)

        T_hist = ego_history.shape[1]
        N = agent_states.shape[1]
        agent_history = agent_states[:, :, :T_hist, :]
        agent_history_flat = agent_history.reshape(B * N, T_hist, -1)
        agent_embed = self.agent_encoder(agent_history_flat).reshape(B, N, -1)

        if agent_types is not None:
            agent_embed = agent_embed + self.type_embedding(agent_types.long() + 1)

        L = lane_centers.shape[1]
        lane_flat = lane_centers.reshape(B * L, -1, 2)
        lane_embed = self.lane_encoder(lane_flat).reshape(B, L, -1)

        all_embeds = torch.cat([ego_embed, agent_embed, lane_embed], dim=1)
        seq_len = all_embeds.shape[1]
        all_embeds = all_embeds + self.position_embedding[:, :seq_len, :]

        attn_mask = torch.zeros(B, 1 + N + L, dtype=torch.bool, device=device)
        if agent_mask is not None:
            attn_mask[:, 1 : 1 + N] = ~agent_mask
        if lane_mask is not None:
            attn_mask[:, 1 + N : 1 + N + L] = ~lane_mask

        x = all_embeds
        for layer in self.layers:
            x = layer(x, attn_mask)

        # Get scene embedding via query attention
        query = self.ego_query.expand(B, -1, -1)
        scene_embed = self.query_attn(query, x, x, attn_mask).squeeze(1)
        scene_embed = self.output_proj(scene_embed)

        return scene_embed, x
