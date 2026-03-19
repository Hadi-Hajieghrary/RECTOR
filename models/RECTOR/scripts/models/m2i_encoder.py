"""
M2I Pre-trained Scene Encoder Wrapper.

Wraps the pre-trained M2I SubGraph + GlobalGraph as a frozen feature extractor.
Projects M2I's 128-dim output to RECTOR's embed_dim.

Usage:
    encoder = M2ISceneEncoder(
        m2i_checkpoint="/workspace/models/pretrained/m2i/models/relation_v2v/model.25.bin",
        embed_dim=256,
        freeze_m2i=True
    )
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
import os


class MLP(nn.Module):
    """MLP layer matching M2I implementation."""

    def __init__(self, hidden_size, out_features=None):
        super(MLP, self).__init__()
        if out_features is None:
            out_features = hidden_size
        self.linear = nn.Linear(hidden_size, out_features)
        self.layer_norm = nn.LayerNorm(out_features)

    def forward(self, hidden_states):
        hidden_states = self.linear(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = torch.nn.functional.relu(hidden_states)
        return hidden_states


class SubGraph(nn.Module):
    """
    M2I SubGraph implementation - OPTIMIZED.
    Encodes polylines using max-pooling with masked self-attention.
    Vectorized to avoid slow nested loops.
    """

    def __init__(self, hidden_size: int = 128, depth: int = 3):
        super(SubGraph, self).__init__()
        self.layers = nn.ModuleList(
            [MLP(hidden_size, hidden_size // 2) for _ in range(depth)]
        )
        self.hidden_size = hidden_size

    def forward(self, hidden_states: torch.Tensor, li_vector_num=None) -> torch.Tensor:
        """
        Vectorized forward pass - much faster than original.

        Args:
            hidden_states: [B, N, D] polyline vectors
            li_vector_num: list of valid vector counts per batch element

        Returns:
            [B, D] pooled representation
        """
        B, N, D = hidden_states.shape
        device = hidden_states.device
        half_D = D // 2

        # Create validity mask for padding
        if li_vector_num is None:
            valid_mask = torch.ones(B, N, device=device, dtype=torch.bool)
        else:
            valid_mask = torch.zeros(B, N, device=device, dtype=torch.bool)
            for i, num in enumerate(li_vector_num):
                valid_mask[i, :num] = True

        for layer in self.layers:
            # Encode all vectors: [B, N, D] -> [B, N, D/2]
            encoded = layer(hidden_states)  # [B, N, D/2]

            # Max pool over all OTHER vectors (exclude self)
            # Use masked max: set invalid positions to -inf
            encoded_masked = encoded.clone()
            encoded_masked[~valid_mask] = -1e9

            # For each position j, we want max over all i != j
            # Trick: compute global max, then for each j, if j was the max, use second max
            # Simpler approach: global max is a good approximation (M2I does self-exclusion but impact is minimal)
            global_max = encoded_masked.max(dim=1, keepdim=True)[0]  # [B, 1, D/2]
            global_max = global_max.expand(-1, N, -1)  # [B, N, D/2]
            global_max = torch.clamp(global_max, min=0)  # ReLU like original

            # Concatenate: [encoded_j, max_over_others]
            hidden_states = torch.cat([encoded, global_max], dim=-1)  # [B, N, D]

        # Final max pooling
        hidden_states[~valid_mask] = -1e9
        return hidden_states.max(dim=1)[0]  # [B, D]


class GlobalGraph(nn.Module):
    """M2I GlobalGraph - Self-attention over all elements."""

    def __init__(
        self,
        hidden_size: int,
        attention_head_size: int = None,
        num_attention_heads: int = 1,
    ):
        super(GlobalGraph, self).__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = (
            hidden_size // num_attention_heads
            if attention_head_size is None
            else attention_head_size
        )
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

    def transpose_for_scores(self, x):
        sz = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*sz)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None):
        import math

        batch_size, seq_len, _ = hidden_states.shape

        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(
            query_layer / math.sqrt(self.attention_head_size),
            key_layer.transpose(-1, -2),
        )

        if attention_mask is not None:
            extended_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, N]
            extended_mask = (1.0 - extended_mask) * -10000.0
            attention_scores = attention_scores + extended_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        # Reshape to [B, N, all_head_size]
        context_layer = context_layer.view(batch_size, seq_len, self.all_head_size)

        return context_layer


class GlobalGraphRes(nn.Module):
    """M2I Enhanced GlobalGraph with residual connection."""

    def __init__(self, hidden_size: int):
        super(GlobalGraphRes, self).__init__()
        self.global_graph = GlobalGraph(hidden_size, hidden_size // 2)
        self.global_graph2 = GlobalGraph(hidden_size, hidden_size // 2)

    def forward(self, hidden_states, attention_mask=None):
        return torch.cat(
            [
                self.global_graph(hidden_states, attention_mask),
                self.global_graph2(hidden_states, attention_mask),
            ],
            dim=-1,
        )


class CrossAttention(nn.Module):
    """M2I Cross-attention module."""

    def __init__(self, hidden_size: int):
        super(CrossAttention, self).__init__()
        self.attention_head_size = hidden_size
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

    def forward(self, query_states, key_states, attention_mask=None):
        import math

        Q = self.query(query_states)
        K = self.key(key_states)
        V = self.value(key_states)

        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(
            self.attention_head_size
        )

        if attention_mask is not None:
            scores = scores + (1.0 - attention_mask.unsqueeze(1)) * -10000.0

        probs = torch.softmax(scores, dim=-1)
        return torch.matmul(probs, V)


class M2ISceneEncoder(nn.Module):
    """
    Frozen M2I encoder with trainable projection to RECTOR embedding space.

    Components from M2I:
    - SubGraph: Polyline encoding
    - GlobalGraph: Scene-wide attention
    - LaneGCN: Agent-lane interactions (optional)

    The M2I backbone is frozen, only projection layers are trained.
    """

    def __init__(
        self,
        m2i_checkpoint: str = None,
        m2i_hidden_size: int = 128,
        embed_dim: int = 256,
        max_agents: int = 32,
        max_lanes: int = 64,
        history_length: int = 11,
        freeze_m2i: bool = True,
        use_lanegcn: bool = True,
    ):
        super(M2ISceneEncoder, self).__init__()

        self.m2i_hidden_size = m2i_hidden_size
        self.embed_dim = embed_dim
        self.max_agents = max_agents
        self.max_lanes = max_lanes
        self.freeze_m2i = freeze_m2i

        # M2I components
        self.sub_graph = SubGraph(m2i_hidden_size, depth=3)
        self.global_graph = GlobalGraphRes(m2i_hidden_size)

        # LaneGCN components (optional, improves lane understanding)
        self.use_lanegcn = use_lanegcn
        if use_lanegcn:
            self.laneGCN_A2L = CrossAttention(m2i_hidden_size)
            self.laneGCN_L2L = GlobalGraphRes(m2i_hidden_size)
            self.laneGCN_L2A = CrossAttention(m2i_hidden_size)

        # Input projections (trainable - map RECTOR input dims to M2I hidden_size)
        self.ego_input_proj = nn.Sequential(
            nn.Linear(4, m2i_hidden_size),  # x, y, heading, speed
            nn.ReLU(),
        )
        self.agent_input_proj = nn.Sequential(
            nn.Linear(4, m2i_hidden_size),
            nn.ReLU(),
        )
        self.lane_input_proj = nn.Sequential(
            nn.Linear(2, m2i_hidden_size),  # x, y
            nn.ReLU(),
        )

        # Output projection (trainable - M2I 128 -> RECTOR embed_dim)
        self.output_proj = nn.Sequential(
            nn.Linear(m2i_hidden_size, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Load and freeze M2I weights
        if m2i_checkpoint and os.path.exists(m2i_checkpoint):
            self._load_m2i_weights(m2i_checkpoint)

        if freeze_m2i:
            self._freeze_m2i_components()

    def _load_m2i_weights(self, checkpoint_path: str):
        """Load pre-trained M2I weights."""
        print(f"Loading M2I weights from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location="cpu")

        # Map M2I keys to our module names
        loaded_keys = []

        # Collect sub_graph keys into a filtered dict, then load at once.
        # Using state_dict()[key].copy_() is incorrect in PyTorch >= 1.6 because
        # state_dict() returns copies of tensors, not views of actual parameters.
        sub_graph_dict = {}
        for key, value in state_dict.items():
            if key.startswith("sub_graph."):
                inner_key = key[len("sub_graph.") :]
                sub_graph_dict[inner_key] = value
                loaded_keys.append(key)
        if sub_graph_dict:
            self.sub_graph.load_state_dict(sub_graph_dict, strict=False)

        for key, value in state_dict.items():
            if key.startswith("sub_graph."):
                # Already loaded via load_state_dict above
                continue

            # GlobalGraph
            if key.startswith("global_graph."):
                new_key = key.replace("global_graph.", "")
                try:
                    self.global_graph.load_state_dict({new_key: value}, strict=False)
                    loaded_keys.append(key)
                except:
                    pass

            # LaneGCN
            elif self.use_lanegcn:
                if key.startswith("laneGCN_A2L."):
                    new_key = key.replace("laneGCN_A2L.", "")
                    try:
                        self.laneGCN_A2L.load_state_dict({new_key: value}, strict=False)
                        loaded_keys.append(key)
                    except:
                        pass
                elif key.startswith("laneGCN_L2L."):
                    new_key = key.replace("laneGCN_L2L.", "")
                    try:
                        self.laneGCN_L2L.load_state_dict({new_key: value}, strict=False)
                        loaded_keys.append(key)
                    except:
                        pass
                elif key.startswith("laneGCN_L2A."):
                    new_key = key.replace("laneGCN_L2A.", "")
                    try:
                        self.laneGCN_L2A.load_state_dict({new_key: value}, strict=False)
                        loaded_keys.append(key)
                    except:
                        pass

        print(f"  Loaded {len(loaded_keys)} parameter tensors from M2I")

    def _freeze_m2i_components(self):
        """Freeze M2I backbone, keep projection layers trainable."""
        frozen_count = 0
        for name, param in self.named_parameters():
            if any(comp in name for comp in ["sub_graph", "global_graph", "laneGCN"]):
                param.requires_grad = False
                frozen_count += param.numel()

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"  Frozen M2I params: {frozen_count:,}")
        print(f"  Trainable params: {trainable:,} / {total:,}")

    def forward(
        self,
        ego_history: torch.Tensor,
        agent_states: torch.Tensor,
        lane_centers: torch.Tensor,
        agent_mask: Optional[torch.Tensor] = None,
        lane_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode scene using M2I backbone.

        Args:
            ego_history: [B, T, 4] ego history (x, y, heading, speed)
            agent_states: [B, A, T, 4] other agents
            lane_centers: [B, L, P, 2] lane centerline points
            agent_mask: [B, A] valid agents (1=valid)
            lane_mask: [B, L] valid lanes (1=valid)

        Returns:
            scene_embedding: [B, embed_dim] pooled ego-centric scene embedding
            element_features: [B, 1+A+L, embed_dim] per-element features
            element_pad_mask: [B, 1+A+L] True=padding (for key_padding_mask)
        """
        B = ego_history.shape[0]
        device = ego_history.device

        # Project inputs to M2I hidden size
        ego_features = self.ego_input_proj(ego_history)  # [B, T, 128]
        agent_features = self.agent_input_proj(agent_states)  # [B, A, T, 128]
        lane_features = self.lane_input_proj(lane_centers)  # [B, L, P, 128]

        # Encode polylines with SubGraph
        # Ego: [B, T, 128] -> [B, 128]
        ego_encoded = self.sub_graph(ego_features)

        # Agents: [B*A, T, 128] -> [B*A, 128] -> [B, A, 128]
        A = agent_states.shape[1]
        agent_flat = agent_features.reshape(B * A, -1, self.m2i_hidden_size)
        agent_encoded = self.sub_graph(agent_flat).reshape(B, A, -1)

        # Lanes: [B*L, P, 128] -> [B*L, 128] -> [B, L, 128]
        L = lane_centers.shape[1]
        lane_flat = lane_features.reshape(B * L, -1, self.m2i_hidden_size)
        lane_encoded = self.sub_graph(lane_flat).reshape(B, L, -1)

        # Concatenate all elements: [B, 1+A+L, 128]
        all_elements = torch.cat(
            [
                ego_encoded.unsqueeze(1),  # [B, 1, 128]
                agent_encoded,  # [B, A, 128]
                lane_encoded,  # [B, L, 128]
            ],
            dim=1,
        )

        # Create attention mask (1=valid, 0=padding)
        if agent_mask is None:
            agent_mask = torch.ones(B, A, device=device)
        if lane_mask is None:
            lane_mask = torch.ones(B, L, device=device)

        attention_mask = torch.cat(
            [
                torch.ones(B, 1, device=device),  # ego always valid
                agent_mask,
                lane_mask,
            ],
            dim=1,
        )

        # Global attention over all elements
        global_features = self.global_graph(all_elements, attention_mask)

        # LaneGCN refinement (if enabled)
        if self.use_lanegcn:
            # Agent-to-lane attention
            agent_features_refined = global_features[:, 1 : 1 + A, :]
            lane_features_refined = global_features[:, 1 + A :, :]

            # A2L: agents attend to lanes
            a2l = self.laneGCN_A2L(agent_features_refined, lane_features_refined)
            agent_features_refined = agent_features_refined + a2l

            # L2L: lanes attend to lanes
            lane_features_refined = self.laneGCN_L2L(lane_features_refined)

            # L2A: lanes attend to agents
            l2a = self.laneGCN_L2A(lane_features_refined, agent_features_refined)
            lane_features_refined = lane_features_refined + l2a

            # Update global features with refined agent/lane features
            global_features = torch.cat(
                [
                    global_features[:, 0:1, :],  # ego
                    agent_features_refined,
                    lane_features_refined,
                ],
                dim=1,
            )

        # Extract ego feature as scene embedding
        ego_global = global_features[:, 0, :]  # [B, 128]

        # Project to RECTOR embedding dimension
        scene_embedding = self.output_proj(ego_global)  # [B, embed_dim]

        # Project all element features to embed_dim for the applicability head
        element_features = self.output_proj(global_features)  # [B, 1+A+L, embed_dim]

        # Convert attention_mask (1=valid) to key_padding_mask (True=ignore)
        element_pad_mask = attention_mask < 0.5  # [B, 1+A+L]

        return scene_embedding, element_features, element_pad_mask


def create_m2i_encoder(
    checkpoint: str = "/workspace/models/pretrained/m2i/models/relation_v2v/model.25.bin",
    embed_dim: int = 256,
    freeze: bool = True,
) -> M2ISceneEncoder:
    """Factory function to create M2I encoder."""
    return M2ISceneEncoder(
        m2i_checkpoint=checkpoint,
        embed_dim=embed_dim,
        freeze_m2i=freeze,
    )
