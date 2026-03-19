"""
Rule Applicability Head.

Predicts which rules are applicable given the scene context.
Uses scene embedding to output logits for BCEWithLogitsLoss.

Key insight: applicability depends on static scene features
(road structure, signals, other agents) but NOT on the trajectory.

Architecture:
- V1: Simple MLP - too weak for 28 rules across 4 tiers
- Current: Tier-aware with rule-specific context attention
  - Separate reasoning pathways for Safety/Legal/Road/Comfort tiers
  - Rule-specific query vectors with cross-attention to scene
  - Captures rule dependencies (e.g., collision implies clearance violation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional, Tuple, List

import sys

sys.path.insert(0, "/workspace/data/WOMD")
sys.path.insert(0, "/workspace/data")
from waymo_rule_eval.rules.rule_constants import (
    NUM_RULES,
    RULE_IDS,
    RULE_INDEX_MAP,
    TIER_0_SAFETY,
    TIER_1_LEGAL,
    TIER_2_ROAD,
    TIER_3_COMFORT,
)


def _get_tier_indices() -> Dict[str, List[int]]:
    """Get rule indices for each tier."""
    return {
        "tier0_safety": [
            RULE_INDEX_MAP[r] for r in TIER_0_SAFETY if r in RULE_INDEX_MAP
        ],
        "tier1_legal": [RULE_INDEX_MAP[r] for r in TIER_1_LEGAL if r in RULE_INDEX_MAP],
        "tier2_road": [RULE_INDEX_MAP[r] for r in TIER_2_ROAD if r in RULE_INDEX_MAP],
        "tier3_comfort": [
            RULE_INDEX_MAP[r] for r in TIER_3_COMFORT if r in RULE_INDEX_MAP
        ],
    }


class TierAwareBlock(nn.Module):
    """
    Processing block for a tier of rules.

    Each tier has distinct patterns:
    - Safety: Depends on agent proximity, velocities
    - Legal: Depends on signals, signs, road markings
    - Road: Depends on lane geometry
    - Comfort: Depends on motion smoothness
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        num_rules_in_tier: int,
        num_heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.num_rules = num_rules_in_tier
        if num_rules_in_tier == 0:
            return

        # Learnable rule query vectors for this tier
        self.rule_queries = nn.Parameter(
            torch.randn(num_rules_in_tier, embed_dim) * 0.02
        )

        # Self-attention among rules (captures dependencies)
        self.rule_self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(embed_dim)

        # Cross-attention to scene context
        self.scene_cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(embed_dim)

        # FFN for each rule
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(embed_dim)

        # Output head per rule
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Initialize output with negative bias (most rules not applicable)
        nn.init.zeros_(self.classifier[-1].weight)
        nn.init.constant_(self.classifier[-1].bias, -1.0)

    def forward(
        self,
        scene_context: torch.Tensor,  # [B, D] pooled scene
        scene_elements: Optional[torch.Tensor] = None,  # [B, N, D] per-element
        element_mask: Optional[torch.Tensor] = None,  # [B, N]
    ) -> torch.Tensor:
        """
        Process rules for this tier.

        Returns:
            logits: [B, num_rules_in_tier]
        """
        if self.num_rules == 0:
            return torch.zeros(
                scene_context.shape[0],
                0,
                device=scene_context.device,
                dtype=scene_context.dtype,
            )

        B = scene_context.shape[0]
        device = scene_context.device

        # Expand rule queries: [num_rules, D] → [B, num_rules, D]
        queries = self.rule_queries.unsqueeze(0).expand(B, -1, -1)

        # 1) Self-attention among rules (captures dependencies like collision→clearance)
        q_self, _ = self.rule_self_attn(queries, queries, queries)
        queries = self.norm1(queries + q_self)

        # 2) Cross-attention to scene
        if scene_elements is not None:
            # Attend to per-element features
            q_cross, _ = self.scene_cross_attn(
                queries,
                scene_elements,
                scene_elements,
                key_padding_mask=element_mask,
            )
        else:
            # Use pooled scene context as single "element"
            scene_expanded = scene_context.unsqueeze(1)  # [B, 1, D]
            q_cross, _ = self.scene_cross_attn(queries, scene_expanded, scene_expanded)

        queries = self.norm2(queries + q_cross)

        # 3) FFN
        queries = self.norm3(queries + self.ffn(queries))

        # 4) Classify each rule
        logits = self.classifier(queries).squeeze(-1)  # [B, num_rules]

        return logits


class RuleApplicabilityHead(nn.Module):
    """
    Tier-Aware Rule Applicability Head.

    Architecture:
    - Scene context projection with multi-scale features
    - Separate processing blocks per tier (safety/legal/road/comfort)
    - Rule-specific learned queries with cross-attention
    - Inter-rule attention to capture dependencies

    Key features:
    - ~1.5M parameters (3x larger than simple MLP)
    - Tier-specific reasoning pathways
    - Rule dependency modeling via self-attention
    """

    def __init__(
        self,
        embed_dim: int = 256,
        hidden_dim: int = 384,  # Increased from embed_dim
        num_layers: int = 3,  # Depth of processing
        num_rules: int = NUM_RULES,
        num_heads: int = 8,
        dropout: float = 0.2,
    ):
        """
        Initialize tier-aware applicability head.

        Args:
            embed_dim: Input/internal embedding dimension
            hidden_dim: FFN hidden dimension (wider = more capacity)
            num_layers: Not directly used (kept for API compat)
            num_rules: Number of rules (default: 28)
            num_heads: Attention heads
            dropout: Dropout rate
        """
        super().__init__()

        self.num_rules = num_rules
        self.embed_dim = embed_dim

        # Scene context projection
        self.scene_proj = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        # Get tier indices
        tier_indices = _get_tier_indices()
        self.register_buffer(
            "tier0_indices",
            torch.tensor(tier_indices["tier0_safety"], dtype=torch.long),
        )
        self.register_buffer(
            "tier1_indices", torch.tensor(tier_indices["tier1_legal"], dtype=torch.long)
        )
        self.register_buffer(
            "tier2_indices", torch.tensor(tier_indices["tier2_road"], dtype=torch.long)
        )
        self.register_buffer(
            "tier3_indices",
            torch.tensor(tier_indices["tier3_comfort"], dtype=torch.long),
        )

        # Tier-specific processing blocks
        self.tier0_block = TierAwareBlock(
            embed_dim, hidden_dim, len(tier_indices["tier0_safety"]), num_heads, dropout
        )
        self.tier1_block = TierAwareBlock(
            embed_dim, hidden_dim, len(tier_indices["tier1_legal"]), num_heads, dropout
        )
        self.tier2_block = TierAwareBlock(
            embed_dim, hidden_dim, len(tier_indices["tier2_road"]), num_heads, dropout
        )
        self.tier3_block = TierAwareBlock(
            embed_dim,
            hidden_dim,
            len(tier_indices["tier3_comfort"]),
            num_heads,
            dropout,
        )

        # Final refinement MLP
        self.output_refine = nn.Sequential(
            nn.Linear(num_rules, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_rules),
        )

        # Residual weight (learnable gating)
        self.refine_gate = nn.Parameter(torch.zeros(1))

        self._init_weights()

        # Restore intentional negative bias on tier classifiers
        # (_init_weights zeros ALL biases, overwriting the -1.0 set in TierAwareBlock)
        for block in [
            self.tier0_block,
            self.tier1_block,
            self.tier2_block,
            self.tier3_block,
        ]:
            if block.num_rules > 0:
                nn.init.zeros_(block.classifier[-1].weight)
                nn.init.constant_(block.classifier[-1].bias, -1.0)

    def _init_weights(self):
        """Initialize weights for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        scene_embedding: torch.Tensor,
        scene_elements: Optional[torch.Tensor] = None,
        element_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict rule applicability with tier-aware processing.

        Args:
            scene_embedding: [B, D] pooled scene features
            scene_elements: [B, N, D] optional per-element embeddings
            element_mask: [B, N] optional mask for elements

        Returns:
            Tuple of:
            - logits: [B, R] raw logits for BCEWithLogitsLoss
            - prob: [B, R] applicability probabilities (sigmoid)
        """
        B = scene_embedding.shape[0]
        device = scene_embedding.device
        dtype = scene_embedding.dtype  # Match input dtype for AMP compatibility

        # Project scene context
        scene_ctx = self.scene_proj(scene_embedding)

        # Process each tier
        tier0_logits = self.tier0_block(scene_ctx, scene_elements, element_mask)
        tier1_logits = self.tier1_block(scene_ctx, scene_elements, element_mask)
        tier2_logits = self.tier2_block(scene_ctx, scene_elements, element_mask)
        tier3_logits = self.tier3_block(scene_ctx, scene_elements, element_mask)

        # Assemble full logits tensor (use same dtype as input for AMP)
        logits = torch.zeros(B, self.num_rules, device=device, dtype=dtype)

        if len(self.tier0_indices) > 0:
            logits.scatter_(
                1, self.tier0_indices.unsqueeze(0).expand(B, -1), tier0_logits
            )
        if len(self.tier1_indices) > 0:
            logits.scatter_(
                1, self.tier1_indices.unsqueeze(0).expand(B, -1), tier1_logits
            )
        if len(self.tier2_indices) > 0:
            logits.scatter_(
                1, self.tier2_indices.unsqueeze(0).expand(B, -1), tier2_logits
            )
        if len(self.tier3_indices) > 0:
            logits.scatter_(
                1, self.tier3_indices.unsqueeze(0).expand(B, -1), tier3_logits
            )

        # Refinement with gated residual
        refined = self.output_refine(logits)
        gate = torch.sigmoid(self.refine_gate)
        logits = logits + gate * refined

        prob = torch.sigmoid(logits)

        return logits, prob

    def predict_binary(
        self,
        scene_embedding: torch.Tensor,
        threshold: float = 0.5,
    ) -> torch.Tensor:
        """
        Predict binary applicability.

        Args:
            scene_embedding: [B, D]
            threshold: Probability threshold

        Returns:
            applicability: [B, R] binary tensor
        """
        _, prob = self.forward(scene_embedding)
        return (prob > threshold).float()


class RuleApplicabilityLoss(nn.Module):
    """
    Loss function for rule applicability prediction.

    Uses BCEWithLogitsLoss with optional class weighting.
    """

    def __init__(
        self,
        pos_weight: Optional[torch.Tensor] = None,
        focal_gamma: float = 0.0,
    ):
        """
        Initialize loss.

        Args:
            pos_weight: Weight for positive class per rule [R]
            focal_gamma: Focal loss gamma (0 = standard BCE)
        """
        super().__init__()

        self.register_buffer(
            "pos_weight", pos_weight if pos_weight is not None else None
        )
        self.focal_gamma = focal_gamma

        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute applicability loss.

        Args:
            logits: [B, R] predicted logits
            targets: [B, R] ground truth applicability (0/1)
            mask: [B, R] optional mask for valid rules

        Returns:
            Scalar loss
        """
        # BCE loss
        loss = self.bce(logits, targets)  # [B, R]

        # Apply positive class weight
        if self.pos_weight is not None:
            pos_weight = self.pos_weight.to(logits.device)
            # Weight positive samples more
            weight = 1.0 + targets * (pos_weight - 1.0)
            loss = loss * weight

        # Focal loss modulation
        if self.focal_gamma > 0:
            prob = torch.sigmoid(logits)
            # p_t = p if y=1 else 1-p
            p_t = targets * prob + (1 - targets) * (1 - prob)
            focal_weight = (1 - p_t) ** self.focal_gamma
            loss = loss * focal_weight

        # Apply mask
        if mask is not None:
            loss = loss * mask
            return loss.sum() / (mask.sum() + 1e-8)

        return loss.mean()
