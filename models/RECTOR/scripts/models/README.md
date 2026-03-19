# RECTOR Neural Architecture

This directory contains the neural-network modules that compose the RECTOR model. The architecture is designed around a core insight: trajectory prediction and rule-compliance reasoning share the same scene understanding, so they should share the same encoder but branch into specialized heads for their distinct tasks.

---

## Conceptual Breakdown

The model stack is organized around four ideas:

1. **Scene understanding** — Encode agents and road context into a unified representation
2. **Rule awareness** — Predict which of 28 rules are active in the current scenario
3. **Trajectory generation** — Propose multiple diverse future hypotheses
4. **Candidate ranking** — Compare modes under structured rule priorities

---

## Architecture Diagram

```
Raw Scene (agent histories, lane polylines, map context)
        |
        v
M2I Scene Encoder (SubGraph + GlobalGraph + LaneGCN)
  - SubGraph: 3-layer MLP per polyline, masked self-attention, max-pool
  - GlobalGraph: Multi-head self-attention over all elements (ego, agents, lanes)
  - LaneGCN refinement: agent->lane, lane->lane, lane->agent cross-attention
  - Projection: 128-dim M2I backbone -> 256-dim RECTOR embedding space
        |
        +---> scene_embedding [B, 256]           Pooled ego-centric scene context
        +---> element_features [B, 1+A+L, 256]   Per-element vectors (ego + agents + lanes)
        +---> element_pad_mask [B, 1+A+L]
                |                    |
                v                    v
    Trajectory Head           Applicability Head
    (CVAE Decoder)            (Tier-Aware Attention, 28 rules)
    |                         |
    +- GoalPredictionHead     +- 4 TierAwareBlocks (Safety, Legal, Road, Comfort)
    |  (6 goal queries ->     |  +- Rule self-attention (inter-rule dependencies)
    |   cross-attn to scene)  |  +- Scene cross-attention (rule->scene elements)
    +- Prior/Posterior nets   |  +- Per-rule classifier
    |  (latent dim=64)        +- Refinement MLP with gated residual
    +- TransformerDecoder     |
    |  (4 layers, 8 heads,    |
    |   parallel decoding)    |
    +- TrajectoryRefiner      |
    +- Confidence head        |
    |                         |
    v                         v
 trajectories [B,6,50,4]   applicability [B,28]
 confidence [B,6]            |
    |                         |
    +--------+----------------+
             v
      Tiered Rule Scorer
      +- Apply applicability mask
      +- Aggregate violations per tier
      +- Lexicographic selection (Safety > Legal > Road > Comfort)
             |
             v
      Best rule-compliant trajectory [B,50,4]
```

**Note:** `RuleAwareGenerator` also instantiates `DifferentiableRuleProxies` (`self.rule_proxies`), but this component is not called during the model's own `forward()` pass. It is used externally by evaluation scripts to compute proxy-based rule violations on the generated trajectories.

---

## Modules in Detail

### rule_aware_generator.py — Top-Level RECTOR Model

The `RuleAwareGenerator` class is the unified model that composes all sub-modules. It manages:
- **Scene encoder selection**: M2ISceneEncoder (pretrained, recommended) or native SceneEncoder
- **Parameter groups with differential learning rates**: M2I backbone fine-tuned at 0.1x base LR to prevent catastrophic forgetting
- **Forward pass orchestration**: scene encoding -> applicability prediction + trajectory generation -> scoring -> selection

**Key parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `embed_dim` | 256 | Scene embedding dimension |
| `num_heads` | 8 | Transformer attention heads |
| `num_encoder_layers` | 4 | SceneEncoder transformer layers (overrides SceneEncoder's default of 3) |
| `trajectory_length` | 50 | Prediction horizon (5s at 10Hz) |
| `num_modes` | 6 | Number of trajectory candidates |
| `latent_dim` | 64 | CVAE latent dimension |
| `decoder_hidden_dim` | 256 | Transformer trajectory decoder hidden dimension |
| `decoder_num_layers` | 4 | Transformer trajectory decoder layers |
| `num_rules` | 28 | Total traffic rules |
| `use_m2i_encoder` | True | Use pretrained M2I backbone |
| `freeze_m2i` | False | Allow M2I fine-tuning in Stage 2 |
| `dropout` | 0.1 | Dropout rate (passed to all sub-modules) |

**Key methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `encode_scene(...)` | scene_embedding, elements, mask | Encode raw scene features |
| `predict_applicability(...)` | logits [B,R], probs [B,R] | Predict 28-rule applicability |
| `generate_trajectories(...)` | dict (trajectories, confidence, KL, goals) | Generate K=6 trajectory modes |
| `forward(...)` | dict (all outputs combined) | Full training forward pass |
| `inference(...)` | dict (trajectories, confidence, applicability) | Inference-time forward pass |
| `get_parameter_groups(base_lr)` | list of param group dicts | Encoder at 0.1x LR, rest at 1x |

**Derived internal settings:** The applicability head is created with `hidden_dim = int(embed_dim * 1.5)` = 384.

### m2i_encoder.py — Pretrained Scene Encoder Wrapper

The `M2ISceneEncoder` wraps Google's Motion-to-Interaction (M2I) pretrained model as a feature extractor. This is the recommended encoder for RECTOR because it provides a scene representation that has already learned to capture interaction dynamics from large-scale trajectory data.

**Internal component classes** (all defined in this file):

| Class | Purpose |
|-------|---------|
| `MLP` | Linear + LayerNorm + ReLU building block |
| `SubGraph` | Polyline encoder: 3-layer vectorized MLP + max-pool (hidden_size=128, depth=3) |
| `GlobalGraph` | Multi-head self-attention with Q/K/V projections |
| `GlobalGraphRes` | Residual pair of GlobalGraphs with concatenated outputs (attention_head_size = hidden/2 each) |
| `CrossAttention` | Cross-attention between query and key states |

**Architecture (from M2I):**
- **SubGraph** (polyline encoder): 3-layer MLP per vector + max-pool over neighboring vectors (self-exclusion masking). Encodes variable-length polylines (ego trajectory, agent trajectories, lane centerlines) into fixed 128-dim vectors.
- **GlobalGraph**: Multi-head self-attention over all polyline embeddings. Captures cross-element interactions (agent-agent, agent-lane, lane-lane).
- **LaneGCN refinement** (optional, controlled by `use_lanegcn=True`): Agent-to-lane, lane-to-lane, lane-to-agent cross-attention for spatially-structured reasoning.
- **Projection**: Linear layers from M2I's 128-dim to RECTOR's 256-dim embedding space (`ego_input_proj`, `agent_input_proj`, `lane_input_proj`, `output_proj`).

**Freezing behavior:** When `freeze_m2i=True` (the default for M2ISceneEncoder), the `_freeze_m2i_components()` method freezes parameters whose names contain `sub_graph`, `global_graph`, or `laneGCN`. The projection layers (`*_input_proj`, `output_proj`) are **always trainable** regardless of the freeze setting.

**Factory function:** `create_m2i_encoder(checkpoint, embed_dim=256, freeze=True)` provides a convenience constructor.

**Input format:**

| Input | Shape | Description |
|-------|-------|-------------|
| `ego_history` | [B, 11, 4] | (x, y, heading, speed) normalized by /50 |
| `agent_states` | [B, 32, 11, 4] | Nearest 32 agents, same format |
| `lane_centers` | [B, 64, 20, 2] | 64 nearest lanes, 20 points each |
| `agent_mask` | [B, 32] | Binary validity |
| `lane_mask` | [B, 64] | Binary validity |

**Output:** Tuple of three tensors:
- `scene_embedding` [B, 256]: Ego-centric pooled scene context
- `element_features` [B, 1+A+L, 256]: Per-element vectors for cross-attention in downstream heads
- `element_pad_mask` [B, 1+A+L]: Validity mask for attention masking

### scene_encoder.py — Native Transformer Encoder (Alternative)

The `SceneEncoder` is an alternative to M2ISceneEncoder when pretrained M2I weights are unavailable. It uses a standard Transformer architecture with:
- **PolylineEncoder** (`input_dim`, `hidden_dim=64`, `output_dim=128`, `num_layers=3`): MLP per point + max-pool for polyline embedding
- **MultiHeadAttention**: Wrapper around `nn.MultiheadAttention` with residual + LayerNorm
- **FeedForward**: Two-layer MLP with residual + LayerNorm
- **TransformerEncoderLayer** (`embed_dim`, `num_heads=8`, `ff_dim=256`, `dropout=0.1`): Self-attention + FFN block
- **Type embeddings**: Learnable per-type tokens (4 types: ego, vehicle, pedestrian, cyclist)
- **Positional encoding**: **Learned** positional embeddings (`nn.Parameter`), not sinusoidal
- **Ego query**: Learnable query token cross-attends to scene elements for final ego-centric embedding

**Key difference from M2ISceneEncoder:** `SceneEncoder.forward()` returns a single `[B, embed_dim]` tensor. The separate `get_all_embeddings()` method returns both the scene embedding and the per-element features.

### applicability_head.py — Rule Applicability Prediction

The `RuleApplicabilityHead` predicts which of 28 traffic rules are relevant to the current scenario. This is a **scene-level** prediction (independent of trajectory content) — it determines the context for subsequent rule evaluation.

**Core architectural innovation — TierAwareBlock** (`embed_dim`, `hidden_dim`, `num_rules_in_tier`, `num_heads=4`, `dropout=0.2`):
- Each priority tier (Safety, Legal, Road, Comfort) has its own processing block
- **Learnable rule queries** [num_rules_in_tier, embed_dim]: Each rule has a learned embedding that acts as an attention query
- **Rule self-attention**: Captures inter-rule dependencies within a tier
- **Scene cross-attention**: Each rule query attends to scene elements, learning which agents, lanes, and signals are relevant to that specific rule
- **Per-rule classifier**: Final linear layer with -1.0 bias initialization (biases toward "not applicable" at init)

**Note on num_heads:** `TierAwareBlock` defaults to `num_heads=4`, but `RuleApplicabilityHead` passes its own `num_heads` parameter (default 8), so in practice the blocks use 8 attention heads.

**Why tier-aware?** Different tiers require different reasoning patterns. Safety rules (collision, clearance) primarily attend to nearby agents. Legal rules (signals, speed limits) attend to map infrastructure. Comfort rules (smoothness, following distance) attend to both. Separate blocks allow specialized attention patterns.

**Key methods:**
- `forward(scene_embedding, scene_elements=None, element_mask=None)` -> `(logits [B,R], prob [B,R])`
- `predict_binary(scene_embedding, threshold=0.5)` -> `[B,R]` — convenience method (only passes `scene_embedding`; does not use element-level cross-attention)

**Parameter count:** ~1.5M (compact relative to the full model)

**Loss:** `RuleApplicabilityLoss(pos_weight=None, focal_gamma=0.0)` — BCE with **optional** focal loss modulation and optional per-rule positive-class weighting. Both features are disabled by default (`focal_gamma=0.0`, `pos_weight=None`). The safety-tier 3x upweighting and focal gamma=2.0 are applied externally by the `train_applicability.py` training script, not built into the loss class itself.

### cvae_head.py — Trajectory Generation

The `CVAETrajectoryHead` generates K=6 diverse ego trajectory candidates using a Conditional Variational Autoencoder with Transformer decoding.

**Internal component classes:**

| Class | Constructor | Purpose |
|-------|------------|---------|
| `PositionalEncoding` | `(d_model, max_len=100, dropout=0.1)` | Sinusoidal positional encoding for trajectory decoder |
| `TrajectoryEncoder` | `(trajectory_length=50, state_dim=4, hidden_dim=128, output_dim=128)` | 1D convolutions + AdaptiveAvgPool1d for posterior encoding |
| `GaussianParams` | `(input_dim, latent_dim)` | Produces mean + logvar (clamped to [-10, 2]) |
| `GoalPredictionHead` | `(scene_dim=256, num_goals=6, goal_dim=64)` | Predicts goal positions, features, and confidence |
| `TransformerTrajectoryDecoder` | `(scene_dim=256, latent_dim=64, goal_dim=64, hidden_dim=256, trajectory_length=50, output_dim=4, num_layers=4, num_heads=8, dropout=0.1)` | Parallel Transformer decoder |
| `TrajectoryRefiner` | `(trajectory_length=50, state_dim=4, hidden_dim=256)` | Learned residual correction (`residual_scale=0.1`) |

**Pipeline:**

1. **Goal prediction**: `GoalPredictionHead` predicts K likely trajectory endpoints using learnable goal queries with cross-attention to the scene. Returns goal_positions [B,K,2], goal_features [B,K,64], goal_confidence [B,K].

2. **Latent sampling**:
   - **Training**: Posterior encoder `q(z|scene, traj_gt)` via `TrajectoryEncoder` -> sample z
   - **Inference**: Prior encoder `p(z|scene)` samples z from the scene alone
   - KL divergence regularizes the latent space
   - `reparameterize(mu, logvar)` always samples (never returns the mean directly)

3. **Trajectory decoding**: `TransformerTrajectoryDecoder` — learnable trajectory queries [T, hidden_dim] decoded in parallel (not autoregressive). Memory tokens combine scene embedding, latent z, and goal features. Output scaling: `output_scale = Parameter([0.15, 0.15, 0.04, 0.10])` for [x, y, heading, speed]. Positions computed as **cumulative sum of deltas** for absolute positions.

4. **Refinement**: `TrajectoryRefiner` applies a learned residual correction (scaled by 0.1) for global smoothness.

5. **Confidence**: A linear head [scene_dim + latent_dim -> hidden_dim -> 1] per mode, normalized with softmax. The confidence loss (KL divergence against oracle best-mode indicator) is computed externally in the training loss module, not within CVAETrajectoryHead.

**Standalone function:** `compute_minADE_minFDE(pred_trajectories, gt_trajectory, scale=50.0)` — utility for computing oracle metrics.

**Key design choice — Parallel decoding:** Unlike RNN/GRU decoders that generate positions sequentially (accumulating errors), the Transformer decoder generates all 50 timesteps simultaneously. This eliminates sequential error propagation and enables GPU-efficient computation.

**Key design choice — Winner-takes-all (WTA):** Only the best mode (lowest ADE to ground truth) receives the reconstruction loss gradient. This focuses each mode on a distinct region of the future distribution rather than averaging.

### tiered_scorer.py — Rule Compliance Scoring

The `TieredRuleScorer` implements the lexicographic scoring that is RECTOR's core theoretical contribution.

**Important:** In its default configuration (`use_learned_weights=False`), `TieredRuleScorer` has **zero learnable parameters**. The tier weights are registered as buffers (non-learnable). This is by design — the scoring logic is purely algorithmic, not learned.

**Scoring modes:**
- **Lexicographic** (proposed): Compares candidates tier-by-tier with epsilon tolerance (default 0.01). Safety violations eliminate candidates before legal violations are even considered. The mathematical guarantee: no finite combination of lower-tier improvements can compensate for a higher-tier violation.
- **Weighted sum** (baseline): Soft aggregation with tier weights (default: safety=1000, legal=100, road=10, comfort=1). Despite large weight ratios, finite weights always permit implicit tier tradeoffs.

**Training-time differentiable approximations:**
- `SoftLexicographicLoss`: Per-tier softmin with temperature scaling. Safety tier gets the coldest temperature (0.01 = strictest), comfort gets the warmest (10.0 = most lenient).
- `DifferentiableTieredSelection` (`temperature=0.1`, `tier_scale=10.0`): Soft attention-based trajectory selection using softmax over lexicographic scores. Hard selection at inference, soft selection at training for gradient flow.

---

## Parameter Summary

| Component | Approx. Parameters | Stage 1 | Stage 2 |
|-----------|-------------------|---------|---------|
| M2I backbone (SubGraph, GlobalGraph, LaneGCN) | ~5M | Frozen | 0.1x LR |
| Projection layers (ego/agent/lane/output_proj) | ~0.5M | **Trainable** (not frozen) | Full LR |
| Applicability head (4 TierAwareBlocks + refine) | ~1.5M | Full LR | Full LR |
| CVAE trajectory head (goal, decoder, refiner, confidence) | ~2M+ | — | Full LR |
| Tiered scorer | 0 (buffers only) | — | N/A (no learnable params) |
| Rule proxies (DifferentiableRuleProxies) | ~0.2M | — | Frozen |
| **Total** | **~8.82M** | **~2M** | **~8.82M** |

**Notes:**
- Projection layers are always trainable, even when M2I backbone is frozen. They are not part of the M2I freezing logic (which matches on `sub_graph`, `global_graph`, `laneGCN` name patterns).
- The tiered scorer contributes zero learnable parameters in the default configuration.
- The CVAE head is larger than 1.5M because the TransformerTrajectoryDecoder alone (4 layers, 256-dim, 8 heads) accounts for significant parameter mass.
- The 8.82M total is an empirical measurement from the trained checkpoint.

---

## Exported Symbols

The `__init__.py` exports the following public API:

| Symbol | Source |
|--------|--------|
| `SceneEncoder`, `PolylineEncoder`, `TransformerEncoderLayer` | `scene_encoder.py` |
| `RuleApplicabilityHead`, `RuleApplicabilityLoss` | `applicability_head.py` |
| `CVAETrajectoryHead`, `TrajectoryEncoder` | `cvae_head.py` |
| `TieredRuleScorer`, `SoftLexicographicLoss`, `DifferentiableTieredSelection` | `tiered_scorer.py` |
| `RuleAwareGenerator` | `rule_aware_generator.py` |

Classes like `M2ISceneEncoder`, `GoalPredictionHead`, `TransformerTrajectoryDecoder`, `TrajectoryRefiner`, `GaussianParams`, and `PositionalEncoding` are not exported directly but are accessible via their respective modules.

---

## Related Documentation

- [../training/README.md](../training/README.md) — Two-stage training strategy and loss components
- [../inference/README.md](../inference/README.md) — How the trained model is used at test time
- [../proxies/README.md](../proxies/README.md) — Differentiable rule proxies used during training
- [../../README.md](../../README.md) — RECTOR scientific overview
