# RECTOR Trained Checkpoints

This directory stores the trained RECTOR model weights used for evaluation, visualization, and deployment. Each checkpoint captures the full model state — scene encoder, trajectory decoder, applicability head, and optimizer state — at the point of best validation performance.

---

## Available Checkpoints

| Checkpoint | Stage | Epochs | Parameters | Key Metric | Purpose |
|-----------|-------|--------|------------|------------|---------|
| `rector_full_20ep_best.pt` | Stage 2 (full) | 20 | 8.82M | minADE=0.684m | Primary evaluation checkpoint |
| `app_head_20ep_best.pt` | Stage 1 (applicability) | 20 | ~3.4M trainable | F1 on 28-rule applicability | Warm-start for Stage 2 |

### rector_full_20ep_best.pt — Production Checkpoint

This is the fully trained RECTOR model after the complete two-stage training pipeline:

**Stage 1** pre-trained the applicability head (28-rule classifier) on a frozen M2I encoder using focal BCE loss with per-rule positive-class weighting and safety-tier 3x multiplier.

**Stage 2** fine-tuned all 8.82M parameters jointly — the M2I scene encoder (at 0.1x learning rate to prevent catastrophic forgetting), the CVAE trajectory decoder (6 modes, 50 timesteps), and the applicability head — using a 7-component composite loss.

**Evaluation results (12,800 WOMD validation scenarios):**

| Metric | Value |
|--------|-------|
| Oracle minADE (best-of-6) | 0.684 m |
| Oracle minFDE (best-of-6) | 1.270 m |
| Miss Rate (FDE > 2.0m) | 18.56% |
| Total Violation Rate (lexicographic) | 15.03% |
| Safety+Legal Violation Rate | 13.15% |

### app_head_20ep_best.pt — Applicability Pre-training

Checkpoint from Stage 1 training. Contains only the scene encoder projections and applicability head weights. Used to initialize Stage 2 training with a warm-started applicability head.

---

## Model Architecture Summary

The checkpoint contains weights for these components:

```
RuleAwareGenerator (8.82M total)
├── scene_encoder (M2ISceneEncoder)
│   ├── M2I backbone (frozen → unfrozen at 0.1x LR in Stage 2)
│   │   ├── SubGraph (polyline encoding, 3 layers)
│   │   ├── GlobalGraph (self-attention over all elements)
│   │   └── CrossAttention (agent-to-lane, lane-to-lane)
│   └── Projection layers (128-dim → 256-dim)
├── applicability_head (RuleApplicabilityHead, ~1.5M)
│   ├── TierAwareBlock × 4 (one per tier)
│   │   ├── Rule self-attention (inter-rule dependencies)
│   │   ├── Scene cross-attention (rule → scene elements)
│   │   └── Per-rule classifier
│   └── Refinement MLP with gated residual
├── trajectory_head (CVAETrajectoryHead)
│   ├── GoalPredictionHead (6 goal queries)
│   ├── Prior/Posterior networks (latent dim=64)
│   ├── TransformerTrajectoryDecoder (4 layers, 8 heads)
│   ├── TrajectoryRefiner
│   └── Confidence head
└── rule_proxies (DifferentiableRuleProxies, frozen)
    ├── CollisionProxy, VRUClearanceProxy
    ├── SmoothnessProxy, LateralAccelerationProxy
    ├── LaneProxy, SpeedLimitProxy
    ├── SignalProxy
    └── InteractionProxy
```

---

## Loading a Checkpoint

```python
import torch
from models.RECTOR.scripts.models import RuleAwareGenerator

model = RuleAwareGenerator(
    embed_dim=256, num_heads=8, num_encoder_layers=4,
    decoder_hidden_dim=256, decoder_num_layers=4,
    trajectory_length=50, num_modes=6, latent_dim=64,
    num_rules=28, use_m2i_encoder=True
)
state = torch.load("checkpoints/rector_full_20ep_best.pt", map_location="cpu")
model.load_state_dict(state["model_state_dict"])
model.eval()
```

---

## Related Documentation

- [../output/README.md](../output/README.md) — Training run history and output directories
- [../scripts/training/README.md](../scripts/training/README.md) — Training pipeline documentation
- [../scripts/models/README.md](../scripts/models/README.md) — Neural architecture documentation
