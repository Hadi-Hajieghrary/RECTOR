# RECTOR Training Outputs

This directory contains all outputs from RECTOR's two-stage training pipeline: checkpoint weights, training metrics, and generated publication artifacts. Each subdirectory corresponds to a specific training run or artifact generation pass.

---

## Training Run History

RECTOR follows a two-stage training strategy. Stage 1 pre-trains the applicability head on a frozen scene encoder; Stage 2 trains the full model end-to-end with all parameters unfrozen. The runs below trace this progression.

### Stage 1: Applicability Head Pretraining

| Directory | Description | Epochs | Best Metric |
|-----------|-------------|--------|-------------|
| `app_head_fresh_20260316_094256/` | Applicability head trained from scratch on frozen M2I encoder | 20 | F1 on 28-rule applicability |

**What it learned:** Which of 28 traffic rules apply to a given driving scenario — a binary classification per rule. The head uses tier-aware attention blocks (one per priority tier) with cross-attention to scene elements, trained with focal BCE loss and per-rule positive-class weighting to handle severe class imbalance.

**Output:** `best.pt` — checkpoint with trained applicability head weights, used to warm-start Stage 2.

### Stage 2: Full End-to-End Training

| Directory | Description | Epochs | Key Metrics |
|-----------|-------------|--------|-------------|
| `rector_full_20260316_123249/` | Full RECTOR from Stage 1 warm start | 20 | minADE=0.684m, minFDE=1.270m |
| `rector_continue_5ep_20260317_014722/` | Continuation from rector_full | 5 | Refinement run |
| `rector_continue_5ep_20260317_015223/` | Final continuation | 5 | **Best checkpoint** |

**What it learned:** Joint trajectory prediction (6 modes × 50 timesteps) + rule applicability + goal prediction. The M2I scene encoder is unfrozen and fine-tuned at 0.1x the base learning rate to adapt its representations for rule compliance without catastrophic forgetting. Training uses a 7-component loss: winner-takes-all reconstruction (weight=20), endpoint (5), KL divergence (0.05), goal (2), smoothness (1), confidence (1), and applicability BCE (1).

**Output:** `best.pt` — the production checkpoint used for all evaluation and visualization.

---

## Current Best Checkpoint

```
rector_continue_5ep_20260317_015223/best.pt
```

This checkpoint is copied to `../models/best.pt` for convenient access by evaluation and visualization scripts. It represents the fully trained RECTOR model with:
- 8.82M total parameters
- M2I-derived scene encoder (fine-tuned)
- CVAE trajectory decoder (6 modes, Transformer-based)
- Rule applicability head (28 rules, 4 tiers)

---

## Artifacts

The `artifacts/` subdirectory contains all publication-ready figures and LaTeX tables generated from evaluation results. See [artifacts/README.md](artifacts/README.md) for a complete inventory.

---

## Directory Structure

```text
output/
├── app_head_fresh_20260316_094256/     Stage 1 training outputs
│   ├── best.pt                        Best applicability head checkpoint
│   ├── epoch_5.pt ... epoch_20.pt     Periodic checkpoints
│   └── config.json                    Training configuration
├── rector_full_20260316_123249/        Stage 2 initial training
│   └── best.pt                        Best full model checkpoint
├── rector_continue_5ep_20260317_014722/  Continuation run 1
│   └── best.pt                        Refined checkpoint
├── rector_continue_5ep_20260317_015223/  Continuation run 2 (final)
│   └── best.pt                        **Best checkpoint** (copied to ../models/best.pt)
└── artifacts/                          Publication-ready outputs
    ├── figures/                        24 PDF + 24 PNG figure files
    ├── tables/                         10 LaTeX table files
    ├── summary.json                    Artifact generation log
    └── README.md                       Artifact inventory
```

---

## Reproducing Training

```bash
cd /workspace/models/RECTOR

# Stage 1: Applicability head pretraining (20 epochs, ~1 hour)
python scripts/training/train_applicability.py \
    --epochs 20 --batch_size 256

# Stage 2: Full end-to-end training (20 epochs, ~3 hours)
python scripts/training/train_rector.py \
    --epochs 20 --batch_size 256 \
    --pretrained_applicability output/app_head_fresh_*/best.pt
```

---

## Related Documentation

- [../README.md](../README.md) — RECTOR scientific overview
- [../scripts/training/README.md](../scripts/training/README.md) — Training pipeline details
- [artifacts/README.md](artifacts/README.md) — Publication artifact inventory
