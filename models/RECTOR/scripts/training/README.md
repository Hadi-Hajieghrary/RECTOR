# RECTOR Training Stack

This directory contains the optimization logic behind RECTOR. It covers loss definitions, callbacks, trainer infrastructure, and data caching for the learned model.

## Training philosophy

The training code is designed around three goals:

- learn accurate multi-modal trajectories,
- preserve behavioral structure through rule-related supervision (focal BCE loss in Stage 1, plain BCE in Stage 2),
- and keep optimization stable enough for long experiments on large motion datasets.

## Two-stage training strategy

RECTOR uses a two-stage training pipeline. This decomposition isolates rule-applicability learning from the harder joint optimization, giving the full model a better starting point.

### Stage 1: Applicability head pretraining (`train_applicability.py`)

The applicability head is trained in isolation first. This stage freezes the M2I encoder (223K params) and trains only the applicability head on top of it, learning to predict which of the 28 traffic rules apply in each scene.

- **Trainable params:** ~3.43M (of 3.65M total)
- **M2I encoder:** frozen --- used as a fixed feature extractor
- **Loss:** `RuleApplicabilityLoss(focal_gamma=2.0, pos_weight=pos_weight)` --- focal BCE with gamma=2.0 and per-rule `pos_weight` (computed from training data: neg/pos ratio clamped to [1.0, 20.0]), safety-tier rules upweighted 3x
- **Output:** a pretrained applicability checkpoint (`best.pt`) consumed by Stage 2

Pretraining the applicability head separately is important because rule-applicability labels are sparse and class-imbalanced (some rules apply in <2% of scenes). Training it jointly from scratch with the trajectory loss would drown out the applicability signal. Focal loss (gamma=2.0) further addresses this by down-weighting easy negatives.

### Stage 2: Full end-to-end training (`train_rector.py`)

The full RECTOR model is trained jointly: trajectory generation, rule applicability, and rule-based scoring all train together. This stage loads the pretrained applicability head from Stage 1 and unfreezes the entire model, including the M2I encoder.

- **Trainable params:** all 8.82M
- **M2I encoder:** unfrozen, fine-tuned at 0.1x the base learning rate
- **Applicability head:** initialized from Stage 1 checkpoint, continues training jointly
- **Loss:** 7-component composite (see "Loss components" below)

The 7 loss components in Stage 2 (`RECTORLoss` from `losses.py`) are:

| # | Component | Weight (default) | Description |
|---|-----------|-------------------|-------------|
| 1 | **WTA Reconstruction** | 20.0 | Winner-takes-all Huber loss (SmoothL1, beta=0.5) on best mode. Uses temporal weighting (linearly 1.0 to 2.0 over the trajectory) and dimension weighting [2.0, 2.0, 0.5, 0.5] for [x, y, heading, speed]. |
| 2 | **Endpoint / FDE** | 5.0 | Extra Huber loss on the final position (x, y) of the best mode. |
| 3 | **KL Divergence** | 0.05 | Regularizes the CVAE latent space. |
| 4 | **Goal** | 2.0 | Huber loss training the goal prediction head to match the GT trajectory endpoint. |
| 5 | **Smoothness** | 1.0 | Penalizes high jerk (third derivative of position) in the best mode. |
| 6 | **Confidence** | 1.0 | KL divergence between predicted mode confidence (log-softmax) and one-hot oracle target (best mode by ADE). |
| 7 | **Applicability BCE** | 1.0 | Plain `binary_cross_entropy_with_logits` on rule applicability logits (no focal weighting in Stage 2). |

Note: there are no separate "heading" or "speed" loss components. Heading and speed are dimensions within the WTA reconstruction loss (dimension indices 2 and 3 with weight 0.5 each), not standalone losses.

Additional `RECTORLoss` defaults: `trajectory_scale=50.0`.

### Why fine-tune the M2I encoder in Stage 2?

The M2I encoder was originally pretrained for generic motion prediction --- forecasting where vehicles will go. RECTOR needs scene representations that also support rule-compliant trajectory generation and rule-applicability prediction. Fine-tuning M2I in Stage 2 allows it to adapt for three reasons:

1. **Domain adaptation.** The encoder learns to emphasize scene features that matter for rule compliance (traffic signal states, lane boundaries, agent proximity) rather than just trajectory accuracy.
2. **Multi-task gradient signal.** The encoder receives gradients from trajectory loss, applicability BCE, and rule-violation scoring simultaneously. These combined signals teach richer, more task-relevant representations than any single objective alone.
3. **Element feature specialization.** The M2I encoder produces per-element features (one vector per agent and lane) that the applicability head consumes directly. Fine-tuning lets these element representations specialize --- e.g., agent features learn to capture collision-relevant dynamics, lane features learn to encode lane-violation geometry.

The 0.1x learning rate multiplier prevents catastrophic forgetting of pretrained knowledge while still allowing gradual adaptation. Keeping M2I frozen in Stage 2 would force downstream heads to compensate for generic, non-task-specific features.

## Main modules

| Module | Role |
|---|---|
| `train_rector.py` | Main training entry point (Stage 2) --- includes `WaymoDataset` (TFRecord), `CachedDataset` (.pt), `Trainer`, data collation |
| `train_applicability.py` | Applicability head pretraining (Stage 1) --- isolated training with frozen M2I encoder |
| `losses.py` | 7-component loss: (1) WTA Reconstruction, (2) Endpoint/FDE, (3) KL Divergence, (4) Goal, (5) Smoothness, (6) Confidence, (7) Applicability BCE |
| `preprocess_cache.py` | One-time TFRecord to `.pt` conversion for 3-5x faster epoch times |
| `callbacks.py` | Exports: `ProxyRefinementCallback`, `ApplicabilityHysteresis`, `EarlyStopping`, `TrainingMetrics` |

### Note on callbacks.py

`callbacks.py` defines `EarlyStopping` and `ProxyRefinementCallback`, but **neither is imported or used by the training scripts**. Both `train_rector.py` and `train_applicability.py` implement their own inline patience counter (`patience=15` by default) for early stopping. `ProxyRefinementCallback` is dead code --- it was designed to validate proxy costs against exact rule evaluation but is never invoked by any training script. The `ApplicabilityHysteresis` module is an inference-time utility, not a training callback.

## Key training features

- **Mixed precision** (AMP) with `GradScaler`
- **OneCycleLR** scheduler (pct_start=0.1) with **AdamW** optimizer
- **Gradient accumulation** (`--grad_accum_steps`) for effective larger batch sizes
- **torch.compile** support (`--compile`) for additional speedup
- **CachedDataset**: random-access `.pt` files for fast data loading with persistent workers
- **WaymoDataset**: streaming `IterableDataset` for TFRecord files (fallback when cache not built)

## Training workflow

```bash
# Step 0: (Optional) Pre-process TFRecords to .pt cache for faster loading
python training/preprocess_cache.py --split train --workers 8
python training/preprocess_cache.py --split val --workers 8

# Step 1: Pretrain the applicability head (Stage 1)
#   Defaults: --epochs 50 --batch_size 256 --learning_rate 0.0003
#   Below overrides are shown for illustration; you can omit them to use defaults.
python training/train_applicability.py \
    --epochs 50 --batch_size 256 --learning_rate 0.0003

# Step 2: Full RECTOR training (Stage 2), loading the Stage 1 checkpoint
#   Defaults: --epochs 50 --batch_size 256 --learning_rate 0.0003
python training/train_rector.py \
    --pretrained_applicability output/applicability_head_.../best.pt \
    --epochs 50 --batch_size 256
```

### CLI reference (key flags)

**Stage 1 (`train_applicability.py`):**

| Flag | Default | Notes |
|------|---------|-------|
| `--epochs` | 50 | |
| `--batch_size` | 256 | |
| `--learning_rate` | 3e-4 | |
| `--weight_decay` | 1e-4 | |
| `--grad_clip` | 1.0 | |
| `--patience` | 15 | Inline early stopping (not the `EarlyStopping` callback) |
| `--val_batches` | 80 | |
| `--freeze_m2i` | True | Use `--no_freeze_m2i` to unfreeze |

**Stage 2 (`train_rector.py`):**

| Flag | Default | Notes |
|------|---------|-------|
| `--pretrained_applicability` | None | Path to Stage 1 checkpoint |
| `--epochs` | 50 | |
| `--batch_size` | 256 | |
| `--learning_rate` | 3e-4 | |
| `--weight_decay` | 1e-4 | |
| `--grad_clip` | 1.0 | |
| `--patience` | 15 | Inline early stopping |
| `--val_batches` | 50 | |
| `--num_workers` | 8 | |
| `--grad_accum_steps` | 1 | Effective batch = batch_size x grad_accum_steps |
| `--freeze_m2i` | False | M2I fine-tuned at 0.1x LR by default |

## Data caching (`preprocess_cache.py`)

Each cached `.pt` file contains the following keys:

| Key | Shape | Dtype |
|-----|-------|-------|
| `ego_history` | [11, 4] | float32 |
| `agent_states` | [32, 11, 4] | float32 |
| `lane_centers` | [64, 20, 2] | float32 |
| `traj_gt` | [50, 4] | float32 |
| `rule_applicability` | [28] | float32 |
| `rule_violations` | [28] | float32 |
| `rule_severity` | [28] | float32 |

## Training metrics reference

| Metric | Meaning |
|---|---|
| **Train/Val Loss** | Composite optimization objective (not directly interpretable in physical units) |
| **ADE (Average Displacement Error)** | Mean distance (meters) between predicted and true positions, averaged across all future timesteps |
| **FDE (Final Displacement Error)** | Distance (meters) between predicted and true position at the final timestep (5 seconds out) |
| **F1 / Precision / Recall** | Applicability head classification metrics (Stage 1 and applicability component of Stage 2) |

## Related documentation

- [../models/README.md](../models/README.md) --- neural architecture.
- [../README.md](../README.md) --- end-to-end workflow.
