# RECTOR: Scientific Overview of the Main Research Pipeline

This directory contains the main research implementation of RECTOR. It includes the learned `RECTOR` model, the `M2I+RECTOR` rule-guided trajectory-selection pipeline, experimental scripts, qualitative visualization tools, and generated artifacts used for analysis.

The purpose of this document is to describe the method in a form that is useful for paper writing: what is being modeled, how the system is organized, what evidence currently exists in the repository, and which scripts reproduce that evidence.

## Research objective

RECTOR is built around a simple question: can traffic-rule reasoning be made explicit in motion prediction and trajectory selection, rather than being left implicit inside a purely geometric forecasting objective?

The repository explores two complementary answers:

- **RECTOR** learns a rule-aware trajectory predictor from rule-augmented WOMD.
- **M2I+RECTOR** takes a pretrained generator and adds explicit rule-based mode selection in a receding-horizon loop.

Together, these two tracks support both quantitative forecasting experiments and qualitative planning-oriented analysis.

## The two active pipelines

### Track A: RECTOR

`RECTOR` is the learned model. It predicts multiple future ego trajectories over a 5-second horizon and also estimates which traffic rules are applicable in the scene.

Methodologically, the model combines:

- an M2I-derived scene encoder (fine-tuned end-to-end at 0.1x LR for domain adaptation to rule-compliance tasks),
- a Transformer-based CVAE decoder (`CVAETrajectoryHead`),
- multi-modal trajectory generation (6 modes, 50 timesteps),
- goal-conditioned prediction, and
- a rule-applicability head (`ApplicabilityHead`) that exposes behavior structure beyond displacement error alone,
- supervised with a 7-component loss (WTA reconstruction, endpoint, KL divergence, goal, smoothness, confidence, applicability BCE).

Training follows a two-stage strategy:

1. **Stage 1 — Applicability head pretraining:** The applicability head is trained in isolation on top of a frozen M2I encoder, learning to predict which of 28 traffic rules apply per scene. This stage uses focal BCE loss (gamma=2.0) with per-rule positive-class weighting and safety-tier 3x multiplier to handle severe class imbalance. This produces a warm-started checkpoint for Stage 2.
2. **Stage 2 — Full end-to-end training:** All 8.82M parameters train jointly — trajectory generation, rule applicability, and rule scoring. The M2I encoder is unfrozen and fine-tuned at 0.1x the base learning rate, allowing its representations to adapt for rule compliance without catastrophic forgetting of pretrained motion features.

See [scripts/training/README.md](scripts/training/README.md) for detailed rationale on the two-stage approach and M2I fine-tuning.

Training supports:
- **Pre-cached data** via `preprocess_cache.py` for 3-5× epoch speedup,
- **Mixed precision** (AMP) with gradient scaling,
- **Gradient accumulation** for effective larger batch sizes,
- **torch.compile** for additional speedup.

This is the pipeline to use for open-loop quantitative reporting.

### Track B: M2I+RECTOR

`M2I+RECTOR` is the analysis and visualization pipeline. Rather than training a new predictor, it starts from pretrained DenseTNT trajectories and re-ranks them with a 28-rule evaluator. The resulting system behaves like a lightweight planning loop:

1. encode the current scene,
2. generate multiple candidate futures,
3. score them under explicit rules,
4. rank candidates lexicographically across safety, legal, road, and comfort criteria,
5. select the best candidate,
6. repeat from the next replanning instant.

This track is especially valuable for behavior interpretation and for paper figures that need to show *why* one mode is preferred over another.

## Method summary

### Inputs

Both tracks operate on WOMD-derived scene information: agent histories, map context, and scenario metadata. In the learned setting, training inputs come from the augmented dataset described in [data/README.md](../../data/README.md).

### Rule representation

The repository uses a canonical set of 28 rules grouped into four high-level tiers:

- **Safety**
- **Legal**
- **Road**
- **Comfort**

The key idea is not only to ask whether a future is accurate, but also whether it is behaviorally acceptable under this structured hierarchy.

### Selection policy

When explicit rule scores are available, candidate trajectories are compared lexicographically by tier. This means that safety violations dominate all downstream considerations, followed by legal violations, then road consistency, then comfort. That ordering is central to the interpretation of the qualitative results.

## Current repository-backed results

The following results are already reflected in the checked-in scripts and generated outputs.

### RECTOR

Reported validation performance in the repository is approximately:

| Metric | Value |
|---|---:|
| minADE (best-of-6) | 0.684 m |
| minFDE (best-of-6) | 1.270 m |
| Miss rate (FDE > 2.0m) | 18.56% |
| Total violation rate (lexicographic) | 15.03% |
| Safety+Legal violation rate | 13.15% |
| Violation reduction vs confidence baseline | 37% relative |

These numbers come from `canonical_results.json` (12,800 WOMD validation scenarios). The recommended workflow is still to regenerate them from the evaluation scripts before final submission.

### M2I+RECTOR

The repository contains a set of receding-horizon demonstration videos showing rule-guided selection over time. These are especially useful for:

- supplementary material,
- qualitative failure/success analysis,
- figure panels illustrating the effect of the rule hierarchy.

## Which scripts matter most

### Quantitative evaluation

- `scripts/evaluation/evaluate_rector.py` — standard open-loop metrics.
- `scripts/evaluation/evaluate_canonical.py` — recommended source for paper tables.
- `scripts/evaluation/compute_bootstrap_cis.py` — uncertainty estimates and confidence intervals.

### Qualitative results

- `scripts/visualization/generate_receding_movies.py` — learned-model videos.
- `scripts/visualization/generate_m2i_movies.py` — pretrained M2I + explicit-rule videos.
- `scripts/visualization/visualize_predictions.py` — static visual figures.

### Training and model development

- `scripts/training/train_applicability.py` — Stage 1: applicability head pretraining (frozen M2I encoder, BCE loss with class-imbalance weighting).
- `scripts/training/train_rector.py` — Stage 2: full end-to-end training (all parameters trainable, M2I fine-tuned at 0.1x LR).
- `scripts/training/preprocess_cache.py` — one-time TFRecord → `.pt` cache conversion.
- `scripts/training/losses.py` — 7-component loss (WTA reconstruction, endpoint, KL divergence, goal, smoothness, confidence, applicability BCE).
- `scripts/inference/run_inference_demo.py` — small-scale sanity checks and example inference.

## Reproducibility workflow

### 1. Evaluate the learned model

```bash
cd /workspace/models/RECTOR
python scripts/evaluation/evaluate_rector.py \
    --checkpoint models/best.pt
```

### 2. Produce canonical paper metrics

```bash
cd /workspace/models/RECTOR/scripts
python evaluation/evaluate_canonical.py \
    --checkpoint /workspace/models/RECTOR/models/best.pt

python evaluation/compute_bootstrap_cis.py \
    --input /workspace/output/evaluation/canonical_results.json
```

### 3. Generate qualitative figures or videos

```bash
cd /workspace/models/RECTOR
python scripts/visualization/generate_receding_movies.py \
    --checkpoint models/best.pt \
    --num_scenarios 20

python scripts/visualization/generate_m2i_movies.py \
    --num_scenarios 40 \
    --predict_interval 20 \
    --min_ego_speed 3.0
```

## Directory guide

```text
models/RECTOR/
├── models/        trained checkpoint weights used by the documented examples
├── checkpoints/   archived checkpoints (Stage 1 + Stage 2)
├── scripts/       code used for training, evaluation, inference, and visualization
├── output/        derived experiment outputs and curated artifact bundles
├── movies/        qualitative videos from RECTOR and M2I+RECTOR
├── docs/          design notes and method-development documents
├── tests/         project-local tests
└── README.md      this document
```

## How this directory should be used in a paper

- Use this document to explain the overall method and the role of each pipeline.
- Use `scripts/evaluation/` for quantitative claims.
- Use `scripts/visualization/` and `movies/` for qualitative figures.
- Use `docs/` only when a lower-level design rationale is needed.

## Related documentation

- [../../README.md](../../README.md) — repository-level overview.
- [../README.md](../README.md) — model inventory across the workspace.
- [scripts/README.md](scripts/README.md) — script-level experimental workflow.
- [../../data/README.md](../../data/README.md) — data preparation and rule augmentation.
