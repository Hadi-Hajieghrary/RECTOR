# RECTOR Scripts: Experimental Workflow and Figure Generation

This directory contains the executable scripts that support training, evaluation, visualization, ablation, and qualitative analysis for the RECTOR project. The structure is intentionally organized by research task rather than by implementation detail.

For paper preparation, this is the most useful place to identify which script generates which table, plot, or video.

## Directory structure

```text
scripts/
├── training/        model training entry points and training utilities
├── evaluation/      metric computation, canonical evaluation, confidence intervals
├── visualization/   static figures and video generation
├── inference/       lightweight inference demos and reusable inference pipeline
├── experiments/     focused studies, diagnostics, and one-off analyses
├── diagnostics/     debugging tools for behavior and trajectory quality
├── lib/             reusable components for the M2I+RECTOR pipeline
├── models/          neural model definitions
├── data/            dataset wrappers and preprocessing helpers used by training
├── proxies/         differentiable rule proxies and aggregation logic
├── tuning/          selection-baseline tuning scripts
└── utils/           shared utility code
```

## Recommended workflow for paper-ready evidence

### 1. Train or load RECTOR

Use the training module when reproducing or extending the learned model:

```bash
cd /workspace/models/RECTOR/scripts

# Option A: Pre-cache data for faster training (recommended, one-time)
python training/preprocess_cache.py --split train --workers 8
python training/preprocess_cache.py --split val --workers 8

# Train with cached data
python training/train_rector.py \
    --cache_dir /workspace/data/WOMD/datasets/waymo_open_dataset/motion_v_1_3_0/processed/augmented/cached \
    --experiment_name my_experiment

# Option B: Train directly from TFRecords (slower)
python training/train_rector.py --experiment_name my_experiment
```

### 2. Produce quantitative tables

For headline metrics, use the evaluation scripts in this order:

```bash
cd /workspace/models/RECTOR/scripts
python evaluation/evaluate_rector.py \
    --checkpoint /workspace/models/RECTOR/models/best.pt

python evaluation/evaluate_canonical.py \
    --checkpoint /workspace/models/RECTOR/models/best.pt

python evaluation/compute_bootstrap_cis.py \
    --input /workspace/output/evaluation/canonical_results.json
```

This sequence gives:

- standard open-loop trajectory metrics,
- canonical protocol outputs suitable for tables,
- confidence intervals for statistical reporting.

The canonical evaluator writes its JSON to `/workspace/output/evaluation/canonical_results.json` by default, which is why the bootstrap step consumes that path.

### 3. Produce qualitative figures

Use the visualization scripts to create figures for the main paper or supplement:

```bash
cd /workspace/models/RECTOR
python scripts/visualization/visualize_predictions.py \
    --checkpoint models/best.pt \
    --num_scenarios 10

python scripts/visualization/generate_receding_movies.py \
    --checkpoint models/best.pt \
    --num_scenarios 20
```

### 4. Produce planning-style demonstrations

For the explicit-rule selection pipeline:

```bash
cd /workspace/models/RECTOR
python scripts/visualization/generate_m2i_movies.py \
    --num_scenarios 40 \
    --predict_interval 20 \
    --min_ego_speed 3.0
```

## Key scripts and what they are for

### Training

| Script | Purpose |
|---|---|
| `training/train_rector.py` | Main training script — includes `WaymoDataset`, `CachedDataset`, `Trainer` |
| `training/preprocess_cache.py` | One-time TFRecord → `.pt` cache conversion for fast training |
| `training/losses.py` | 7-component loss (WTA, KL, goal, smoothness, heading, speed, applicability) |
| `training/callbacks.py` | Monitoring, validation, early stopping helpers |

### Evaluation

| Script | Purpose |
|---|---|
| `evaluation/evaluate_rector.py` | Standard open-loop validation metrics |
| `evaluation/evaluate_canonical.py` | Canonical evaluation for tables and protocol comparisons |
| `evaluation/compute_bootstrap_cis.py` | Confidence intervals and nonparametric statistical summaries |

### Visualization

| Script | Purpose |
|---|---|
| `visualization/visualize_predictions.py` | Static qualitative figures |
| `visualization/generate_movies.py` | Standard prediction videos |
| `visualization/generate_receding_movies.py` | RECTOR receding-horizon videos |
| `visualization/generate_m2i_movies.py` | M2I+RECTOR receding-horizon videos |
| `visualization/visualize_model_architecture.py` | Architecture illustrations and sanity checks |

`visualization/visualize_model_architecture.py` requires the Graphviz `dot` executable to render diagram outputs.

### Inference and baselines

| Script | Purpose |
|---|---|
| `inference/run_inference_demo.py` | Small-scale end-to-end inference example |
| `tuning/tune_weighted_sum.py` | Tuning of weighted-sum selection baselines |

### Research support scripts

| Folder | Typical use |
|---|---|
| `experiments/` | paper-specific analyses, ablations, or metric studies |
| `diagnostics/` | model debugging and trajectory-quality inspection |
| `lib/` | reusable components for the explicit-rule M2I+RECTOR pipeline |

## Which outputs are most useful for a journal paper

If the goal is a paper draft, the most valuable outputs from this directory are usually:

- canonical JSON results from `evaluation/evaluate_canonical.py`,
- confidence intervals from `evaluation/compute_bootstrap_cis.py`,
- qualitative figures from `visualization/visualize_predictions.py`,
- demonstration videos from `visualization/generate_receding_movies.py` and `visualization/generate_m2i_movies.py`.

## Guidance for extending the scripts

When adding new scripts, keep the structure task-oriented:

- new training entry points belong in `training/`,
- paper metric scripts belong in `evaluation/`,
- figure and movie generation belongs in `visualization/`,
- one-off studies belong in `experiments/` unless they become part of the standard workflow.

This helps the documentation remain interpretable to readers who are trying to reproduce results rather than inspect every implementation detail.

## Related documentation

- [../README.md](../README.md) — method-level overview for the RECTOR subproject.
- [../../README.md](../../README.md) — repository-level overview.
- [../../../data/README.md](../../../data/README.md) — data preparation and augmentation.
