# Models Directory Guide

This directory contains the trained and pretrained model assets used throughout the RECTOR workspace. The goal of this document is not to enumerate every implementation detail, but to explain which model family each subdirectory represents, what role it plays in the research pipeline, and which artifacts are most relevant for manuscript preparation.

## What lives here

The `models/` directory contains three distinct kinds of assets:

1. **RECTOR** — the learned rule-aware trajectory predictor trained on rule-augmented WOMD.
2. **M2I+RECTOR** — the rule-guided trajectory selection pipeline built on top of pretrained M2I trajectories.
3. **Pretrained M2I checkpoints** — external trajectory-prediction backbones used as fixed or lightly adapted components.

## At a glance

| Asset | Role in the project | Typical use in a paper | Main location |
|---|---|---|---|
| RECTOR | Learned multi-modal ego forecasting model | Quantitative open-loop results | [models/RECTOR](RECTOR) |
| M2I+RECTOR | Receding-horizon rule-guided selection pipeline | Qualitative behavior studies and rule-based comparisons | [models/RECTOR](RECTOR) |
| Pretrained M2I | External scene encoder / trajectory generator | Backbone dependency and baseline component | [models/pretrained/m2i](pretrained/m2i) |

## RECTOR

`RECTOR` is the learned model in the repository. It combines:

- an M2I-derived scene representation,
- a Transformer CVAE trajectory decoder (`CVAETrajectoryHead`),
- multi-modal prediction (K=6 modes, 50 timesteps) over a 5-second horizon,
- rule applicability prediction (`ApplicabilityHead`) to expose behavior semantics alongside motion output, and
- a 7-component loss including applicability BCE supervision.

Training supports pre-cached data loading, mixed precision, gradient accumulation, and `torch.compile`.

This is the model family to use when reporting classical motion-prediction metrics such as minADE, minFDE, and miss rate.

Relevant artifacts:

- trained checkpoints in `models/RECTOR/output/`,
- evaluation scripts in `models/RECTOR/scripts/evaluation/`,
- visualization scripts in `models/RECTOR/scripts/visualization/`.

## M2I+RECTOR

`M2I+RECTOR` is not primarily a training pipeline. Instead, it uses pretrained DenseTNT trajectory candidates and then applies RECTOR's explicit rule evaluator to rank and select them in a receding-horizon loop.

This pipeline is especially useful when the paper needs to show:

- how rule-aware selection changes behavior over time,
- how safety/legal/road/comfort trade-offs appear qualitatively,
- and how explicit rule reasoning can sit on top of an existing predictor.

The outputs are most visible in the generated movies and experiment summaries under `models/RECTOR/`.

## Pretrained M2I assets

The `pretrained/m2i/` directory stores the external M2I checkpoints used by this repository. These weights should be understood as dependencies rather than original contributions of this workspace. In the manuscript, they are best described as the pretrained backbone used for scene encoding or candidate generation.

## Recommended citation strategy for repository artifacts

When preparing figures or tables from this codebase, it is helpful to separate model assets by evidentiary role:

- **Primary learned model**: `RECTOR`
- **Qualitative planning-style demonstrations**: `M2I+RECTOR`
- **External dependency**: pretrained `M2I`

## Directory structure

```text
models/
├── RECTOR/            primary research subproject: code, outputs, movies, docs
├── pretrained/        pretrained M2I assets used by RECTOR
└── checkpoints/       additional or exported checkpoint storage
```

## Common workflows

### Evaluate RECTOR

```bash
cd /workspace/models/RECTOR
python scripts/evaluation/evaluate_rector.py \
    --checkpoint output/trained_RECTOR/best.pt
```

### Generate RECTOR qualitative videos

```bash
cd /workspace/models/RECTOR
python scripts/visualization/generate_receding_movies.py \
    --checkpoint output/trained_RECTOR/best.pt \
    --num_scenarios 20
```

### Generate M2I+RECTOR demonstrations

```bash
cd /workspace/models/RECTOR
python scripts/visualization/generate_m2i_movies.py \
    --num_scenarios 40 \
    --predict_interval 20 \
    --min_ego_speed 3.0
```

## Where to read next

- [models/RECTOR/README.md](RECTOR/README.md) for the scientific description of the main method.
- [models/RECTOR/scripts/README.md](RECTOR/scripts/README.md) for experiment, evaluation, and figure-generation scripts.
