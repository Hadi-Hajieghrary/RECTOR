# Data Pipeline and Rule Augmentation

This directory documents how raw WOMD scenarios become training-ready and evaluation-ready inputs for RECTOR. For paper writing, this is the most important place to understand data provenance, preprocessing assumptions, and the meaning of the added rule annotations.

## Role of the data pipeline

The data layer supports three goals:

1. **preserve the original WOMD motion context**,
2. **convert scenarios into formats that are practical for learning and visualization**, and
3. **attach explicit rule metadata that can be consumed by models and analyses**.

The repository therefore does more than store dataset files. It turns WOMD into a rule-aware research dataset tailored to the RECTOR experiments.

## Core processing stages

### 1. Raw WOMD scenarios

The starting point is the Waymo Open Motion Dataset scenario format. These records provide trajectories, dynamic-map information, traffic-light state, and scenario metadata.

### 2. Interactive-scenario filtering

The project focuses on interaction-rich driving behavior. Filtering utilities identify interactive subsets that are more relevant for motion prediction and behavior analysis than purely static or weakly coupled scenes.

### 3. Format conversion

The repository contains converters that map raw scenario protos into tensor-friendly or M2I-compatible formats. This step standardizes inputs for training and analysis.

### 4. Rule augmentation

The most distinctive step is rule augmentation. For each scenario, the rule-evaluation framework determines:

- whether a rule is applicable,
- whether it is violated,
- and how severe the violation is when a graded score is available.

These signals become part of the processed dataset and are later used for learning, analysis, or explicit candidate scoring.

## Why the augmentation matters

Without augmentation, the dataset supports forecasting but not structured reasoning about behavior quality. The added rule metadata makes it possible to ask questions such as:

- which modes are geometrically accurate but behaviorally poor,
- whether a model understands where rules apply,
- and whether explicit rule scoring changes selection outcomes.

This is a central design decision in RECTOR and one of the key connections between the data pipeline and the scientific claims of the project.

## Main directory layout

```text
data/
├── WOMD/
│   ├── datasets/         raw and processed WOMD files
│   ├── scripts/          filtering, conversion, and visualization utilities
│   ├── src/              low-level conversion utilities
│   ├── visualizations/   generated static scene visualizations
│   ├── movies/           generated BEV videos
│   └── waymo_rule_eval/  rule evaluation and augmentation framework
└── README.md
```

## Processed outputs used by the project

### Raw scenario records

Used when the original Waymo protocol or visualization fidelity is required.

### Converted tensor or example format

Used when fixed-shape access patterns are needed for model training and scripted evaluation.

### Rule-augmented records

Used when the experiment needs structured behavioral supervision. These records are the most important processed artifacts for RECTOR training.

## Typical reproduction workflow

### Prepare interactive subsets

```bash
python data/WOMD/scripts/lib/filter_interactive_scenario.py \
    --input-dir data/WOMD/datasets/waymo_open_dataset/motion_v_1_3_0/raw/scenario/training \
    --output-dir data/WOMD/datasets/waymo_open_dataset/motion_v_1_3_0/raw/scenario/training_interactive
```

### Augment with rule metadata

```bash
cd /workspace/data/WOMD
python -m waymo_rule_eval.augmentation.augment_cli \
    --input /workspace/data/WOMD/datasets/waymo_open_dataset/motion_v_1_3_0/raw/scenario/training_interactive \
    --output /workspace/data/WOMD/datasets/waymo_open_dataset/motion_v_1_3_0/processed/augmented/scenario/training_interactive
```

### Pre-cache for fast training (optional but recommended)

```bash
cd /workspace/models/RECTOR/scripts
python training/preprocess_cache.py --split train --workers 8
python training/preprocess_cache.py --split val --workers 8
```

This converts augmented TFRecords into individual `.pt` files stored in `.../processed/augmented/cached/`, enabling 3-5× faster training epochs via random-access `CachedDataset`.

## Recommended interpretation for a paper

When describing the dataset pipeline in a manuscript, the following language is usually the most faithful to the codebase:

- the project uses WOMD as the base source of motion data,
- interaction-focused subsets are extracted for modeling convenience,
- scenarios are converted into model-friendly representations,
- and RECTOR contributes an explicit rule-augmentation stage that enriches each sample with applicability, violation, and severity information.

## Related documentation

- [../README.md](../README.md) — repository-level overview.
- [../models/RECTOR/README.md](../models/RECTOR/README.md) — how the data is used by the main models.
- [WOMD/waymo_rule_eval/README.md](WOMD/waymo_rule_eval/README.md) — detailed rule-evaluation framework documentation.
