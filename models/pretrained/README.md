# `models/pretrained/` — Pretrained Model Weights

This directory contains pretrained model weights for the RECTOR workspace, primarily the **M2I (Marginal-to-Interactive)** models used for scene encoding and trajectory prediction.

---

## Table of Contents

1. [Overview](#overview)
2. [Directory Structure](#directory-structure)
3. [M2I Models](#m2i-models)
4. [Weight File Locations](#weight-file-locations)
5. [Usage](#usage)

---

## Overview

| Model | File | Size | Purpose |
|-------|------|------|---------|
| **DenseTNT** | `m2i/models/densetnt/model.24.bin` | ~120MB | Marginal trajectory prediction |
| **Relation V2V** | `m2i/models/relation_v2v/model.25.bin` | ~120MB | Relationship classification, **RECTOR encoder** |
| **Conditional V2V** | `m2i/models/conditional_v2v/model.29.bin` | ~120MB | Conditioned trajectory prediction |

---

## Directory Structure

```
pretrained/
├── README.md                    # This file
├── .gitkeep                     # Keeps directory in git
│
└── m2i/                         # M2I model ecosystem
    ├── README.md               # M2I documentation
    ├── models/                 # Weight files
    │   ├── densetnt/
    │   │   └── model.24.bin   # Marginal predictor (epoch 24)
    │   ├── relation_v2v/
    │   │   └── model.25.bin   # Relation classifier (epoch 25)
    │   └── conditional_v2v/
    │       └── model.29.bin   # Conditional predictor (epoch 29)
    ├── scripts/                # Inference scripts
    │   ├── bash/              # Shell wrappers
    │   └── lib/               # Python implementations
    ├── movies/                 # Generated visualizations
    └── relations/              # Relation prediction outputs
```

---

## M2I Models

### 1. DenseTNT (Marginal Predictor)

- **File**: `m2i/models/densetnt/model.24.bin`
- **Purpose**: Predict trajectories for individual agents independently
- **Output**: 6 trajectory modes per agent with confidence scores
- **Architecture**: VectorNet encoder + dense goal prediction

### 2. Relation V2V (Relation Classifier)

- **File**: `m2i/models/relation_v2v/model.25.bin`
- **Purpose**: Classify agent pair relationships (PASS/YIELD/NONE)
- **Output**: Influencer/reactor assignment
- **Used By**: **RECTOR scene encoder** (this is the key file for RECTOR)

### 3. Conditional V2V (Conditional Predictor)

- **File**: `m2i/models/conditional_v2v/model.29.bin`
- **Purpose**: Predict reactor trajectories conditioned on influencer
- **Output**: 6 reactor modes for each influencer mode
- **Input**: Influencer trajectory + scene context

---

## Weight File Locations

### Required for RECTOR

```bash
# The only weight file RECTOR needs:
/workspace/models/pretrained/m2i/models/relation_v2v/model.25.bin
```

### Required for Full M2I Pipeline

```bash
# All three models needed for M2I inference:
/workspace/models/pretrained/m2i/models/densetnt/model.24.bin
/workspace/models/pretrained/m2i/models/relation_v2v/model.25.bin
/workspace/models/pretrained/m2i/models/conditional_v2v/model.29.bin
```

---

## Usage

### In RECTOR

```python
from models.rule_aware_generator import RuleAwareGenerator

model = RuleAwareGenerator(
    use_m2i_encoder=True,
    m2i_checkpoint='/workspace/models/pretrained/m2i/models/relation_v2v/model.25.bin',
    freeze_m2i=True,  # Keep pretrained M2I encoder frozen
)
```

### In M2I Inference

```bash
# Run full M2I pipeline
bash models/pretrained/m2i/scripts/bash/run_m2i_pipeline.sh -n 10
```

### Loading Weights Directly

```python
import torch

# Load relation model weights
weights = torch.load('/workspace/models/pretrained/m2i/models/relation_v2v/model.25.bin')
print(f"Loaded {len(weights)} weight tensors")
```

---

## Downloading Weights

If weights are not present, download from the M2I repository:

```bash
# Check for missing weights
ls -la m2i/models/*/

# Download if missing (placeholder - adjust URL as needed)
# wget -O m2i/models/densetnt/model.24.bin <URL>
# wget -O m2i/models/relation_v2v/model.25.bin <URL>
# wget -O m2i/models/conditional_v2v/model.29.bin <URL>
```

---

## Related Documentation

| Document | Description |
|----------|-------------|
| [m2i/README.md](m2i/README.md) | M2I inference pipeline |
| [../RECTOR/README.md](../RECTOR/README.md) | RECTOR model |
| [m2i/models/README.md](m2i/models/README.md) | Model architecture details |

---

*Last updated: March 10, 2026*
