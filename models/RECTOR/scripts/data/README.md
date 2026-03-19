# RECTOR Training Data Utilities

This directory contains the dataset wrappers and preprocessing utilities used by the model-training code. It is the bridge between the augmented WOMD records and the tensors consumed by RECTOR.

## Responsibilities

The code here is responsible for:

- loading prepared scenario data,
- assembling scene features into model inputs,
- exposing trajectory and rule metadata in a stable format,
- and supporting training-time sampling behavior.

## Main modules

| Module | Role |
|---|---|
| `dataset.py` | Primary dataset wrapper used by training and evaluation |
| `label_generator.py` | Logic for constructing labels or training targets from processed data |
| `perturbations.py` | Data perturbations and controlled modifications for robustness studies |
| `sample.py` | Lightweight access helpers or sample-level utilities |

## Why this directory matters

A forecasting paper often focuses on model architecture, but reproducibility depends just as strongly on how inputs and labels are formed. This directory documents that practical layer of the pipeline.

## Related documentation

- [../../../data/README.md](../../../data/README.md) — repository-level data pipeline.
- [../training/README.md](../training/README.md) — how these data tensors are consumed during optimization.
