# RECTOR Inference

This directory contains the code used to run trained RECTOR models at test time. It is intentionally smaller than the training stack and focuses on turning a checkpoint plus scene tensors into predictions and associated metadata.

## Main files

| File | Role |
|---|---|
| `pipeline.py` | Reusable inference logic for model loading, prediction, and optional rule-aware post-processing |
| `run_inference_demo.py` | Minimal script for verifying that a trained checkpoint produces sensible outputs |

## Typical use

This directory is most useful for:

- sanity-checking a checkpoint,
- demonstrating output format,
- and integrating the trained model into downstream analysis scripts.

For large-scale evaluation, use the scripts in `../evaluation/`.

## Why it matters for a paper

Inference code clarifies what the model actually outputs at test time: trajectory modes, confidence scores, and rule-related predictions. That makes it useful when writing the test-time inference paragraph of the methods section.
