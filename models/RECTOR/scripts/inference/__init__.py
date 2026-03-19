"""
RECTOR Inference Module.

Provides inference pipeline for rule-aware trajectory generation.

Components:
- RECTORInference: Main inference engine
- StreamingInference: Real-time inference with caching
- compute_metrics: Evaluation metrics
"""

from .pipeline import (
    InferenceConfig,
    RECTORInference,
    StreamingInference,
    compute_metrics,
)


__all__ = [
    "InferenceConfig",
    "RECTORInference",
    "StreamingInference",
    "compute_metrics",
]
