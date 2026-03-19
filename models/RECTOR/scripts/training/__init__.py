"""
RECTOR Training Module.

Provides training pipeline for rule-aware trajectory generation.

Components:
- RECTORLoss: Combined loss function (WTA + applicability + KL + smoothness)
- WaymoDataset: Augmented TFRecord dataset with rule labels
- Callbacks: EarlyStopping, ApplicabilityHysteresis, etc.
"""

from .losses import RECTORLoss, TRAJECTORY_SCALE
from .train_rector import WaymoDataset, collate_fn

from .callbacks import (
    ProxyRefinementCallback,
    ApplicabilityHysteresis,
    EarlyStopping,
    TrainingMetrics,
)


__all__ = [
    # Loss
    "RECTORLoss",
    "TRAJECTORY_SCALE",
    # Data
    "WaymoDataset",
    "collate_fn",
    # Callbacks
    "ProxyRefinementCallback",
    "ApplicabilityHysteresis",
    "EarlyStopping",
    "TrainingMetrics",
]
