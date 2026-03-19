"""
Data pipeline for RECTOR rule-aware trajectory generation.

This module provides:
- AugmentedSample: Data structure for training samples
- AugmentedDataset: PyTorch Dataset implementation
- WeightedSampler: Balanced sampling for rare rules
- TrajectoryPerturbation: Data augmentation strategies
- Label generation utilities
"""

from .sample import AugmentedSample, BatchedSamples, SceneFeatures
from .dataset import (
    AugmentedDataset,
    WeightedSampler,
    WeightedRuleSampler,
    collate_fn,
    create_dataloader,
)
from .perturbations import (
    TrajectoryPerturbation,
    PerturbationResult,
    PerturbationPipeline,
    SpeedScaling,
    LateralOffset,
    HardBraking,
    SharpTurn,
    SignalIgnore,
    GaussianNoise,
)
from .label_generator import (
    generate_labels,
    generate_applicability_only,
    generate_contrastive_labels,
    OtherFuturesAugmenter,
    PredictorErrorStats,
)


__all__ = [
    # Data structures
    "AugmentedSample",
    "BatchedSamples",
    "SceneFeatures",
    # Dataset
    "AugmentedDataset",
    "WeightedSampler",
    "collate_fn",
    "create_dataloader",
    # Perturbations
    "TrajectoryPerturbation",
    "PerturbationResult",
    "PerturbationPipeline",
    "SpeedScaling",
    "LateralOffset",
    "HardBraking",
    "SharpTurn",
    "SignalIgnore",
    "GaussianNoise",
    # Label generation
    "generate_labels",
    "generate_applicability_only",
    "generate_contrastive_labels",
    "OtherFuturesAugmenter",
    "PredictorErrorStats",
]
