"""
RECTOR (REactive Conditional Trajectory ORchestrator)

An interactive trajectory planner built on M2I's conditional prediction.
"""

__version__ = "0.1.0"
__author__ = "RECTOR Team"

from .data_contracts import (
    EgoCandidate,
    ReactorTensorPack,
    SceneEmbeddingCache,
    PredictionResult,
    PlanningConfig,
)

__all__ = [
    "EgoCandidate",
    "ReactorTensorPack",
    "SceneEmbeddingCache",
    "PredictionResult",
    "PlanningConfig",
]
