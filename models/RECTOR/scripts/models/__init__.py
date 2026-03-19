"""
RECTOR Models Module.

Provides neural network components for rule-aware trajectory generation.

Components:
- SceneEncoder: Encode scene features (ego, agents, lanes)
- RuleApplicabilityHead: Predict which rules apply
- CVAETrajectoryHead: Generate multi-modal trajectories
- TieredRuleScorer: Rank by compliance priority
- RuleAwareGenerator: Main model combining all components
"""

from .scene_encoder import (
    SceneEncoder,
    PolylineEncoder,
    TransformerEncoderLayer,
)

from .applicability_head import (
    RuleApplicabilityHead,
    RuleApplicabilityLoss,
)

from .cvae_head import (
    CVAETrajectoryHead,
    TrajectoryEncoder,
)

from .tiered_scorer import (
    TieredRuleScorer,
    SoftLexicographicLoss,
    DifferentiableTieredSelection,
)

from .rule_aware_generator import (
    RuleAwareGenerator,
)


__all__ = [
    # Encoder
    "SceneEncoder",
    "PolylineEncoder",
    "TransformerEncoderLayer",
    # Applicability
    "RuleApplicabilityHead",
    "RuleApplicabilityLoss",
    # Trajectory
    "CVAETrajectoryHead",
    "TrajectoryEncoder",
    # Scoring
    "TieredRuleScorer",
    "SoftLexicographicLoss",
    "DifferentiableTieredSelection",
    # Main models
    "RuleAwareGenerator",
]
