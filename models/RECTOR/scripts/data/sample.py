"""
Data structures for rule-aware trajectory generation.

Contains dataclasses for training samples with rule annotations.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import numpy as np


@dataclass
class AugmentedSample:
    """
    Single augmented training sample with rule annotations.

    Contains all data needed for training the rule-aware trajectory
    generator, including scene features, trajectories, and rule labels.
    """

    # Identifiers
    scenario_id: str
    sample_id: str = ""

    # Ego state history (before prediction)
    # [T_hist, 4] - (x, y, yaw, speed)
    ego_history: np.ndarray = field(default_factory=lambda: np.zeros((11, 4)))

    # Ground truth ego trajectory (future)
    # [H, 4] - (x, y, yaw, speed)
    ego_future_gt: np.ndarray = field(default_factory=lambda: np.zeros((80, 4)))

    # Perturbed ego trajectory (for contrastive learning)
    # [H, 4] - (x, y, yaw, speed)
    ego_future_perturbed: Optional[np.ndarray] = None

    # Other agent states
    # [N_agents, T_total, 4] - (x, y, yaw, speed)
    agent_states: np.ndarray = field(default_factory=lambda: np.zeros((0, 91, 4)))

    # Agent metadata
    # [N_agents, 3] - (type, length, width)
    agent_metadata: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))

    # Agent validity masks
    # [N_agents, T_total]
    agent_valid: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 91), dtype=bool)
    )

    # Map features - lane centerlines
    # [N_lanes, N_points, 2] - (x, y)
    lane_centers: np.ndarray = field(default_factory=lambda: np.zeros((0, 20, 2)))

    # Lane headings
    # [N_lanes, N_points]
    lane_headings: np.ndarray = field(default_factory=lambda: np.zeros((0, 20)))

    # Lane validity masks
    # [N_lanes]
    lane_valid: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=bool))

    # Road edges
    # [N_edges, 2]
    road_edges: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))

    # Traffic control features
    # Stoplines: [N_stoplines, 2]
    stoplines: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))

    # Signal states per stopline: [N_stoplines, T_total]
    signal_states: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 91), dtype=int)
    )

    # Crosswalk polygons: [N_crosswalks, 4, 2]
    crosswalk_polygons: np.ndarray = field(default_factory=lambda: np.zeros((0, 4, 2)))

    # Rule labels
    # Applicability: [NUM_RULES] binary
    rule_applicability: np.ndarray = field(
        default_factory=lambda: np.zeros((28,), dtype=bool)
    )

    # Violations for GT trajectory: [NUM_RULES] binary
    rule_violations_gt: np.ndarray = field(
        default_factory=lambda: np.zeros((28,), dtype=bool)
    )

    # Violation severity for GT: [NUM_RULES] float in [0, 1]
    rule_severity_gt: np.ndarray = field(
        default_factory=lambda: np.zeros((28,), dtype=float)
    )

    # Violations for perturbed trajectory: [NUM_RULES] binary
    rule_violations_perturbed: Optional[np.ndarray] = None

    # Violation severity for perturbed: [NUM_RULES] float
    rule_severity_perturbed: Optional[np.ndarray] = None

    # Perturbation metadata
    perturbation_type: str = "none"
    perturbation_params: Dict[str, Any] = field(default_factory=dict)

    # Scenario metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate and fix array types."""
        self.ego_history = np.atleast_2d(self.ego_history).astype(float)
        self.ego_future_gt = np.atleast_2d(self.ego_future_gt).astype(float)
        self.rule_applicability = np.atleast_1d(self.rule_applicability).astype(bool)
        self.rule_violations_gt = np.atleast_1d(self.rule_violations_gt).astype(bool)
        self.rule_severity_gt = np.atleast_1d(self.rule_severity_gt).astype(float)

    @property
    def num_agents(self) -> int:
        """Number of other agents."""
        return self.agent_states.shape[0]

    @property
    def num_lanes(self) -> int:
        """Number of lane polylines."""
        return self.lane_centers.shape[0]

    @property
    def history_length(self) -> int:
        """Number of history timesteps."""
        return self.ego_history.shape[0]

    @property
    def future_length(self) -> int:
        """Number of future timesteps."""
        return self.ego_future_gt.shape[0]

    @property
    def has_perturbation(self) -> bool:
        """Check if sample has a perturbed trajectory."""
        return self.ego_future_perturbed is not None

    def get_ego_size(self) -> np.ndarray:
        """Get ego vehicle size [length, width]."""
        return self.metadata.get("ego_size", np.array([4.5, 2.0]))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "scenario_id": self.scenario_id,
            "sample_id": self.sample_id,
            "ego_history": self.ego_history,
            "ego_future_gt": self.ego_future_gt,
            "ego_future_perturbed": self.ego_future_perturbed,
            "agent_states": self.agent_states,
            "agent_metadata": self.agent_metadata,
            "agent_valid": self.agent_valid,
            "lane_centers": self.lane_centers,
            "lane_headings": self.lane_headings,
            "lane_valid": self.lane_valid,
            "road_edges": self.road_edges,
            "stoplines": self.stoplines,
            "signal_states": self.signal_states,
            "crosswalk_polygons": self.crosswalk_polygons,
            "rule_applicability": self.rule_applicability,
            "rule_violations_gt": self.rule_violations_gt,
            "rule_severity_gt": self.rule_severity_gt,
            "rule_violations_perturbed": self.rule_violations_perturbed,
            "rule_severity_perturbed": self.rule_severity_perturbed,
            "perturbation_type": self.perturbation_type,
            "perturbation_params": self.perturbation_params,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AugmentedSample":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class BatchedSamples:
    """
    Batched samples ready for model input.

    All tensors have batch dimension as first axis.
    """

    # [B, T_hist, 4]
    ego_history: np.ndarray

    # [B, H, 4]
    ego_future_gt: np.ndarray

    # [B, N_max, T_total, 4] (padded)
    agent_states: np.ndarray

    # [B, N_max, 3]
    agent_metadata: np.ndarray

    # [B, N_max, T_total]
    agent_valid: np.ndarray

    # [B, L_max, P, 2]
    lane_centers: np.ndarray

    # [B, L_max, P]
    lane_headings: np.ndarray

    # [B, L_max]
    lane_valid: np.ndarray

    # [B, NUM_RULES]
    rule_applicability: np.ndarray

    # [B, NUM_RULES]
    rule_violations: np.ndarray

    # [B, NUM_RULES]
    rule_severity: np.ndarray

    # Batch size
    batch_size: int = 0

    def __post_init__(self):
        self.batch_size = self.ego_history.shape[0]


@dataclass
class SceneFeatures:
    """
    Scene features for model input (as tensors).

    Used by proxies and model forward pass.
    """

    # Agent features
    agent_positions: Any = None  # [B, N, H, 2]
    agent_velocities: Any = None  # [B, N, H, 2]
    agent_sizes: Any = None  # [B, N, 2]
    agent_types: Any = None  # [B, N]
    agent_valid: Any = None  # [B, N, H]

    # Ego features
    ego_size: Any = None  # [B, 2]

    # Lane features
    lane_centers: Any = None  # [B, L, P, 2]
    lane_headings: Any = None  # [B, L, P]
    lane_valid: Any = None  # [B, L]

    # Road features
    road_edges: Any = None  # [B, E, 2]
    road_edge_valid: Any = None  # [B, E]

    # Signal features
    stoplines: Any = None  # [B, S, 2]
    signal_states: Any = None  # [B, S, H]

    # Crosswalk features
    crosswalk_polygons: Any = None  # [B, C, 4, 2]

    # VRU features (extracted from agents)
    vru_positions: Any = None  # [B, V, H, 2]
    vru_valid: Any = None  # [B, V, H]

    # Speed limits
    speed_limits: Any = None  # [B] or [B, H]

    # Context flags
    is_turning_left: Any = None  # [B]
    in_intersection: Any = None  # [B]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for proxy input."""
        return {k: v for k, v in self.__dict__.items() if v is not None}
