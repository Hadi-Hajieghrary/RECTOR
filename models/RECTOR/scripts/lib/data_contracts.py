"""
RECTOR Data Contracts

Immutable data structures that define the core interfaces for RECTOR.
These contracts ensure type safety and prevent data contamination.

Design Principles:
1. IMMUTABLE: All dataclasses use frozen=True to prevent mutation
2. TYPED: Full type annotations for static analysis
3. GPU-READY: Tensor fields are torch.Tensor, ready for batched ops
4. SERIALIZABLE: Support for checkpointing and debugging

Phase 2 Implementation - December 2024
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import numpy as np

# Conditional import for torch (may not be available in all contexts)
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False


@dataclass(frozen=True)
class PlanningConfig:
    """
    Configuration for RECTOR planning loop.

    All parameters that affect planning behavior are collected here.
    Immutable to prevent accidental modification during planning.
    """

    # Ego candidate generation
    num_candidates: int = 16  # M: number of ego trajectory candidates
    candidate_horizon: int = 80  # Future timesteps (8 seconds at 10Hz)
    candidate_dt: float = 0.1  # Timestep duration

    # Reactor selection
    max_reactors: int = 3  # K: max reactors to consider
    reactor_distance_threshold: float = 50.0  # Max distance to consider
    reactor_ttc_threshold: float = 10.0  # Time-to-collision threshold

    # Prediction modes
    num_prediction_modes: int = 6  # M2I outputs 6 modes

    # Safety thresholds
    collision_radius: float = 3.0  # Conservative collision check radius
    min_safety_margin: float = 2.0  # Minimum clearance to reactors

    # CVaR risk parameters
    cvar_alpha: float = 0.1  # Risk level (lower = more conservative)
    cvar_num_samples: int = 100  # Samples for CVaR estimation
    cvar_seed: int = 42  # For reproducibility (common random numbers)

    # Non-cooperative envelope
    max_decel: float = 8.0  # m/s^2, max assumed deceleration
    max_lateral_accel: float = 4.0  # m/s^2, max lateral acceleration
    reaction_time: float = 0.5  # seconds, assumed reaction delay

    # Model paths (relative to model root)
    densetnt_path: str = "densetnt/model.24.bin"
    relation_path: str = "relation_v2v/model.24.bin"
    conditional_path: str = "conditional_v2v/model.29.bin"

    # Device
    device: str = "cuda"

    def validate(self) -> bool:
        """Validate configuration parameters."""
        assert self.num_candidates > 0, "Must have at least 1 candidate"
        assert self.max_reactors > 0, "Must have at least 1 reactor"
        assert self.candidate_horizon > 0, "Horizon must be positive"
        assert 0.0 < self.cvar_alpha <= 1.0, "CVaR alpha must be in (0, 1]"
        return True


@dataclass(frozen=True)
class ModelConfig:
    """
    M2I model configuration parameters.

    Mirrors the essential fields from M2I's Args class.
    """

    hidden_size: int = 128
    future_frame_num: int = 80
    mode_num: int = 6
    sub_graph_depth: int = 3
    global_graph_depth: int = 1
    nms_threshold: float = 7.0


@dataclass(frozen=True)
class AgentState:
    """
    State of a single agent at a specific timestep.

    Coordinates are in world frame (Waymo global coordinates).
    """

    x: float  # World X position (meters)
    y: float  # World Y position (meters)
    heading: float  # Heading angle (radians)
    velocity_x: float  # Velocity X component (m/s)
    velocity_y: float  # Velocity Y component (m/s)
    agent_id: int = -1  # Agent identifier (-1 for ego)
    length: float = 4.5  # Agent length (meters)
    width: float = 2.0  # Agent width (meters)
    agent_type: int = 1  # 1=vehicle, 2=pedestrian, 3=cyclist

    @property
    def position(self) -> np.ndarray:
        """Position as numpy array [x, y]."""
        return np.array([self.x, self.y], dtype=np.float32)

    @property
    def velocity(self) -> np.ndarray:
        """Velocity as numpy array [vx, vy]."""
        return np.array([self.velocity_x, self.velocity_y], dtype=np.float32)

    @property
    def speed(self) -> float:
        """Scalar speed (m/s)."""
        return np.sqrt(self.velocity_x**2 + self.velocity_y**2)


@dataclass(frozen=True)
class AgentHistory:
    """
    Historical trajectory of an agent.

    Contains T timesteps of state information.
    """

    agent_id: int
    states: Tuple[AgentState, ...]  # Tuple for immutability

    @property
    def num_timesteps(self) -> int:
        return len(self.states)

    @property
    def current_state(self) -> AgentState:
        """Most recent state."""
        return self.states[-1]

    def to_array(self) -> np.ndarray:
        """Convert to numpy array [T, 7] (x, y, heading, vx, vy, length, width)."""
        return np.array(
            [
                [s.x, s.y, s.heading, s.velocity_x, s.velocity_y, s.length, s.width]
                for s in self.states
            ],
            dtype=np.float32,
        )


@dataclass(frozen=True)
class EgoCandidate:
    """
    A candidate ego trajectory for evaluation.

    Each candidate represents a possible future trajectory for the ego vehicle.
    The planner generates M candidates and evaluates each against K reactors.

    CRITICAL: Trajectory is in WORLD frame. Transform to reactor-local frame
    before feeding to conditional model.
    """

    candidate_id: int

    # Trajectory: [H, 2] where H = horizon (80 timesteps = 8 seconds)
    # In WORLD coordinates
    trajectory: np.ndarray  # [H, 2] - (x, y) positions

    # Optional velocity profile: [H, 2] - (vx, vy) velocities
    velocities: Optional[np.ndarray] = None

    # Generation metadata
    generation_method: str = "sampling"  # "sampling", "optimization", "lattice"
    parent_id: Optional[int] = None  # For iterative refinement

    @property
    def horizon(self) -> int:
        return self.trajectory.shape[0]

    @property
    def start_position(self) -> np.ndarray:
        return self.trajectory[0]

    @property
    def end_position(self) -> np.ndarray:
        return self.trajectory[-1]

    def get_position_at(self, t: int) -> np.ndarray:
        """Get position at timestep t."""
        assert 0 <= t < self.horizon, f"t={t} out of range [0, {self.horizon})"
        return self.trajectory[t]

    def compute_travel_distance(self) -> float:
        """Compute total distance traveled along trajectory."""
        diffs = np.diff(self.trajectory, axis=0)
        return float(np.sum(np.linalg.norm(diffs, axis=1)))

    def to_tensor(self, device: str = "cuda") -> "torch.Tensor":
        """Convert to torch tensor on specified device."""
        if not TORCH_AVAILABLE:
            raise ImportError("torch not available")
        return torch.tensor(self.trajectory, dtype=torch.float32, device=device)


@dataclass(frozen=True)
class EgoCandidateBatch:
    """
    Batch of ego candidates for parallel evaluation.

    Stores M candidates as a single tensor for efficient batched operations.
    """

    candidates: Tuple[EgoCandidate, ...]

    # Batched tensor: [M, H, 2] - all trajectories stacked
    trajectories_tensor: np.ndarray  # [M, H, 2]

    @property
    def num_candidates(self) -> int:
        return len(self.candidates)

    @property
    def horizon(self) -> int:
        return self.trajectories_tensor.shape[1]

    def to_tensor(self, device: str = "cuda") -> "torch.Tensor":
        """Convert to torch tensor [M, H, 2]."""
        if not TORCH_AVAILABLE:
            raise ImportError("torch not available")
        return torch.tensor(
            self.trajectories_tensor, dtype=torch.float32, device=device
        )

    @classmethod
    def from_candidates(cls, candidates: List[EgoCandidate]) -> "EgoCandidateBatch":
        """Create batch from list of candidates."""
        trajectories = np.stack([c.trajectory for c in candidates], axis=0)
        return cls(
            candidates=tuple(candidates),
            trajectories_tensor=trajectories,
        )


@dataclass(frozen=True)
class ReactorTensorPack:
    """
    Pre-computed tensors for a single reactor, in reactor-centric frame.

    These are computed ONCE per reactor per tick and reused across all
    M ego candidates. This is the key to efficient batched inference.

    CRITICAL: All position data is in REACTOR-LOCAL frame.
    The coordinate transform is defined by (origin, rotation).
    """

    reactor_id: int

    # Reactor's current state
    position_world: np.ndarray  # [2] world position
    heading_world: float  # World heading (radians)

    # Coordinate transform: reactor-centric frame
    # To transform world→local: rotate by -rotation, then subtract origin
    origin: np.ndarray  # [2] = position_world
    rotation: float  # = heading_world

    # Agent history in reactor-local frame: [T_hist, state_dim]
    reactor_history_local: np.ndarray

    # Agent type (1=vehicle, 2=pedestrian, 3=cyclist)
    reactor_type: int

    # M2I mapping fields (pre-computed for this reactor)
    # These are the essential fields needed by the conditional model
    matrix: np.ndarray  # [N_polylines, 128] feature vectors
    polyline_spans: List[slice]  # Index ranges for each polyline
    map_start_polyline_idx: int  # Where map polylines start

    # Scene context (for conditional model)
    goals_2D: np.ndarray  # [G, 2] candidate goal positions

    # Optional: Pre-computed embeddings if using cached encoder
    # These would be GPU tensors in a full implementation
    agent_embeddings: Optional[np.ndarray] = None  # [N_agents, hidden_dim]
    lane_embeddings: Optional[np.ndarray] = None  # [N_lanes, hidden_dim]
    context_embedding: Optional[np.ndarray] = None  # [hidden_dim]

    def transform_to_local(self, world_coords: np.ndarray) -> np.ndarray:
        """
        Transform world coordinates to reactor-local frame.

        Args:
            world_coords: [N, 2] or [2] world coordinates

        Returns:
            local_coords: Same shape, in reactor-local frame
        """
        import math

        # Translate to reactor origin
        translated = world_coords - self.origin

        # Rotate to reactor heading
        cos_a = math.cos(-self.rotation)
        sin_a = math.sin(-self.rotation)

        if translated.ndim == 1:
            x_local = translated[0] * cos_a - translated[1] * sin_a
            y_local = translated[0] * sin_a + translated[1] * cos_a
            return np.array([x_local, y_local])
        else:
            x_local = translated[:, 0] * cos_a - translated[:, 1] * sin_a
            y_local = translated[:, 0] * sin_a + translated[:, 1] * cos_a
            return np.stack([x_local, y_local], axis=1)

    def transform_to_world(self, local_coords: np.ndarray) -> np.ndarray:
        """
        Transform reactor-local coordinates to world frame.

        Args:
            local_coords: [N, 2] or [2] local coordinates

        Returns:
            world_coords: Same shape, in world frame
        """
        import math

        # Rotate back to world heading
        cos_a = math.cos(self.rotation)
        sin_a = math.sin(self.rotation)

        if local_coords.ndim == 1:
            x_world = local_coords[0] * cos_a - local_coords[1] * sin_a
            y_world = local_coords[0] * sin_a + local_coords[1] * cos_a
            return np.array([x_world + self.origin[0], y_world + self.origin[1]])
        else:
            x_world = local_coords[:, 0] * cos_a - local_coords[:, 1] * sin_a
            y_world = local_coords[:, 0] * sin_a + local_coords[:, 1] * cos_a
            return np.stack(
                [x_world + self.origin[0], y_world + self.origin[1]], axis=1
            )


@dataclass(frozen=True)
class SceneEmbeddingCache:
    """
    Cached encoder outputs for entire scene.

    This is what "encode once per tick" actually means:
    - Parse scenario data ONCE
    - Compute base mappings ONCE
    - Store reusable data for all M×K queries

    GPU Tensor Caching:
    The optional fields below can store GPU tensors from the encoder
    to avoid redundant computation across M×K queries:
    - raster_image_hidden: Cached CNN encoder output [224, 224, hidden_size]
    - element_states: Cached subgraph encoder outputs
    """

    # Scenario metadata
    scenario_id: str
    timestep: int

    # Ego state
    ego_position: np.ndarray  # [2] world position
    ego_heading: float  # World heading
    ego_history: np.ndarray  # [T_hist, state_dim]

    # All agents in scene
    agent_ids: Tuple[int, ...]
    agent_positions: np.ndarray  # [N, 2] world positions
    agent_headings: np.ndarray  # [N] headings
    agent_types: np.ndarray  # [N] agent types
    agent_histories: np.ndarray  # [N, T_hist, state_dim]
    objects_of_interest: np.ndarray  # [N] binary mask for interactive agents

    # Road network (in world frame)
    roadgraph_xyz: np.ndarray  # [R, 3] road points
    roadgraph_type: np.ndarray  # [R] road point types
    roadgraph_valid: np.ndarray  # [R] validity mask
    roadgraph_id: np.ndarray  # [R] lane IDs

    # Traffic lights
    traffic_light_state: Optional[np.ndarray] = None

    # BEV raster image (60 channels for scene, expandable to 150 for conditional)
    bev_image: Optional[np.ndarray] = None  # [224, 224, 60]

    # goals_2D grid (candidate goal positions in local frame)
    goals_2D: Optional[np.ndarray] = None  # [G, 2]

    # ==========================================================================
    # GPU Tensor Cache (for encoder output reuse)
    # ==========================================================================
    # These are populated by the adapter after first encoding pass
    # and reused across all M×K queries to avoid redundant computation.

    # Cached CNN encoder output for raster image (GPU tensor)
    # Shape: [224, 224, hidden_size] on GPU
    raster_image_hidden: Optional[Any] = (
        None  # torch.Tensor (use Any for frozen dataclass)
    )

    # Cached base mapping template (numpy, can be reused)
    base_mapping_template: Optional[Dict] = None

    # Flag indicating if GPU cache is populated
    gpu_cache_valid: bool = False

    def get_interactive_agent_ids(self) -> List[int]:
        """Get IDs of interactive agents (objects of interest)."""
        mask = self.objects_of_interest > 0
        return [self.agent_ids[i] for i in np.where(mask)[0]]

    def get_agent_state(self, agent_id: int) -> AgentState:
        """Get current state of an agent by ID."""
        idx = self.agent_ids.index(agent_id)
        return AgentState(
            x=float(self.agent_positions[idx, 0]),
            y=float(self.agent_positions[idx, 1]),
            heading=float(self.agent_headings[idx]),
            velocity_x=float(self.agent_histories[idx, -1, 3]),
            velocity_y=float(self.agent_histories[idx, -1, 4]),
            agent_type=int(self.agent_types[idx]),
        )

    def with_gpu_cache(
        self,
        raster_image_hidden: Any = None,
        base_mapping_template: Optional[Dict] = None,
    ) -> "SceneEmbeddingCache":
        """
        Create a new cache with GPU tensor data populated.

        Since this dataclass is frozen, we create a new instance
        with the cached GPU tensors included.

        Args:
            raster_image_hidden: Cached CNN encoder output (GPU tensor)
            base_mapping_template: Cached base mapping dict

        Returns:
            New SceneEmbeddingCache with GPU cache populated
        """
        return SceneEmbeddingCache(
            scenario_id=self.scenario_id,
            timestep=self.timestep,
            ego_position=self.ego_position,
            ego_heading=self.ego_heading,
            ego_history=self.ego_history,
            agent_ids=self.agent_ids,
            agent_positions=self.agent_positions,
            agent_headings=self.agent_headings,
            agent_types=self.agent_types,
            agent_histories=self.agent_histories,
            objects_of_interest=self.objects_of_interest,
            roadgraph_xyz=self.roadgraph_xyz,
            roadgraph_type=self.roadgraph_type,
            roadgraph_valid=self.roadgraph_valid,
            roadgraph_id=self.roadgraph_id,
            traffic_light_state=self.traffic_light_state,
            bev_image=self.bev_image,
            goals_2D=self.goals_2D,
            raster_image_hidden=raster_image_hidden or self.raster_image_hidden,
            base_mapping_template=base_mapping_template or self.base_mapping_template,
            gpu_cache_valid=True,
        )

    def clear_gpu_cache(self) -> "SceneEmbeddingCache":
        """Create a new cache with GPU tensors cleared (for memory management)."""
        return SceneEmbeddingCache(
            scenario_id=self.scenario_id,
            timestep=self.timestep,
            ego_position=self.ego_position,
            ego_heading=self.ego_heading,
            ego_history=self.ego_history,
            agent_ids=self.agent_ids,
            agent_positions=self.agent_positions,
            agent_headings=self.agent_headings,
            agent_types=self.agent_types,
            agent_histories=self.agent_histories,
            objects_of_interest=self.objects_of_interest,
            roadgraph_xyz=self.roadgraph_xyz,
            roadgraph_type=self.roadgraph_type,
            roadgraph_valid=self.roadgraph_valid,
            roadgraph_id=self.roadgraph_id,
            traffic_light_state=self.traffic_light_state,
            bev_image=self.bev_image,
            goals_2D=self.goals_2D,
            raster_image_hidden=None,
            base_mapping_template=None,
            gpu_cache_valid=False,
        )  # =============================================================================


# Prediction Result Contracts


@dataclass(frozen=True)
class SinglePrediction:
    """
    Prediction for a single (ego_candidate, reactor) pair.

    Contains multiple trajectory modes with confidence scores.
    """

    ego_candidate_id: int
    reactor_id: int

    # Predicted trajectories: [N_modes, H, 2] in WORLD frame
    trajectories: np.ndarray

    # Confidence scores: [N_modes]
    scores: np.ndarray

    @property
    def num_modes(self) -> int:
        return self.trajectories.shape[0]

    @property
    def horizon(self) -> int:
        return self.trajectories.shape[1]

    @property
    def best_mode(self) -> int:
        """Index of highest-confidence mode."""
        return int(np.argmax(self.scores))

    @property
    def best_trajectory(self) -> np.ndarray:
        """Highest-confidence trajectory [H, 2]."""
        return self.trajectories[self.best_mode]

    def get_position_at(self, t: int, mode: int = -1) -> np.ndarray:
        """Get predicted position at timestep t."""
        if mode < 0:
            mode = self.best_mode
        return self.trajectories[mode, t]


@dataclass(frozen=True)
class PredictionResult:
    """
    Complete prediction result for all (ego_candidate, reactor) pairs.

    Shape: [M, K, N_modes, H, 2] where:
    - M = number of ego candidates
    - K = number of reactors
    - N_modes = prediction modes per reactor (typically 6)
    - H = horizon (typically 80 timesteps)
    """

    # Full prediction tensor: [M, K, N_modes, H, 2]
    trajectories: np.ndarray

    # Score tensor: [M, K, N_modes]
    scores: np.ndarray

    # Metadata
    ego_candidate_ids: Tuple[int, ...]
    reactor_ids: Tuple[int, ...]

    @property
    def num_candidates(self) -> int:
        return self.trajectories.shape[0]

    @property
    def num_reactors(self) -> int:
        return self.trajectories.shape[1]

    @property
    def num_modes(self) -> int:
        return self.trajectories.shape[2]

    @property
    def horizon(self) -> int:
        return self.trajectories.shape[3]

    def get_prediction(self, candidate_idx: int, reactor_idx: int) -> SinglePrediction:
        """Get prediction for specific (candidate, reactor) pair."""
        return SinglePrediction(
            ego_candidate_id=self.ego_candidate_ids[candidate_idx],
            reactor_id=self.reactor_ids[reactor_idx],
            trajectories=self.trajectories[candidate_idx, reactor_idx],
            scores=self.scores[candidate_idx, reactor_idx],
        )

    def get_best_trajectories(self) -> np.ndarray:
        """
        Get best (highest-score) trajectory for each (candidate, reactor).

        Returns:
            [M, K, H, 2] - Best trajectory for each pair
        """
        best_modes = np.argmax(self.scores, axis=2)  # [M, K]
        M, K, _, H, _ = self.trajectories.shape

        result = np.zeros((M, K, H, 2), dtype=np.float32)
        for m in range(M):
            for k in range(K):
                result[m, k] = self.trajectories[m, k, best_modes[m, k]]
        return result


@dataclass(frozen=True)
class CollisionCheckResult:
    """
    Result of collision check for a single (ego_candidate, reactor_prediction) pair.
    """

    has_collision: bool
    collision_time: Optional[int] = None  # Timestep of first collision
    min_distance: float = float("inf")  # Minimum distance over horizon
    min_distance_time: Optional[int] = None  # Timestep of minimum distance


@dataclass(frozen=True)
class SafetyScore:
    """
    Safety evaluation for a single ego candidate across all reactors.
    """

    candidate_id: int

    # Per-reactor collision results
    collision_results: Tuple[CollisionCheckResult, ...]

    # Aggregate safety metrics
    has_any_collision: bool
    min_clearance: float  # Minimum distance to any reactor
    worst_reactor_id: Optional[int]  # Reactor with highest risk

    # CVaR-based risk score (lower = safer)
    cvar_risk: float

    # Overall safety score (higher = safer, 0-1 scale)
    safety_score: float

    @property
    def is_safe(self) -> bool:
        """Check if candidate passes all safety thresholds."""
        return not self.has_any_collision and self.safety_score > 0.5


@dataclass(frozen=True)
class PlanningResult:
    """
    Final result of RECTOR planning for a single timestep.
    """

    # Selected trajectory
    selected_candidate_id: int
    selected_trajectory: np.ndarray  # [H, 2]

    # All evaluated candidates with scores
    candidate_scores: Tuple[float, ...]

    # Safety evaluation
    safety_scores: Tuple[SafetyScore, ...]

    # Reactor predictions for selected candidate
    reactor_predictions: PredictionResult

    # Planning metadata
    planning_time_ms: float
    num_candidates_evaluated: int
    num_reactors: int

    @property
    def is_safe(self) -> bool:
        """Check if selected trajectory is safe."""
        selected_safety = self.safety_scores[self.selected_candidate_id]
        return selected_safety.is_safe


def create_ego_candidate(
    trajectory: np.ndarray,
    candidate_id: int = 0,
    velocities: Optional[np.ndarray] = None,
    generation_method: str = "sampling",
) -> EgoCandidate:
    """
    Factory function to create an EgoCandidate.

    Args:
        trajectory: [H, 2] trajectory in world coordinates
        candidate_id: Unique identifier
        velocities: Optional [H, 2] velocity profile
        generation_method: How the candidate was generated

    Returns:
        EgoCandidate instance
    """
    assert (
        trajectory.ndim == 2 and trajectory.shape[1] == 2
    ), f"trajectory must be [H, 2], got {trajectory.shape}"

    return EgoCandidate(
        candidate_id=candidate_id,
        trajectory=trajectory.astype(np.float32),
        velocities=velocities.astype(np.float32) if velocities is not None else None,
        generation_method=generation_method,
    )


def create_candidate_batch(
    trajectories: np.ndarray,
    generation_method: str = "sampling",
) -> EgoCandidateBatch:
    """
    Factory function to create an EgoCandidateBatch.

    Args:
        trajectories: [M, H, 2] batch of trajectories in world coordinates
        generation_method: How candidates were generated

    Returns:
        EgoCandidateBatch instance
    """
    assert (
        trajectories.ndim == 3 and trajectories.shape[2] == 2
    ), f"trajectories must be [M, H, 2], got {trajectories.shape}"

    M = trajectories.shape[0]
    candidates = [
        create_ego_candidate(
            trajectory=trajectories[i],
            candidate_id=i,
            generation_method=generation_method,
        )
        for i in range(M)
    ]

    return EgoCandidateBatch(
        candidates=tuple(candidates),
        trajectories_tensor=trajectories.astype(np.float32),
    )
