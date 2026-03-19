"""
Temporal spatial indexing for efficient multi-frame neighbor queries.

Pre-builds KD-trees for all frames at once, avoiding O(N log N)
construction on every frame query.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np

try:
    from scipy.spatial import cKDTree as KDTree
except ImportError:
    from scipy.spatial import KDTree

if TYPE_CHECKING:
    from .context import Agent


@dataclass
class FrameSpatialIndex:
    """
    Spatial index for a single frame.

    Provides efficient radius and k-nearest-neighbor queries for
    agents at a specific timestep.
    """

    frame_idx: int
    _tree: Optional[KDTree] = field(default=None, repr=False)
    _positions: np.ndarray = field(default_factory=lambda: np.empty((0, 2)), repr=False)
    _agent_indices: List[int] = field(default_factory=list, repr=False)
    _agents: List["Agent"] = field(default_factory=list, repr=False)

    @property
    def n_agents(self) -> int:
        """Number of agents with valid positions at this frame."""
        return len(self._agent_indices)

    @property
    def is_empty(self) -> bool:
        """True if no agents have valid positions at this frame."""
        return self._tree is None or len(self._positions) == 0

    def query_radius(
        self, x: float, y: float, radius: float, agent_type: Optional[str] = None
    ) -> List[Tuple["Agent", float]]:
        """Find all agents within radius of (x, y), optionally filtered by type."""
        if self.is_empty:
            return []

        indices = self._tree.query_ball_point([x, y], radius)
        results = []

        for idx in indices:
            agent = self._agents[self._agent_indices[idx]]

            # Optional type filter
            if agent_type is not None:
                if agent.type.lower() != agent_type.lower():
                    continue

            dist = np.sqrt(
                (self._positions[idx, 0] - x) ** 2 + (self._positions[idx, 1] - y) ** 2
            )
            results.append((agent, dist))

        # Sort by distance
        results.sort(key=lambda pair: pair[1])
        return results

    def query_k_nearest(
        self, x: float, y: float, k: int = 5, agent_type: Optional[str] = None
    ) -> List[Tuple["Agent", float]]:
        """Find k nearest agents to (x, y), optionally filtered by type."""
        if self.is_empty:
            return []

        # Query more than k if filtering by type
        query_k = min(k * 3, self.n_agents) if agent_type else min(k, self.n_agents)

        if query_k == 0:
            return []

        distances, indices = self._tree.query([x, y], k=query_k)

        # Handle single result case
        if np.isscalar(distances):
            distances = [distances]
            indices = [indices]

        results = []
        for dist, idx in zip(distances, indices):
            if idx >= len(self._agent_indices):
                continue

            agent = self._agents[self._agent_indices[idx]]

            if agent_type is not None:
                if agent.type.lower() != agent_type.lower():
                    continue

            results.append((agent, float(dist)))

            if len(results) >= k:
                break

        return results

    def query_vrus(
        self, x: float, y: float, radius: float
    ) -> List[Tuple["Agent", float]]:
        """
        Find VRUs (pedestrians and cyclists) within radius.

        Args:
            x, y: Query point
            radius: Search radius

        Returns:
            List of (agent, distance) tuples for VRUs only
        """
        all_nearby = self.query_radius(x, y, radius)
        return [(a, d) for a, d in all_nearby if a.is_vru]

    def query_vehicles(
        self, x: float, y: float, radius: float
    ) -> List[Tuple["Agent", float]]:
        """
        Find vehicles within radius.

        Args:
            x, y: Query point
            radius: Search radius

        Returns:
            List of (agent, distance) tuples for vehicles only
        """
        return self.query_radius(x, y, radius, agent_type="vehicle")


class TemporalSpatialIndex:
    """
    Spatial index across all frames.

    Pre-builds KD-trees for all frames at construction time,
    enabling O(log N) queries at any frame.
    """

    def __init__(self, agents: List["Agent"], n_frames: int):
        """
        Build spatial indices for all frames.

        Args:
            agents: List of Agent objects
            n_frames: Number of frames in scenario
        """
        self._agents = agents
        self._n_frames = n_frames
        self._frame_indices: List[FrameSpatialIndex] = []

        # Pre-build indices for all frames
        for t in range(n_frames):
            frame_index = self._build_frame_index(t)
            self._frame_indices.append(frame_index)

    def _build_frame_index(self, t: int) -> FrameSpatialIndex:
        """Build spatial index for a single frame."""
        positions = []
        agent_indices = []

        for i, agent in enumerate(self._agents):
            if agent.is_valid_at(t):
                positions.append([agent.x[t], agent.y[t]])
                agent_indices.append(i)

        frame_index = FrameSpatialIndex(
            frame_idx=t,
            _agents=self._agents,
            _agent_indices=agent_indices,
        )

        if positions:
            pos_array = np.array(positions)
            frame_index._positions = pos_array
            frame_index._tree = KDTree(pos_array)

        return frame_index

    @property
    def n_frames(self) -> int:
        """Number of frames indexed."""
        return self._n_frames

    def at_frame(self, t: int) -> FrameSpatialIndex:
        """
        Get spatial index for a specific frame.

        Args:
            t: Frame index

        Returns:
            FrameSpatialIndex for that frame
        """
        if t < 0 or t >= self._n_frames:
            # Return empty index for out-of-bounds
            return FrameSpatialIndex(frame_idx=t, _agents=self._agents)
        return self._frame_indices[t]

    def query_radius(
        self,
        t: int,
        x: float,
        y: float,
        radius: float,
        agent_type: Optional[str] = None,
    ) -> List[Tuple["Agent", float]]:
        """
        Query agents within radius at a specific frame.

        Args:
            t: Frame index
            x, y: Query point
            radius: Search radius
            agent_type: Optional type filter

        Returns:
            List of (agent, distance) tuples
        """
        return self.at_frame(t).query_radius(x, y, radius, agent_type)

    def query_k_nearest(
        self, t: int, x: float, y: float, k: int = 5, agent_type: Optional[str] = None
    ) -> List[Tuple["Agent", float]]:
        """
        Query k nearest agents at a specific frame.

        Args:
            t: Frame index
            x, y: Query point
            k: Number of neighbors
            agent_type: Optional type filter

        Returns:
            List of (agent, distance) tuples
        """
        return self.at_frame(t).query_k_nearest(x, y, k, agent_type)
