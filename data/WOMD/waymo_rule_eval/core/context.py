"""
Context classes for scenario representation.

Contains data classes for:
- EgoState: Ego vehicle trajectory with validity tracking
- Agent: Other traffic participants (vehicles, pedestrians, cyclists)
- MapContext: Static map features
- MapSignals: Dynamic traffic signal states
- ScenarioContext: Complete scenario container
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from ..utils.constants import (
    DEFAULT_DT_S,
    DEFAULT_EGO_LENGTH,
    DEFAULT_EGO_WIDTH,
    GREEN_SIGNAL_STATES,
    RED_SIGNAL_STATES,
    SIGNAL_STATE_UNKNOWN,
    YELLOW_SIGNAL_STATES,
)


@dataclass
class EgoState:
    """
    Ego vehicle state trajectory with validity tracking.

    Stores position, heading, and speed trajectories along with
    a validity mask indicating which timesteps have valid data.
    """

    x: np.ndarray
    y: np.ndarray
    yaw: np.ndarray
    speed: np.ndarray
    length: float = DEFAULT_EGO_LENGTH
    width: float = DEFAULT_EGO_WIDTH
    valid: Optional[np.ndarray] = None

    # Optional Frenet coordinates (computed by frenet_projector)
    s: Optional[np.ndarray] = None
    d: Optional[np.ndarray] = None

    # Cached kinematic features
    _acceleration: Optional[np.ndarray] = field(default=None, repr=False, compare=False)
    _jerk: Optional[np.ndarray] = field(default=None, repr=False, compare=False)
    _yaw_rate: Optional[np.ndarray] = field(default=None, repr=False, compare=False)
    _lateral_acceleration: Optional[np.ndarray] = field(
        default=None, repr=False, compare=False
    )
    _cached_dt: Optional[float] = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        """Ensure arrays and create default validity mask."""
        self.x = np.atleast_1d(np.asarray(self.x, dtype=float))
        self.y = np.atleast_1d(np.asarray(self.y, dtype=float))
        self.yaw = np.atleast_1d(np.asarray(self.yaw, dtype=float))
        self.speed = np.atleast_1d(np.asarray(self.speed, dtype=float))

        if self.valid is None:
            self.valid = ~(np.isnan(self.x) | np.isnan(self.y))
        else:
            self.valid = np.atleast_1d(np.asarray(self.valid, dtype=bool))

    def __len__(self) -> int:
        return len(self.x)

    @property
    def n_valid(self) -> int:
        """Number of valid timesteps."""
        return int(np.sum(self.valid))

    def is_valid_at(self, t: int) -> bool:
        """Check if ego has valid state at timestep t."""
        if t < 0 or t >= len(self.valid):
            return False
        return bool(self.valid[t])

    def position_at(self, t: int) -> Optional[Tuple[float, float]]:
        """Get (x, y) at timestep t, or None if invalid."""
        if not self.is_valid_at(t):
            return None
        return (float(self.x[t]), float(self.y[t]))

    def state_at(self, t: int) -> Optional[Dict[str, Any]]:
        """Get state at timestep t as dict, or None if invalid."""
        if not self.is_valid_at(t):
            return None
        return {
            "x": float(self.x[t]),
            "y": float(self.y[t]),
            "yaw": float(self.yaw[t]),
            "speed": float(self.speed[t]),
        }

    def get_acceleration(self, dt: float = DEFAULT_DT_S) -> np.ndarray:
        """Acceleration via finite differences (cached)."""
        if self._acceleration is None or self._cached_dt != dt:
            self._invalidate_cache(dt)
            acc = np.zeros_like(self.speed)
            acc[1:] = (self.speed[1:] - self.speed[:-1]) / dt
            acc[0] = acc[1] if len(acc) > 1 else 0.0
            self._acceleration = acc
        return self._acceleration

    def get_jerk(self, dt: float = DEFAULT_DT_S) -> np.ndarray:
        """Jerk (d(acceleration)/dt) via finite differences (cached)."""
        if self._jerk is None or self._cached_dt != dt:
            acc = self.get_acceleration(dt)
            jerk = np.zeros_like(acc)
            jerk[1:] = (acc[1:] - acc[:-1]) / dt
            jerk[0] = jerk[1] if len(jerk) > 1 else 0.0
            self._jerk = jerk
        return self._jerk

    def get_yaw_rate(self, dt: float = DEFAULT_DT_S) -> np.ndarray:
        """Yaw rate via finite differences with wraparound handling (cached)."""
        if self._yaw_rate is None or self._cached_dt != dt:
            yaw_rate = np.zeros_like(self.yaw)
            yaw_diff = np.diff(self.yaw)
            # Handle angle wraparound
            yaw_diff = np.arctan2(np.sin(yaw_diff), np.cos(yaw_diff))
            yaw_rate[1:] = yaw_diff / dt
            yaw_rate[0] = yaw_rate[1] if len(yaw_rate) > 1 else 0.0
            self._yaw_rate = yaw_rate
        return self._yaw_rate

    def get_lateral_acceleration(self, dt: float = DEFAULT_DT_S) -> np.ndarray:
        """Lateral acceleration (speed * yaw_rate), cached."""
        if self._lateral_acceleration is None or self._cached_dt != dt:
            yaw_rate = self.get_yaw_rate(dt)
            self._lateral_acceleration = self.speed * yaw_rate
        return self._lateral_acceleration

    def _invalidate_cache(self, new_dt: float) -> None:
        """Invalidate cached kinematic properties when dt changes."""
        self._acceleration = None
        self._jerk = None
        self._yaw_rate = None
        self._lateral_acceleration = None
        self._cached_dt = new_dt


@dataclass
class Agent:
    """
    Other traffic participant with full type support.

    Supports vehicles, pedestrians, and cyclists with proper
    VRU (Vulnerable Road User) detection.
    """

    id: int
    type: str  # "vehicle", "pedestrian", "cyclist"
    x: np.ndarray
    y: np.ndarray
    yaw: np.ndarray
    speed: np.ndarray
    length: float
    width: float
    valid: Optional[np.ndarray] = None

    def __post_init__(self):
        """Ensure arrays and create default validity mask."""
        self.x = np.atleast_1d(np.asarray(self.x, dtype=float))
        self.y = np.atleast_1d(np.asarray(self.y, dtype=float))
        self.yaw = np.atleast_1d(np.asarray(self.yaw, dtype=float))
        self.speed = np.atleast_1d(np.asarray(self.speed, dtype=float))

        # Validate array lengths are consistent
        n = len(self.x)
        if len(self.y) != n or len(self.yaw) != n or len(self.speed) != n:
            raise ValueError(
                f"Agent(id={self.id}) array length mismatch: "
                f"x={len(self.x)}, y={len(self.y)}, "
                f"yaw={len(self.yaw)}, speed={len(self.speed)}"
            )

        if self.valid is None:
            self.valid = ~(np.isnan(self.x) | np.isnan(self.y))
        else:
            self.valid = np.atleast_1d(np.asarray(self.valid, dtype=bool))

    def __len__(self) -> int:
        return len(self.x)

    @property
    def n_valid(self) -> int:
        """Number of valid timesteps."""
        return int(np.sum(self.valid))

    @property
    def is_vehicle(self) -> bool:
        return self.type.lower() in ("vehicle", "car", "truck", "bus")

    @property
    def is_pedestrian(self) -> bool:
        return self.type.lower() in ("pedestrian", "ped")

    @property
    def is_cyclist(self) -> bool:
        return self.type.lower() in ("cyclist", "bicycle", "bike", "motorcyclist")

    @property
    def is_vru(self) -> bool:
        """Is this a Vulnerable Road User (pedestrian or cyclist)?"""
        return self.is_pedestrian or self.is_cyclist

    def is_valid_at(self, t: int) -> bool:
        """Check if agent has valid state at timestep t."""
        if t < 0 or t >= len(self.valid):
            return False
        return bool(self.valid[t])

    def position_at(self, t: int) -> Optional[Tuple[float, float]]:
        """Get (x, y) at timestep t, or None if invalid."""
        if not self.is_valid_at(t):
            return None
        return (float(self.x[t]), float(self.y[t]))

    def state_at(self, t: int) -> Optional[Dict[str, Any]]:
        """Get state at timestep t as dict, or None if invalid."""
        if not self.is_valid_at(t):
            return None
        return {
            "x": float(self.x[t]),
            "y": float(self.y[t]),
            "yaw": float(self.yaw[t]),
            "speed": float(self.speed[t]),
            "type": self.type,
        }

    def distance_to(self, x: float, y: float, t: int) -> Optional[float]:
        """Get distance from agent to point at timestep t."""
        if not self.is_valid_at(t):
            return None
        return float(np.sqrt((self.x[t] - x) ** 2 + (self.y[t] - y) ** 2))


@dataclass
class MapContext:
    """
    Static map features for the scenario.

    Contains lane geometry, stop lines, crosswalks, road boundaries,
    and other map elements needed for rule evaluation.
    """

    lane_center_xy: np.ndarray  # (N, 2) lane centerline points
    lane_id: Optional[int] = None  # ID of ego's primary lane
    stopline_xy: Optional[np.ndarray] = None  # (M, 2) stopline positions
    crosswalk_polys: List[np.ndarray] = field(default_factory=list)
    road_edges: List[Dict[str, Any]] = field(default_factory=list)
    stop_signs: List[Dict[str, Any]] = field(default_factory=list)
    speed_limit: Optional[np.ndarray] = None  # Per-point speed limits (m/s)
    speed_limit_mask: Optional[np.ndarray] = None

    # Additional lane info for multi-lane scenarios
    all_lanes: List[Dict[str, Any]] = field(default_factory=list)

    # Optional zone/boundary data used by regulatory rules
    construction_zone_xy: Optional[np.ndarray] = (
        None  # (K, 2) construction zone polygon
    )
    school_zone_xy: Optional[np.ndarray] = None  # (K, 2) school zone polygon
    boundary_types: Optional[np.ndarray] = None  # Per-point boundary type codes

    def __post_init__(self):
        """Ensure arrays are properly shaped."""
        if self.lane_center_xy is None:
            self.lane_center_xy = np.empty((0, 2))
        else:
            self.lane_center_xy = np.atleast_2d(np.asarray(self.lane_center_xy))

        if self.stopline_xy is None:
            self.stopline_xy = np.empty((0, 2))
        else:
            self.stopline_xy = np.atleast_2d(np.asarray(self.stopline_xy))

    @property
    def has_lane_geometry(self) -> bool:
        return self.lane_center_xy.shape[0] >= 2

    @property
    def has_stoplines(self) -> bool:
        return self.stopline_xy.shape[0] > 0

    @property
    def has_crosswalks(self) -> bool:
        return len(self.crosswalk_polys) > 0

    @property
    def has_road_edges(self) -> bool:
        return len(self.road_edges) > 0

    @property
    def has_stop_signs(self) -> bool:
        return len(self.stop_signs) > 0

    @property
    def has_speed_limits(self) -> bool:
        return self.speed_limit is not None and len(self.speed_limit) > 0


@dataclass
class MapSignals:
    """
    Dynamic traffic signal states over time.

    Contains per-frame signal states and the lane ID used for association.
    """

    signal_state: np.ndarray  # (T,) signal state per frame
    ego_lane_id: Optional[int] = None  # Lane ID used for signal association
    confidence: Optional[np.ndarray] = None

    def __post_init__(self):
        self.signal_state = np.atleast_1d(np.asarray(self.signal_state, dtype=int))
        if self.confidence is None:
            self.confidence = np.ones(len(self.signal_state))

    def __len__(self) -> int:
        return len(self.signal_state)

    def is_red_at(self, t: int) -> bool:
        """Check if signal is red (stop) at timestep t."""
        if t < 0 or t >= len(self.signal_state):
            return False
        return int(self.signal_state[t]) in RED_SIGNAL_STATES

    def is_yellow_at(self, t: int) -> bool:
        """Check if signal is yellow (caution) at timestep t."""
        if t < 0 or t >= len(self.signal_state):
            return False
        return int(self.signal_state[t]) in YELLOW_SIGNAL_STATES

    def is_green_at(self, t: int) -> bool:
        """Check if signal is green (go) at timestep t."""
        if t < 0 or t >= len(self.signal_state):
            return False
        return int(self.signal_state[t]) in GREEN_SIGNAL_STATES

    def state_at(self, t: int) -> int:
        """Get signal state at timestep t."""
        if t < 0 or t >= len(self.signal_state):
            return SIGNAL_STATE_UNKNOWN
        return int(self.signal_state[t])


@dataclass
class ScenarioContext:
    """
    Complete scenario container for rule evaluation.

    Bundles ego state, other agents, map context, and signals
    into a single object for rule evaluators.
    """

    scenario_id: str
    ego: EgoState
    agents: List[Agent]
    map_context: MapContext
    signals: Optional[MapSignals] = None
    dt: float = DEFAULT_DT_S
    # Window metadata (set when slicing)
    window_start_ts: Optional[float] = None
    window_size: Optional[int] = None
    dataset_kind: str = "motion_scenario"
    trajectory_injection: Optional[Any] = field(default=None, repr=False, compare=False)

    @property
    def map(self) -> MapContext:
        """Alias for map_context for compatibility."""
        return self.map_context

    @property
    def n_frames(self) -> int:
        """Total number of frames in scenario."""
        return len(self.ego.x)

    @property
    def vehicles(self) -> List[Agent]:
        """List of vehicle agents."""
        return [a for a in self.agents if a.is_vehicle]

    @property
    def pedestrians(self) -> List[Agent]:
        """List of pedestrian agents."""
        return [a for a in self.agents if a.is_pedestrian]

    @property
    def cyclists(self) -> List[Agent]:
        """List of cyclist agents."""
        return [a for a in self.agents if a.is_cyclist]

    @property
    def vrus(self) -> List[Agent]:
        """List of vulnerable road users (pedestrians + cyclists)."""
        return [a for a in self.agents if a.is_vru]

    @property
    def has_signals(self) -> bool:
        """Check if scenario has traffic signal data."""
        return self.signals is not None

    def agents_of_type(self, agent_type: str) -> List[Agent]:
        """Get agents of a specific type."""
        return [a for a in self.agents if a.type.lower() == agent_type.lower()]

    def valid_agents_at(self, t: int) -> List[Agent]:
        """Get agents with valid state at timestep t."""
        return [a for a in self.agents if a.is_valid_at(t)]
