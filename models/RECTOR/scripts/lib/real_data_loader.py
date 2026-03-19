#!/usr/bin/env python3
"""
Real Data Loader for RECTOR Testing and Benchmarking.

Provides utilities to load real Waymo scenarios from augmented TFRecords.
This replaces synthetic data generation across the codebase.

Usage:
    from real_data_loader import RealDataLoader, get_default_loader

    loader = get_default_loader()
    scenario = loader.get_random_scenario()
    # scenario contains: ego_state, other_agents, lane, metadata
"""

import os
import glob
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterator
import numpy as np

# TensorFlow for reading TFRecords
import tensorflow as tf

tf.config.set_visible_devices([], "GPU")  # CPU only

# Waymo protos
try:
    from waymo_open_dataset.protos import scenario_pb2

    WAYMO_AVAILABLE = True
except ImportError:
    scenario_pb2 = None
    WAYMO_AVAILABLE = False


# Default paths - use raw data for full 91 timesteps
DEFAULT_DATA_DIR = (
    "/workspace/data/WOMD/datasets/waymo_open_dataset/motion_v_1_3_0/raw/scenario"
)
TRAIN_DIR = os.path.join(DEFAULT_DATA_DIR, "training_interactive")
VAL_DIR = os.path.join(DEFAULT_DATA_DIR, "validation_interactive")
TEST_DIR = os.path.join(DEFAULT_DATA_DIR, "testing_interactive")

# Augmented data (only has 11 timesteps - history only)
AUGMENTED_DATA_DIR = "/workspace/data/WOMD/datasets/waymo_open_dataset/motion_v_1_3_0/processed/augmented/scenario"


@dataclass
class AgentState:
    """Agent state from real data."""

    x: float
    y: float
    heading: float
    velocity_x: float
    velocity_y: float
    agent_id: int
    length: float = 4.5
    width: float = 2.0
    agent_type: str = "VEHICLE"


@dataclass
class LaneInfo:
    """Lane information from map."""

    centerline: np.ndarray
    speed_limit: float = 25.0
    lane_type: str = "LANE"


@dataclass
class RealScenario:
    """A real scenario loaded from TFRecord."""

    scenario_id: str
    ego_state: AgentState
    other_agents: List[AgentState]
    lane: LaneInfo
    ego_history: np.ndarray  # [11, 4]
    ego_future: np.ndarray  # [80, 4]
    agent_histories: np.ndarray  # [N, 11, 4]
    rule_applicability: np.ndarray  # [28]
    rule_violations: np.ndarray  # [28]
    metadata: Dict


class RealDataLoader:
    """
    Loads real scenarios from Waymo raw TFRecords.

    Uses raw data which has full 91 timesteps (11 history + 80 future).
    Replaces synthetic data generation for:
    - Benchmarks
    - Visualization demos
    - Integration tests
    - Parameter tuning
    """

    def __init__(
        self,
        data_dir: str = DEFAULT_DATA_DIR,
        split: str = "training_interactive",
        max_files: int = 10,
        cache_size: int = 100,
        seed: int = 42,
    ):
        """
        Initialize loader.

        Args:
            data_dir: Base directory for raw data
            split: Which split to use (training_interactive, testing_interactive, etc.)
            max_files: Maximum number of TFRecord files to load (for speed)
            cache_size: Number of scenarios to cache in memory
            seed: Random seed for reproducibility
        """
        self.data_dir = data_dir
        self.split = split
        self.max_files = max_files
        self.cache_size = cache_size
        self.rng = np.random.default_rng(seed)

        # Get TFRecord files
        split_dir = os.path.join(data_dir, split)
        pattern = os.path.join(split_dir, "*.tfrecord*")
        all_files = sorted(glob.glob(pattern))

        if len(all_files) == 0:
            raise FileNotFoundError(f"No TFRecord files found in {split_dir}")

        # Limit files for faster loading
        self.tfrecord_files = all_files[:max_files]
        print(f"RealDataLoader: Using {len(self.tfrecord_files)} files from {split}")

        # Cache
        self._cache: List[RealScenario] = []
        self._loaded = False

    def _load_cache(self):
        """Load scenarios into cache."""
        if self._loaded:
            return

        print(f"Loading {self.cache_size} scenarios from real data...")

        dataset = tf.data.TFRecordDataset(
            self.tfrecord_files,
            num_parallel_reads=min(4, len(self.tfrecord_files)),
        )

        count = 0
        for raw_record in dataset:
            if count >= self.cache_size:
                break

            try:
                scenario = self._parse_record(raw_record)
                if scenario is not None:
                    self._cache.append(scenario)
                    count += 1
            except Exception as e:
                continue

        print(f"  Loaded {len(self._cache)} scenarios")
        self._loaded = True

    def _parse_record(self, raw_record) -> Optional[RealScenario]:
        """Parse a single TFRecord into RealScenario."""
        if not WAYMO_AVAILABLE:
            return None

        # Raw data is just the serialized proto
        scenario = scenario_pb2.Scenario()
        scenario.ParseFromString(raw_record.numpy())

        scenario_id = scenario.scenario_id

        # No rule annotations in raw data - use zeros
        rule_applicability = np.zeros(28, dtype=np.float32)
        rule_violations = np.zeros(28, dtype=np.float32)

        # Extract data
        return self._extract_scenario(
            scenario, scenario_id, rule_applicability, rule_violations
        )

    def _extract_scenario(
        self,
        scenario,
        scenario_id: str,
        rule_applicability: np.ndarray,
        rule_violations: np.ndarray,
    ) -> Optional[RealScenario]:
        """Extract scenario data from proto."""
        tracks = list(scenario.tracks)
        sdc_idx = scenario.sdc_track_index

        if len(tracks) == 0 or sdc_idx >= len(tracks):
            return None

        sdc_track = tracks[sdc_idx]
        # Use the scenario's declared current timestep instead of hardcoding 10
        current_ts = scenario.current_time_index
        history_length = current_ts + 1  # frames 0..current_ts
        future_length = 80  # 8 seconds at 10 Hz

        # Check validity
        if len(sdc_track.states) < current_ts + future_length:
            return None

        if not sdc_track.states[current_ts].valid:
            return None

        # Current ego state
        state = sdc_track.states[current_ts]
        ego_state = AgentState(
            x=state.center_x,
            y=state.center_y,
            heading=state.heading,
            velocity_x=state.velocity_x,
            velocity_y=state.velocity_y,
            agent_id=0,
            length=state.length,
            width=state.width,
        )

        # Ego history [history_length, 4]
        ego_history = np.zeros((history_length, 4), dtype=np.float32)
        for t in range(history_length):
            if t < len(sdc_track.states) and sdc_track.states[t].valid:
                s = sdc_track.states[t]
                ego_history[t] = [
                    s.center_x,
                    s.center_y,
                    s.heading,
                    np.sqrt(s.velocity_x**2 + s.velocity_y**2),
                ]

        # Ego future [future_length, 4]
        ego_future = np.zeros((future_length, 4), dtype=np.float32)
        for t in range(future_length):
            ts_idx = current_ts + 1 + t
            if ts_idx < len(sdc_track.states) and sdc_track.states[ts_idx].valid:
                s = sdc_track.states[ts_idx]
                ego_future[t] = [
                    s.center_x,
                    s.center_y,
                    s.heading,
                    np.sqrt(s.velocity_x**2 + s.velocity_y**2),
                ]

        # Other agents
        other_agents = []
        agent_histories = []

        for i, track in enumerate(tracks):
            if i == sdc_idx:
                continue
            if len(other_agents) >= 30:  # Limit
                break

            if current_ts < len(track.states) and track.states[current_ts].valid:
                s = track.states[current_ts]
                other_agents.append(
                    AgentState(
                        x=s.center_x,
                        y=s.center_y,
                        heading=s.heading,
                        velocity_x=s.velocity_x,
                        velocity_y=s.velocity_y,
                        agent_id=i,
                        length=s.length,
                        width=s.width,
                    )
                )

                # History
                hist = np.zeros((11, 4), dtype=np.float32)
                for t in range(11):
                    if t < len(track.states) and track.states[t].valid:
                        ts = track.states[t]
                        hist[t] = [
                            ts.center_x,
                            ts.center_y,
                            ts.heading,
                            np.sqrt(ts.velocity_x**2 + ts.velocity_y**2),
                        ]
                agent_histories.append(hist)

        agent_histories = (
            np.array(agent_histories) if agent_histories else np.zeros((0, 11, 4))
        )

        # Extract lane
        lane = self._extract_lane(scenario.map_features, ego_state.x, ego_state.y)

        # Metadata
        metadata = {
            "scenario_id": scenario_id,
            "num_agents": len(other_agents),
            "ego_speed": np.sqrt(ego_state.velocity_x**2 + ego_state.velocity_y**2),
            "timestamps_sec": (
                scenario.timestamps_seconds[:91] if scenario.timestamps_seconds else []
            ),
        }

        return RealScenario(
            scenario_id=scenario_id,
            ego_state=ego_state,
            other_agents=other_agents,
            lane=lane,
            ego_history=ego_history,
            ego_future=ego_future,
            agent_histories=agent_histories,
            rule_applicability=rule_applicability,
            rule_violations=rule_violations,
            metadata=metadata,
        )

    def _extract_lane(self, map_features, ego_x: float, ego_y: float) -> LaneInfo:
        """Extract nearest lane to ego."""
        best_lane = None
        best_dist = float("inf")

        for feature in map_features:
            if feature.HasField("lane"):
                polyline = list(feature.lane.polyline)
                if len(polyline) < 2:
                    continue

                # Sample centerline
                points = np.array([[p.x, p.y] for p in polyline])

                # Distance to ego
                dists = np.sqrt(
                    (points[:, 0] - ego_x) ** 2 + (points[:, 1] - ego_y) ** 2
                )
                min_dist = dists.min()

                if min_dist < best_dist:
                    best_dist = min_dist
                    # Resample to fixed size
                    if len(points) > 50:
                        indices = np.linspace(0, len(points) - 1, 50).astype(int)
                        best_lane = points[indices]
                    else:
                        best_lane = points

        if best_lane is None:
            # Fallback: straight lane from ego
            best_lane = np.array([[ego_x + i * 2, ego_y] for i in range(50)])

        return LaneInfo(centerline=best_lane, speed_limit=25.0)

    def get_random_scenario(self) -> RealScenario:
        """Get a random scenario from cache."""
        self._load_cache()
        return self.rng.choice(self._cache)

    def get_scenarios(self, n: int) -> List[RealScenario]:
        """Get n random scenarios."""
        self._load_cache()
        return list(
            self.rng.choice(self._cache, size=min(n, len(self._cache)), replace=False)
        )

    def iter_scenarios(self, n: Optional[int] = None) -> Iterator[RealScenario]:
        """Iterate through scenarios."""
        self._load_cache()
        scenarios = self._cache[:n] if n else self._cache
        for s in scenarios:
            yield s

    def __len__(self) -> int:
        self._load_cache()
        return len(self._cache)


# Global singleton
_default_loader: Optional[RealDataLoader] = None


def get_default_loader() -> RealDataLoader:
    """Get or create default loader."""
    global _default_loader
    if _default_loader is None:
        _default_loader = RealDataLoader()
    return _default_loader


def get_test_scenario() -> RealScenario:
    """Get a single test scenario for quick testing."""
    return get_default_loader().get_random_scenario()


def get_test_scenarios(n: int = 10) -> List[RealScenario]:
    """Get multiple test scenarios."""
    return get_default_loader().get_scenarios(n)


# Compatibility functions - return data in format expected by existing code
def create_scenario_from_real(
    num_agents: int = 10,
    horizon: int = 80,
) -> Tuple[AgentState, List[AgentState], LaneInfo]:
    """
    Create scenario data from real TFRecords.

    Drop-in replacement for create_synthetic_scenario().
    """
    scenario = get_test_scenario()

    # Limit agents if requested
    agents = scenario.other_agents[:num_agents]

    return scenario.ego_state, agents, scenario.lane


def create_test_scenarios_from_real(num_scenarios: int = 10) -> List[Dict]:
    """
    Create test scenarios from real data.

    Drop-in replacement for create_test_scenarios().
    """
    scenarios = get_test_scenarios(num_scenarios)

    result = []
    for s in scenarios:
        result.append(
            {
                "ego_state": s.ego_state,
                "other_agents": s.other_agents,
                "lane": s.lane,
                "metadata": s.metadata,
            }
        )

    return result
