"""
Waymo Open Motion Dataset adapter with full agent type support.

This module provides data loading from Waymo Motion Dataset TFRecord files,
extracting vehicles, pedestrians, and cyclists with proper dimensions
and validity handling.

CRITICAL FEATURES:
1. Extracts ALL agent types (vehicles, pedestrians, cyclists)
2. Uses actual per-track dimensions from ObjectState
3. Respects validity flags (marks invalid as NaN)
4. Associates traffic signals to ego's lane
5. Extracts road edges and stop signs from map features
"""

import logging
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

# Import proto only when available
try:
    from waymo_open_dataset.protos import scenario_pb2

    WAYMO_AVAILABLE = True
except ImportError:
    WAYMO_AVAILABLE = False
    log.warning("Waymo Open Dataset not installed, using mock proto")


from ..core.context import Agent, EgoState, MapContext, MapSignals, ScenarioContext
from ..utils.constants import (
    AGENT_TYPE_MAP,
    DEFAULT_DIMENSIONS,
    DEFAULT_DT_S,
    DEFAULT_EGO_LENGTH,
    DEFAULT_EGO_WIDTH,
    SIGNAL_STATE_ARROW_CAUTION,
    SIGNAL_STATE_ARROW_GO,
    SIGNAL_STATE_ARROW_STOP,
    SIGNAL_STATE_CAUTION,
    SIGNAL_STATE_FLASHING_CAUTION,
    SIGNAL_STATE_FLASHING_STOP,
    SIGNAL_STATE_GO,
    SIGNAL_STATE_STOP,
    SIGNAL_STATE_UNKNOWN,
    WAYMO_TYPE_CYCLIST,
    WAYMO_TYPE_PEDESTRIAN,
    WAYMO_TYPE_VEHICLE,
)


def _get_agent_type(object_type: int) -> str:
    """Convert Waymo object type to string."""
    return AGENT_TYPE_MAP.get(object_type, "vehicle")


def _get_default_dimensions(agent_type: str) -> Tuple[float, float]:
    """Get default (length, width) for agent type."""
    return DEFAULT_DIMENSIONS.get(agent_type, (4.5, 1.8))


class MotionScenarioReader:
    """
    Reads Waymo Motion Dataset scenarios with full agent type support.

    Usage:
        reader = MotionScenarioReader()
        for context in reader.read_tfrecord("path/to/file.tfrecord"):
            print(f"Scenario {context.scenario_id} has {len(context.agents)} agents")
    """

    def __init__(self, dt: float = DEFAULT_DT_S):
        """
        Initialize the reader.

        Args:
            dt: Time step in seconds (Waymo uses 0.1s = 10Hz)
        """
        self.dt = dt

    def read_tfrecord(self, path: str) -> Iterator[ScenarioContext]:
        """
        Read all scenarios from a TFRecord file.

        Args:
            path: Path to TFRecord file

        Yields:
            ScenarioContext for each scenario in the file
        """
        if not WAYMO_AVAILABLE:
            log.error("Waymo Open Dataset not installed")
            return

        import tensorflow as tf

        dataset = tf.data.TFRecordDataset(path, compression_type="")
        skipped_records = 0

        for record in dataset:
            record_bytes = record.numpy()
            scenario = self._parse_record_to_scenario(record_bytes)
            if scenario is None:
                skipped_records += 1
                continue
            yield self.parse_scenario(scenario)

        if skipped_records > 0:
            log.warning(
                "Skipped %d malformed/unsupported records from %s",
                skipped_records,
                path,
            )

    def _parse_record_to_scenario(self, record_bytes: bytes):
        """Parse TFRecord bytes into a Scenario proto.

        Supports:
        1) Raw Scenario records (serialized scenario_pb2.Scenario)
        2) Augmented records (serialized tf.train.Example with scenario/proto)
        """
        scenario = scenario_pb2.Scenario()

        try:
            scenario.ParseFromString(record_bytes)
            return scenario
        except Exception:
            pass

        try:
            import tensorflow as tf

            example = tf.train.Example()
            example.ParseFromString(record_bytes)
            features = example.features.feature

            if "scenario/proto" not in features:
                return None

            proto_values = features["scenario/proto"].bytes_list.value
            if not proto_values:
                return None

            scenario.ParseFromString(proto_values[0])
            return scenario
        except Exception:
            return None

    def parse_scenario(self, scenario) -> ScenarioContext:
        """
        Parse a Waymo Scenario proto into ScenarioContext.

        Args:
            scenario: Waymo Scenario proto

        Returns:
            ScenarioContext with all extracted data
        """
        scenario_id = scenario.scenario_id
        n_frames = len(scenario.timestamps_seconds)

        # Find ego (sdc) track
        ego_track = None
        ego_idx = scenario.sdc_track_index
        if 0 <= ego_idx < len(scenario.tracks):
            ego_track = scenario.tracks[ego_idx]

        if ego_track is None:
            # Fallback: use first track
            ego_track = scenario.tracks[0] if scenario.tracks else None

        # Extract ego state
        ego = self._extract_ego(ego_track, n_frames)

        # Extract all other agents (ALL types)
        agents = self._extract_agents(scenario.tracks, ego_idx, n_frames)

        # Extract map context
        map_context = self._extract_map(scenario.map_features, ego)

        # Extract signals
        signals = self._extract_signals(
            scenario.dynamic_map_states, n_frames, map_context.lane_id
        )

        return ScenarioContext(
            scenario_id=scenario_id,
            ego=ego,
            agents=agents,
            map_context=map_context,
            signals=signals,
            dt=self.dt,
        )

    def _extract_ego(self, track, n_frames: int) -> EgoState:
        """Extract ego state from track proto."""
        x = np.full(n_frames, np.nan)
        y = np.full(n_frames, np.nan)
        yaw = np.full(n_frames, np.nan)
        speed = np.full(n_frames, np.nan)
        valid = np.zeros(n_frames, dtype=bool)

        length = DEFAULT_EGO_LENGTH
        width = DEFAULT_EGO_WIDTH

        if track is not None:
            for t, state in enumerate(track.states):
                if state.valid:
                    if 0 <= t < n_frames:
                        x[t] = state.center_x
                        y[t] = state.center_y
                        yaw[t] = state.heading
                        vx = state.velocity_x if hasattr(state, "velocity_x") else 0
                        vy = state.velocity_y if hasattr(state, "velocity_y") else 0
                        speed[t] = np.sqrt(vx**2 + vy**2)
                        valid[t] = True

                        # Use actual dimensions if available
                        if hasattr(state, "length") and state.length > 0:
                            length = state.length
                        if hasattr(state, "width") and state.width > 0:
                            width = state.width

        return EgoState(
            x=x, y=y, yaw=yaw, speed=speed, length=length, width=width, valid=valid
        )

    def _extract_agents(self, tracks, ego_idx: int, n_frames: int) -> List[Agent]:
        """
        Extract ALL agents (vehicles, pedestrians, cyclists).

        This is CRITICAL: the original code only extracted vehicles.
        """
        agents = []

        for i, track in enumerate(tracks):
            if i == ego_idx:
                continue

            # Get agent type - MUST include all types
            obj_type = track.object_type
            if obj_type not in AGENT_TYPE_MAP:
                continue  # Skip unknown types

            agent_type = AGENT_TYPE_MAP[obj_type]

            x = np.full(n_frames, np.nan)
            y = np.full(n_frames, np.nan)
            yaw = np.full(n_frames, np.nan)
            speed = np.full(n_frames, np.nan)
            valid = np.zeros(n_frames, dtype=bool)

            # Default dimensions for this type
            default_len, default_wid = _get_default_dimensions(agent_type)
            length = default_len
            width = default_wid

            for t, state in enumerate(track.states):
                if state.valid:
                    if 0 <= t < n_frames:
                        x[t] = state.center_x
                        y[t] = state.center_y
                        yaw[t] = state.heading
                        vx = getattr(state, "velocity_x", 0)
                        vy = getattr(state, "velocity_y", 0)
                        speed[t] = np.sqrt(vx**2 + vy**2)
                        valid[t] = True

                        # Use actual dimensions from proto
                        if hasattr(state, "length") and state.length > 0:
                            length = state.length
                        if hasattr(state, "width") and state.width > 0:
                            width = state.width

            if np.any(valid):
                agents.append(
                    Agent(
                        id=int(track.id),
                        type=agent_type,
                        x=x,
                        y=y,
                        yaw=yaw,
                        speed=speed,
                        length=length,
                        width=width,
                        valid=valid,
                    )
                )

        return agents

    def _extract_map(self, map_features, ego: EgoState) -> MapContext:
        """Extract map features from proto."""
        all_lanes = []
        stoplines = []
        crosswalk_polys = []
        road_edges = []
        stop_signs = []

        # Find closest lane to ego start position
        ego_start_x = ego.x[ego.valid][0] if ego.n_valid > 0 else 0
        ego_start_y = ego.y[ego.valid][0] if ego.n_valid > 0 else 0
        best_lane_id = None
        best_lane_dist = float("inf")
        best_lane_points = np.empty((0, 2))

        for feature in map_features:
            if feature.HasField("lane"):
                lane = feature.lane
                points = np.array([[p.x, p.y] for p in lane.polyline])
                if len(points) > 0:
                    all_lanes.append(
                        {
                            "id": feature.id,
                            "points": points,
                            "type": lane.type if hasattr(lane, "type") else 0,
                        }
                    )

                    # Check distance to ego start
                    dists = np.sqrt(
                        (points[:, 0] - ego_start_x) ** 2
                        + (points[:, 1] - ego_start_y) ** 2
                    )
                    min_dist = np.min(dists)
                    if min_dist < best_lane_dist:
                        best_lane_dist = min_dist
                        best_lane_id = feature.id
                        best_lane_points = points

            elif feature.HasField("stop_sign"):
                stop_sign = feature.stop_sign
                if hasattr(stop_sign, "position"):
                    stop_signs.append(
                        {
                            "id": feature.id,
                            "x": stop_sign.position.x,
                            "y": stop_sign.position.y,
                        }
                    )

            elif feature.HasField("crosswalk"):
                crosswalk = feature.crosswalk
                points = np.array([[p.x, p.y] for p in crosswalk.polygon])
                if len(points) >= 3:
                    crosswalk_polys.append(points)

            elif feature.HasField("road_edge"):
                edge = feature.road_edge
                points = np.array([[p.x, p.y] for p in edge.polyline])
                if len(points) > 0:
                    road_edges.append(
                        {
                            "id": feature.id,
                            "points": points,
                            "type": edge.type if hasattr(edge, "type") else 0,
                        }
                    )

            elif feature.HasField("road_line"):
                road_line = feature.road_line
                if hasattr(road_line, "type"):
                    # Stop lines are typically a type of road line
                    if "STOP" in str(road_line.type).upper():
                        for p in road_line.polyline:
                            stoplines.append([p.x, p.y])

        stopline_xy = np.array(stoplines) if stoplines else np.empty((0, 2))

        return MapContext(
            lane_center_xy=best_lane_points,
            lane_id=best_lane_id,
            all_lanes=all_lanes,
            stopline_xy=stopline_xy,
            crosswalk_polys=crosswalk_polys,
            road_edges=road_edges,
            stop_signs=stop_signs,
        )

    def _extract_signals(
        self, dynamic_states, n_frames: int, ego_lane_id: Optional[int]
    ) -> MapSignals:
        """
        Extract traffic signal states over time.

        Associates signals to ego's lane, not the maximum of all signals.
        """
        signal_state = np.full(n_frames, SIGNAL_STATE_UNKNOWN, dtype=int)

        for t, frame_state in enumerate(dynamic_states):
            if t < 0 or t >= n_frames:
                continue

            # Find signal for ego's lane
            for sig in frame_state.lane_states:
                if ego_lane_id is not None and sig.lane != ego_lane_id:
                    continue  # Only use ego's lane signal

                # Map Waymo signal state
                state_map = {
                    0: SIGNAL_STATE_UNKNOWN,
                    1: SIGNAL_STATE_ARROW_STOP,
                    2: SIGNAL_STATE_ARROW_CAUTION,
                    3: SIGNAL_STATE_ARROW_GO,
                    4: SIGNAL_STATE_STOP,
                    5: SIGNAL_STATE_CAUTION,
                    6: SIGNAL_STATE_GO,
                    7: SIGNAL_STATE_FLASHING_STOP,
                    8: SIGNAL_STATE_FLASHING_CAUTION,
                }
                signal_state[t] = state_map.get(sig.state, SIGNAL_STATE_UNKNOWN)
                break  # Use first matching signal

        return MapSignals(
            signal_state=signal_state,
            ego_lane_id=ego_lane_id,
        )

    def iter_windows(
        self,
        path: str,
        window_size_s: float = 8.0,
        stride_s: float = 2.0,
    ) -> Iterator[Tuple[ScenarioContext, int, int, int]]:
        """
        Iterate over sliding windows from TFRecord file.

        This method provides memory-efficient streaming over windows
        without loading entire scenarios into memory at once.

        Args:
            path: Path to TFRecord file
            window_size_s: Window size in seconds
            stride_s: Stride between windows in seconds

        Yields:
            Tuple of (windowed_context, window_idx, start_idx, end_idx)
        """
        from ..pipeline.rule_executor import extract_window
        from ..pipeline.window_scheduler import make_windows_timed

        for scenario_ctx in self.read_tfrecord(path):
            n_frames = scenario_ctx.n_frames
            dt = scenario_ctx.dt or self.dt

            # Generate windows
            specs = make_windows_timed(
                T=n_frames,
                dt=dt,
                window_size_s=window_size_s,
                stride_s=stride_s,
            )

            for window_idx, spec in enumerate(specs):
                windowed_ctx = extract_window(
                    scenario_ctx, spec.start_idx, spec.end_idx
                )
                yield (windowed_ctx, window_idx, spec.start_idx, spec.end_idx)

    def iter_scenarios_from_glob(
        self,
        glob_pattern: str,
    ) -> Iterator[ScenarioContext]:
        """
        Iterate over scenarios from multiple TFRecord files.

        Args:
            glob_pattern: Glob pattern for TFRecord files

        Yields:
            ScenarioContext for each scenario
        """
        import glob as glob_module

        files = sorted(glob_module.glob(glob_pattern))
        for fpath in files:
            try:
                yield from self.read_tfrecord(fpath)
            except Exception as e:
                log.warning(f"Failed to read {fpath}: {e}")
                continue


def create_scenario_from_arrays(
    scenario_id: str,
    ego_x: np.ndarray,
    ego_y: np.ndarray,
    ego_yaw: np.ndarray,
    ego_speed: np.ndarray,
    agent_data: Optional[List[Dict[str, Any]]] = None,
    dt: float = DEFAULT_DT_S,
) -> ScenarioContext:
    """
    Create a ScenarioContext from numpy arrays.

    Useful for testing or synthetic scenario generation.

    Args:
        scenario_id: Unique scenario identifier
        ego_x, ego_y, ego_yaw, ego_speed: Ego trajectory arrays
        agent_data: List of agent dicts with keys:
            id, type, x, y, yaw, speed, length, width
        dt: Time step in seconds

    Returns:
        ScenarioContext with the provided data
    """
    ego = EgoState(
        x=ego_x,
        y=ego_y,
        yaw=ego_yaw,
        speed=ego_speed,
        length=DEFAULT_EGO_LENGTH,
        width=DEFAULT_EGO_WIDTH,
    )

    agents = []
    if agent_data:
        for data in agent_data:
            agent = Agent(
                id=data.get("id", 0),
                type=data.get("type", "vehicle"),
                x=data["x"],
                y=data["y"],
                yaw=data["yaw"],
                speed=data["speed"],
                length=data.get("length", 4.5),
                width=data.get("width", 1.8),
            )
            agents.append(agent)

    map_context = MapContext(
        lane_center_xy=np.empty((0, 2)),
    )

    return ScenarioContext(
        scenario_id=scenario_id,
        ego=ego,
        agents=agents,
        map_context=map_context,
        dt=dt,
    )
