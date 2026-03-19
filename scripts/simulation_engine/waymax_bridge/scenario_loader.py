"""
Scenario Loader: Loads WOMD scenarios from local TFRecords.

Reads the augmented TFRecords containing serialised Scenario protos
and converts them to Waymax-compatible feature dicts that can be
transformed into SimulatorState objects.

The WOMD data on disk uses the `scenario/proto` key (protobuf-serialised
``Scenario`` messages), which differs from Waymax's expected 70-key
flat TFExample schema. This module bridges that gap.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple

import jax.numpy as jnp
import numpy as np
import tensorflow as tf

from waymo_open_dataset.protos import scenario_pb2
from waymax import datatypes


_DEFAULT_DATA_ROOT = (
    "/workspace/data/WOMD/datasets/waymo_open_dataset"
    "/motion_v_1_3_0/processed/augmented/scenario"
)

_SPLIT_DIRS = {
    "validation": "validation_interactive",
    "training": "training_interactive",
    "testing": "testing_interactive",
}


@dataclass
class ScenarioLoaderConfig:
    """Configuration for scenario loading."""

    data_root: str = _DEFAULT_DATA_ROOT
    split: str = "validation"
    max_objects: int = 128
    max_rg_points: int = 30_000
    max_tl_states: int = 16
    num_past: int = 10  # WOMD convention: 10 past + 1 current + 80 future
    num_current: int = 1
    num_future: int = 80
    shuffle: bool = False
    seed: int = 2026
    max_scenarios: Optional[int] = None


def _pad_or_truncate(arr: np.ndarray, target: int, axis: int = 0) -> np.ndarray:
    """Pad with zeros or truncate *arr* along *axis* to *target* size."""
    current = arr.shape[axis]
    if current >= target:
        slices = [slice(None)] * arr.ndim
        slices[axis] = slice(0, target)
        return arr[tuple(slices)]
    # Pad
    pad_width = [(0, 0)] * arr.ndim
    pad_width[axis] = (0, target - current)
    return np.pad(arr, pad_width, mode="constant", constant_values=0.0)


def scenario_proto_to_simulator_state(
    proto: scenario_pb2.Scenario,
    max_objects: int = 128,
    max_rg_points: int = 30_000,
    max_tl_states: int = 16,
) -> datatypes.SimulatorState:
    """
    Convert a WOMD Scenario proto into a Waymax SimulatorState.

    This builds the JAX arrays by hand, matching the shapes that Waymax's
    ``PlanningAgentEnvironment`` expects.

    Parameters
    ----------
    proto : scenario_pb2.Scenario
        Decoded WOMD scenario protobuf.
    max_objects : int
        Pad/truncate objects dimension to this size.
    max_rg_points : int
        Pad/truncate roadgraph points to this size.
    max_tl_states : int
        Pad/truncate traffic light lanes to this size.

    Returns
    -------
    datatypes.SimulatorState
    """
    T = len(proto.timestamps_seconds)  # 91

    # Build track index ordering: ensure SDC is always included.
    # If SDC index >= max_objects, swap it into slot 0.
    sdc_idx = proto.sdc_track_index
    total_tracks = len(proto.tracks)
    num_objects = min(total_tracks, max_objects)

    track_indices = list(range(min(total_tracks, max_objects)))
    if sdc_idx >= max_objects:
        # Swap SDC into slot 0
        track_indices[0] = sdc_idx
        sdc_slot = 0
    else:
        sdc_slot = sdc_idx

    # ---- Object trajectories ------------------------------------------------
    x = np.zeros((num_objects, T), dtype=np.float32)
    y = np.zeros((num_objects, T), dtype=np.float32)
    z = np.zeros((num_objects, T), dtype=np.float32)
    vel_x = np.zeros((num_objects, T), dtype=np.float32)
    vel_y = np.zeros((num_objects, T), dtype=np.float32)
    yaw = np.zeros((num_objects, T), dtype=np.float32)
    valid = np.zeros((num_objects, T), dtype=np.bool_)
    length = np.zeros((num_objects, T), dtype=np.float32)
    width = np.zeros((num_objects, T), dtype=np.float32)
    height = np.zeros((num_objects, T), dtype=np.float32)
    timestamp_micros = np.zeros((num_objects, T), dtype=np.int64)

    obj_ids = np.zeros(num_objects, dtype=np.int32)
    obj_types = np.zeros(num_objects, dtype=np.int32)
    is_sdc = np.zeros(num_objects, dtype=np.bool_)
    is_valid_obj = np.zeros(num_objects, dtype=np.bool_)
    objects_of_interest = np.zeros(num_objects, dtype=np.bool_)

    # Track indices marked as tracks_to_predict
    ttp_set: set = set()
    if hasattr(proto, "tracks_to_predict"):
        for rp in proto.tracks_to_predict:
            ttp_set.add(rp.track_index)

    for slot, orig_idx in enumerate(track_indices):
        track = proto.tracks[orig_idx]
        obj_ids[slot] = track.id
        obj_types[slot] = track.object_type  # 1=vehicle, 2=ped, 3=cyclist
        if slot == sdc_slot:
            is_sdc[slot] = True
        if orig_idx in ttp_set:
            objects_of_interest[slot] = True

        for t_idx, state in enumerate(track.states):
            if t_idx >= T:
                break
            if state.valid:
                x[slot, t_idx] = state.center_x
                y[slot, t_idx] = state.center_y
                z[slot, t_idx] = state.center_z
                vel_x[slot, t_idx] = state.velocity_x
                vel_y[slot, t_idx] = state.velocity_y
                yaw[slot, t_idx] = state.heading
                valid[slot, t_idx] = True
                length[slot, t_idx] = state.length
                width[slot, t_idx] = state.width
                height[slot, t_idx] = state.height
                timestamp_micros[slot, t_idx] = int(
                    proto.timestamps_seconds[t_idx] * 1e6
                )

        # Object is "valid" if it has any valid state
        is_valid_obj[slot] = valid[slot].any()

    # Pad objects dim to max_objects
    def _pad_obj(arr: np.ndarray) -> np.ndarray:
        return _pad_or_truncate(arr, max_objects, axis=0)

    trajectory = datatypes.Trajectory(
        x=jnp.array(_pad_obj(x)),
        y=jnp.array(_pad_obj(y)),
        z=jnp.array(_pad_obj(z)),
        vel_x=jnp.array(_pad_obj(vel_x)),
        vel_y=jnp.array(_pad_obj(vel_y)),
        yaw=jnp.array(_pad_obj(yaw)),
        valid=jnp.array(_pad_obj(valid)),
        length=jnp.array(_pad_obj(length)),
        width=jnp.array(_pad_obj(width)),
        height=jnp.array(_pad_obj(height)),
        timestamp_micros=jnp.array(_pad_obj(timestamp_micros).astype(np.int64)),
    )

    metadata = datatypes.ObjectMetadata(
        ids=jnp.array(_pad_or_truncate(obj_ids, max_objects)),
        object_types=jnp.array(_pad_or_truncate(obj_types, max_objects)),
        is_sdc=jnp.array(_pad_or_truncate(is_sdc, max_objects)),
        is_modeled=jnp.array(_pad_or_truncate(is_sdc, max_objects)),
        is_valid=jnp.array(_pad_or_truncate(is_valid_obj, max_objects)),
        objects_of_interest=jnp.array(
            _pad_or_truncate(objects_of_interest, max_objects)
        ),
        is_controlled=jnp.array(_pad_or_truncate(is_sdc, max_objects)),
    )

    # ---- Roadgraph --------------------------------------------------------
    rg_x_list: List[float] = []
    rg_y_list: List[float] = []
    rg_z_list: List[float] = []
    rg_dir_x_list: List[float] = []
    rg_dir_y_list: List[float] = []
    rg_dir_z_list: List[float] = []
    rg_type_list: List[int] = []
    rg_id_list: List[int] = []
    rg_valid_list: List[bool] = []

    # Map feature type codes (Waymax convention)
    _MAP_TYPE = {
        "lane": 1,
        "road_line": 2,
        "road_edge": 3,
        "stop_sign": 4,
        "crosswalk": 5,
        "speed_bump": 6,
    }

    for mf in proto.map_features:
        for feat_name in [
            "lane",
            "road_line",
            "road_edge",
            "crosswalk",
            "stop_sign",
            "speed_bump",
        ]:
            if mf.HasField(feat_name):
                feat = getattr(mf, feat_name)
                polyline = feat.polyline if hasattr(feat, "polyline") else []
                mtype = _MAP_TYPE.get(feat_name, 0)
                for j, pt in enumerate(polyline):
                    rg_x_list.append(pt.x)
                    rg_y_list.append(pt.y)
                    rg_z_list.append(pt.z)
                    # Direction from consecutive points
                    if j + 1 < len(polyline):
                        npt = polyline[j + 1]
                        dx = npt.x - pt.x
                        dy = npt.y - pt.y
                        dz = npt.z - pt.z
                        norm = max(np.sqrt(dx**2 + dy**2 + dz**2), 1e-6)
                        rg_dir_x_list.append(dx / norm)
                        rg_dir_y_list.append(dy / norm)
                        rg_dir_z_list.append(dz / norm)
                    else:
                        rg_dir_x_list.append(0.0)
                        rg_dir_y_list.append(0.0)
                        rg_dir_z_list.append(0.0)
                    rg_type_list.append(mtype)
                    rg_id_list.append(mf.id)
                    rg_valid_list.append(True)
                break  # only one field per map feature

    n_rg = len(rg_x_list)
    rg_x_arr = (
        np.array(rg_x_list, dtype=np.float32) if n_rg else np.zeros(0, dtype=np.float32)
    )
    rg_y_arr = (
        np.array(rg_y_list, dtype=np.float32) if n_rg else np.zeros(0, dtype=np.float32)
    )
    rg_z_arr = (
        np.array(rg_z_list, dtype=np.float32) if n_rg else np.zeros(0, dtype=np.float32)
    )
    rg_dir_x_arr = (
        np.array(rg_dir_x_list, dtype=np.float32)
        if n_rg
        else np.zeros(0, dtype=np.float32)
    )
    rg_dir_y_arr = (
        np.array(rg_dir_y_list, dtype=np.float32)
        if n_rg
        else np.zeros(0, dtype=np.float32)
    )
    rg_dir_z_arr = (
        np.array(rg_dir_z_list, dtype=np.float32)
        if n_rg
        else np.zeros(0, dtype=np.float32)
    )
    rg_type_arr = (
        np.array(rg_type_list, dtype=np.int32) if n_rg else np.zeros(0, dtype=np.int32)
    )
    rg_id_arr = (
        np.array(rg_id_list, dtype=np.int32) if n_rg else np.zeros(0, dtype=np.int32)
    )
    rg_valid_arr = (
        np.array(rg_valid_list, dtype=np.bool_) if n_rg else np.zeros(0, dtype=np.bool_)
    )

    def _pad_rg(arr: np.ndarray) -> np.ndarray:
        return _pad_or_truncate(arr, max_rg_points, axis=0)

    roadgraph_points = datatypes.RoadgraphPoints(
        x=jnp.array(_pad_rg(rg_x_arr)),
        y=jnp.array(_pad_rg(rg_y_arr)),
        z=jnp.array(_pad_rg(rg_z_arr)),
        dir_x=jnp.array(_pad_rg(rg_dir_x_arr)),
        dir_y=jnp.array(_pad_rg(rg_dir_y_arr)),
        dir_z=jnp.array(_pad_rg(rg_dir_z_arr)),
        types=jnp.array(_pad_rg(rg_type_arr)),
        ids=jnp.array(_pad_rg(rg_id_arr)),
        valid=jnp.array(_pad_rg(rg_valid_arr)),
    )

    # ---- Traffic Lights ---------------------------------------------------
    tl_x = np.zeros((T, max_tl_states), dtype=np.float32)
    tl_y = np.zeros((T, max_tl_states), dtype=np.float32)
    tl_z = np.zeros((T, max_tl_states), dtype=np.float32)
    tl_state = np.zeros((T, max_tl_states), dtype=np.int32)
    tl_id = np.zeros((T, max_tl_states), dtype=np.int32)
    tl_valid = np.zeros((T, max_tl_states), dtype=np.bool_)

    for t_idx, dms in enumerate(proto.dynamic_map_states):
        if t_idx >= T:
            break
        for s_idx, ls in enumerate(dms.lane_states):
            if s_idx >= max_tl_states:
                break
            tl_x[t_idx, s_idx] = ls.stop_point.x
            tl_y[t_idx, s_idx] = ls.stop_point.y
            tl_z[t_idx, s_idx] = ls.stop_point.z
            tl_state[t_idx, s_idx] = ls.state
            tl_id[t_idx, s_idx] = ls.lane
            tl_valid[t_idx, s_idx] = True

    traffic_lights = datatypes.TrafficLights(
        x=jnp.array(tl_x),
        y=jnp.array(tl_y),
        z=jnp.array(tl_z),
        state=jnp.array(tl_state),
        lane_ids=jnp.array(tl_id),
        valid=jnp.array(tl_valid),
    )

    # ---- Assemble SimulatorState ------------------------------------------
    state = datatypes.SimulatorState(
        sim_trajectory=trajectory,
        log_trajectory=trajectory,  # same at init
        log_traffic_light=traffic_lights,
        object_metadata=metadata,
        timestep=jnp.int32(0),  # will be set to init_steps by env
        roadgraph_points=roadgraph_points,
    )

    return state


def load_scenarios(
    cfg: Optional[ScenarioLoaderConfig] = None,
) -> Iterator[Tuple[str, datatypes.SimulatorState]]:
    """
    Yield ``(scenario_id, SimulatorState)`` from local WOMD TFRecords.

    Parameters
    ----------
    cfg : ScenarioLoaderConfig, optional
        Defaults to validation split.

    Yields
    ------
    (scenario_id, SimulatorState)
    """
    if cfg is None:
        cfg = ScenarioLoaderConfig()

    split_dir = _SPLIT_DIRS.get(cfg.split, cfg.split)
    data_dir = os.path.join(cfg.data_root, split_dir)

    # Glob shards
    pattern = os.path.join(data_dir, "*.tfrecord*")
    import glob

    shard_paths = sorted(glob.glob(pattern))
    if not shard_paths:
        raise FileNotFoundError(f"No TFRecord files found at {pattern}")

    if cfg.shuffle:
        rng = np.random.RandomState(cfg.seed)
        rng.shuffle(shard_paths)

    count = 0
    for shard_path in shard_paths:
        ds = tf.data.TFRecordDataset(shard_path)
        for raw_record in ds:
            raw_bytes = raw_record.numpy()
            example = tf.train.Example()
            example.ParseFromString(raw_bytes)

            # Augmented format: proto under 'scenario/proto'
            if "scenario/proto" in example.features.feature:
                proto_bytes = example.features.feature[
                    "scenario/proto"
                ].bytes_list.value[0]
            else:
                # Skip unrecognised format
                continue

            scenario = scenario_pb2.Scenario()
            scenario.ParseFromString(proto_bytes)

            sim_state = scenario_proto_to_simulator_state(
                scenario,
                max_objects=cfg.max_objects,
                max_rg_points=cfg.max_rg_points,
                max_tl_states=cfg.max_tl_states,
            )

            yield scenario.scenario_id, sim_state

            count += 1
            if cfg.max_scenarios is not None and count >= cfg.max_scenarios:
                return
