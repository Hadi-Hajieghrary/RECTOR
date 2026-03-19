#!/usr/bin/env python3
"""
Pure Python converter: Waymo Scenario proto → TFExample format.

This converts raw Scenario protos (v1.3+) to the TFExample format expected
by M2I and other motion prediction models that use the older tf.train.Example
format with fixed-length features.

Usage:
    python scenario_to_example.py \
        --input_dir /path/to/scenario_tfrecords \
        --output_dir /path/to/output \
        --num_map_samples 20000 \
        --max_agents 128
"""

import argparse
import logging
import os
from glob import glob
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tqdm import tqdm

# Import Waymo protos
from waymo_open_dataset.protos import scenario_pb2


# Constants matching M2I expectations
NUM_PAST_STEPS = 10
NUM_CURRENT_STEPS = 1
NUM_FUTURE_STEPS = 80
TOTAL_STEPS = NUM_PAST_STEPS + NUM_CURRENT_STEPS + NUM_FUTURE_STEPS  # 91

DEFAULT_MAX_AGENTS = 128
DEFAULT_NUM_MAP_SAMPLES = 20000
DEFAULT_NUM_TRAFFIC_LIGHTS = 16

INVALID_VALUE = -1.0

log = logging.getLogger(__name__)


def _bytes_feature(value: bytes) -> tf.train.Feature:
    """Create a bytes feature."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(values: List[float]) -> tf.train.Feature:
    """Create a float feature."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))


def _int64_feature(values: List[int]) -> tf.train.Feature:
    """Create an int64 feature."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _pad_or_truncate(
    arr: np.ndarray, target_len: int, pad_value: float = INVALID_VALUE
) -> np.ndarray:
    """Pad or truncate array to target length."""
    if len(arr) >= target_len:
        return arr[:target_len]
    padded = np.full(target_len, pad_value, dtype=arr.dtype)
    padded[: len(arr)] = arr
    return padded


def _get_reordered_track_indices(scenario: scenario_pb2.Scenario) -> List[int]:
    """Get track indices reordered so that tracks_to_predict come first."""
    num_tracks = len(scenario.tracks)

    # Get the indices of tracks to predict
    tracks_to_predict_indices = set()
    for ttp in scenario.tracks_to_predict:
        tracks_to_predict_indices.add(ttp.track_index)

    # Get objects of interest
    objects_of_interest_ids = set(scenario.objects_of_interest)

    # Build reordering: tracks to predict first, then others
    predict_first = []
    others = []

    for i in range(num_tracks):
        track = scenario.tracks[i]
        # Include track if it's in tracks_to_predict OR objects_of_interest
        if i in tracks_to_predict_indices or track.id in objects_of_interest_ids:
            predict_first.append(i)
        else:
            others.append(i)

    return predict_first + others


def _extract_agent_states(
    scenario: scenario_pb2.Scenario,
    max_agents: int = DEFAULT_MAX_AGENTS,
) -> Dict[str, np.ndarray]:
    """Extract agent state features from scenario proto."""

    # Get reordered track indices so tracks_to_predict are first
    reordered_indices = _get_reordered_track_indices(scenario)
    num_tracks = len(scenario.tracks)

    # Initialize arrays with invalid values
    # Shape: [max_agents] for 1D, [max_agents, T] for 2D
    agent_id = np.full(max_agents, -1, dtype=np.int64)
    agent_type = np.full(max_agents, INVALID_VALUE, dtype=np.float32)
    is_sdc = np.zeros(max_agents, dtype=np.int64)
    tracks_to_predict = np.zeros(max_agents, dtype=np.int64)
    objects_of_interest = np.zeros(max_agents, dtype=np.int64)

    # Past states: [max_agents, NUM_PAST_STEPS]
    past_x = np.full((max_agents, NUM_PAST_STEPS), INVALID_VALUE, dtype=np.float32)
    past_y = np.full((max_agents, NUM_PAST_STEPS), INVALID_VALUE, dtype=np.float32)
    past_z = np.full((max_agents, NUM_PAST_STEPS), INVALID_VALUE, dtype=np.float32)
    past_bbox_yaw = np.full(
        (max_agents, NUM_PAST_STEPS), INVALID_VALUE, dtype=np.float32
    )
    past_length = np.full((max_agents, NUM_PAST_STEPS), INVALID_VALUE, dtype=np.float32)
    past_width = np.full((max_agents, NUM_PAST_STEPS), INVALID_VALUE, dtype=np.float32)
    past_height = np.full((max_agents, NUM_PAST_STEPS), INVALID_VALUE, dtype=np.float32)
    past_velocity_x = np.full(
        (max_agents, NUM_PAST_STEPS), INVALID_VALUE, dtype=np.float32
    )
    past_velocity_y = np.full(
        (max_agents, NUM_PAST_STEPS), INVALID_VALUE, dtype=np.float32
    )
    past_vel_yaw = np.full(
        (max_agents, NUM_PAST_STEPS), INVALID_VALUE, dtype=np.float32
    )
    past_valid = np.zeros((max_agents, NUM_PAST_STEPS), dtype=np.int64)
    past_timestamp = np.full((max_agents, NUM_PAST_STEPS), -1, dtype=np.int64)

    # Current states: [max_agents, 1]
    current_x = np.full((max_agents, 1), INVALID_VALUE, dtype=np.float32)
    current_y = np.full((max_agents, 1), INVALID_VALUE, dtype=np.float32)
    current_z = np.full((max_agents, 1), INVALID_VALUE, dtype=np.float32)
    current_bbox_yaw = np.full((max_agents, 1), INVALID_VALUE, dtype=np.float32)
    current_length = np.full((max_agents, 1), INVALID_VALUE, dtype=np.float32)
    current_width = np.full((max_agents, 1), INVALID_VALUE, dtype=np.float32)
    current_height = np.full((max_agents, 1), INVALID_VALUE, dtype=np.float32)
    current_velocity_x = np.full((max_agents, 1), INVALID_VALUE, dtype=np.float32)
    current_velocity_y = np.full((max_agents, 1), INVALID_VALUE, dtype=np.float32)
    current_vel_yaw = np.full((max_agents, 1), INVALID_VALUE, dtype=np.float32)
    current_valid = np.zeros((max_agents, 1), dtype=np.int64)
    current_timestamp = np.full((max_agents, 1), -1, dtype=np.int64)

    # Future states: [max_agents, NUM_FUTURE_STEPS]
    future_x = np.full((max_agents, NUM_FUTURE_STEPS), INVALID_VALUE, dtype=np.float32)
    future_y = np.full((max_agents, NUM_FUTURE_STEPS), INVALID_VALUE, dtype=np.float32)
    future_z = np.full((max_agents, NUM_FUTURE_STEPS), INVALID_VALUE, dtype=np.float32)
    future_bbox_yaw = np.full(
        (max_agents, NUM_FUTURE_STEPS), INVALID_VALUE, dtype=np.float32
    )
    future_length = np.full(
        (max_agents, NUM_FUTURE_STEPS), INVALID_VALUE, dtype=np.float32
    )
    future_width = np.full(
        (max_agents, NUM_FUTURE_STEPS), INVALID_VALUE, dtype=np.float32
    )
    future_height = np.full(
        (max_agents, NUM_FUTURE_STEPS), INVALID_VALUE, dtype=np.float32
    )
    future_velocity_x = np.full(
        (max_agents, NUM_FUTURE_STEPS), INVALID_VALUE, dtype=np.float32
    )
    future_velocity_y = np.full(
        (max_agents, NUM_FUTURE_STEPS), INVALID_VALUE, dtype=np.float32
    )
    future_vel_yaw = np.full(
        (max_agents, NUM_FUTURE_STEPS), INVALID_VALUE, dtype=np.float32
    )
    future_valid = np.zeros((max_agents, NUM_FUTURE_STEPS), dtype=np.int64)
    future_timestamp = np.full((max_agents, NUM_FUTURE_STEPS), -1, dtype=np.int64)

    # Get SDC track index
    sdc_track_index = scenario.sdc_track_index

    # Get tracks to predict
    tracks_to_predict_indices = set()
    for ttp in scenario.tracks_to_predict:
        tracks_to_predict_indices.add(ttp.track_index)

    # Get objects of interest (by track ID, not index)
    objects_of_interest_ids = set(scenario.objects_of_interest)

    # Current timestep index in Scenario proto (usually step 10 for 11 history steps, 0-indexed)
    current_time_index = scenario.current_time_index

    # Extract timestamps
    timestamps_seconds = list(scenario.timestamps_seconds)

    # Use reordered indices so tracks_to_predict come first
    for agent_idx, orig_track_idx in enumerate(reordered_indices):
        if agent_idx >= max_agents:
            break

        track = scenario.tracks[orig_track_idx]

        agent_id[agent_idx] = int(track.id)
        agent_type[agent_idx] = float(track.object_type)

        if orig_track_idx == sdc_track_index:
            is_sdc[agent_idx] = 1

        if orig_track_idx in tracks_to_predict_indices:
            tracks_to_predict[agent_idx] = 1

        if track.id in objects_of_interest_ids:
            objects_of_interest[agent_idx] = 1

        # Extract states for each timestep
        for state_idx, state in enumerate(track.states):
            if state_idx < current_time_index:
                # Past state
                past_idx = state_idx - (current_time_index - NUM_PAST_STEPS)
                if 0 <= past_idx < NUM_PAST_STEPS:
                    past_x[agent_idx, past_idx] = state.center_x
                    past_y[agent_idx, past_idx] = state.center_y
                    past_z[agent_idx, past_idx] = state.center_z
                    past_bbox_yaw[agent_idx, past_idx] = state.heading
                    past_length[agent_idx, past_idx] = state.length
                    past_width[agent_idx, past_idx] = state.width
                    past_height[agent_idx, past_idx] = state.height
                    past_velocity_x[agent_idx, past_idx] = state.velocity_x
                    past_velocity_y[agent_idx, past_idx] = state.velocity_y
                    past_vel_yaw[agent_idx, past_idx] = 0.0  # Not in proto
                    past_valid[agent_idx, past_idx] = 1 if state.valid else 0
                    if state_idx < len(timestamps_seconds):
                        past_timestamp[agent_idx, past_idx] = int(
                            timestamps_seconds[state_idx] * 1e6
                        )

            elif state_idx == current_time_index:
                # Current state
                current_x[agent_idx, 0] = state.center_x
                current_y[agent_idx, 0] = state.center_y
                current_z[agent_idx, 0] = state.center_z
                current_bbox_yaw[agent_idx, 0] = state.heading
                current_length[agent_idx, 0] = state.length
                current_width[agent_idx, 0] = state.width
                current_height[agent_idx, 0] = state.height
                current_velocity_x[agent_idx, 0] = state.velocity_x
                current_velocity_y[agent_idx, 0] = state.velocity_y
                current_vel_yaw[agent_idx, 0] = 0.0
                current_valid[agent_idx, 0] = 1 if state.valid else 0
                if state_idx < len(timestamps_seconds):
                    current_timestamp[agent_idx, 0] = int(
                        timestamps_seconds[state_idx] * 1e6
                    )

            else:
                # Future state
                future_idx = state_idx - current_time_index - 1
                if 0 <= future_idx < NUM_FUTURE_STEPS:
                    future_x[agent_idx, future_idx] = state.center_x
                    future_y[agent_idx, future_idx] = state.center_y
                    future_z[agent_idx, future_idx] = state.center_z
                    future_bbox_yaw[agent_idx, future_idx] = state.heading
                    future_length[agent_idx, future_idx] = state.length
                    future_width[agent_idx, future_idx] = state.width
                    future_height[agent_idx, future_idx] = state.height
                    future_velocity_x[agent_idx, future_idx] = state.velocity_x
                    future_velocity_y[agent_idx, future_idx] = state.velocity_y
                    future_vel_yaw[agent_idx, future_idx] = 0.0
                    future_valid[agent_idx, future_idx] = 1 if state.valid else 0
                    if state_idx < len(timestamps_seconds):
                        future_timestamp[agent_idx, future_idx] = int(
                            timestamps_seconds[state_idx] * 1e6
                        )

    return {
        "state/id": agent_id,
        "state/type": agent_type,
        "state/is_sdc": is_sdc,
        "state/tracks_to_predict": tracks_to_predict,
        "state/objects_of_interest": objects_of_interest,
        # Past
        "state/past/x": past_x,
        "state/past/y": past_y,
        "state/past/z": past_z,
        "state/past/bbox_yaw": past_bbox_yaw,
        "state/past/length": past_length,
        "state/past/width": past_width,
        "state/past/height": past_height,
        "state/past/velocity_x": past_velocity_x,
        "state/past/velocity_y": past_velocity_y,
        "state/past/vel_yaw": past_vel_yaw,
        "state/past/valid": past_valid,
        "state/past/timestamp_micros": past_timestamp,
        # Current
        "state/current/x": current_x,
        "state/current/y": current_y,
        "state/current/z": current_z,
        "state/current/bbox_yaw": current_bbox_yaw,
        "state/current/length": current_length,
        "state/current/width": current_width,
        "state/current/height": current_height,
        "state/current/velocity_x": current_velocity_x,
        "state/current/velocity_y": current_velocity_y,
        "state/current/vel_yaw": current_vel_yaw,
        "state/current/valid": current_valid,
        "state/current/timestamp_micros": current_timestamp,
        # Future
        "state/future/x": future_x,
        "state/future/y": future_y,
        "state/future/z": future_z,
        "state/future/bbox_yaw": future_bbox_yaw,
        "state/future/length": future_length,
        "state/future/width": future_width,
        "state/future/height": future_height,
        "state/future/velocity_x": future_velocity_x,
        "state/future/velocity_y": future_velocity_y,
        "state/future/vel_yaw": future_vel_yaw,
        "state/future/valid": future_valid,
        "state/future/timestamp_micros": future_timestamp,
    }


def _extract_roadgraph(
    scenario: scenario_pb2.Scenario,
    num_map_samples: int = DEFAULT_NUM_MAP_SAMPLES,
) -> Dict[str, np.ndarray]:
    """Extract roadgraph features from scenario proto."""

    xyz_list = []
    dir_list = []
    type_list = []
    id_list = []
    valid_list = []

    for map_feature in scenario.map_features:
        feature_id = map_feature.id

        # Determine feature type based on which field is set
        if map_feature.HasField("lane"):
            lane = map_feature.lane
            points = list(lane.polyline)
            feature_type = lane.type  # LaneType enum
        elif map_feature.HasField("road_line"):
            road_line = map_feature.road_line
            points = list(road_line.polyline)
            feature_type = 6 + road_line.type  # Offset for road lines
        elif map_feature.HasField("road_edge"):
            road_edge = map_feature.road_edge
            points = list(road_edge.polyline)
            feature_type = 15 + road_edge.type  # Offset for road edges
        elif map_feature.HasField("stop_sign"):
            stop_sign = map_feature.stop_sign
            points = [stop_sign.position]
            feature_type = 17
        elif map_feature.HasField("crosswalk"):
            crosswalk = map_feature.crosswalk
            points = list(crosswalk.polygon)
            feature_type = 18
        elif map_feature.HasField("speed_bump"):
            speed_bump = map_feature.speed_bump
            points = list(speed_bump.polygon)
            feature_type = 19
        elif map_feature.HasField("driveway"):
            driveway = map_feature.driveway
            points = list(driveway.polygon)
            feature_type = 20
        else:
            continue

        # Extract points and compute directions
        for i, pt in enumerate(points):
            xyz_list.append([pt.x, pt.y, pt.z])

            # Compute direction to next point
            if i + 1 < len(points):
                next_pt = points[i + 1]
                dx = next_pt.x - pt.x
                dy = next_pt.y - pt.y
                dz = next_pt.z - pt.z
                norm = np.sqrt(dx * dx + dy * dy + dz * dz)
                if norm > 1e-6:
                    dir_list.append([dx / norm, dy / norm, dz / norm])
                else:
                    dir_list.append([0.0, 0.0, 0.0])
            else:
                dir_list.append([0.0, 0.0, 0.0])

            type_list.append(feature_type)
            id_list.append(feature_id)
            valid_list.append(1)

    # Pad to num_map_samples
    num_points = len(xyz_list)

    xyz = np.full((num_map_samples, 3), INVALID_VALUE, dtype=np.float32)
    direction = np.full((num_map_samples, 3), INVALID_VALUE, dtype=np.float32)
    types = np.full((num_map_samples, 1), INVALID_VALUE, dtype=np.int64)
    ids = np.full((num_map_samples, 1), INVALID_VALUE, dtype=np.int64)
    valid = np.zeros((num_map_samples, 1), dtype=np.int64)

    copy_len = min(num_points, num_map_samples)
    if copy_len > 0:
        xyz[:copy_len] = np.array(xyz_list[:copy_len])
        direction[:copy_len] = np.array(dir_list[:copy_len])
        types[:copy_len, 0] = np.array(type_list[:copy_len])
        ids[:copy_len, 0] = np.array(id_list[:copy_len])
        valid[:copy_len, 0] = np.array(valid_list[:copy_len])

    return {
        "roadgraph_samples/xyz": xyz,
        "roadgraph_samples/dir": direction,
        "roadgraph_samples/type": types,
        "roadgraph_samples/id": ids,
        "roadgraph_samples/valid": valid,
    }


def _extract_traffic_lights(
    scenario: scenario_pb2.Scenario,
    num_traffic_lights: int = DEFAULT_NUM_TRAFFIC_LIGHTS,
) -> Dict[str, np.ndarray]:
    """Extract traffic light state features from scenario proto."""

    current_time_index = scenario.current_time_index

    # Past: [NUM_PAST_STEPS, num_traffic_lights]
    past_state = np.zeros((NUM_PAST_STEPS, num_traffic_lights), dtype=np.int64)
    past_x = np.zeros((NUM_PAST_STEPS, num_traffic_lights), dtype=np.float32)
    past_y = np.zeros((NUM_PAST_STEPS, num_traffic_lights), dtype=np.float32)
    past_z = np.zeros((NUM_PAST_STEPS, num_traffic_lights), dtype=np.float32)
    past_id = np.zeros((NUM_PAST_STEPS, num_traffic_lights), dtype=np.int64)
    past_valid = np.zeros((NUM_PAST_STEPS, num_traffic_lights), dtype=np.int64)

    # Current: [1, num_traffic_lights]
    current_state = np.zeros((1, num_traffic_lights), dtype=np.int64)
    current_x = np.zeros((1, num_traffic_lights), dtype=np.float32)
    current_y = np.zeros((1, num_traffic_lights), dtype=np.float32)
    current_z = np.zeros((1, num_traffic_lights), dtype=np.float32)
    current_id = np.zeros((1, num_traffic_lights), dtype=np.int64)
    current_valid = np.zeros((1, num_traffic_lights), dtype=np.int64)

    # Future: [NUM_FUTURE_STEPS, num_traffic_lights]
    future_state = np.zeros((NUM_FUTURE_STEPS, num_traffic_lights), dtype=np.int64)
    future_x = np.zeros((NUM_FUTURE_STEPS, num_traffic_lights), dtype=np.float32)
    future_y = np.zeros((NUM_FUTURE_STEPS, num_traffic_lights), dtype=np.float32)
    future_z = np.zeros((NUM_FUTURE_STEPS, num_traffic_lights), dtype=np.float32)
    future_id = np.zeros((NUM_FUTURE_STEPS, num_traffic_lights), dtype=np.int64)
    future_valid = np.zeros((NUM_FUTURE_STEPS, num_traffic_lights), dtype=np.int64)

    dropped_future_steps = 0
    dropped_lane_states = 0

    # Extract from dynamic map states
    for step_idx, dms in enumerate(scenario.dynamic_map_states):
        if step_idx < current_time_index:
            # Past
            past_idx = step_idx - (current_time_index - NUM_PAST_STEPS)
            if 0 <= past_idx < NUM_PAST_STEPS:
                for tl_idx, tl_state in enumerate(dms.lane_states):
                    if tl_idx >= num_traffic_lights:
                        dropped_lane_states += 1
                        break
                    past_state[past_idx, tl_idx] = tl_state.state
                    past_x[past_idx, tl_idx] = tl_state.stop_point.x
                    past_y[past_idx, tl_idx] = tl_state.stop_point.y
                    past_z[past_idx, tl_idx] = tl_state.stop_point.z
                    past_id[past_idx, tl_idx] = tl_state.lane
                    past_valid[past_idx, tl_idx] = 1

        elif step_idx == current_time_index:
            # Current
            for tl_idx, tl_state in enumerate(dms.lane_states):
                if tl_idx >= num_traffic_lights:
                    dropped_lane_states += 1
                    break
                current_state[0, tl_idx] = tl_state.state
                current_x[0, tl_idx] = tl_state.stop_point.x
                current_y[0, tl_idx] = tl_state.stop_point.y
                current_z[0, tl_idx] = tl_state.stop_point.z
                current_id[0, tl_idx] = tl_state.lane
                current_valid[0, tl_idx] = 1

        else:
            # Future
            future_idx = step_idx - current_time_index - 1
            if 0 <= future_idx < NUM_FUTURE_STEPS:
                for tl_idx, tl_state in enumerate(dms.lane_states):
                    if tl_idx >= num_traffic_lights:
                        dropped_lane_states += 1
                        break
                    future_state[future_idx, tl_idx] = tl_state.state
                    future_x[future_idx, tl_idx] = tl_state.stop_point.x
                    future_y[future_idx, tl_idx] = tl_state.stop_point.y
                    future_z[future_idx, tl_idx] = tl_state.stop_point.z
                    future_id[future_idx, tl_idx] = tl_state.lane
                    future_valid[future_idx, tl_idx] = 1
            elif future_idx >= NUM_FUTURE_STEPS:
                dropped_future_steps += 1

    if dropped_future_steps > 0:
        log.warning(
            "Scenario %s: dropped %d future traffic-light timesteps beyond horizon=%d",
            scenario.scenario_id,
            dropped_future_steps,
            NUM_FUTURE_STEPS,
        )

    if dropped_lane_states > 0:
        log.warning(
            "Scenario %s: dropped %d traffic-light lane states beyond max_traffic_lights=%d",
            scenario.scenario_id,
            dropped_lane_states,
            num_traffic_lights,
        )

    return {
        "traffic_light_state/past/state": past_state,
        "traffic_light_state/past/x": past_x,
        "traffic_light_state/past/y": past_y,
        "traffic_light_state/past/z": past_z,
        "traffic_light_state/past/id": past_id,
        "traffic_light_state/past/valid": past_valid,
        "traffic_light_state/current/state": current_state,
        "traffic_light_state/current/x": current_x,
        "traffic_light_state/current/y": current_y,
        "traffic_light_state/current/z": current_z,
        "traffic_light_state/current/id": current_id,
        "traffic_light_state/current/valid": current_valid,
        "traffic_light_state/future/state": future_state,
        "traffic_light_state/future/x": future_x,
        "traffic_light_state/future/y": future_y,
        "traffic_light_state/future/z": future_z,
        "traffic_light_state/future/id": future_id,
        "traffic_light_state/future/valid": future_valid,
    }


def scenario_to_example(
    scenario: scenario_pb2.Scenario,
    max_agents: int = DEFAULT_MAX_AGENTS,
    num_map_samples: int = DEFAULT_NUM_MAP_SAMPLES,
) -> tf.train.Example:
    """Convert a Scenario proto to a tf.train.Example."""

    features = {}

    # Add scenario ID
    features["scenario/id"] = _bytes_feature(scenario.scenario_id.encode("utf-8"))

    # Extract and add agent states
    agent_features = _extract_agent_states(scenario, max_agents)
    for key, value in agent_features.items():
        if value.dtype == np.float32:
            features[key] = _float_feature(value.flatten().tolist())
        else:
            features[key] = _int64_feature(value.flatten().tolist())

    # Extract and add roadgraph
    roadgraph_features = _extract_roadgraph(scenario, num_map_samples)
    for key, value in roadgraph_features.items():
        if value.dtype == np.float32:
            features[key] = _float_feature(value.flatten().tolist())
        else:
            features[key] = _int64_feature(value.flatten().tolist())

    # Extract and add traffic lights
    traffic_features = _extract_traffic_lights(scenario)
    for key, value in traffic_features.items():
        if value.dtype == np.float32:
            features[key] = _float_feature(value.flatten().tolist())
        else:
            features[key] = _int64_feature(value.flatten().tolist())

    return tf.train.Example(features=tf.train.Features(feature=features))


def convert_tfrecord(
    input_path: str,
    output_path: str,
    max_agents: int = DEFAULT_MAX_AGENTS,
    num_map_samples: int = DEFAULT_NUM_MAP_SAMPLES,
) -> int:
    """Convert a single TFRecord file from Scenario to Example format."""

    dataset = tf.data.TFRecordDataset(input_path)

    count = 0
    with tf.io.TFRecordWriter(output_path) as writer:
        for record in dataset:
            scenario = scenario_pb2.Scenario.FromString(record.numpy())
            example = scenario_to_example(scenario, max_agents, num_map_samples)
            writer.write(example.SerializeToString())
            count += 1

    return count


def main():
    parser = argparse.ArgumentParser(
        description="Convert Scenario TFRecords to Example format"
    )
    parser.add_argument(
        "--input_dir", required=True, help="Input directory with Scenario TFRecords"
    )
    parser.add_argument(
        "--output_dir", required=True, help="Output directory for Example TFRecords"
    )
    parser.add_argument(
        "--max_agents",
        type=int,
        default=DEFAULT_MAX_AGENTS,
        help="Max agents per example",
    )
    parser.add_argument(
        "--num_map_samples",
        type=int,
        default=DEFAULT_NUM_MAP_SAMPLES,
        help="Number of map samples",
    )
    parser.add_argument(
        "--pattern", default="*.tfrecord*", help="Glob pattern for input files"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    input_files = sorted(glob(os.path.join(args.input_dir, args.pattern)))
    print(f"Found {len(input_files)} input files")

    total_scenarios = 0
    for input_path in tqdm(input_files, desc="Converting files"):
        basename = os.path.basename(input_path)
        output_path = os.path.join(args.output_dir, basename)

        count = convert_tfrecord(
            input_path,
            output_path,
            max_agents=args.max_agents,
            num_map_samples=args.num_map_samples,
        )
        total_scenarios += count

    print(f"Converted {total_scenarios} scenarios from {len(input_files)} files")


if __name__ == "__main__":
    main()
