#!/usr/bin/env python3
"""
Full Rule Evaluation for RECTOR Predictions.

This script evaluates RECTOR's trajectory predictions against all 28 Waymo
rule tiers including Legal and Road rules that require full map context.

The process:
1. Load augmented TFRecords containing full scenario protos
2. Run RECTOR inference to get trajectory predictions
3. Inject predictions into ScenarioContext as ego trajectory
4. Run RuleExecutor with all 28 rules
5. Aggregate results by tier

Output:
- Per-scenario rule violations
- Tier-wise aggregated statistics
- Comparison with ground truth trajectories
"""

import os
import sys
import json
import glob
import time
import logging
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))
sys.path.insert(0, "/workspace/data/WOMD")
sys.path.insert(0, "/workspace/data")

# TensorFlow for TFRecords
import tensorflow as tf

tf.config.set_visible_devices([], "GPU")

# PyTorch for RECTOR
import torch

torch.set_num_threads(4)

# Waymo protos
from waymo_open_dataset.protos import scenario_pb2

# Rule evaluation framework
from waymo_rule_eval.core.context import (
    Agent,
    EgoState,
    MapContext,
    MapSignals,
    ScenarioContext,
)
from waymo_rule_eval.pipeline.rule_executor import RuleExecutor, ScenarioResult
from waymo_rule_eval.utils.constants import (
    AGENT_TYPE_MAP,
    DEFAULT_DT_S,
    DEFAULT_EGO_LENGTH,
    DEFAULT_EGO_WIDTH,
    SIGNAL_STATE_UNKNOWN,
    SIGNAL_STATE_STOP,
    SIGNAL_STATE_CAUTION,
    SIGNAL_STATE_GO,
    SIGNAL_STATE_ARROW_STOP,
    SIGNAL_STATE_ARROW_CAUTION,
    SIGNAL_STATE_ARROW_GO,
    SIGNAL_STATE_FLASHING_STOP,
    SIGNAL_STATE_FLASHING_CAUTION,
)

# Rule tier definitions
from waymo_rule_eval.rules.rule_constants import (
    TIER_0_SAFETY,
    TIER_1_LEGAL,
    TIER_2_ROAD,
    TIER_3_COMFORT,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)


@dataclass
class TierStats:
    """Statistics for a rule tier."""

    tier_name: str
    total_applicable: int = 0
    total_violations: int = 0
    total_severity: float = 0.0
    rule_stats: Dict[str, Dict] = field(default_factory=dict)

    @property
    def violation_rate(self) -> float:
        if self.total_applicable == 0:
            return 0.0
        return 100.0 * self.total_violations / self.total_applicable


@dataclass
class EvaluationResults:
    """Complete evaluation results."""

    checkpoint: str
    total_scenarios: int
    elapsed_time: float
    tier_stats: Dict[str, TierStats] = field(default_factory=dict)
    per_scenario_results: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "checkpoint": self.checkpoint,
            "total_scenarios": self.total_scenarios,
            "elapsed_time": self.elapsed_time,
            "tier_violation_rates": {
                name: stats.violation_rate for name, stats in self.tier_stats.items()
            },
            "tier_violations": {
                name: stats.total_violations for name, stats in self.tier_stats.items()
            },
            "tier_applicable": {
                name: stats.total_applicable for name, stats in self.tier_stats.items()
            },
            "per_rule_violations": {
                name: {
                    rule_id: stats.get("violations", 0)
                    for rule_id, stats in tier_stats.rule_stats.items()
                }
                for name, tier_stats in self.tier_stats.items()
            },
            "per_rule_applicable": {
                name: {
                    rule_id: stats.get("applicable", 0)
                    for rule_id, stats in tier_stats.rule_stats.items()
                }
                for name, tier_stats in self.tier_stats.items()
            },
        }


class AugmentedTFRecordReader:
    """Reads augmented TFRecords with embedded scenario protos."""

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.tfrecord_files = sorted(glob.glob(os.path.join(data_dir, "*.tfrecord*")))
        log.info(f"Found {len(self.tfrecord_files)} TFRecord files in {data_dir}")

    def iter_scenarios(self, max_scenarios: Optional[int] = None):
        """Iterate through scenarios yielding (scenario_id, scenario_proto, example)."""
        count = 0

        for tfrecord_path in self.tfrecord_files:
            if max_scenarios and count >= max_scenarios:
                break

            dataset = tf.data.TFRecordDataset(tfrecord_path)

            for record in dataset:
                if max_scenarios and count >= max_scenarios:
                    break

                try:
                    example = tf.train.Example.FromString(record.numpy())

                    # Extract scenario proto
                    scenario_proto_bytes = example.features.feature[
                        "scenario/proto"
                    ].bytes_list.value[0]
                    scenario_id = (
                        example.features.feature["scenario/id"]
                        .bytes_list.value[0]
                        .decode("utf-8")
                    )

                    scenario = scenario_pb2.Scenario()
                    scenario.ParseFromString(scenario_proto_bytes)

                    yield scenario_id, scenario, example
                    count += 1

                except Exception as e:
                    log.warning(f"Failed to parse record: {e}")
                    continue

        log.info(f"Processed {count} scenarios")


class ScenarioContextBuilder:
    """Builds ScenarioContext from Waymo scenario proto."""

    def __init__(self, dt: float = DEFAULT_DT_S):
        self.dt = dt

    def build(
        self, scenario, predicted_trajectory: Optional[np.ndarray] = None
    ) -> ScenarioContext:
        """
        Build ScenarioContext from scenario proto.

        Args:
            scenario: Waymo Scenario proto
            predicted_trajectory: Optional predicted trajectory [T, 2] to use as ego trajectory
                                 If None, uses ground truth trajectory

        Returns:
            ScenarioContext with all required data for rule evaluation
        """
        scenario_id = scenario.scenario_id
        n_frames = len(scenario.timestamps_seconds)

        # Find ego track: sdc_track_index is an array index, NOT a track object ID
        ego_idx = scenario.sdc_track_index
        if 0 <= ego_idx < len(scenario.tracks):
            ego_track = scenario.tracks[ego_idx]
        elif len(scenario.tracks) > 0:
            ego_track = scenario.tracks[0]
        else:
            ego_track = None

        # Extract ego state
        ego = self._extract_ego(ego_track, n_frames, predicted_trajectory)

        # Extract other agents
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

    def _extract_ego(
        self, track, n_frames: int, predicted_trajectory: Optional[np.ndarray] = None
    ) -> EgoState:
        """Extract ego state, optionally injecting predicted trajectory."""
        x = np.full(n_frames, np.nan)
        y = np.full(n_frames, np.nan)
        yaw = np.full(n_frames, np.nan)
        speed = np.full(n_frames, np.nan)
        valid = np.zeros(n_frames, dtype=bool)

        length = DEFAULT_EGO_LENGTH
        width = DEFAULT_EGO_WIDTH

        if track is not None:
            for i, state in enumerate(track.states):
                if i >= n_frames:
                    break
                if state.valid:
                    x[i] = state.center_x
                    y[i] = state.center_y
                    yaw[i] = state.heading
                    vx = state.velocity_x if hasattr(state, "velocity_x") else 0
                    vy = state.velocity_y if hasattr(state, "velocity_y") else 0
                    speed[i] = np.sqrt(vx**2 + vy**2)
                    valid[i] = True

                    if hasattr(state, "length") and state.length > 0:
                        length = state.length
                    if hasattr(state, "width") and state.width > 0:
                        width = state.width

        # Inject predicted trajectory for future frames (t >= 11)
        if predicted_trajectory is not None:
            current_frame = 10  # History frames 0-10, predictions start at 11
            pred_len = min(len(predicted_trajectory), n_frames - current_frame - 1)

            for i in range(pred_len):
                t = current_frame + 1 + i
                if t < n_frames:
                    x[t] = predicted_trajectory[i, 0]
                    y[t] = predicted_trajectory[i, 1]
                    valid[t] = True

                    # Compute heading from trajectory direction
                    if i > 0:
                        dx = predicted_trajectory[i, 0] - predicted_trajectory[i - 1, 0]
                        dy = predicted_trajectory[i, 1] - predicted_trajectory[i - 1, 1]
                        yaw[t] = np.arctan2(dy, dx)
                    elif i == 0 and current_frame > 0:
                        dx = predicted_trajectory[0, 0] - x[current_frame]
                        dy = predicted_trajectory[0, 1] - y[current_frame]
                        yaw[t] = np.arctan2(dy, dx)

                    # Compute speed from displacement
                    if i > 0:
                        disp = np.sqrt(
                            (
                                predicted_trajectory[i, 0]
                                - predicted_trajectory[i - 1, 0]
                            )
                            ** 2
                            + (
                                predicted_trajectory[i, 1]
                                - predicted_trajectory[i - 1, 1]
                            )
                            ** 2
                        )
                        speed[t] = disp / self.dt

        return EgoState(
            x=x, y=y, yaw=yaw, speed=speed, length=length, width=width, valid=valid
        )

    def _extract_agents(self, tracks, ego_idx: int, n_frames: int) -> List[Agent]:
        """Extract all other agents."""
        agents = []

        for track in tracks:
            if track.id == ego_idx:
                continue

            obj_type = track.object_type
            if obj_type not in AGENT_TYPE_MAP:
                continue

            agent_type = AGENT_TYPE_MAP[obj_type]

            x = np.full(n_frames, np.nan)
            y = np.full(n_frames, np.nan)
            yaw = np.full(n_frames, np.nan)
            speed = np.full(n_frames, np.nan)
            valid = np.zeros(n_frames, dtype=bool)

            length, width = 4.5, 1.8  # Defaults

            for i, state in enumerate(track.states):
                if i >= n_frames:
                    break
                if state.valid:
                    x[i] = state.center_x
                    y[i] = state.center_y
                    yaw[i] = state.heading
                    vx = getattr(state, "velocity_x", 0)
                    vy = getattr(state, "velocity_y", 0)
                    speed[i] = np.sqrt(vx**2 + vy**2)
                    valid[i] = True

                    if hasattr(state, "length") and state.length > 0:
                        length = state.length
                    if hasattr(state, "width") and state.width > 0:
                        width = state.width

            if np.any(valid):
                agents.append(
                    Agent(
                        id=track.id,
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
        """Extract map features."""
        all_lanes = []
        stoplines = []
        crosswalk_polys = []
        road_edges = []
        stop_signs = []

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
        """Extract traffic signal states."""
        signal_state = np.full(n_frames, SIGNAL_STATE_UNKNOWN, dtype=int)

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

        for frame_state in dynamic_states:
            t = frame_state.timestep if hasattr(frame_state, "timestep") else 0
            if t < 0 or t >= n_frames:
                continue

            for sig in frame_state.lane_states:
                if ego_lane_id is not None and sig.lane != ego_lane_id:
                    continue
                signal_state[t] = state_map.get(sig.state, SIGNAL_STATE_UNKNOWN)
                break

        return MapSignals(
            signal_state=signal_state,
            ego_lane_id=ego_lane_id,
        )


class RECTORInference:
    """Run RECTOR inference to get trajectory predictions."""

    TRAJECTORY_SCALE = 50.0
    HISTORY_LENGTH = 11
    MAX_AGENTS = 32
    MAX_LANES = 64
    LANE_POINTS = 20

    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        self.checkpoint_path = checkpoint_path
        self.model = self._load_model()

    def _load_model(self):
        """Load RECTOR model from checkpoint."""
        log.info(f"Loading RECTOR model from {self.checkpoint_path}")

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        # Import model from RECTOR source
        sys.path.insert(0, "/workspace/models/RECTOR/scripts")
        from models.rule_aware_generator import RuleAwareGenerator

        # Inferred from checkpoint weights
        model = RuleAwareGenerator(
            embed_dim=256,
            num_heads=8,
            num_encoder_layers=4,
            history_length=11,
            max_agents=self.MAX_AGENTS,
            max_lanes=self.MAX_LANES,
            trajectory_length=50,
            num_modes=6,
            latent_dim=64,
            decoder_hidden_dim=256,
            decoder_num_layers=4,
            num_rules=28,
            dropout=0.1,
            use_m2i_encoder=True,  # Checkpoint uses M2I encoder
        )

        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()

        log.info("Model loaded successfully")
        return model

    def predict_from_scenario(
        self,
        scenario,  # Scenario proto
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Run RECTOR inference from a Waymo Scenario proto.

        Returns:
            trajectories: [K, T, 2] - K modes, T timesteps (in world coords)
            confidences: [K] - mode confidences
            applicability: [28] - rule applicability predictions
            None, None, None if extraction fails
        """
        # Extract features from scenario
        features = self._extract_features(scenario)
        if features is None:
            return None, None, None

        with torch.no_grad():
            # Unpack features
            ego_history = (
                torch.from_numpy(features["ego_history"]).unsqueeze(0).to(self.device)
            )
            agent_states = (
                torch.from_numpy(features["agent_states"]).unsqueeze(0).to(self.device)
            )
            lane_centers = (
                torch.from_numpy(features["lane_centers"]).unsqueeze(0).to(self.device)
            )

            # Forward pass
            outputs = self.model(
                ego_history=ego_history,
                agent_states=agent_states,
                lane_centers=lane_centers,
            )

            # Get outputs
            trajectories_norm = outputs["trajectories"].cpu().numpy()[0]  # [K, T, 4]
            confidence = outputs["confidence"].cpu().numpy()[0]
            applicability = outputs["applicability"].cpu().numpy()[0]

            # Denormalize trajectories to world coords
            ref_x = features["ref_x"]
            ref_y = features["ref_y"]
            ref_h = features["ref_h"]
            cos_h, sin_h = np.cos(ref_h), np.sin(ref_h)

            # Unscale and rotate back
            trajectories = trajectories_norm[:, :, :2] * self.TRAJECTORY_SCALE
            # Rotate from ego-centric to world
            traj_world = np.zeros_like(trajectories)
            for k in range(trajectories.shape[0]):
                for t in range(trajectories.shape[1]):
                    x_local = trajectories[k, t, 0]
                    y_local = trajectories[k, t, 1]
                    # Inverse rotation
                    x_world = x_local * cos_h - y_local * sin_h + ref_x
                    y_world = x_local * sin_h + y_local * cos_h + ref_y
                    traj_world[k, t, 0] = x_world
                    traj_world[k, t, 1] = y_world

            return traj_world, confidence, applicability

    def _extract_features(self, scenario) -> Optional[Dict[str, np.ndarray]]:
        """Extract model input features from Scenario proto."""
        tracks = list(scenario.tracks)
        if not tracks:
            return None

        # Find SDC: sdc_track_index is an array index, NOT a track object ID
        sdc_idx = scenario.sdc_track_index
        if sdc_idx < 0 or sdc_idx >= len(tracks):
            sdc_idx = 0

        sdc_track = tracks[sdc_idx]
        current_ts = scenario.current_time_index

        # Reference state
        ref_state = (
            sdc_track.states[current_ts] if current_ts < len(sdc_track.states) else None
        )
        if ref_state is None or not ref_state.valid:
            return None

        ref_x, ref_y = ref_state.center_x, ref_state.center_y
        if abs(ref_x) < 0.001 and abs(ref_y) < 0.001:
            return None
        ref_h = ref_state.heading
        cos_h, sin_h = np.cos(-ref_h), np.sin(-ref_h)

        def normalize_coords(x, y):
            dx, dy = x - ref_x, y - ref_y
            return dx * cos_h - dy * sin_h, dx * sin_h + dy * cos_h

        def normalize_heading(h):
            rel_h = h - ref_h
            while rel_h > np.pi:
                rel_h -= 2 * np.pi
            while rel_h < -np.pi:
                rel_h += 2 * np.pi
            return rel_h

        # Ego history
        ego_history = np.zeros((self.HISTORY_LENGTH, 4), dtype=np.float32)
        for t in range(self.HISTORY_LENGTH):
            if t < len(sdc_track.states) and sdc_track.states[t].valid:
                state = sdc_track.states[t]
                x, y = normalize_coords(state.center_x, state.center_y)
                ego_history[t, 0] = x
                ego_history[t, 1] = y
                ego_history[t, 2] = normalize_heading(state.heading)
                ego_history[t, 3] = np.sqrt(state.velocity_x**2 + state.velocity_y**2)

        # Agent states
        agent_states = np.zeros(
            (self.MAX_AGENTS, self.HISTORY_LENGTH, 4), dtype=np.float32
        )
        agent_count = 0
        for i, track in enumerate(tracks):
            if i == sdc_idx or agent_count >= self.MAX_AGENTS:
                continue
            has_valid = False
            for t in range(self.HISTORY_LENGTH):
                if t < len(track.states) and track.states[t].valid:
                    state = track.states[t]
                    x, y = normalize_coords(state.center_x, state.center_y)
                    agent_states[agent_count, t, 0] = x
                    agent_states[agent_count, t, 1] = y
                    agent_states[agent_count, t, 2] = normalize_heading(state.heading)
                    agent_states[agent_count, t, 3] = np.sqrt(
                        state.velocity_x**2 + state.velocity_y**2
                    )
                    has_valid = True
            if has_valid:
                agent_count += 1

        # Lane centers [L, P, 2]
        lane_centers = np.zeros((self.MAX_LANES, self.LANE_POINTS, 2), dtype=np.float32)
        lane_count = 0
        for map_feature in scenario.map_features:
            if lane_count >= self.MAX_LANES:
                break
            if map_feature.HasField("lane"):
                polyline = list(map_feature.lane.polyline)
                if polyline:
                    indices = np.linspace(
                        0, len(polyline) - 1, self.LANE_POINTS
                    ).astype(int)
                    for p_idx, src_idx in enumerate(indices):
                        x, y = normalize_coords(
                            polyline[src_idx].x, polyline[src_idx].y
                        )
                        lane_centers[lane_count, p_idx, 0] = x
                        lane_centers[lane_count, p_idx, 1] = y
                    lane_count += 1

        # Normalize by scale
        ego_history[:, :2] /= self.TRAJECTORY_SCALE
        agent_states[:, :, :2] /= self.TRAJECTORY_SCALE
        lane_centers /= self.TRAJECTORY_SCALE

        return {
            "ego_history": ego_history,
            "agent_states": agent_states,
            "lane_centers": lane_centers,
            "ref_x": ref_x,
            "ref_y": ref_y,
            "ref_h": ref_h,
        }


def get_tier_for_rule(rule_id: str) -> str:
    """Get tier name for a rule ID."""
    if rule_id in TIER_0_SAFETY:
        return "Safety"
    elif rule_id in TIER_1_LEGAL:
        return "Legal"
    elif rule_id in TIER_2_ROAD:
        return "Road"
    elif rule_id in TIER_3_COMFORT:
        return "Comfort"
    else:
        return "Other"


def prepare_scenario_data(scenario) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prepare scenario data for RECTOR inference."""
    current_ts = 10

    # Find ego track: sdc_track_index is an array index, NOT a track object ID
    ego_idx = scenario.sdc_track_index
    if 0 <= ego_idx < len(scenario.tracks):
        ego_track = scenario.tracks[ego_idx]
    else:
        ego_track = None

    if ego_track is None:
        return None, None, None

    # Ego history [11, 4]: x, y, heading, speed
    ego_history = np.zeros((11, 4), dtype=np.float32)
    for t in range(11):
        if t < len(ego_track.states) and ego_track.states[t].valid:
            s = ego_track.states[t]
            speed = np.sqrt(s.velocity_x**2 + s.velocity_y**2)
            ego_history[t] = [s.center_x, s.center_y, s.heading, speed]

    # Agent histories [N, 11, 4]
    agent_histories = []
    for track in scenario.tracks:
        if track.id == ego_idx:
            continue
        if len(agent_histories) >= 30:
            break

        hist = np.zeros((11, 4), dtype=np.float32)
        has_valid = False
        for t in range(11):
            if t < len(track.states) and track.states[t].valid:
                s = track.states[t]
                speed = np.sqrt(s.velocity_x**2 + s.velocity_y**2)
                hist[t] = [s.center_x, s.center_y, s.heading, speed]
                has_valid = True

        if has_valid:
            agent_histories.append(hist)

    agent_histories = (
        np.array(agent_histories) if agent_histories else np.zeros((0, 11, 4))
    )

    # Lane points [M, 2]
    ego_x = ego_history[current_ts, 0]
    ego_y = ego_history[current_ts, 1]

    best_lane = None
    best_dist = float("inf")

    for feature in scenario.map_features:
        if feature.HasField("lane"):
            points = np.array([[p.x, p.y] for p in feature.lane.polyline])
            if len(points) > 0:
                dists = np.sqrt(
                    (points[:, 0] - ego_x) ** 2 + (points[:, 1] - ego_y) ** 2
                )
                min_dist = dists.min()
                if min_dist < best_dist:
                    best_dist = min_dist
                    best_lane = points

    if best_lane is None:
        best_lane = np.array([[ego_x + i * 2, ego_y] for i in range(50)])

    # Resample to 50 points
    if len(best_lane) > 50:
        indices = np.linspace(0, len(best_lane) - 1, 50).astype(int)
        best_lane = best_lane[indices]
    elif len(best_lane) < 50:
        # Pad
        pad_len = 50 - len(best_lane)
        last_point = best_lane[-1]
        padding = np.tile(last_point, (pad_len, 1))
        best_lane = np.vstack([best_lane, padding])

    return ego_history, agent_histories, best_lane.astype(np.float32)


def run_evaluation(
    checkpoint_path: str,
    data_dir: str,
    output_path: str,
    max_scenarios: int = 1000,
    use_predictions: bool = True,
    device: str = "cpu",
):
    """
    Run full rule evaluation.

    Args:
        checkpoint_path: Path to RECTOR checkpoint
        data_dir: Directory with augmented TFRecords
        output_path: Path to save results
        max_scenarios: Maximum scenarios to evaluate
        use_predictions: If True, use RECTOR predictions; if False, use GT
        device: Device for inference
    """
    log.info("=" * 60)
    log.info("Full Rule Evaluation for RECTOR")
    log.info("=" * 60)
    log.info(f"Checkpoint: {checkpoint_path}")
    log.info(f"Data dir: {data_dir}")
    log.info(f"Max scenarios: {max_scenarios}")
    log.info(f"Use predictions: {use_predictions}")

    # Initialize components
    reader = AugmentedTFRecordReader(data_dir)
    context_builder = ScenarioContextBuilder()

    if use_predictions:
        inference = RECTORInference(checkpoint_path, device)
    else:
        inference = None

    # Initialize rule executor
    rule_executor = RuleExecutor()
    rule_executor.register_all_rules()
    log.info(f"Registered {len(rule_executor.rules)} rules")

    # Initialize tier stats
    results = EvaluationResults(
        checkpoint=checkpoint_path,
        total_scenarios=0,
        elapsed_time=0,
        tier_stats={
            "Safety": TierStats("Safety"),
            "Legal": TierStats("Legal"),
            "Road": TierStats("Road"),
            "Comfort": TierStats("Comfort"),
        },
    )

    start_time = time.time()

    # Process scenarios
    for scenario_id, scenario, example in reader.iter_scenarios(max_scenarios):
        try:
            if use_predictions:
                # Run RECTOR inference directly from scenario
                trajectories, confidences, _ = inference.predict_from_scenario(scenario)

                if trajectories is None:
                    continue

                # Use best mode
                best_mode = np.argmax(confidences)
                predicted_traj = trajectories[best_mode]  # [T, 2] in world coords
            else:
                predicted_traj = None

            # Build scenario context
            ctx = context_builder.build(scenario, predicted_traj)

            # Run rule evaluation
            scenario_result = rule_executor.evaluate(ctx)

            # Aggregate results
            for rule_result in scenario_result.rule_results:
                tier = get_tier_for_rule(rule_result.rule_id)
                if tier not in results.tier_stats:
                    continue

                tier_stats = results.tier_stats[tier]

                if rule_result.rule_id not in tier_stats.rule_stats:
                    tier_stats.rule_stats[rule_result.rule_id] = {
                        "applicable": 0,
                        "violations": 0,
                        "severity": 0.0,
                    }

                rule_stats = tier_stats.rule_stats[rule_result.rule_id]

                if rule_result.applies:
                    tier_stats.total_applicable += 1
                    rule_stats["applicable"] += 1

                    if rule_result.has_violation:
                        tier_stats.total_violations += 1
                        tier_stats.total_severity += rule_result.severity
                        rule_stats["violations"] += 1
                        rule_stats["severity"] += rule_result.severity

            # Save per-scenario result
            results.per_scenario_results.append(
                {
                    "scenario_id": scenario_id,
                    "n_violations": scenario_result.n_violations,
                    "n_applicable": scenario_result.n_applicable,
                    "total_severity": scenario_result.total_severity,
                }
            )

            results.total_scenarios += 1

            if results.total_scenarios % 100 == 0:
                elapsed = time.time() - start_time
                log.info(
                    f"Processed {results.total_scenarios} scenarios in {elapsed:.1f}s"
                )

        except Exception as e:
            log.warning(f"Error processing scenario {scenario_id}: {e}")
            continue

    results.elapsed_time = time.time() - start_time

    # Log summary
    log.info("=" * 60)
    log.info("Evaluation Complete")
    log.info("=" * 60)
    log.info(f"Total scenarios: {results.total_scenarios}")
    log.info(f"Elapsed time: {results.elapsed_time:.1f}s")
    log.info("")
    log.info("Tier-wise Results:")
    for tier_name, stats in results.tier_stats.items():
        log.info(f"  {tier_name}:")
        log.info(f"    Applicable: {stats.total_applicable}")
        log.info(f"    Violations: {stats.total_violations}")
        log.info(f"    Violation Rate: {stats.violation_rate:.2f}%")

    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results.to_dict(), f, indent=2)
    log.info(f"Results saved to {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Full Rule Evaluation for RECTOR")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/workspace/models/RECTOR/models/best.pt",
        help="Path to RECTOR checkpoint",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/workspace/data/WOMD/datasets/waymo_open_dataset/motion_v_1_3_0/processed/augmented/scenario/validation_interactive",
        help="Directory with augmented TFRecords",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/workspace/output/rule_evaluation/full_rule_evaluation_results.json",
        help="Output path for results",
    )
    parser.add_argument(
        "--max-scenarios", type=int, default=1000, help="Maximum scenarios to evaluate"
    )
    parser.add_argument(
        "--use-gt",
        action="store_true",
        help="Use ground truth trajectories instead of predictions",
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device for inference"
    )

    args = parser.parse_args()

    run_evaluation(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        output_path=args.output,
        max_scenarios=args.max_scenarios,
        use_predictions=not args.use_gt,
        device=args.device,
    )


if __name__ == "__main__":
    main()
