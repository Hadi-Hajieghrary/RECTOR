#!/usr/bin/env python3
"""
RECTOR V2 Receding Horizon Movie Generator with Rule Violation Display.

At each timestep, the model predicts the next 5 seconds of trajectory.
Shows rule applicability and violation metrics in a side panel.

Usage:
    python visualization/generate_receding_movies.py --num_scenarios 5
"""

import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import argparse
import glob
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.interpolate import CubicSpline, UnivariateSpline

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec

import tensorflow as tf

tf.config.set_visible_devices([], "GPU")

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, "/workspace/data/WOMD")
sys.path.insert(0, "/workspace/data")

from models.rule_aware_generator import RuleAwareGenerator

# Try to import Waymo protos
try:
    from waymo_open_dataset.protos import scenario_pb2

    WAYMO_AVAILABLE = True
except ImportError:
    WAYMO_AVAILABLE = False
    print("Warning: waymo_open_dataset not available")

# Import rule constants
try:
    from waymo_rule_eval.rules.rule_constants import (
        RULE_IDS,
        NUM_RULES,
        RULE_INDEX_MAP,
        TIER_0_SAFETY,
        TIER_1_LEGAL,
        TIER_2_ROAD,
        TIER_3_COMFORT,
        TIER_DEFINITIONS,
        get_rule_tier,
    )

    RULES_AVAILABLE = True
except ImportError:
    RULES_AVAILABLE = False
    print("Warning: rule constants not available")
    RULE_IDS = []
    NUM_RULES = 28
    RULE_INDEX_MAP = {}


# Colors matching M2I style
COLORS = {
    "ego": "#e74c3c",  # Red
    "ego_future_gt": "#27ae60",  # Green
    "planned_traj": "#3498db",  # Blue - planned trajectory
    "planned_fade": "#85c1e9",  # Light blue - fading planned
    "other_modes": "#f39c12",  # Orange - other modes
    "agents": "#9b59b6",  # Purple
    "pedestrian": "#2ecc71",  # Green
    "cyclist": "#e67e22",  # Dark orange
    "lane": "#7f8c8d",  # Gray
    "road_edge": "#2c3e50",  # Dark
    "crosswalk": "#f1c40f",  # Yellow
}

# Tier colors for rule bars
TIER_COLORS = {
    0: "#e74c3c",  # Red - Safety
    1: "#f39c12",  # Orange - Legal
    2: "#3498db",  # Blue - Road
    3: "#2ecc71",  # Green - Comfort
}

# Rule short names for display
RULE_SHORT_NAMES = {
    "L0.R2": "Long. Distance",
    "L0.R3": "Lat. Clearance",
    "L0.R4": "Crosswalk Occ.",
    "L10.R1": "Veh. Collision",
    "L10.R2": "VRU Clearance",
    "L5.R1": "Traffic Signal",
    "L5.R2": "Priority",
    "L7.R4": "Speed Limit",
    "L8.R1": "Red Light",
    "L8.R2": "Stop Sign",
    "L8.R3": "Crosswalk Yield",
    "L8.R5": "Wrong Way",
    "L3.R3": "Drivable Surface",
    "L7.R3": "Lane Departure",
    "L1.R1": "Accel Smooth",
    "L1.R2": "Brake Smooth",
    "L1.R3": "Steer Smooth",
    "L1.R4": "Speed Consist.",
    "L1.R5": "Lane Change",
    "L4.R3": "Left Turn Gap",
    "L5.R3": "Parking",
    "L5.R4": "School Zone",
    "L5.R5": "Construction",
    "L6.R1": "Coop. Lane Chg",
    "L6.R2": "Following Dist",
    "L6.R3": "Intersection",
    "L6.R4": "Ped. Interact",
    "L6.R5": "Cyclist Inter.",
}

TRAJECTORY_SCALE = 50.0
HISTORY_LENGTH = 11
FUTURE_LENGTH = 50
TOTAL_TIMESTEPS = 91  # 9.1 seconds at 10Hz
DT = 0.1  # Time step in seconds (10Hz)


def smooth_trajectory(traj: np.ndarray, n_anchors: int = 5) -> np.ndarray:
    """
    Post-process raw model trajectory to remove zig-zag artifacts.

    The Transformer decoder generates all 50 steps in parallel as cumulated
    deltas.  This produces high-frequency lateral oscillation that is
    kinematically infeasible for a vehicle.

    Strategy: sample a small number of *anchor* points from the raw
    trajectory (including start and end), then fit a smooth cubic spline
    through them and resample at the original 10 Hz rate.

    Unlike Gaussian smoothing, this:
      - Preserves the exact endpoint (no distance shrinkage)
      - Preserves the overall trajectory shape / curvature
      - Produces a kinematically smooth path between anchors

    Args:
        traj: [T, 4] trajectory (x, y, heading, speed) in metres,
              ego-centric frame (origin at current ego).
        n_anchors: Number of waypoints to keep from the raw trajectory
                   (including first and last).  5 works well: produces
                   very smooth curves while following the model's intent.

    Returns:
        smoothed: [T, 4] smoothed trajectory (same shape, same endpoints).
    """
    T = len(traj)
    if T < 4:
        return traj.copy()

    smoothed = traj.copy()

    # 1. Select anchor indices (always includes first and last)
    anchor_idx = np.unique(np.linspace(0, T - 1, n_anchors).astype(int))
    t_anchor = anchor_idx * DT
    t_full = np.arange(T) * DT

    # 2. Fit cubic splines through anchors
    sp_x = CubicSpline(t_anchor, traj[anchor_idx, 0], bc_type="clamped")
    sp_y = CubicSpline(t_anchor, traj[anchor_idx, 1], bc_type="clamped")
    smoothed[:, 0] = sp_x(t_full)
    smoothed[:, 1] = sp_y(t_full)

    # 3. Recompute heading from tangent of smoothed path
    dx = np.gradient(smoothed[:, 0])
    dy = np.gradient(smoothed[:, 1])
    displacement = np.sqrt(dx**2 + dy**2)
    mask = displacement > 1e-3
    if mask.any():
        smoothed[mask, 2] = np.arctan2(dy[mask], dx[mask])
        if not mask[0]:
            smoothed[0, 2] = traj[0, 2]
        for t in range(1, T):
            if not mask[t]:
                smoothed[t, 2] = smoothed[t - 1, 2]

    # 4. Recompute speed from smoothed positions
    if T > 1:
        dx_dt = np.diff(smoothed[:, 0]) / DT
        dy_dt = np.diff(smoothed[:, 1]) / DT
        smoothed[1:, 3] = np.sqrt(dx_dt**2 + dy_dt**2)
        smoothed[0, 3] = smoothed[1, 3]

    return smoothed


def compute_kinematic_violations(trajectory: np.ndarray) -> np.ndarray:
    """
    Compute violation estimates from trajectory kinematics.

    When full proxy evaluation isn't available, estimate violations
    from acceleration, jerk, and speed patterns.

    Args:
        trajectory: [T, 4] trajectory (x, y, heading, velocity)

    Returns:
        violations: [NUM_RULES] estimated violation levels
    """
    violations = np.zeros(NUM_RULES)

    T = len(trajectory)
    if T < 3:
        return violations

    x, y = trajectory[:, 0], trajectory[:, 1]
    heading = trajectory[:, 2] if trajectory.shape[1] > 2 else np.zeros(T)

    # Compute velocities from positions
    dx = np.diff(x) / DT
    dy = np.diff(y) / DT
    speeds = np.sqrt(dx**2 + dy**2)

    # Compute accelerations
    if len(speeds) > 1:
        accel = np.diff(speeds) / DT
        max_accel = np.max(np.abs(accel))

        # L1.R1 - Smooth acceleration (threshold ~3 m/s²)
        if "L1.R1" in RULE_INDEX_MAP:
            idx = RULE_INDEX_MAP["L1.R1"]
            violations[idx] = min(1.0, max(0, max_accel - 2.0) / 5.0)

        # L1.R2 - Smooth braking (threshold ~4 m/s²)
        if "L1.R2" in RULE_INDEX_MAP:
            idx = RULE_INDEX_MAP["L1.R2"]
            max_decel = np.min(accel)  # Most negative
            violations[idx] = min(1.0, max(0, -max_decel - 3.0) / 5.0)

    # Compute lateral acceleration (steering)
    if len(heading) > 1:
        heading_rate = np.diff(heading)
        # Normalize heading rate
        heading_rate = np.arctan2(np.sin(heading_rate), np.cos(heading_rate))
        max_yaw_rate = np.max(np.abs(heading_rate)) / DT

        # L1.R3 - Smooth steering (threshold ~0.5 rad/s)
        if "L1.R3" in RULE_INDEX_MAP:
            idx = RULE_INDEX_MAP["L1.R3"]
            violations[idx] = min(1.0, max(0, max_yaw_rate - 0.3) / 1.0)

    # Speed consistency
    if len(speeds) > 0:
        speed_std = np.std(speeds)
        mean_speed = np.mean(speeds)

        # L1.R4 - Speed consistency
        if "L1.R4" in RULE_INDEX_MAP:
            idx = RULE_INDEX_MAP["L1.R4"]
            violations[idx] = min(1.0, speed_std / (mean_speed + 1.0))

        # L7.R4 - Speed limit (assume 25 m/s ~ 90 km/h limit)
        if "L7.R4" in RULE_INDEX_MAP:
            idx = RULE_INDEX_MAP["L7.R4"]
            max_speed = np.max(speeds)
            violations[idx] = min(1.0, max(0, max_speed - 25.0) / 10.0)

    # Compute jerk for comfort
    if len(speeds) > 2:
        jerk = np.diff(accel) / DT
        max_jerk = np.max(np.abs(jerk))

        # L1.R5 - Lane change smoothness (use jerk as proxy)
        if "L1.R5" in RULE_INDEX_MAP:
            idx = RULE_INDEX_MAP["L1.R5"]
            violations[idx] = min(1.0, max_jerk / 20.0)

    return violations


def fit_trajectory_spline(
    waypoints: np.ndarray,
    dt: float = DT,
    upsample_factor: int = 3,
    smoothing: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit a smooth spline to trajectory waypoints considering timestamps.

    The network outputs waypoints at fixed time intervals (10Hz = 0.1s).
    This function fits a cubic spline for smooth visualization.

    Args:
        waypoints: [T, 4] trajectory (x, y, heading, velocity) at 10Hz
        dt: Time step between waypoints (0.1s for 10Hz)
        upsample_factor: How many times to upsample for smoother curve
        smoothing: Smoothing factor (0 = interpolate exactly through points)

    Returns:
        smooth_xy: [T*upsample_factor, 2] smoothed (x, y) positions
        smooth_times: [T*upsample_factor] corresponding timestamps
    """
    T = len(waypoints)
    if T < 4:
        # Not enough points for cubic spline
        return waypoints[:, :2], np.arange(T) * dt

    # Create timestamps for each waypoint
    times = np.arange(T) * dt  # [0, 0.1, 0.2, ..., 4.9] for 50 points

    # Extract x, y positions
    x = waypoints[:, 0]
    y = waypoints[:, 1]

    try:
        if smoothing > 0:
            # Use smoothing spline (approximates but smooths noise)
            # k=3 for cubic, s is smoothing factor
            spline_x = UnivariateSpline(times, x, k=3, s=smoothing)
            spline_y = UnivariateSpline(times, y, k=3, s=smoothing)
        else:
            # Use interpolating cubic spline (passes through all points)
            spline_x = CubicSpline(times, x, bc_type="natural")
            spline_y = CubicSpline(times, y, bc_type="natural")

        # Create upsampled time array
        smooth_times = np.linspace(times[0], times[-1], T * upsample_factor)

        # Evaluate splines at upsampled times
        smooth_x = spline_x(smooth_times)
        smooth_y = spline_y(smooth_times)

        smooth_xy = np.stack([smooth_x, smooth_y], axis=1)

        return smooth_xy, smooth_times

    except Exception as e:
        # Fallback: return original points
        return waypoints[:, :2], times


def fit_heading_spline(
    waypoints: np.ndarray,
    dt: float = DT,
) -> np.ndarray:
    """
    Compute smooth heading from spline derivatives.

    Instead of using raw heading from network, compute heading
    from the tangent of the smoothed trajectory.

    Args:
        waypoints: [T, 4] trajectory
        dt: Time step

    Returns:
        smooth_headings: [T] heading at each timestep
    """
    T = len(waypoints)
    if T < 4:
        return waypoints[:, 2]

    times = np.arange(T) * dt
    x = waypoints[:, 0]
    y = waypoints[:, 1]

    try:
        spline_x = CubicSpline(times, x, bc_type="natural")
        spline_y = CubicSpline(times, y, bc_type="natural")

        # Get derivatives (velocity components)
        dx = spline_x(times, 1)  # First derivative
        dy = spline_y(times, 1)

        # Heading from atan2
        headings = np.arctan2(dy, dx)

        return headings

    except Exception:
        return waypoints[:, 2]


def parse_args():
    parser = argparse.ArgumentParser(
        description="RECTOR V2 Receding Horizon Movies with Rule Display"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/workspace/models/RECTOR/models/best.pt",
        help="Path to checkpoint",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/workspace/data/WOMD/datasets/waymo_open_dataset/motion_v_1_3_0/processed/augmented/scenario/validation_interactive",
        help="Path to data",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/workspace/models/RECTOR/movies/receding",
        help="Output directory",
    )
    parser.add_argument(
        "--num_scenarios", type=int, default=5, help="Number of scenarios"
    )
    parser.add_argument("--fps", type=int, default=10, help="Frames per second")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (omit for different scenarios each run)",
    )
    parser.add_argument(
        "--view_range", type=float, default=80, help="View range in meters (full scene)"
    )

    # Trajectory stabilization
    parser.add_argument(
        "--ema_alpha",
        type=float,
        default=0.4,
        help="EMA blending factor for trajectory smoothing (0=max smooth, 1=no smooth)",
    )
    parser.add_argument(
        "--no_ema",
        action="store_true",
        help="Disable EMA trajectory smoothing (raw predictions)",
    )
    parser.add_argument(
        "--n_anchors",
        type=int,
        default=5,
        help="Number of anchor waypoints for cubic-spline trajectory smoothing (0=disable, 5=default)",
    )
    parser.add_argument(
        "--min_ego_speed",
        type=float,
        default=2.0,
        help="Minimum average ego speed (m/s) to select a scenario. Filters out parked/static scenarios.",
    )

    # Model config
    parser.add_argument(
        "--m2i_checkpoint",
        type=str,
        default="/workspace/models/pretrained/m2i/models/relation_v2v/model.25.bin",
    )

    return parser.parse_args()


class ScenarioLoader:
    """Load full scenario data for receding horizon visualization."""

    def __init__(self, tfrecord_paths: List[str]):
        self.tfrecord_paths = tfrecord_paths
        self.scenarios = []
        self._load_scenarios()

    def _load_scenarios(self):
        """Load all scenarios from TFRecords."""
        for path in self.tfrecord_paths:
            try:
                dataset = tf.data.TFRecordDataset([path])
                for raw_record in dataset:
                    scenario_data = self._parse_record(raw_record.numpy())
                    if scenario_data is not None:
                        self.scenarios.append(scenario_data)
            except Exception as e:
                print(f"  Warning: Failed to load {path}: {e}")

    def _parse_record(self, raw_bytes) -> Optional[Dict]:
        """Parse a TFRecord into scenario data."""
        # Try augmented format first
        try:
            example = tf.train.Example()
            example.ParseFromString(raw_bytes)
            features = example.features.feature

            if "scenario/proto" in features:
                scenario_bytes = features["scenario/proto"].bytes_list.value[0]
                scenario = scenario_pb2.Scenario()
                scenario.ParseFromString(scenario_bytes)
                return self._extract_scenario(scenario)
        except:
            pass

        # Try raw Scenario format
        try:
            scenario = scenario_pb2.Scenario()
            scenario.ParseFromString(raw_bytes)
            if scenario.scenario_id:
                return self._extract_scenario(scenario)
        except:
            pass

        return None

    def _extract_scenario(self, scenario) -> Optional[Dict]:
        """Extract full scenario data for all 91 timesteps."""
        tracks = list(scenario.tracks)
        if not tracks:
            return None

        # Find SDC: sdc_track_index is an array index, NOT a track object ID
        sdc_idx = scenario.sdc_track_index
        if sdc_idx < 0 or sdc_idx >= len(tracks):
            sdc_idx = 0

        sdc_track = tracks[sdc_idx]

        # Need full 91 timesteps
        if len(sdc_track.states) < TOTAL_TIMESTEPS:
            return None

        # Get reference frame from current timestep (t=10)
        current_ts = 10
        ref_state = sdc_track.states[current_ts]
        if not ref_state.valid:
            return None

        ref_x, ref_y = ref_state.center_x, ref_state.center_y
        ref_h = ref_state.heading
        cos_h, sin_h = np.cos(-ref_h), np.sin(-ref_h)

        def to_local(x, y):
            dx, dy = x - ref_x, y - ref_y
            return dx * cos_h - dy * sin_h, dx * sin_h + dy * cos_h

        def norm_heading(h):
            rel_h = h - ref_h
            while rel_h > np.pi:
                rel_h -= 2 * np.pi
            while rel_h < -np.pi:
                rel_h += 2 * np.pi
            return rel_h

        # Extract ego trajectory (all 91 timesteps)
        ego_traj = np.zeros((TOTAL_TIMESTEPS, 5), dtype=np.float32)  # x, y, h, v, valid
        for t in range(TOTAL_TIMESTEPS):
            if t < len(sdc_track.states) and sdc_track.states[t].valid:
                state = sdc_track.states[t]
                x, y = to_local(state.center_x, state.center_y)
                ego_traj[t, 0] = x
                ego_traj[t, 1] = y
                ego_traj[t, 2] = norm_heading(state.heading)
                ego_traj[t, 3] = np.sqrt(state.velocity_x**2 + state.velocity_y**2)
                ego_traj[t, 4] = 1.0

        # Extract other agents (all timesteps)
        # IMPORTANT: Match training pipeline agent ordering —
        # only include agents that have at least one valid state during
        # the history period (t=0..10). This matches WaymoDataset._extract_features()
        # which iterates range(self.history_length) to check validity.
        max_agents = 32
        agent_trajs = np.zeros(
            (max_agents, TOTAL_TIMESTEPS, 6), dtype=np.float32
        )  # x, y, h, v, valid, type
        agent_count = 0

        history_end = 11  # history_length used in training
        for i, track in enumerate(tracks):
            if i == sdc_idx or agent_count >= max_agents:
                continue

            # Check validity ONLY during history period (matching training)
            has_valid_history = False
            for t in range(min(history_end, len(track.states))):
                if track.states[t].valid:
                    has_valid_history = True
                    break

            if not has_valid_history:
                continue

            # Store ALL timesteps for this agent (for receding horizon visualization)
            for t in range(min(TOTAL_TIMESTEPS, len(track.states))):
                if track.states[t].valid:
                    state = track.states[t]
                    x, y = to_local(state.center_x, state.center_y)
                    agent_trajs[agent_count, t, 0] = x
                    agent_trajs[agent_count, t, 1] = y
                    agent_trajs[agent_count, t, 2] = norm_heading(state.heading)
                    agent_trajs[agent_count, t, 3] = np.sqrt(
                        state.velocity_x**2 + state.velocity_y**2
                    )
                    agent_trajs[agent_count, t, 4] = 1.0
                    agent_trajs[agent_count, t, 5] = (
                        track.object_type
                    )  # 1=vehicle, 2=ped, 3=cyclist

            agent_count += 1

        # Extract lanes
        max_lanes = 128
        lanes = []
        road_edges = []
        crosswalks = []

        for map_feature in scenario.map_features:
            if map_feature.HasField("lane"):
                polyline = [(p.x, p.y) for p in map_feature.lane.polyline]
                if polyline:
                    lane_local = np.array([to_local(p[0], p[1]) for p in polyline])
                    lanes.append(lane_local)
            elif map_feature.HasField("road_edge"):
                polyline = [(p.x, p.y) for p in map_feature.road_edge.polyline]
                if polyline:
                    edge_local = np.array([to_local(p[0], p[1]) for p in polyline])
                    road_edges.append(edge_local)
            elif map_feature.HasField("crosswalk"):
                polygon = [(p.x, p.y) for p in map_feature.crosswalk.polygon]
                if polygon:
                    cw_local = np.array([to_local(p[0], p[1]) for p in polygon])
                    crosswalks.append(cw_local)

        # Compute ego displacement stats for filtering
        valid_mask = ego_traj[:, 4] > 0
        if valid_mask.sum() > 1:
            valid_pos = ego_traj[valid_mask, :2]
            total_dist = np.sqrt(np.sum(np.diff(valid_pos, axis=0) ** 2, axis=1)).sum()
            duration = (valid_mask.sum() - 1) * DT
            avg_speed = total_dist / max(duration, 1e-6)
        else:
            total_dist = 0.0
            avg_speed = 0.0

        return {
            "scenario_id": scenario.scenario_id,
            "ego_traj": ego_traj,
            "agent_trajs": agent_trajs,
            "agent_count": agent_count,
            "lanes": lanes[:max_lanes],
            "road_edges": road_edges,
            "crosswalks": crosswalks,
            "ego_total_dist": total_dist,
            "ego_avg_speed": avg_speed,
        }

    def __len__(self):
        return len(self.scenarios)

    def __getitem__(self, idx):
        return self.scenarios[idx]


def prepare_model_input(
    scenario: Dict, current_t: int, device: torch.device
) -> Dict[str, torch.Tensor]:
    """
    Prepare model input for prediction from current timestep.

    At timestep t, we use history [t-10:t+1] and predict [t+1:t+51].

    All coordinates are normalized to the ego-centric frame at current_t:
    - Origin at ego's position at current_t
    - X-axis aligned with ego's heading at current_t
    - Headings are relative to ego's heading
    This matches the training pipeline (WaymoDataset._extract_features).
    """
    ego_traj = scenario["ego_traj"]
    agent_trajs = scenario["agent_trajs"]

    # Reference pose: ego at current_t
    if ego_traj[current_t, 4] > 0:
        ref_x, ref_y = ego_traj[current_t, 0], ego_traj[current_t, 1]
        ref_h = ego_traj[current_t, 2]
    else:
        # Fallback: find nearest valid timestep
        ref_x, ref_y, ref_h = 0.0, 0.0, 0.0
        for dt in range(10):
            for sign in [0, -1, 1]:
                t_try = current_t + sign * dt
                if 0 <= t_try < ego_traj.shape[0] and ego_traj[t_try, 4] > 0:
                    ref_x, ref_y = ego_traj[t_try, 0], ego_traj[t_try, 1]
                    ref_h = ego_traj[t_try, 2]
                    break
            else:
                continue
            break

    cos_h, sin_h = np.cos(-ref_h), np.sin(-ref_h)

    def normalize_coords(x, y):
        """Transform from global to ego-centric frame."""
        dx, dy = x - ref_x, y - ref_y
        return dx * cos_h - dy * sin_h, dx * sin_h + dy * cos_h

    def normalize_heading(h):
        """Normalize heading relative to ego heading."""
        rel_h = h - ref_h
        while rel_h > np.pi:
            rel_h -= 2 * np.pi
        while rel_h < -np.pi:
            rel_h += 2 * np.pi
        return rel_h

    # History window: [current_t - 10, current_t] inclusive = 11 frames
    hist_start = max(0, current_t - 10)
    hist_end = current_t + 1

    # Ego history - normalized to ego-centric frame
    ego_history = np.zeros((HISTORY_LENGTH, 4), dtype=np.float32)
    for i, t in enumerate(range(hist_start, hist_end)):
        idx = i + (HISTORY_LENGTH - (hist_end - hist_start))
        if t >= 0 and t < ego_traj.shape[0] and ego_traj[t, 4] > 0:
            x, y = normalize_coords(ego_traj[t, 0], ego_traj[t, 1])
            ego_history[idx, 0] = x
            ego_history[idx, 1] = y
            ego_history[idx, 2] = normalize_heading(ego_traj[t, 2])
            ego_history[idx, 3] = ego_traj[t, 3]  # speed is scalar, no transform

    # Agent states - normalized to ego-centric frame
    max_agents = 32
    agent_states = np.zeros((max_agents, HISTORY_LENGTH, 4), dtype=np.float32)
    for a in range(min(scenario["agent_count"], max_agents)):
        for i, t in enumerate(range(hist_start, hist_end)):
            idx = i + (HISTORY_LENGTH - (hist_end - hist_start))
            if t >= 0 and t < agent_trajs.shape[1] and agent_trajs[a, t, 4] > 0:
                x, y = normalize_coords(agent_trajs[a, t, 0], agent_trajs[a, t, 1])
                agent_states[a, idx, 0] = x
                agent_states[a, idx, 1] = y
                agent_states[a, idx, 2] = normalize_heading(agent_trajs[a, t, 2])
                agent_states[a, idx, 3] = agent_trajs[a, t, 3]  # speed

    # Lane centers - normalized to ego-centric frame
    # IMPORTANT: Match training pipeline — include ALL lanes in protobuf order
    # (no distance filtering). Training uses first max_lanes lanes without filtering.
    max_lanes = 64
    lane_points = 20
    lane_centers = np.zeros((max_lanes, lane_points, 2), dtype=np.float32)

    lane_idx = 0
    for lane in scenario["lanes"]:
        if lane_idx >= max_lanes:
            break
        indices = np.linspace(0, len(lane) - 1, lane_points).astype(int)
        for p_idx, src_idx in enumerate(indices):
            x, y = normalize_coords(lane[src_idx, 0], lane[src_idx, 1])
            lane_centers[lane_idx, p_idx, 0] = x
            lane_centers[lane_idx, p_idx, 1] = y
        lane_idx += 1

    # Normalize by trajectory scale (matches training pipeline)
    ego_history[:, :2] /= TRAJECTORY_SCALE
    agent_states[:, :, :2] /= TRAJECTORY_SCALE
    lane_centers /= TRAJECTORY_SCALE

    # Convert to tensors and add batch dimension
    return {
        "ego_history": torch.from_numpy(ego_history).unsqueeze(0).to(device),
        "agent_states": torch.from_numpy(agent_states).unsqueeze(0).to(device),
        "lane_centers": torch.from_numpy(lane_centers).unsqueeze(0).to(device),
    }


def draw_vehicle(
    ax,
    x,
    y,
    heading,
    length=4.5,
    width=2.0,
    color="red",
    alpha=1.0,
    zorder=10,
    label=None,
):
    """Draw a vehicle as a rotated rectangle."""
    corners = np.array(
        [
            [-length / 2, -width / 2],
            [length / 2, -width / 2],
            [length / 2, width / 2],
            [-length / 2, width / 2],
        ]
    )

    # Rotate
    cos_h, sin_h = np.cos(heading), np.sin(heading)
    rot = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
    corners = corners @ rot.T
    corners += np.array([x, y])

    polygon = plt.Polygon(
        corners,
        facecolor=color,
        edgecolor="black",
        linewidth=1,
        alpha=alpha,
        zorder=zorder,
        label=label,
    )
    ax.add_patch(polygon)

    # Heading arrow
    arrow_len = length * 0.4
    ax.arrow(
        x,
        y,
        arrow_len * np.cos(heading),
        arrow_len * np.sin(heading),
        head_width=0.4,
        head_length=0.2,
        fc="white",
        ec="black",
        zorder=zorder + 1,
        alpha=alpha,
    )


def draw_rule_panel(
    ax_rules, applicability: np.ndarray, violations: np.ndarray, current_t: int
):
    """
    Draw the rule violation panel.

    Args:
        ax_rules: Matplotlib axis for rule panel
        applicability: [NUM_RULES] applicability probabilities (0-1)
        violations: [NUM_RULES] violation costs (0-1)
        current_t: Current timestep
    """
    ax_rules.clear()
    ax_rules.set_xlim(0, 1.5)

    # Group rules by tier
    tier_rules = {
        0: TIER_0_SAFETY if RULES_AVAILABLE else [],
        1: TIER_1_LEGAL if RULES_AVAILABLE else [],
        2: TIER_2_ROAD if RULES_AVAILABLE else [],
        3: TIER_3_COMFORT if RULES_AVAILABLE else [],
    }
    tier_names = [
        "SAFETY (Tier 0)",
        "LEGAL (Tier 1)",
        "ROAD (Tier 2)",
        "COMFORT (Tier 3)",
    ]

    y_pos = 0
    y_positions = []
    rule_data = []

    # Count active rules per tier
    tier_active_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    tier_violation_counts = {0: 0, 1: 0, 2: 0, 3: 0}

    for tier in [0, 1, 2, 3]:
        rules = tier_rules[tier]
        if not rules:
            continue

        # Tier header
        y_positions.append(y_pos)
        rule_data.append(
            {
                "type": "header",
                "name": tier_names[tier],
                "tier": tier,
            }
        )
        y_pos += 1

        for rule_id in rules:
            if rule_id not in RULE_INDEX_MAP:
                continue

            idx = RULE_INDEX_MAP[rule_id]
            app = applicability[idx] if idx < len(applicability) else 0
            viol = violations[idx] if idx < len(violations) else 0

            # Count active rules
            if app > 0.3:  # Consider rule active if prob > 0.3
                tier_active_counts[tier] += 1
                if viol > 0.1:
                    tier_violation_counts[tier] += 1

            y_positions.append(y_pos)
            rule_data.append(
                {
                    "type": "rule",
                    "rule_id": rule_id,
                    "name": RULE_SHORT_NAMES.get(rule_id, rule_id),
                    "tier": tier,
                    "applicability": app,
                    "violation": viol,
                }
            )
            y_pos += 1

        y_pos += 0.5  # Gap between tiers

    ax_rules.set_ylim(-0.5, y_pos + 0.5)
    ax_rules.invert_yaxis()

    # Draw rules
    for i, (y, data) in enumerate(zip(y_positions, rule_data)):
        if data["type"] == "header":
            # Tier header with active count
            tier = data["tier"]
            active = tier_active_counts[tier]
            viol_count = tier_violation_counts[tier]
            header_text = f"{data['name']}"
            if active > 0:
                header_text += f" [{active} active"
                if viol_count > 0:
                    header_text += f", {viol_count} violated"
                header_text += "]"
            ax_rules.text(
                0.02,
                y,
                header_text,
                fontsize=8,
                fontweight="bold",
                color=TIER_COLORS[tier],
                va="center",
            )
            # Horizontal line under header
            ax_rules.axhline(
                y=y + 0.4,
                xmin=0,
                xmax=1,
                color=TIER_COLORS[tier],
                linewidth=0.5,
                alpha=0.3,
            )
        else:
            # Rule row
            app = data["applicability"]
            viol = data["violation"]
            tier = data["tier"]

            # Rule name
            name_alpha = 1.0 if app > 0.3 else 0.4
            ax_rules.text(
                0.02, y, data["name"], fontsize=7, va="center", alpha=name_alpha
            )

            # Applicability probability (as percentage)
            app_text = f"{app*100:.0f}%"
            app_color = TIER_COLORS[tier] if app > 0.3 else "gray"
            ax_rules.text(
                0.55,
                y,
                app_text,
                fontsize=6,
                va="center",
                color=app_color,
                alpha=0.9,
                ha="center",
            )

            # Violation bar and value (only if applicable)
            if app > 0.3:
                bar_width = 0.45
                bar_x = 0.7

                # Background bar
                ax_rules.barh(
                    y,
                    bar_width,
                    left=bar_x,
                    height=0.5,
                    color="#e0e0e0",
                    alpha=0.5,
                    edgecolor="gray",
                    linewidth=0.5,
                )

                # Violation fill
                if viol > 0.005:
                    # Color based on severity
                    if viol < 0.2:
                        bar_color = "#27ae60"  # Green - low
                    elif viol < 0.5:
                        bar_color = "#f1c40f"  # Yellow - medium
                    elif viol < 0.7:
                        bar_color = "#e67e22"  # Orange - high
                    else:
                        bar_color = "#c0392b"  # Red - severe

                    fill_width = min(viol, 1.0) * bar_width
                    ax_rules.barh(
                        y,
                        fill_width,
                        left=bar_x,
                        height=0.5,
                        color=bar_color,
                        alpha=0.9,
                    )

                # Violation value text
                viol_text = f"{viol:.3f}" if viol > 0 else "0"
                ax_rules.text(
                    bar_x + bar_width + 0.03,
                    y,
                    viol_text,
                    fontsize=6,
                    va="center",
                    alpha=0.9,
                    fontweight="bold" if viol > 0.3 else "normal",
                )
            else:
                # Show "N/A" for non-applicable rules
                ax_rules.text(
                    0.85,
                    y,
                    "N/A",
                    fontsize=6,
                    va="center",
                    color="gray",
                    alpha=0.4,
                    ha="center",
                )

    # Title with timestamp
    time_sec = (current_t - 10) / 10.0
    ax_rules.set_title(
        f"Rule Evaluation @ t={time_sec:+.1f}s\n" f"[Prob%] [Violation Cost 0-1]",
        fontsize=9,
        fontweight="bold",
        loc="left",
    )
    ax_rules.axis("off")

    ax_rules.set_title(
        f"Rule Violations (t={current_t})", fontsize=10, fontweight="bold"
    )
    ax_rules.axis("off")


def _ema_blend_trajectory(
    new_planned: np.ndarray,
    prev_planned: Optional[np.ndarray],
    alpha: float,
) -> np.ndarray:
    """
    Blend new trajectory prediction with the previous one using EMA.

    The previous trajectory is shifted by one timestep (since ego advanced)
    before blending, so corresponding physical times are aligned.

    Args:
        new_planned: [T, 4] new prediction in absolute coords (x, y, heading, speed)
        prev_planned: [T, 4] previous smoothed prediction (or None on first frame)
        alpha: blending weight for new prediction (0 = keep old, 1 = use new)

    Returns:
        blended: [T, 4] EMA-smoothed trajectory
    """
    if prev_planned is None:
        return new_planned.copy()

    T_new = len(new_planned)
    T_prev = len(prev_planned)

    # Shift previous trajectory by 1 step: prev[1] aligns with new[0], etc.
    # After shifting, we lose the first point and pad the end.
    if T_prev <= 1:
        return new_planned.copy()

    shifted_prev = prev_planned[1:]  # [T_prev-1, 4]

    # Match lengths
    T = min(T_new, len(shifted_prev))
    blended = new_planned.copy()

    # Blend x, y positions
    blended[:T, 0] = alpha * new_planned[:T, 0] + (1 - alpha) * shifted_prev[:T, 0]
    blended[:T, 1] = alpha * new_planned[:T, 1] + (1 - alpha) * shifted_prev[:T, 1]

    # Blend heading using circular interpolation
    for i in range(T):
        h_new = new_planned[i, 2]
        h_old = shifted_prev[i, 2]
        diff = np.arctan2(np.sin(h_new - h_old), np.cos(h_new - h_old))
        blended[i, 2] = h_old + alpha * diff

    # Blend speed linearly
    if new_planned.shape[1] > 3 and shifted_prev.shape[1] > 3:
        blended[:T, 3] = alpha * new_planned[:T, 3] + (1 - alpha) * shifted_prev[:T, 3]

    return blended


def generate_receding_movie(
    model: RuleAwareGenerator,
    scenario: Dict,
    scenario_idx: int,
    output_path: str,
    fps: int = 10,
    view_range: float = 80,
    device: torch.device = None,
    ema_alpha: float = 0.4,
    n_anchors: int = 5,
):
    """Generate a 9-second receding horizon movie with rule panel.

    Uses EMA (Exponential Moving Average) to smooth trajectories across
    consecutive frames, reducing jitter from independent re-predictions.
    Set ema_alpha=1.0 to disable smoothing (raw predictions).
    """
    # Print smoothing info
    if ema_alpha < 1.0:
        print(f"    EMA smoothing: alpha={ema_alpha:.2f}")

    ego_traj = scenario["ego_traj"]
    agent_trajs = scenario["agent_trajs"]
    lanes = scenario["lanes"]
    road_edges = scenario["road_edges"]
    crosswalks = scenario["crosswalks"]

    # Compute scene bounds (full scenario view)
    all_points = []
    for t in range(TOTAL_TIMESTEPS):
        if ego_traj[t, 4] > 0:
            all_points.append(ego_traj[t, :2])
    all_points = np.array(all_points)

    x_center = all_points[:, 0].mean()
    y_center = all_points[:, 1].mean()

    # Create figure with two panels: scene (left) and rules (right)
    fig = plt.figure(figsize=(16, 10), dpi=100)
    gs = GridSpec(1, 2, width_ratios=[2.5, 1], wspace=0.05)
    ax_scene = fig.add_subplot(gs[0])
    ax_rules = fig.add_subplot(gs[1])

    # EMA state for trajectory smoothing (use mutable container for closure access)
    ema_state = {
        "planned": None,  # Previous smoothed best trajectory (absolute coords)
        "others": {},  # Previous smoothed other-mode trajectories
    }

    # Store predictions
    prediction_cache = {}

    def animate(frame):
        ax_scene.clear()

        current_t = frame  # Current timestep (0-90)

        # Draw road edges
        for edge in road_edges:
            ax_scene.plot(
                edge[:, 0],
                edge[:, 1],
                color=COLORS["road_edge"],
                linewidth=2,
                alpha=0.8,
                zorder=1,
            )

        # Draw lanes
        for lane in lanes:
            ax_scene.plot(
                lane[:, 0],
                lane[:, 1],
                color=COLORS["lane"],
                linewidth=1,
                alpha=0.5,
                zorder=2,
            )

        # Draw crosswalks
        for cw in crosswalks:
            if len(cw) >= 3:
                polygon = plt.Polygon(
                    cw, facecolor=COLORS["crosswalk"], alpha=0.3, zorder=1
                )
                ax_scene.add_patch(polygon)

        # Draw other agents at current time
        for a in range(scenario["agent_count"]):
            if agent_trajs[a, current_t, 4] > 0:
                x, y = agent_trajs[a, current_t, 0], agent_trajs[a, current_t, 1]
                h = agent_trajs[a, current_t, 2]
                agent_type = int(agent_trajs[a, current_t, 5])

                if agent_type == 2:  # Pedestrian
                    ax_scene.scatter(
                        x, y, c=COLORS["pedestrian"], s=80, zorder=5, marker="o"
                    )
                elif agent_type == 3:  # Cyclist
                    ax_scene.scatter(
                        x, y, c=COLORS["cyclist"], s=100, zorder=5, marker="^"
                    )
                else:  # Vehicle
                    draw_vehicle(
                        ax_scene,
                        x,
                        y,
                        h,
                        length=4.0,
                        width=1.8,
                        color=COLORS["agents"],
                        alpha=0.7,
                        zorder=5,
                    )

        # Draw ego history trail (last 10 frames)
        hist_start = max(0, current_t - 10)
        hist_points = []
        for t in range(hist_start, current_t + 1):
            if ego_traj[t, 4] > 0:
                hist_points.append(ego_traj[t, :2])
        if hist_points:
            hist_points = np.array(hist_points)
            ax_scene.plot(
                hist_points[:, 0],
                hist_points[:, 1],
                color=COLORS["ego"],
                linewidth=2,
                alpha=0.5,
                linestyle="--",
                zorder=8,
            )

        # Draw ground truth future (faded)
        future_end = min(current_t + FUTURE_LENGTH, TOTAL_TIMESTEPS)
        gt_future = []
        for t in range(current_t, future_end):
            if ego_traj[t, 4] > 0:
                gt_future.append(ego_traj[t, :2])
        if gt_future:
            gt_future = np.array(gt_future)
            ax_scene.plot(
                gt_future[:, 0],
                gt_future[:, 1],
                color=COLORS["ego_future_gt"],
                linewidth=2,
                alpha=0.4,
                linestyle=":",
                zorder=7,
                label="Ground Truth Future",
            )

        # Initialize rule values
        applicability = np.zeros(NUM_RULES)
        violations = np.zeros(NUM_RULES)

        # Run model prediction from current timestep (if we have enough history)
        if current_t >= 10 and current_t < TOTAL_TIMESTEPS - 1:
            try:
                inputs = prepare_model_input(scenario, current_t, device)

                with torch.no_grad():
                    outputs = model(
                        ego_history=inputs["ego_history"],
                        agent_states=inputs["agent_states"],
                        lane_centers=inputs["lane_centers"],
                    )

                # Get best trajectory
                trajectories = outputs["trajectories"][0].cpu().numpy()  # [6, 50, 4]
                confidence = outputs["confidence"][0].cpu().numpy()
                confidence = np.exp(confidence) / np.exp(confidence).sum()
                best_mode = confidence.argmax()

                # Get applicability probabilities
                if "applicability_prob" in outputs:
                    app_probs = (
                        torch.sigmoid(outputs["applicability_prob"][0]).cpu().numpy()
                    )
                    applicability = app_probs
                elif "applicability" in outputs:
                    # Logits - apply sigmoid
                    app_logits = outputs["applicability"][0].cpu().numpy()
                    applicability = 1.0 / (1.0 + np.exp(-app_logits))

                # Compute violations using rule proxies
                traj_tensor = outputs["trajectories"]  # [1, 6, 50, 4]
                best_traj = traj_tensor[
                    :, best_mode : best_mode + 1, :, :
                ]  # [1, 1, 50, 4]

                # Prepare scene features for proxy evaluation
                scene_features = {
                    "ego_history": inputs["ego_history"],
                    "agent_states": inputs["agent_states"],
                    "lane_centers": inputs["lane_centers"],
                }

                # Compute violations via proxies
                try:
                    violation_costs = model.rule_proxies(
                        best_traj, scene_features
                    )  # [1, 1, NUM_RULES]
                    violations = violation_costs[0, 0].cpu().numpy()  # [NUM_RULES]
                    violations = np.clip(violations, 0, 1)
                except Exception as proxy_err:
                    # Fallback: estimate violations from trajectory kinematics
                    violations = compute_kinematic_violations(trajectories[best_mode])

                # Convert to meters and apply trajectory smoothing
                planned = trajectories[best_mode].copy()
                planned[:, :2] *= TRAJECTORY_SCALE

                # Smooth the trajectory in ego-centric frame BEFORE
                # transforming to scene frame.  This removes the high-freq
                # lateral oscillation produced by the parallel Transformer
                # decoder while preserving the overall path shape.
                if n_anchors > 0:
                    planned = smooth_trajectory(planned, n_anchors=n_anchors)

                # The prediction is relative to current position
                # Add current ego position to get absolute coordinates
                if ego_traj[current_t, 4] > 0:
                    ego_x, ego_y = ego_traj[current_t, 0], ego_traj[current_t, 1]
                    ego_h = ego_traj[current_t, 2]

                    # Rotate and translate
                    cos_h, sin_h = np.cos(ego_h), np.sin(ego_h)
                    for i in range(len(planned)):
                        rx, ry = planned[i, 0], planned[i, 1]
                        planned[i, 0] = ego_x + rx * cos_h - ry * sin_h
                        planned[i, 1] = ego_y + rx * sin_h + ry * cos_h
                        planned[i, 2] += ego_h

                    # --- EMA smoothing across frames ---
                    planned = _ema_blend_trajectory(
                        planned, ema_state["planned"], ema_alpha
                    )
                    ema_state["planned"] = planned.copy()  # Store for next frame

                    # Fit spline for smooth trajectory visualization
                    # Waypoints are timestamped at 10Hz (0.1s intervals)
                    smooth_xy, smooth_times = fit_trajectory_spline(
                        planned,
                        dt=DT,
                        upsample_factor=5,  # 5x upsampling for smooth curve
                        smoothing=0.1,  # Slight smoothing to reduce jitter
                    )

                    # Draw planned trajectory (5 seconds ahead) - smoothed
                    ax_scene.plot(
                        smooth_xy[:, 0],
                        smooth_xy[:, 1],
                        color=COLORS["planned_traj"],
                        linewidth=3,
                        alpha=0.9,
                        zorder=9,
                        label=f"Planned (5s ahead)",
                    )

                    # Also draw waypoint markers (show the actual predicted points)
                    ax_scene.scatter(
                        planned[::5, 0],
                        planned[::5, 1],
                        c=COLORS["planned_traj"],
                        s=30,
                        marker="o",
                        zorder=9,
                        alpha=0.6,
                        edgecolors="white",
                        linewidths=0.5,
                    )

                    # Mark endpoint
                    ax_scene.scatter(
                        planned[-1, 0],
                        planned[-1, 1],
                        c=COLORS["planned_traj"],
                        s=150,
                        marker="*",
                        zorder=10,
                        edgecolors="white",
                        linewidths=1,
                    )

                    # Draw other modes (faded) - also with trajectory smoothing + EMA
                    for m in range(trajectories.shape[0]):
                        if m != best_mode:
                            other = trajectories[m].copy()
                            other[:, :2] *= TRAJECTORY_SCALE
                            if n_anchors > 0:
                                other = smooth_trajectory(other, n_anchors=n_anchors)
                            for i in range(len(other)):
                                rx, ry = other[i, 0], other[i, 1]
                                other[i, 0] = ego_x + rx * cos_h - ry * sin_h
                                other[i, 1] = ego_y + rx * sin_h + ry * cos_h

                            # EMA smooth other modes too
                            prev_other = ema_state["others"].get(m, None)
                            other = _ema_blend_trajectory(other, prev_other, ema_alpha)
                            ema_state["others"][m] = other.copy()

                            # Smooth other modes too
                            other_smooth, _ = fit_trajectory_spline(
                                other, dt=DT, upsample_factor=3, smoothing=0.1
                            )
                            ax_scene.plot(
                                other_smooth[:, 0],
                                other_smooth[:, 1],
                                color=COLORS["other_modes"],
                                linewidth=1,
                                alpha=0.3,
                                zorder=8,
                            )

            except Exception as e:
                pass  # Skip prediction on error

        # Draw ego vehicle at current position
        if ego_traj[current_t, 4] > 0:
            ego_x, ego_y = ego_traj[current_t, 0], ego_traj[current_t, 1]
            ego_h = ego_traj[current_t, 2]
            draw_vehicle(
                ax_scene,
                ego_x,
                ego_y,
                ego_h,
                length=4.8,
                width=2.1,
                color=COLORS["ego"],
                alpha=1.0,
                zorder=15,
                label="Ego Vehicle",
            )

        # Set axis properties - full scene view
        ax_scene.set_xlim(x_center - view_range / 2, x_center + view_range / 2)
        ax_scene.set_ylim(y_center - view_range / 2, y_center + view_range / 2)
        ax_scene.set_aspect("equal")
        ax_scene.grid(True, alpha=0.2)

        # Legend
        ax_scene.legend(loc="upper left", fontsize=9)

        # Time display
        time_sec = (current_t - 10) / 10.0  # t=10 is current (0.0s)
        time_str = f"t = {time_sec:+.1f}s"
        if current_t < 10:
            phase = "History"
        else:
            phase = "Live Planning"

        ax_scene.set_xlabel("X (meters)", fontsize=11)
        ax_scene.set_ylabel("Y (meters)", fontsize=11)
        ax_scene.set_title(
            f"RECTOR V2 Receding Horizon - Scenario {scenario_idx}\n"
            f"{phase} | {time_str} | Frame {current_t+1}/{TOTAL_TIMESTEPS}",
            fontsize=12,
        )

        # Draw rule panel
        draw_rule_panel(ax_rules, applicability, violations, current_t)

        return []

    # Create animation (91 frames = 9.1 seconds)
    print(f"    Generating {TOTAL_TIMESTEPS} frames...")
    anim = animation.FuncAnimation(
        fig, animate, frames=TOTAL_TIMESTEPS, interval=1000 // fps, blit=False
    )

    # Save
    writer = animation.FFMpegWriter(fps=fps, bitrate=4000)
    anim.save(output_path, writer=writer)
    print(f"    Saved: {output_path}")

    plt.close(fig)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set seed (None = different each run)
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("RECTOR V2 Receding Horizon Movie Generator (with Rule Display)")
    print("=" * 70)
    print(f"\n  Duration: 9.1 seconds ({TOTAL_TIMESTEPS} frames)")
    print(f"  Planning: At each step, predict next 5 seconds")
    print(f"  View: Full scene ({args.view_range}m range)")
    print(f"  Rule display: {NUM_RULES} rules across 4 tiers")

    # Load model
    print("\n[1/3] Loading Model...")
    model = RuleAwareGenerator(
        embed_dim=256,
        decoder_hidden_dim=256,
        decoder_num_layers=4,
        latent_dim=64,
        num_modes=6,
        use_m2i_encoder=True,
        m2i_checkpoint=args.m2i_checkpoint,
        freeze_m2i=True,
        trajectory_length=50,
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"  Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")

    # Load scenarios
    print("\n[2/3] Loading Scenarios...")
    data_files = sorted(glob.glob(os.path.join(args.data_dir, "*")))
    sample_files = random.sample(data_files, min(20, len(data_files)))

    loader = ScenarioLoader(sample_files)
    print(f"  Loaded {len(loader)} scenarios from {len(sample_files)} files")

    if len(loader) == 0:
        print("ERROR: No valid scenarios found!")
        return

    # Filter for moving scenarios (skip parked/static ego)
    min_speed = args.min_ego_speed
    moving_indices = []
    for idx in range(len(loader)):
        s = loader[idx]
        if s["ego_avg_speed"] >= min_speed:
            moving_indices.append(idx)

    print(
        f"  Moving scenarios (avg speed >= {min_speed:.1f} m/s): "
        f"{len(moving_indices)}/{len(loader)}"
    )

    if not moving_indices:
        print(
            "WARNING: No scenarios with sufficient ego motion. "
            "Falling back to all scenarios."
        )
        moving_indices = list(range(len(loader)))

    # Randomly sample from moving scenarios
    n_gen = min(args.num_scenarios, len(moving_indices))
    scenarios_to_use = random.sample(moving_indices, n_gen)

    print(
        f"\n[3/3] Generating {n_gen} Movies (randomly sampled from {len(moving_indices)} moving scenarios)..."
    )

    for i, scenario_idx in enumerate(scenarios_to_use):
        scenario = loader[scenario_idx]
        output_path = str(
            output_dir / f'receding_{i:03d}_{scenario["scenario_id"][:8]}.mp4'
        )

        print(
            f"\n  Scenario {i+1}/{len(scenarios_to_use)}: {scenario['scenario_id'][:16]}... "
            f"(ego dist={scenario['ego_total_dist']:.1f}m, avg={scenario['ego_avg_speed']:.1f}m/s)"
        )

        generate_receding_movie(
            model=model,
            scenario=scenario,
            scenario_idx=i,
            output_path=output_path,
            fps=args.fps,
            view_range=args.view_range,
            device=device,
            ema_alpha=1.0 if args.no_ema else args.ema_alpha,
            n_anchors=args.n_anchors,
        )

    print("\n" + "=" * 70)
    print("MOVIE GENERATION COMPLETE")
    print("=" * 70)
    print(f"\n  Movies generated: {len(scenarios_to_use)}")
    print(f"  Duration: 9.1 seconds each")
    print(f"  Output: {output_dir}")
    print(f"  Features:")
    print(f"    - Receding horizon planning (re-predict each frame)")
    print(
        f"    - EMA trajectory smoothing (alpha={1.0 if args.no_ema else args.ema_alpha:.2f})"
    )
    print(f"    - Rule violation panel with 4 tiers")
    print(f"    - Color-coded violation severity")
    print("=" * 70)


if __name__ == "__main__":
    main()
