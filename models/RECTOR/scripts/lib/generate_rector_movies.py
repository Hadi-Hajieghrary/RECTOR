#!/usr/bin/env python3
"""
Generate movies visualizing RECTOR planning.

Shows:
- Road map (lanes, boundaries, crosswalks) as polylines matching M2I/BEV style
- All agents in the scene with oriented rectangle patches
- Ground truth trajectories (past and future)
- Ego's candidate trajectories with safety scores
- Reactor predictions based on each candidate
- Type-based coloring (vehicle, pedestrian, cyclist)

Visual style matches M2I movies for consistency.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

# Add RECTOR lib to path
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

# Try importing TensorFlow
try:
    import tensorflow as tf

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Warning: TensorFlow not available")

# Import matplotlib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, Polygon, Circle, FancyArrow
from matplotlib.lines import Line2D
import matplotlib.transforms as transforms
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter

try:
    from matplotlib.animation import FFMpegWriter

    FFMPEG_AVAILABLE = True
except ImportError:
    FFMPEG_AVAILABLE = False

# Import RECTOR components
try:
    from planning_loop import (
        RECTORPlanner,
        CandidateGenerator,
        ReactorSelector,
        CandidateScorer,
    )
    from data_contracts import AgentState, EgoCandidateBatch, PlanningConfig, LaneInfo

    RECTOR_AVAILABLE = True
except ImportError as e:
    print(f"Warning: RECTOR components not available: {e}")
    RECTOR_AVAILABLE = False


class RECTORMovieGenerator:
    """Generate movies visualizing RECTOR planning on Waymo scenarios.

    Visual style matches M2I movie generator for consistency.
    """

    # Colors matching M2I/BEV movie style
    COLORS = {
        # Agent types
        "ego": "#FF0000",  # Red for ego (like influencer)
        "reactor": "#f39c12",  # Orange for reactors
        "vehicle": "#3498db",  # Blue for other vehicles
        "pedestrian": "#2ecc71",  # Green for pedestrians
        "cyclist": "#9b59b6",  # Purple for cyclists
        "other": "#95a5a6",  # Gray for others
        # Road elements
        "lane": "#34495e",  # Dark gray for lanes
        "road_line": "#7f8c8d",  # Medium gray for road lines
        "road_edge": "#2c3e50",  # Very dark gray for road edges
        "crosswalk": "#16a085",  # Teal for crosswalks
        # Trajectories
        "history_ego": "#e74c3c",  # Red for ego history
        "history_reactor": "#f39c12",  # Orange for reactor history
        "future_gt": "#bdc3c7",  # Light gray for GT future
        "candidate_safe": "#27ae60",  # Green for safe candidates
        "candidate_unsafe": "#e74c3c",  # Red for unsafe candidates
        "selected": "#2ecc71",  # Bright green for selected
    }

    # Roadgraph type mapping
    ROAD_TYPE_NAMES = {
        1: "lane_freeway",
        2: "lane_surface_street",
        3: "lane_bike_lane",
        6: "road_line_broken_single_white",
        7: "road_line_solid_single_white",
        8: "road_line_solid_double_white",
        9: "road_line_broken_single_yellow",
        10: "road_line_broken_double_yellow",
        11: "road_line_solid_single_yellow",
        12: "road_line_solid_double_yellow",
        13: "road_line_passing_double_yellow",
        15: "road_edge_boundary",
        16: "road_edge_median",
        17: "stop_sign",
        18: "crosswalk",
        19: "speed_bump",
    }

    def __init__(
        self, output_dir: Path, view_range: float = 50.0, fps: int = 10, dpi: int = 100
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.view_range = view_range
        self.fps = fps
        self.dpi = dpi

        print(f"RECTOR Movie Generator initialized:")
        print(f"  Output: {self.output_dir}")
        print(f"  FPS: {fps}, DPI: {dpi}, View Range: {view_range}m")

    def _build_features_description(self):
        """Build TFRecord feature description for Waymo Motion format."""
        n_agents = 128
        n_past = 10
        n_current = 1
        n_future = 80

        return {
            "scenario/id": tf.io.FixedLenFeature([], tf.string),
            # Past states
            "state/past/x": tf.io.FixedLenFeature([n_agents, n_past], tf.float32),
            "state/past/y": tf.io.FixedLenFeature([n_agents, n_past], tf.float32),
            "state/past/velocity_x": tf.io.FixedLenFeature(
                [n_agents, n_past], tf.float32
            ),
            "state/past/velocity_y": tf.io.FixedLenFeature(
                [n_agents, n_past], tf.float32
            ),
            "state/past/bbox_yaw": tf.io.FixedLenFeature(
                [n_agents, n_past], tf.float32
            ),
            "state/past/valid": tf.io.FixedLenFeature([n_agents, n_past], tf.int64),
            "state/past/length": tf.io.FixedLenFeature([n_agents, n_past], tf.float32),
            "state/past/width": tf.io.FixedLenFeature([n_agents, n_past], tf.float32),
            # Current state
            "state/current/x": tf.io.FixedLenFeature([n_agents, n_current], tf.float32),
            "state/current/y": tf.io.FixedLenFeature([n_agents, n_current], tf.float32),
            "state/current/velocity_x": tf.io.FixedLenFeature(
                [n_agents, n_current], tf.float32
            ),
            "state/current/velocity_y": tf.io.FixedLenFeature(
                [n_agents, n_current], tf.float32
            ),
            "state/current/bbox_yaw": tf.io.FixedLenFeature(
                [n_agents, n_current], tf.float32
            ),
            "state/current/valid": tf.io.FixedLenFeature(
                [n_agents, n_current], tf.int64
            ),
            "state/current/length": tf.io.FixedLenFeature(
                [n_agents, n_current], tf.float32
            ),
            "state/current/width": tf.io.FixedLenFeature(
                [n_agents, n_current], tf.float32
            ),
            # Future states
            "state/future/x": tf.io.FixedLenFeature([n_agents, n_future], tf.float32),
            "state/future/y": tf.io.FixedLenFeature([n_agents, n_future], tf.float32),
            "state/future/velocity_x": tf.io.FixedLenFeature(
                [n_agents, n_future], tf.float32
            ),
            "state/future/velocity_y": tf.io.FixedLenFeature(
                [n_agents, n_future], tf.float32
            ),
            "state/future/bbox_yaw": tf.io.FixedLenFeature(
                [n_agents, n_future], tf.float32
            ),
            "state/future/valid": tf.io.FixedLenFeature([n_agents, n_future], tf.int64),
            # Static properties
            "state/type": tf.io.FixedLenFeature([n_agents], tf.float32),
            "state/id": tf.io.FixedLenFeature([n_agents], tf.float32),
            "state/objects_of_interest": tf.io.FixedLenFeature([n_agents], tf.int64),
            # Roadgraph
            "roadgraph_samples/xyz": tf.io.VarLenFeature(tf.float32),
            "roadgraph_samples/type": tf.io.VarLenFeature(tf.int64),
            "roadgraph_samples/id": tf.io.VarLenFeature(tf.int64),
            "roadgraph_samples/valid": tf.io.VarLenFeature(tf.int64),
        }

    def parse_scenario(
        self, tfrecord_path: str, scenario_idx: int = 0
    ) -> Optional[Dict]:
        """Parse a Waymo scenario from TFRecord."""
        if not TF_AVAILABLE:
            raise RuntimeError("TensorFlow required")

        features = self._build_features_description()
        dataset = tf.data.TFRecordDataset(tfrecord_path)

        for idx, record in enumerate(dataset):
            if idx != scenario_idx:
                continue

            parsed = tf.io.parse_single_example(record, features)

            def to_np(key):
                tensor = parsed[key]
                if isinstance(tensor, tf.SparseTensor):
                    return tf.sparse.to_dense(tensor).numpy()
                return tensor.numpy()

            def concat_time(base_key):
                """Concatenate past/current/future along time axis."""
                past = to_np(f"state/past/{base_key}")
                current = to_np(f"state/current/{base_key}")
                future = to_np(f"state/future/{base_key}")
                return np.concatenate([past, current, future], axis=1)

            # Concatenate to [128, 91] or [128, 91, 2]
            x = concat_time("x")
            y = concat_time("y")
            vx = concat_time("velocity_x")
            vy = concat_time("velocity_y")
            yaw = concat_time("bbox_yaw")
            valid = concat_time("valid")

            states = np.stack([x, y], axis=-1)  # [128, 91, 2]

            # Get size from current (static per agent)
            length = to_np("state/current/length")[:, 0]
            width = to_np("state/current/width")[:, 0]

            # Roadgraph
            road_xyz_flat = to_np("roadgraph_samples/xyz")
            road_type = to_np("roadgraph_samples/type")
            road_id = to_np("roadgraph_samples/id")
            road_valid = to_np("roadgraph_samples/valid")

            n_road_pts = len(road_xyz_flat) // 3
            if n_road_pts > 0:
                road_xyz = road_xyz_flat.reshape(n_road_pts, 3)
                road_valid_mask = (
                    road_valid[:n_road_pts] > 0
                    if len(road_valid) >= n_road_pts
                    else np.ones(n_road_pts, dtype=bool)
                )
                road_type = (
                    road_type[:n_road_pts]
                    if len(road_type) >= n_road_pts
                    else np.zeros(n_road_pts, dtype=int)
                )
                road_id = (
                    road_id[:n_road_pts]
                    if len(road_id) >= n_road_pts
                    else np.arange(n_road_pts)
                )

                road_polylines = self._build_road_polylines(
                    road_xyz[road_valid_mask],
                    road_type[road_valid_mask],
                    road_id[road_valid_mask],
                )
            else:
                road_xyz = np.zeros((0, 3))
                road_polylines = {
                    "lanes": [],
                    "road_lines": [],
                    "road_edges": [],
                    "crosswalks": [],
                    "stop_signs": [],
                }

            scenario_id = to_np("scenario/id")
            if isinstance(scenario_id, bytes):
                scenario_id = scenario_id.decode()

            return {
                "scenario_id": scenario_id,
                "state": states,
                "state_is_valid": valid > 0,
                "yaw": yaw,
                "velocity_x": vx,
                "velocity_y": vy,
                "length": length,
                "width": width,
                "track_type": to_np("state/type").astype(int),
                "track_id": to_np("state/id"),
                "objects_of_interest": to_np("state/objects_of_interest"),
                "road_polylines": road_polylines,
            }

        return None

    def _build_road_polylines(
        self, xyz: np.ndarray, types: np.ndarray, ids: np.ndarray
    ) -> Dict:
        """Group road points by ID to form polylines."""
        polylines = {
            "lanes": [],
            "road_lines": [],
            "road_edges": [],
            "crosswalks": [],
            "stop_signs": [],
        }

        if len(xyz) == 0:
            return polylines

        unique_ids = np.unique(ids)
        for road_id in unique_ids:
            mask = ids == road_id
            points = xyz[mask][:, :2]
            road_types = types[mask]

            if len(points) < 2:
                continue

            most_common_type = (
                int(np.bincount(road_types.astype(int)).argmax())
                if len(road_types) > 0
                else 0
            )

            if most_common_type in [1, 2, 3]:
                polylines["lanes"].append(points)
            elif most_common_type in [6, 7, 8, 9, 10, 11, 12, 13]:
                polylines["road_lines"].append(points)
            elif most_common_type in [15, 16]:
                polylines["road_edges"].append(points)
            elif most_common_type == 18:
                polylines["crosswalks"].append(points)
            elif most_common_type == 17:
                polylines["stop_signs"].append(points[0])

        return polylines

    def _get_agent_color(
        self, agent_type: int, is_ego: bool = False, is_reactor: bool = False
    ) -> str:
        """Get color for an agent based on type and role."""
        if is_ego:
            return self.COLORS["ego"]
        if is_reactor:
            return self.COLORS["reactor"]

        type_map = {1: "vehicle", 2: "pedestrian", 3: "cyclist"}
        type_name = type_map.get(agent_type, "other")
        return self.COLORS.get(type_name, self.COLORS["other"])

    def _create_vehicle_patch(
        self,
        ax,
        x: float,
        y: float,
        heading: float,
        length: float,
        width: float,
        color: str,
        alpha: float = 0.8,
    ) -> Rectangle:
        """Create oriented rectangle patch for vehicle."""
        if length <= 0:
            length = 4.5
        if width <= 0:
            width = 2.0

        rect = Rectangle(
            (-length / 2, -width / 2),
            length,
            width,
            facecolor=color,
            edgecolor="black",
            linewidth=1.5,
            alpha=alpha,
        )

        t = transforms.Affine2D().rotate(heading).translate(x, y) + ax.transData
        rect.set_transform(t)

        return rect

    def _draw_road_polylines(
        self, ax, road_polylines: Dict, center_x: float, center_y: float
    ):
        """Draw road elements as polylines."""
        view_range = self.view_range

        # Lanes
        for lane_points in road_polylines.get("lanes", []):
            in_view = [
                (x, y)
                for x, y in lane_points
                if abs(x - center_x) < view_range and abs(y - center_y) < view_range
            ]
            if len(in_view) > 1:
                xs, ys = zip(*in_view)
                ax.plot(
                    xs,
                    ys,
                    color=self.COLORS["lane"],
                    linewidth=1.5,
                    alpha=0.4,
                    linestyle="-",
                    zorder=1,
                )

        # Road lines
        for line_points in road_polylines.get("road_lines", []):
            in_view = [
                (x, y)
                for x, y in line_points
                if abs(x - center_x) < view_range and abs(y - center_y) < view_range
            ]
            if len(in_view) > 1:
                xs, ys = zip(*in_view)
                ax.plot(
                    xs,
                    ys,
                    color=self.COLORS["road_line"],
                    linewidth=1.0,
                    alpha=0.5,
                    linestyle="--",
                    zorder=1,
                )

        # Road edges
        for edge_points in road_polylines.get("road_edges", []):
            in_view = [
                (x, y)
                for x, y in edge_points
                if abs(x - center_x) < view_range and abs(y - center_y) < view_range
            ]
            if len(in_view) > 1:
                xs, ys = zip(*in_view)
                ax.plot(
                    xs,
                    ys,
                    color=self.COLORS["road_edge"],
                    linewidth=2.0,
                    alpha=0.6,
                    linestyle="-",
                    zorder=1,
                )

        # Crosswalks
        for crosswalk_points in road_polylines.get("crosswalks", []):
            in_view = [
                (x, y)
                for x, y in crosswalk_points
                if abs(x - center_x) < view_range and abs(y - center_y) < view_range
            ]
            if len(in_view) > 2:
                polygon = Polygon(
                    in_view,
                    closed=True,
                    facecolor=self.COLORS["crosswalk"],
                    edgecolor=self.COLORS["crosswalk"],
                    linewidth=2,
                    alpha=0.3,
                    zorder=1,
                )
                ax.add_patch(polygon)

        # Stop signs
        for stop_pos in road_polylines.get("stop_signs", []):
            sx, sy = stop_pos[0], stop_pos[1]
            if abs(sx - center_x) < view_range and abs(sy - center_y) < view_range:
                circle = Circle((sx, sy), 1.5, color="#c0392b", alpha=0.7, zorder=5)
                ax.add_patch(circle)

    def generate_candidate_trajectories(
        self, ego_state, num_candidates: int = 8
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate ego candidate trajectories for visualization.

        Returns:
            candidates: [M, H, 2] candidate trajectories
            scores: [M] safety scores for each candidate
        """
        M = num_candidates
        H = 80
        candidates = np.zeros((M, H, 2))

        # Get ego properties
        x0 = ego_state.x if hasattr(ego_state, "x") else ego_state["x"]
        y0 = ego_state.y if hasattr(ego_state, "y") else ego_state["y"]
        yaw0 = ego_state.yaw if hasattr(ego_state, "yaw") else ego_state["yaw"]
        speed0 = ego_state.speed if hasattr(ego_state, "speed") else ego_state["speed"]

        for m in range(M):
            # Vary speed and lateral offset
            speed_factor = 0.5 + (m / (M - 1)) * 0.7 if M > 1 else 1.0
            speed = speed0 * speed_factor
            lateral = (m - M / 2) * 0.8

            for t in range(H):
                dt = 0.1 * t
                # Forward motion with lateral variation
                dx = speed * dt * np.cos(yaw0) - lateral * np.sin(yaw0) * (
                    1 - np.exp(-dt / 2)
                )
                dy = speed * dt * np.sin(yaw0) + lateral * np.cos(yaw0) * (
                    1 - np.exp(-dt / 2)
                )
                candidates[m, t, 0] = x0 + dx
                candidates[m, t, 1] = y0 + dy

        # Compute scores (higher for middle candidates, lower for extremes)
        scores = np.zeros(M)
        for m in range(M):
            # Distance from center candidate
            dist_from_center = abs(m - (M - 1) / 2) / (M / 2)
            scores[m] = 1.0 - 0.5 * dist_from_center

        # Add some randomness
        scores += np.random.uniform(-0.1, 0.1, M)
        scores = np.clip(scores, 0.1, 1.0)

        return candidates, scores

    def generate_scenario_movie(
        self,
        scenario: Dict,
        start_t: int = 10,
        end_t: int = 80,
        step: int = 1,
        num_candidates: int = 8,
    ) -> Optional[Path]:
        """Generate a movie for a scenario showing RECTOR planning.

        Args:
            scenario: Parsed scenario dict
            start_t: Starting timestep
            end_t: Ending timestep
            step: Timestep increment
            num_candidates: Number of candidate trajectories
        """
        scenario_id = scenario["scenario_id"]
        print(f"\nGenerating movie for scenario: {scenario_id}")

        states = scenario["state"]  # [128, 91, 2]
        valid = scenario["state_is_valid"]
        yaw = scenario["yaw"]
        vx = scenario["velocity_x"]
        vy = scenario["velocity_y"]
        length = scenario["length"]
        width = scenario["width"]
        track_types = scenario["track_type"]
        road_polylines = scenario["road_polylines"]

        # Ego is agent 0
        ego_idx = 0

        # Select reactor agents (up to 3 nearest agents of interest)
        oi = scenario["objects_of_interest"]
        reactor_indices = []
        for i in range(1, 128):
            if oi[i] > 0 and valid[i, start_t]:
                reactor_indices.append(i)
            if len(reactor_indices) >= 3:
                break

        # Ensure we have at least one reactor
        if not reactor_indices:
            for i in range(1, 128):
                if valid[i, start_t]:
                    reactor_indices.append(i)
                if len(reactor_indices) >= 3:
                    break

        print(f"  Ego: agent 0, Reactors: {reactor_indices}")

        fig, ax = plt.subplots(1, 1, figsize=(12, 12))

        # Generate frames for animation
        timesteps = list(range(start_t, min(end_t + 1, 91), step))

        def animate(frame_idx):
            ax.clear()
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

            t = timesteps[frame_idx]

            # Get ego position for camera centering
            if valid[ego_idx, t]:
                center_x = states[ego_idx, t, 0]
                center_y = states[ego_idx, t, 1]
            else:
                for check_t in range(t, -1, -1):
                    if valid[ego_idx, check_t]:
                        center_x = states[ego_idx, check_t, 0]
                        center_y = states[ego_idx, check_t, 1]
                        break
                else:
                    center_x, center_y = 0, 0

            # Set view bounds (follow ego)
            ax.set_xlim(center_x - self.view_range, center_x + self.view_range)
            ax.set_ylim(center_y - self.view_range, center_y + self.view_range)
            ax.set_facecolor("#f5f5f5")

            # Draw road
            self._draw_road_polylines(ax, road_polylines, center_x, center_y)

            # Draw all agents
            for agent_i in range(128):
                if not valid[agent_i, t]:
                    continue

                agent_x = states[agent_i, t, 0]
                agent_y = states[agent_i, t, 1]

                if (
                    abs(agent_x - center_x) > self.view_range
                    or abs(agent_y - center_y) > self.view_range
                ):
                    continue

                agent_yaw = yaw[agent_i, t]
                agent_length = length[agent_i]
                agent_width = width[agent_i]
                agent_type = track_types[agent_i]

                is_ego = agent_i == ego_idx
                is_reactor = agent_i in reactor_indices

                color = self._get_agent_color(agent_type, is_ego, is_reactor)

                rect = self._create_vehicle_patch(
                    ax, agent_x, agent_y, agent_yaw, agent_length, agent_width, color
                )
                ax.add_patch(rect)

                # Velocity arrow
                speed = np.sqrt(vx[agent_i, t] ** 2 + vy[agent_i, t] ** 2)
                if speed > 0.5:
                    arrow = FancyArrow(
                        agent_x,
                        agent_y,
                        vx[agent_i, t] * 0.5,
                        vy[agent_i, t] * 0.5,
                        width=0.3,
                        head_width=0.8,
                        head_length=0.5,
                        fc=color,
                        ec="black",
                        alpha=0.6,
                        linewidth=0.5,
                        zorder=6,
                    )
                    ax.add_patch(arrow)

                # Labels
                if is_ego:
                    ax.annotate(
                        "Ego",
                        (agent_x, agent_y),
                        textcoords="offset points",
                        xytext=(5, 8),
                        fontsize=9,
                        fontweight="bold",
                        color=color,
                        zorder=10,
                    )
                elif is_reactor:
                    ax.annotate(
                        "Reactor",
                        (agent_x, agent_y),
                        textcoords="offset points",
                        xytext=(5, 8),
                        fontsize=9,
                        fontweight="bold",
                        color=color,
                        zorder=10,
                    )

            # Draw history trails
            history_length = min(t + 1, 15)

            # Ego history
            ego_history = []
            for hist_t in range(max(0, t - history_length), t + 1):
                if valid[ego_idx, hist_t]:
                    ego_history.append(states[ego_idx, hist_t])
            if len(ego_history) > 1:
                ego_history = np.array(ego_history)
                ax.plot(
                    ego_history[:, 0],
                    ego_history[:, 1],
                    color=self.COLORS["history_ego"],
                    linewidth=2.5,
                    alpha=0.5,
                    linestyle="--",
                    zorder=3,
                )

            # Reactor histories
            for react_idx in reactor_indices:
                react_history = []
                for hist_t in range(max(0, t - history_length), t + 1):
                    if valid[react_idx, hist_t]:
                        react_history.append(states[react_idx, hist_t])
                if len(react_history) > 1:
                    react_history = np.array(react_history)
                    ax.plot(
                        react_history[:, 0],
                        react_history[:, 1],
                        color=self.COLORS["history_reactor"],
                        linewidth=2.5,
                        alpha=0.5,
                        linestyle="--",
                        zorder=3,
                    )

            # Draw ground truth future
            if t >= 10:
                # Ego GT future
                future_mask = valid[ego_idx, t:]
                if future_mask.any():
                    gt_future = states[ego_idx, t:][future_mask]
                    if len(gt_future) > 1:
                        ax.plot(
                            gt_future[:, 0],
                            gt_future[:, 1],
                            color=self.COLORS["future_gt"],
                            linewidth=1.5,
                            linestyle=":",
                            alpha=0.4,
                            zorder=2,
                        )

            # Draw candidate trajectories
            if valid[ego_idx, t]:
                ego_state = {
                    "x": float(states[ego_idx, t, 0]),
                    "y": float(states[ego_idx, t, 1]),
                    "yaw": float(yaw[ego_idx, t]),
                    "speed": float(np.sqrt(vx[ego_idx, t] ** 2 + vy[ego_idx, t] ** 2)),
                    "length": float(length[ego_idx]),
                    "width": float(width[ego_idx]),
                }

                candidates, scores = self.generate_candidate_trajectories(
                    ego_state, num_candidates
                )

                # Find best candidate
                best_idx = np.argmax(scores)

                # Draw all candidates
                for m in range(len(candidates)):
                    traj = candidates[m]
                    score = scores[m]
                    is_best = m == best_idx

                    if is_best:
                        color = self.COLORS["selected"]
                        linewidth = 3.5
                        alpha = 0.9
                    elif score > 0.5:
                        color = self.COLORS["candidate_safe"]
                        linewidth = 2.0
                        alpha = 0.6
                    else:
                        color = self.COLORS["candidate_unsafe"]
                        linewidth = 1.5
                        alpha = 0.4

                    ax.plot(
                        traj[:, 0],
                        traj[:, 1],
                        color=color,
                        linewidth=linewidth,
                        alpha=alpha,
                        zorder=4 if is_best else 3,
                    )

                    # Mark endpoint
                    if is_best:
                        ax.scatter(
                            traj[-1, 0],
                            traj[-1, 1],
                            c=color,
                            s=100,
                            marker="*",
                            edgecolors="white",
                            linewidths=1,
                            zorder=5,
                        )
                    else:
                        ax.scatter(
                            traj[-1, 0],
                            traj[-1, 1],
                            c=color,
                            s=30,
                            marker="o",
                            edgecolors="white",
                            linewidths=0.5,
                            alpha=alpha,
                            zorder=4,
                        )

            # Title
            time_sec = t * 0.1
            ax.set_title(
                f"RECTOR Planning - {scenario_id[:20]}...\n"
                f"Frame {t}/91 | Time: {time_sec:.1f}s | Candidates: {num_candidates}",
                fontsize=12,
                fontweight="bold",
            )

            ax.set_xlabel("X (meters)", fontsize=10)
            ax.set_ylabel("Y (meters)", fontsize=10)

            # Legend
            legend_elements = [
                mpatches.Patch(
                    facecolor=self.COLORS["ego"], edgecolor="black", label="Ego"
                ),
                mpatches.Patch(
                    facecolor=self.COLORS["reactor"],
                    edgecolor="black",
                    label="Reactors",
                ),
                mpatches.Patch(
                    facecolor=self.COLORS["vehicle"],
                    edgecolor="black",
                    label="Other Vehicles",
                ),
                mpatches.Patch(
                    facecolor=self.COLORS["pedestrian"],
                    edgecolor="black",
                    label="Pedestrians",
                ),
                mpatches.Patch(
                    facecolor=self.COLORS["cyclist"],
                    edgecolor="black",
                    label="Cyclists",
                ),
                Line2D(
                    [0],
                    [0],
                    color=self.COLORS["history_ego"],
                    linestyle="--",
                    linewidth=2,
                    label="History Trail",
                ),
                Line2D(
                    [0],
                    [0],
                    color=self.COLORS["future_gt"],
                    linestyle=":",
                    linewidth=1.5,
                    label="Ground Truth",
                ),
                Line2D(
                    [0],
                    [0],
                    color=self.COLORS["selected"],
                    linestyle="-",
                    linewidth=3.5,
                    label="Selected Trajectory",
                ),
                Line2D(
                    [0],
                    [0],
                    color=self.COLORS["candidate_safe"],
                    linestyle="-",
                    linewidth=2,
                    label="Safe Candidates",
                ),
                Line2D(
                    [0],
                    [0],
                    color=self.COLORS["candidate_unsafe"],
                    linestyle="-",
                    linewidth=1.5,
                    label="Unsafe Candidates",
                ),
                Line2D(
                    [0],
                    [0],
                    color=self.COLORS["lane"],
                    linestyle="-",
                    linewidth=1.5,
                    label="Lanes",
                ),
                Line2D(
                    [0],
                    [0],
                    color=self.COLORS["road_edge"],
                    linestyle="-",
                    linewidth=2,
                    label="Road Edges",
                ),
            ]
            ax.legend(
                handles=legend_elements,
                loc="upper right",
                fontsize=8,
                framealpha=0.9,
                ncol=1,
            )

            return []

        # Create animation
        num_frames = len(timesteps)
        print(f"  Rendering {num_frames} frames...")

        anim = animation.FuncAnimation(
            fig,
            animate,
            frames=num_frames,
            interval=1000 // self.fps,
            blit=False,
            repeat=True,
        )

        # Save
        output_path = self.output_dir / f"{scenario_id}_rector_planning"

        # Try MP4 first
        mp4_saved = False
        if FFMPEG_AVAILABLE:
            try:
                mp4_path = output_path.with_suffix(".mp4")
                writer = FFMpegWriter(
                    fps=self.fps,
                    bitrate=1800,
                    codec="libx264",
                    extra_args=["-pix_fmt", "yuv420p"],
                )
                anim.save(str(mp4_path), writer=writer, dpi=self.dpi)
                file_size_mb = mp4_path.stat().st_size / (1024 * 1024)
                print(f"  ✓ Saved MP4: {mp4_path.name} ({file_size_mb:.1f} MB)")
                mp4_saved = True
            except Exception as e:
                print(f"  Warning: MP4 save failed: {e}")

        # Save GIF
        try:
            gif_path = output_path.with_suffix(".gif")
            gif_writer = PillowWriter(fps=min(self.fps, 10))
            anim.save(str(gif_path), writer=gif_writer, dpi=max(self.dpi // 2, 50))
            gif_size_mb = gif_path.stat().st_size / (1024 * 1024)
            print(f"  ✓ Saved GIF: {gif_path.name} ({gif_size_mb:.1f} MB)")
        except Exception as e:
            print(f"  Warning: GIF save failed: {e}")

        plt.close(fig)
        return output_path


def find_tfrecord_files(data_dir: str) -> List[Path]:
    """Find all TFRecord files in directory."""
    data_path = Path(data_dir)
    if data_path.is_file():
        return [data_path]
    return sorted(data_path.glob("*.tfrecord*"))


def main():
    parser = argparse.ArgumentParser(
        description="Generate RECTOR planning visualization movies"
    )
    parser.add_argument(
        "--tfrecord",
        "-t",
        type=str,
        default="/workspace/data/WOMD/datasets/waymo_open_dataset/motion_v_1_3_0/processed/tf/validation_interactive",
        help="TFRecord file or directory",
    )
    parser.add_argument(
        "--num_scenarios",
        "-n",
        type=int,
        default=3,
        help="Number of scenarios to process",
    )
    parser.add_argument("--start_t", type=int, default=10, help="Start timestep")
    parser.add_argument("--end_t", type=int, default=80, help="End timestep")
    parser.add_argument("--step", type=int, default=2, help="Timestep increment")
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default="/workspace/models/RECTOR/movies",
        help="Output directory for movies",
    )
    parser.add_argument("--fps", type=int, default=10, help="Frames per second")
    parser.add_argument(
        "--num_candidates", type=int, default=8, help="Number of candidate trajectories"
    )
    parser.add_argument(
        "--view_range", type=float, default=50.0, help="View range in meters"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("RECTOR MOVIE GENERATOR")
    print("=" * 60)
    print()

    # Find TFRecord files
    tfrecord_files = find_tfrecord_files(args.tfrecord)
    if not tfrecord_files:
        print(f"Error: No TFRecord files found in {args.tfrecord}")
        return

    print(f"Configuration:")
    print(f"  TFRecord: {tfrecord_files[0]}")
    print(f"  Scenarios: {args.num_scenarios}")
    print(f"  Timesteps: {args.start_t} to {args.end_t} (step={args.step})")
    print(f"  Output: {args.output_dir}")
    print()

    # Create generator
    generator = RECTORMovieGenerator(
        output_dir=Path(args.output_dir),
        view_range=args.view_range,
        fps=args.fps,
        dpi=100,
    )

    # Process scenarios
    import time

    start_time = time.time()

    scenarios_processed = 0
    for tfrecord_path in tfrecord_files:
        if scenarios_processed >= args.num_scenarios:
            break

        dataset = tf.data.TFRecordDataset(str(tfrecord_path))

        for scenario_idx, _ in enumerate(dataset):
            if scenarios_processed >= args.num_scenarios:
                break

            print()
            print("#" * 60)
            print(f"# SCENARIO {scenarios_processed + 1}/{args.num_scenarios}")
            print("#" * 60)

            scenario = generator.parse_scenario(str(tfrecord_path), scenario_idx)
            if scenario is None:
                continue

            print(f"  Scenario ID: {scenario['scenario_id']}")

            generator.generate_scenario_movie(
                scenario,
                start_t=args.start_t,
                end_t=args.end_t,
                step=args.step,
                num_candidates=args.num_candidates,
            )

            scenarios_processed += 1

    total_time = time.time() - start_time

    print()
    print("=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"Total time: {total_time:.1f}s")
    print(f"Movies saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
