#!/usr/bin/env python3
"""
Generate Bird's-Eye View (BEV) Movies from Waymo Scenarios

Creates animated MP4/GIF movies showing a top-down view of the scenario
from the ego vehicle's perspective, simulating a LiDAR/sensor view.

This script works with both:
- Raw Scenario format (raw/scenario/)
- Processed TF format (processed/tf/)

Usage:
    # Single file
    python generate_bev_movie.py --tfrecord <path> --scenario-index 0

    # Process split from scenario format
    python generate_bev_movie.py --format scenario --split validation_interactive --num 5

    # Process split from tf format
    python generate_bev_movie.py --format tf --split training_interactive --num 5

    # Process all splits from both formats
    python generate_bev_movie.py --all --num 5
"""

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.patches import Patch, Polygon, Rectangle

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from waymo_open_dataset.protos import scenario_pb2

    HAS_WAYMO = True
except ImportError:
    print("Warning: waymo_open_dataset not found. Scenario format not available.")
    HAS_WAYMO = False


class BEVMovieGenerator:
    """Generate bird's-eye view movies following the ego vehicle."""

    # Colors for different elements
    COLORS = {
        "ego": "#FF0000",  # Red for ego vehicle
        "vehicle": "#3498db",  # Blue for other vehicles
        "pedestrian": "#2ecc71",  # Green for pedestrians
        "cyclist": "#f39c12",  # Orange for cyclists
        "other": "#95a5a6",  # Gray for others
        "lane": "#34495e",  # Dark gray for lanes
        "road_line": "#7f8c8d",  # Medium gray for road lines
        "road_edge": "#2c3e50",  # Very dark gray for road edges
        "crosswalk": "#16a085",  # Teal for crosswalks
        "stop_sign": "#c0392b",  # Dark red for stop signs
        "speed_bump": "#d35400",  # Dark orange for speed bumps
        "history": "#e74c3c",  # Red for ego history trail
    }

    def __init__(
        self, output_dir: Path, fps: int = 10, dpi: int = 100, view_range: float = 50.0
    ):
        """
        Initialize BEV movie generator.

        Args:
            output_dir: Directory for output files
            fps: Frames per second
            dpi: Resolution quality
            view_range: Viewing range in meters (how far to show around ego)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.dpi = dpi
        self.view_range = view_range

        print(f"BEV Movie Generator initialized:")
        print(f"  Output: {self.output_dir}")
        print(f"  FPS: {fps}, DPI: {dpi}, Range: {view_range}m")

    def load_scenario(self, tfrecord_path: str, scenario_index: int = 0):
        """Load scenario from TFRecord file."""
        if not HAS_WAYMO:
            raise ImportError(
                "waymo_open_dataset is required to load scenarios. "
                "Install with: pip install waymo-open-dataset-tf-2-11-0"
            )
        dataset = tf.data.TFRecordDataset([tfrecord_path], compression_type="")

        for idx, data in enumerate(dataset):
            if idx == scenario_index:
                scenario = scenario_pb2.Scenario()
                scenario.ParseFromString(data.numpy())
                return scenario

        raise ValueError(
            f"Scenario index {scenario_index} not found in {tfrecord_path}"
        )

    def extract_data(self, scenario):
        """Extract trajectories and map features from scenario."""
        # Find ego vehicle (SDC)
        sdc_track_index = scenario.sdc_track_index
        sdc_track = scenario.tracks[sdc_track_index]

        # Extract ego trajectory
        ego_trajectory = []
        for state in sdc_track.states:
            if state.valid:
                ego_trajectory.append(
                    {
                        "x": state.center_x,
                        "y": state.center_y,
                        "heading": state.heading,
                        "length": state.length,
                        "width": state.width,
                        "velocity_x": state.velocity_x,
                        "velocity_y": state.velocity_y,
                    }
                )
            else:
                ego_trajectory.append(None)

        # Extract other agents
        agents_trajectories = []
        for track_idx, track in enumerate(scenario.tracks):
            if track_idx == sdc_track_index:
                continue  # Skip ego

            trajectory = []
            for state in track.states:
                if state.valid:
                    trajectory.append(
                        {
                            "x": state.center_x,
                            "y": state.center_y,
                            "heading": state.heading,
                            "length": state.length,
                            "width": state.width,
                            "velocity_x": state.velocity_x,
                            "velocity_y": state.velocity_y,
                            "type": track.object_type,
                        }
                    )
                else:
                    trajectory.append(None)

            agents_trajectories.append(trajectory)

        # Extract map features
        lanes = []
        road_lines = []
        road_edges = []
        crosswalks = []
        stop_signs = []
        speed_bumps = []

        for feature in scenario.map_features:
            if feature.HasField("lane"):
                lane_points = [(p.x, p.y) for p in feature.lane.polyline]
                lanes.append(lane_points)
            elif feature.HasField("road_line"):
                line_points = [(p.x, p.y) for p in feature.road_line.polyline]
                road_lines.append(line_points)
            elif feature.HasField("road_edge"):
                edge_points = [(p.x, p.y) for p in feature.road_edge.polyline]
                road_edges.append(edge_points)
            elif feature.HasField("crosswalk"):
                crosswalk_points = [(p.x, p.y) for p in feature.crosswalk.polygon]
                crosswalks.append(crosswalk_points)
            elif feature.HasField("stop_sign"):
                pos = feature.stop_sign.position
                stop_signs.append((pos.x, pos.y))
            elif feature.HasField("speed_bump"):
                bump_points = [(p.x, p.y) for p in feature.speed_bump.polygon]
                speed_bumps.append(bump_points)

        return {
            "ego": ego_trajectory,
            "agents": agents_trajectories,
            "lanes": lanes,
            "road_lines": road_lines,
            "road_edges": road_edges,
            "crosswalks": crosswalks,
            "stop_signs": stop_signs,
            "speed_bumps": speed_bumps,
            "scenario_id": scenario.scenario_id,
            "num_frames": len(ego_trajectory),
        }

    def generate_movie(
        self, scenario, output_filename: str = None, follow_ego: bool = True
    ):
        """
        Generate BEV movie.

        Args:
            scenario: Scenario protobuf
            output_filename: Output file name (default: based on scenario ID)
            follow_ego: If True, camera follows ego vehicle
        """
        if output_filename is None:
            output_filename = f"bev_scenario_{scenario.scenario_id}.mp4"

        output_path = self.output_dir / output_filename

        print(f"\nGenerating BEV movie for scenario {scenario.scenario_id}")

        # Extract data
        data = self.extract_data(scenario)

        # Setup figure
        fig, ax = plt.subplots(figsize=(12, 12))

        # Initialize elements
        ego_patch = None
        agent_patches = []
        lane_lines = []
        road_line_lines = []
        crosswalk_patches = []
        history_line = None

        def init():
            """Initialize animation."""
            ax.clear()
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)
            ax.set_xlabel("X (meters)", fontsize=12)
            ax.set_ylabel("Y (meters)", fontsize=12)
            return []

        def update(frame_idx):
            """Update animation frame."""
            ax.clear()
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

            # Get ego state at this frame
            ego_state = data["ego"][frame_idx]

            if ego_state is None:
                # Ego not valid at this frame
                ax.set_title(f"Frame {frame_idx}/{data['num_frames']} - Ego Not Valid")
                return []

            ego_x, ego_y = ego_state["x"], ego_state["y"]
            ego_heading = ego_state["heading"]

            # Set view range around ego (if following)
            if follow_ego:
                ax.set_xlim(ego_x - self.view_range, ego_x + self.view_range)
                ax.set_ylim(ego_y - self.view_range, ego_y + self.view_range)
                ax.set_xlabel("X relative to ego (meters)", fontsize=12)
                ax.set_ylabel("Y relative to ego (meters)", fontsize=12)

            # Draw map features (lanes, road lines, road edges, crosswalks, stop signs, speed bumps)
            # Draw lanes
            for lane_points in data["lanes"]:
                if follow_ego:
                    lane_points = [
                        (x, y)
                        for x, y in lane_points
                        if abs(x - ego_x) < self.view_range
                        and abs(y - ego_y) < self.view_range
                    ]
                if len(lane_points) > 1:
                    xs, ys = zip(*lane_points)
                    ax.plot(
                        xs,
                        ys,
                        color=self.COLORS["lane"],
                        linewidth=1.5,
                        alpha=0.4,
                        linestyle="-",
                    )

            # Draw road lines
            for line_points in data["road_lines"]:
                if follow_ego:
                    line_points = [
                        (x, y)
                        for x, y in line_points
                        if abs(x - ego_x) < self.view_range
                        and abs(y - ego_y) < self.view_range
                    ]
                if len(line_points) > 1:
                    xs, ys = zip(*line_points)
                    ax.plot(
                        xs,
                        ys,
                        color=self.COLORS["road_line"],
                        linewidth=1.0,
                        alpha=0.5,
                        linestyle="--",
                    )

            # Draw road edges (solid boundary lines)
            for edge_points in data.get("road_edges", []):
                if follow_ego:
                    edge_points = [
                        (x, y)
                        for x, y in edge_points
                        if abs(x - ego_x) < self.view_range
                        and abs(y - ego_y) < self.view_range
                    ]
                if len(edge_points) > 1:
                    xs, ys = zip(*edge_points)
                    ax.plot(
                        xs,
                        ys,
                        color=self.COLORS["road_edge"],
                        linewidth=2.0,
                        alpha=0.6,
                        linestyle="-",
                    )

            # Draw crosswalks
            for crosswalk_points in data["crosswalks"]:
                if follow_ego:
                    crosswalk_points = [
                        (x, y)
                        for x, y in crosswalk_points
                        if abs(x - ego_x) < self.view_range
                        and abs(y - ego_y) < self.view_range
                    ]
                if len(crosswalk_points) > 2:
                    polygon = Polygon(
                        crosswalk_points,
                        closed=True,
                        facecolor=self.COLORS["crosswalk"],
                        edgecolor=self.COLORS["crosswalk"],
                        linewidth=2,
                        alpha=0.3,
                    )
                    ax.add_patch(polygon)

            # Draw stop signs
            from matplotlib.patches import Circle

            for stop_pos in data.get("stop_signs", []):
                sx, sy = stop_pos
                if follow_ego:
                    if (
                        abs(sx - ego_x) > self.view_range
                        or abs(sy - ego_y) > self.view_range
                    ):
                        continue
                circle = Circle(
                    (sx, sy), 1.5, color=self.COLORS["stop_sign"], alpha=0.7, zorder=5
                )
                ax.add_patch(circle)
                ax.text(
                    sx,
                    sy,
                    "STOP",
                    ha="center",
                    va="center",
                    fontsize=5,
                    fontweight="bold",
                    color="white",
                    zorder=6,
                )

            # Draw speed bumps
            for bump_points in data.get("speed_bumps", []):
                if follow_ego:
                    bump_points = [
                        (x, y)
                        for x, y in bump_points
                        if abs(x - ego_x) < self.view_range
                        and abs(y - ego_y) < self.view_range
                    ]
                if len(bump_points) > 2:
                    polygon = Polygon(
                        bump_points,
                        closed=True,
                        facecolor=self.COLORS["speed_bump"],
                        edgecolor=self.COLORS["speed_bump"],
                        linewidth=1.5,
                        alpha=0.5,
                    )
                    ax.add_patch(polygon)

            # Draw ego history trail with dashed line
            history_length = 15
            history_points = []
            for i in range(max(0, frame_idx - history_length), frame_idx):
                if data["ego"][i] is not None:
                    history_points.append((data["ego"][i]["x"], data["ego"][i]["y"]))

            if len(history_points) > 1:
                hx, hy = zip(*history_points)
                ax.plot(
                    hx,
                    hy,
                    color=self.COLORS["history"],
                    linewidth=2.5,
                    alpha=0.4,
                    linestyle="--",
                )
                # Mark start position with circle
                ax.plot(
                    hx[0],
                    hy[0],
                    "o",
                    color=self.COLORS["history"],
                    markersize=6,
                    markeredgecolor="black",
                    markeredgewidth=1,
                    alpha=0.7,
                )

            # Draw other agents with velocity arrows
            from matplotlib.patches import FancyArrow

            for agent_traj in data["agents"]:
                if frame_idx >= len(agent_traj):
                    continue
                agent_state = agent_traj[frame_idx]
                if agent_state is None:
                    continue

                # Check if in view range
                if follow_ego:
                    if (
                        abs(agent_state["x"] - ego_x) > self.view_range
                        or abs(agent_state["y"] - ego_y) > self.view_range
                    ):
                        continue

                # Get color based on type
                type_name = self._get_object_type_name(agent_state["type"])
                color = self.COLORS.get(type_name.lower(), self.COLORS["other"])

                # Draw as oriented rectangle
                rect = self._create_vehicle_patch(
                    agent_state["x"],
                    agent_state["y"],
                    agent_state["heading"],
                    agent_state["length"],
                    agent_state["width"],
                    color,
                )
                ax.add_patch(rect)

                # Draw velocity arrow for moving agents
                if "velocity_x" in agent_state and "velocity_y" in agent_state:
                    vx, vy = agent_state.get("velocity_x", 0), agent_state.get(
                        "velocity_y", 0
                    )
                    speed = np.sqrt(vx**2 + vy**2)
                    if speed > 0.5:  # Only show if moving
                        arrow = FancyArrow(
                            agent_state["x"],
                            agent_state["y"],
                            vx * 0.5,
                            vy * 0.5,
                            width=0.3,
                            head_width=0.8,
                            head_length=0.5,
                            fc=color,
                            ec="black",
                            alpha=0.6,
                            linewidth=0.5,
                        )
                        ax.add_patch(arrow)

            # Draw ego vehicle (always on top)
            ego_rect = self._create_vehicle_patch(
                ego_x,
                ego_y,
                ego_heading,
                ego_state["length"],
                ego_state["width"],
                self.COLORS["ego"],
            )
            ax.add_patch(ego_rect)

            # Add velocity arrow for ego
            vel_scale = 2.0
            ax.arrow(
                ego_x,
                ego_y,
                ego_state["velocity_x"] * vel_scale,
                ego_state["velocity_y"] * vel_scale,
                head_width=1.5,
                head_length=1.0,
                fc="red",
                ec="red",
                alpha=0.7,
                linewidth=2,
            )

            # Add coordinate frame indicator
            if follow_ego:
                # Draw heading indicator
                arrow_len = 5
                ax.arrow(
                    ego_x,
                    ego_y,
                    arrow_len * np.cos(ego_heading),
                    arrow_len * np.sin(ego_heading),
                    head_width=1,
                    head_length=1,
                    fc="yellow",
                    ec="yellow",
                    alpha=0.8,
                    linewidth=2,
                )

            # Title with frame info
            speed = np.sqrt(ego_state["velocity_x"] ** 2 + ego_state["velocity_y"] ** 2)
            ax.set_title(
                f"Bird's-Eye View - Frame {frame_idx}/{data['num_frames']}\n"
                f"Scenario: {data['scenario_id'][:16]}... | "
                f"Speed: {speed:.1f} m/s ({speed * 3.6:.1f} km/h)",
                fontsize=14,
                fontweight="bold",
            )

            # Add comprehensive legend
            from matplotlib.lines import Line2D
            from matplotlib.patches import Patch

            legend_elements = [
                Patch(
                    facecolor=self.COLORS["ego"],
                    edgecolor="black",
                    label="Ego Vehicle (SDC)",
                ),
                Patch(
                    facecolor=self.COLORS["vehicle"],
                    edgecolor="black",
                    label="Other Vehicles",
                ),
                Patch(
                    facecolor=self.COLORS["pedestrian"],
                    edgecolor="black",
                    label="Pedestrians",
                ),
                Patch(
                    facecolor=self.COLORS["cyclist"],
                    edgecolor="black",
                    label="Cyclists",
                ),
                Line2D(
                    [0],
                    [0],
                    color=self.COLORS["history"],
                    linestyle="--",
                    linewidth=2,
                    label="History Trail",
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
                Patch(
                    facecolor=self.COLORS["crosswalk"], alpha=0.5, label="Crosswalks"
                ),
            ]
            ax.legend(
                handles=legend_elements,
                loc="upper right",
                fontsize=9,
                framealpha=0.9,
                ncol=1,
            )

            return []

        # Create animation
        num_frames = data["num_frames"]
        anim = FuncAnimation(
            fig,
            update,
            init_func=init,
            frames=num_frames,
            interval=1000 / self.fps,
            blit=False,
            repeat=True,
        )

        # Save animation
        print(f"  Rendering {num_frames} frames...")
        try:
            # Save MP4
            writer = FFMpegWriter(
                fps=self.fps,
                bitrate=1800,
                codec="libx264",
                extra_args=["-pix_fmt", "yuv420p"],
            )
            anim.save(str(output_path), writer=writer, dpi=self.dpi)

            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"  ✓ Saved MP4: {output_path.name} ({file_size_mb:.1f} MB)")

            # Save GIF alongside MP4
            gif_path = output_path.with_suffix(".gif")
            gif_fps = min(self.fps, 10)  # GIFs work better with lower fps
            gif_writer = PillowWriter(fps=gif_fps)
            anim.save(str(gif_path), writer=gif_writer, dpi=max(self.dpi // 2, 50))

            gif_size_mb = gif_path.stat().st_size / (1024 * 1024)
            print(f"  ✓ Saved GIF: {gif_path.name} ({gif_size_mb:.1f} MB)")

            return True

        except Exception as e:
            print(f"  ✗ Error saving movie: {e}")
            return False
        finally:
            plt.close(fig)

    def _create_vehicle_patch(self, x, y, heading, length, width, color, ax=None):
        """Create oriented rectangle patch for vehicle."""
        # Create rectangle centered at origin
        rect = Rectangle(
            (-length / 2, -width / 2),
            length,
            width,
            facecolor=color,
            edgecolor="black",
            linewidth=1.5,
            alpha=0.8,
        )

        # Apply rotation and translation
        import matplotlib.transforms as transforms

        if ax is None:
            ax = plt.gca()
        t = transforms.Affine2D().rotate(heading).translate(x, y) + ax.transData
        rect.set_transform(t)

        return rect

    def _get_object_type_name(self, type_id: int) -> str:
        """Convert object type ID to name."""
        type_map = {
            0: "other",
            1: "vehicle",
            2: "pedestrian",
            3: "cyclist",
            4: "other",
        }
        return type_map.get(type_id, "other")


# Base data path
DATA_BASE = Path("/workspace/data/WOMD/datasets/waymo_open_dataset/motion_v_1_3_0")
MOVIES_BASE = Path("/workspace/data/WOMD/movies")

# Available splits
SPLITS = [
    "training_interactive",
    "validation_interactive",
    "testing_interactive",
]


def get_data_dir(format_type: str, split: str) -> Path:
    """Get data directory for given format and split."""
    if format_type == "scenario":
        return DATA_BASE / "raw" / "scenario" / split
    elif format_type == "tf":
        return DATA_BASE / "processed" / "tf" / split
    else:
        raise ValueError(f"Unknown format: {format_type}")


def get_output_dir(format_type: str, split: str) -> Path:
    """Get output directory for given format and split."""
    return MOVIES_BASE / "bev" / format_type / split


def process_scenario_format(
    generator, data_dir: Path, output_dir: Path, num_scenarios: int, follow_ego: bool
) -> int:
    """Process scenarios from raw scenario format."""
    if not HAS_WAYMO:
        print("  ERROR: waymo_open_dataset package required for scenario format")
        return 0

    tfrecord_files = sorted(data_dir.glob("*.tfrecord*"))
    if not tfrecord_files:
        print(f"  No tfrecord files found in {data_dir}")
        return 0

    print(f"  Found {len(tfrecord_files)} files")

    processed = 0
    for tfrecord_path in tfrecord_files[:num_scenarios]:
        try:
            scenario = generator.load_scenario(str(tfrecord_path), 0)
            output_name = f"bev_{scenario.scenario_id}.mp4"

            # Temporarily change output dir
            old_output = generator.output_dir
            generator.output_dir = output_dir

            success = generator.generate_movie(
                scenario, output_filename=output_name, follow_ego=follow_ego
            )

            generator.output_dir = old_output

            if success:
                processed += 1

        except Exception as e:
            print(f"  Error processing {tfrecord_path.name}: {e}")
            continue

    return processed


def process_tf_format(
    generator, data_dir: Path, output_dir: Path, num_scenarios: int, follow_ego: bool
) -> int:
    """
    Process scenarios from processed tf.Example format.

    TF format has flattened features - we parse them and create visualizations.
    """
    tfrecord_files = sorted(data_dir.glob("*.tfrecord*"))
    if not tfrecord_files:
        print(f"  No tfrecord files found in {data_dir}")
        return 0

    print(f"  Found {len(tfrecord_files)} files")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load examples from tfrecords
    examples = []
    for tfrecord_file in tfrecord_files:
        if len(examples) >= num_scenarios:
            break

        dataset = tf.data.TFRecordDataset(str(tfrecord_file))
        for i, raw_record in enumerate(dataset):
            if len(examples) >= num_scenarios:
                break

            example = tf.train.Example()
            example.ParseFromString(raw_record.numpy())
            file_id = f"{tfrecord_file.stem}_{i:04d}"
            examples.append((file_id, example))

    if not examples:
        print("  No examples loaded")
        return 0

    print(f"  Loaded {len(examples)} examples")

    processed = 0
    for file_id, example in examples:
        try:
            print(f"\n  Processing: {file_id}")

            # Parse TF example
            data = parse_tf_example(example)

            # Generate BEV movie for TF format
            output_file = output_dir / f"bev_tf_{file_id}.mp4"
            success = generate_tf_bev_movie(
                data,
                output_file,
                fps=generator.fps,
                dpi=generator.dpi,
                view_range=generator.view_range,
                follow_ego=follow_ego,
            )

            if success:
                processed += 1

        except Exception as e:
            print(f"  Error processing {file_id}: {e}")
            import traceback

            traceback.print_exc()
            continue

    return processed


def parse_tf_example(example) -> dict:
    """Parse tf.train.Example into visualization data."""
    features = example.features.feature

    def get_float_list(key):
        if key in features:
            return np.array(features[key].float_list.value)
        return np.array([])

    def get_int_list(key):
        if key in features:
            return np.array(features[key].int64_list.value)
        return np.array([])

    # Get number of agents
    num_agents = len(get_float_list("state/current/x"))

    # Parse state data - reshape from flat to [num_agents, num_timesteps]
    def parse_state(prefix, num_timesteps):
        x = get_float_list(f"{prefix}/x")
        y = get_float_list(f"{prefix}/y")
        valid = get_int_list(f"{prefix}/valid")
        heading = get_float_list(f"{prefix}/bbox_yaw")
        length = get_float_list(f"{prefix}/length")
        width = get_float_list(f"{prefix}/width")

        expected = num_agents * num_timesteps
        if len(x) == expected:
            return {
                "x": x.reshape(num_agents, num_timesteps),
                "y": y.reshape(num_agents, num_timesteps),
                "valid": valid.reshape(num_agents, num_timesteps).astype(bool),
                "heading": heading.reshape(num_agents, num_timesteps),
                "length": length.reshape(num_agents, num_timesteps),
                "width": width.reshape(num_agents, num_timesteps),
            }
        return None

    state_past = parse_state("state/past", 10)
    state_current = parse_state("state/current", 1)
    state_future = parse_state("state/future", 80)

    # Parse roadgraph
    roadgraph_xyz = get_float_list("roadgraph_samples/xyz")
    roadgraph_valid = get_int_list("roadgraph_samples/valid")

    # Parse agent types
    agent_type = get_float_list("state/type")
    is_sdc = get_int_list("state/is_sdc")

    return {
        "state_past": state_past,
        "state_current": state_current,
        "state_future": state_future,
        "roadgraph_xyz": roadgraph_xyz,
        "roadgraph_valid": roadgraph_valid,
        "agent_type": agent_type,
        "is_sdc": is_sdc,
        "num_agents": num_agents,
    }


def generate_tf_bev_movie(
    data: dict,
    output_path: Path,
    fps: int = 10,
    dpi: int = 100,
    view_range: float = 50.0,
    follow_ego: bool = True,
) -> bool:
    """Generate BEV movie from TF format data."""

    # Colors
    COLORS = {
        "ego": "#FF0000",
        "vehicle": "#3498db",
        "pedestrian": "#2ecc71",
        "cyclist": "#f39c12",
        "other": "#95a5a6",
        "roadmap": "#7f8c8d",
        "history": "#e74c3c",
    }

    # Find SDC (ego vehicle)
    sdc_idx = None
    if len(data["is_sdc"]) > 0:
        sdc_indices = np.where(data["is_sdc"] == 1)[0]
        if len(sdc_indices) > 0:
            sdc_idx = sdc_indices[0]

    # If no SDC found, use first valid agent
    if sdc_idx is None:
        if data["state_current"] is not None:
            valid_agents = np.where(data["state_current"]["valid"][:, 0])[0]
            if len(valid_agents) > 0:
                sdc_idx = valid_agents[0]

    if sdc_idx is None:
        print("  No valid ego vehicle found")
        return False

    # Total frames (past + current + future)
    total_frames = 10 + 1 + 80  # 91 frames

    fig, ax = plt.subplots(figsize=(12, 12))

    # Compute plot limits
    all_x, all_y = [], []
    if len(data["roadgraph_xyz"]) > 0:
        xyz = data["roadgraph_xyz"].reshape(-1, 3)
        valid = data["roadgraph_valid"].reshape(-1) > 0
        if np.any(valid):
            all_x.extend(xyz[valid, 0])
            all_y.extend(xyz[valid, 1])

    # Add agent positions
    for state_key in ["state_past", "state_current", "state_future"]:
        if data[state_key] is not None:
            x = data[state_key]["x"]
            y = data[state_key]["y"]
            valid = data[state_key]["valid"]
            if len(x) > 0 and np.any(valid):
                all_x.extend(x[valid])
                all_y.extend(y[valid])

    if len(all_x) > 0:
        x_center = np.mean(all_x)
        y_center = np.mean(all_y)
    else:
        x_center, y_center = 0, 0

    def get_state_at_frame(frame_idx):
        """Get state data for a specific frame index."""
        if frame_idx < 10:
            # Past frames
            state = data["state_past"]
            t_idx = frame_idx
        elif frame_idx == 10:
            # Current frame
            state = data["state_current"]
            t_idx = 0
        else:
            # Future frames
            state = data["state_future"]
            t_idx = frame_idx - 11

        if state is None:
            return None

        return {
            "x": state["x"][:, t_idx],
            "y": state["y"][:, t_idx],
            "valid": state["valid"][:, t_idx],
            "heading": state["heading"][:, t_idx],
            "length": state["length"][:, t_idx],
            "width": state["width"][:, t_idx],
        }

    def init():
        ax.clear()
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        return []

    def update(frame_idx):
        ax.clear()
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

        state = get_state_at_frame(frame_idx)
        if state is None or not state["valid"][sdc_idx]:
            ax.set_title(f"Frame {frame_idx}/{total_frames} - No Valid Data")
            return []

        ego_x = state["x"][sdc_idx]
        ego_y = state["y"][sdc_idx]
        ego_heading = state["heading"][sdc_idx]

        # Set view
        if follow_ego:
            ax.set_xlim(ego_x - view_range, ego_x + view_range)
            ax.set_ylim(ego_y - view_range, ego_y + view_range)
        else:
            ax.set_xlim(x_center - view_range * 2, x_center + view_range * 2)
            ax.set_ylim(y_center - view_range * 2, y_center + view_range * 2)

        # Draw roadgraph
        if len(data["roadgraph_xyz"]) > 0:
            xyz = data["roadgraph_xyz"].reshape(-1, 3)
            valid = data["roadgraph_valid"].reshape(-1) > 0
            if np.any(valid):
                pts = xyz[valid]
                # Filter to view range
                if follow_ego:
                    in_range = (np.abs(pts[:, 0] - ego_x) < view_range) & (
                        np.abs(pts[:, 1] - ego_y) < view_range
                    )
                    pts = pts[in_range]
                if len(pts) > 0:
                    ax.scatter(
                        pts[:, 0], pts[:, 1], c=COLORS["roadmap"], s=1, alpha=0.3
                    )

        # Draw other agents
        for agent_idx in range(min(data["num_agents"], 32)):
            if agent_idx == sdc_idx:
                continue
            if not state["valid"][agent_idx]:
                continue

            x = state["x"][agent_idx]
            y = state["y"][agent_idx]

            # Check if in view
            if follow_ego:
                if abs(x - ego_x) > view_range or abs(y - ego_y) > view_range:
                    continue

            # Get agent type color
            if len(data["agent_type"]) > agent_idx:
                atype = int(data["agent_type"][agent_idx])
                type_map = {1: "vehicle", 2: "pedestrian", 3: "cyclist"}
                color = COLORS.get(type_map.get(atype, "other"), COLORS["other"])
            else:
                color = COLORS["other"]

            # Draw agent rectangle
            heading = state["heading"][agent_idx]
            length = max(state["length"][agent_idx], 1.0)
            width = max(state["width"][agent_idx], 0.5)

            rect = Rectangle(
                (-length / 2, -width / 2),
                length,
                width,
                facecolor=color,
                edgecolor="black",
                linewidth=1,
                alpha=0.7,
            )
            import matplotlib.transforms as transforms

            t = transforms.Affine2D().rotate(heading).translate(x, y) + ax.transData
            rect.set_transform(t)
            ax.add_patch(rect)

        # Draw ego vehicle
        ego_length = max(state["length"][sdc_idx], 4.0)
        ego_width = max(state["width"][sdc_idx], 2.0)

        ego_rect = Rectangle(
            (-ego_length / 2, -ego_width / 2),
            ego_length,
            ego_width,
            facecolor=COLORS["ego"],
            edgecolor="black",
            linewidth=2,
            alpha=0.9,
        )
        import matplotlib.transforms as transforms

        t = (
            transforms.Affine2D().rotate(ego_heading).translate(ego_x, ego_y)
            + ax.transData
        )
        ego_rect.set_transform(t)
        ax.add_patch(ego_rect)

        # Draw heading arrow
        arrow_len = 5
        ax.arrow(
            ego_x,
            ego_y,
            arrow_len * np.cos(ego_heading),
            arrow_len * np.sin(ego_heading),
            head_width=1,
            head_length=1,
            fc="yellow",
            ec="yellow",
            alpha=0.8,
            linewidth=2,
        )

        # Phase label
        if frame_idx < 10:
            phase = f"Past: t={frame_idx - 10}"
        elif frame_idx == 10:
            phase = "Current: t=0"
        else:
            phase = f"Future: t={frame_idx - 10}"

        ax.set_title(
            f"BEV (TF Format) - Frame {frame_idx}/{total_frames}\n{phase}",
            fontsize=14,
            fontweight="bold",
        )

        ax.set_xlabel("X (meters)", fontsize=12)
        ax.set_ylabel("Y (meters)", fontsize=12)

        return []

    anim = FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=total_frames,
        interval=1000 / fps,
        blit=False,
        repeat=True,
    )

    print(f"  Rendering {total_frames} frames...")
    try:
        # Save MP4
        writer = FFMpegWriter(
            fps=fps, bitrate=1800, codec="libx264", extra_args=["-pix_fmt", "yuv420p"]
        )
        anim.save(str(output_path), writer=writer, dpi=dpi)

        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"  ✓ Saved MP4: {output_path.name} ({file_size_mb:.1f} MB)")

        # Save GIF alongside MP4
        gif_path = output_path.with_suffix(".gif")
        gif_fps = min(fps, 10)  # GIFs work better with lower fps
        gif_writer = PillowWriter(fps=gif_fps)
        anim.save(str(gif_path), writer=gif_writer, dpi=max(dpi // 2, 50))

        gif_size_mb = gif_path.stat().st_size / (1024 * 1024)
        print(f"  ✓ Saved GIF: {gif_path.name} ({gif_size_mb:.1f} MB)")

        return True

    except Exception as e:
        print(f"  ✗ Error saving: {e}")
        return False
    finally:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Generate bird's-eye view movies from Waymo data"
    )

    # Input options
    parser.add_argument("--tfrecord", type=str, help="Path to single TFRecord file")
    parser.add_argument(
        "--format",
        type=str,
        choices=["scenario", "tf"],
        help="Data format: scenario (raw) or tf (processed)",
    )
    parser.add_argument(
        "--split", type=str, choices=SPLITS, help="Dataset split to process"
    )
    parser.add_argument(
        "--all", action="store_true", help="Process all formats and splits"
    )

    # Processing options
    parser.add_argument(
        "--num",
        "-n",
        type=int,
        default=5,
        help="Number of scenarios per split (default: 5)",
    )
    parser.add_argument(
        "--scenario-index",
        type=int,
        default=0,
        help="Scenario index in file (for --tfrecord)",
    )

    # Output options
    parser.add_argument("--output-dir", type=Path, help="Custom output directory")
    parser.add_argument(
        "--fps", type=int, default=10, help="Frames per second (default: 10)"
    )
    parser.add_argument(
        "--dpi", type=int, default=100, help="Resolution quality (default: 100)"
    )
    parser.add_argument(
        "--view-range",
        type=float,
        default=50.0,
        help="Viewing range around ego in meters (default: 50)",
    )
    parser.add_argument(
        "--no-follow",
        action="store_true",
        help="Do not follow ego vehicle (fixed view)",
    )
    parser.add_argument("--gif", action="store_true", help="Also generate GIF versions")

    args = parser.parse_args()

    # Validate arguments
    if not args.tfrecord and not args.format and not args.all:
        parser.error("Must specify --tfrecord, --format/--split, or --all")

    if args.format and not args.split:
        parser.error("--format requires --split")

    print("=" * 60)
    print("BEV Movie Generator")
    print("=" * 60)

    total_processed = 0

    # Single file mode
    if args.tfrecord:
        output_dir = args.output_dir or MOVIES_BASE / "bev" / "single"
        generator = BEVMovieGenerator(
            output_dir=output_dir,
            fps=args.fps,
            dpi=args.dpi,
            view_range=args.view_range,
        )

        print(f"Processing: {args.tfrecord}")
        scenario = generator.load_scenario(args.tfrecord, args.scenario_index)
        if generator.generate_movie(scenario, follow_ego=not args.no_follow):
            total_processed = 1

    # Format/split mode
    elif args.format and args.split:
        data_dir = get_data_dir(args.format, args.split)
        output_dir = args.output_dir or get_output_dir(args.format, args.split)

        generator = BEVMovieGenerator(
            output_dir=output_dir,
            fps=args.fps,
            dpi=args.dpi,
            view_range=args.view_range,
        )

        print(f"\nFormat: {args.format}")
        print(f"Split: {args.split}")
        print(f"Input: {data_dir}")
        print(f"Output: {output_dir}")
        print(f"Scenarios: {args.num}")

        if not data_dir.exists():
            print(f"ERROR: Directory not found: {data_dir}")
            return 1

        if args.format == "scenario":
            total_processed = process_scenario_format(
                generator, data_dir, output_dir, args.num, not args.no_follow
            )
        else:
            total_processed = process_tf_format(
                generator, data_dir, output_dir, args.num, not args.no_follow
            )

    # Process all formats and splits
    elif args.all:
        print(f"Processing ALL formats and splits")
        print(f"Scenarios per split: {args.num}")

        for format_type in ["scenario", "tf"]:
            print(f"\n{'=' * 60}")
            print(f"Format: {format_type.upper()}")
            print("=" * 60)

            for split in SPLITS:
                data_dir = get_data_dir(format_type, split)
                output_dir = get_output_dir(format_type, split)

                print(f"\n--- {split} ---")
                print(f"  Input: {data_dir}")
                print(f"  Output: {output_dir}")

                if not data_dir.exists():
                    print(f"  SKIP: Directory not found")
                    continue

                output_dir.mkdir(parents=True, exist_ok=True)

                generator = BEVMovieGenerator(
                    output_dir=output_dir,
                    fps=args.fps,
                    dpi=args.dpi,
                    view_range=args.view_range,
                )

                if format_type == "scenario":
                    count = process_scenario_format(
                        generator, data_dir, output_dir, args.num, not args.no_follow
                    )
                else:
                    count = process_tf_format(
                        generator, data_dir, output_dir, args.num, not args.no_follow
                    )

                total_processed += count
                print(f"  Generated: {count} movies")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"COMPLETE: Generated {total_processed} BEV movies")
    print(f"Output: {MOVIES_BASE / 'bev'}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
