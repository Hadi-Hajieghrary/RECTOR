#!/usr/bin/env python3
"""
RECTOR Visualization Utilities

Create Bird's-Eye-View (BEV) visualizations for RECTOR planning results.

Features:
1. Static frame rendering
2. Animated planning evolution
3. Multi-candidate comparison views
4. Safety envelope visualization
5. Movie generation

Output Formats:
- PNG frames
- MP4 animations
- Interactive matplotlib displays
"""

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

# Setup paths
RECTOR_ROOT = Path(__file__).parent.parent.parent
RECTOR_LIB = Path(__file__).parent  # Already in lib
sys.path.insert(0, str(RECTOR_LIB))

# Visualization imports
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection

# RECTOR imports
from data_contracts import (
    AgentState,
    EgoCandidate,
    EgoCandidateBatch,
)
from planning_loop import PlanningResult, LaneInfo

# Optional moviepy for animations
try:
    import imageio

    IMAGEIO_AVAILABLE = True
except ImportError:
    IMAGEIO_AVAILABLE = False


# Color palette for candidates (gradient from bad to good)
CANDIDATE_CMAP = LinearSegmentedColormap.from_list(
    "candidate_score", ["#ff4444", "#ffaa44", "#44ff44"]
)

# Fixed colors
COLOR_EGO = "#2196F3"  # Blue
COLOR_EGO_SELECTED = "#4CAF50"  # Green
COLOR_REACTOR = "#FF5722"  # Orange
COLOR_OTHER = "#9E9E9E"  # Gray
COLOR_ROAD = "#E0E0E0"  # Light gray
COLOR_LANE = "#90A4AE"  # Blue-gray
COLOR_ENVELOPE = "#FFEB3B"  # Yellow (warning)
COLOR_COLLISION = "#F44336"  # Red


@dataclass
class BEVConfig:
    """Configuration for BEV rendering."""

    width: int = 1200  # Image width in pixels
    height: int = 800  # Image height in pixels
    meters_per_pixel: float = 0.3  # Scale factor
    ego_centered: bool = True  # Center view on ego
    x_offset: float = 0.0  # Additional X offset (meters)
    y_offset: float = 0.0  # Additional Y offset (meters)
    show_grid: bool = True  # Show coordinate grid
    show_legend: bool = True  # Show legend
    trail_length: int = 10  # History trail length (timesteps)

    @property
    def x_range(self) -> Tuple[float, float]:
        """X axis range in meters."""
        half = (self.width / 2) * self.meters_per_pixel
        return (-half + self.x_offset, half + self.x_offset)

    @property
    def y_range(self) -> Tuple[float, float]:
        """Y axis range in meters."""
        half = (self.height / 2) * self.meters_per_pixel
        return (-half + self.y_offset, half + self.y_offset)


class BEVRenderer:
    """
    Render Bird's-Eye-View visualizations of RECTOR planning.

    All coordinates are in the EGO-CENTRIC frame where:
    - Ego is at origin
    - X points right
    - Y points forward
    """

    def __init__(self, config: BEVConfig = None):
        self.config = config or BEVConfig()
        self.fig = None
        self.ax = None

    def _setup_figure(self):
        """Create figure and axes."""
        dpi = 100
        fig, ax = plt.subplots(
            figsize=(self.config.width / dpi, self.config.height / dpi), dpi=dpi
        )

        # Set axis limits
        ax.set_xlim(self.config.x_range)
        ax.set_ylim(self.config.y_range)
        ax.set_aspect("equal")

        # Style
        ax.set_facecolor("#1a1a2e")  # Dark background
        ax.grid(self.config.show_grid, color="#333", linestyle="--", alpha=0.3)
        ax.tick_params(colors="#888")
        for spine in ax.spines.values():
            spine.set_color("#333")

        self.fig = fig
        self.ax = ax
        return fig, ax

    def _transform_to_ego_frame(
        self,
        points: np.ndarray,
        ego_state: AgentState,
    ) -> np.ndarray:
        """Transform world coordinates to ego-centric frame."""
        if len(points.shape) == 1:
            points = points.reshape(1, -1)

        # Translate to ego position
        centered = points - np.array([ego_state.x, ego_state.y])

        # Rotate to align with ego heading
        cos_h = np.cos(-ego_state.heading)
        sin_h = np.sin(-ego_state.heading)
        rotated = np.column_stack(
            [
                centered[:, 0] * cos_h - centered[:, 1] * sin_h,
                centered[:, 0] * sin_h + centered[:, 1] * cos_h,
            ]
        )

        # Swap axes so ego faces up (Y direction)
        return rotated[:, [1, 0]] * np.array([1, -1])

    def render_road_graph(
        self,
        road_points: np.ndarray,
        road_types: np.ndarray,
        ego_state: AgentState,
    ):
        """Render road network points."""
        if road_points is None or len(road_points) == 0:
            return

        transformed = self._transform_to_ego_frame(road_points[:, :2], ego_state)

        # Different colors for different road types
        colors = np.where(road_types == 1, COLOR_LANE, COLOR_ROAD)

        self.ax.scatter(
            transformed[:, 0],
            transformed[:, 1],
            c=colors,
            s=1,
            alpha=0.5,
        )

    def render_lane(
        self,
        lane: LaneInfo,
        ego_state: AgentState,
        color: str = COLOR_LANE,
        linewidth: float = 2.0,
    ):
        """Render a lane centerline."""
        if lane is None or lane.centerline is None:
            return

        transformed = self._transform_to_ego_frame(lane.centerline, ego_state)

        self.ax.plot(
            transformed[:, 0],
            transformed[:, 1],
            color=color,
            linewidth=linewidth,
            linestyle="--",
            alpha=0.6,
            label="Lane",
        )

    def render_agent(
        self,
        agent: AgentState,
        ego_state: AgentState,
        color: str = COLOR_OTHER,
        label: Optional[str] = None,
        show_velocity: bool = True,
    ):
        """Render an agent as an oriented box."""
        pos = self._transform_to_ego_frame(agent.position, ego_state)[0]
        heading = agent.heading - ego_state.heading + np.pi / 2  # Adjust for ego frame

        # Draw oriented rectangle
        rect = patches.Rectangle(
            (-agent.length / 2, -agent.width / 2),
            agent.length,
            agent.width,
            linewidth=2,
            edgecolor=color,
            facecolor=color,
            alpha=0.6,
        )

        # Transform rectangle to agent position and heading
        t = matplotlib.transforms.Affine2D().rotate(heading).translate(*pos)
        rect.set_transform(t + self.ax.transData)
        self.ax.add_patch(rect)

        # Velocity arrow
        if show_velocity and agent.speed > 0.5:
            vel = (
                self._transform_to_ego_frame(
                    agent.position + agent.velocity, ego_state
                )[0]
                - pos
            )
            self.ax.arrow(
                pos[0],
                pos[1],
                vel[0] * 0.5,
                vel[1] * 0.5,
                head_width=0.5,
                head_length=0.3,
                fc=color,
                ec=color,
                alpha=0.8,
            )

        # Label
        if label:
            self.ax.text(
                pos[0],
                pos[1] + agent.width,
                label,
                fontsize=8,
                color=color,
                ha="center",
                va="bottom",
            )

    def render_ego(
        self,
        ego_state: AgentState,
        color: str = COLOR_EGO,
    ):
        """Render ego vehicle (always at origin in ego frame)."""
        # Ego is at origin, facing up
        rect = patches.Rectangle(
            (-ego_state.length / 2, -ego_state.width / 2),
            ego_state.length,
            ego_state.width,
            linewidth=3,
            edgecolor=color,
            facecolor=color,
            alpha=0.8,
        )

        # Rotate 90 degrees to face up
        t = matplotlib.transforms.Affine2D().rotate(np.pi / 2)
        rect.set_transform(t + self.ax.transData)
        self.ax.add_patch(rect)

        # Arrow indicating heading
        self.ax.arrow(
            0,
            ego_state.length / 2,
            0,
            2,
            head_width=0.8,
            head_length=0.5,
            fc=color,
            ec="white",
            linewidth=2,
        )

    def render_trajectory(
        self,
        trajectory: np.ndarray,
        ego_state: AgentState,
        color: str = COLOR_EGO,
        linewidth: float = 2.0,
        linestyle: str = "-",
        alpha: float = 0.8,
        label: Optional[str] = None,
        show_endpoint: bool = False,
    ):
        """Render a trajectory."""
        transformed = self._transform_to_ego_frame(trajectory, ego_state)

        self.ax.plot(
            transformed[:, 0],
            transformed[:, 1],
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            alpha=alpha,
            label=label,
        )

        if show_endpoint:
            self.ax.scatter(
                transformed[-1, 0],
                transformed[-1, 1],
                c=color,
                s=50,
                marker="*",
                zorder=10,
            )

    def render_candidates(
        self,
        candidates: EgoCandidateBatch,
        ego_state: AgentState,
        scores: Optional[np.ndarray] = None,
        selected_idx: Optional[int] = None,
    ):
        """Render all ego trajectory candidates with score-based coloring."""
        if scores is not None:
            # Normalize scores for colormap
            s_min, s_max = scores.min(), scores.max()
            if s_max > s_min:
                norm_scores = (scores - s_min) / (s_max - s_min)
            else:
                norm_scores = np.ones_like(scores) * 0.5
        else:
            norm_scores = np.linspace(0, 1, len(candidates.candidates))

        for i, cand in enumerate(candidates.candidates):
            is_selected = i == selected_idx

            color = CANDIDATE_CMAP(norm_scores[i])
            linewidth = 3.0 if is_selected else 1.5
            alpha = 1.0 if is_selected else 0.4

            label = "Selected" if is_selected else None

            self.render_trajectory(
                cand.trajectory,
                ego_state,
                color=color if not is_selected else COLOR_EGO_SELECTED,
                linewidth=linewidth,
                alpha=alpha,
                label=label,
                show_endpoint=is_selected,
            )

    def render_reactor_predictions(
        self,
        reactor_id: int,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        ego_state: AgentState,
        color: str = COLOR_REACTOR,
    ):
        """Render multi-modal predictions for a reactor."""
        # Sort by probability
        sorted_idx = np.argsort(probabilities)[::-1]

        for i, idx in enumerate(sorted_idx[:3]):  # Show top 3 modes
            traj = predictions[idx]
            prob = probabilities[idx]

            alpha = 0.3 + 0.4 * prob
            linewidth = 1.0 + 2.0 * prob

            self.render_trajectory(
                traj,
                ego_state,
                color=color,
                linewidth=linewidth,
                alpha=alpha,
                linestyle="--" if i > 0 else "-",
            )

    def render_safety_envelope(
        self,
        envelope: np.ndarray,
        ego_state: AgentState,
        color: str = COLOR_ENVELOPE,
    ):
        """Render a safety envelope polygon."""
        transformed = self._transform_to_ego_frame(envelope, ego_state)

        polygon = patches.Polygon(
            transformed,
            fill=True,
            facecolor=color,
            edgecolor=color,
            alpha=0.2,
            linewidth=1,
        )
        self.ax.add_patch(polygon)

    def render_collision_point(
        self,
        point: np.ndarray,
        ego_state: AgentState,
    ):
        """Render a collision point."""
        transformed = self._transform_to_ego_frame(point.reshape(1, 2), ego_state)[0]

        self.ax.scatter(
            transformed[0],
            transformed[1],
            c=COLOR_COLLISION,
            s=200,
            marker="X",
            zorder=20,
            edgecolors="white",
            linewidths=2,
        )

    def render_planning_result(
        self,
        result: PlanningResult,
        ego_state: AgentState,
        other_agents: List[AgentState],
        lane: Optional[LaneInfo] = None,
        reactor_ids: Optional[List[int]] = None,
        title: Optional[str] = None,
    ) -> plt.Figure:
        """
        Render a complete planning result frame.

        Args:
            result: PlanningResult from RECTOR planner
            ego_state: Current ego state
            other_agents: List of other agent states
            lane: Optional lane info
            reactor_ids: Optional list of reactor agent IDs
            title: Optional title

        Returns:
            matplotlib Figure
        """
        fig, ax = self._setup_figure()

        # Render lane
        if lane:
            self.render_lane(lane, ego_state)

        # Render other agents
        agent_by_id = {a.agent_id: a for a in other_agents}
        reactor_ids = reactor_ids or []

        for agent in other_agents:
            is_reactor = agent.agent_id in reactor_ids
            color = COLOR_REACTOR if is_reactor else COLOR_OTHER
            label = f"R{agent.agent_id}" if is_reactor else None
            self.render_agent(agent, ego_state, color=color, label=label)

        # Render candidates with scores
        # We need to reconstruct candidates from result
        # For now, just show selected trajectory
        self.render_trajectory(
            result.selected_candidate.trajectory,
            ego_state,
            color=COLOR_EGO_SELECTED,
            linewidth=3.0,
            label="Selected",
            show_endpoint=True,
        )

        # Render ego
        self.render_ego(ego_state)

        # Title and info
        if title:
            ax.set_title(title, color="white", fontsize=14)

        # Info box
        info_text = (
            f"Iteration: {result.iteration}\n"
            f"Selected: #{result.selected_idx}\n"
            f"Score: {result.all_scores[result.selected_idx]:.2f}"
        )
        ax.text(
            0.02,
            0.98,
            info_text,
            transform=ax.transAxes,
            fontsize=10,
            color="white",
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="#333", alpha=0.8),
        )

        # Legend
        if self.config.show_legend:
            ax.legend(
                loc="upper right",
                facecolor="#333",
                edgecolor="#555",
                labelcolor="white",
            )

        plt.tight_layout()
        return fig

    def save_frame(
        self,
        filepath: str,
        dpi: int = 100,
    ):
        """Save current figure to file."""
        if self.fig is not None:
            self.fig.savefig(filepath, dpi=dpi, facecolor="#1a1a2e", edgecolor="none")
            plt.close(self.fig)
            self.fig = None
            self.ax = None


class RECTORMovieGenerator:
    """Generate movies from RECTOR planning sequences."""

    def __init__(
        self,
        output_dir: str,
        fps: int = 10,
        config: BEVConfig = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.renderer = BEVRenderer(config)
        self.frames = []

    def add_frame(
        self,
        result: PlanningResult,
        ego_state: AgentState,
        other_agents: List[AgentState],
        lane: Optional[LaneInfo] = None,
        title: Optional[str] = None,
    ):
        """Add a frame to the movie."""
        fig = self.renderer.render_planning_result(
            result, ego_state, other_agents, lane, title=title
        )

        # Convert to numpy array
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        self.frames.append(data)
        plt.close(fig)

    def save_movie(self, filename: str = "rector_planning.mp4"):
        """Save accumulated frames as movie."""
        if not IMAGEIO_AVAILABLE:
            print("Warning: imageio not available, saving as PNG sequence")
            self.save_frames()
            return

        if not self.frames:
            print("No frames to save")
            return

        output_path = self.output_dir / filename

        # Use imageio to create movie
        imageio.mimwrite(
            str(output_path),
            self.frames,
            fps=self.fps,
            codec="libx264",
            quality=8,
        )

        print(f"Saved movie: {output_path}")
        self.frames = []

    def save_frames(self, prefix: str = "frame"):
        """Save frames as individual PNGs."""
        for i, frame in enumerate(self.frames):
            filepath = self.output_dir / f"{prefix}_{i:04d}.png"
            imageio.imwrite(str(filepath), frame)

        print(f"Saved {len(self.frames)} frames to {self.output_dir}")
        self.frames = []


def render_candidate_comparison(
    candidates: EgoCandidateBatch,
    scores: np.ndarray,
    ego_state: AgentState,
    lane: Optional[LaneInfo] = None,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Render a comparison view of all candidates side-by-side.

    Creates a grid of subplots, one per candidate, sorted by score.
    """
    n = len(candidates.candidates)
    cols = min(4, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(4 * cols, 4 * rows),
        squeeze=False,
    )
    fig.patch.set_facecolor("#1a1a2e")

    # Sort by score
    sorted_idx = np.argsort(scores)[::-1]

    for ax_idx, cand_idx in enumerate(sorted_idx):
        row = ax_idx // cols
        col = ax_idx % cols
        ax = axes[row, col]

        cand = candidates.candidates[cand_idx]
        score = scores[cand_idx]

        # Setup subplot
        ax.set_facecolor("#1a1a2e")
        ax.set_aspect("equal")
        ax.set_xlim(-20, 60)
        ax.set_ylim(-30, 30)
        ax.grid(True, color="#333", alpha=0.3)

        # Render lane
        if lane is not None:
            renderer = BEVRenderer()
            renderer.ax = ax
            renderer.render_lane(lane, ego_state, color="#444")

        # Render trajectory
        traj = cand.trajectory - ego_state.position
        ax.plot(
            traj[:, 0],
            traj[:, 1],
            color=CANDIDATE_CMAP(score / scores.max()),
            linewidth=2,
        )

        # Ego marker
        ax.scatter([0], [0], c=COLOR_EGO, s=100, marker="s")

        # Title
        rank = ax_idx + 1
        ax.set_title(
            f"#{cand_idx} (rank {rank})\nScore: {score:.2f}", color="white", fontsize=10
        )

    # Hide unused subplots
    for ax_idx in range(n, rows * cols):
        row = ax_idx // cols
        col = ax_idx % cols
        axes[row, col].axis("off")

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=100, facecolor="#1a1a2e")
        plt.close(fig)
        print(f"Saved comparison: {output_path}")

    return fig


def main():
    parser = argparse.ArgumentParser(description="RECTOR Visualization Utilities")

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="output/visualizations",
        help="Output directory",
    )
    parser.add_argument(
        "--demo", action="store_true", help="Generate demo visualization"
    )

    args = parser.parse_args()

    if args.demo:
        # Generate demo visualization with real Waymo data
        from planning_loop import CandidateGenerator, RECTORPlanner, PlanningConfig
        from real_data_loader import get_test_scenario

        # Load real scenario from TFRecords
        scenario = get_test_scenario()
        ego_state = scenario.ego_state
        other_agents = scenario.other_agents
        lane = scenario.lane

        # Run planner
        config = PlanningConfig(num_candidates=8, device="cpu")
        planner = RECTORPlanner(config=config, adapter=None)
        result = planner.plan_tick(
            ego_state=ego_state,
            agent_states=other_agents,
            current_lane=lane,
        )

        # Render
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        renderer = BEVRenderer()
        fig = renderer.render_planning_result(
            result, ego_state, other_agents, lane, title="RECTOR Planning Demo"
        )

        output_path = output_dir / "demo_frame.png"
        renderer.save_frame(str(output_path))
        print(f"Saved demo: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
