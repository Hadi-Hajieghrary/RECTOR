#!/usr/bin/env python3
"""
Waymo TF Example Visualization and Movie Creator

This script creates animated visualizations of Waymo tf_example format data, including:
- Agent trajectories (history + future)
- Map features (lanes, road edges)
- Agent states over time

The tf_example format contains flattened features optimized for training.

Usage:
    python scripts/visualize_waymo_tfexample.py --data_dir /path/to/waymo/tf_example \\
                                                 --num_scenarios 10 \\
                                                 --output_dir movies/
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from tqdm import tqdm

# Try importing TensorFlow for Waymo data loading
try:
    import tensorflow as tf
    HAS_WAYMO = True
except ImportError:
    print("Error: TensorFlow not found. Install with:")
    print("pip install tensorflow")
    HAS_WAYMO = False
    sys.exit(1)


# Color scheme
COLORS = {
    'history': '#3498db',  # Blue
    'future': '#2ecc71',  # Green
    'vehicle': '#e74c3c',  # Red
    'pedestrian': '#f39c12',  # Orange
    'cyclist': '#9b59b6',  # Purple
    'roadmap': '#95a5a6',  # Gray
}

# Constants from Waymo dataset
PAST_TIMESTEPS = 11  # 1 second at 10Hz
FUTURE_TIMESTEPS = 80  # 8 seconds at 10Hz


class WaymoTFExampleVisualizer:
    """Visualizer for Waymo tf_example format data."""
    
    def __init__(self, figsize=(12, 10), dpi=100):
        """
        Initialize visualizer.
        
        Args:
            figsize: Figure size in inches
            dpi: Dots per inch for rendering
        """
        self.figsize = figsize
        self.dpi = dpi
        
    def parse_tfexample(self, example: tf.train.Example) -> Dict:
        """
        Parse a tf.train.Example into a structured dictionary.
        
        Args:
            example: TensorFlow Example proto
            
        Returns:
            Dictionary with parsed features
        """
        features = example.features.feature
        
        # Determine number of agents (usually 128)
        num_agents = len(self._get_float_list(features, 'state/current/x'))
        
        # Extract state features - data is flattened as [agent0_t0, agent0_t1, ..., agent1_t0, agent1_t1, ...]
        # Need to reshape to [num_agents, num_timesteps]
        state_past = self._parse_and_reshape_feature(
            features, num_agents, 10,  # 10 past timesteps
            'state/past/x', 'state/past/y', 'state/past/valid',
            'state/past/bbox_yaw', 'state/past/length', 'state/past/width'
        )
        state_current = self._parse_and_reshape_feature(
            features, num_agents, 1,  # 1 current timestep
            'state/current/x', 'state/current/y', 'state/current/valid',
            'state/current/bbox_yaw', 'state/current/length', 'state/current/width'
        )
        state_future = self._parse_and_reshape_feature(
            features, num_agents, 80,  # 80 future timesteps
            'state/future/x', 'state/future/y', 'state/future/valid',
            'state/future/bbox_yaw', 'state/future/length', 'state/future/width'
        )
        
        # Extract roadgraph features
        roadgraph_xyz = self._get_float_list(features, 'roadgraph_samples/xyz')
        roadgraph_type = self._get_int_list(features, 'roadgraph_samples/type')
        roadgraph_valid = self._get_int_list(features, 'roadgraph_samples/valid')
        
        # Extract agent types and validity
        agent_type = self._get_float_list(features, 'state/type')
        is_sdc = self._get_int_list(features, 'state/is_sdc')
        tracks_to_predict = self._get_int_list(features, 'state/tracks_to_predict')
        
        return {
            'state_past': state_past,
            'state_current': state_current,
            'state_future': state_future,
            'roadgraph_xyz': roadgraph_xyz,
            'roadgraph_type': roadgraph_type,
            'roadgraph_valid': roadgraph_valid,
            'agent_type': agent_type,
            'is_sdc': is_sdc,
            'tracks_to_predict': tracks_to_predict,
            'num_agents': num_agents,
        }
    
    def _get_float_list(self, features, key: str) -> np.ndarray:
        """Get float list from features."""
        if key in features:
            return np.array(features[key].float_list.value)
        return np.array([])
    
    def _get_int_list(self, features, key: str) -> np.ndarray:
        """Get int64 list from features."""
        if key in features:
            return np.array(features[key].int64_list.value)
        return np.array([])
    
    def _parse_and_reshape_feature(self, features, num_agents: int,
                                   num_timesteps: int, x_key: str,
                                   y_key: str, valid_key: str,
                                   heading_key: str, length_key: str,
                                   width_key: str) -> Dict:
        """
        Parse and reshape features from flattened to [num_agents, num_timesteps].
        
        TF Example format stores data flattened:
        [agent0_t0, agent0_t1, ..., agent1_t0, agent1_t1, ...]
        """
        x = self._get_float_list(features, x_key)
        y = self._get_float_list(features, y_key)
        valid = self._get_int_list(features, valid_key)
        heading = self._get_float_list(features, heading_key)
        length = self._get_float_list(features, length_key)
        width = self._get_float_list(features, width_key)
        
        # Reshape from flat to [num_agents, num_timesteps]
        expected_size = num_agents * num_timesteps
        
        if len(x) == expected_size:
            x = x.reshape(num_agents, num_timesteps)
            y = y.reshape(num_agents, num_timesteps)
            valid = valid.reshape(num_agents, num_timesteps).astype(bool)
            heading = heading.reshape(num_agents, num_timesteps)
            length = length.reshape(num_agents, num_timesteps)
            width = width.reshape(num_agents, num_timesteps)
        else:
            # If sizes don't match, return empty arrays
            x = np.zeros((num_agents, num_timesteps))
            y = np.zeros((num_agents, num_timesteps))
            valid = np.zeros((num_agents, num_timesteps), dtype=bool)
            heading = np.zeros((num_agents, num_timesteps))
            length = np.zeros((num_agents, num_timesteps))
            width = np.zeros((num_agents, num_timesteps))
        
        return {
            'x': x,
            'y': y,
            'valid': valid,
            'heading': heading,
            'length': length,
            'width': width,
        }
    
    def plot_roadgraph(self, ax: plt.Axes, roadgraph_xyz: np.ndarray, 
                      roadgraph_type: np.ndarray, roadgraph_valid: np.ndarray):
        """
        Plot road graph features.
        
        Args:
            ax: Matplotlib axes
            roadgraph_xyz: Road graph points [N, 3]
            roadgraph_type: Road graph types [N]
            roadgraph_valid: Road graph validity [N]
        """
        if len(roadgraph_xyz) == 0:
            return
            
        # Reshape to [N, 3]
        roadgraph_xyz = roadgraph_xyz.reshape(-1, 3)
        roadgraph_valid = roadgraph_valid.reshape(-1)
        roadgraph_type = roadgraph_type.reshape(-1)
        
        valid_points = roadgraph_valid > 0
        if not np.any(valid_points):
            return
        
        xyz = roadgraph_xyz[valid_points]
        types = roadgraph_type[valid_points]
        
        # Plot road features
        ax.scatter(xyz[:, 0], xyz[:, 1], c=COLORS['roadmap'], 
                  s=1, alpha=0.3, label='Road Graph')
    
    def plot_agent_trajectory(self, ax: plt.Axes, x: np.ndarray, y: np.ndarray,
                             valid: np.ndarray, color: str, label: str, 
                             alpha: float = 1.0, linewidth: float = 2):
        """
        Plot agent trajectory.
        
        Args:
            ax: Matplotlib axes
            x: X coordinates
            y: Y coordinates
            valid: Valid mask
            color: Line color
            label: Legend label
            alpha: Transparency
            linewidth: Line width
        """
        if len(x) == 0 or not np.any(valid):
            return
            
        # Plot valid points
        valid_indices = np.where(valid)[0]
        if len(valid_indices) == 0:
            return
        
        ax.plot(x[valid], y[valid], color=color, linewidth=linewidth, 
               alpha=alpha, label=label)
        ax.scatter(x[valid], y[valid], color=color, s=20, alpha=alpha, zorder=5)
    
    def plot_agent_box(self, ax: plt.Axes, x: float, y: float, heading: float,
                      length: float = 4.0, width: float = 2.0, color: str = 'red',
                      alpha: float = 0.8):
        """
        Plot agent bounding box.
        
        Args:
            ax: Matplotlib axes
            x, y: Agent center position
            heading: Agent heading in radians
            length, width: Box dimensions
            color: Box color
            alpha: Transparency
        """
        # Create rectangle at origin
        rect = patches.Rectangle((-length/2, -width/2), length, width,
                                linewidth=2, edgecolor=color,
                                facecolor=color, alpha=alpha)
        
        # Create transformation: rotate then translate
        t = matplotlib.transforms.Affine2D().rotate(heading).translate(x, y) + ax.transData
        rect.set_transform(t)
        ax.add_patch(rect)
    
    def create_static_plot(self, data: Dict, title: str = "Waymo TF Example") -> plt.Figure:
        """
        Create a static plot of the scenario.
        
        Args:
            data: Parsed tf_example data
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Plot roadgraph
        self.plot_roadgraph(ax, data['roadgraph_xyz'],
                          data['roadgraph_type'], data['roadgraph_valid'])
        
        # Determine number of agents
        num_agents = data['num_agents']
        tracks_to_predict = data.get('tracks_to_predict', np.zeros(num_agents))
        
        # Plot agent trajectories - data is now [num_agents, num_timesteps]
        for agent_idx in range(min(num_agents, 32)):  # Limit display
            # Skip invalid agents
            if not np.any(data['state_current']['valid'][agent_idx]):
                continue
            
            # Get agent data (already reshaped)
            past_x = data['state_past']['x'][agent_idx]
            past_y = data['state_past']['y'][agent_idx]
            past_valid = data['state_past']['valid'][agent_idx]
            
            future_x = data['state_future']['x'][agent_idx]
            future_y = data['state_future']['y'][agent_idx]
            future_valid = data['state_future']['valid'][agent_idx]
            
            # Plot trajectories (only if agent has valid data)
            if np.any(past_valid):
                self.plot_agent_trajectory(ax, past_x, past_y, past_valid,
                                          COLORS['history'],
                                          f'Agent {agent_idx}',
                                          alpha=0.6, linewidth=1.5)
            
            if np.any(future_valid):
                self.plot_agent_trajectory(ax, future_x, future_y,
                                          future_valid,
                                          COLORS['future'],
                                          f'Agent {agent_idx}',
                                          alpha=0.6, linewidth=1.5)
        
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        return fig
    
    def create_animation(self, data: Dict, title: str = "Waymo TF Example",
                        fps: int = 10) -> Tuple[plt.Figure, FuncAnimation]:
        """
        Create an animation of the scenario.
        
        Args:
            data: Parsed tf_example data
            title: Plot title
            fps: Frames per second
            
        Returns:
            Tuple of (figure, animation)
        """
        # Calculate total frames (past + future)
        total_frames = PAST_TIMESTEPS + FUTURE_TIMESTEPS
        
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Determine number of agents
        num_agents = len(data['agent_type'])
        max_agents = min(num_agents, 16)  # Limit for performance
        
        # Pre-compute plot limits from roadgraph
        if len(data['roadgraph_xyz']) > 0:
            roadgraph_xyz = data['roadgraph_xyz'].reshape(-1, 3)
            x_min, x_max = roadgraph_xyz[:, 0].min(), roadgraph_xyz[:, 0].max()
            y_min, y_max = roadgraph_xyz[:, 1].min(), roadgraph_xyz[:, 1].max()
            margin = 20
            xlim = (x_min - margin, x_max + margin)
            ylim = (y_min - margin, y_max + margin)
        else:
            xlim = (-100, 100)
            ylim = (-100, 100)
        
        def init():
            """Initialize animation."""
            ax.clear()
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            return []
        
        def animate(frame):
            """Update animation frame."""
            ax.clear()
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            
            # Plot roadgraph (static)
            self.plot_roadgraph(ax, data['roadgraph_xyz'],
                              data['roadgraph_type'], data['roadgraph_valid'])
            
            # Calculate which timestep we're at
            if frame < PAST_TIMESTEPS:
                # Showing past data
                current_step = frame
                data_source = 'past'
                time_label = f"Past: t={frame - PAST_TIMESTEPS + 1}"
            else:
                # Showing future data
                current_step = frame - PAST_TIMESTEPS
                data_source = 'future'
                time_label = f"Future: t={current_step}"
            
            # Plot agents at current timestep - data is [num_agents, num_timesteps]
            for agent_idx in range(max_agents):
                if data_source == 'past':
                    x_all = data['state_past']['x'][agent_idx]
                    y_all = data['state_past']['y'][agent_idx]
                    valid_all = data['state_past']['valid'][agent_idx]
                    heading_all = data['state_past']['heading'][agent_idx]
                    length_all = data['state_past']['length'][agent_idx]
                    width_all = data['state_past']['width'][agent_idx]
                else:
                    x_all = data['state_future']['x'][agent_idx]
                    y_all = data['state_future']['y'][agent_idx]
                    valid_all = data['state_future']['valid'][agent_idx]
                    heading_all = data['state_future']['heading'][agent_idx]
                    length_all = data['state_future']['length'][agent_idx]
                    width_all = data['state_future']['width'][agent_idx]
                
                if current_step < len(x_all) and valid_all[current_step]:
                    x = x_all[current_step]
                    y = y_all[current_step]
                    heading = heading_all[current_step]
                    length = length_all[current_step]
                    width = width_all[current_step]
                    
                    # Plot trajectory history up to current point
                    if current_step > 0:
                        hist_x = x_all[:current_step+1]
                        hist_y = y_all[:current_step+1]
                        hist_valid = valid_all[:current_step+1]
                        ax.plot(hist_x[hist_valid], hist_y[hist_valid],
                               color=COLORS['vehicle'], alpha=0.4,
                               linewidth=1)
                    
                    # Plot current position with actual dimensions
                    self.plot_agent_box(ax, x, y, heading,
                                      length=max(length, 1.0),
                                      width=max(width, 0.5),
                                      color=COLORS['vehicle'], alpha=0.8)
            
            ax.set_xlabel('X (m)', fontsize=12)
            ax.set_ylabel('Y (m)', fontsize=12)
            ax.set_title(f'{title}\n{time_label}', fontsize=14,
                        fontweight='bold')
            
            return []
        
        anim = FuncAnimation(fig, animate, init_func=init,
                           frames=total_frames, interval=1000/fps,
                           blit=True, repeat=True)
        
        return fig, anim
    
    def save_animation(self, anim: FuncAnimation, output_path: Path,
                      fps: int = 10, writer: str = 'ffmpeg'):
        """
        Save animation to file.
        
        Args:
            anim: Matplotlib animation
            output_path: Output file path
            fps: Frames per second
            writer: Writer to use ('ffmpeg' or 'pillow')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if writer == 'ffmpeg':
            writer_obj = FFMpegWriter(fps=fps, bitrate=1800)
        else:
            writer_obj = PillowWriter(fps=fps)
        
        anim.save(str(output_path), writer=writer_obj)
        print(f"Animation saved: {output_path}")


def load_tfexamples(data_dir: Path, num_scenarios: int = 10) -> List[Tuple[str, tf.train.Example]]:
    """
    Load tf_examples from tfrecord files.
    
    Args:
        data_dir: Directory containing tfrecord files
        num_scenarios: Maximum number of scenarios to load
        
    Returns:
        List of (file_id, example) tuples
    """
    tfrecord_files = sorted(data_dir.glob("*.tfrecord*"))
    
    if not tfrecord_files:
        print(f"No tfrecord files found in {data_dir}")
        return []
    
    print(f"Found {len(tfrecord_files)} tfrecord files")
    
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
            
            # Use file name + record index as ID
            file_id = f"{tfrecord_file.stem}_{i:04d}"
            examples.append((file_id, example))
    
    return examples


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Visualize Waymo tf_example format data and create movies"
    )
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Directory containing Waymo tfrecord files")
    parser.add_argument("--num_scenarios", type=int, default=10,
                       help="Number of scenarios to process")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for movies")
    parser.add_argument("--fps", type=int, default=10,
                       help="Frames per second for animation")
    parser.add_argument("--dpi", type=int, default=100,
                       help="DPI for rendering")
    parser.add_argument("--format", type=str, default="mp4",
                       choices=["mp4", "gif"],
                       help="Output format")
    parser.add_argument("--static", action="store_true",
                       help="Create static plots instead of animations")
    
    args = parser.parse_args()
    
    # Setup paths
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not data_dir.exists():
        print(f"Error: Data directory {data_dir} does not exist")
        return 1
    
    # Load examples
    examples = load_tfexamples(data_dir, args.num_scenarios)
    
    if not examples:
        print("No examples loaded. Exiting.")
        return 1
    
    print(f"Loaded {len(examples)} examples")
    
    # Create visualizer
    visualizer = WaymoTFExampleVisualizer(dpi=args.dpi)
    
    # Process each example
    processed = 0
    for file_id, example in tqdm(examples, desc="Processing scenarios"):
        try:
            # Parse example
            data = visualizer.parse_tfexample(example)
            
            # Create output filename
            output_file = output_dir / f"{file_id}.{args.format}"
            
            if args.static:
                # Create static plot
                fig = visualizer.create_static_plot(data, title=f"TF Example: {file_id}")
                fig.savefig(output_file, dpi=args.dpi, bbox_inches='tight')
                plt.close(fig)
                print(f"Saved: {output_file}")
            else:
                # Create animation
                fig, anim = visualizer.create_animation(data, title=f"TF Example: {file_id}",
                                                       fps=args.fps)
                
                writer = 'ffmpeg' if args.format == 'mp4' else 'pillow'
                visualizer.save_animation(anim, output_file, fps=args.fps, writer=writer)
                plt.close(fig)
            
            processed += 1
            
        except Exception as e:
            print(f"Error processing {file_id}: {e}")
            continue
    
    print(f"\nProcessed {processed} scenario(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
