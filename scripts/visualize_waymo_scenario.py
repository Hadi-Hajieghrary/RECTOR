#!/usr/bin/env python3
"""
Waymo Scenario Visualization and Movie Creator

This script creates animated visualizations of Waymo scenarios, including:
- Agent trajectories (history + future)
- Map features (lanes, crosswalks, road edges)
- Interactive agent pairs with predictions
- Comparison with ground truth

Usage:
    python scripts/visualize_waymo_scenario.py --data_dir /path/to/waymo/data \\
                                       --scenario_id SCENARIO_ID \\
                                       --output movie.mp4
    
    # Or process multiple scenarios
    python scripts/visualize_waymo_scenario.py --data_dir /path/to/waymo/data \\
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
    from waymo_open_dataset.protos import scenario_pb2
    HAS_WAYMO = True
except ImportError:
    print("Warning: waymo_open_dataset not found. Install with:")
    print("pip install waymo-open-dataset-tf-2-11-0")
    HAS_WAYMO = False


# Color scheme
COLORS = {
    'history': '#3498db',  # Blue
    'future_gt': '#2ecc71',  # Green
    'prediction': '#e74c3c',  # Red
    'influencer': '#9b59b6',  # Purple
    'reactor': '#f39c12',  # Orange
    'other_agents': '#95a5a6',  # Gray
    'lane': '#34495e',  # Dark gray
    'road_edge': '#7f8c8d',  # Medium gray
    'crosswalk': '#ecf0f1',  # Light gray
}


class WaymoScenarioVisualizer:
    """Visualize Waymo scenarios with animation support."""
    
    def __init__(
        self,
        figsize: Tuple[int, int] = (12, 12),
        dpi: int = 100,
        fps: int = 10,
    ):
        """
        Args:
            figsize: Figure size in inches
            dpi: Dots per inch for output
            fps: Frames per second for animation
        """
        self.figsize = figsize
        self.dpi = dpi
        self.fps = fps
        
    def load_scenario_from_tfrecord(self, tfrecord_path: str, scenario_id: Optional[str] = None) -> Dict:
        """
        Load a scenario from Waymo TFRecord file.
        
        Args:
            tfrecord_path: Path to TFRecord file
            scenario_id: Specific scenario ID to load (None = first scenario)
            
        Returns:
            Dictionary with scenario data
        """
        if not HAS_WAYMO:
            raise RuntimeError("Waymo Open Dataset not installed")
            
        dataset = tf.data.TFRecordDataset(tfrecord_path)
        
        for data in dataset:
            scenario = scenario_pb2.Scenario()
            scenario.ParseFromString(data.numpy())
            
            # If specific ID requested, skip until found
            if scenario_id and scenario.scenario_id != scenario_id:
                continue
                
            return self._process_scenario(scenario)
        
        raise ValueError(f"Scenario {scenario_id} not found in {tfrecord_path}")
    
    def _process_scenario(self, scenario) -> Dict:
        """Convert Waymo protobuf to visualization format."""
        
        # Extract tracks
        tracks = []
        for track in scenario.tracks:
            track_data = {
                'id': track.id,
                'type': track.object_type,
                'states': [],
            }
            
            for state in track.states:
                if state.valid:
                    track_data['states'].append({
                        'x': state.center_x,
                        'y': state.center_y,
                        'heading': state.heading,
                        'vx': state.velocity_x,
                        'vy': state.velocity_y,
                        'width': state.width,
                        'length': state.length,
                        'timestep': state.center_x,  # Will be fixed by index
                    })
            
            tracks.append(track_data)
        
        # Extract map features
        map_features = {
            'lanes': [],
            'road_edges': [],
            'crosswalks': [],
        }
        
        for feature in scenario.map_features:
            if feature.HasField('lane'):
                polyline = [(p.x, p.y) for p in feature.lane.polyline]
                map_features['lanes'].append(polyline)
                
            elif feature.HasField('road_edge'):
                polyline = [(p.x, p.y) for p in feature.road_edge.polyline]
                map_features['road_edges'].append(polyline)
                
            elif feature.HasField('crosswalk'):
                polygon = [(p.x, p.y) for p in feature.crosswalk.polygon]
                map_features['crosswalks'].append(polygon)
        
        return {
            'scenario_id': scenario.scenario_id,
            'tracks': tracks,
            'map': map_features,
            'timestamps_seconds': scenario.timestamps_seconds,
            'sdc_track_index': scenario.sdc_track_index,
            'objects_of_interest': scenario.tracks_to_predict,
        }
    
    def plot_map(self, ax: plt.Axes, map_features: Dict):
        """Plot map features."""
        
        # Plot lanes
        for lane in map_features['lanes']:
            if len(lane) > 1:
                lane_array = np.array(lane)
                ax.plot(lane_array[:, 0], lane_array[:, 1], 
                       color=COLORS['lane'], linewidth=0.5, alpha=0.5, zorder=1)
        
        # Plot road edges
        for edge in map_features['road_edges']:
            if len(edge) > 1:
                edge_array = np.array(edge)
                ax.plot(edge_array[:, 0], edge_array[:, 1],
                       color=COLORS['road_edge'], linewidth=1.0, alpha=0.6, zorder=1)
        
        # Plot crosswalks
        for crosswalk in map_features['crosswalks']:
            if len(crosswalk) > 2:
                polygon = patches.Polygon(crosswalk, closed=True,
                                        facecolor=COLORS['crosswalk'],
                                        edgecolor='none', alpha=0.5, zorder=1)
                ax.add_patch(polygon)
    
    def plot_agent_box(self, ax: plt.Axes, x: float, y: float, heading: float,
                      length: float, width: float, color: str, alpha: float = 1.0):
        """Plot agent as oriented bounding box."""
        
        # Create rectangle in local frame
        rect_local = np.array([
            [-length/2, -width/2],
            [length/2, -width/2],
            [length/2, width/2],
            [-length/2, width/2],
        ])
        
        # Rotation matrix
        cos_h = np.cos(heading)
        sin_h = np.sin(heading)
        R = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
        
        # Transform to global frame
        rect_global = rect_local @ R.T + np.array([x, y])
        
        # Plot
        polygon = patches.Polygon(rect_global, closed=True,
                                facecolor=color, edgecolor='black',
                                linewidth=0.5, alpha=alpha, zorder=10)
        ax.add_patch(polygon)
    
    def visualize_static(
        self,
        scenario: Dict,
        current_frame: Optional[int] = None,
        history_frames: int = 10,
        future_frames: int = 80,
        prediction: Optional[Dict] = None,
        output_path: Optional[str] = None,
    ):
        """
        Create static visualization of scenario at a specific frame.
        
        Args:
            scenario: Scenario data dictionary
            current_frame: Frame to visualize (None = middle frame)
            history_frames: Number of past frames to show
            future_frames: Number of future frames to show
            prediction: Optional predictions to overlay
            output_path: Path to save figure
        """
        
        total_frames = len(scenario['timestamps_seconds'])
        if current_frame is None:
            current_frame = total_frames // 2
        
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Plot map
        self.plot_map(ax, scenario['map'])
        
        # Plot all tracks
        for track in scenario['tracks']:
            if len(track['states']) == 0:
                continue
            
            # Get valid states around current frame
            history_start = max(0, current_frame - history_frames)
            history_end = current_frame
            future_start = current_frame
            future_end = min(len(track['states']), current_frame + future_frames)
            
            # History trajectory
            if history_end > history_start:
                history_states = track['states'][history_start:history_end]
                history_xy = np.array([[s['x'], s['y']] for s in history_states])
                ax.plot(history_xy[:, 0], history_xy[:, 1],
                       color=COLORS['history'], linewidth=1.5, alpha=0.6, zorder=5)
            
            # Future trajectory (ground truth)
            if future_end > future_start:
                future_states = track['states'][future_start:future_end]
                future_xy = np.array([[s['x'], s['y']] for s in future_states])
                ax.plot(future_xy[:, 0], future_xy[:, 1],
                       color=COLORS['future_gt'], linewidth=1.5, linestyle='--',
                       alpha=0.4, zorder=5)
            
            # Current position as box
            if current_frame < len(track['states']):
                state = track['states'][current_frame]
                self.plot_agent_box(
                    ax, state['x'], state['y'], state['heading'],
                    state['length'], state['width'],
                    color=COLORS['other_agents'], alpha=0.8
                )
        
        # Highlight SDC (self-driving car)
        sdc_track = scenario['tracks'][scenario['sdc_track_index']]
        if current_frame < len(sdc_track['states']):
            state = sdc_track['states'][current_frame]
            self.plot_agent_box(
                ax, state['x'], state['y'], state['heading'],
                state['length'], state['width'],
                color='#e74c3c', alpha=1.0
            )
        
        # Plot predictions if provided
        if prediction is not None:
            pred_xy = prediction['trajectory']  # (T, 2)
            ax.plot(pred_xy[:, 0], pred_xy[:, 1],
                   color=COLORS['prediction'], linewidth=2.0,
                   linestyle=':', label='Prediction', zorder=15)
        
        # Set equal aspect and limits
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X (meters)', fontsize=12)
        ax.set_ylabel('Y (meters)', fontsize=12)
        
        # Title
        time_s = scenario['timestamps_seconds'][current_frame]
        ax.set_title(f"Scenario: {scenario['scenario_id']}\nTime: {time_s:.1f}s (Frame {current_frame}/{total_frames})",
                    fontsize=14, fontweight='bold')
        
        # Legend
        ax.legend(loc='upper right', fontsize=10)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved: {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def create_animation(
        self,
        scenario: Dict,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        history_frames: int = 10,
        prediction: Optional[Dict] = None,
        output_path: str = "scenario_animation.mp4",
        writer: str = "ffmpeg",  # or "pillow" for GIF
    ):
        """
        Create animated visualization of scenario.
        
        Args:
            scenario: Scenario data dictionary
            start_frame: Starting frame
            end_frame: Ending frame (None = all frames)
            history_frames: Number of past frames to show in trails
            prediction: Optional predictions to overlay
            output_path: Output file path (.mp4 or .gif)
            writer: 'ffmpeg' for MP4 or 'pillow' for GIF
        """
        
        total_frames = len(scenario['timestamps_seconds'])
        if end_frame is None:
            end_frame = total_frames
        
        frames_to_animate = range(start_frame, end_frame)
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Plot static map
        self.plot_map(ax, scenario['map'])
        
        # Initialize plot elements
        trajectory_lines = {}
        agent_patches = {}
        
        for track in scenario['tracks']:
            track_id = track['id']
            # History line
            line, = ax.plot([], [], color=COLORS['history'], linewidth=1.5, alpha=0.6, zorder=5)
            trajectory_lines[track_id] = line
            
        # Title text
        title_text = ax.text(0.5, 1.05, '', transform=ax.transAxes,
                           fontsize=14, fontweight='bold', ha='center')
        
        # Set initial view limits (will be updated)
        all_x, all_y = [], []
        for track in scenario['tracks']:
            for state in track['states']:
                all_x.append(state['x'])
                all_y.append(state['y'])
        
        if all_x and all_y:
            margin = 20  # meters
            ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
            ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
        
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X (meters)', fontsize=12)
        ax.set_ylabel('Y (meters)', fontsize=12)
        
        def update(frame_idx):
            """Update function for animation."""
            frame = frames_to_animate[frame_idx]
            
            # Clear previous agent patches
            for patch_list in agent_patches.values():
                for patch in patch_list:
                    patch.remove()
            agent_patches.clear()
            
            # Update trajectories and agents
            for track in scenario['tracks']:
                track_id = track['id']
                
                if frame >= len(track['states']):
                    continue
                
                # History trail
                history_start = max(0, frame - history_frames)
                history_states = track['states'][history_start:frame+1]
                if history_states:
                    history_xy = np.array([[s['x'], s['y']] for s in history_states])
                    trajectory_lines[track_id].set_data(history_xy[:, 0], history_xy[:, 1])
                
                # Current agent box
                state = track['states'][frame]
                color = COLORS['other_agents']
                if track_id == scenario['tracks'][scenario['sdc_track_index']]['id']:
                    color = '#e74c3c'  # Highlight SDC
                
                # Create box polygon
                rect_local = np.array([
                    [-state['length']/2, -state['width']/2],
                    [state['length']/2, -state['width']/2],
                    [state['length']/2, state['width']/2],
                    [-state['length']/2, state['width']/2],
                ])
                
                cos_h = np.cos(state['heading'])
                sin_h = np.sin(state['heading'])
                R = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
                rect_global = rect_local @ R.T + np.array([state['x'], state['y']])
                
                polygon = patches.Polygon(rect_global, closed=True,
                                        facecolor=color, edgecolor='black',
                                        linewidth=0.5, alpha=0.8, zorder=10)
                ax.add_patch(polygon)
                
                if track_id not in agent_patches:
                    agent_patches[track_id] = []
                agent_patches[track_id].append(polygon)
            
            # Update title
            time_s = scenario['timestamps_seconds'][frame]
            title_text.set_text(f"Scenario: {scenario['scenario_id']}\n"
                              f"Time: {time_s:.1f}s (Frame {frame}/{total_frames})")
            
            return list(trajectory_lines.values()) + [title_text]
        
        # Create animation
        print(f"Creating animation with {len(frames_to_animate)} frames...")
        anim = FuncAnimation(
            fig, update,
            frames=len(frames_to_animate),
            interval=1000/self.fps,
            blit=False,
        )
        
        # Save animation as MP4 and GIF
        output_path = Path(output_path)
        
        # Save MP4 version
        if writer.lower() == "ffmpeg":
            writer_obj = FFMpegWriter(fps=self.fps, bitrate=5000)
            anim.save(output_path, writer=writer_obj)
            print(f"MP4 saved: {output_path}")
        elif writer.lower() == "pillow":
            writer_obj = PillowWriter(fps=self.fps)
            anim.save(output_path, writer=writer_obj)
            print(f"Animation saved: {output_path}")
        else:
            raise ValueError(f"Unknown writer: {writer}")
        
        # Also save GIF version for GitHub display (reduced fps for smaller size)
        if writer.lower() == "ffmpeg" and output_path.suffix.lower() == ".mp4":
            gif_path = output_path.with_suffix('.gif')
            print(f"Creating GIF (5 fps): {gif_path}")
            pillow_writer = PillowWriter(fps=5)  # Reduced fps for smaller GIF
            anim.save(gif_path, writer=pillow_writer)
            print(f"GIF saved: {gif_path}")
        
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize Waymo scenarios")
    parser.add_argument("--data_dir", required=True, help="Path to Waymo tfrecord files")
    parser.add_argument("--scenario_id", help="Specific scenario ID to visualize")
    parser.add_argument("--num_scenarios", type=int, default=1, help="Number of scenarios to process")
    parser.add_argument("--output", default="scenario.mp4", help="Output file (mp4 or gif)")
    parser.add_argument("--output_dir", help="Output directory for multiple scenarios")
    parser.add_argument("--figsize", type=int, nargs=2, default=[12, 12], help="Figure size")
    parser.add_argument("--dpi", type=int, default=100, help="DPI for output")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second")
    parser.add_argument("--static", action="store_true", help="Create static image instead of animation")
    parser.add_argument("--frame", type=int, help="Frame to visualize (for static mode)")
    
    args = parser.parse_args()
    
    # Check if Waymo dataset is available
    if not HAS_WAYMO:
        print("ERROR: waymo_open_dataset package not found!")
        print("Install with: pip install waymo-open-dataset-tf-2-11-0")
        return 1
    
    # Find tfrecord files
    data_dir = Path(args.data_dir)
    tfrecord_files = list(data_dir.glob("*.tfrecord*"))
    
    if not tfrecord_files:
        print(f"No tfrecord files found in {data_dir}")
        return 1
    
    print(f"Found {len(tfrecord_files)} tfrecord files")
    
    # Create visualizer
    visualizer = WaymoScenarioVisualizer(
        figsize=tuple(args.figsize),
        dpi=args.dpi,
        fps=args.fps,
    )
    
    # Process scenarios
    scenarios_processed = 0
    
    for tfrecord_path in tqdm(tfrecord_files[:args.num_scenarios], desc="Processing scenarios"):
        try:
            # Load scenario
            scenario = visualizer.load_scenario_from_tfrecord(
                str(tfrecord_path),
                scenario_id=args.scenario_id
            )
            
            # Determine output path
            if args.output_dir:
                output_dir = Path(args.output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                ext = '.png' if args.static else ('.gif' if args.output.endswith('.gif') else '.mp4')
                output_path = output_dir / f"{scenario['scenario_id']}{ext}"
            else:
                output_path = args.output
            
            # Create visualization
            if args.static:
                visualizer.visualize_static(
                    scenario,
                    current_frame=args.frame,
                    output_path=str(output_path),
                )
            else:
                writer = "pillow" if str(output_path).endswith('.gif') else "ffmpeg"
                visualizer.create_animation(
                    scenario,
                    output_path=str(output_path),
                    writer=writer,
                )
            
            scenarios_processed += 1
            
            # If specific scenario requested, stop after finding it
            if args.scenario_id:
                break
                
        except Exception as e:
            print(f"Error processing {tfrecord_path}: {e}")
            continue
    
    print(f"\nProcessed {scenarios_processed} scenario(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())