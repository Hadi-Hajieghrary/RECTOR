#!/usr/bin/env python3
"""
Create Static Visualizations of Waymo Scenarios

Generates static plots showing trajectories, map features, and multi-frame sequences.
These can be used as previews before generating full movies.

This script works with both:
- Raw Scenario format (raw/scenario/)
- Processed TF format (processed/tf/)

Usage:
    # Single file
    python visualize_scenario.py --tfrecord <path> --scenario-index 0
    
    # Process split from scenario format
    python visualize_scenario.py --format scenario --split validation_interactive --num 5
    
    # Process split from tf format
    python visualize_scenario.py --format tf --split training_interactive --num 5
    
    # Process all splits from both formats
    python visualize_scenario.py --all --num 5
    
    # With multi-frame visualization
    python visualize_scenario.py --format scenario --split validation_interactive --num 5 --multi-frame
"""

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Polygon, Rectangle

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from waymo_open_dataset.protos import scenario_pb2
    HAS_WAYMO = True
except ImportError:
    print("Warning: waymo_open_dataset not found. Scenario format not available.")
    HAS_WAYMO = False


# Base data path
DATA_BASE = Path("/workspace/data/WOMD/datasets/waymo_open_dataset/motion_v_1_3_0")
VIZ_BASE = Path("/workspace/data/WOMD/visualizations")

# Available splits
SPLITS = [
    'training_interactive',
    'validation_interactive',
    'testing_interactive',
]


class ScenarioVisualizer:
    """Create static visualizations of Waymo scenarios."""
    
    # Colors matching generate_bev_movie.py
    COLORS = {
        'ego': '#FF0000',           # Red for ego vehicle
        'vehicle': '#3498db',       # Blue for other vehicles
        'pedestrian': '#2ecc71',    # Green for pedestrians
        'cyclist': '#f39c12',       # Orange for cyclists
        'other': '#95a5a6',         # Gray for others
        'lane': '#34495e',          # Dark gray for lanes
        'road_line': '#7f8c8d',     # Medium gray for road lines
        'road_edge': '#2c3e50',     # Very dark gray for road edges
        'crosswalk': '#16a085',     # Teal for crosswalks
        'stop_sign': '#c0392b',     # Dark red for stop signs
        'speed_bump': '#d35400',    # Dark orange for speed bumps
        'history': '#e74c3c',       # Red for history trail
    }
    
    def __init__(self, output_dir: Path, dpi: int = 150):
        """
        Initialize scenario visualizer.
        
        Args:
            output_dir: Directory for output files
            dpi: Resolution quality for saved images
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        
        print(f"Scenario Visualizer initialized:")
        print(f"  Output: {self.output_dir}")
        print(f"  DPI: {dpi}")
    
    def load_scenario(self, tfrecord_path: str, scenario_index: int = 0):
        """Load scenario from TFRecord file.
        
        Handles two storage formats:
        1. Raw Scenario proto serialized directly as TFRecord entries
        2. tf.Example wrapper with the Scenario proto in a 'scenario/proto' bytes field
           (produced by the rule evaluation pipeline)
        """
        if not HAS_WAYMO:
            raise RuntimeError("waymo_open_dataset package required for scenario format")
        
        dataset = tf.data.TFRecordDataset([tfrecord_path], compression_type='')
        
        for idx, data in enumerate(dataset):
            if idx == scenario_index:
                raw_bytes = data.numpy()
                
                # Try direct Scenario proto first
                scenario = scenario_pb2.Scenario()
                try:
                    scenario.ParseFromString(raw_bytes)
                    if scenario.scenario_id:
                        return scenario
                except Exception:
                    pass
                
                # Try tf.Example wrapper with 'scenario/proto' field
                try:
                    example = tf.train.Example()
                    example.ParseFromString(raw_bytes)
                    if 'scenario/proto' in example.features.feature:
                        proto_bytes = example.features.feature['scenario/proto'].bytes_list.value[0]
                        scenario = scenario_pb2.Scenario()
                        scenario.ParseFromString(proto_bytes)
                        return scenario
                except Exception:
                    pass
                
                raise ValueError(
                    f"Record {scenario_index} in {tfrecord_path} could not be parsed "
                    f"as a Scenario proto or as a tf.Example with 'scenario/proto' field"
                )
        
        raise ValueError(f"Scenario index {scenario_index} not found in {tfrecord_path}")
    
    def extract_data(self, scenario):
        """Extract trajectories and map features from scenario."""
        # Find ego vehicle (SDC)
        sdc_track_index = scenario.sdc_track_index
        sdc_track = scenario.tracks[sdc_track_index]
        
        # Extract ego trajectory
        ego_trajectory = []
        for state in sdc_track.states:
            if state.valid:
                ego_trajectory.append({
                    'x': state.center_x,
                    'y': state.center_y,
                    'heading': state.heading,
                    'length': state.length,
                    'width': state.width,
                    'velocity_x': state.velocity_x,
                    'velocity_y': state.velocity_y,
                })
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
                    trajectory.append({
                        'x': state.center_x,
                        'y': state.center_y,
                        'heading': state.heading,
                        'length': state.length,
                        'width': state.width,
                        'velocity_x': state.velocity_x,
                        'velocity_y': state.velocity_y,
                        'type': track.object_type,
                    })
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
            if feature.HasField('lane'):
                lane_points = [(p.x, p.y) for p in feature.lane.polyline]
                lanes.append(lane_points)
            elif feature.HasField('road_line'):
                line_points = [(p.x, p.y) for p in feature.road_line.polyline]
                road_lines.append(line_points)
            elif feature.HasField('road_edge'):
                edge_points = [(p.x, p.y) for p in feature.road_edge.polyline]
                road_edges.append(edge_points)
            elif feature.HasField('crosswalk'):
                crosswalk_points = [(p.x, p.y) for p in feature.crosswalk.polygon]
                crosswalks.append(crosswalk_points)
            elif feature.HasField('stop_sign'):
                pos = feature.stop_sign.position
                stop_signs.append((pos.x, pos.y))
            elif feature.HasField('speed_bump'):
                bump_points = [(p.x, p.y) for p in feature.speed_bump.polygon]
                speed_bumps.append(bump_points)
        
        return {
            'ego': ego_trajectory,
            'agents': agents_trajectories,
            'lanes': lanes,
            'road_lines': road_lines,
            'road_edges': road_edges,
            'crosswalks': crosswalks,
            'stop_signs': stop_signs,
            'speed_bumps': speed_bumps,
            'scenario_id': scenario.scenario_id,
            'num_frames': len(ego_trajectory),
            'timestamps': list(scenario.timestamps_seconds),
            'sdc_track_index': sdc_track_index,
        }
    
    def visualize_overview(self, scenario, output_name: str = None) -> bool:
        """
        Create overview visualization showing all trajectories.
        
        Args:
            scenario: Scenario protobuf
            output_name: Output filename (default: based on scenario ID)
        """
        data = self.extract_data(scenario)
        
        if output_name is None:
            output_name = f"overview_{data['scenario_id']}.png"
        
        output_path = self.output_dir / output_name
        
        print(f"\nCreating overview visualization for {data['scenario_id']}")
        print(f"  Tracks: {len(data['agents']) + 1}, Frames: {data['num_frames']}")
        
        fig, ax = plt.subplots(figsize=(14, 12))
        
        # Draw map features first (background)
        self._draw_map_features(ax, data)
        
        # Plot ego trajectory
        ego_positions = [(s['x'], s['y']) for s in data['ego'] if s is not None]
        if ego_positions:
            xs, ys = zip(*ego_positions)
            ax.plot(xs, ys, color=self.COLORS['ego'], linewidth=3.0, 
                   alpha=1.0, label='Ego Vehicle (SDC)')
            # Mark start and end
            ax.plot(xs[0], ys[0], 'o', color=self.COLORS['ego'], markersize=10,
                   markeredgecolor='black', markeredgewidth=2)
            ax.plot(xs[-1], ys[-1], 's', color=self.COLORS['ego'], markersize=10,
                   markeredgecolor='black', markeredgewidth=2)
        
        # Plot other agent trajectories
        labeled_types = set()
        for agent_traj in data['agents']:
            positions = [(s['x'], s['y']) for s in agent_traj if s is not None]
            if not positions:
                continue
            
            # Get agent type from first valid state
            first_valid = next((s for s in agent_traj if s is not None), None)
            if first_valid is None:
                continue
            
            type_name = self._get_object_type_name(first_valid['type'])
            color = self.COLORS.get(type_name.lower(), self.COLORS['other'])
            
            # Only add label once per type
            label = type_name.title() if type_name not in labeled_types else None
            if label:
                labeled_types.add(type_name)
            
            xs, ys = zip(*positions)
            ax.plot(xs, ys, color=color, linewidth=1.5, alpha=0.6, label=label)
            
            # Mark start and end
            ax.plot(xs[0], ys[0], 'o', color=color, markersize=6,
                   markeredgecolor='black', markeredgewidth=1, alpha=0.7)
            ax.plot(xs[-1], ys[-1], 's', color=color, markersize=6,
                   markeredgecolor='black', markeredgewidth=1, alpha=0.7)
        
        # Calculate bounds
        all_positions = ego_positions.copy()
        for agent_traj in data['agents']:
            all_positions.extend([(s['x'], s['y']) for s in agent_traj if s is not None])
        
        if all_positions:
            all_x, all_y = zip(*all_positions)
            margin = 20
            ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
            ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
        
        # Labels and title
        duration = len(data['timestamps']) * 0.1 if data['timestamps'] else data['num_frames'] * 0.1
        ax.set_xlabel('X (meters)', fontsize=12)
        ax.set_ylabel('Y (meters)', fontsize=12)
        ax.set_title(f"Scenario Overview: {data['scenario_id']}\n"
                    f"Duration: {duration:.1f}s | Tracks: {len(data['agents']) + 1}",
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
        ax.set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"  ✓ Saved: {output_path.name} ({file_size_mb:.2f} MB)")
        return True
    
    def visualize_multi_frame(self, scenario, output_name: str = None,
                             num_frames: int = 6) -> bool:
        """
        Create multi-frame visualization showing scenario evolution.
        
        Args:
            scenario: Scenario protobuf
            output_name: Output filename
            num_frames: Number of frames to show (default: 6)
        """
        data = self.extract_data(scenario)
        
        if output_name is None:
            output_name = f"multiframe_{data['scenario_id']}.png"
        
        output_path = self.output_dir / output_name
        
        print(f"\nCreating multi-frame visualization for {data['scenario_id']}")
        
        total_frames = data['num_frames']
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        # Calculate global bounds for consistent view across frames
        all_positions = []
        for s in data['ego']:
            if s is not None:
                all_positions.append((s['x'], s['y']))
        for agent_traj in data['agents']:
            for s in agent_traj:
                if s is not None:
                    all_positions.append((s['x'], s['y']))
        
        if all_positions:
            all_x, all_y = zip(*all_positions)
            margin = 15
            x_min, x_max = min(all_x) - margin, max(all_x) + margin
            y_min, y_max = min(all_y) - margin, max(all_y) + margin
        else:
            x_min, x_max, y_min, y_max = -50, 50, -50, 50
        
        rows = 2
        cols = 3
        fig, axes = plt.subplots(rows, cols, figsize=(18, 12))
        axes = axes.flatten()
        
        for plot_idx, frame_idx in enumerate(frame_indices):
            ax = axes[plot_idx]
            # Use actual timestamp if available
            if data['timestamps'] and frame_idx < len(data['timestamps']):
                time = data['timestamps'][frame_idx]
            else:
                time = frame_idx * 0.1
            
            # Set consistent view limits
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            
            # Draw map features
            self._draw_map_features(ax, data)
            
            # Draw agents at this frame
            # Ego vehicle
            ego_state = data['ego'][frame_idx] if frame_idx < len(data['ego']) else None
            if ego_state is not None:
                rect = self._create_vehicle_patch(
                    ego_state['x'], ego_state['y'], ego_state['heading'],
                    ego_state['length'], ego_state['width'], self.COLORS['ego'], ax
                )
                ax.add_patch(rect)
                
                # Draw velocity arrow
                speed = np.sqrt(ego_state['velocity_x']**2 + ego_state['velocity_y']**2)
                if speed > 0.5:
                    ax.arrow(ego_state['x'], ego_state['y'],
                            ego_state['velocity_x'] * 0.5, ego_state['velocity_y'] * 0.5,
                            head_width=0.8, head_length=0.5,
                            fc=self.COLORS['ego'], ec='black', alpha=0.6, linewidth=0.5)
            
            # Other agents
            for agent_traj in data['agents']:
                if frame_idx >= len(agent_traj):
                    continue
                agent_state = agent_traj[frame_idx]
                if agent_state is None:
                    continue
                
                type_name = self._get_object_type_name(agent_state['type'])
                color = self.COLORS.get(type_name.lower(), self.COLORS['other'])
                
                rect = self._create_vehicle_patch(
                    agent_state['x'], agent_state['y'], agent_state['heading'],
                    agent_state['length'], agent_state['width'], color, ax
                )
                ax.add_patch(rect)
                
                # Draw velocity arrow for moving agents
                speed = np.sqrt(agent_state['velocity_x']**2 + agent_state['velocity_y']**2)
                if speed > 0.5:
                    ax.arrow(agent_state['x'], agent_state['y'],
                            agent_state['velocity_x'] * 0.5, agent_state['velocity_y'] * 0.5,
                            head_width=0.8, head_length=0.5,
                            fc=color, ec='black', alpha=0.6, linewidth=0.5)
            
            ax.set_title(f'Frame {frame_idx} (t={time:.1f}s)', fontsize=11, fontweight='bold')
            ax.set_xlabel('X (m)', fontsize=10)
            ax.set_ylabel('Y (m)', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
        
        plt.suptitle(f'Scenario Timeline: {data["scenario_id"]}\n'
                    f'Duration: {total_frames * 0.1:.1f}s | Tracks: {len(data["agents"]) + 1}',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"  ✓ Saved: {output_path.name} ({file_size_mb:.2f} MB)")
        return True

    def visualize_combined(self, scenario, output_name: str = None) -> bool:
        """
        Create high-quality combined visualization with overview + timeline.
        
        Creates a 2-row layout:
        - Top row: Large overview with trajectories
        - Bottom row: 4 key frames showing scenario evolution
        """
        data = self.extract_data(scenario)
        
        if output_name is None:
            output_name = f"combined_{data['scenario_id']}.png"
        
        output_path = self.output_dir / output_name
        
        print(f"\nCreating combined visualization for {data['scenario_id']}")
        
        # Calculate global bounds
        all_positions = []
        for s in data['ego']:
            if s is not None:
                all_positions.append((s['x'], s['y']))
        for agent_traj in data['agents']:
            for s in agent_traj:
                if s is not None:
                    all_positions.append((s['x'], s['y']))
        
        if all_positions:
            all_x, all_y = zip(*all_positions)
            margin = 15
            x_min, x_max = min(all_x) - margin, max(all_x) + margin
            y_min, y_max = min(all_y) - margin, max(all_y) + margin
        else:
            x_min, x_max, y_min, y_max = -50, 50, -50, 50
        
        # Create figure with custom grid
        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(2, 4, height_ratios=[1.2, 1], hspace=0.25, wspace=0.2)
        
        # Top row: Large overview spanning 4 columns
        ax_overview = fig.add_subplot(gs[0, :])
        self._draw_overview_on_ax(ax_overview, data, x_min, x_max, y_min, y_max)
        
        # Bottom row: 4 key frames
        total_frames = data['num_frames']
        frame_indices = np.linspace(0, total_frames - 1, 4, dtype=int)
        
        for i, frame_idx in enumerate(frame_indices):
            ax = fig.add_subplot(gs[1, i])
            self._draw_frame_on_ax(ax, data, frame_idx, x_min, x_max, y_min, y_max)
        
        # Main title
        duration = total_frames * 0.1
        plt.suptitle(
            f'Scenario Analysis: {data["scenario_id"]}\n'
            f'Duration: {duration:.1f}s | Total Tracks: {len(data["agents"]) + 1}',
            fontsize=16, fontweight='bold', y=0.98
        )
        
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()
        
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"  ✓ Saved: {output_path.name} ({file_size_mb:.2f} MB)")
        return True

    def _draw_overview_on_ax(self, ax, data, x_min, x_max, y_min, y_max):
        """Draw overview visualization on given axis."""
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        
        # Draw map features
        self._draw_map_features(ax, data)
        
        # Draw ego trajectory with gradient
        ego_positions = [(s['x'], s['y']) for s in data['ego'] if s is not None]
        if len(ego_positions) > 1:
            xs, ys = zip(*ego_positions)
            # Draw gradient trail
            for i in range(len(xs) - 1):
                alpha = 0.3 + 0.7 * (i / len(xs))
                ax.plot(xs[i:i+2], ys[i:i+2], color=self.COLORS['ego'],
                       linewidth=3.0, alpha=alpha)
            # Mark start and end
            ax.plot(xs[0], ys[0], 'o', color=self.COLORS['ego'],
                   markersize=12, markeredgecolor='white', markeredgewidth=2,
                   label='Ego Start', zorder=10)
            ax.plot(xs[-1], ys[-1], 's', color=self.COLORS['ego'],
                   markersize=12, markeredgecolor='white', markeredgewidth=2,
                   label='Ego End', zorder=10)
        
        # Draw agent trajectories
        labeled_types = set()
        for agent_traj in data['agents']:
            positions = [(s['x'], s['y']) for s in agent_traj if s is not None]
            if len(positions) < 2:
                continue
            
            first_valid = next((s for s in agent_traj if s is not None), None)
            if first_valid is None:
                continue
            
            type_name = self._get_object_type_name(first_valid['type'])
            color = self.COLORS.get(type_name.lower(), self.COLORS['other'])
            
            label = type_name.title() if type_name not in labeled_types else None
            if label:
                labeled_types.add(type_name)
            
            xs, ys = zip(*positions)
            # Draw gradient trail
            for i in range(len(xs) - 1):
                alpha = 0.2 + 0.6 * (i / len(xs))
                ax.plot(xs[i:i+2], ys[i:i+2], color=color,
                       linewidth=2.0, alpha=alpha)
            
            # Mark start and end
            ax.plot(xs[0], ys[0], 'o', color=color, markersize=6,
                   markeredgecolor='black', markeredgewidth=0.5, alpha=0.8,
                   label=label)
            ax.plot(xs[-1], ys[-1], 's', color=color, markersize=6,
                   markeredgecolor='black', markeredgewidth=0.5, alpha=0.8)
        
        ax.set_xlabel('X (meters)', fontsize=12)
        ax.set_ylabel('Y (meters)', fontsize=12)
        ax.set_title('Trajectory Overview', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=10, framealpha=0.95,
                 fancybox=True, shadow=True)
        ax.set_aspect('equal')

    def _draw_frame_on_ax(self, ax, data, frame_idx, x_min, x_max, y_min, y_max):
        """Draw single frame visualization on given axis."""
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        
        # Get time
        if data['timestamps'] and frame_idx < len(data['timestamps']):
            time = data['timestamps'][frame_idx]
        else:
            time = frame_idx * 0.1
        
        # Draw map features
        self._draw_map_features(ax, data)
        
        # Draw history trails up to this frame
        self._draw_history_trails(ax, data, frame_idx)
        
        # Draw ego vehicle
        if frame_idx < len(data['ego']):
            ego_state = data['ego'][frame_idx]
            if ego_state is not None:
                rect = self._create_vehicle_patch(
                    ego_state['x'], ego_state['y'], ego_state['heading'],
                    ego_state['length'], ego_state['width'], self.COLORS['ego'], ax
                )
                ax.add_patch(rect)
                
                # Velocity arrow
                vx, vy = ego_state['velocity_x'], ego_state['velocity_y']
                speed = np.sqrt(vx**2 + vy**2)
                if speed > 0.5:
                    ax.arrow(ego_state['x'], ego_state['y'], vx * 0.5, vy * 0.5,
                            head_width=0.8, head_length=0.4,
                            fc=self.COLORS['ego'], ec='black',
                            alpha=0.7, linewidth=0.5, zorder=6)
        
        # Draw other agents
        for agent_traj in data['agents']:
            if frame_idx >= len(agent_traj):
                continue
            agent_state = agent_traj[frame_idx]
            if agent_state is None:
                continue
            
            type_name = self._get_object_type_name(agent_state['type'])
            color = self.COLORS.get(type_name.lower(), self.COLORS['other'])
            
            rect = self._create_vehicle_patch(
                agent_state['x'], agent_state['y'], agent_state['heading'],
                agent_state['length'], agent_state['width'], color, ax
            )
            ax.add_patch(rect)
            
            # Velocity arrow
            vx, vy = agent_state['velocity_x'], agent_state['velocity_y']
            speed = np.sqrt(vx**2 + vy**2)
            if speed > 0.5:
                ax.arrow(agent_state['x'], agent_state['y'], vx * 0.5, vy * 0.5,
                        head_width=0.6, head_length=0.3,
                        fc=color, ec='black', alpha=0.6,
                        linewidth=0.5, zorder=6)
        
        ax.set_title(f'Frame {frame_idx} (t={time:.1f}s)',
                    fontsize=11, fontweight='bold')
        ax.set_xlabel('X (m)', fontsize=9)
        ax.set_ylabel('Y (m)', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

    def _draw_history_trails(self, ax, data, current_frame):
        """Draw fading history trails up to current frame."""
        # Ego history
        trail_length = min(20, current_frame)
        if trail_length > 0:
            start_frame = max(0, current_frame - trail_length)
            ego_trail = [
                (data['ego'][i]['x'], data['ego'][i]['y'])
                for i in range(start_frame, current_frame + 1)
                if i < len(data['ego']) and data['ego'][i] is not None
            ]
            if len(ego_trail) > 1:
                xs, ys = zip(*ego_trail)
                for i in range(len(xs) - 1):
                    alpha = 0.2 + 0.6 * (i / len(xs))
                    ax.plot(xs[i:i+2], ys[i:i+2], color=self.COLORS['ego'],
                           linewidth=2.0, alpha=alpha)
        
        # Agent histories
        for agent_traj in data['agents']:
            start_frame = max(0, current_frame - trail_length)
            agent_trail = []
            for i in range(start_frame, current_frame + 1):
                if i < len(agent_traj) and agent_traj[i] is not None:
                    agent_trail.append((agent_traj[i]['x'], agent_traj[i]['y']))
            
            if len(agent_trail) > 1:
                first_state = next((s for s in agent_traj if s is not None), None)
                if first_state:
                    type_name = self._get_object_type_name(first_state['type'])
                    color = self.COLORS.get(type_name.lower(), self.COLORS['other'])
                    
                    xs, ys = zip(*agent_trail)
                    for i in range(len(xs) - 1):
                        alpha = 0.15 + 0.45 * (i / len(xs))
                        ax.plot(xs[i:i+2], ys[i:i+2], color=color,
                               linewidth=1.5, alpha=alpha)
    
    def _draw_map_features(self, ax, data: dict):
        """Draw map features on axis."""
        # Draw lanes
        for lane_points in data.get('lanes', []):
            if len(lane_points) > 1:
                xs, ys = zip(*lane_points)
                ax.plot(xs, ys, color=self.COLORS['lane'], linewidth=1.5, alpha=0.4)
        
        # Draw road lines
        for line_points in data.get('road_lines', []):
            if len(line_points) > 1:
                xs, ys = zip(*line_points)
                ax.plot(xs, ys, color=self.COLORS['road_line'], 
                       linewidth=1.0, alpha=0.5, linestyle='--')
        
        # Draw road edges
        for edge_points in data.get('road_edges', []):
            if len(edge_points) > 1:
                xs, ys = zip(*edge_points)
                ax.plot(xs, ys, color=self.COLORS['road_edge'], linewidth=2.0, alpha=0.6)
        
        # Draw crosswalks
        for crosswalk_points in data.get('crosswalks', []):
            if len(crosswalk_points) > 2:
                poly = Polygon(crosswalk_points, closed=True,
                              facecolor=self.COLORS['crosswalk'],
                              edgecolor=self.COLORS['crosswalk'],
                              alpha=0.3, linewidth=1.5)
                ax.add_patch(poly)
        
        # Draw stop signs
        for stop_pos in data.get('stop_signs', []):
            sx, sy = stop_pos
            circle = Circle((sx, sy), 1.5, color=self.COLORS['stop_sign'], 
                           alpha=0.7, zorder=5)
            ax.add_patch(circle)
            ax.text(sx, sy, 'STOP', ha='center', va='center',
                   fontsize=5, fontweight='bold', color='white', zorder=6)
        
        # Draw speed bumps
        for bump_points in data.get('speed_bumps', []):
            if len(bump_points) > 2:
                poly = Polygon(bump_points, closed=True,
                              facecolor=self.COLORS['speed_bump'],
                              edgecolor=self.COLORS['speed_bump'],
                              alpha=0.5, linewidth=1.5)
                ax.add_patch(poly)
    
    def _create_vehicle_patch(self, x, y, heading, length, width, color, ax):
        """Create oriented rectangle patch for vehicle."""
        length = max(length, 1.0)
        width = max(width, 0.5)

        rect = Rectangle((-length/2, -width/2), length, width,
                         facecolor=color, edgecolor='black',
                         linewidth=1.5, alpha=0.8)

        # Apply rotation and translation
        import matplotlib.transforms as transforms
        t = transforms.Affine2D().rotate(heading).translate(x, y) + ax.transData
        rect.set_transform(t)
        
        return rect
    
    def _get_object_type_name(self, type_id: int) -> str:
        """Convert object type ID to name."""
        type_map = {
            0: 'other',
            1: 'vehicle',
            2: 'pedestrian',
            3: 'cyclist',
            4: 'other',
        }
        return type_map.get(type_id, 'other')


def get_data_dir(format_type: str, split: str) -> Path:
    """Get data directory for given format and split."""
    if format_type == 'scenario':
        return DATA_BASE / 'raw' / 'scenario' / split
    elif format_type == 'tf':
        return DATA_BASE / 'processed' / 'tf' / split
    else:
        raise ValueError(f"Unknown format: {format_type}")


def get_output_dir(format_type: str, split: str) -> Path:
    """Get output directory for given format and split."""
    return VIZ_BASE / format_type / split


def process_scenario_format(visualizer, data_dir: Path, output_dir: Path,
                            num_scenarios: int, multi_frame: bool,
                            combined: bool = False) -> int:
    """Process scenarios from raw scenario format."""
    if not HAS_WAYMO:
        print("  ERROR: waymo_open_dataset package required for scenario format")
        return 0
    
    tfrecord_files = sorted(data_dir.glob('*.tfrecord*'))
    if not tfrecord_files:
        print(f"  No tfrecord files found in {data_dir}")
        return 0
    
    print(f"  Found {len(tfrecord_files)} files")
    
    processed = 0
    for tfrecord_path in tfrecord_files[:num_scenarios]:
        try:
            scenario = visualizer.load_scenario(str(tfrecord_path), 0)
            
            # Temporarily change output dir
            old_output = visualizer.output_dir
            visualizer.output_dir = output_dir
            
            # Generate overview
            visualizer.visualize_overview(scenario)
            
            # Generate multi-frame if requested
            if multi_frame:
                visualizer.visualize_multi_frame(scenario)
            
            # Generate combined if requested
            if combined:
                visualizer.visualize_combined(scenario)
            
            visualizer.output_dir = old_output
            processed += 1
            
        except Exception as e:
            print(f"  Error processing {tfrecord_path.name}: {e}")
            continue
    
    return processed


def process_tf_format(visualizer, data_dir: Path, output_dir: Path,
                      num_scenarios: int, multi_frame: bool,
                      combined: bool = False) -> int:
    """
    Process scenarios from processed tf.Example format.
    
    Note: TF format visualization is limited since tf.Example doesn't contain
    the full scenario protobuf structure. Creates basic trajectory plots.
    """
    tfrecord_files = sorted(data_dir.glob('*.tfrecord*'))
    if not tfrecord_files:
        print(f"  No tfrecord files found in {data_dir}")
        return 0
    
    print(f"  Found {len(tfrecord_files)} files")
    print("  Note: TF format has limited visualization (no full scenario data)")
    
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
            
            # Create TF format visualization
            output_path = output_dir / f"overview_tf_{file_id}.png"
            success = create_tf_visualization(example, output_path, visualizer.dpi)
            
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
    num_agents = len(get_float_list('state/current/x'))
    
    # Parse state data
    def parse_state(prefix, num_timesteps):
        x = get_float_list(f'{prefix}/x')
        y = get_float_list(f'{prefix}/y')
        valid = get_int_list(f'{prefix}/valid')
        
        expected = num_agents * num_timesteps
        if len(x) == expected:
            return {
                'x': x.reshape(num_agents, num_timesteps),
                'y': y.reshape(num_agents, num_timesteps),
                'valid': valid.reshape(num_agents, num_timesteps).astype(bool),
            }
        return None
    
    state_past = parse_state('state/past', 10)
    state_current = parse_state('state/current', 1)
    state_future = parse_state('state/future', 80)
    
    # Parse roadgraph
    roadgraph_xyz = get_float_list('roadgraph_samples/xyz')
    roadgraph_valid = get_int_list('roadgraph_samples/valid')
    
    # Parse agent types
    agent_type = get_float_list('state/type')
    is_sdc = get_int_list('state/is_sdc')
    
    return {
        'state_past': state_past,
        'state_current': state_current,
        'state_future': state_future,
        'roadgraph_xyz': roadgraph_xyz,
        'roadgraph_valid': roadgraph_valid,
        'agent_type': agent_type,
        'is_sdc': is_sdc,
        'num_agents': num_agents,
    }


def create_tf_visualization(example, output_path: Path, dpi: int = 150) -> bool:
    """Create visualization from TF format data."""
    
    data = parse_tf_example(example)
    
    # Colors
    COLORS = {
        'ego': '#FF0000',
        'vehicle': '#3498db',
        'pedestrian': '#2ecc71',
        'cyclist': '#f39c12',
        'other': '#95a5a6',
        'roadmap': '#7f8c8d',
        'past': '#e74c3c',
        'future': '#27ae60',
    }
    
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Find SDC (ego vehicle)
    sdc_idx = None
    if len(data['is_sdc']) > 0:
        sdc_indices = np.where(data['is_sdc'] == 1)[0]
        if len(sdc_indices) > 0:
            sdc_idx = sdc_indices[0]
    
    if sdc_idx is None and data['state_current'] is not None:
        valid_agents = np.where(data['state_current']['valid'][:, 0])[0]
        if len(valid_agents) > 0:
            sdc_idx = valid_agents[0]
    
    # Draw roadgraph
    if len(data['roadgraph_xyz']) > 0:
        xyz = data['roadgraph_xyz'].reshape(-1, 3)
        valid = data['roadgraph_valid'].reshape(-1) > 0
        if np.any(valid):
            pts = xyz[valid]
            ax.scatter(pts[:, 0], pts[:, 1], c=COLORS['roadmap'], s=1, alpha=0.3, label='Road')
    
    # Plot trajectories for each agent
    labeled_types = set()
    for agent_idx in range(min(data['num_agents'], 32)):
        is_ego = (agent_idx == sdc_idx)
        
        # Collect all positions
        all_x, all_y = [], []
        past_x, past_y = [], []
        future_x, future_y = [], []
        
        if data['state_past'] is not None:
            for t in range(10):
                if data['state_past']['valid'][agent_idx, t]:
                    past_x.append(data['state_past']['x'][agent_idx, t])
                    past_y.append(data['state_past']['y'][agent_idx, t])
        
        if data['state_current'] is not None:
            if data['state_current']['valid'][agent_idx, 0]:
                curr_x = data['state_current']['x'][agent_idx, 0]
                curr_y = data['state_current']['y'][agent_idx, 0]
                all_x.append(curr_x)
                all_y.append(curr_y)
        
        if data['state_future'] is not None:
            for t in range(80):
                if data['state_future']['valid'][agent_idx, t]:
                    future_x.append(data['state_future']['x'][agent_idx, t])
                    future_y.append(data['state_future']['y'][agent_idx, t])
        
        if not past_x and not future_x and not all_x:
            continue
        
        # Get agent color
        if is_ego:
            color = COLORS['ego']
            linewidth = 3.0
            alpha = 1.0
            label = 'Ego (SDC)'
        else:
            if len(data['agent_type']) > agent_idx:
                atype = int(data['agent_type'][agent_idx])
                type_map = {1: 'vehicle', 2: 'pedestrian', 3: 'cyclist'}
                type_name = type_map.get(atype, 'other')
            else:
                type_name = 'other'
            
            color = COLORS.get(type_name, COLORS['other'])
            linewidth = 1.5
            alpha = 0.6
            label = type_name.title() if type_name not in labeled_types else None
            if label:
                labeled_types.add(type_name)
        
        # Plot past trajectory
        if len(past_x) > 1:
            ax.plot(past_x, past_y, color=color, linewidth=linewidth, 
                   alpha=alpha * 0.5, linestyle='--')
        
        # Plot future trajectory
        if len(future_x) > 1:
            ax.plot(future_x, future_y, color=color, linewidth=linewidth, 
                   alpha=alpha, linestyle='-', label=label)
        
        # Mark current position
        if all_x:
            ax.plot(all_x[0], all_y[0], 'o', color=color, markersize=8 if is_ego else 5,
                   markeredgecolor='black', markeredgewidth=1)
    
    ax.set_xlabel('X (meters)', fontsize=12)
    ax.set_ylabel('Y (meters)', fontsize=12)
    ax.set_title(f'TF Format Trajectory Overview\nAgents: {data["num_agents"]}',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  ✓ Saved: {output_path.name} ({file_size_mb:.2f} MB)")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Create static visualizations from Waymo data"
    )
    
    # Input options
    parser.add_argument('--tfrecord', type=str,
                       help='Path to single TFRecord file')
    parser.add_argument('--format', type=str, choices=['scenario', 'tf'],
                       help='Data format: scenario (raw) or tf (processed)')
    parser.add_argument('--split', type=str,
                       choices=SPLITS,
                       help='Dataset split to process')
    parser.add_argument('--all', action='store_true',
                       help='Process all formats and splits')
    
    # Processing options
    parser.add_argument('--num', '-n', type=int, default=5,
                       help='Number of scenarios per split (default: 5)')
    parser.add_argument('--scenario-index', type=int, default=0,
                       help='Scenario index in file (for --tfrecord)')
    parser.add_argument('--multi-frame', action='store_true',
                       help='Also create multi-frame visualization')
    parser.add_argument('--combined', action='store_true',
                       help='Create high-quality combined visualization')
    
    # Output options
    parser.add_argument('--output-dir', type=Path,
                       help='Custom output directory')
    parser.add_argument('--dpi', type=int, default=150,
                       help='Resolution quality (default: 150)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.tfrecord and not args.format and not args.all:
        parser.error("Must specify --tfrecord, --format/--split, or --all")
    
    if args.format and not args.split:
        parser.error("--format requires --split")
    
    print("=" * 60)
    print("Static Scenario Visualizer")
    print("=" * 60)
    
    total_processed = 0
    
    # Single file mode
    if args.tfrecord:
        output_dir = args.output_dir or VIZ_BASE / 'single'
        visualizer = ScenarioVisualizer(output_dir=output_dir, dpi=args.dpi)
        
        print(f"Processing: {args.tfrecord}")
        scenario = visualizer.load_scenario(args.tfrecord, args.scenario_index)
        
        visualizer.visualize_overview(scenario)
        if args.multi_frame:
            visualizer.visualize_multi_frame(scenario)
        if args.combined:
            visualizer.visualize_combined(scenario)
        
        total_processed = 1
    
    # Format/split mode
    elif args.format and args.split:
        data_dir = get_data_dir(args.format, args.split)
        output_dir = args.output_dir or get_output_dir(args.format, args.split)
        
        visualizer = ScenarioVisualizer(output_dir=output_dir, dpi=args.dpi)
        
        print(f"\nFormat: {args.format}")
        print(f"Split: {args.split}")
        print(f"Input: {data_dir}")
        print(f"Output: {output_dir}")
        print(f"Scenarios: {args.num}")
        
        if not data_dir.exists():
            print(f"ERROR: Directory not found: {data_dir}")
            return 1
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if args.format == 'scenario':
            total_processed = process_scenario_format(
                visualizer, data_dir, output_dir,
                args.num, args.multi_frame, args.combined
            )
        else:
            total_processed = process_tf_format(
                visualizer, data_dir, output_dir,
                args.num, args.multi_frame, args.combined
            )
    
    # Process all formats and splits
    elif args.all:
        print(f"Processing ALL formats and splits")
        print(f"Scenarios per split: {args.num}")
        
        for format_type in ['scenario', 'tf']:
            print(f"\n{'=' * 60}")
            print(f"Format: {format_type.upper()}")
            print('=' * 60)
            
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
                
                visualizer = ScenarioVisualizer(output_dir=output_dir, dpi=args.dpi)
                
                if format_type == 'scenario':
                    count = process_scenario_format(
                        visualizer, data_dir, output_dir,
                        args.num, args.multi_frame, args.combined
                    )
                else:
                    count = process_tf_format(
                        visualizer, data_dir, output_dir,
                        args.num, args.multi_frame, args.combined
                    )
                
                total_processed += count
                print(f"  Generated: {count} visualizations")
    
    # Summary
    print(f"\n{'=' * 60}")
    print(f"COMPLETE: Generated {total_processed} visualizations")
    print(f"Output: {VIZ_BASE}")
    print('=' * 60)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
