#!/usr/bin/env python3
"""
Generate movies visualizing M2I predictions.

Shows:
- Road map (lanes, boundaries, crosswalks) as polylines matching BEV style
- All agents in the scene with oriented rectangle patches
- Ground truth trajectories (past and future) 
- Influencer's predicted trajectory
- Reactor's 6 conditional predicted trajectories with confidence scores
- Type-based coloring (vehicle, pedestrian, cyclist)

Visual style matches data/scripts/lib/generate_bev_movie.py for consistency.
"""

import os
import sys
import pickle
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np

# Add M2I source to path
sys.path.insert(0, '/workspace/externals/M2I/src')

import tensorflow as tf
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, Polygon, Circle, FancyArrow
from matplotlib.collections import LineCollection
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter, PillowWriter
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.transforms as transforms
from matplotlib.lines import Line2D


class M2IMovieGenerator:
    """Generate movies visualizing M2I predictions on Waymo scenarios."""
    
    # Colors matching BEV movie generator style
    COLORS = {
        # Agent types
        'influencer': '#FF0000',        # Red for influencer (like ego)
        'reactor': '#f39c12',           # Orange for reactor
        'vehicle': '#3498db',           # Blue for other vehicles
        'pedestrian': '#2ecc71',        # Green for pedestrians
        'cyclist': '#9b59b6',           # Purple for cyclists
        'other': '#95a5a6',             # Gray for others
        # Road elements
        'lane': '#34495e',              # Dark gray for lanes
        'road_line': '#7f8c8d',         # Medium gray for road lines
        'road_edge': '#2c3e50',         # Very dark gray for road edges
        'crosswalk': '#16a085',         # Teal for crosswalks
        # Trajectories
        'history_influencer': '#e74c3c', # Red for influencer history
        'history_reactor': '#f39c12',    # Orange for reactor history
        'future_gt': '#bdc3c7',          # Light gray for GT future
        'pred_influencer': '#27ae60',    # Green for influencer prediction
        'pred_reactor': plt.cm.plasma,   # Colormap for reactor modes
    }
    
    # Roadgraph type mapping (Waymo Motion Dataset)
    ROAD_TYPE_NAMES = {
        1: 'lane_freeway',
        2: 'lane_surface_street',
        3: 'lane_bike_lane',
        6: 'road_line_broken_single_white',
        7: 'road_line_solid_single_white',
        8: 'road_line_solid_double_white',
        9: 'road_line_broken_single_yellow',
        10: 'road_line_broken_double_yellow',
        11: 'road_line_solid_single_yellow',
        12: 'road_line_solid_double_yellow',
        13: 'road_line_passing_double_yellow',
        15: 'road_edge_boundary',
        16: 'road_edge_median',
        17: 'stop_sign',
        18: 'crosswalk',
        19: 'speed_bump',
    }
    
    def __init__(self, 
                 data_dir: Path,
                 marginal_path: Path,
                 relation_path: Path,
                 conditional_path: Path,
                 output_dir: Path,
                 view_range: float = 50.0,
                 fps: int = 10,
                 dpi: int = 100):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.view_range = view_range
        self.fps = fps
        self.dpi = dpi
        
        # Load predictions
        print("Loading predictions...")
        with open(marginal_path, 'rb') as f:
            self.marginal = pickle.load(f)
        with open(relation_path, 'rb') as f:
            self.relations = pickle.load(f)
        with open(conditional_path, 'rb') as f:
            self.conditional = pickle.load(f)
        
        print(f"  Marginal: {len(self.marginal)} scenarios")
        print(f"  Relations: {len(self.relations)} pairs")
        print(f"  Conditional: {len(self.conditional)} scenarios")
        
        # Find scenarios with all predictions
        self.valid_scenarios = set(self.marginal.keys()) & set(self.relations.keys()) & set(self.conditional.keys())
        print(f"  Valid scenarios (all 3 stages): {len(self.valid_scenarios)}")
        
        # TFRecord files
        self.tfrecord_files = sorted(self.data_dir.glob("*.tfrecord*"))
        print(f"  TFRecord files: {len(self.tfrecord_files)}")
        
        # Feature description for parsing
        self.features_description = self._build_features_description()
        
        print(f"M2I Movie Generator initialized:")
        print(f"  Output: {self.output_dir}")
        print(f"  FPS: {fps}, DPI: {dpi}, View Range: {view_range}m")
    
    def _build_features_description(self):
        """Build TFRecord feature description for flat format with all agent data."""
        return {
            'scenario/id': tf.io.FixedLenFeature([], tf.string),
            # Agent state info
            'state/id': tf.io.FixedLenFeature([128], tf.float32),
            'state/type': tf.io.FixedLenFeature([128], tf.float32),
            'state/is_sdc': tf.io.FixedLenFeature([128], tf.int64),
            'state/tracks_to_predict': tf.io.FixedLenFeature([128], tf.int64),
            # Current state
            'state/current/x': tf.io.FixedLenFeature([128], tf.float32),
            'state/current/y': tf.io.FixedLenFeature([128], tf.float32),
            'state/current/bbox_yaw': tf.io.FixedLenFeature([128], tf.float32),
            'state/current/length': tf.io.FixedLenFeature([128], tf.float32),
            'state/current/width': tf.io.FixedLenFeature([128], tf.float32),
            'state/current/velocity_x': tf.io.FixedLenFeature([128], tf.float32),
            'state/current/velocity_y': tf.io.FixedLenFeature([128], tf.float32),
            'state/current/valid': tf.io.FixedLenFeature([128], tf.int64),
            # Past state
            'state/past/x': tf.io.FixedLenFeature([1280], tf.float32),
            'state/past/y': tf.io.FixedLenFeature([1280], tf.float32),
            'state/past/bbox_yaw': tf.io.FixedLenFeature([1280], tf.float32),
            'state/past/length': tf.io.FixedLenFeature([1280], tf.float32),
            'state/past/width': tf.io.FixedLenFeature([1280], tf.float32),
            'state/past/velocity_x': tf.io.FixedLenFeature([1280], tf.float32),
            'state/past/velocity_y': tf.io.FixedLenFeature([1280], tf.float32),
            'state/past/valid': tf.io.FixedLenFeature([1280], tf.int64),
            # Future state
            'state/future/x': tf.io.FixedLenFeature([10240], tf.float32),
            'state/future/y': tf.io.FixedLenFeature([10240], tf.float32),
            'state/future/bbox_yaw': tf.io.FixedLenFeature([10240], tf.float32),
            'state/future/length': tf.io.FixedLenFeature([10240], tf.float32),
            'state/future/width': tf.io.FixedLenFeature([10240], tf.float32),
            'state/future/velocity_x': tf.io.FixedLenFeature([10240], tf.float32),
            'state/future/velocity_y': tf.io.FixedLenFeature([10240], tf.float32),
            'state/future/valid': tf.io.FixedLenFeature([10240], tf.int64),
            # Roadgraph
            'roadgraph_samples/xyz': tf.io.FixedLenFeature([90000], tf.float32),
            'roadgraph_samples/type': tf.io.FixedLenFeature([30000], tf.int64),
            'roadgraph_samples/id': tf.io.FixedLenFeature([30000], tf.int64),
            'roadgraph_samples/valid': tf.io.FixedLenFeature([30000], tf.int64),
        }
    
    def parse_scenario(self, record) -> Optional[Dict]:
        """Parse a single TFRecord into a scenario dict with full agent data."""
        try:
            parsed = tf.io.parse_single_example(record, self.features_description)
            
            scenario_id = parsed['scenario/id'].numpy()
            if isinstance(scenario_id, bytes):
                scenario_id = scenario_id.decode()
            
            # Reshape states: 128 agents x timesteps
            # Position
            past_x = parsed['state/past/x'].numpy().reshape(128, 10)
            past_y = parsed['state/past/y'].numpy().reshape(128, 10)
            past_valid = parsed['state/past/valid'].numpy().reshape(128, 10)
            
            cur_x = parsed['state/current/x'].numpy().reshape(128, 1)
            cur_y = parsed['state/current/y'].numpy().reshape(128, 1)
            cur_valid = parsed['state/current/valid'].numpy().reshape(128, 1)
            
            future_x = parsed['state/future/x'].numpy().reshape(128, 80)
            future_y = parsed['state/future/y'].numpy().reshape(128, 80)
            future_valid = parsed['state/future/valid'].numpy().reshape(128, 80)
            
            # Heading/yaw
            past_yaw = parsed['state/past/bbox_yaw'].numpy().reshape(128, 10)
            cur_yaw = parsed['state/current/bbox_yaw'].numpy().reshape(128, 1)
            future_yaw = parsed['state/future/bbox_yaw'].numpy().reshape(128, 80)
            
            # Size
            past_length = parsed['state/past/length'].numpy().reshape(128, 10)
            past_width = parsed['state/past/width'].numpy().reshape(128, 10)
            cur_length = parsed['state/current/length'].numpy().reshape(128, 1)
            cur_width = parsed['state/current/width'].numpy().reshape(128, 1)
            future_length = parsed['state/future/length'].numpy().reshape(128, 80)
            future_width = parsed['state/future/width'].numpy().reshape(128, 80)
            
            # Velocity
            past_vx = parsed['state/past/velocity_x'].numpy().reshape(128, 10)
            past_vy = parsed['state/past/velocity_y'].numpy().reshape(128, 10)
            cur_vx = parsed['state/current/velocity_x'].numpy().reshape(128, 1)
            cur_vy = parsed['state/current/velocity_y'].numpy().reshape(128, 1)
            future_vx = parsed['state/future/velocity_x'].numpy().reshape(128, 80)
            future_vy = parsed['state/future/velocity_y'].numpy().reshape(128, 80)
            
            # Combine into [128, 91, ...] arrays
            x = np.concatenate([past_x, cur_x, future_x], axis=1)
            y = np.concatenate([past_y, cur_y, future_y], axis=1)
            yaw = np.concatenate([past_yaw, cur_yaw, future_yaw], axis=1)
            length = np.concatenate([past_length, cur_length, future_length], axis=1)
            width = np.concatenate([past_width, cur_width, future_width], axis=1)
            vx = np.concatenate([past_vx, cur_vx, future_vx], axis=1)
            vy = np.concatenate([past_vy, cur_vy, future_vy], axis=1)
            valid = np.concatenate([past_valid, cur_valid, future_valid], axis=1) > 0
            
            states = np.stack([x, y], axis=-1)  # [128, 91, 2]
            
            # Road graph - group by ID for polyline rendering
            road_xyz = parsed['roadgraph_samples/xyz'].numpy().reshape(30000, 3)
            road_type = parsed['roadgraph_samples/type'].numpy()
            road_id = parsed['roadgraph_samples/id'].numpy()
            road_valid = parsed['roadgraph_samples/valid'].numpy() > 0
            
            # Build polylines grouped by road ID
            road_polylines = self._build_road_polylines(
                road_xyz[road_valid], 
                road_type[road_valid], 
                road_id[road_valid]
            )
            
            return {
                'scenario_id': scenario_id,
                'state': states,
                'state_is_valid': valid,
                'yaw': yaw,
                'length': length,
                'width': width,
                'velocity_x': vx,
                'velocity_y': vy,
                'track_id': parsed['state/id'].numpy(),
                'track_type': parsed['state/type'].numpy().astype(int),
                'is_sdc': parsed['state/is_sdc'].numpy() > 0,
                'tracks_to_predict': parsed['state/tracks_to_predict'].numpy() > 0,
                'roadgraph_xyz': road_xyz[road_valid],
                'roadgraph_type': road_type[road_valid],
                'road_polylines': road_polylines,
            }
        except Exception as e:
            print(f"  Parse error: {e}")
            return None
    
    def _build_road_polylines(self, xyz: np.ndarray, types: np.ndarray, ids: np.ndarray) -> Dict:
        """Group road points by ID to form polylines for rendering."""
        polylines = {
            'lanes': [],           # Types 1-3
            'road_lines': [],      # Types 6-13
            'road_edges': [],      # Types 15-16
            'crosswalks': [],      # Type 18
            'stop_signs': [],      # Type 17
            'speed_bumps': [],     # Type 19
        }
        
        # Group points by road segment ID
        unique_ids = np.unique(ids)
        for road_id in unique_ids:
            mask = ids == road_id
            points = xyz[mask][:, :2]  # Just x, y
            road_types = types[mask]
            
            if len(points) < 2:
                continue
                
            # Determine category based on most common type
            most_common_type = int(np.bincount(road_types.astype(int)).argmax())
            
            if most_common_type in [1, 2, 3]:
                polylines['lanes'].append(points)
            elif most_common_type in [6, 7, 8, 9, 10, 11, 12, 13]:
                polylines['road_lines'].append(points)
            elif most_common_type in [15, 16]:
                polylines['road_edges'].append(points)
            elif most_common_type == 18:
                polylines['crosswalks'].append(points)
            elif most_common_type == 17:
                # Stop signs are points, not polylines
                polylines['stop_signs'].append(points[0])
            elif most_common_type == 19:
                polylines['speed_bumps'].append(points)
        
        return polylines
    
    def find_scenario_in_tfrecords(self, scenario_id: str) -> Optional[Dict]:
        """Find a scenario across all TFRecord files."""
        for tfrecord_path in self.tfrecord_files:
            dataset = tf.data.TFRecordDataset(str(tfrecord_path))
            for record in dataset:
                parsed = self.parse_scenario(record)
                if parsed is not None and parsed['scenario_id'] == scenario_id:
                    return parsed
        return None
    
    def _get_agent_color(self, agent_type: int, is_influencer: bool = False, is_reactor: bool = False) -> str:
        """Get color for an agent based on type and role."""
        if is_influencer:
            return self.COLORS['influencer']
        if is_reactor:
            return self.COLORS['reactor']
        
        type_map = {
            1: 'vehicle',
            2: 'pedestrian', 
            3: 'cyclist',
        }
        type_name = type_map.get(agent_type, 'other')
        return self.COLORS.get(type_name, self.COLORS['other'])
    
    def _create_vehicle_patch(self, ax, x: float, y: float, heading: float, 
                               length: float, width: float, color: str, 
                               alpha: float = 0.8) -> Rectangle:
        """Create oriented rectangle patch for vehicle (matching BEV style)."""
        # Default size if not available
        if length <= 0:
            length = 4.5
        if width <= 0:
            width = 2.0
            
        rect = Rectangle((-length/2, -width/2), length, width,
                         facecolor=color, edgecolor='black',
                         linewidth=1.5, alpha=alpha)
        
        # Apply rotation and translation
        t = transforms.Affine2D().rotate(heading).translate(x, y) + ax.transData
        rect.set_transform(t)
        
        return rect
    
    def _draw_road_polylines(self, ax, road_polylines: Dict, center_x: float, center_y: float):
        """Draw road elements as polylines (matching BEV style)."""
        view_range = self.view_range
        
        # Draw lanes (solid gray lines)
        for lane_points in road_polylines.get('lanes', []):
            # Filter points in view
            in_view = [(x, y) for x, y in lane_points 
                       if abs(x - center_x) < view_range and abs(y - center_y) < view_range]
            if len(in_view) > 1:
                xs, ys = zip(*in_view)
                ax.plot(xs, ys, color=self.COLORS['lane'], linewidth=1.5, alpha=0.4, linestyle='-', zorder=1)
        
        # Draw road lines (dashed)
        for line_points in road_polylines.get('road_lines', []):
            in_view = [(x, y) for x, y in line_points 
                       if abs(x - center_x) < view_range and abs(y - center_y) < view_range]
            if len(in_view) > 1:
                xs, ys = zip(*in_view)
                ax.plot(xs, ys, color=self.COLORS['road_line'], linewidth=1.0, alpha=0.5, linestyle='--', zorder=1)
        
        # Draw road edges (solid boundary)
        for edge_points in road_polylines.get('road_edges', []):
            in_view = [(x, y) for x, y in edge_points 
                       if abs(x - center_x) < view_range and abs(y - center_y) < view_range]
            if len(in_view) > 1:
                xs, ys = zip(*in_view)
                ax.plot(xs, ys, color=self.COLORS['road_edge'], linewidth=2.0, alpha=0.6, linestyle='-', zorder=1)
        
        # Draw crosswalks (filled polygons)
        for crosswalk_points in road_polylines.get('crosswalks', []):
            in_view = [(x, y) for x, y in crosswalk_points 
                       if abs(x - center_x) < view_range and abs(y - center_y) < view_range]
            if len(in_view) > 2:
                polygon = Polygon(in_view, closed=True,
                                  facecolor=self.COLORS['crosswalk'],
                                  edgecolor=self.COLORS['crosswalk'],
                                  linewidth=2, alpha=0.3, zorder=1)
                ax.add_patch(polygon)
        
        # Draw stop signs (circles with text)
        for stop_pos in road_polylines.get('stop_signs', []):
            sx, sy = stop_pos[0], stop_pos[1]
            if abs(sx - center_x) < view_range and abs(sy - center_y) < view_range:
                circle = Circle((sx, sy), 1.5, color='#c0392b', alpha=0.7, zorder=5)
                ax.add_patch(circle)
    
    def generate_scenario_movie(self, scenario_id: str, fps: int = None, 
                                 follow_agent: bool = True, show_all_agents: bool = True) -> Optional[Path]:
        """
        Generate a movie for a single scenario showing M2I predictions.
        
        Visual style matches data/scripts/lib/generate_bev_movie.py:
        - Polyline road rendering
        - Oriented vehicle rectangles  
        - All agents displayed with type-based colors
        - Follow-agent camera mode
        - Comprehensive legend
        
        Args:
            scenario_id: Waymo scenario ID
            fps: Frames per second (default: self.fps)
            follow_agent: If True, camera follows the influencer
            show_all_agents: If True, show all agents in scene (not just interacting pair)
        """
        if fps is None:
            fps = self.fps
            
        print(f"\nGenerating movie for scenario: {scenario_id}")
        
        # Check if we have all predictions
        if scenario_id not in self.valid_scenarios:
            print(f"  Skipping - missing predictions")
            return None
        
        # Get predictions
        marginal_data = self.marginal[scenario_id]
        relation_label, relation_score = self.relations[scenario_id]
        conditional_data = self.conditional[scenario_id]
        
        # Find scenario data in TFRecords
        scenario_data = self.find_scenario_in_tfrecords(scenario_id)
        if scenario_data is None:
            print(f"  Skipping - scenario not found in TFRecords")
            return None
        
        # Extract data
        states = scenario_data['state']  # [128, 91, 2]
        valid = scenario_data['state_is_valid']  # [128, 91]
        yaw = scenario_data['yaw']  # [128, 91]
        length = scenario_data['length']  # [128, 91]
        width = scenario_data['width']  # [128, 91]
        vx = scenario_data['velocity_x']  # [128, 91]
        vy = scenario_data['velocity_y']  # [128, 91]
        track_ids = scenario_data['track_id'].astype(int)
        track_types = scenario_data['track_type']
        road_polylines = scenario_data.get('road_polylines', {})
        
        # Map predicted agent IDs to indices in the TFRecord
        marginal_ids = list(marginal_data['ids'])
        conditional_ids = list(conditional_data['ids'])
        
        if len(marginal_ids) < 2:
            print(f"  Skipping - not enough agents in marginal predictions")
            return None
        
        # Find agent indices for reactor (conditional) and influencer (marginal)
        reactor_id = conditional_ids[0] if len(conditional_ids) > 0 else marginal_ids[0]
        influencer_id = marginal_ids[1] if len(marginal_ids) > 1 else marginal_ids[0]
        
        reactor_idx = np.where(track_ids == reactor_id)[0]
        influencer_idx = np.where(track_ids == influencer_id)[0]
        
        if len(reactor_idx) == 0 or len(influencer_idx) == 0:
            print(f"  Skipping - cannot find agents {reactor_id}, {influencer_id} in TFRecord")
            return None
        
        reactor_idx = reactor_idx[0]
        influencer_idx = influencer_idx[0]
        
        # Create the movie
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        
        # Time parameters
        history_steps = 11  # 1.1 seconds of history (timestep 10 is "current")
        future_steps = 80   # 8 seconds of future
        total_steps = history_steps + future_steps
        
        # Calculate coordinate offsets for predictions
        gt_current_reactor = states[reactor_idx, history_steps - 1, :2]
        gt_current_influencer = states[influencer_idx, history_steps - 1, :2]
        
        # Prediction offsets
        pred_offset_reactor = gt_current_reactor - conditional_data['rst'][0, 0, 0, :2]
        
        inf_idx_in_marg = marginal_ids.index(influencer_id) if influencer_id in marginal_ids else -1
        if inf_idx_in_marg >= 0:
            pred_offset_influencer = gt_current_influencer - marginal_data['rst'][inf_idx_in_marg, 0, 0, :2]
            inf_traj_all = marginal_data['rst'][inf_idx_in_marg].copy() + pred_offset_influencer
            inf_best_mode = np.argmax(marginal_data['score'][inf_idx_in_marg])
        else:
            inf_traj_all = None
        
        reactor_traj_all = conditional_data['rst'][0].copy() + pred_offset_reactor
        reactor_scores = conditional_data['score'][0]
        scores_norm = np.exp(reactor_scores - np.max(reactor_scores))
        scores_norm = scores_norm / np.sum(scores_norm)
        
        # Calculate plot bounds (if not following agent)
        if not follow_agent:
            all_valid_positions = []
            for i in range(128):
                v_mask = valid[i]
                if v_mask.any():
                    all_valid_positions.extend(states[i, v_mask].tolist())
            if all_valid_positions:
                all_valid_positions = np.array(all_valid_positions)
                x_min = all_valid_positions[:, 0].min() - 20
                x_max = all_valid_positions[:, 0].max() + 20
                y_min = all_valid_positions[:, 1].min() - 20
                y_max = all_valid_positions[:, 1].max() + 20
            else:
                x_min, x_max, y_min, y_max = -50, 50, -50, 50
        
        def animate(frame):
            ax.clear()
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            
            t = frame
            
            # Get influencer position for camera centering
            if valid[influencer_idx, t]:
                center_x = states[influencer_idx, t, 0]
                center_y = states[influencer_idx, t, 1]
            else:
                # Fall back to last valid position
                for check_t in range(t, -1, -1):
                    if valid[influencer_idx, check_t]:
                        center_x = states[influencer_idx, check_t, 0]
                        center_y = states[influencer_idx, check_t, 1]
                        break
                else:
                    center_x, center_y = 0, 0
            
            # Set view bounds
            if follow_agent:
                ax.set_xlim(center_x - self.view_range, center_x + self.view_range)
                ax.set_ylim(center_y - self.view_range, center_y + self.view_range)
            else:
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)
            
            ax.set_facecolor('#f5f5f5')
            
            # Draw road (polylines matching BEV style)
            self._draw_road_polylines(ax, road_polylines, center_x, center_y)
            
            # Draw all agents
            for agent_i in range(128):
                if not valid[agent_i, t]:
                    continue
                
                agent_x = states[agent_i, t, 0]
                agent_y = states[agent_i, t, 1]
                
                # Skip if out of view
                if follow_agent:
                    if abs(agent_x - center_x) > self.view_range or abs(agent_y - center_y) > self.view_range:
                        continue
                
                agent_yaw = yaw[agent_i, t]
                agent_length = length[agent_i, t]
                agent_width = width[agent_i, t]
                agent_type = track_types[agent_i]
                
                # Determine if this is influencer or reactor
                is_influencer = (agent_i == influencer_idx)
                is_reactor = (agent_i == reactor_idx)
                
                # Get color based on role and type
                color = self._get_agent_color(agent_type, is_influencer, is_reactor)
                
                # Draw oriented rectangle
                if show_all_agents or is_influencer or is_reactor:
                    rect = self._create_vehicle_patch(ax, agent_x, agent_y, agent_yaw,
                                                      agent_length, agent_width, color)
                    ax.add_patch(rect)
                    
                    # Draw velocity arrow for moving agents
                    speed = np.sqrt(vx[agent_i, t]**2 + vy[agent_i, t]**2)
                    if speed > 0.5:
                        arrow = FancyArrow(agent_x, agent_y,
                                          vx[agent_i, t] * 0.5, vy[agent_i, t] * 0.5,
                                          width=0.3, head_width=0.8, head_length=0.5,
                                          fc=color, ec='black', alpha=0.6, linewidth=0.5, zorder=6)
                        ax.add_patch(arrow)
                    
                    # Label influencer and reactor
                    if is_influencer:
                        ax.annotate('Influencer', (agent_x, agent_y),
                                   textcoords="offset points", xytext=(5, 8),
                                   fontsize=9, fontweight='bold', color=color, zorder=10)
                    elif is_reactor:
                        ax.annotate('Reactor', (agent_x, agent_y),
                                   textcoords="offset points", xytext=(5, 8),
                                   fontsize=9, fontweight='bold', color=color, zorder=10)
            
            # Draw history trails (dashed lines)
            history_length = min(t + 1, 15)  # Show up to 1.5 seconds of history
            
            # Influencer history
            inf_history = []
            for hist_t in range(max(0, t - history_length), t + 1):
                if valid[influencer_idx, hist_t]:
                    inf_history.append(states[influencer_idx, hist_t])
            if len(inf_history) > 1:
                inf_history = np.array(inf_history)
                ax.plot(inf_history[:, 0], inf_history[:, 1],
                       color=self.COLORS['history_influencer'], linewidth=2.5,
                       alpha=0.5, linestyle='--', zorder=3)
            
            # Reactor history
            react_history = []
            for hist_t in range(max(0, t - history_length), t + 1):
                if valid[reactor_idx, hist_t]:
                    react_history.append(states[reactor_idx, hist_t])
            if len(react_history) > 1:
                react_history = np.array(react_history)
                ax.plot(react_history[:, 0], react_history[:, 1],
                       color=self.COLORS['history_reactor'], linewidth=2.5,
                       alpha=0.5, linestyle='--', zorder=3)
            
            # Draw ground truth future (after history ends)
            if t >= history_steps - 1:
                # Influencer GT future (light gray dashed)
                future_mask = valid[influencer_idx, history_steps:]
                if future_mask.any():
                    gt_future = states[influencer_idx, history_steps:][future_mask]
                    if len(gt_future) > 1:
                        ax.plot(gt_future[:, 0], gt_future[:, 1],
                               color=self.COLORS['future_gt'], linewidth=1.5,
                               linestyle=':', alpha=0.4, zorder=2)
                
                # Reactor GT future
                future_mask = valid[reactor_idx, history_steps:]
                if future_mask.any():
                    gt_future = states[reactor_idx, history_steps:][future_mask]
                    if len(gt_future) > 1:
                        ax.plot(gt_future[:, 0], gt_future[:, 1],
                               color=self.COLORS['future_gt'], linewidth=1.5,
                               linestyle=':', alpha=0.4, zorder=2)
            
            # Draw predicted trajectories after history ends
            if t >= history_steps:
                pred_step = t - history_steps
                
                # Draw influencer's predicted trajectory (from marginal)
                if inf_traj_all is not None and pred_step < 80:
                    inf_pred = inf_traj_all[inf_best_mode, :pred_step+1, :]
                    if len(inf_pred) > 0:
                        ax.plot(inf_pred[:, 0], inf_pred[:, 1],
                               color=self.COLORS['pred_influencer'], linewidth=3,
                               linestyle='-', alpha=0.9, zorder=4)
                        # Star at endpoint
                        ax.scatter(inf_pred[-1, 0], inf_pred[-1, 1],
                                  c=self.COLORS['pred_influencer'], s=100, 
                                  marker='*', edgecolors='white', linewidths=1, zorder=5)
                
                # Draw reactor's 6 conditional trajectories
                if pred_step < 80:
                    for mode in range(6):
                        traj = reactor_traj_all[mode, :pred_step+1, :]
                        score = scores_norm[mode]
                        
                        if len(traj) > 0:
                            color = plt.cm.plasma(mode / 5)
                            alpha = 0.3 + 0.7 * score
                            linewidth = 1.5 + 3 * score
                            
                            ax.plot(traj[:, 0], traj[:, 1],
                                   color=color, linewidth=linewidth,
                                   alpha=alpha, zorder=4)
                            
                            # Mark trajectory endpoint with star
                            if pred_step > 0:
                                ax.scatter(traj[-1, 0], traj[-1, 1],
                                          c=[color], s=50 + 50*score, marker='*',
                                          edgecolors='white', linewidths=0.5,
                                          alpha=alpha, zorder=5)
            
            # Title
            relation_str = ['Smaller ID → Influencer', 'Larger ID → Influencer', 'No Interaction'][relation_label]
            time_sec = t * 0.1  # 10Hz
            ax.set_title(
                f"M2I Prediction - {scenario_id[:20]}...\n"
                f"Relation: {relation_str} | Frame {t}/{total_steps-1} | Time: {time_sec:.1f}s",
                fontsize=12, fontweight='bold'
            )
            
            ax.set_xlabel('X (meters)', fontsize=10)
            ax.set_ylabel('Y (meters)', fontsize=10)
            
            # Comprehensive legend (matching BEV style)
            legend_elements = [
                mpatches.Patch(facecolor=self.COLORS['influencer'], edgecolor='black', label='Influencer'),
                mpatches.Patch(facecolor=self.COLORS['reactor'], edgecolor='black', label='Reactor'),
                mpatches.Patch(facecolor=self.COLORS['vehicle'], edgecolor='black', label='Other Vehicles'),
                mpatches.Patch(facecolor=self.COLORS['pedestrian'], edgecolor='black', label='Pedestrians'),
                mpatches.Patch(facecolor=self.COLORS['cyclist'], edgecolor='black', label='Cyclists'),
                Line2D([0], [0], color=self.COLORS['history_influencer'], linestyle='--', linewidth=2, label='History Trail'),
                Line2D([0], [0], color=self.COLORS['future_gt'], linestyle=':', linewidth=1.5, label='Ground Truth'),
                Line2D([0], [0], color=self.COLORS['pred_influencer'], linestyle='-', linewidth=3, label='Influencer Pred'),
                mpatches.Patch(facecolor='purple', alpha=0.7, label='Reactor Modes (6)'),
                Line2D([0], [0], color=self.COLORS['lane'], linestyle='-', linewidth=1.5, label='Lanes'),
                Line2D([0], [0], color=self.COLORS['road_edge'], linestyle='-', linewidth=2, label='Road Edges'),
            ]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=8,
                     framealpha=0.9, ncol=1)
            
            return []
        
        # Create animation
        num_frames = min(total_steps, 91)
        anim = animation.FuncAnimation(fig, animate, frames=num_frames,
                                       interval=1000//fps, blit=False, repeat=True)
        
        # Save MP4
        output_path = self.output_dir / f"{scenario_id}.mp4"
        try:
            print(f"  Rendering {num_frames} frames...")
            writer = FFMpegWriter(fps=fps, bitrate=1800,
                                  codec='libx264', extra_args=['-pix_fmt', 'yuv420p'])
            anim.save(str(output_path), writer=writer, dpi=self.dpi)
            
            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"  ✓ Saved MP4: {output_path.name} ({file_size_mb:.1f} MB)")
            
            # Also save GIF
            gif_path = output_path.with_suffix('.gif')
            gif_fps = min(fps, 10)
            gif_writer = PillowWriter(fps=gif_fps)
            anim.save(str(gif_path), writer=gif_writer, dpi=max(self.dpi // 2, 50))
            
            gif_size_mb = gif_path.stat().st_size / (1024 * 1024)
            print(f"  ✓ Saved GIF: {gif_path.name} ({gif_size_mb:.1f} MB)")
            
            plt.close(fig)
            return output_path
            
        except Exception as e:
            print(f"  ✗ Error saving movie: {e}")
            import traceback
            traceback.print_exc()
            plt.close(fig)
            return None
    
    def generate_static_visualization(self, scenario_id: str, show_all_agents: bool = True) -> Optional[Path]:
        """Generate a static image showing the full prediction (matching BEV style)."""
        
        print(f"\nGenerating visualization for scenario: {scenario_id}")
        
        if scenario_id not in self.valid_scenarios:
            print(f"  Skipping - missing predictions")
            return None
        
        # Get predictions
        marginal_data = self.marginal[scenario_id]
        relation_label, relation_score = self.relations[scenario_id]
        conditional_data = self.conditional[scenario_id]
        
        # Find scenario data
        scenario_data = self.find_scenario_in_tfrecords(scenario_id)
        if scenario_data is None:
            print(f"  Skipping - scenario not found in TFRecords")
            return None
        
        # Extract data
        states = scenario_data['state']  # [128, 91, 2]
        valid = scenario_data['state_is_valid']  # [128, 91]
        yaw = scenario_data['yaw']  # [128, 91]
        length = scenario_data['length']  # [128, 91]
        width = scenario_data['width']  # [128, 91]
        track_ids = scenario_data['track_id'].astype(int)
        track_types = scenario_data['track_type']
        road_polylines = scenario_data.get('road_polylines', {})
        
        # Map predicted agent IDs to indices in the TFRecord
        marginal_ids = list(marginal_data['ids'])
        conditional_ids = list(conditional_data['ids'])
        
        # Find agent indices for reactor (conditional) and influencer (marginal)
        reactor_id = conditional_ids[0] if len(conditional_ids) > 0 else marginal_ids[0]
        influencer_id = marginal_ids[1] if len(marginal_ids) > 1 else marginal_ids[0]
        
        reactor_idx = np.where(track_ids == reactor_id)[0]
        influencer_idx = np.where(track_ids == influencer_id)[0]
        
        if len(reactor_idx) == 0 or len(influencer_idx) == 0:
            print(f"  Skipping - cannot find agents {reactor_id}, {influencer_id} in TFRecord")
            return None
        
        reactor_idx = reactor_idx[0]
        influencer_idx = influencer_idx[0]
        
        history_steps = 11
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(14, 14))
        
        # Calculate coordinate offset for predictions
        gt_current_reactor = states[reactor_idx, history_steps - 1, :2]
        gt_current_influencer = states[influencer_idx, history_steps - 1, :2]
        
        pred_offset_reactor = gt_current_reactor - conditional_data['rst'][0, 0, 0, :2]
        
        inf_idx_in_marg = marginal_ids.index(influencer_id) if influencer_id in marginal_ids else -1
        if inf_idx_in_marg >= 0:
            pred_offset_influencer = gt_current_influencer - marginal_data['rst'][inf_idx_in_marg, 0, 0, :2]
        
        # Calculate plot bounds based on all valid agent positions
        all_positions = []
        for i in range(128):
            v_mask = valid[i]
            if v_mask.any():
                all_positions.extend(states[i, v_mask].tolist())
        
        if all_positions:
            all_positions = np.array(all_positions)
            x_min = all_positions[:, 0].min() - 30
            x_max = all_positions[:, 0].max() + 30
            y_min = all_positions[:, 1].min() - 30
            y_max = all_positions[:, 1].max() + 30
        else:
            x_min, x_max, y_min, y_max = -50, 50, -50, 50
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')
        ax.set_facecolor('#f5f5f5')
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        # Draw road polylines (matching BEV style)
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        
        # Lanes
        for lane_points in road_polylines.get('lanes', []):
            if len(lane_points) > 1:
                xs, ys = zip(*lane_points)
                ax.plot(xs, ys, color=self.COLORS['lane'], linewidth=1.5, alpha=0.4, linestyle='-', zorder=1)
        
        # Road edges
        for edge_points in road_polylines.get('road_edges', []):
            if len(edge_points) > 1:
                xs, ys = zip(*edge_points)
                ax.plot(xs, ys, color=self.COLORS['road_edge'], linewidth=2.0, alpha=0.6, linestyle='-', zorder=1)
        
        # Road lines
        for line_points in road_polylines.get('road_lines', []):
            if len(line_points) > 1:
                xs, ys = zip(*line_points)
                ax.plot(xs, ys, color=self.COLORS['road_line'], linewidth=1.0, alpha=0.5, linestyle='--', zorder=1)
        
        # Crosswalks
        for crosswalk_points in road_polylines.get('crosswalks', []):
            if len(crosswalk_points) > 2:
                polygon = Polygon(crosswalk_points, closed=True,
                                  facecolor=self.COLORS['crosswalk'],
                                  edgecolor=self.COLORS['crosswalk'],
                                  linewidth=2, alpha=0.3, zorder=1)
                ax.add_patch(polygon)
        
        # Draw all agents at the current time (history_steps - 1)
        t = history_steps - 1  # Current time = end of history
        for agent_i in range(128):
            if not valid[agent_i, t]:
                continue
            
            agent_x = states[agent_i, t, 0]
            agent_y = states[agent_i, t, 1]
            agent_yaw = yaw[agent_i, t]
            agent_length = length[agent_i, t]
            agent_width = width[agent_i, t]
            agent_type = track_types[agent_i]
            
            is_influencer = (agent_i == influencer_idx)
            is_reactor = (agent_i == reactor_idx)
            
            # Get color based on role and type
            color = self._get_agent_color(agent_type, is_influencer, is_reactor)
            
            # Draw oriented rectangle
            if show_all_agents or is_influencer or is_reactor:
                rect = self._create_vehicle_patch(ax, agent_x, agent_y, agent_yaw,
                                                  agent_length, agent_width, color)
                ax.add_patch(rect)
                
                # Label influencer and reactor
                if is_influencer:
                    ax.annotate('Influencer', (agent_x, agent_y),
                               textcoords="offset points", xytext=(5, 10),
                               fontsize=10, fontweight='bold', color=color, zorder=10)
                elif is_reactor:
                    ax.annotate('Reactor', (agent_x, agent_y),
                               textcoords="offset points", xytext=(5, 10),
                               fontsize=10, fontweight='bold', color=color, zorder=10)
        
        # Draw ground truth trajectories
        # Influencer history (red dashed)
        inf_history = states[influencer_idx, :history_steps]
        inf_hist_valid = valid[influencer_idx, :history_steps]
        if inf_hist_valid.any():
            inf_history = inf_history[inf_hist_valid]
            if len(inf_history) > 1:
                ax.plot(inf_history[:, 0], inf_history[:, 1],
                       color=self.COLORS['history_influencer'], linewidth=2.5,
                       linestyle='--', alpha=0.6, zorder=3)
        
        # Reactor history (orange dashed)
        react_history = states[reactor_idx, :history_steps]
        react_hist_valid = valid[reactor_idx, :history_steps]
        if react_hist_valid.any():
            react_history = react_history[react_hist_valid]
            if len(react_history) > 1:
                ax.plot(react_history[:, 0], react_history[:, 1],
                       color=self.COLORS['history_reactor'], linewidth=2.5,
                       linestyle='--', alpha=0.6, zorder=3)
        
        # Influencer GT future (gray dotted)
        inf_future = states[influencer_idx, history_steps:]
        inf_fut_valid = valid[influencer_idx, history_steps:]
        if inf_fut_valid.any():
            inf_future = inf_future[inf_fut_valid]
            if len(inf_future) > 1:
                ax.plot(inf_future[:, 0], inf_future[:, 1],
                       color=self.COLORS['future_gt'], linewidth=2,
                       linestyle=':', alpha=0.5, zorder=2)
        
        # Reactor GT future (gray dotted)
        react_future = states[reactor_idx, history_steps:]
        react_fut_valid = valid[reactor_idx, history_steps:]
        if react_fut_valid.any():
            react_future = react_future[react_fut_valid]
            if len(react_future) > 1:
                ax.plot(react_future[:, 0], react_future[:, 1],
                       color=self.COLORS['future_gt'], linewidth=2,
                       linestyle=':', alpha=0.5, zorder=2)
        
        # Draw predicted trajectories
        # Influencer (from marginal, best mode)
        if inf_idx_in_marg >= 0:
            inf_traj = marginal_data['rst'][inf_idx_in_marg].copy()
            inf_scores = marginal_data['score'][inf_idx_in_marg]
            inf_traj = inf_traj + pred_offset_influencer
            
            best_mode = np.argmax(inf_scores)
            inf_pred = inf_traj[best_mode]
            ax.plot(inf_pred[:, 0], inf_pred[:, 1],
                   color=self.COLORS['pred_influencer'], linewidth=4,
                   linestyle='-', alpha=0.9, zorder=4)
            ax.scatter(inf_pred[-1, 0], inf_pred[-1, 1],
                      c=self.COLORS['pred_influencer'], s=150, marker='*',
                      edgecolors='white', linewidths=1.5, zorder=5)
        
        # Reactor (6 conditional modes)
        if len(conditional_data['rst']) > 0:
            reactor_trajs = conditional_data['rst'][0].copy()
            reactor_scores = conditional_data['score'][0]
            reactor_trajs = reactor_trajs + pred_offset_reactor
            
            scores_norm = np.exp(reactor_scores - np.max(reactor_scores))
            scores_norm = scores_norm / np.sum(scores_norm)
            
            for mode in range(6):
                traj = reactor_trajs[mode]
                score = scores_norm[mode]
                
                color = plt.cm.plasma(mode / 5)
                alpha = 0.4 + 0.6 * score
                linewidth = 2 + 4 * score
                
                ax.plot(traj[:, 0], traj[:, 1],
                       color=color, linewidth=linewidth,
                       alpha=alpha, zorder=4)
                
                ax.scatter(traj[-1, 0], traj[-1, 1],
                          c=[color], s=80 + 80*score, marker='*',
                          edgecolors='white', linewidths=0.5,
                          alpha=alpha, zorder=5)
        
        # Title
        relation_str = ['Smaller ID → Influencer', 'Larger ID → Influencer', 'No Interaction'][relation_label]
        ax.set_title(f'M2I Prediction: {scenario_id}\nRelation: {relation_str}',
                    fontsize=14, fontweight='bold')
        
        # Comprehensive legend (matching BEV style)
        legend_elements = [
            mpatches.Patch(facecolor=self.COLORS['influencer'], edgecolor='black', label='Influencer'),
            mpatches.Patch(facecolor=self.COLORS['reactor'], edgecolor='black', label='Reactor'),
            mpatches.Patch(facecolor=self.COLORS['vehicle'], edgecolor='black', label='Other Vehicles'),
            mpatches.Patch(facecolor=self.COLORS['pedestrian'], edgecolor='black', label='Pedestrians'),
            Line2D([0], [0], color=self.COLORS['history_influencer'], linestyle='--', linewidth=2, label='History'),
            Line2D([0], [0], color=self.COLORS['future_gt'], linestyle=':', linewidth=2, label='Ground Truth'),
            Line2D([0], [0], color=self.COLORS['pred_influencer'], linestyle='-', linewidth=3, label='Influencer Pred'),
            mpatches.Patch(facecolor='purple', alpha=0.7, label='Reactor Modes (6)'),
            Line2D([0], [0], color=self.COLORS['lane'], linestyle='-', linewidth=1.5, label='Lanes'),
            Line2D([0], [0], color=self.COLORS['road_edge'], linestyle='-', linewidth=2, label='Road Edges'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=9,
                 framealpha=0.9, ncol=1)
        
        ax.set_xlabel('X (meters)', fontsize=11)
        ax.set_ylabel('Y (meters)', fontsize=11)
        
        # Save
        output_path = self.output_dir / f"{scenario_id}.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ Saved: {output_path}")
        
        return output_path


def main():
    parser = argparse.ArgumentParser(description='Generate M2I prediction visualizations (BEV style)')
    parser.add_argument('--data_dir', type=str,
                       default='/workspace/data/datasets/waymo_open_dataset/motion_v_1_3_0/processed/tf/validation_interactive',
                       help='Directory containing TFRecord files')
    parser.add_argument('--predictions_dir', type=str,
                       default='/workspace/output/m2i_live',
                       help='Directory containing M2I predictions')
    parser.add_argument('--output_dir', type=str,
                       default='/workspace/models/pretrained/m2i/movies/m2i_predictions',
                       help='Output directory for movies')
    parser.add_argument('--num_scenarios', type=int, default=5,
                       help='Number of scenarios to visualize')
    parser.add_argument('--mode', type=str, choices=['movie', 'image', 'both'], default='both',
                       help='Visualization mode')
    parser.add_argument('--fps', type=int, default=10,
                       help='Frames per second for movies')
    parser.add_argument('--dpi', type=int, default=100,
                       help='DPI for output images/movies')
    parser.add_argument('--view_range', type=float, default=50.0,
                       help='View range in meters around agent (for follow mode)')
    parser.add_argument('--follow_agent', action='store_true', default=True,
                       help='Camera follows the influencer agent')
    parser.add_argument('--no_follow', action='store_true',
                       help='Disable follow mode (fixed viewport)')
    parser.add_argument('--show_all_agents', action='store_true', default=True,
                       help='Show all agents in scene (not just interacting pair)')
    parser.add_argument('--interacting_only', action='store_true',
                       help='Show only the interacting pair')
    
    args = parser.parse_args()
    
    # Handle flag conflicts
    follow_agent = not args.no_follow
    show_all_agents = not args.interacting_only
    
    # Paths
    predictions_dir = Path(args.predictions_dir)
    marginal_path = predictions_dir / 'marginal' / 'marginal_predictions.pickle'
    relation_path = predictions_dir / 'relation' / 'relation_predictions.pickle'
    conditional_path = predictions_dir / 'conditional' / 'conditional_predictions.pickle'
    
    # Check files exist
    for p in [marginal_path, relation_path, conditional_path]:
        if not p.exists():
            print(f"Error: {p} not found. Run m2i_live_inference.py first.")
            return
    
    # Create generator with BEV-style parameters
    generator = M2IMovieGenerator(
        data_dir=Path(args.data_dir),
        marginal_path=marginal_path,
        relation_path=relation_path,
        conditional_path=conditional_path,
        output_dir=Path(args.output_dir),
        view_range=args.view_range,
        fps=args.fps,
        dpi=args.dpi
    )
    
    # Generate visualizations
    scenarios = list(generator.valid_scenarios)[:args.num_scenarios]
    print(f"\nGenerating visualizations for {len(scenarios)} scenarios...")
    print(f"  Follow agent: {follow_agent}")
    print(f"  Show all agents: {show_all_agents}")
    
    for scenario_id in scenarios:
        if args.mode in ['image', 'both']:
            generator.generate_static_visualization(scenario_id, show_all_agents=show_all_agents)
        if args.mode in ['movie', 'both']:
            generator.generate_scenario_movie(scenario_id, fps=args.fps,
                                              follow_agent=follow_agent,
                                              show_all_agents=show_all_agents)
    
    print(f"\nDone! Outputs saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
