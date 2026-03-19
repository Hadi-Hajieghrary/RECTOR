#!/usr/bin/env python3
"""
M2I Full Receding Horizon Inference
====================================

This script implements receding horizon trajectory prediction using the M2I 
(Multi-agent Interaction) model, simulating real-time autonomous vehicle 
operation where predictions are continuously updated as new observations 
become available.

Overview
--------
In autonomous driving, trajectory prediction must be performed continuously
as the vehicle moves through the environment. This "receding horizon" approach:

1. At each timestep t, uses the past 1 second of observations (10 frames at 10Hz)
2. Predicts future trajectories for the next 8 seconds (80 frames)
3. Advances to timestep t+1 and repeats with updated observations

This allows analysis of how prediction quality evolves with more observations
and how the model adapts to changing agent behaviors.

Architecture
------------
The implementation includes four key components:

1. **BEV Raster Rendering** (`_render_bev_raster`)
   - Renders road network onto a 224×224×60 bird's-eye-view image
   - Channel layout:
     * Channels 0-10: Ego agent trajectory history
     * Channels 20-30: Other agents' trajectories
     * Channels 40-59: Road lane types (20 different types)
   - Coordinates are transformed to agent-centric frame using normalizer

2. **Road Network Vectors** (`_create_road_vectors`)
   - Creates 128-dimensional polyline vectors for VectorNet encoding
   - Groups road points by lane ID for proper polyline structure
   - Vector format:
     * vec[0:2] = start point (x, y) × scale
     * vec[20:22] = end point (x, y) × scale
     * vec[50+lane_type] = 1 (one-hot encoding)
   - Uses scale=0.03 matching M2I's 'stride_10_2' configuration

3. **Receding Horizon Prediction** (`run_receding_horizon`)
   - Parses complete 9.1-second scenarios (91 timesteps at 10Hz)
   - Runs DenseTNT inference at each timestep from start_t to end_t
   - Computes ADE/FDE metrics against ground truth
   - Supports configurable step size for temporal resolution

4. **Movie Visualization** (`RecedingHorizonVisualizer`)
   - Generates MP4 movies showing predictions evolving over time
   - Each frame displays:
     * Gray road network points
     * Colored agent history trajectories (solid lines)
     * Ground truth future (dashed lines)
     * Predicted trajectories (solid with star endpoint)
     * Current position markers (circles)
   - Uses 10 FPS to match Waymo dataset temporal resolution

Coordinate Systems
------------------
- **World coordinates**: Global Waymo coordinate frame (meters)
- **Agent-centric**: Rotated/translated so agent is at origin facing +Y
- **Raster pixels**: x_pix = x_local * scale + 112, y_pix = y_local * scale + 56

Model Configuration
-------------------
- DenseTNT with raster CNN encoder (model.24.bin, 270 weights)
- Hidden size: 128
- Future prediction: 80 frames (8 seconds)
- Mode number: 6 trajectory hypotheses
- NMS threshold: 7.2 meters

Usage Examples
--------------
Basic usage with DenseTNT only:
    python m2i_receding_horizon_full.py \\
        --num_scenarios 10 \\
        --start_t 10 --end_t 90 --step 1 \\
        --generate_movies

With custom TFRecord file:
    python m2i_receding_horizon_full.py \\
        --tfrecord /path/to/validation.tfrecord \\
        --num_scenarios 5 \\
        --generate_movies

Command Line Arguments
----------------------
--tfrecord      : Path to TFRecord file (default: validation_interactive)
--num_scenarios : Number of scenarios to process (default: 3)
--start_t       : First prediction timestep, must be >= 10 (default: 10)
--end_t         : Last prediction timestep, must be <= 90 (default: 30)
--step          : Timestep increment (default: 5, use 1 for full resolution)
--output        : Output path for predictions pickle file
--movies_dir    : Output directory for visualization movies
--generate_movies : Flag to enable movie generation
--device        : Device for inference (cuda or cpu)
--full-pipeline : Enable 3-stage M2I (DenseTNT + Relation + Conditional) - DEPRECATED
--subprocess-pipeline : Enable 3-stage M2I with subprocess isolation (RECOMMENDED)

Pipeline Modes
--------------
1. **DenseTNT Only** (default):
   - Runs independent trajectory prediction for all agents
   - Fast and reliable, no model interference issues
   - Recommended for general use

2. **Subprocess Pipeline** (--subprocess-pipeline):
   - Stage 1: DenseTNT in main process
   - Stage 2: Relation model in subprocess (subprocess_relation.py)
   - Stage 3: Conditional model in subprocess (subprocess_conditional.py)
   - Avoids global args conflicts in M2I's VectorNet implementation
   - Relation stage correctly identifies influencer/reactor relationships

3. **Full Pipeline** (--full-pipeline, DEPRECATED):
   - All models in same process
   - Suffers from global args conflicts
   - Use subprocess-pipeline instead

Output Format
-------------
The predictions are saved as a pickle file with structure:
{
    scenario_idx: {
        'scenario_id': str,
        'predictions': {
            timestep: {
                'timestep': int,
                'agents': {
                    agent_id: {
                        'pred_traj': np.array [6, 80, 2],  # 6 modes, 80 frames, xy
                        'pred_scores': np.array [6],       # confidence scores
                        'current_pos': np.array [2],       # current position
                    }
                },
                'ground_truth': {
                    agent_id: {
                        'current_pos': np.array [2],
                        'future_traj': np.array [T, 2],
                    }
                }
            }
        },
        'metadata': {...}
    }
}

Notes
-----
- Waymo Open Dataset scenarios are 9.1 seconds (91 timesteps at 10Hz)
- Predictions require 10 history frames, so earliest start is t=10
- Movies at 10 FPS match real-time playback of Waymo scenarios
- The full 3-stage M2I pipeline (--full-pipeline) is implemented but requires
  careful handling of global state in the M2I codebase

Author: RECTOR Project
Date: December 2025
"""

import argparse
import os
import pickle
import sys
import time
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch
import torch.nn as nn

# Environment setup
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')

# Add paths
M2I_SRC = Path("/workspace/externals/M2I/src")
sys.path.insert(0, str(M2I_SRC))
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

import tensorflow as tf


class OtherParams(list):
    """Hybrid list/dict for M2I's other_params."""
    def __init__(self, items=None, defaults=None):
        super().__init__(items or [])
        self._defaults = defaults or {}
        self._removed = set()
    
    def get(self, key, default=None):
        if key in self:
            return True
        return self._defaults.get(key, default)
    
    def __getitem__(self, key):
        if isinstance(key, int):
            return super().__getitem__(key)
        if key in self:
            return True
        if key in self._defaults:
            return self._defaults[key]
        raise KeyError(key)
    
    def remove(self, item):
        self._removed.add(item)


class RecedingHorizonM2I:
    """
    Full M2I receding horizon inference pipeline.
    
    Runs DenseTNT predictions at multiple timesteps, creating a sequence
    of predictions that incorporate increasingly recent observations.
    
    Optionally runs Relation and Conditional stages at each timestep.
    """
    
    def __init__(self, device: str = 'cuda', enable_relation: bool = False, enable_conditional: bool = False):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.relation_model = None
        self.conditional_model = None
        self.args = None
        self.relation_args = None
        self.conditional_args = None
        
        # Enable 3-stage pipeline
        self.enable_relation = enable_relation
        self.enable_conditional = enable_conditional
        
        # Model paths - navigate up from scripts/lib to pretrained/m2i/models
        M2I_ROOT = SCRIPT_DIR.parent.parent  # /workspace/models/pretrained/m2i
        self.densetnt_path = M2I_ROOT / 'models' / 'densetnt' / 'model.24.bin'
        self.relation_path = M2I_ROOT / 'models' / 'relation_v2v' / 'model.25.bin'
        self.conditional_path = M2I_ROOT / 'models' / 'conditional_v2v' / 'model.29.bin'
        
        print(f"Device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    def _set_seeds(self, seed: int = 42):
        """Set random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def _build_args(self):
        """Build args for DenseTNT model."""
        import utils
        
        args = utils.Args()
        args.hidden_size = 128
        args.future_frame_num = 80
        args.future_test_frame_num = 80
        args.mode_num = 6
        args.sub_graph_batch_size = 4096
        args.train_batch_size = 1
        args.eval_batch_size = 1
        args.core_num = 1
        args.nms_threshold = 7.2
        args.agent_type = 'vehicle'
        args.inter_agent_types = None
        args.single_agent = True
        args.infMLP = 0
        
        # Architecture
        args.sub_graph_depth = 3
        args.global_graph_depth = 1
        args.hidden_dropout_prob = 0.1
        args.initializer_range = 0.02
        args.max_distance = 50.0
        args.no_sub_graph = False
        args.no_agents = False
        args.attention_decay = False
        args.use_map = False
        args.old_version = False
        args.lstm = False
        args.visualize = False
        args.no_cuda = False
        args.waymo = True
        args.nuscenes = False
        args.argoverse = False
        args.do_train = False
        args.do_eval = True
        args.do_test = False
        
        # Goal classification
        args.classify_sub_goals = False
        
        # Eval params list (used by utils.add_eval_param)
        args.eval_params = []
        
        # For raster generation - must be an actual array, not None, 
        # or get_agents will segfault
        args.image = np.zeros([224, 224, 60], dtype=np.int8)
        
        # Logging directories (needed by utils.logging)
        output_dir = Path('/workspace/output/m2i_live/receding_horizon')
        output_dir.mkdir(parents=True, exist_ok=True)
        args.log_dir = str(output_dir / 'logs')
        args.output_dir = str(output_dir)
        args.temp_file_dir = str(output_dir / 'temp')
        os.makedirs(args.log_dir, exist_ok=True)
        os.makedirs(args.temp_file_dir, exist_ok=True)
        
        # Other params - MUST include raster for pretrained model
        # NOTE: Do NOT include 'train_pair_interest' or 'train_reactor' for marginal predictions
        # Those modes expect gt_influencer_traj_idx which we don't have for independent predictions
        args.other_params = OtherParams([
            'l1_loss',
            'densetnt',
            'goals_2D',
            'enhance_global_graph',
            'laneGCN',
            'point_sub_graph',
            'laneGCN-4',
            'stride_10_2',
            'raster',
        ], {'eval_time': 80})
        
        utils.args = args
        return args
    
    def load_model(self):
        """Load DenseTNT model."""
        from modeling.vectornet import VectorNet
        
        print(f"\nLoading DenseTNT model...")
        print(f"  Model path: {self.densetnt_path}")
        
        if not self.densetnt_path.exists():
            raise FileNotFoundError(f"Model not found: {self.densetnt_path}")
        
        self.args = self._build_args()
        self.model = VectorNet(self.args)
        
        # Load weights (checkpoint is a direct OrderedDict)
        checkpoint = torch.load(self.densetnt_path, map_location='cpu', weights_only=False)
        model_dict = self.model.state_dict()
        
        loaded = 0
        for key, value in checkpoint.items():
            clean_key = key.replace('module.', '') if key.startswith('module.') else key
            if clean_key in model_dict and model_dict[clean_key].shape == value.shape:
                model_dict[clean_key] = value
                loaded += 1
        
        self.model.load_state_dict(model_dict)
        self.model.to(self.device)
        self.model.eval()
        print(f"  Loaded {loaded}/{len(checkpoint)} weights")
        
        # Load Relation model if enabled
        if self.enable_relation and self.relation_path.exists():
            self._load_relation_model()
        
        # Load Conditional model if enabled
        if self.enable_conditional and self.conditional_path.exists():
            self._load_conditional_model()
    
    def _build_relation_args(self):
        """Build args for Relation V2V model."""
        import utils
        
        args = utils.Args()
        args.hidden_size = 128
        args.future_frame_num = 80
        args.future_test_frame_num = 80
        args.mode_num = 6
        args.nms_threshold = 7.0
        args.use_centerline = False
        args.waymo = True
        args.nuscenes = False
        args.argoverse = False
        args.do_train = False
        args.do_eval = True
        args.do_test = False
        args.classify_sub_goals = False
        args.eval_params = []
        args.infMLP = 0
        args.image = None  # Relation doesn't use raster
        
        # Architecture params (required by VectorNet)
        args.sub_graph_depth = 3
        args.global_graph_depth = 1
        args.hidden_dropout_prob = 0.1
        args.initializer_range = 0.02
        args.max_distance = 50.0
        args.no_sub_graph = False
        args.no_agents = False
        args.attention_decay = False
        args.use_map = False
        args.old_version = False
        args.lstm = False
        args.visualize = False
        args.no_cuda = False
        args.single_agent = False  # Relation uses pairs
        args.agent_type = 'vehicle'
        args.inter_agent_types = None
        
        # Logging directories
        output_dir = Path('/workspace/output/m2i_live/receding_horizon')
        args.log_dir = str(output_dir / 'logs')
        args.output_dir = str(output_dir)
        args.temp_file_dir = str(output_dir / 'temp')
        
        # Relation-specific params
        args.other_params = OtherParams([
            'l1_loss',
            'densetnt',
            'goals_2D',
            'enhance_global_graph',
            'laneGCN',
            'point_sub_graph',
            'laneGCN-4',
            'stride_10_2',
            'train_pair_interest',
            'train_relation',
        ], {'eval_time': 80})
        
        return args
    
    def _build_conditional_args(self):
        """Build args for Conditional V2V model."""
        import utils
        
        args = utils.Args()
        args.hidden_size = 128
        args.future_frame_num = 80
        args.future_test_frame_num = 80
        args.mode_num = 6
        args.nms_threshold = 7.0
        args.use_centerline = False
        args.waymo = True
        args.nuscenes = False
        args.argoverse = False
        args.do_train = False
        args.do_eval = True
        args.do_test = False
        args.classify_sub_goals = False
        args.eval_params = []
        args.infMLP = 6
        args.inf_pred_num = 6
        # Conditional model REQUIRES raster for influencer encoding
        # The image will be populated per-inference with influencer trajectory in channels 60+
        args.image = np.zeros([224, 224, 150], dtype=np.int8)
        
        # Architecture params (required by VectorNet)
        args.sub_graph_depth = 3
        args.global_graph_depth = 1
        args.hidden_dropout_prob = 0.1
        args.initializer_range = 0.02
        args.max_distance = 50.0
        args.no_sub_graph = False
        args.no_agents = False
        args.attention_decay = False
        args.use_map = False
        args.old_version = False
        args.lstm = False
        args.visualize = False
        args.no_cuda = False
        args.single_agent = False  # Conditional uses reactor
        args.agent_type = 'vehicle'
        args.inter_agent_types = None
        
        # Logging directories
        output_dir = Path('/workspace/output/m2i_live/receding_horizon')
        args.log_dir = str(output_dir / 'logs')
        args.output_dir = str(output_dir)
        args.temp_file_dir = str(output_dir / 'temp')
        
        # Conditional-specific params - MUST include 'raster' and 'raster_inf'
        # for the CNN encoder to process influencer trajectory
        args.other_params = OtherParams([
            'l1_loss',
            'densetnt',
            'goals_2D',
            'enhance_global_graph',
            'laneGCN',
            'point_sub_graph',
            'laneGCN-4',
            'stride_10_2',
            'train_pair_interest',
            'train_reactor',
            'raster',      # Enable raster CNN encoder
            'raster_inf',  # Enable influencer trajectory rasterization
        ], {'eval_time': 80})
        
        return args
    
    def _load_relation_model(self):
        """Load Relation V2V model."""
        from modeling.vectornet import VectorNet
        
        print(f"Loading Relation V2V model...")
        self.relation_args = self._build_relation_args()
        self.relation_model = VectorNet(self.relation_args)
        
        checkpoint = torch.load(self.relation_path, map_location='cpu')
        model_dict = self.relation_model.state_dict()
        
        loaded = 0
        for key, value in checkpoint.items():
            clean_key = key.replace('module.', '') if key.startswith('module.') else key
            if clean_key in model_dict and model_dict[clean_key].shape == value.shape:
                model_dict[clean_key] = value
                loaded += 1
        
        self.relation_model.load_state_dict(model_dict)
        self.relation_model.to(self.device)
        self.relation_model.eval()
        print(f"  Loaded {loaded}/{len(checkpoint)} weights (Relation)")
    
    def _load_conditional_model(self):
        """Load Conditional V2V model."""
        from modeling.vectornet import VectorNet
        
        print(f"Loading Conditional V2V model...")
        self.conditional_args = self._build_conditional_args()
        self.conditional_model = VectorNet(self.conditional_args)
        
        checkpoint = torch.load(self.conditional_path, map_location='cpu')
        model_dict = self.conditional_model.state_dict()
        
        loaded = 0
        for key, value in checkpoint.items():
            clean_key = key.replace('module.', '') if key.startswith('module.') else key
            if clean_key in model_dict and model_dict[clean_key].shape == value.shape:
                model_dict[clean_key] = value
                loaded += 1
        
        self.conditional_model.load_state_dict(model_dict)
        self.conditional_model.to(self.device)
        self.conditional_model.eval()
        print(f"  Loaded {loaded}/{len(checkpoint)} weights (Conditional)")
    
    def _generate_raster_goals_grid(self, raster_scale: int = 1) -> np.ndarray:
        """
        Generate goals_2D grid for raster mode.
        
        In raster mode, each pixel in the 224×224 raster image is a candidate goal.
        The coordinate system is:
          - x = (pixel_i - 112) / scale  (ranges from -112 to +111)
          - y = (pixel_j - 56) / scale   (ranges from -56 to +167)
        
        Args:
            raster_scale: Scaling factor (default 1 for full resolution)
            
        Returns:
            goals_2D: Array of shape [N, 2] where N = (224/scale)^2
        """
        grid_size = 224
        goals_list = []
        
        for i in range(0, grid_size, raster_scale):
            x_float = (i - 112) / float(raster_scale)
            for j in range(0, grid_size, raster_scale):
                y_float = (j - 56) / float(raster_scale)
                goals_list.append([x_float, y_float])
        
        return np.array(goals_list, dtype=np.float32)
    
    def _render_bev_raster(
        self,
        roadgraph_xyz: np.ndarray,
        roadgraph_type: np.ndarray,
        roadgraph_valid: np.ndarray,
        normalizer,
        raster_scale: int = 1,
    ) -> np.ndarray:
        """
        Render a bird's-eye-view raster image with road lanes.
        
        Raster channel layout (60 channels):
          - Channels 0-10:  Ego agent history trajectory (filled by get_agents)
          - Channels 20-30: Other agents history (filled by get_agents)
          - Channels 33-39: Traffic lights (not implemented here)
          - Channels 40-59: Road lane types (0-19 lane types → channels 40-59)
        
        Args:
            roadgraph_xyz: Road points [N, 3] in world coordinates
            roadgraph_type: Road point types [N]
            roadgraph_valid: Valid mask [N]
            normalizer: Coordinate normalizer (has x, y, yaw)
            raster_scale: Pixel scale (default 1)
            
        Returns:
            image: [224, 224, 60] int8 raster image
        """
        import math
        
        image = np.zeros([224, 224, 60], dtype=np.int8)
        
        # Coordinate transform params
        cent_x = normalizer.x
        cent_y = normalizer.y
        angle = normalizer.yaw
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        
        # Helper: world to local normalized coordinates
        def world_to_local(x, y):
            dx = x - cent_x
            dy = y - cent_y
            x_local = dx * cos_a - dy * sin_a
            y_local = dx * sin_a + dy * cos_a
            return x_local, y_local
        
        # Helper: local coords to raster pixel
        def local_to_pixel(x_local, y_local):
            x_pix = int(x_local * raster_scale + 112 + 0.5)
            y_pix = int(y_local * raster_scale + 56 + 0.5)
            return x_pix, y_pix
        
        # Render road lanes
        max_dist = 80.0  # meters
        for i in range(len(roadgraph_xyz)):
            if not roadgraph_valid[i]:
                continue
            
            x_world = roadgraph_xyz[i, 0]
            y_world = roadgraph_xyz[i, 1]
            x_local, y_local = world_to_local(x_world, y_world)
            
            # Skip points too far away
            dist = math.sqrt(x_local**2 + (y_local - 30)**2)
            if dist > max_dist:
                continue
            
            x_pix, y_pix = local_to_pixel(x_local, y_local)
            
            # Check bounds
            if 0 <= x_pix < 224 and 0 <= y_pix < 224:
                lane_type = int(roadgraph_type[i])
                if 0 <= lane_type < 20:
                    image[x_pix, y_pix, 40 + lane_type] = 1
        
        return image
    
    def _create_road_vectors(
        self,
        roadgraph_xyz: np.ndarray,
        roadgraph_type: np.ndarray,
        roadgraph_valid: np.ndarray,
        roadgraph_id: np.ndarray,
        normalizer,
    ) -> Tuple[np.ndarray, List]:
        """
        Create road polyline vectors from roadgraph data.
        
        Similar to M2I's get_roads but in pure Python for compatibility.
        
        Args:
            roadgraph_xyz: Road points [N, 3] in world coordinates
            roadgraph_type: Road point types [N]
            roadgraph_valid: Valid mask [N]
            roadgraph_id: Lane IDs [N]
            normalizer: Coordinate normalizer
            
        Returns:
            vectors: [M, 128] road polyline vectors
            polyline_spans: List of slices for each polyline
        """
        import math
        
        cent_x = normalizer.x
        cent_y = normalizer.y
        angle = normalizer.yaw
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        scale = 0.03  # Same scale as M2I uses with 'stride_10_2'
        
        max_dist = 80.0
        visible_y = 30.0
        
        # Group points by lane ID
        lane_points = {}
        for i in range(len(roadgraph_xyz)):
            if not roadgraph_valid[i]:
                continue
            
            x_world = roadgraph_xyz[i, 0]
            y_world = roadgraph_xyz[i, 1]
            
            # Transform to local coords
            dx = x_world - cent_x
            dy = y_world - cent_y
            x_local = dx * cos_a - dy * sin_a
            y_local = dx * sin_a + dy * cos_a
            
            # Distance filter
            dist = math.sqrt(x_local**2 + (y_local - visible_y)**2)
            if dist > max_dist:
                continue
            
            lane_id = int(roadgraph_id[i])
            lane_type = int(roadgraph_type[i])
            
            if lane_id not in lane_points:
                lane_points[lane_id] = []
            lane_points[lane_id].append((x_local, y_local, lane_type))
        
        # Create vectors for each lane
        vectors_list = []
        polyline_spans = []
        
        stride = 10  # Sample every 10 points
        
        for lane_id, points in lane_points.items():
            if len(points) < 2:
                continue
            
            start_idx = len(vectors_list)
            
            # Sample points along lane
            for c in range(0, len(points) - 1, stride):
                p1 = points[c]
                p2 = points[min(c + stride, len(points) - 1)]
                
                # Create 128-dim vector
                vec = np.zeros(128, dtype=np.float32)
                
                # Start point (scaled)
                vec[0] = p1[0] * scale
                vec[1] = p1[1] * scale
                
                # End point (scaled)
                vec[20] = p2[0] * scale
                vec[21] = p2[1] * scale
                
                # Lane type one-hot (channels 50+)
                lane_type = p1[2]
                if 0 <= lane_type < 20:
                    vec[50 + lane_type] = 1
                
                vectors_list.append(vec)
            
            if len(vectors_list) > start_idx:
                polyline_spans.append(slice(start_idx, len(vectors_list)))
        
        if len(vectors_list) == 0:
            return np.zeros((0, 128), dtype=np.float32), []
        
        return np.array(vectors_list, dtype=np.float32), polyline_spans
    
    def parse_full_scenario(self, tfrecord_path: str, scenario_idx: int = 0) -> Optional[Dict]:
        """
        Parse a scenario and return all 91 timesteps of data.
        
        Returns complete scenario data that can be sliced for any prediction timestep.
        """
        from waymo_flat_parser import features_description_flat
        
        dataset = tf.data.TFRecordDataset(tfrecord_path)
        
        for idx, record in enumerate(dataset):
            if idx != scenario_idx:
                continue
            
            # Parse the record
            parsed = tf.io.parse_single_example(record, features_description_flat)
            
            # Helper to convert to numpy
            def to_np(key):
                return parsed[key].numpy()
            
            n_agents = 128
            n_past = 10
            n_future = 80
            
            # Get all state data and reshape
            # Past: [128*10] -> [128, 10], stored most recent first, so reverse
            past_x = to_np('state/past/x').reshape(n_agents, n_past)[:, ::-1]
            past_y = to_np('state/past/y').reshape(n_agents, n_past)[:, ::-1]
            past_vx = to_np('state/past/velocity_x').reshape(n_agents, n_past)[:, ::-1]
            past_vy = to_np('state/past/velocity_y').reshape(n_agents, n_past)[:, ::-1]
            past_yaw = to_np('state/past/bbox_yaw').reshape(n_agents, n_past)[:, ::-1]
            past_valid = to_np('state/past/valid').reshape(n_agents, n_past)[:, ::-1]
            past_length = to_np('state/past/length').reshape(n_agents, n_past)[:, ::-1]
            past_width = to_np('state/past/width').reshape(n_agents, n_past)[:, ::-1]
            
            # Current: [128]
            current_x = to_np('state/current/x').reshape(n_agents, 1)
            current_y = to_np('state/current/y').reshape(n_agents, 1)
            current_vx = to_np('state/current/velocity_x').reshape(n_agents, 1)
            current_vy = to_np('state/current/velocity_y').reshape(n_agents, 1)
            current_yaw = to_np('state/current/bbox_yaw').reshape(n_agents, 1)
            current_valid = to_np('state/current/valid').reshape(n_agents, 1)
            current_length = to_np('state/current/length').reshape(n_agents, 1)
            current_width = to_np('state/current/width').reshape(n_agents, 1)
            
            # Future: [128*80] -> [128, 80]
            future_x = to_np('state/future/x').reshape(n_agents, n_future)
            future_y = to_np('state/future/y').reshape(n_agents, n_future)
            future_vx = to_np('state/future/velocity_x').reshape(n_agents, n_future)
            future_vy = to_np('state/future/velocity_y').reshape(n_agents, n_future)
            future_yaw = to_np('state/future/bbox_yaw').reshape(n_agents, n_future)
            future_valid = to_np('state/future/valid').reshape(n_agents, n_future)
            future_length = to_np('state/future/length').reshape(n_agents, n_future)
            future_width = to_np('state/future/width').reshape(n_agents, n_future)
            
            # Stack all timesteps: [128, 91, ...]
            all_x = np.concatenate([past_x, current_x, future_x], axis=1)
            all_y = np.concatenate([past_y, current_y, future_y], axis=1)
            all_vx = np.concatenate([past_vx, current_vx, future_vx], axis=1)
            all_vy = np.concatenate([past_vy, current_vy, future_vy], axis=1)
            all_yaw = np.concatenate([past_yaw, current_yaw, future_yaw], axis=1)
            all_valid = np.concatenate([past_valid, current_valid, future_valid], axis=1)
            all_length = np.concatenate([past_length, current_length, future_length], axis=1)
            all_width = np.concatenate([past_width, current_width, future_width], axis=1)
            
            # Road graph (xyz is flat: 90000 = 30000 * 3)
            roadgraph_xyz = to_np('roadgraph_samples/xyz').reshape(-1, 3)  # [30000, 3]
            roadgraph_type = to_np('roadgraph_samples/type')  # [30000]
            roadgraph_valid = to_np('roadgraph_samples/valid')  # [30000]
            roadgraph_id = to_np('roadgraph_samples/id')  # [30000]
            
            # Metadata
            scenario_id = to_np('scenario/id')
            agent_types = to_np('state/type')
            agent_ids = to_np('state/id')
            tracks_to_predict_mask = to_np('state/tracks_to_predict')
            objects_of_interest = to_np('state/objects_of_interest')
            
            return {
                # Full trajectories [128, 91]
                'x': all_x,
                'y': all_y,
                'vx': all_vx,
                'vy': all_vy,
                'yaw': all_yaw,
                'valid': all_valid,
                'length': all_length,
                'width': all_width,
                
                # Roadgraph
                'roadgraph_xyz': roadgraph_xyz,
                'roadgraph_type': roadgraph_type,
                'roadgraph_valid': roadgraph_valid,
                'roadgraph_id': roadgraph_id,
                
                # Metadata
                'scenario_id': scenario_id,
                'agent_types': agent_types,
                'agent_ids': agent_ids,
                'tracks_to_predict': tracks_to_predict_mask,
                'objects_of_interest': objects_of_interest,
            }
        
        return None
    
    def create_mapping_at_timestep(
        self,
        scenario_data: Dict,
        current_t: int,
        agent_idx: int,
    ) -> Optional[Dict]:
        """
        Create M2I-compatible mapping for a specific agent at a specific timestep.
        
        Uses M2I's utils_cython functions to create proper 128-dim vectors.
        
        Args:
            scenario_data: Full scenario data from parse_full_scenario
            current_t: Current timestep (10 <= t <= 80 for valid windows)
            agent_idx: Which agent to create mapping for
        
        Returns:
            M2I mapping dict compatible with VectorNet model
        """
        import math
        import utils
        import utils_cython
        from utils_cython import get_normalized
        Normalizer = utils.Normalizer
        
        n_history = 11  # 10 past + 1 current  
        n_future = 80
        
        # Check valid prediction window
        if current_t < 10 or current_t > 80:
            return None
        
        # Extract data for shifted time window
        # History: [current_t - 10, current_t] = 11 timesteps
        # Future: [current_t + 1, current_t + 80] = 80 timesteps
        
        hist_start = current_t - 10
        hist_end = current_t + 1  # exclusive
        future_start = current_t + 1
        future_end = min(current_t + 81, 91)
        
        # Get agent data for this window
        x = scenario_data['x'][:, hist_start:hist_end]
        y = scenario_data['y'][:, hist_start:hist_end]
        vx = scenario_data['vx'][:, hist_start:hist_end]
        vy = scenario_data['vy'][:, hist_start:hist_end]
        yaw = scenario_data['yaw'][:, hist_start:hist_end]
        valid = scenario_data['valid'][:, hist_start:hist_end]
        length = scenario_data['length'][:, hist_start:hist_end]
        width = scenario_data['width'][:, hist_start:hist_end]
        
        # Future for labels
        future_x = scenario_data['x'][:, future_start:future_end]
        future_y = scenario_data['y'][:, future_start:future_end]
        future_yaw = scenario_data['yaw'][:, future_start:future_end]
        future_vx = scenario_data['vx'][:, future_start:future_end]
        future_vy = scenario_data['vy'][:, future_start:future_end]
        future_length = scenario_data['length'][:, future_start:future_end]
        future_width = scenario_data['width'][:, future_start:future_end]
        future_valid = scenario_data['valid'][:, future_start:future_end]
        
        # Pad future if needed to get 80 frames
        if future_x.shape[1] < n_future:
            pad_len = n_future - future_x.shape[1]
            future_x = np.pad(future_x, ((0, 0), (0, pad_len)), mode='edge')
            future_y = np.pad(future_y, ((0, 0), (0, pad_len)), mode='edge')
            future_yaw = np.pad(future_yaw, ((0, 0), (0, pad_len)), mode='edge')
            future_vx = np.pad(future_vx, ((0, 0), (0, pad_len)), mode='edge')
            future_vy = np.pad(future_vy, ((0, 0), (0, pad_len)), mode='edge')
            future_length = np.pad(future_length, ((0, 0), (0, pad_len)), mode='edge')
            future_width = np.pad(future_width, ((0, 0), (0, pad_len)), mode='edge')
            future_valid = np.pad(future_valid, ((0, 0), (0, pad_len)), mode='constant', constant_values=0)
        
        # Build trajectory tensor [n_agents, 91, 7]
        # Format must match M2I: x, y, length, width, yaw, vx, vy
        n_agents = x.shape[0]  # Usually 128
        gt_trajectory = np.zeros((n_agents, n_history + n_future, 7), dtype=np.float32)
        gt_trajectory[:, :n_history, 0] = x
        gt_trajectory[:, :n_history, 1] = y  
        gt_trajectory[:, :n_history, 2] = length
        gt_trajectory[:, :n_history, 3] = width
        gt_trajectory[:, :n_history, 4] = yaw
        gt_trajectory[:, :n_history, 5] = vx
        gt_trajectory[:, :n_history, 6] = vy
        gt_trajectory[:, n_history:, 0] = future_x
        gt_trajectory[:, n_history:, 1] = future_y
        gt_trajectory[:, n_history:, 2] = future_length
        gt_trajectory[:, n_history:, 3] = future_width
        gt_trajectory[:, n_history:, 4] = future_yaw
        gt_trajectory[:, n_history:, 5] = future_vx
        gt_trajectory[:, n_history:, 6] = future_vy
        
        # Build validity tensor
        gt_is_valid = np.zeros((n_agents, n_history + n_future), dtype=np.float32)
        gt_is_valid[:, :n_history] = (valid > 0).astype(np.float32)
        gt_is_valid[:, n_history:] = (future_valid > 0).astype(np.float32)
        
        # Get agent types (replace -1 with 0 to avoid segfault in cython code)
        tracks_type = scenario_data['agent_types'].astype(np.int32)
        tracks_type[tracks_type < 0] = 0  # Replace invalid types with 0
        
        # Find target agent (must be valid at current time)
        current_valid_mask = valid[:, -1] > 0
        valid_agent_indices = np.where(current_valid_mask)[0]
        
        if len(valid_agent_indices) == 0:
            return None
        
        if agent_idx >= len(valid_agent_indices):
            return None
            
        target_idx = valid_agent_indices[agent_idx]
        
        # Reorder so target is first
        if target_idx != 0:
            gt_trajectory[[0, target_idx]] = gt_trajectory[[target_idx, 0]]
            gt_is_valid[[0, target_idx]] = gt_is_valid[[target_idx, 0]]
            tracks_type = tracks_type.copy()
            tracks_type[[0, target_idx]] = tracks_type[[target_idx, 0]]
        
        # Compute normalizer (center on target at current time)
        cent_x = gt_trajectory[0, n_history - 1, 0]
        cent_y = gt_trajectory[0, n_history - 1, 1]
        waymo_yaw = gt_trajectory[0, n_history - 1, 4]
        angle = -waymo_yaw + math.radians(90)
        
        normalizer = Normalizer(cent_x, cent_y, angle)
        
        # Normalize trajectories
        gt_trajectory[:, :, :2] = get_normalized(gt_trajectory[:, :, :2], normalizer)
        
        # Render BEV raster with road lanes (channels 40-59)
        # This is done BEFORE get_agents so agent trajectories can be added
        bev_image = self._render_bev_raster(
            scenario_data['roadgraph_xyz'],
            scenario_data['roadgraph_type'],
            scenario_data['roadgraph_valid'],
            normalizer,
            raster_scale=1,
        )
        
        # Ensure utils.args is set and image is the rendered BEV for get_agents
        utils.args = self.args
        self.args.image = bev_image  # get_agents will add agent trajectories to this
        
        # Create vectors using M2I's get_agents (produces 128-dim vectors)
        # This also populates channels 0-10 (ego) and 20-30 (others) of self.args.image
        agent_vectors, agent_polyline_spans, trajs = utils_cython.get_agents(
            gt_trajectory, gt_is_valid, tracks_type, False, self.args
        )
        
        # Create road vectors (pure Python implementation)
        road_vectors, road_polyline_spans = self._create_road_vectors(
            scenario_data['roadgraph_xyz'],
            scenario_data['roadgraph_type'],
            scenario_data['roadgraph_valid'],
            scenario_data['roadgraph_id'],
            normalizer,
        )
        
        # Combine agent and road vectors
        map_start_polyline_idx = len(agent_polyline_spans)
        
        if len(road_vectors) > 0:
            # Offset road polyline spans
            offset = len(agent_vectors)
            road_polyline_spans_adjusted = [
                slice(s.start + offset, s.stop + offset) for s in road_polyline_spans
            ]
            
            vectors = np.concatenate([agent_vectors, road_vectors], axis=0)
            polyline_spans = [slice(int(each[0]), int(each[1])) for each in agent_polyline_spans]
            polyline_spans.extend(road_polyline_spans_adjusted)
        else:
            vectors = agent_vectors
            polyline_spans = [slice(int(each[0]), int(each[1])) for each in agent_polyline_spans]
        
        # Labels (future trajectory in normalized coords)
        labels = gt_trajectory[0, n_history:, :2].copy()
        labels_is_valid = gt_is_valid[0, n_history:].copy()
        
        # Generate goals_2D grid for raster mode
        # In raster mode, goals_2D must have 224*224 = 50176 points
        # Each pixel in the raster is a candidate goal
        # Coordinate system: x = (pixel - 112), y = (pixel - 56)
        if 'raster' in self.args.other_params:
            goals_2D = self._generate_raster_goals_grid(raster_scale=1)
        else:
            # For non-raster mode, just use final label
            goals_2D = labels[-1:].copy()
        
        # Get agent info
        agent_id = scenario_data['agent_ids'][target_idx] if target_idx < len(scenario_data['agent_ids']) else 0
        agent_type = scenario_data['agent_types'][target_idx] if target_idx < len(scenario_data['agent_types']) else 1
        
        # Compute speed from velocity at last history frame
        last_hist_idx = n_history - 1
        vx = scenario_data['vx'][target_idx, current_t] if current_t < 91 else 0.0
        vy = scenario_data['vy'][target_idx, current_t] if current_t < 91 else 0.0
        speed = float(np.sqrt(vx**2 + vy**2))
        
        # Get heading at last history frame
        waymo_yaw = float(scenario_data['yaw'][target_idx, current_t]) if current_t < 91 else 0.0
        
        # Create mapping
        mapping = {
            'matrix': vectors,
            'polyline_spans': polyline_spans,
            'labels': labels,
            'labels_is_valid': labels_is_valid,
            'normalizer': normalizer,
            'goals_2D': goals_2D,
            'scenario_id': scenario_data['scenario_id'],
            'object_id': agent_id,
            'cent_x': cent_x,
            'cent_y': cent_y,
            'angle': angle,
            'current_timestep': current_t,
            'eval_time': 80,
            'map_start_polyline_idx': map_start_polyline_idx,
            # Additional required fields
            'speed': speed,
            'waymo_yaw': waymo_yaw,
            'track_type_int': int(agent_type),
            'stage_one_label': 0,  # GT endpoint index (not used for inference)
            'predict_agent_num': gt_trajectory.shape[0],
        }
        
        # Add raster image - use the populated BEV image (roads + agents)
        if 'raster' in self.args.other_params:
            # self.args.image was populated by _render_bev_raster (roads) 
            # and get_agents (agent trajectories)
            mapping['image'] = self.args.image.astype(np.float32)
        
        return mapping
    
    def run_inference_single(self, mapping: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run DenseTNT inference on a single mapping.
        
        Returns:
            pred_traj: [6, 80, 2] - 6 trajectory modes
            pred_scores: [6] - confidence scores
        """
        import utils
        import modeling.vectornet as vectornet_module
        # CRITICAL: Restore DenseTNT args before inference
        # VectorNet uses a module-level 'args' variable that gets overwritten
        # when we load Relation/Conditional models in their __init__
        utils.args = self.args
        vectornet_module.args = self.args
        
        with torch.no_grad():
            try:
                pred_traj, pred_score, _ = self.model([mapping], self.device)
                
                pred_traj = np.array(pred_traj[0]) if pred_traj is not None else None
                pred_score = np.array(pred_score[0]) if pred_score is not None else None
                
                return pred_traj, pred_score
            except Exception as e:
                print(f"    Inference error: {e}")
                return None, None
    
    def run_relation_inference(
        self,
        mapping_agent1: Dict,
        mapping_agent2: Dict,
    ) -> int:
        """
        Run Relation model to determine influencer/reactor relationship.
        
        The Relation model takes a pair of agents and predicts:
          - 0: No clear relation
          - 1: Agent 1 is influencer, Agent 2 is reactor
          - 2: Agent 2 is influencer, Agent 1 is reactor
        
        Args:
            mapping_agent1: Mapping for first agent
            mapping_agent2: Mapping for second agent
            
        Returns:
            relation_label: 0, 1, or 2
        """
        if self.relation_model is None:
            print("  Relation model not loaded, loading now...")
            self._load_relation_model()
            if self.relation_model is None:
                return 0  # Default: no relation
        
        import utils
        import modeling.vectornet as vectornet_module
        # Restore Relation args before inference
        utils.args = self.relation_args
        vectornet_module.args = self.relation_args
        
        # Create pair mapping for relation model
        # The relation model expects a special 'pair_inf' format
        pair_mapping = self._create_pair_mapping(mapping_agent1, mapping_agent2)
        
        with torch.no_grad():
            try:
                # Relation model returns relation logits for the pair
                outputs = self.relation_model([pair_mapping], self.device)
                
                # outputs is typically (pred_traj, pred_scores, relation_logits)
                if outputs is not None and len(outputs) >= 3:
                    relation_logits = outputs[2]
                    if relation_logits is not None:
                        # Get relation label: argmax of logits
                        relation_label = int(np.argmax(relation_logits[0]))
                        return relation_label
                        
                return 0  # Default
                
            except Exception as e:
                print(f"    Relation inference error: {e}")
                return 0
    
    def _create_pair_mapping(
        self,
        mapping1: Dict,
        mapping2: Dict,
    ) -> Dict:
        """
        Create a pair mapping for the Relation model.
        
        The Relation model processes two agents together to determine
        their influencer/reactor relationship.
        """
        # Copy first mapping as base
        pair_mapping = mapping1.copy()
        
        # Add second agent's info as 'reactor' fields
        pair_mapping['reactor'] = {
            'agent_id': mapping2.get('object_id'),
            'cent_x': mapping2.get('cent_x'),
            'cent_y': mapping2.get('cent_y'),
            'angle': mapping2.get('angle'),
            'polyline_spans': mapping2.get('polyline_spans', []),
            'matrix': mapping2.get('matrix', []),
        }
        
        # If VectorNet data exists, combine matrices
        if 'matrix' in mapping1 and 'matrix' in mapping2:
            # Stack agent polylines together
            combined_matrix = []
            combined_spans = []
            
            # Add agent 1's polylines
            for polyline in mapping1.get('matrix', []):
                start = len(combined_matrix)
                combined_matrix.extend(polyline)
                combined_spans.append([start, len(combined_matrix)])
            
            # Add agent 2's polylines  
            for polyline in mapping2.get('matrix', []):
                start = len(combined_matrix)
                combined_matrix.extend(polyline)
                combined_spans.append([start, len(combined_matrix)])
            
            pair_mapping['combined_matrix'] = combined_matrix
            pair_mapping['combined_spans'] = combined_spans
        
        return pair_mapping
    
    def run_conditional_inference(
        self,
        reactor_mapping: Dict,
        influencer_trajectory: np.ndarray,
        influencer_scores: Optional[np.ndarray] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Run Conditional model to predict reactor trajectory given influencer.
        
        The Conditional model takes the reactor agent and the influencer's
        predicted trajectory, then predicts the reactor's trajectory conditioned
        on the influencer's behavior.
        
        Args:
            reactor_mapping: Mapping for the reactor agent
            influencer_trajectory: [K, T, 2] predicted trajectories for influencer
            influencer_scores: [K] confidence scores for influencer trajectories
            
        Returns:
            pred_traj: [6, 80, 2] conditional predictions for reactor
            pred_scores: [6] confidence scores
        """
        if self.conditional_model is None:
            print("  Conditional model not loaded, loading now...")
            self._load_conditional_model()
            if self.conditional_model is None:
                return None, None
        
        import utils
        import modeling.vectornet as vectornet_module
        # Restore Conditional args before inference
        utils.args = self.conditional_args
        vectornet_module.args = self.conditional_args
        
        # Create conditional mapping
        conditional_mapping = self._create_conditional_mapping(
            reactor_mapping, 
            influencer_trajectory,
            influencer_scores
        )
        
        with torch.no_grad():
            try:
                pred_traj, pred_score, _ = self.conditional_model(
                    [conditional_mapping], self.device
                )
                
                pred_traj = np.array(pred_traj[0]) if pred_traj is not None else None
                pred_score = np.array(pred_score[0]) if pred_score is not None else None
                
                return pred_traj, pred_score
                
            except Exception as e:
                print(f"    Conditional inference error: {e}")
                return None, None
    
    def _create_conditional_mapping(
        self,
        reactor_mapping: Dict,
        influencer_trajectory: np.ndarray,
        influencer_scores: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Create a conditional mapping for the Conditional model.
        
        Adds the influencer's predicted trajectory to the reactor's mapping
        and rasterizes it into image channels 60+ for CNN encoder.
        
        The conditional model uses raster-based influencer encoding:
        - Base scene image: channels 0-59 (roads, ego history, other agents)
        - Influencer trajectory: channels 60-139 (80 future timesteps)
        """
        import math
        
        # Deep copy to avoid modifying original
        cond_mapping = {}
        for key, value in reactor_mapping.items():
            if isinstance(value, np.ndarray):
                cond_mapping[key] = value.copy()
            else:
                cond_mapping[key] = value
        
        # Add influencer trajectory info
        if influencer_trajectory is not None:
            # Use best mode if scores provided
            if influencer_scores is not None:
                best_mode = np.argmax(influencer_scores)
                best_traj = influencer_trajectory[best_mode]  # [80, 2]
            else:
                best_traj = influencer_trajectory[0]  # Use first mode
            
            cond_mapping['influencer_traj'] = best_traj.astype(np.float32)
            cond_mapping['influencer_all_trajs'] = influencer_trajectory.astype(np.float32)
            
            if influencer_scores is not None:
                cond_mapping['influencer_scores'] = influencer_scores.astype(np.float32)
            
            # CRITICAL: Rasterize influencer trajectory into image channels 60+
            # This is how the conditional model receives influencer information
            if 'raster' in self.conditional_args.other_params and 'raster_inf' in self.conditional_args.other_params:
                # Get or create image
                if 'image' in cond_mapping and cond_mapping['image'] is not None:
                    base_image = cond_mapping['image']
                else:
                    # Create empty image if not provided
                    base_image = np.zeros([224, 224, 60], dtype=np.int8)
                
                # Expand to 150 channels for influencer encoding
                if base_image.shape[2] < 150:
                    full_image = np.zeros([224, 224, 150], dtype=np.int8)
                    full_image[:, :, :base_image.shape[2]] = base_image
                else:
                    full_image = base_image.copy()
                
                # Get normalizer params from mapping
                cent_x = cond_mapping.get('cent_x', 0.0)
                cent_y = cond_mapping.get('cent_y', 0.0)
                angle = cond_mapping.get('angle', 0.0)
                cos_a = math.cos(angle)
                sin_a = math.sin(angle)
                raster_scale = 1
                
                # Rasterize each timestep of influencer trajectory
                # best_traj shape: [80, 2] in WORLD coordinates
                num_timesteps = min(80, best_traj.shape[0])
                for t in range(num_timesteps):
                    x_world = best_traj[t, 0]
                    y_world = best_traj[t, 1]
                    
                    # Transform to local coordinates (relative to reactor)
                    dx = x_world - cent_x
                    dy = y_world - cent_y
                    x_local = dx * cos_a - dy * sin_a
                    y_local = dx * sin_a + dy * cos_a
                    
                    # Convert to pixel coordinates
                    x_pix = int(x_local * raster_scale + 112 + 0.5)
                    y_pix = int(y_local * raster_scale + 56 + 0.5)
                    
                    # Set pixel if in bounds
                    if 0 <= x_pix < 224 and 0 <= y_pix < 224:
                        full_image[x_pix, y_pix, 60 + t] = 1
                
                cond_mapping['image'] = full_image.astype(np.float32)
        
        return cond_mapping
    
    def run_full_m2i_pipeline(
        self,
        scenario_data: Dict,
        t: int,
        agent1_idx: int,
        agent2_idx: int,
    ) -> Dict:
        """
        Run the full M2I 3-stage pipeline for a pair of agents.
        
        Pipeline:
          1. DenseTNT: Predict trajectories for both agents independently
          2. Relation: Determine which agent influences which
          3. Conditional: Re-predict reactor conditioned on influencer
        
        Args:
            scenario_data: Parsed scenario data
            t: Current timestep
            agent1_idx: Index of first interactive agent
            agent2_idx: Index of second interactive agent
            
        Returns:
            Dict with predictions for both agents, relation label, and 
            conditional predictions
        """
        result = {
            'agent1_id': None,
            'agent2_id': None,
            'stage1_predictions': {},  # DenseTNT predictions
            'relation_label': 0,       # 0=none, 1=agent1->agent2, 2=agent2->agent1
            'influencer_id': None,
            'reactor_id': None,
            'conditional_predictions': None,  # Reactor conditioned on influencer
        }
        
        # Get agent IDs
        interactive_indices = np.where(scenario_data['objects_of_interest'] > 0)[0]
        global_idx1 = interactive_indices[agent1_idx]
        global_idx2 = interactive_indices[agent2_idx]
        agent1_id = int(scenario_data['agent_ids'][global_idx1])
        agent2_id = int(scenario_data['agent_ids'][global_idx2])
        
        result['agent1_id'] = agent1_id
        result['agent2_id'] = agent2_id
        
        # Stage 1: DenseTNT predictions for both agents
        mapping1 = self.create_mapping_at_timestep(scenario_data, t, agent1_idx)
        mapping2 = self.create_mapping_at_timestep(scenario_data, t, agent2_idx)
        
        if mapping1 is None or mapping2 is None:
            return result
        
        pred1, scores1 = self.run_inference_single(mapping1)
        pred2, scores2 = self.run_inference_single(mapping2)
        
        if pred1 is not None:
            result['stage1_predictions'][agent1_id] = {
                'pred_traj': pred1,
                'pred_scores': scores1,
            }
        
        if pred2 is not None:
            result['stage1_predictions'][agent2_id] = {
                'pred_traj': pred2,
                'pred_scores': scores2,
            }
        
        # Stage 2: Relation prediction
        if pred1 is not None and pred2 is not None:
            relation_label = self.run_relation_inference(mapping1, mapping2)
            result['relation_label'] = relation_label
            
            # Determine influencer/reactor based on relation
            if relation_label == 1:
                # Agent 1 influences Agent 2
                result['influencer_id'] = agent1_id
                result['reactor_id'] = agent2_id
                influencer_traj, influencer_scores = pred1, scores1
                reactor_mapping = mapping2
            elif relation_label == 2:
                # Agent 2 influences Agent 1
                result['influencer_id'] = agent2_id
                result['reactor_id'] = agent1_id
                influencer_traj, influencer_scores = pred2, scores2
                reactor_mapping = mapping1
            else:
                # No clear relation, skip conditional
                return result
            
            # Stage 3: Conditional prediction for reactor
            cond_pred, cond_scores = self.run_conditional_inference(
                reactor_mapping, 
                influencer_traj,
                influencer_scores
            )
            
            if cond_pred is not None:
                result['conditional_predictions'] = {
                    'reactor_id': result['reactor_id'],
                    'pred_traj': cond_pred,
                    'pred_scores': cond_scores,
                }
        
        return result
    
    def run_receding_horizon(
        self,
        tfrecord_path: str,
        scenario_idx: int = 0,
        start_t: int = 10,
        end_t: int = 30,
        step: int = 1,
    ) -> Dict:
        """
        Run full receding horizon prediction on a single scenario.
        
        Args:
            tfrecord_path: Path to TFRecord file
            scenario_idx: Which scenario in the file
            start_t: First prediction timestep (must be >= 10)
            end_t: Last prediction timestep (must be <= 80)
            step: Timestep increment (1 = every frame, 10 = every second)
        
        Returns:
            Dict with predictions at each timestep
        """
        print(f"\n{'='*70}")
        print("RECEDING HORIZON M2I INFERENCE")
        print(f"{'='*70}")
        print(f"Prediction timesteps: {start_t} to {end_t} (step={step})")
        
        # Load model if needed
        if self.model is None:
            self.load_model()
        
        # Parse full scenario
        print(f"\nParsing scenario {scenario_idx}...")
        scenario_data = self.parse_full_scenario(tfrecord_path, scenario_idx)
        
        if scenario_data is None:
            print("  Failed to parse scenario")
            return {}
        
        scenario_id = scenario_data['scenario_id']
        if isinstance(scenario_id, bytes):
            scenario_id = scenario_id.decode()
        print(f"  Scenario ID: {scenario_id}")
        
        # Get interactive agents
        objects_of_interest = scenario_data['objects_of_interest']
        interactive_indices = np.where(objects_of_interest > 0)[0]
        print(f"  Interactive agents: {len(interactive_indices)}")
        
        if len(interactive_indices) == 0:
            print("  No interactive agents found")
            return {}
        
        # Run predictions at each timestep
        all_predictions = {}
        
        for t in range(start_t, end_t + 1, step):
            print(f"\n--- Timestep t={t} ---")
            print(f"  History: [{t-10} : {t}], Predict: [{t+1} : {min(t+80, 90)}]")
            
            timestep_predictions = {
                'timestep': t,
                'agents': {},
                'ground_truth': {},
            }
            
            self._set_seeds(42 + t)  # Different seed per timestep but reproducible
            
            # Predict for each interactive agent
            for agent_idx in range(len(interactive_indices)):
                global_idx = interactive_indices[agent_idx]
                agent_id = scenario_data['agent_ids'][global_idx]
                
                # Get current GT position for this agent
                if t < 91:
                    gt_pos = np.array([
                        scenario_data['x'][global_idx, t],
                        scenario_data['y'][global_idx, t]
                    ])
                    gt_valid = scenario_data['valid'][global_idx, t] > 0
                else:
                    gt_valid = False
                
                if not gt_valid:
                    continue
                
                # Store ground truth
                future_end = min(t + 81, 91)
                gt_future = np.stack([
                    scenario_data['x'][global_idx, t+1:future_end],
                    scenario_data['y'][global_idx, t+1:future_end]
                ], axis=-1)
                
                timestep_predictions['ground_truth'][int(agent_id)] = {
                    'current_pos': gt_pos,
                    'future_traj': gt_future,
                }
                
                # Create mapping for this agent at this timestep
                mapping = self.create_mapping_at_timestep(
                    scenario_data, t, agent_idx
                )
                
                if mapping is None:
                    print(f"    Agent {agent_id}: Failed to create mapping")
                    continue
                
                # Run inference
                pred_traj, pred_scores = self.run_inference_single(mapping)
                
                if pred_traj is not None:
                    # ============================================================
                    # CRITICAL: M2I decoder ALREADY denormalizes predictions!
                    # 
                    # The decoder.py (lines 310-320) applies:
                    #   normalizer(predict_trajs, reverse=True)
                    #
                    # This already transforms from agent-centric to world coords.
                    # DO NOT apply another denormalization here!
                    # ============================================================
                    pred_world = pred_traj  # Already in world coordinates
                    
                    timestep_predictions['agents'][int(agent_id)] = {
                        'pred_traj': pred_world,  # [6, 80, 2]
                        'pred_scores': pred_scores,  # [6]
                        'current_pos': gt_pos,
                    }
                    
                    # Compute error for best mode
                    if len(gt_future) > 0:
                        best_mode = np.argmax(pred_scores) if pred_scores is not None else 0
                        pred_best = pred_world[best_mode, :len(gt_future)]
                        errors = np.linalg.norm(pred_best - gt_future, axis=-1)
                        ade = errors.mean()
                        fde = errors[-1] if len(errors) > 0 else 0
                        print(f"    Agent {agent_id}: ADE={ade:.2f}m, FDE={fde:.2f}m")
                else:
                    print(f"    Agent {agent_id}: Inference failed")
            
            all_predictions[t] = timestep_predictions
        
        return {
            'scenario_id': scenario_id,
            'predictions': all_predictions,
            'metadata': {
                'start_t': start_t,
                'end_t': end_t,
                'step': step,
                'tfrecord': tfrecord_path,
                'scenario_idx': scenario_idx,
            }
        }
    
    def run_multiple_scenarios(
        self,
        tfrecord_path: str,
        num_scenarios: int = 5,
        start_t: int = 10,
        end_t: int = 30,
        step: int = 5,
        output_path: Optional[Path] = None,
    ) -> Dict:
        """
        Run receding horizon on multiple scenarios.
        """
        print(f"\n{'='*70}")
        print("RECEDING HORIZON - MULTIPLE SCENARIOS")
        print(f"{'='*70}")
        print(f"Scenarios: {num_scenarios}")
        print(f"Timesteps: {start_t} to {end_t} (step={step})")
        
        all_results = {}
        
        for scenario_idx in range(num_scenarios):
            print(f"\n\n{'#'*70}")
            print(f"# SCENARIO {scenario_idx + 1}/{num_scenarios}")
            print(f"{'#'*70}")
            
            try:
                result = self.run_receding_horizon(
                    tfrecord_path,
                    scenario_idx=scenario_idx,
                    start_t=start_t,
                    end_t=end_t,
                    step=step,
                )
                
                if result:
                    all_results[scenario_idx] = result
            except Exception as e:
                print(f"  Error: {e}")
                continue
        
        # Save results
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'wb') as f:
                pickle.dump(all_results, f)
            print(f"\nSaved results to: {output_path}")
        
        return all_results
    
    def run_receding_horizon_full_pipeline(
        self,
        tfrecord_path: str,
        scenario_idx: int = 0,
        start_t: int = 10,
        end_t: int = 30,
        step: int = 1,
        use_conditional: bool = True,
    ) -> Dict:
        """
        Run full 3-stage M2I pipeline at each receding horizon timestep.
        
        This method runs the complete M2I pipeline:
          1. DenseTNT: Independent trajectory prediction for all agents
          2. Relation: Determine influencer/reactor pairs
          3. Conditional: Re-predict reactors conditioned on influencers
        
        Args:
            tfrecord_path: Path to TFRecord file
            scenario_idx: Which scenario in the file
            start_t: First prediction timestep (must be >= 10)
            end_t: Last prediction timestep (must be <= 80)
            step: Timestep increment
            use_conditional: Whether to run full 3-stage pipeline
        
        Returns:
            Dict with predictions at each timestep including relation info
        """
        print(f"\n{'='*70}")
        print("RECEDING HORIZON - FULL M2I PIPELINE")
        print(f"{'='*70}")
        print(f"Prediction timesteps: {start_t} to {end_t} (step={step})")
        print(f"3-Stage Pipeline: {'Enabled' if use_conditional else 'Disabled'}")
        
        # Load all models
        if self.model is None:
            print("\nLoading DenseTNT model...")
            self.load_model()
        
        if use_conditional:
            if self.relation_model is None:
                print("Loading Relation model...")
                self._load_relation_model()
            if self.conditional_model is None:
                print("Loading Conditional model...")
                self._load_conditional_model()
        
        # Parse full scenario
        print(f"\nParsing scenario {scenario_idx}...")
        scenario_data = self.parse_full_scenario(tfrecord_path, scenario_idx)
        
        if scenario_data is None:
            print("  Failed to parse scenario")
            return {}
        
        scenario_id = scenario_data['scenario_id']
        if isinstance(scenario_id, bytes):
            scenario_id = scenario_id.decode()
        print(f"  Scenario ID: {scenario_id}")
        
        # Get interactive agents
        objects_of_interest = scenario_data['objects_of_interest']
        interactive_indices = np.where(objects_of_interest > 0)[0]
        num_agents = len(interactive_indices)
        print(f"  Interactive agents: {num_agents}")
        
        if num_agents == 0:
            print("  No interactive agents found")
            return {}
        
        # Run predictions at each timestep
        all_predictions = {}
        
        for t in range(start_t, end_t + 1, step):
            print(f"\n--- Timestep t={t} ---")
            print(f"  History: [{t-10} : {t}], Predict: [{t+1} : {min(t+80, 90)}]")
            
            timestep_predictions = {
                'timestep': t,
                'agents': {},
                'ground_truth': {},
                'relations': [],
                'conditional_predictions': {},
            }
            
            self._set_seeds(42 + t)
            
            # Stage 1: DenseTNT predictions for all agents
            print(f"  Stage 1: DenseTNT predictions...")
            agent_mappings = {}
            agent_predictions = {}
            
            for agent_idx in range(num_agents):
                global_idx = interactive_indices[agent_idx]
                agent_id = int(scenario_data['agent_ids'][global_idx])
                
                # Check validity
                if t >= 91 or scenario_data['valid'][global_idx, t] <= 0:
                    continue
                
                # Get GT
                gt_pos = np.array([
                    scenario_data['x'][global_idx, t],
                    scenario_data['y'][global_idx, t]
                ])
                
                future_end = min(t + 81, 91)
                gt_future = np.stack([
                    scenario_data['x'][global_idx, t+1:future_end],
                    scenario_data['y'][global_idx, t+1:future_end]
                ], axis=-1)
                
                timestep_predictions['ground_truth'][agent_id] = {
                    'current_pos': gt_pos,
                    'future_traj': gt_future,
                }
                
                # Create mapping
                mapping = self.create_mapping_at_timestep(scenario_data, t, agent_idx)
                if mapping is None:
                    continue
                
                agent_mappings[agent_id] = mapping
                
                # Run DenseTNT inference
                pred_traj, pred_scores = self.run_inference_single(mapping)
                
                if pred_traj is not None:
                    agent_predictions[agent_id] = {
                        'pred_traj': pred_traj,
                        'pred_scores': pred_scores,
                        'current_pos': gt_pos,
                    }
                    
                    # Compute initial error
                    if len(gt_future) > 0:
                        best_mode = np.argmax(pred_scores) if pred_scores is not None else 0
                        pred_best = pred_traj[best_mode, :len(gt_future)]
                        errors = np.linalg.norm(pred_best - gt_future, axis=-1)
                        ade = errors.mean()
                        fde = errors[-1] if len(errors) > 0 else 0
                        print(f"    Agent {agent_id} (DenseTNT): ADE={ade:.2f}m, FDE={fde:.2f}m")
            
            # Store DenseTNT predictions
            timestep_predictions['agents'] = agent_predictions.copy()
            
            # Stage 2 & 3: Relation + Conditional for agent pairs
            if use_conditional and num_agents >= 2 and len(agent_predictions) >= 2:
                print(f"  Stage 2 & 3: Relation + Conditional...")
                
                # Process pairs of agents
                agent_ids = list(agent_predictions.keys())
                for i in range(len(agent_ids)):
                    for j in range(i + 1, len(agent_ids)):
                        id1, id2 = agent_ids[i], agent_ids[j]
                        
                        if id1 not in agent_mappings or id2 not in agent_mappings:
                            continue
                        
                        # Run relation inference
                        relation_label = self.run_relation_inference(
                            agent_mappings[id1],
                            agent_mappings[id2]
                        )
                        
                        timestep_predictions['relations'].append({
                            'agent1_id': id1,
                            'agent2_id': id2,
                            'relation': relation_label,  # 0=none, 1=id1->id2, 2=id2->id1
                        })
                        
                        if relation_label == 0:
                            continue
                        
                        # Determine influencer/reactor
                        if relation_label == 1:
                            influencer_id, reactor_id = id1, id2
                        else:
                            influencer_id, reactor_id = id2, id1
                        
                        print(f"    Pair ({id1}, {id2}): Agent {influencer_id} influences Agent {reactor_id}")
                        
                        # Run conditional prediction for reactor
                        influencer_pred = agent_predictions[influencer_id]['pred_traj']
                        influencer_scores = agent_predictions[influencer_id]['pred_scores']
                        reactor_mapping = agent_mappings[reactor_id]
                        
                        cond_pred, cond_scores = self.run_conditional_inference(
                            reactor_mapping,
                            influencer_pred,
                            influencer_scores
                        )
                        
                        if cond_pred is not None:
                            # Store conditional prediction
                            timestep_predictions['conditional_predictions'][reactor_id] = {
                                'pred_traj': cond_pred,
                                'pred_scores': cond_scores,
                                'influencer_id': influencer_id,
                            }
                            
                            # Update main prediction with conditional result
                            timestep_predictions['agents'][reactor_id]['pred_traj'] = cond_pred
                            timestep_predictions['agents'][reactor_id]['pred_scores'] = cond_scores
                            timestep_predictions['agents'][reactor_id]['conditioned_on'] = influencer_id
                            
                            # Compute conditional error
                            gt_future = timestep_predictions['ground_truth'].get(reactor_id, {}).get('future_traj')
                            if gt_future is not None and len(gt_future) > 0:
                                best_mode = np.argmax(cond_scores) if cond_scores is not None else 0
                                pred_best = cond_pred[best_mode, :len(gt_future)]
                                errors = np.linalg.norm(pred_best - gt_future, axis=-1)
                                ade = errors.mean()
                                fde = errors[-1] if len(errors) > 0 else 0
                                print(f"    Agent {reactor_id} (Conditional): ADE={ade:.2f}m, FDE={fde:.2f}m")
            
            all_predictions[t] = timestep_predictions
        
        return {
            'scenario_id': scenario_id,
            'predictions': all_predictions,
            'metadata': {
                'start_t': start_t,
                'end_t': end_t,
                'step': step,
                'tfrecord': tfrecord_path,
                'scenario_idx': scenario_idx,
                'use_conditional': use_conditional,
                'pipeline': 'full_m2i_3stage',
            }
        }

    def run_receding_horizon_subprocess_pipeline(
        self,
        tfrecord_path: str,
        scenario_idx: int = 0,
        start_t: int = 10,
        end_t: int = 90,
        step: int = 1,
        use_conditional: bool = True,
    ) -> Dict:
        """
        Run full 3-stage M2I pipeline using subprocesses for Relation/Conditional.
        
        This method avoids global args conflicts by running Relation and Conditional
        models in separate Python processes. The workflow:
        
        1. DenseTNT: Run in main process (first model loaded)
        2. Relation: Run via subprocess_relation.py 
        3. Conditional: Run via subprocess_conditional.py
        
        Data is passed between stages via temporary pickle files.
        
        Args:
            tfrecord_path: Path to TFRecord file
            scenario_idx: Which scenario in the file
            start_t: First prediction timestep (must be >= 10)
            end_t: Last prediction timestep (must be <= 90)
            step: Timestep increment
            use_conditional: Whether to run full 3-stage pipeline
        
        Returns:
            Dict with predictions at each timestep including relation info
        """
        import subprocess
        import tempfile
        
        print(f"\n{'='*70}")
        print("RECEDING HORIZON - SUBPROCESS M2I PIPELINE")
        print(f"{'='*70}")
        print(f"Prediction timesteps: {start_t} to {end_t} (step={step})")
        print(f"3-Stage Pipeline: {'Enabled (subprocess)' if use_conditional else 'Disabled'}")
        
        # Subprocess script paths
        relation_script = Path(__file__).parent / 'subprocess_relation.py'
        conditional_script = Path(__file__).parent / 'subprocess_conditional.py'
        
        # Load DenseTNT model (only this model in main process)
        if self.model is None:
            print("\nLoading DenseTNT model...")
            self.load_model()
        
        # Parse full scenario
        print(f"\nParsing scenario {scenario_idx}...")
        scenario_data = self.parse_full_scenario(tfrecord_path, scenario_idx)
        
        if scenario_data is None:
            print("  Failed to parse scenario")
            return {}
        
        scenario_id = scenario_data['scenario_id']
        if isinstance(scenario_id, bytes):
            scenario_id = scenario_id.decode()
        print(f"  Scenario ID: {scenario_id}")
        
        # Get interactive agents
        objects_of_interest = scenario_data['objects_of_interest']
        interactive_indices = np.where(objects_of_interest > 0)[0]
        num_agents = len(interactive_indices)
        print(f"  Interactive agents: {num_agents}")
        
        if num_agents == 0:
            print("  No interactive agents found")
            return {}
        
        # Stage 1: Run DenseTNT for all timesteps
        print(f"\n{'='*50}")
        print("STAGE 1: DenseTNT Predictions (all timesteps)")
        print(f"{'='*50}")
        
        all_predictions = {}
        all_mappings = {}  # For subprocess communication
        
        for t in range(start_t, end_t + 1, step):
            print(f"\n--- Timestep t={t} ---")
            
            timestep_predictions = {
                'timestep': t,
                'agents': {},
                'ground_truth': {},
                'relations': [],
                'conditional_predictions': {},
            }
            timestep_mappings = {}
            
            self._set_seeds(42 + t)
            
            for agent_idx in range(num_agents):
                global_idx = interactive_indices[agent_idx]
                agent_id = int(scenario_data['agent_ids'][global_idx])
                
                # Check validity
                if t >= 91 or scenario_data['valid'][global_idx, t] <= 0:
                    continue
                
                # Get GT
                gt_pos = np.array([
                    scenario_data['x'][global_idx, t],
                    scenario_data['y'][global_idx, t]
                ])
                
                future_end = min(t + 81, 91)
                gt_future = np.stack([
                    scenario_data['x'][global_idx, t+1:future_end],
                    scenario_data['y'][global_idx, t+1:future_end]
                ], axis=-1)
                
                timestep_predictions['ground_truth'][agent_id] = {
                    'current_pos': gt_pos,
                    'future_traj': gt_future,
                }
                
                # Create mapping
                mapping = self.create_mapping_at_timestep(scenario_data, t, agent_idx)
                if mapping is None:
                    continue
                
                timestep_mappings[agent_id] = mapping
                
                # Run DenseTNT inference
                pred_traj, pred_scores = self.run_inference_single(mapping)
                
                if pred_traj is not None:
                    timestep_predictions['agents'][agent_id] = {
                        'pred_traj': pred_traj,
                        'pred_scores': pred_scores,
                        'current_pos': gt_pos,
                    }
                    
                    # Compute error
                    if len(gt_future) > 0:
                        best_mode = np.argmax(pred_scores) if pred_scores is not None else 0
                        pred_best = pred_traj[best_mode, :len(gt_future)]
                        errors = np.linalg.norm(pred_best - gt_future, axis=-1)
                        ade = errors.mean()
                        fde = errors[-1] if len(errors) > 0 else 0
                        print(f"    Agent {agent_id}: ADE={ade:.2f}m, FDE={fde:.2f}m")
            
            all_predictions[t] = timestep_predictions
            all_mappings[t] = {
                'mappings': timestep_mappings,
                'predictions': timestep_predictions['agents'],
            }
        
        # Stage 2 & 3: Relation + Conditional via subprocesses
        if use_conditional and num_agents >= 2:
            print(f"\n{'='*50}")
            print("STAGE 2: Relation Inference (subprocess)")
            print(f"{'='*50}")
            
            # Create temp files for inter-process communication
            with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as f:
                pickle.dump(all_mappings, f)
                input_pickle = f.name
            
            with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as f:
                relation_output = f.name
            
            # Run Relation subprocess
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            cmd = [
                sys.executable, str(relation_script),
                '--input_pickle', input_pickle,
                '--output_pickle', relation_output,
                '--device', device,
            ]
            
            print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"Relation subprocess failed: {result.stderr}")
            else:
                print(result.stdout)
                
                # Load relation results
                with open(relation_output, 'rb') as f:
                    relation_results = pickle.load(f)
                
                # Merge relation results into predictions
                for t, relations in relation_results.items():
                    if t in all_predictions:
                        all_predictions[t]['relations'] = relations
                        
                        # Update mappings with relations for conditional
                        all_mappings[t]['relations'] = relations
            
            # Clean up relation temp file
            os.unlink(input_pickle)
            os.unlink(relation_output)
            
            print(f"\n{'='*50}")
            print("STAGE 3: Conditional Inference (subprocess)")
            print(f"{'='*50}")
            
            # Create new temp files with relation info
            with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as f:
                pickle.dump(all_mappings, f)
                input_pickle = f.name
            
            with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as f:
                conditional_output = f.name
            
            # Run Conditional subprocess
            cmd = [
                sys.executable, str(conditional_script),
                '--input_pickle', input_pickle,
                '--output_pickle', conditional_output,
                '--device', device,
            ]
            
            print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"Conditional subprocess failed: {result.stderr}")
            else:
                print(result.stdout)
                
                # Load conditional results
                with open(conditional_output, 'rb') as f:
                    conditional_results = pickle.load(f)
                
                # Merge conditional results into predictions
                for t, conditionals in conditional_results.items():
                    if t in all_predictions:
                        all_predictions[t]['conditional_predictions'] = conditionals
                        
                        # Update agent predictions with conditional results
                        for agent_id, cond_pred in conditionals.items():
                            if agent_id in all_predictions[t]['agents']:
                                pred_traj = cond_pred.get('pred_traj')
                                pred_scores = cond_pred.get('pred_scores')
                                
                                # Skip if prediction is invalid
                                if pred_traj is None or not hasattr(pred_traj, 'shape') or pred_traj.ndim < 2:
                                    print(f"  t={t} Agent {agent_id}: Skipping invalid conditional prediction")
                                    continue
                                
                                all_predictions[t]['agents'][agent_id]['pred_traj'] = pred_traj
                                all_predictions[t]['agents'][agent_id]['pred_scores'] = pred_scores
                                all_predictions[t]['agents'][agent_id]['conditioned_on'] = cond_pred['influencer_id']
                                
                                # Recompute error with conditional prediction
                                gt = all_predictions[t]['ground_truth'].get(agent_id, {})
                                gt_future = gt.get('future_traj')
                                if gt_future is not None and len(gt_future) > 0 and pred_traj.ndim >= 2:
                                    best_mode = np.argmax(pred_scores) if pred_scores is not None and pred_scores.size > 0 else 0
                                    try:
                                        pred_best = pred_traj[best_mode, :len(gt_future)]
                                        errors = np.linalg.norm(pred_best - gt_future, axis=-1)
                                        ade = errors.mean()
                                        fde = errors[-1] if len(errors) > 0 else 0
                                        print(f"  t={t} Agent {agent_id} (Conditional): ADE={ade:.2f}m, FDE={fde:.2f}m")
                                    except (IndexError, ValueError) as e:
                                        print(f"  t={t} Agent {agent_id}: Error computing conditional metrics: {e}")
            
            # Clean up conditional temp files
            os.unlink(input_pickle)
            os.unlink(conditional_output)
        
        return {
            'scenario_id': scenario_id,
            'predictions': all_predictions,
            'metadata': {
                'start_t': start_t,
                'end_t': end_t,
                'step': step,
                'tfrecord': tfrecord_path,
                'scenario_idx': scenario_idx,
                'use_conditional': use_conditional,
                'pipeline': 'subprocess_3stage',
            }
        }


class RecedingHorizonVisualizer:
    """
    Visualize receding horizon predictions as movies.
    
    Visual style matches generate_m2i_movies.py for consistency:
    - Polyline road rendering (lanes, road lines, road edges, crosswalks)
    - Oriented vehicle rectangles for all agents
    - Type-based coloring (vehicle, pedestrian, cyclist)
    - Velocity arrows for moving agents
    - History trails with dashed lines
    - Ground truth future with dotted lines
    - Predicted trajectories with proper styling
    - Comprehensive legend
    - Follow-agent camera mode
    """
    
    # Colors matching BEV movie generator and M2I movie generator style
    COLORS = {
        # Agent types
        'agent_0': '#FF0000',           # Red for first interactive agent
        'agent_1': '#f39c12',           # Orange for second interactive agent
        'vehicle': '#3498db',           # Blue for other vehicles
        'pedestrian': '#2ecc71',        # Green for pedestrians
        'cyclist': '#9b59b6',           # Purple for cyclists
        'other': '#95a5a6',             # Gray for others
        # Road elements
        'lane': '#34495e',              # Dark gray for lanes
        'road_line': '#7f8c8d',         # Medium gray for road lines
        'road_edge': '#2c3e50',         # Very dark gray for road edges
        'crosswalk': '#16a085',         # Teal for crosswalks
        'stop_sign': '#c0392b',         # Dark red for stop signs
        # Trajectories
        'history': '#e74c3c',           # Red for history trail
        'future_gt': '#bdc3c7',         # Light gray for GT future
        'pred': '#27ae60',              # Green for prediction
    }
    
    def __init__(self, output_dir: Path, view_range: float = 50.0, fps: int = 10, dpi: int = 100):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.view_range = view_range
        self.fps = fps
        self.dpi = dpi
        
        # Import matplotlib
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        import matplotlib.patches as mpatches
        import matplotlib.transforms as transforms
        from matplotlib.patches import Rectangle, Polygon, Circle, FancyArrow
        from matplotlib.lines import Line2D
        from matplotlib.animation import FFMpegWriter, PillowWriter
        self.plt = plt
        self.animation = animation
        self.mpatches = mpatches
        self.transforms = transforms
        self.Rectangle = Rectangle
        self.Polygon = Polygon
        self.Circle = Circle
        self.FancyArrow = FancyArrow
        self.Line2D = Line2D
        self.FFMpegWriter = FFMpegWriter
        self.PillowWriter = PillowWriter
    
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
                polygon = self.Polygon(in_view, closed=True,
                                        facecolor=self.COLORS['crosswalk'],
                                        edgecolor=self.COLORS['crosswalk'],
                                        linewidth=2, alpha=0.3, zorder=1)
                ax.add_patch(polygon)
        
        # Draw stop signs (circles)
        for stop_pos in road_polylines.get('stop_signs', []):
            sx, sy = stop_pos[0], stop_pos[1]
            if abs(sx - center_x) < view_range and abs(sy - center_y) < view_range:
                circle = self.Circle((sx, sy), 1.5, color=self.COLORS['stop_sign'], alpha=0.7, zorder=5)
                ax.add_patch(circle)
    
    def _get_agent_color(self, agent_type: int, agent_idx: int, interactive_indices: List[int]) -> str:
        """Get color for an agent based on type and role."""
        if agent_idx in interactive_indices:
            # Interactive agents get special colors
            pos = interactive_indices.index(agent_idx)
            if pos == 0:
                return self.COLORS['agent_0']
            else:
                return self.COLORS['agent_1']
        
        type_map = {
            1: 'vehicle',
            2: 'pedestrian', 
            3: 'cyclist',
        }
        type_name = type_map.get(int(agent_type), 'other')
        return self.COLORS.get(type_name, self.COLORS['other'])
    
    def _create_vehicle_patch(self, ax, x: float, y: float, heading: float, 
                               length: float, width: float, color: str, 
                               alpha: float = 0.8):
        """Create oriented rectangle patch for vehicle (matching BEV style)."""
        # Default size if not available
        if length <= 0:
            length = 4.5
        if width <= 0:
            width = 2.0
            
        rect = self.Rectangle((-length/2, -width/2), length, width,
                              facecolor=color, edgecolor='black',
                              linewidth=1.5, alpha=alpha)
        
        # Apply rotation and translation
        t = self.transforms.Affine2D().rotate(heading).translate(x, y) + ax.transData
        rect.set_transform(t)
        
        return rect
    
    def generate_movie(
        self,
        scenario_data: Dict,
        predictions: Dict,
        scenario_id: str,
        fps: int = None,
        follow_agent: bool = True,
        show_all_agents: bool = True,
    ) -> Path:
        """
        Generate a movie showing receding horizon predictions.
        
        Visual style matches generate_m2i_movies.py:
        - Polyline road rendering
        - Oriented vehicle rectangles  
        - All agents displayed with type-based colors
        - Follow-agent camera mode
        - Comprehensive legend
        
        Each frame shows the prediction made at timestep t, with:
        - Road map as polylines
        - All agents as oriented rectangles
        - History trajectories as dashed lines
        - Ground truth future as dotted lines
        - Predicted trajectories as solid lines with star endpoint
        
        Args:
            scenario_data: Full scenario data from parse_full_scenario
            predictions: Dict from run_receding_horizon
            scenario_id: Scenario ID for filename
            fps: Frames per second (default: self.fps)
            follow_agent: If True, camera follows the first interactive agent
            show_all_agents: If True, show all agents in scene
            
        Returns:
            Path to the generated movie
        """
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        
        if fps is None:
            fps = self.fps
        
        # Get prediction timesteps
        timesteps = sorted(predictions.keys())
        if len(timesteps) == 0:
            print("  No predictions to visualize")
            return None
        
        # Setup figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        
        # Get road data and build polylines
        roadgraph_xyz = scenario_data['roadgraph_xyz']
        roadgraph_type = scenario_data['roadgraph_type']
        roadgraph_id = scenario_data['roadgraph_id']
        roadgraph_valid = scenario_data['roadgraph_valid']
        
        valid_mask = roadgraph_valid > 0
        road_polylines = self._build_road_polylines(
            roadgraph_xyz[valid_mask],
            roadgraph_type[valid_mask],
            roadgraph_id[valid_mask]
        )
        
        # Get agent data
        all_x = scenario_data['x']      # [128, 91]
        all_y = scenario_data['y']      # [128, 91]
        all_yaw = scenario_data['yaw']  # [128, 91]
        all_length = scenario_data['length']  # [128, 91]
        all_width = scenario_data['width']    # [128, 91]
        all_vx = scenario_data['vx']    # [128, 91]
        all_vy = scenario_data['vy']    # [128, 91]
        all_valid = scenario_data['valid']    # [128, 91]
        agent_types = scenario_data['agent_types']  # [128]
        agent_ids = scenario_data['agent_ids']  # [128]
        tracks_to_predict = scenario_data['tracks_to_predict']  # [128]
        
        # Find interactive agent indices (agents we're predicting for)
        # First, from the predictions keys
        first_timestep = timesteps[0]
        gt_data = predictions[first_timestep].get('ground_truth', {})
        interactive_agent_ids = list(gt_data.keys())
        
        # Map agent IDs to indices
        interactive_indices = []
        for aid in interactive_agent_ids:
            matches = np.where(agent_ids.astype(int) == int(aid))[0]
            if len(matches) > 0:
                interactive_indices.append(matches[0])
        
        # If no matches, use tracks_to_predict
        if len(interactive_indices) == 0:
            interactive_indices = list(np.where(tracks_to_predict > 0)[0][:2])
        
        # Calculate plot bounds for non-follow mode
        if not follow_agent:
            all_valid_positions = []
            for i in range(128):
                v_mask = all_valid[i] > 0
                if v_mask.any():
                    all_valid_positions.extend(list(zip(all_x[i, v_mask], all_y[i, v_mask])))
            if all_valid_positions:
                all_valid_positions = np.array(all_valid_positions)
                x_min = all_valid_positions[:, 0].min() - 20
                x_max = all_valid_positions[:, 0].max() + 20
                y_min = all_valid_positions[:, 1].min() - 20
                y_max = all_valid_positions[:, 1].max() + 20
            else:
                x_min, x_max, y_min, y_max = -50, 50, -50, 50
        
        # Agent colors for predictions (use plasma colormap for modes)
        agent_colors = ['#FF0000', '#f39c12', '#2196F3', '#4CAF50', '#9C27B0']
        
        def animate(frame_idx):
            ax.clear()
            t = timesteps[frame_idx]
            
            # Get center position for camera (first interactive agent)
            if len(interactive_indices) > 0 and all_valid[interactive_indices[0], t] > 0:
                center_x = all_x[interactive_indices[0], t]
                center_y = all_y[interactive_indices[0], t]
            else:
                # Fall back to finding any valid position
                center_x, center_y = 0, 0
                for check_idx in interactive_indices:
                    for check_t in range(t, -1, -1):
                        if all_valid[check_idx, check_t] > 0:
                            center_x = all_x[check_idx, check_t]
                            center_y = all_y[check_idx, check_t]
                            break
                    if center_x != 0 or center_y != 0:
                        break
            
            # Set view bounds
            if follow_agent:
                ax.set_xlim(center_x - self.view_range, center_x + self.view_range)
                ax.set_ylim(center_y - self.view_range, center_y + self.view_range)
            else:
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)
            
            ax.set_aspect('equal')
            ax.set_facecolor('#f5f5f5')
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            
            # Draw road polylines
            self._draw_road_polylines(ax, road_polylines, center_x, center_y)
            
            # Draw all agents as oriented rectangles
            for agent_i in range(128):
                if all_valid[agent_i, t] <= 0:
                    continue
                
                agent_x = all_x[agent_i, t]
                agent_y = all_y[agent_i, t]
                
                # Skip if out of view (when following)
                if follow_agent:
                    if abs(agent_x - center_x) > self.view_range or abs(agent_y - center_y) > self.view_range:
                        continue
                
                agent_yaw = all_yaw[agent_i, t]
                agent_length = all_length[agent_i, t]
                agent_width = all_width[agent_i, t]
                agent_type = agent_types[agent_i]
                
                # Get color based on role and type
                color = self._get_agent_color(agent_type, agent_i, interactive_indices)
                
                # Draw oriented rectangle
                if show_all_agents or agent_i in interactive_indices:
                    rect = self._create_vehicle_patch(ax, agent_x, agent_y, agent_yaw,
                                                      agent_length, agent_width, color)
                    ax.add_patch(rect)
                    
                    # Draw velocity arrow for moving agents
                    speed = np.sqrt(all_vx[agent_i, t]**2 + all_vy[agent_i, t]**2)
                    if speed > 0.5 and agent_i in interactive_indices:
                        arrow = self.FancyArrow(agent_x, agent_y,
                                               all_vx[agent_i, t] * 0.5, all_vy[agent_i, t] * 0.5,
                                               width=0.3, head_width=0.8, head_length=0.5,
                                               fc=color, ec='black', alpha=0.6, linewidth=0.5, zorder=6)
                        ax.add_patch(arrow)
                    
                    # Label interactive agents
                    if agent_i in interactive_indices:
                        label = f'Agent {int(agent_ids[agent_i])}'
                        ax.annotate(label, (agent_x, agent_y),
                                   textcoords="offset points", xytext=(5, 8),
                                   fontsize=8, fontweight='bold', color=color, zorder=10)
            
            # Draw history trails for interactive agents
            history_length = min(t + 1, 15)  # Show up to 1.5 seconds of history
            
            for idx, agent_i in enumerate(interactive_indices):
                color = agent_colors[idx % len(agent_colors)]
                
                # History trail (dashed line)
                hist_mask = all_valid[agent_i, max(0, t - history_length):t + 1] > 0
                hist_x = all_x[agent_i, max(0, t - history_length):t + 1][hist_mask]
                hist_y = all_y[agent_i, max(0, t - history_length):t + 1][hist_mask]
                if len(hist_x) > 1:
                    ax.plot(hist_x, hist_y, color=color, linewidth=2.5,
                           alpha=0.5, linestyle='--', zorder=3)
            
            # Get predictions at this timestep
            preds = predictions.get(t, {})
            gt_data = preds.get('ground_truth', {})
            agent_preds = preds.get('agents', {})
            
            # Draw ground truth future and predictions for each interactive agent
            for idx, (agent_id, gt) in enumerate(gt_data.items()):
                color = agent_colors[idx % len(agent_colors)]
                
                # Draw ground truth future (dotted line)
                if 'future_traj' in gt and len(gt['future_traj']) > 0:
                    fut = gt['future_traj']
                    ax.plot(fut[:, 0], fut[:, 1], 
                           color=self.COLORS['future_gt'], linewidth=1.5,
                           linestyle=':', alpha=0.5, zorder=2,
                           label=f'GT Future' if idx == 0 else None)
                
                # Draw predicted trajectories
                if agent_id in agent_preds:
                    pred_data = agent_preds[agent_id]
                    pred_traj = pred_data['pred_traj']  # [6, 80, 2]
                    pred_scores = pred_data['pred_scores']  # [6]
                    
                    if pred_scores is not None and len(pred_scores) > 0:
                        best_mode = np.argmax(pred_scores)
                        # Normalize scores for visualization
                        scores_norm = np.exp(pred_scores - np.max(pred_scores))
                        scores_norm = scores_norm / np.sum(scores_norm)
                    else:
                        best_mode = 0
                        scores_norm = np.ones(6) / 6
                    
                    # Draw all 6 modes with varying opacity
                    for mode in range(min(6, pred_traj.shape[0])):
                        traj = pred_traj[mode]
                        score = scores_norm[mode] if mode < len(scores_norm) else 0.1
                        
                        if len(traj) > 0:
                            # Use different color intensity based on score
                            mode_color = plt.cm.plasma(mode / 5)
                            alpha = 0.3 + 0.7 * score
                            linewidth = 1.5 + 3 * score
                            
                            ax.plot(traj[:, 0], traj[:, 1],
                                   color=mode_color, linewidth=linewidth,
                                   alpha=alpha, zorder=4)
                            
                            # Mark endpoint with star for best mode
                            if mode == best_mode:
                                ax.scatter(traj[-1, 0], traj[-1, 1],
                                          c=[color], s=100, marker='*',
                                          edgecolors='white', linewidths=1,
                                          zorder=5)
            
            # Title with receding horizon info
            time_sec = t * 0.1  # 10Hz dataset
            ax.set_title(
                f"Receding Horizon Prediction - {scenario_id[:20]}...\n"
                f"Prediction at t={t} ({time_sec:.1f}s) | {len(agent_preds)} agents | 6 modes each",
                fontsize=12, fontweight='bold'
            )
            
            ax.set_xlabel('X (meters)', fontsize=10)
            ax.set_ylabel('Y (meters)', fontsize=10)
            
            # Comprehensive legend
            legend_elements = [
                self.mpatches.Patch(facecolor=self.COLORS['agent_0'], edgecolor='black', label='Agent 1 (Interactive)'),
                self.mpatches.Patch(facecolor=self.COLORS['agent_1'], edgecolor='black', label='Agent 2 (Interactive)'),
                self.mpatches.Patch(facecolor=self.COLORS['vehicle'], edgecolor='black', label='Other Vehicles'),
                self.mpatches.Patch(facecolor=self.COLORS['pedestrian'], edgecolor='black', label='Pedestrians'),
                self.mpatches.Patch(facecolor=self.COLORS['cyclist'], edgecolor='black', label='Cyclists'),
                self.Line2D([0], [0], color=self.COLORS['agent_0'], linestyle='--', linewidth=2, label='History Trail'),
                self.Line2D([0], [0], color=self.COLORS['future_gt'], linestyle=':', linewidth=1.5, label='Ground Truth Future'),
                self.mpatches.Patch(facecolor=plt.cm.plasma(0.5), alpha=0.7, label='Predicted Modes (6)'),
                self.Line2D([0], [0], color=self.COLORS['lane'], linestyle='-', linewidth=1.5, label='Lanes'),
                self.Line2D([0], [0], color=self.COLORS['road_edge'], linestyle='-', linewidth=2, label='Road Edges'),
            ]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=8,
                     framealpha=0.9, ncol=1)
            
            return []
        
        # Create animation
        anim = animation.FuncAnimation(
            fig, animate, frames=len(timesteps),
            interval=1000/fps, blit=False, repeat=True
        )
        
        # Save movie
        output_path = self.output_dir / f'{scenario_id}_receding_horizon.mp4'
        print(f"  Rendering {len(timesteps)} frames...")
        
        try:
            writer = self.FFMpegWriter(fps=fps, bitrate=1800,
                                       codec='libx264', extra_args=['-pix_fmt', 'yuv420p'])
            anim.save(str(output_path), writer=writer, dpi=self.dpi)
            
            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"  ✓ Saved MP4: {output_path.name} ({file_size_mb:.1f} MB)")
            
            # Also save GIF
            gif_path = output_path.with_suffix('.gif')
            gif_fps = min(fps, 10)
            gif_writer = self.PillowWriter(fps=gif_fps)
            anim.save(str(gif_path), writer=gif_writer, dpi=max(self.dpi // 2, 50))
            
            gif_size_mb = gif_path.stat().st_size / (1024 * 1024)
            print(f"  ✓ Saved GIF: {gif_path.name} ({gif_size_mb:.1f} MB)")
            
            self.plt.close(fig)
            return output_path
            
        except Exception as e:
            print(f"  Error saving movie: {e}")
            import traceback
            traceback.print_exc()
            
            # Try saving as gif only as fallback
            try:
                gif_path = self.output_dir / f'{scenario_id}_receding_horizon.gif'
                anim.save(str(gif_path), writer='pillow', fps=fps)
                self.plt.close(fig)
                return gif_path
            except Exception as e2:
                print(f"  Error saving gif: {e2}")
                self.plt.close(fig)
                return None


def main():
    parser = argparse.ArgumentParser(description="M2I Full Receding Horizon Inference")
    parser.add_argument('--tfrecord', type=str,
                        default='/workspace/data/datasets/waymo_open_dataset/motion_v_1_3_0/processed/tf/validation_interactive/validation_interactive.tfrecord-00001-of-00150',
                        help='Path to TFRecord file')
    parser.add_argument('--num_scenarios', type=int, default=3,
                        help='Number of scenarios to process')
    parser.add_argument('--start_t', type=int, default=10,
                        help='First prediction timestep (default: 10)')
    parser.add_argument('--end_t', type=int, default=30,
                        help='Last prediction timestep (default: 30)')
    parser.add_argument('--step', type=int, default=5,
                        help='Timestep increment (default: 5)')
    parser.add_argument('--output', type=str,
                        default='/workspace/output/m2i_live/receding_horizon/predictions.pickle',
                        help='Output path for predictions')
    parser.add_argument('--movies_dir', type=str,
                        default='/workspace/models/pretrained/m2i/movies/receding_horizon',
                        help='Output directory for movies')
    parser.add_argument('--generate_movies', action='store_true',
                        help='Generate visualization movies')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    parser.add_argument('--full-pipeline', action='store_true',
                        help='Use full 3-stage M2I pipeline (DenseTNT + Relation + Conditional) - DEPRECATED due to global args issue')
    parser.add_argument('--subprocess-pipeline', action='store_true',
                        help='Use subprocess 3-stage M2I pipeline (recommended for full pipeline)')
    
    args = parser.parse_args()
    
    # Create predictor
    predictor = RecedingHorizonM2I(device=args.device)
    
    # Run receding horizon - also save scenario data for visualization
    if args.subprocess_pipeline:
        print("\nUsing SUBPROCESS 3-stage M2I pipeline (DenseTNT in main, Relation/Conditional in subprocesses)...")
        results = {}
        for scenario_idx in range(args.num_scenarios):
            result = predictor.run_receding_horizon_subprocess_pipeline(
                tfrecord_path=args.tfrecord,
                scenario_idx=scenario_idx,
                start_t=args.start_t,
                end_t=args.end_t,
                step=args.step,
                use_conditional=True,
            )
            if result:
                results[scenario_idx] = result
        
        # Save results
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'wb') as f:
                pickle.dump(results, f)
            print(f"\nSaved results to: {output_path}")
    elif args.full_pipeline:
        print("\nUsing FULL 3-stage M2I pipeline...")
        results = {}
        for scenario_idx in range(args.num_scenarios):
            result = predictor.run_receding_horizon_full_pipeline(
                tfrecord_path=args.tfrecord,
                scenario_idx=scenario_idx,
                start_t=args.start_t,
                end_t=args.end_t,
                step=args.step,
                use_conditional=True,
            )
            if result:
                results[scenario_idx] = result
        
        # Save results
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'wb') as f:
                pickle.dump(results, f)
            print(f"\nSaved results to: {output_path}")
    else:
        results = predictor.run_multiple_scenarios(
            tfrecord_path=args.tfrecord,
            num_scenarios=args.num_scenarios,
            start_t=args.start_t,
            end_t=args.end_t,
            step=args.step,
            output_path=args.output,
        )
    
    # Generate movies if requested
    if args.generate_movies:
        print(f"\n{'='*70}")
        print("GENERATING VISUALIZATION MOVIES")
        print(f"{'='*70}")
        
        visualizer = RecedingHorizonVisualizer(Path(args.movies_dir))
        
        # We need to re-parse scenarios to get the full data for visualization
        for scenario_idx, result in results.items():
            scenario_id = result['scenario_id']
            print(f"\nGenerating movie for scenario {scenario_id}...")
            
            # Re-parse scenario for visualization data
            scenario_data = predictor.parse_full_scenario(args.tfrecord, scenario_idx)
            if scenario_data:
                movie_path = visualizer.generate_movie(
                    scenario_data=scenario_data,
                    predictions=result['predictions'],
                    scenario_id=scenario_id,
                    fps=10,  # Match BEV movies: 10 FPS for ~9 second playback
                )
                if movie_path:
                    print(f"  Saved: {movie_path}")
    
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Processed {len(results)} scenarios")
    if args.subprocess_pipeline:
        print(f"Pipeline: Subprocess 3-stage M2I (DenseTNT + Relation + Conditional)")
    elif args.full_pipeline:
        print(f"Pipeline: Full 3-stage M2I (DenseTNT + Relation + Conditional)")
    else:
        print(f"Pipeline: DenseTNT only")
    
    # Compute aggregate statistics
    all_ades = []
    all_fdes = []
    
    for scenario_idx, result in results.items():
        for t, preds in result['predictions'].items():
            for agent_id, agent_preds in preds['agents'].items():
                if agent_id in preds['ground_truth']:
                    gt = preds['ground_truth'][agent_id]['future_traj']
                    pred = agent_preds['pred_traj']
                    scores = agent_preds['pred_scores']
                    
                    if scores is not None:
                        best_mode = np.argmax(scores)
                    else:
                        best_mode = 0
                    
                    pred_best = pred[best_mode, :len(gt)]
                    if len(pred_best) > 0 and len(gt) > 0:
                        errors = np.linalg.norm(pred_best - gt, axis=-1)
                        all_ades.append(errors.mean())
                        all_fdes.append(errors[-1])
    
    if all_ades:
        print(f"\nAggregate metrics across all timesteps:")
        print(f"  Mean ADE: {np.mean(all_ades):.3f} m")
        print(f"  Mean FDE: {np.mean(all_fdes):.3f} m")


if __name__ == '__main__':
    main()
