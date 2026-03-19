#!/usr/bin/env python3
"""
M2I Receding Horizon Inference

This script demonstrates running M2I predictions at multiple timesteps,
simulating what an autonomous vehicle would do in real-time:

At each timestep t:
  1. Observe history [t-10 : t]
  2. Predict future [t+1 : t+80]
  3. Advance to t+1, incorporate new observation
  4. Repeat

This is in contrast to the standard Waymo benchmark which predicts only once at t=10.

Usage:
    python m2i_receding_horizon.py --num_scenarios 5 --prediction_steps 10

Author: RECTOR Project
"""

import argparse
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import torch

# Add paths
M2I_SRC = Path("/workspace/externals/M2I/src")
sys.path.insert(0, str(M2I_SRC))
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

import tensorflow as tf


class RecedingHorizonPredictor:
    """
    M2I predictor that runs at multiple timesteps (receding horizon).
    
    Key Concept:
    - Standard M2I: Predicts once at t=10 for t=11-90
    - Receding Horizon: Predicts at t=10, t=11, t=12, ... using updated history
    
    This simulates real-time operation where:
    1. At each timestep, you observe new agent positions
    2. You re-run prediction with updated history
    3. Predictions should improve as you get more current data
    """
    
    def __init__(self, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = None
        
    def load_model(self, model_path: Path):
        """Load DenseTNT model for marginal predictions."""
        from modeling.vectornet import VectorNet
        
        # Build args for DenseTNT
        args = self._build_densetnt_args()
        
        # Create model
        self.model = VectorNet(args)
        
        # Load weights
        checkpoint = torch.load(model_path, map_location='cpu')
        model_dict = self.model.state_dict()
        
        loaded = 0
        for key, value in checkpoint['model'].items():
            clean_key = key.replace('module.', '') if key.startswith('module.') else key
            if clean_key in model_dict and model_dict[clean_key].shape == value.shape:
                model_dict[clean_key] = value
                loaded += 1
        
        self.model.load_state_dict(model_dict)
        self.model.to(self.device)
        self.model.eval()
        print(f"Loaded {loaded}/{len(checkpoint['model'])} weights")
        
        return args
    
    def _build_densetnt_args(self):
        """Build args for DenseTNT model."""
        import utils
        
        args = utils.Args()
        args.hidden_size = 128
        args.future_frame_num = 80
        args.mode_num = 6
        args.sub_graph_batch_size = 4096
        args.eval_batch_size = 1
        args.core_num = 1
        args.agent_type = 'vehicle'
        args.sub_graph_depth = 3
        args.global_graph_depth = 1
        args.hidden_dropout_prob = 0.1
        args.initializer_range = 0.02
        args.max_distance = 50.0
        args.nms_threshold = 7.2
        
        # other_params with raster support
        from m2i_live_inference import OtherParams
        args.other_params = OtherParams([
            'densetnt',
            'raster',
            'goals_2D',
            'enhance_global_graph',
            'laneGCN',
            'laneGCN-4',
            'nms',
            'point_sub_graph',
        ], {'eval_time': 80})
        
        return args
    
    def parse_scenario_data(self, tfrecord_path: str, scenario_idx: int = 0) -> Dict:
        """
        Parse a scenario and return all 91 timesteps of data.
        
        Returns dict with:
            - positions: [128, 91, 2] - x, y for all agents at all times
            - headings: [128, 91] - heading angles
            - valid: [128, 91] - validity masks
            - types: [128] - agent types
            - ids: [128] - agent IDs
            - tracks_to_predict: [2] - indices of interactive agents
            - roadgraph: road features
            - scenario_id: string
        """
        # Use the flat format parser
        from waymo_flat_parser import features_description_flat
        
        dataset = tf.data.TFRecordDataset(tfrecord_path)
        
        for idx, record in enumerate(dataset):
            if idx != scenario_idx:
                continue
            
            # Parse the record
            parsed = tf.io.parse_single_example(record, features_description_flat)
            
            # Convert to numpy
            def to_np(key):
                return parsed[key].numpy()
            
            # Get scenario ID
            scenario_id = to_np('scenario/id')
            
            # Get agent data - note: flat format has shape [128*timesteps] 
            # We need to reshape appropriately
            n_agents = 128
            n_past = 10
            n_future = 80
            
            # Past positions [128*10] -> [128, 10]
            past_x = to_np('state/past/x').reshape(n_agents, n_past)
            past_y = to_np('state/past/y').reshape(n_agents, n_past)
            past_valid = to_np('state/past/valid').reshape(n_agents, n_past)
            
            # Current positions [128] 
            current_x = to_np('state/current/x').reshape(n_agents, 1)
            current_y = to_np('state/current/y').reshape(n_agents, 1)
            current_valid = to_np('state/current/valid').reshape(n_agents, 1)
            
            # Future positions [128*80] -> [128, 80]
            future_x = to_np('state/future/x').reshape(n_agents, n_future)
            future_y = to_np('state/future/y').reshape(n_agents, n_future)
            future_valid = to_np('state/future/valid').reshape(n_agents, n_future)
            
            # Past is stored most recent first, so reverse to chronological order
            past_x = past_x[:, ::-1]
            past_y = past_y[:, ::-1]
            past_valid = past_valid[:, ::-1]
            
            # Stack all timesteps: past (10) + current (1) + future (80) = 91
            all_x = np.concatenate([past_x, current_x, future_x], axis=1)
            all_y = np.concatenate([past_y, current_y, future_y], axis=1)
            positions = np.stack([all_x, all_y], axis=-1)
            
            # Validity masks
            all_valid = np.concatenate([past_valid, current_valid, future_valid], axis=1)
            
            # Agent types and IDs
            agent_types = to_np('state/type')
            agent_ids = to_np('state/id')
            
            # Tracks to predict (which agents are interactive)
            tracks_to_predict_mask = to_np('state/tracks_to_predict')
            tracks_indices = np.where(tracks_to_predict_mask > 0)[0]
            
            return {
                'positions': positions,   # [128, 91, 2]
                'valid': all_valid,        # [128, 91]
                'types': agent_types,
                'ids': agent_ids,
                'tracks_to_predict': tracks_indices,
                'scenario_id': scenario_id,
            }
        
        return None
    
    def create_input_at_timestep(self, scenario_data: Dict, current_t: int) -> Dict:
        """
        Create M2I-compatible input for prediction at timestep current_t.
        
        Args:
            scenario_data: Full scenario data with all 91 timesteps
            current_t: Current timestep (10-80 for valid prediction window)
                       - Need 10 history steps (t-10 to t-1) + current (t)
                       - Will predict 80 future steps (t+1 to t+80)
        
        Returns:
            Dict with 'past', 'current', 'future' formatted for M2I
        """
        positions = scenario_data['positions']  # [128, 91, 2]
        valid = scenario_data['valid']          # [128, 91]
        
        # Extract windows
        # Past: 10 timesteps before current
        past_start = current_t - 10
        past_end = current_t
        past_positions = positions[:, past_start:past_end, :]  # [128, 10, 2]
        past_valid = valid[:, past_start:past_end]              # [128, 10]
        
        # Current: single timestep
        current_positions = positions[:, current_t:current_t+1, :]  # [128, 1, 2]
        current_valid = valid[:, current_t:current_t+1]              # [128, 1]
        
        # Future: up to 80 timesteps after current
        future_start = current_t + 1
        future_end = min(current_t + 81, 91)  # Don't exceed available data
        future_positions = positions[:, future_start:future_end, :]  # [128, <=80, 2]
        future_valid = valid[:, future_start:future_end]              # [128, <=80]
        
        # Pad future if needed
        if future_positions.shape[1] < 80:
            pad_len = 80 - future_positions.shape[1]
            future_positions = np.pad(future_positions, 
                                      ((0, 0), (0, pad_len), (0, 0)), 
                                      mode='edge')
            future_valid = np.pad(future_valid, 
                                  ((0, 0), (0, pad_len)), 
                                  mode='constant', constant_values=0)
        
        return {
            'past': {
                'x': past_positions[:, :, 0],      # [128, 10]
                'y': past_positions[:, :, 1],      # [128, 10]
                'valid': past_valid,               # [128, 10]
            },
            'current': {
                'x': current_positions[:, 0, 0],   # [128]
                'y': current_positions[:, 0, 1],   # [128]
                'valid': current_valid[:, 0],      # [128]
            },
            'future': {
                'x': future_positions[:, :, 0],    # [128, 80]
                'y': future_positions[:, :, 1],    # [128, 80]
                'valid': future_valid,             # [128, 80]
            },
            'types': scenario_data['types'],
            'ids': scenario_data['ids'],
            'tracks_to_predict': scenario_data['tracks_to_predict'],
            'scenario_id': scenario_data['scenario_id'],
            'current_timestep': current_t,
            'roadgraph_xyz': scenario_data.get('roadgraph_xyz'),
            'roadgraph_type': scenario_data.get('roadgraph_type'),
        }
    
    def run_receding_horizon(
        self,
        tfrecord_path: str,
        scenario_idx: int = 0,
        start_t: int = 10,
        end_t: int = 20,
        model_path: Path = None,
    ) -> Dict[int, Dict]:
        """
        Run predictions at multiple timesteps using receding horizon.
        
        Args:
            tfrecord_path: Path to TFRecord file
            scenario_idx: Which scenario in the file to process
            start_t: First timestep to predict at (must be >= 10 for history)
            end_t: Last timestep to predict at (must be <= 80 for future window)
            model_path: Path to DenseTNT model
        
        Returns:
            Dict mapping timestep -> predictions at that timestep
        """
        print(f"\n{'='*60}")
        print("RECEDING HORIZON PREDICTION")
        print(f"{'='*60}")
        print(f"Prediction timesteps: {start_t} to {end_t}")
        print(f"At each timestep t, observe [t-10:t] and predict [t+1:t+80]")
        
        # Load scenario data
        print(f"\nLoading scenario {scenario_idx} from {tfrecord_path}...")
        scenario_data = self.parse_scenario_data(tfrecord_path, scenario_idx)
        if scenario_data is None:
            raise ValueError(f"Could not load scenario {scenario_idx}")
        
        scenario_id = scenario_data['scenario_id']
        if isinstance(scenario_id, bytes):
            scenario_id = scenario_id.decode()
        print(f"Scenario ID: {scenario_id}")
        
        # Load model if needed
        if model_path and self.model is None:
            args = self.load_model(model_path)
        
        # Run predictions at each timestep
        all_predictions = {}
        
        for t in range(start_t, end_t + 1):
            print(f"\n--- Timestep t={t} ---")
            print(f"  History window: [{t-10} : {t}]")
            print(f"  Prediction window: [{t+1} : {min(t+80, 90)}]")
            
            # Create input for this timestep
            input_data = self.create_input_at_timestep(scenario_data, t)
            
            # Get ground truth at current position (for validation)
            tracks = scenario_data['tracks_to_predict']
            for agent_idx in tracks:
                if agent_idx < len(scenario_data['ids']):
                    agent_id = scenario_data['ids'][agent_idx]
                    curr_pos = scenario_data['positions'][agent_idx, t, :]
                    print(f"  Agent {agent_id} current position: ({curr_pos[0]:.2f}, {curr_pos[1]:.2f})")
            
            # Store the input data (for demo - in production you'd run the model)
            all_predictions[t] = {
                'input': input_data,
                'scenario_id': scenario_id,
                'current_timestep': t,
                'history_window': (t-10, t),
                'prediction_window': (t+1, min(t+80, 90)),
            }
        
        return all_predictions


def demonstrate_concept():
    """
    Demonstrate the difference between single-shot and receding horizon prediction.
    """
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    SINGLE-SHOT vs RECEDING HORIZON                            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  SINGLE-SHOT (Current Implementation):                                       ║
║  ─────────────────────────────────────                                       ║
║  • Predict ONCE at t=10                                                      ║
║  • Use history [t=0 → t=10]                                                  ║
║  • Predict future [t=11 → t=90]                                              ║
║  • Never update predictions as time passes                                   ║
║                                                                              ║
║  Timeline:                                                                   ║
║  ├──────────────┬─────────────────────────────────────────────────────────┤ ║
║  0              10                                                       90  ║
║  │   history    │                    prediction                          │  ║
║  └──────────────┴─────────────────────────────────────────────────────────┘  ║
║                                                                              ║
║                                                                              ║
║  RECEDING HORIZON (What Autonomous Vehicles Actually Do):                    ║
║  ──────────────────────────────────────────────────────                      ║
║  • At EACH timestep t, make a new prediction                                 ║
║  • Use history [t-10 → t]                                                    ║
║  • Predict future [t+1 → t+80]                                               ║
║  • As time passes, predictions improve with new observations                 ║
║                                                                              ║
║  Timeline (predictions at t=10, 11, 12):                                     ║
║                                                                              ║
║  t=10: ├──────────────┬─────────────────────────────────────────────────┤   ║
║        0              10                                               90    ║
║        │   history    │                prediction                       │   ║
║                                                                              ║
║  t=11:  ├──────────────┬────────────────────────────────────────────────┤   ║
║         1              11                                              91    ║
║         │   history    │               prediction                       │   ║
║                        ↑                                                     ║
║                   New observation incorporated!                              ║
║                                                                              ║
║  t=12:   ├──────────────┬───────────────────────────────────────────────┤   ║
║          2              12                                             92    ║
║          │   history    │              prediction                       │   ║
║                         ↑                                                    ║
║                    Even more current data!                                   ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  WHY RECEDING HORIZON IS BETTER:                                             ║
║  • Predictions incorporate latest observations                               ║
║  • Can react to unexpected agent behavior                                    ║
║  • Errors don't accumulate as badly over time                               ║
║  • More realistic for actual AV deployment                                   ║
║                                                                              ║
║  WHY WAYMO BENCHMARK USES SINGLE-SHOT:                                       ║
║  • Simpler evaluation (one prediction per scenario)                          ║
║  • Standardized comparison across methods                                    ║
║  • Focuses on prediction quality at fixed horizon                            ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")


def main():
    parser = argparse.ArgumentParser(description="M2I Receding Horizon Inference")
    parser.add_argument('--tfrecord', type=str, 
                        default='/workspace/data/datasets/waymo_open_dataset/motion_v_1_3_0/processed/tf/validation_interactive/validation_interactive.tfrecord-00001-of-00150',
                        help='Path to TFRecord file')
    parser.add_argument('--scenario_idx', type=int, default=0,
                        help='Which scenario in the file to process')
    parser.add_argument('--start_t', type=int, default=10,
                        help='First timestep to predict at (default: 10)')
    parser.add_argument('--end_t', type=int, default=20,
                        help='Last timestep to predict at (default: 20)')
    parser.add_argument('--explain', action='store_true',
                        help='Show explanation of single-shot vs receding horizon')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    if args.explain:
        demonstrate_concept()
        return
    
    # Create predictor
    predictor = RecedingHorizonPredictor(device=args.device)
    
    # Run receding horizon demo
    predictions = predictor.run_receding_horizon(
        tfrecord_path=args.tfrecord,
        scenario_idx=args.scenario_idx,
        start_t=args.start_t,
        end_t=args.end_t,
    )
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Generated predictions at {len(predictions)} timesteps")
    print(f"Timesteps: {list(predictions.keys())}")
    
    print("""
To run full receding horizon inference with the model, you would need to:
1. Load the DenseTNT model
2. For each timestep, create the proper M2I input format
3. Run inference and store predictions
4. Optionally run relation and conditional stages at each timestep

This requires modifying the dataset loader to accept arbitrary time windows,
which is a more involved change to the M2I codebase.
""")


if __name__ == '__main__':
    main()
