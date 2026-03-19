#!/usr/bin/env python3
"""
Subprocess script for M2I Conditional inference.

This script is designed to be called as a subprocess to avoid global args
conflicts in the M2I VectorNet implementation.

Usage:
    python subprocess_conditional.py --input_pickle <path> --output_pickle <path> --device cuda
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

# Environment setup
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')

# Add paths
M2I_SRC = Path("/workspace/externals/M2I/src")
sys.path.insert(0, str(M2I_SRC))

import numpy as np
import torch


class OtherParams(list):
    """Hybrid list/dict for M2I's other_params."""
    def __init__(self, items=None, defaults=None):
        super().__init__(items or [])
        self._defaults = defaults or {}
    
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


def build_conditional_args():
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
    args.do_eval = False
    args.do_test = True  # Test mode for inference without GT labels
    args.classify_sub_goals = False
    args.eval_params = []
    args.infMLP = 6
    args.inf_pred_num = 6
    args.image = None
    
    # Architecture params
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
    args.single_agent = False
    args.agent_type = 'vehicle'
    args.inter_agent_types = None
    
    # Logging directories
    output_dir = Path('/workspace/output/m2i_live/receding_horizon')
    output_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir = str(output_dir / 'logs')
    args.output_dir = str(output_dir)
    args.temp_file_dir = str(output_dir / 'temp')
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.temp_file_dir, exist_ok=True)
    
    # Conditional-specific params
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
    ], {'eval_time': 80})
    
    utils.args = args
    return args


def load_conditional_model(device):
    """Load Conditional V2V model."""
    from modeling.vectornet import VectorNet
    
    conditional_path = Path("/workspace/models/pretrained/m2i/models/conditional_v2v/model.29.bin")
    
    print("Loading Conditional V2V model...")
    args = build_conditional_args()
    model = VectorNet(args)
    
    checkpoint = torch.load(conditional_path, map_location='cpu')
    model_dict = model.state_dict()
    
    loaded = 0
    for key, value in checkpoint.items():
        clean_key = key.replace('module.', '') if key.startswith('module.') else key
        if clean_key in model_dict and model_dict[clean_key].shape == value.shape:
            model_dict[clean_key] = value
            loaded += 1
    
    model.load_state_dict(model_dict)
    model.to(device)
    model.eval()
    print(f"  Loaded {loaded}/{len(checkpoint)} weights")
    
    return model, args


def run_conditional_inference(model, args, reactor_mapping, influencer_traj, influencer_scores, device):
    """
    Run Conditional model to predict reactor trajectory given influencer.
    
    Returns:
        pred_traj: [6, 80, 2] conditional predictions
        pred_scores: [6] confidence scores
    """
    import utils
    import modeling.vectornet as vectornet_module
    
    # Ensure args are set
    utils.args = args
    vectornet_module.args = args
    
    # Create conditional mapping
    cond_mapping = reactor_mapping.copy()
    
    # Add influencer trajectory info
    if influencer_traj is not None:
        if influencer_scores is not None:
            best_mode = np.argmax(influencer_scores)
            best_traj = influencer_traj[best_mode]
        else:
            best_traj = influencer_traj[0]
        
        cond_mapping['influencer_traj'] = best_traj.astype(np.float32)
        cond_mapping['influencer_all_trajs'] = influencer_traj.astype(np.float32)
        
        if influencer_scores is not None:
            cond_mapping['influencer_scores'] = influencer_scores.astype(np.float32)
        
        # Add gt_influencer_traj_idx (required by conditional model)
        # This should point to the influencer's polyline in the matrix
        map_start = cond_mapping.get('map_start_polyline_idx', 0)
        cond_mapping['gt_influencer_traj_idx'] = map_start + 1  # After map polylines
    
    with torch.no_grad():
        try:
            outputs = model([cond_mapping], device)
            
            # Handle various output formats
            if outputs is None:
                return None, None
            
            pred_traj, pred_score = None, None
            
            if isinstance(outputs, tuple) and len(outputs) >= 2:
                raw_traj, raw_score = outputs[0], outputs[1]
                
                # Handle trajectory output
                if raw_traj is not None:
                    if isinstance(raw_traj, np.ndarray):
                        if raw_traj.ndim >= 3:
                            pred_traj = raw_traj[0] if raw_traj.shape[0] > 0 else raw_traj
                        else:
                            pred_traj = raw_traj
                    elif isinstance(raw_traj, list) and len(raw_traj) > 0:
                        pred_traj = np.array(raw_traj[0])
                    elif hasattr(raw_traj, 'cpu'):
                        pred_traj = raw_traj.cpu().numpy()
                
                # Handle score output
                if raw_score is not None:
                    if isinstance(raw_score, np.ndarray):
                        if raw_score.ndim >= 1 and raw_score.shape[0] > 0:
                            pred_score = raw_score[0] if raw_score.ndim > 1 else raw_score
                        else:
                            pred_score = raw_score
                    elif isinstance(raw_score, list) and len(raw_score) > 0:
                        pred_score = np.array(raw_score[0])
                    elif hasattr(raw_score, 'cpu'):
                        pred_score = raw_score.cpu().numpy()
            
            return pred_traj, pred_score
            
        except Exception as e:
            print(f"  Conditional inference error: {e}")
            import traceback
            traceback.print_exc()
            return None, None


def main():
    parser = argparse.ArgumentParser(description="Subprocess Conditional Inference")
    parser.add_argument('--input_pickle', type=str, required=True,
                        help='Input pickle with predictions, mappings, and relations')
    parser.add_argument('--output_pickle', type=str, required=True,
                        help='Output pickle for conditional predictions')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Conditional subprocess: device={device}")
    
    # Load input data
    with open(args.input_pickle, 'rb') as f:
        input_data = pickle.load(f)
    
    # Load model
    model, model_args = load_conditional_model(device)
    
    # Process each timestep
    results = {}
    
    for timestep, data in input_data.items():
        mappings = data.get('mappings', {})
        predictions = data.get('predictions', {})
        relations = data.get('relations', [])
        
        timestep_conditional = {}
        
        # Process each relation to generate conditional predictions
        for rel in relations:
            relation_label = rel['relation']
            if relation_label == 0:
                continue
            
            # Determine influencer/reactor
            if relation_label == 1:
                influencer_id = rel['agent1_id']
                reactor_id = rel['agent2_id']
            else:
                influencer_id = rel['agent2_id']
                reactor_id = rel['agent1_id']
            
            # Get influencer prediction and reactor mapping
            if influencer_id not in predictions or reactor_id not in mappings:
                continue
            
            influencer_pred = predictions[influencer_id].get('pred_traj')
            influencer_scores = predictions[influencer_id].get('pred_scores')
            reactor_mapping = mappings[reactor_id]
            
            if influencer_pred is None:
                continue
            
            # Run conditional inference
            cond_pred, cond_scores = run_conditional_inference(
                model, model_args,
                reactor_mapping,
                influencer_pred,
                influencer_scores,
                device
            )
            
            if cond_pred is not None:
                timestep_conditional[reactor_id] = {
                    'pred_traj': cond_pred,
                    'pred_scores': cond_scores,
                    'influencer_id': influencer_id,
                }
                print(f"  t={timestep}: Conditional prediction for Agent {reactor_id} (influenced by {influencer_id})")
        
        results[timestep] = timestep_conditional
    
    # Save results
    with open(args.output_pickle, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Conditional results saved to: {args.output_pickle}")


if __name__ == '__main__':
    main()
