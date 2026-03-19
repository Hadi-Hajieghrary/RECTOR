#!/usr/bin/env python3
"""
Subprocess script for M2I Relation inference.

This script is designed to be called as a subprocess to avoid global args
conflicts in the M2I VectorNet implementation.

Usage:
    python subprocess_relation.py --input_pickle <path> --output_pickle <path> --device cuda
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


def build_relation_args():
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
    args.do_eval = False
    args.do_test = True  # Test mode for inference without GT labels
    args.classify_sub_goals = False
    args.eval_params = []
    args.infMLP = 0
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
    
    utils.args = args
    return args


def load_relation_model(device):
    """Load Relation V2V model."""
    from modeling.vectornet import VectorNet
    
    relation_path = Path("/workspace/models/pretrained/m2i/models/relation_v2v/model.25.bin")
    
    print("Loading Relation V2V model...")
    args = build_relation_args()
    model = VectorNet(args)
    
    checkpoint = torch.load(relation_path, map_location='cpu')
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


def run_relation_inference(model, args, mapping1, mapping2, device):
    """
    Run Relation model to determine influencer/reactor relationship.
    
    Returns:
        relation_label: 0=no relation, 1=agent1->agent2, 2=agent2->agent1
    """
    import utils
    import globals as m2i_globals
    import modeling.vectornet as vectornet_module
    
    # Ensure args are set
    utils.args = args
    vectornet_module.args = args
    
    # Clear previous relation predictions
    m2i_globals.sun_1_pred_relations = {}
    
    # Create pair mapping by copying mapping1 and adding reactor info
    pair_mapping = {}
    for key, value in mapping1.items():
        if isinstance(value, np.ndarray):
            pair_mapping[key] = value.copy()
        elif isinstance(value, list):
            pair_mapping[key] = list(value)
        else:
            pair_mapping[key] = value
    
    # Add reactor information
    pair_mapping['reactor'] = {
        'agent_id': mapping2.get('object_id'),
        'cent_x': mapping2.get('cent_x'),
        'cent_y': mapping2.get('cent_y'),
        'angle': mapping2.get('angle'),
        'polyline_spans': mapping2.get('polyline_spans', []),
        'matrix': mapping2.get('matrix', []),
    }
    
    # Add dummy interaction_label for inference (required by model)
    pair_mapping['interaction_label'] = 0  # Dummy value, not used during inference
    pair_mapping['influencer_idx'] = 0  # Dummy value
    
    # Ensure scenario_id is set (used by globals to store result)
    if 'scenario_id' not in pair_mapping:
        pair_mapping['scenario_id'] = b'temp_scenario'
    elif isinstance(pair_mapping['scenario_id'], str):
        pair_mapping['scenario_id'] = pair_mapping['scenario_id'].encode()
    
    with torch.no_grad():
        try:
            outputs = model([pair_mapping], device)
            
            # Check if relation was stored in globals (preferred method)
            if m2i_globals.sun_1_pred_relations:
                for scenario_key, result in m2i_globals.sun_1_pred_relations.items():
                    if isinstance(result, list) and len(result) >= 1:
                        relation_label = int(result[0])
                        return relation_label
            
            # Fallback: check direct output
            if outputs is not None and len(outputs) >= 3:
                relation_logits = outputs[2]
                if relation_logits is not None:
                    relation_label = int(np.argmax(relation_logits[0]))
                    return relation_label
            
            return 0
            
        except Exception as e:
            print(f"  Relation inference error: {e}")
            import traceback
            traceback.print_exc()
            return 0


def main():
    parser = argparse.ArgumentParser(description="Subprocess Relation Inference")
    parser.add_argument('--input_pickle', type=str, required=True,
                        help='Input pickle with DenseTNT predictions and mappings')
    parser.add_argument('--output_pickle', type=str, required=True,
                        help='Output pickle for relation predictions')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Relation subprocess: device={device}")
    
    # Load input data
    with open(args.input_pickle, 'rb') as f:
        input_data = pickle.load(f)
    
    # Load model
    model, model_args = load_relation_model(device)
    
    # Process each timestep's agent pairs
    results = {}
    
    for timestep, data in input_data.items():
        mappings = data.get('mappings', {})
        agent_ids = list(mappings.keys())
        
        timestep_relations = []
        
        # Process all pairs
        for i in range(len(agent_ids)):
            for j in range(i + 1, len(agent_ids)):
                id1, id2 = agent_ids[i], agent_ids[j]
                
                if id1 not in mappings or id2 not in mappings:
                    continue
                
                relation_label = run_relation_inference(
                    model, model_args,
                    mappings[id1], mappings[id2],
                    device
                )
                
                timestep_relations.append({
                    'agent1_id': id1,
                    'agent2_id': id2,
                    'relation': relation_label,
                })
                
                if relation_label != 0:
                    inf_id = id1 if relation_label == 1 else id2
                    react_id = id2 if relation_label == 1 else id1
                    print(f"  t={timestep}: Agent {inf_id} influences Agent {react_id}")
        
        results[timestep] = timestep_relations
    
    # Save results
    with open(args.output_pickle, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Relation results saved to: {args.output_pickle}")


if __name__ == '__main__':
    main()
