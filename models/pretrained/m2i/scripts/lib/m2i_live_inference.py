#!/usr/bin/env python3
"""
M2I Live Inference Pipeline - NO PRE-COMPUTED FILES

This script runs the M2I models (DenseTNT, Relation, Conditional) directly
on TFRecord data WITHOUT using any pre-computed pickle files.

Models:
- DenseTNT (model.24.bin): Marginal trajectory predictor for ALL agents
- Relation V2V (model.25.bin): Relation predictor (who influences whom)  
- Conditional V2V (model.29.bin): Conditional trajectory predictor

Usage:
    python m2i_live_inference.py --num_scenarios 10 \
        --data_dir /workspace/data/datasets/waymo_open_dataset/motion_v_1_3_0/processed/tf/validation_interactive \
        --output_dir /workspace/output/m2i_live \
        --device cuda

Author: RECTOR Project
"""

import argparse
import glob
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


class OtherParams(list):
    """
    A list subclass that also supports dict-like access.
    
    M2I uses other_params inconsistently - sometimes as a list (for 'in' checks)
    and sometimes as a dict (for .get() calls). This class supports both.
    
    Note: The remove() method is overridden to be a no-op because M2I's dataset_waymo.py
    mutates other_params by removing 'save_rst', which breaks subsequent scenarios.
    """
    def __init__(self, items=None, defaults=None):
        """
        Initialize with a list of items and optional dict of default values.
        
        Args:
            items: List of string flags (e.g., ['densetnt', 'raster'])
            defaults: Dict of key->value for .get() fallbacks (e.g., {'eval_time': 80})
        """
        super().__init__(items or [])
        self._defaults = defaults or {}
        self._removed = set()  # Track "removed" items without actually removing
    
    def get(self, key, default=None):
        """Dict-like get method for backward compatibility."""
        if key in self:
            return True
        return self._defaults.get(key, default)
    
    def __getitem__(self, key):
        """Support both list indexing and dict key access."""
        if isinstance(key, int):
            return super().__getitem__(key)
        if key in self:
            return True
        if key in self._defaults:
            return self._defaults[key]
        raise KeyError(key)
    
    def remove(self, item):
        """
        Override remove to track removal but not actually remove the item.
        This prevents the bug where save_rst gets permanently removed.
        """
        self._removed.add(item)
        # Don't actually remove - M2I calls this but we need the item to persist
        # for subsequent scenario processing


# Add M2I source to path
M2I_SRC = Path("/workspace/externals/M2I/src")
M2I_CONFIG_DIR = M2I_SRC.parent / "configs"
sys.path.insert(0, str(M2I_SRC))

# Add our custom parser path
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings('ignore')

# Patch the waymo_tutorial parser to use flat format BEFORE importing M2I modules
def patch_waymo_parser():
    """Patch waymo_tutorial to use flat format parser for RECTOR workspace TFRecords."""
    import waymo_tutorial
    from waymo_flat_parser import _parse_flat
    waymo_tutorial._parse = _parse_flat
    print("  Patched waymo_tutorial._parse to use flat format parser")

# Import waymo_tutorial first, then patch it
import waymo_tutorial
patch_waymo_parser()


# =============================================================================
# Configuration
# =============================================================================

class Config:
    """M2I Pipeline Configuration - No pre-computed files."""
    
    # Paths
    MODEL_DIR = Path("/workspace/models/pretrained/m2i/models")
    DATA_DIR = Path("/workspace/data/datasets/waymo_open_dataset/motion_v_1_3_0/processed/tf/validation_interactive")
    OUTPUT_DIR = Path("/workspace/output/m2i_live")
    
    # Model files
    DENSETNT_MODEL = MODEL_DIR / "densetnt" / "model.24.bin"
    RELATION_MODEL = MODEL_DIR / "relation_v2v" / "model.25.bin"
    CONDITIONAL_MODEL = MODEL_DIR / "conditional_v2v" / "model.29.bin"
    
    # Model parameters
    HIDDEN_SIZE = 128
    MODE_NUM = 6
    FUTURE_FRAMES = 80


# =============================================================================
# M2I Dataset and Model Integration
# =============================================================================

class M2ILiveRunner:
    """
    Run M2I inference LIVE without pre-computed files.
    
    This class directly uses the M2I Dataset and VectorNet to:
    1. Run DenseTNT marginal predictions for all agents
    2. Run Relation prediction to determine influencer/reactor
    3. Run Conditional prediction for reactor trajectories
    """
    
    def __init__(self, device: str = 'cuda', seed: int = 42):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.args_cache = {}
        self.seed = seed
        
        # Set reproducible random seeds
        self._set_seeds(seed)
        
        print(f"\n{'='*60}")
        print("M2I LIVE INFERENCE - NO PRE-COMPUTED FILES")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    def _set_seeds(self, seed: int):
        """Set random seeds for reproducibility."""
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def _build_args_for_densetnt(self, data_dir: Path, output_dir: Path):
        """Build args for DenseTNT marginal predictor."""
        import utils
        
        args = utils.Args()
        args.data_dir = [str(data_dir)]
        args.output_dir = str(output_dir)
        args.log_dir = str(output_dir)
        # Use shared temp_file_dir to ensure all stages process the same scenarios
        args.temp_file_dir = os.path.join(str(output_dir.parent), 'shared_temp')
        
        # Eval mode
        args.do_eval = True
        args.do_train = False
        args.do_test = False
        args.debug_mode = False
        args.debug = False
        
        # Dataset
        args.waymo = True
        args.nuscenes = False
        args.argoverse = False
        
        # Model params (from densetnt.yaml)
        args.hidden_size = 128
        args.future_frame_num = 80
        args.future_test_frame_num = 80
        args.mode_num = 6
        args.sub_graph_batch_size = 4096
        args.train_batch_size = 1
        args.eval_batch_size = 1
        args.core_num = 1
        args.infMLP = 0
        args.nms_threshold = 7.2
        args.agent_type = 'vehicle'  # Can be: vehicle, pedestrian, cyclist
        args.inter_agent_types = None
        args.single_agent = True
        
        # Model architecture params
        args.sub_graph_depth = 3
        args.global_graph_depth = 1
        args.hidden_dropout_prob = 0.1
        args.initializer_range = 0.02
        args.max_distance = 50.0
        args.no_sub_graph = False
        args.no_agents = False
        args.attention_decay = False
        args.use_map = False
        args.reuse_temp_file = False
        args.old_version = False
        args.use_centerline = False
        args.autoregression = None
        args.lstm = False
        args.add_prefix = None
        args.multi = None
        args.placeholder = 0.0
        args.method_span = [0, 1]
        args.visualize = False
        args.train_extra = False
        args.not_use_api = False
        args.no_cuda = False
        args.learning_rate = 0.001
        args.weight_decay = 0.3
        args.num_train_epochs = 30
        args.joint_target_each = 80
        args.joint_target_type = "no"
        args.classify_sub_goals = False
        args.traj_loss_coeff = 1.0
        args.short_term_loss_coeff = 0.0
        args.relation_pred_threshold = 0.9
        args.direct_relation_path = None
        args.all_agent_ids_path = None
        args.vehicle_r_pred_threshold = None
        args.config = None
        
        # Model recovery
        args.model_recover_path = str(Config.DENSETNT_MODEL)
        
        # No relation/conditional files - we compute everything live
        args.relation_file_path = None
        args.relation_pred_file_path = None
        args.influencer_pred_file_path = None
        
        # Other params for DenseTNT (from densetnt.yaml) - Use OtherParams for hybrid list/dict access
        # IMPORTANT: The pretrained model.24.bin was trained WITH raster input (CNN encoder)
        # so we MUST include 'raster' to match the model architecture
        args.other_params = OtherParams([
            'l1_loss',
            'densetnt',
            'goals_2D',
            'enhance_global_graph',
            'laneGCN',
            'point_sub_graph',
            'laneGCN-4',
            'stride_10_2',
            'train_pair_interest',  # Process interactive pairs
            'save_rst',  # Save results
            'raster',  # REQUIRED: Model was trained with raster CNN encoder
        ])
        
        args.eval_params = []
        args.train_params = []
        args.seed = 42
        args.cuda_visible_device_num = None
        args.distributed_training = 1
        args.validation_model = 24
        args.reverse_pred_relation = False
        args.eval_rst_saving_number = None
        args.eval_exp_path = str(output_dir / 'marginal_predictions')
        
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(args.log_dir, exist_ok=True)
        os.makedirs(args.temp_file_dir, exist_ok=True)
        
        # Register globally
        utils.args = args
        return args
    
    def _build_args_for_relation(self, data_dir: Path, output_dir: Path):
        """Build args for Relation V2V predictor."""
        import utils
        
        args = utils.Args()
        args.data_dir = [str(data_dir)]
        args.output_dir = str(output_dir)
        args.log_dir = str(output_dir)
        # Use shared temp_file_dir to ensure all stages process the same scenarios
        args.temp_file_dir = os.path.join(str(output_dir.parent), 'shared_temp')
        
        args.do_eval = True
        args.do_train = False
        args.do_test = True  # Use test mode to allow dummy interaction_label when no GT relations
        args.debug_mode = False
        args.debug = False
        
        args.waymo = True
        args.nuscenes = False
        args.argoverse = False
        
        # Model params (from relation.yaml)
        args.hidden_size = 128
        args.future_frame_num = 80
        args.future_test_frame_num = 80
        args.mode_num = 6
        args.sub_graph_batch_size = 4096
        args.train_batch_size = 1
        args.eval_batch_size = 1
        args.core_num = 1
        args.infMLP = 0
        args.nms_threshold = 7.2
        args.agent_type = 'vehicle'
        args.inter_agent_types = None
        args.relation_pred_threshold = 0.9
        args.single_agent = True
        
        # Model architecture params
        args.sub_graph_depth = 3
        args.global_graph_depth = 1
        args.hidden_dropout_prob = 0.1
        args.initializer_range = 0.02
        args.max_distance = 50.0
        args.no_sub_graph = False
        args.no_agents = False
        args.attention_decay = False
        args.use_map = False
        args.reuse_temp_file = False
        args.old_version = False
        args.use_centerline = False
        args.autoregression = None
        args.lstm = False
        args.add_prefix = None
        args.multi = None
        args.placeholder = 0.0
        args.method_span = [0, 1]
        args.visualize = False
        args.train_extra = False
        args.not_use_api = False
        args.no_cuda = False
        args.learning_rate = 0.001
        args.weight_decay = 0.3
        args.num_train_epochs = 30
        args.joint_target_each = 80
        args.joint_target_type = "no"
        args.classify_sub_goals = False
        args.traj_loss_coeff = 1.0
        args.short_term_loss_coeff = 0.0
        args.direct_relation_path = None
        args.all_agent_ids_path = None
        args.vehicle_r_pred_threshold = None
        args.config = None
        
        args.model_recover_path = str(Config.RELATION_MODEL)
        args.validation_model = 25
        
        # No pre-computed files
        args.relation_file_path = None
        args.relation_pred_file_path = None
        args.influencer_pred_file_path = None
        
        # Other params for Relation (from relation.yaml) - Use OtherParams for hybrid list/dict access
        # Note: 'save_rst' is critical - it tells the code to save results instead of computing loss with GT labels
        args.other_params = OtherParams([
            'train_relation',
            'pair_vv',
            'pred_with_threshold',
            'save_rst',  # CRITICAL: enables eval mode, saves predictions instead of needing GT relations
            'l1_loss',
            'densetnt',
            'goals_2D',
            'enhance_global_graph',
            'laneGCN',
            'point_sub_graph',
            'laneGCN-4',
            'stride_10_2',
            'raster',  # Relation needs raster
        ])
        
        args.eval_params = []
        args.train_params = []
        args.seed = 42
        args.distributed_training = 1
        args.reverse_pred_relation = False
        args.eval_rst_saving_number = None
        args.eval_exp_path = str(output_dir / 'relation_predictions')
        
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(args.temp_file_dir, exist_ok=True)
        
        utils.args = args
        return args
    
    def _build_args_for_conditional(self, data_dir: Path, output_dir: Path,
                                     marginal_pred_path: Path, relation_pred_path: Path):
        """Build args for Conditional V2V predictor."""
        import utils
        
        args = utils.Args()
        args.data_dir = [str(data_dir)]
        args.output_dir = str(output_dir)
        args.log_dir = str(output_dir)
        # Use shared temp_file_dir to ensure all stages process the same scenarios
        args.temp_file_dir = os.path.join(str(output_dir.parent), 'shared_temp')
        
        args.do_eval = True
        args.do_train = False
        args.do_test = False
        args.debug_mode = False
        args.debug = False
        
        args.waymo = True
        args.nuscenes = False
        args.argoverse = False
        
        # Model params (from conditional_pred.yaml)
        args.hidden_size = 128
        args.future_frame_num = 80
        args.future_test_frame_num = 80
        args.mode_num = 6
        args.sub_graph_batch_size = 4096
        args.train_batch_size = 1
        args.eval_batch_size = 1
        args.core_num = 1
        args.infMLP = 8  # Conditional uses MLP for influencer encoding
        args.nms_threshold = 7.2
        args.agent_type = 'vehicle'
        args.inter_agent_types = None
        args.single_agent = True
        args.relation_pred_threshold = 0.9
        
        # Model architecture params
        args.sub_graph_depth = 3
        args.global_graph_depth = 1
        args.hidden_dropout_prob = 0.1
        args.initializer_range = 0.02
        args.max_distance = 50.0
        args.no_sub_graph = False
        args.no_agents = False
        args.attention_decay = False
        args.use_map = False
        args.reuse_temp_file = False
        args.old_version = False
        args.use_centerline = False
        args.autoregression = None
        args.lstm = False
        args.add_prefix = None
        args.multi = None
        args.placeholder = 0.0
        args.method_span = [0, 1]
        args.visualize = False
        args.train_extra = False
        args.not_use_api = False
        args.no_cuda = False
        args.learning_rate = 0.001
        args.weight_decay = 0.3
        args.num_train_epochs = 30
        args.joint_target_each = 80
        args.joint_target_type = "no"
        args.classify_sub_goals = False
        args.traj_loss_coeff = 1.0
        args.short_term_loss_coeff = 0.0
        args.direct_relation_path = None
        args.all_agent_ids_path = None
        args.vehicle_r_pred_threshold = None
        args.config = None
        
        args.model_recover_path = str(Config.CONDITIONAL_MODEL)
        args.validation_model = 29
        
        # Use the LIVE computed files (not pre-computed)
        args.relation_file_path = None  # No GT relations
        args.relation_pred_file_path = str(relation_pred_path)  # Live relation predictions
        args.influencer_pred_file_path = str(marginal_pred_path)  # Live marginal predictions
        
        # Other params for Conditional (from conditional_pred.yaml) - Use OtherParams for hybrid list/dict access
        args.other_params = OtherParams([
            'train_reactor',
            # 'gt_influencer_traj',  # Commented out - we use predicted trajectories, not GT
            'save_rst',
            'raster_inf',
            'raster',
            'pair_vv',
            'l1_loss',
            'densetnt',
            'goals_2D',
            'enhance_global_graph',
            'laneGCN',
            'point_sub_graph',
            'laneGCN-4',
            'stride_10_2',
        ])
        
        args.eval_params = []
        args.train_params = []
        args.seed = 42
        args.distributed_training = 1
        args.reverse_pred_relation = False
        args.eval_rst_saving_number = None  # Use all 6 influencer prediction modes
        args.eval_exp_path = str(output_dir / 'conditional_predictions')
        
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(args.temp_file_dir, exist_ok=True)
        
        utils.args = args
        return args
    
    def load_model(self, model_path: Path, args) -> nn.Module:
        """Load a VectorNet model with given args."""
        from modeling.vectornet import VectorNet
        
        model = VectorNet(args)
        
        # Load weights
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        model_dict = model.state_dict()
        
        # Filter compatible weights
        pretrained_dict = {k: v for k, v in checkpoint.items() 
                           if k in model_dict and v.shape == model_dict[k].shape}
        
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        
        model = model.to(self.device)
        model.eval()
        
        loaded = len(pretrained_dict)
        total = len(model_dict)
        print(f"  Loaded {loaded}/{total} weights")
        
        return model
    
    def run_densetnt_marginal(
        self,
        data_dir: Path,
        output_dir: Path,
        max_scenarios: int,
    ) -> Tuple[Path, Dict[str, Dict]]:
        """
        Run DenseTNT to compute marginal trajectory predictions for ALL agents.
        
        Returns:
            marginal_path: Path to saved marginal predictions pickle
            predictions: Dict mapping scenario_id -> prediction data
        """
        print(f"\n{'='*60}")
        print("STEP 1: Running DenseTNT Marginal Prediction")
        print(f"{'='*60}")
        
        from dataset_waymo import Dataset
        from modeling.vectornet import VectorNet
        import utils
        
        args = self._build_args_for_densetnt(data_dir, output_dir)
        
        print(f"  Data: {data_dir}")
        print(f"  Model: {Config.DENSETNT_MODEL}")
        print(f"  Max scenarios: {max_scenarios}")
        
        # Load model
        print("\n  Loading DenseTNT model...")
        
        # Reset seeds before dataset creation for reproducibility
        self._set_seeds(self.seed)
        
        model = self.load_model(Config.DENSETNT_MODEL, args)
        
        # Create dataset
        print("  Creating dataset...")
        dataset = Dataset(args, batch_size=1, to_screen=False)
        
        # Run inference
        predictions = {}
        seen_scenarios = set()
        batch_count = 0
        # Track scenario-agent pairs to ensure we get both agents per scenario
        scenario_agent_pairs = set()
        
        print("  Running inference...")
        start_time = time.time()
        
        iterator = iter(dataset)
        # Process more batches to ensure we get both agents per scenario
        # Each scenario needs ~2 agents, so process 3x more to account for filtering
        max_batches = max_scenarios * 5
        while batch_count < max_batches:
            try:
                batch = next(iterator)
            except StopIteration:
                break
            
            if batch is None:
                # Need to generate more data
                sufficient, length = dataset.waymo_generate()
                if not sufficient and length == 0:
                    break
                continue
            
            batch_count += 1
            mappings = batch if isinstance(batch, list) else [batch]
            
            with torch.no_grad():
                try:
                    pred_traj, pred_score, _ = model(mappings, self.device)
                except Exception as e:
                    print(f"    Batch {batch_count} error: {e}")
                    continue
            
            pred_traj = np.array(pred_traj) if pred_traj is not None else None
            pred_score = np.array(pred_score) if pred_score is not None else None
            
            if pred_traj is None:
                continue
            
            for idx, mapping in enumerate(mappings):
                scenario_id = mapping['scenario_id']
                if isinstance(scenario_id, bytes):
                    scenario_id = scenario_id.decode()
                
                agent_id = int(mapping['object_id'])
                
                # Get this agent's predictions
                if idx < len(pred_traj):
                    traj_modes = pred_traj[idx]  # [6, 80, 2]
                    scores = pred_score[idx] if pred_score is not None else np.ones(6) / 6
                else:
                    continue
                
                # NOTE: Predictions are ALREADY in world coordinates!
                # The M2I decoder applies normalizer(traj, reverse=True) in decoder.py line 289-290
                # DO NOT denormalize again - that would double-transform the coordinates
                
                # Store prediction
                if scenario_id not in predictions:
                    predictions[scenario_id] = {'rst': [], 'score': [], 'ids': []}
                
                # Avoid duplicate agent predictions
                pair_key = (scenario_id, agent_id)
                if pair_key not in scenario_agent_pairs:
                    predictions[scenario_id]['rst'].append(traj_modes)
                    predictions[scenario_id]['score'].append(scores)
                    predictions[scenario_id]['ids'].append(agent_id)
                    scenario_agent_pairs.add(pair_key)
                
                seen_scenarios.add(scenario_id)
            
            if batch_count % 50 == 0:
                print(f"    Processed {batch_count} batches, {len(seen_scenarios)} scenarios, {len(scenario_agent_pairs)} agents...")
        
        elapsed = time.time() - start_time
        print(f"\n  Completed: {len(seen_scenarios)} scenarios, {len(scenario_agent_pairs)} agents in {elapsed:.1f}s")
        
        # Convert lists to arrays
        for scenario_id, data in predictions.items():
            if len(data['rst']) > 0:
                data['rst'] = np.stack(data['rst'], axis=0)
                data['score'] = np.stack(data['score'], axis=0)
                data['ids'] = np.array(data['ids'])
        
        # Save predictions
        output_dir.mkdir(parents=True, exist_ok=True)
        marginal_path = output_dir / 'marginal_predictions.pickle'
        with open(marginal_path, 'wb') as f:
            pickle.dump(predictions, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"  Saved: {marginal_path}")
        print(f"  Predictions: {len(predictions)} scenarios")
        
        return marginal_path, predictions
    
    def run_relation_prediction(
        self,
        data_dir: Path,
        output_dir: Path,
        max_scenarios: int,
    ) -> Tuple[Path, Dict]:
        """
        Run Relation V2V to predict influencer-reactor relationships.
        
        Returns:
            relation_path: Path to saved relation predictions
            relations: Dict mapping scenario_id -> relation predictions
        """
        print(f"\n{'='*60}")
        print("STEP 2: Running Relation V2V Prediction")
        print(f"{'='*60}")
        
        from dataset_waymo import Dataset
        from modeling.vectornet import VectorNet
        import globals as m2i_globals
        import utils
        
        args = self._build_args_for_relation(data_dir, output_dir)
        
        print(f"  Data: {data_dir}")
        print(f"  Model: {Config.RELATION_MODEL}")
        
        # Initialize globals for relation storage
        m2i_globals.sun_1_pred_relations = {}
        
        # Load model
        print("\n  Loading Relation V2V model...")
        
        # Reset seeds before dataset creation for reproducibility
        self._set_seeds(self.seed)
        
        model = self.load_model(Config.RELATION_MODEL, args)
        
        # Create dataset
        print("  Creating dataset...")
        dataset = Dataset(args, batch_size=1, to_screen=False)
        
        # Run inference
        seen_scenarios = set()
        batch_count = 0
        
        print("  Running inference...")
        start_time = time.time()
        
        iterator = iter(dataset)
        while len(seen_scenarios) < max_scenarios:
            try:
                batch = next(iterator)
            except StopIteration:
                break
            
            if batch is None:
                sufficient, length = dataset.waymo_generate()
                if not sufficient and length == 0:
                    break
                continue
            
            batch_count += 1
            mappings = batch if isinstance(batch, list) else [batch]
            
            with torch.no_grad():
                try:
                    # Relation model stores predictions in globals
                    model(mappings, self.device)
                except Exception as e:
                    print(f"    Batch {batch_count} error: {e}")
                    continue
            
            for mapping in mappings:
                scenario_id = mapping['scenario_id']
                if isinstance(scenario_id, bytes):
                    scenario_id = scenario_id.decode()
                seen_scenarios.add(scenario_id)
            
            if batch_count % 10 == 0:
                print(f"    Processed {batch_count} batches, {len(seen_scenarios)} scenarios...")
            
            if len(seen_scenarios) >= max_scenarios:
                break
        
        elapsed = time.time() - start_time
        
        # Get relation predictions from globals
        relations = m2i_globals.sun_1_pred_relations.copy()
        
        print(f"\n  Completed: {len(seen_scenarios)} scenarios in {elapsed:.1f}s")
        print(f"  Relations: {len(relations)} pairs")
        
        # Save predictions
        output_dir.mkdir(parents=True, exist_ok=True)
        relation_path = output_dir / 'relation_predictions.pickle'
        with open(relation_path, 'wb') as f:
            pickle.dump(relations, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"  Saved: {relation_path}")
        
        return relation_path, relations
    
    def run_conditional_prediction(
        self,
        data_dir: Path,
        output_dir: Path,
        marginal_path: Path,
        relation_path: Path,
        max_scenarios: int,
    ) -> Tuple[Path, Dict[str, Dict]]:
        """
        Run Conditional V2V to predict reactor trajectories conditioned on influencer.
        
        This uses the LIVE computed marginal and relation predictions.
        
        Returns:
            conditional_path: Path to saved conditional predictions
            predictions: Dict mapping scenario_id -> conditional prediction data
        """
        print(f"\n{'='*60}")
        print("STEP 3: Running Conditional V2V Prediction")
        print(f"{'='*60}")
        
        from dataset_waymo import Dataset
        from modeling.vectornet import VectorNet
        import utils
        
        args = self._build_args_for_conditional(data_dir, output_dir, marginal_path, relation_path)
        
        print(f"  Data: {data_dir}")
        print(f"  Model: {Config.CONDITIONAL_MODEL}")
        print(f"  Marginal predictions: {marginal_path}")
        print(f"  Relation predictions: {relation_path}")
        
        # Load model
        print("\n  Loading Conditional V2V model...")
        
        # Reset seeds before dataset creation for reproducibility
        self._set_seeds(self.seed)
        
        model = self.load_model(Config.CONDITIONAL_MODEL, args)
        
        # Create dataset
        print("  Creating dataset...")
        dataset = Dataset(args, batch_size=1, to_screen=False)
        
        # Run inference
        predictions = {}
        seen_scenarios = set()
        batch_count = 0
        
        print("  Running inference...")
        start_time = time.time()
        
        iterator = iter(dataset)
        while len(seen_scenarios) < max_scenarios:
            try:
                batch = next(iterator)
            except StopIteration:
                break
            
            if batch is None:
                sufficient, length = dataset.waymo_generate()
                if not sufficient and length == 0:
                    break
                continue
            
            batch_count += 1
            mappings = batch if isinstance(batch, list) else [batch]
            
            with torch.no_grad():
                try:
                    pred_traj, pred_score, _ = model(mappings, self.device)
                except Exception as e:
                    print(f"    Batch {batch_count} error: {e}")
                    continue
            
            pred_traj = np.array(pred_traj) if pred_traj is not None else None
            pred_score = np.array(pred_score) if pred_score is not None else None
            
            if pred_traj is None:
                continue
            
            for idx, mapping in enumerate(mappings):
                scenario_id = mapping['scenario_id']
                if isinstance(scenario_id, bytes):
                    scenario_id = scenario_id.decode()
                
                agent_id = int(mapping['object_id'])
                
                if idx < len(pred_traj):
                    traj_modes = pred_traj[idx]
                    scores = pred_score[idx] if pred_score is not None else np.ones(6) / 6
                else:
                    continue
                
                # NOTE: Predictions are ALREADY in world coordinates!
                # The M2I decoder applies normalizer(traj, reverse=True) in decoder.py line 289-290
                # DO NOT denormalize again - that would double-transform the coordinates
                
                # Store
                if scenario_id not in predictions:
                    predictions[scenario_id] = {'rst': [], 'score': [], 'ids': []}
                
                predictions[scenario_id]['rst'].append(traj_modes)
                predictions[scenario_id]['score'].append(scores)
                predictions[scenario_id]['ids'].append(agent_id)
                
                seen_scenarios.add(scenario_id)
            
            if batch_count % 10 == 0:
                print(f"    Processed {batch_count} batches, {len(seen_scenarios)} scenarios...")
            
            if len(seen_scenarios) >= max_scenarios:
                break
        
        elapsed = time.time() - start_time
        print(f"\n  Completed: {len(seen_scenarios)} scenarios in {elapsed:.1f}s")
        
        # Convert lists to arrays
        for scenario_id, data in predictions.items():
            if len(data['rst']) > 0:
                data['rst'] = np.stack(data['rst'], axis=0)
                data['score'] = np.stack(data['score'], axis=0)
                data['ids'] = np.array(data['ids'])
        
        # Save
        output_dir.mkdir(parents=True, exist_ok=True)
        conditional_path = output_dir / 'conditional_predictions.pickle'
        with open(conditional_path, 'wb') as f:
            pickle.dump(predictions, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"  Saved: {conditional_path}")
        print(f"  Predictions: {len(predictions)} scenarios")
        
        return conditional_path, predictions
    
    def run_full_pipeline(
        self,
        data_dir: Path,
        output_dir: Path,
        max_scenarios: int,
    ) -> Dict[str, Any]:
        """
        Run the complete M2I pipeline:
        1. DenseTNT marginal prediction
        2. Relation V2V prediction
        3. Conditional V2V prediction
        
        All predictions are computed LIVE, no pre-computed files used.
        """
        print(f"\n{'='*70}")
        print("M2I FULL LIVE PIPELINE")
        print(f"{'='*70}")
        print(f"\nData directory: {data_dir}")
        print(f"Output directory: {output_dir}")
        print(f"Max scenarios: {max_scenarios}")
        print("\nNO PRE-COMPUTED FILES - All predictions computed live!")
        
        pipeline_start = time.time()
        
        # Step 1: Marginal predictions
        marginal_path, marginal_preds = self.run_densetnt_marginal(
            data_dir, output_dir / 'marginal', max_scenarios
        )
        
        # Step 2: Relation predictions
        relation_path, relations = self.run_relation_prediction(
            data_dir, output_dir / 'relation', max_scenarios
        )
        
        # Step 3: Conditional predictions (uses outputs from steps 1 and 2)
        conditional_path, conditional_preds = self.run_conditional_prediction(
            data_dir, output_dir / 'conditional',
            marginal_path, relation_path, max_scenarios
        )
        
        pipeline_elapsed = time.time() - pipeline_start
        
        # Summary
        print(f"\n{'='*70}")
        print("PIPELINE COMPLETE")
        print(f"{'='*70}")
        print(f"\nTotal time: {pipeline_elapsed:.1f}s")
        print(f"\nOutput files:")
        print(f"  Marginal:    {marginal_path}")
        print(f"  Relations:   {relation_path}")
        print(f"  Conditional: {conditional_path}")
        print(f"\nStatistics:")
        print(f"  Marginal predictions:    {len(marginal_preds)} scenarios")
        print(f"  Relation predictions:    {len(relations)} pairs")
        print(f"  Conditional predictions: {len(conditional_preds)} scenarios")
        
        if torch.cuda.is_available():
            peak_mem = torch.cuda.max_memory_allocated(0) / 1e9
            print(f"\nPeak GPU memory: {peak_mem:.2f} GB")
        
        return {
            'marginal': marginal_preds,
            'marginal_path': marginal_path,
            'relations': relations,
            'relation_path': relation_path,
            'conditional': conditional_preds,
            'conditional_path': conditional_path,
            'elapsed_time': pipeline_elapsed,
        }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='M2I Live Inference Pipeline - NO PRE-COMPUTED FILES',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument('--num_scenarios', type=int, default=10,
                        help='Number of scenarios to process')
    parser.add_argument('--data_dir', type=str, 
                        default=str(Config.DATA_DIR),
                        help='Directory containing TFRecord files')
    parser.add_argument('--output_dir', type=str,
                        default=str(Config.OUTPUT_DIR),
                        help='Output directory for predictions')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    parser.add_argument('--stage', type=str, default='all',
                        choices=['marginal', 'relation', 'conditional', 'all'],
                        help='Which stage to run (default: all)')
    
    args = parser.parse_args()
    
    # Update config
    Config.DATA_DIR = Path(args.data_dir)
    Config.OUTPUT_DIR = Path(args.output_dir)
    
    # Create runner
    runner = M2ILiveRunner(device=args.device)
    
    if args.stage == 'all':
        # Run full pipeline
        results = runner.run_full_pipeline(
            data_dir=Config.DATA_DIR,
            output_dir=Config.OUTPUT_DIR,
            max_scenarios=args.num_scenarios,
        )
    elif args.stage == 'marginal':
        # Run only marginal prediction
        marginal_path, predictions = runner.run_densetnt_marginal(
            data_dir=Config.DATA_DIR,
            output_dir=Config.OUTPUT_DIR / 'marginal',
            max_scenarios=args.num_scenarios,
        )
    elif args.stage == 'relation':
        # Run only relation prediction
        relation_path, relations = runner.run_relation_prediction(
            data_dir=Config.DATA_DIR,
            output_dir=Config.OUTPUT_DIR / 'relation',
            max_scenarios=args.num_scenarios,
        )
    elif args.stage == 'conditional':
        # Need marginal and relation paths for conditional
        marginal_path = Config.OUTPUT_DIR / 'marginal' / 'marginal_predictions.pickle'
        relation_path = Config.OUTPUT_DIR / 'relation' / 'relation_predictions.pickle'
        
        if not marginal_path.exists() or not relation_path.exists():
            print("ERROR: Conditional stage requires marginal and relation predictions.")
            print("Run with --stage all first, or run marginal and relation stages separately.")
            return
        
        conditional_path, predictions = runner.run_conditional_prediction(
            data_dir=Config.DATA_DIR,
            output_dir=Config.OUTPUT_DIR / 'conditional',
            marginal_path=marginal_path,
            relation_path=relation_path,
            max_scenarios=args.num_scenarios,
        )
    
    print("\nDone!")


if __name__ == '__main__':
    main()
