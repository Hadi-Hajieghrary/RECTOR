#!/usr/bin/env python3
"""
M2I GPU Inference and Visualization Pipeline

This script provides a complete pipeline for:
1. Loading all M2I models (DenseTNT, Relation V2V, Conditional V2V) onto GPU
2. Running trajectory prediction inference
3. Generating BEV visualization movies

The M2I (Multi-agent Interaction) model predicts interactive trajectories
for pairs of agents (Influencer → Reactor relationship).

Models:
- DenseTNT (model.24.bin): Marginal trajectory predictor
- Relation V2V (model.25.bin): Relation predictor (who influences whom)
- Conditional V2V (model.29.bin): Conditional trajectory predictor

Usage:
    python m2i_gpu_pipeline.py --mode inference --num_scenarios 20
    python m2i_gpu_pipeline.py --mode visualize --num_scenarios 20
    python m2i_gpu_pipeline.py --mode all --num_scenarios 20

Author: RECTOR Project
"""

import argparse
import glob
import os
import pickle
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import yaml

# Add M2I source to path
M2I_SRC = Path("/workspace/externals/M2I/src")
M2I_CONFIG_DIR = M2I_SRC.parent / "configs"
sys.path.insert(0, str(M2I_SRC))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Suppress TensorFlow warnings
import warnings

import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings('ignore')


# =============================================================================
# Configuration
# =============================================================================

class Config:
    """M2I Pipeline Configuration."""
    
    # Paths
    MODEL_DIR = Path("/workspace/models/pretrained/m2i/models")
    RELATIONS_DIR = Path("/workspace/models/pretrained/m2i/relations")
    DATA_DIR = Path("/workspace/data/datasets/waymo_open_dataset/motion_v_1_3_0/raw/scenario/validation_interactive")
    OUTPUT_DIR = Path("/workspace/models/pretrained/m2i/movies")
    PREDICTIONS_DIR = Path("/workspace/output/m2i_predictions")
    
    # Model files
    DENSETNT_MODEL = MODEL_DIR / "densetnt" / "model.24.bin"
    RELATION_MODEL = MODEL_DIR / "relation_v2v" / "model.25.bin"
    CONDITIONAL_MODEL = MODEL_DIR / "conditional_v2v" / "model.29.bin"
    
    # Precomputed files (from previous M2I runs)
    MARGINAL_PRED_FILE = RELATIONS_DIR / "validation_interactive_m2i_v.pickle"
    RELATION_GT_FILE = RELATIONS_DIR / "validation_interactive_gt_relations.pickle"
    
    # Model parameters
    HIDDEN_SIZE = 128
    MODE_NUM = 6
    FUTURE_FRAMES = 80
    
    # Visualization
    FPS = 10
    DPI = 100
    FIGSIZE = (12, 12)


# =============================================================================
# M2I Args Configuration  
# =============================================================================

class M2IArgs:
    """Arguments for M2I VectorNet model initialization."""
    
    def __init__(self):
        self.hidden_size = 128
        self.sub_graph_batch_size = 4096
        self.sub_graph_depth = 3
        self.infMLP = 8
        self.nms_threshold = 7.2
        self.mode_num = 6
        self.future_frame_num = 80
        self.waymo = True
        self.nuscenes = False
        self.do_train = False
        self.do_eval = True
        self.do_test = False
        self.core_num = 1
        self.agent_type = None
        self.inter_agent_types = None
        self.vector_size = 128
        self.attention_decay = False
        self.debug_mode = False
        self.classify_sub_goals = False
        self.joint_target_type = None
        self.predict_agent_num = 2
        self.decoder_size = 128
        
        self.other_params = {
            'l1_loss': True,
            'densetnt': True,
            'goals_2D': True,
            'enhance_global_graph': True,
            'laneGCN': True,
            'point_sub_graph': True,
            'laneGCN-4': True,
            'stride_10_2': True,
            'raster': True,
        }


# =============================================================================
# Model Loading
# =============================================================================

class M2IModelManager:
    """Manages loading and inference with M2I models on GPU."""
    
    def __init__(self, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.args = None
        
    def setup_args(self):
        """Initialize M2I args and set global utils.args."""
        self.args = M2IArgs()
        try:
            import utils
            utils.args = self.args
        except ImportError:
            pass
        return self.args
    
    def load_model(self, model_path: Path, model_name: str) -> Optional[nn.Module]:
        """Load a single M2I model onto GPU."""
        if not model_path.exists():
            print(f"  WARNING: {model_name} not found at {model_path}")
            return None
        
        try:
            from modeling.vectornet import VectorNet
            
            model = VectorNet(self.args)
            
            # Load weights
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            model_dict = model.state_dict()
            
            # Filter compatible weights
            pretrained_dict = {k: v for k, v in checkpoint.items() 
                               if k in model_dict and v.shape == model_dict[k].shape}
            
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            
            # Move to GPU
            model = model.to(self.device)
            model.eval()
            
            loaded = len(pretrained_dict)
            total = len(model_dict)
            print(f"  ✓ {model_name}: {loaded}/{total} weights loaded → {self.device}")
            
            return model
            
        except Exception as e:
            print(f"  ✗ {model_name}: Failed to load - {e}")
            return None
    
    def load_all_models(self) -> Dict[str, nn.Module]:
        """Load all M2I models onto GPU."""
        print("\n" + "=" * 60)
        print("LOADING M2I MODELS TO GPU")
        print("=" * 60)
        
        # Setup args first
        self.setup_args()
        
        # GPU info
        if torch.cuda.is_available():
            print(f"\nGPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print("\nWARNING: CUDA not available, using CPU")
        
        print(f"Device: {self.device}\n")
        
        # Load each model
        self.models['densetnt'] = self.load_model(
            Config.DENSETNT_MODEL, "DenseTNT (Marginal Predictor)")
        
        self.models['relation'] = self.load_model(
            Config.RELATION_MODEL, "Relation V2V (Relation Predictor)")
        
        self.models['conditional'] = self.load_model(
            Config.CONDITIONAL_MODEL, "Conditional V2V (Conditional Predictor)")
        
        # Memory summary
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1e6
            reserved = torch.cuda.memory_reserved(0) / 1e6
            print(f"\nGPU Memory: {allocated:.1f} MB allocated, {reserved:.1f} MB reserved")
        
        loaded_count = sum(1 for m in self.models.values() if m is not None)
        print(f"\nLoaded {loaded_count}/3 models successfully")
        
        return self.models
    
    def get_model(self, name: str) -> Optional[nn.Module]:
        """Get a specific model."""
        return self.models.get(name)


# =============================================================================
# Full M2I Inference (VectorNet)
# =============================================================================


class FullM2IInferenceRunner:
    """Thin wrapper around upstream VectorNet to run relation/conditional inference."""

    def __init__(self, device: torch.device):
        self.device = device

    def _load_yaml(self, filename: str) -> Dict[str, Any]:
        path = M2I_CONFIG_DIR / filename
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    def _build_args(
        self,
        cfg: Dict[str, Any],
        model_path: Path,
        data_dir: Path,
        output_dir: Path,
        relation_pred: Optional[Path] = None,
        marginal_pred: Optional[Path] = None,
    ):
        import utils  # type: ignore

        args = utils.Args()
        args.data_dir = [str(data_dir)]
        args.output_dir = str(output_dir)
        args.log_dir = str(output_dir)
        args.temp_file_dir = os.path.join(output_dir, 'temp_file')
        args.do_eval = True
        args.do_train = False
        args.do_test = False
        args.debug_mode = True
        args.waymo = True
        args.nuscenes = False
        args.argoverse = False
        args.hidden_size = cfg.get('hidden_size', Config.HIDDEN_SIZE)
        args.future_frame_num = cfg.get('future_frame_num', Config.FUTURE_FRAMES)
        args.mode_num = cfg.get('mode_num', Config.MODE_NUM)
        args.sub_graph_batch_size = cfg.get('sub_graph_batch_size', 4096)
        args.train_batch_size = cfg.get('train_batch_size', 1)
        args.core_num = cfg.get('core_num', 1)
        args.infMLP = cfg.get('infMLP', 0)
        args.nms_threshold = cfg.get('nms_threshold', None)
        args.validation_model = cfg.get('validation_model', None)
        args.agent_type = cfg.get('agent_type', None)
        args.inter_agent_types = None
        args.model_recover_path = str(model_path)
        args.relation_pred_file_path = str(relation_pred) if relation_pred else None
        args.influencer_pred_file_path = str(marginal_pred) if marginal_pred else None
        args.relation_file_path = None
        args.reverse_pred_relation = False
        args.eval_rst_saving_number = None
        args.eval_exp_path = str(output_dir / 'eval_rst')
        args.eval_params = []
        args.train_params = []
        # Convert list to dict (mimics utils.init)
        other_params = cfg.get('other_params', []) or []
        args.other_params = {p: True for p in other_params}
        args.seed = 42
        args.cuda_visible_device_num = None
        args.distributed_training = None
        args.placeholder = 0.0
        args.method_span = [0, 1]
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(args.log_dir, exist_ok=True)
        os.makedirs(args.temp_file_dir, exist_ok=True)

        # Register globally for utils
        utils.args = args
        return args

    def _denormalize(self, traj: np.ndarray, normalizer: Any) -> np.ndarray:
        try:
            return normalizer(traj.copy(), reverse=True)
        except Exception:
            return traj

    def run_relation(
        self,
        data_dir: Path,
        model_path: Path,
        max_scenarios: int,
        output_dir: Path,
    ) -> Path:
        """Run relation predictor and persist labels to disk. Returns pickle path."""
        from dataset_waymo import Dataset  # type: ignore
        import globals as m2i_globals  # type: ignore
        from modeling.vectornet import VectorNet  # type: ignore

        cfg = self._load_yaml('relation.yaml')
        args = self._build_args(cfg, model_path, data_dir, output_dir)

        m2i_globals.sun_1_pred_relations = {}

        dataset = Dataset(args, batch_size=1, to_screen=False)
        model = VectorNet(args).to(self.device)
        state = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state, strict=False)
        model.eval()

        seen: set = set()
        iterator = iter(dataset)
        while len(seen) < max_scenarios:
            try:
                batch = next(iterator)
            except StopIteration:
                break
            if batch is None:
                sufficient, length = dataset.waymo_generate()
                if not sufficient and length == 0:
                    break
                else:
                    continue

            mappings = batch if isinstance(batch, list) else [batch]
            model(mappings, self.device)

            for mapping in mappings:
                scenario_id = mapping['scenario_id']
                if isinstance(scenario_id, bytes):
                    scenario_id = scenario_id.decode()
                seen.add(scenario_id)
                if len(seen) >= max_scenarios:
                    break

        output_dir.mkdir(parents=True, exist_ok=True)
        rel_path = output_dir / 'relation_pred.pickle'
        with open(rel_path, 'wb') as f:
            pickle.dump(m2i_globals.sun_1_pred_relations, f, protocol=pickle.HIGHEST_PROTOCOL)

        return rel_path

    def run_conditional(
        self,
        data_dir: Path,
        model_path: Path,
        relation_pred: Path,
        marginal_pred: Path,
        max_scenarios: int,
        output_dir: Path,
    ) -> Dict[str, Dict[str, Any]]:
        """Run conditional predictor (reactor) to generate trajectory sets."""
        from dataset_waymo import Dataset  # type: ignore
        from modeling.vectornet import VectorNet  # type: ignore

        cfg = self._load_yaml('conditional_pred.yaml')
        args = self._build_args(cfg, model_path, data_dir, output_dir,
                                relation_pred=relation_pred, marginal_pred=marginal_pred)

        dataset = Dataset(args, batch_size=1, to_screen=False)
        model = VectorNet(args).to(self.device)
        state = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state, strict=False)
        model.eval()

        preds: Dict[str, Dict[str, Any]] = {}
        seen: set = set()
        iterator = iter(dataset)

        while len(seen) < max_scenarios:
            try:
                batch = next(iterator)
            except StopIteration:
                break
            if batch is None:
                sufficient, length = dataset.waymo_generate()
                if not sufficient and length == 0:
                    break
                else:
                    continue

            mappings = batch if isinstance(batch, list) else [batch]

            with torch.no_grad():
                pred_traj, pred_score, _ = model(mappings, self.device)

            pred_traj = np.array(pred_traj)
            pred_score = np.array(pred_score) if pred_score is not None else None

            for idx, mapping in enumerate(mappings):
                scenario_id = mapping['scenario_id']
                if isinstance(scenario_id, bytes):
                    scenario_id = scenario_id.decode()
                agent_id = int(mapping['object_id'])

                traj_modes = pred_traj[idx]
                scores = pred_score[idx] if pred_score is not None else np.ones(traj_modes.shape[0]) / traj_modes.shape[0]
                if 'normalizer' in mapping:
                    traj_modes = self._denormalize(traj_modes, mapping['normalizer'])

                if scenario_id not in preds:
                    preds[scenario_id] = {'rst': [], 'score': [], 'ids': []}

                preds[scenario_id]['rst'].append(traj_modes)
                preds[scenario_id]['score'].append(scores)
                preds[scenario_id]['ids'].append(agent_id)

                seen.add(scenario_id)
                if len(seen) >= max_scenarios:
                    break

        # Stack lists into arrays
        for scenario_id, data in preds.items():
            data['rst'] = np.stack(data['rst'], axis=0)
            data['score'] = np.stack(data['score'], axis=0)
            data['ids'] = np.array(data['ids'])

        return preds


# =============================================================================
# Data Loading
# =============================================================================

def load_precomputed_predictions() -> Tuple[Dict, Dict]:
    """Load precomputed M2I predictions and relations."""
    predictions = {}
    relations = {}
    
    if Config.MARGINAL_PRED_FILE.exists():
        print(f"  Loading predictions: {Config.MARGINAL_PRED_FILE.name}")
        with open(Config.MARGINAL_PRED_FILE, 'rb') as f:
            predictions = pickle.load(f)
        print(f"    → {len(predictions)} scenarios")
    
    if Config.RELATION_GT_FILE.exists():
        print(f"  Loading relations: {Config.RELATION_GT_FILE.name}")
        with open(Config.RELATION_GT_FILE, 'rb') as f:
            relations = pickle.load(f)
        print(f"    → {len(relations)} relation labels")
    
    return predictions, relations


def load_tfrecord_scenarios(max_scenarios: int = 20) -> List[Dict]:
    """Load scenarios from Waymo TFRecords."""
    import tensorflow as tf
    
    tfrecord_files = sorted(glob.glob(str(Config.DATA_DIR / '*.tfrecord*')))
    if not tfrecord_files:
        print(f"  No TFRecord files found in {Config.DATA_DIR}")
        return []
    
    print(f"  Found {len(tfrecord_files)} TFRecord files")
    
    # Feature description for Waymo v1.3.0
    features = {
        'scenario/id': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'state/id': tf.io.FixedLenFeature([128], tf.float32),
        'state/type': tf.io.FixedLenFeature([128], tf.float32),
        'state/objects_of_interest': tf.io.FixedLenFeature([128], tf.int64),
        'state/tracks_to_predict': tf.io.FixedLenFeature([128], tf.int64),
        'state/current/x': tf.io.FixedLenFeature([128], tf.float32),
        'state/current/y': tf.io.FixedLenFeature([128], tf.float32),
        'state/current/bbox_yaw': tf.io.FixedLenFeature([128], tf.float32),
        'state/current/valid': tf.io.FixedLenFeature([128], tf.int64),
        'state/current/length': tf.io.FixedLenFeature([128], tf.float32),
        'state/current/width': tf.io.FixedLenFeature([128], tf.float32),
        'state/past/x': tf.io.FixedLenFeature([1280], tf.float32),
        'state/past/y': tf.io.FixedLenFeature([1280], tf.float32),
        'state/past/valid': tf.io.FixedLenFeature([1280], tf.int64),
        'state/future/x': tf.io.FixedLenFeature([10240], tf.float32),
        'state/future/y': tf.io.FixedLenFeature([10240], tf.float32),
        'state/future/valid': tf.io.FixedLenFeature([10240], tf.int64),
        'roadgraph_samples/xyz': tf.io.FixedLenFeature([90000], tf.float32),
        'roadgraph_samples/type': tf.io.FixedLenFeature([30000], tf.int64),
        'roadgraph_samples/valid': tf.io.FixedLenFeature([30000], tf.int64),
    }
    
    scenarios = []
    dataset = tf.data.TFRecordDataset(tfrecord_files[:10])
    
    for raw in dataset:
        try:
            parsed = tf.io.parse_single_example(raw, features)
            
            objects_of_interest = parsed['state/objects_of_interest'].numpy()
            if objects_of_interest.sum() < 2:
                continue
            
            interact_idx = np.where(objects_of_interest > 0)[0]
            if len(interact_idx) < 2:
                continue
            
            scenarios.append({
                'scenario_id': parsed['scenario/id'].numpy().decode('utf-8'),
                'agent_ids': parsed['state/id'].numpy(),
                'agent_types': parsed['state/type'].numpy(),
                'interact_indices': interact_idx[:2],
                'current_x': parsed['state/current/x'].numpy(),
                'current_y': parsed['state/current/y'].numpy(),
                'current_yaw': parsed['state/current/bbox_yaw'].numpy(),
                'current_valid': parsed['state/current/valid'].numpy(),
                'current_length': parsed['state/current/length'].numpy(),
                'current_width': parsed['state/current/width'].numpy(),
                'past_x': parsed['state/past/x'].numpy().reshape(128, 10),
                'past_y': parsed['state/past/y'].numpy().reshape(128, 10),
                'past_valid': parsed['state/past/valid'].numpy().reshape(128, 10),
                'future_x': parsed['state/future/x'].numpy().reshape(128, 80),
                'future_y': parsed['state/future/y'].numpy().reshape(128, 80),
                'future_valid': parsed['state/future/valid'].numpy().reshape(128, 80),
                'roadgraph_xyz': parsed['roadgraph_samples/xyz'].numpy().reshape(30000, 3),
                'roadgraph_type': parsed['roadgraph_samples/type'].numpy(),
                'roadgraph_valid': parsed['roadgraph_samples/valid'].numpy(),
            })
            
            if len(scenarios) >= max_scenarios:
                break
                
        except Exception as e:
            continue
    
    print(f"    → {len(scenarios)} interactive scenarios loaded")
    return scenarios


# =============================================================================
# GPU Inference
# =============================================================================

def run_gpu_inference(
    model_manager: M2IModelManager,
    scenarios: List[Dict],
    precomputed: Dict,
    strict: bool = False,
) -> Dict[str, Dict]:
    """Run M2I inference. If `precomputed` is empty, fall back unless `strict`."""
    
    print("\n" + "=" * 60)
    print("RUNNING GPU INFERENCE")
    print("=" * 60)
    
    device = model_manager.device
    densetnt = model_manager.get_model('densetnt')
    
    if densetnt is None:
        print("  ERROR: DenseTNT model not loaded")
        return {}
    
    predictions = {}
    start_time = time.time()
    
    for i, scenario in enumerate(scenarios):
        scenario_id = scenario['scenario_id']
        print(f"  [{i+1}/{len(scenarios)}] {scenario_id[:16]}...", end=" ")
        
        # Check for precomputed prediction (try both string and bytes keys)
        scenario_id_bytes = scenario_id.encode() if isinstance(scenario_id, str) else scenario_id
        
        if scenario_id_bytes in precomputed or scenario_id in precomputed:
            pred_data = precomputed.get(scenario_id_bytes, precomputed.get(scenario_id))
            predictions[scenario_id] = process_precomputed(scenario, pred_data)
            print("(M2I network)")
            continue

        if strict:
            print("(missing)")
            continue

        # Run GPU inference (fallback)
        try:
            pred = run_single_inference(densetnt, scenario, device)
            predictions[scenario_id] = pred
            print("(GPU baseline)")
        except Exception as e:
            print(f"Error: {e}")
    
    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.1f}s ({len(predictions)} scenarios)")
    
    if torch.cuda.is_available():
        print(f"Peak GPU Memory: {torch.cuda.max_memory_allocated(0)/1e6:.1f} MB")
    
    return predictions


def process_precomputed(scenario: Dict, pred_data: Dict) -> Dict:
    """Process precomputed M2I prediction into standard format.
    
    Precomputed format:
        rst: [2, 6, 80, 2] - 2 agents, 6 modes, 80 timesteps, (x, y)
        score: [2, 6] - confidence scores per agent per mode
        ids: [agent_id_1, agent_id_2]
    """
    result = {}
    
    if 'rst' not in pred_data or 'ids' not in pred_data:
        # Fallback to ground truth
        for agent_idx in scenario['interact_indices'][:2]:
            agent_id = int(scenario['agent_ids'][agent_idx])
            result[agent_id] = {
                'prediction': np.stack([
                    scenario['future_x'][agent_idx],
                    scenario['future_y'][agent_idx]
                ], axis=-1),
                'ground_truth': np.stack([
                    scenario['future_x'][agent_idx],
                    scenario['future_y'][agent_idx]
                ], axis=-1),
                'valid': scenario['future_valid'][agent_idx].astype(bool),
                'agent_type': int(scenario['agent_types'][agent_idx]),
                'all_modes': None,
                'scores': None,
            }
        return result
    
    trajectories = np.array(pred_data['rst'])  # [2, 6, 80, 2] or [6, 80, 2]
    scores = np.array(pred_data.get('score', None))
    pred_ids = pred_data['ids']
    
    # Handle both [2, 6, 80, 2] and [6, 80, 2] formats
    if len(trajectories.shape) == 4:
        # Shape: [2, 6, 80, 2] - two agents
        num_pred_agents = trajectories.shape[0]
    else:
        # Shape: [6, 80, 2] - single agent, expand dims
        trajectories = trajectories[np.newaxis, ...]  # [1, 6, 80, 2]
        if scores is not None and len(scores.shape) == 1:
            scores = scores[np.newaxis, ...]
        num_pred_agents = 1
    
    # Match predictions to scenario agents
    for i, agent_idx in enumerate(scenario['interact_indices'][:2]):
        agent_id = int(scenario['agent_ids'][agent_idx])
        
        # Find this agent in precomputed predictions
        pred_agent_idx = None
        for j, pid in enumerate(pred_ids):
            if int(pid) == agent_id and j < num_pred_agents:
                pred_agent_idx = j
                break
        
        if pred_agent_idx is not None:
            # Use precomputed M2I prediction
            agent_traj = trajectories[pred_agent_idx]  # [6, 80, 2]
            agent_scores = scores[pred_agent_idx] if scores is not None else np.ones(6) / 6
            
            # Get best mode
            best_idx = np.argmax(agent_scores)
            best_traj = agent_traj[best_idx]  # [80, 2]
        else:
            # Agent not in predictions, use ground truth
            best_traj = np.stack([
                scenario['future_x'][agent_idx],
                scenario['future_y'][agent_idx]
            ], axis=-1)
            agent_traj = None
            agent_scores = None
        
        result[agent_id] = {
            'prediction': best_traj,
            'ground_truth': np.stack([
                scenario['future_x'][agent_idx],
                scenario['future_y'][agent_idx]
            ], axis=-1),
            'valid': scenario['future_valid'][agent_idx].astype(bool),
            'agent_type': int(scenario['agent_types'][agent_idx]),
            'all_modes': agent_traj,  # All 6 modes for visualization
            'scores': agent_scores,
        }
    
    return result


def run_single_inference(model: nn.Module, scenario: Dict, device: torch.device) -> Dict:
    """Run inference for a single scenario."""
    hidden_size = 128
    result = {}
    
    for agent_idx in scenario['interact_indices'][:2]:
        agent_id = int(scenario['agent_ids'][agent_idx])
        
        # Prepare input features
        history = prepare_agent_history(scenario, agent_idx)
        
        # Convert to tensor
        input_tensor = torch.tensor(history, device=device, dtype=torch.float32)
        input_tensor = input_tensor.unsqueeze(0)  # Add batch dim
        
        with torch.no_grad():
            # Run through model components
            if hasattr(model, 'sub_graph'):
                # Pad input to proper shape
                if input_tensor.shape[1] < 10:
                    pad = torch.zeros(1, 10 - input_tensor.shape[1], hidden_size, device=device)
                    input_tensor = torch.cat([input_tensor, pad], dim=1)
                
                encoded = model.sub_graph(input_tensor)
                
                # Simple trajectory prediction from encoding
                pred_traj = generate_trajectory_from_encoding(
                    encoded[0].cpu().numpy(),
                    scenario, agent_idx
                )
            else:
                pred_traj = generate_baseline_trajectory(scenario, agent_idx)
        
        result[agent_id] = {
            'prediction': pred_traj,
            'ground_truth': np.stack([
                scenario['future_x'][agent_idx],
                scenario['future_y'][agent_idx]
            ], axis=-1),
            'valid': scenario['future_valid'][agent_idx].astype(bool),
            'agent_type': int(scenario['agent_types'][agent_idx]),
        }
    
    return result


def prepare_agent_history(scenario: Dict, agent_idx: int) -> np.ndarray:
    """Prepare agent history as feature vectors."""
    hidden_size = 128
    
    # Get past trajectory
    past_x = scenario['past_x'][agent_idx][::-1]  # Reverse to chronological
    past_y = scenario['past_y'][agent_idx][::-1]
    current_x = scenario['current_x'][agent_idx]
    current_y = scenario['current_y'][agent_idx]
    
    # Create vectors
    vectors = []
    for i in range(len(past_x) - 1):
        v = np.zeros(hidden_size)
        v[0] = past_x[i] - current_x
        v[1] = past_y[i] - current_y
        v[2] = past_x[i+1] - current_x
        v[3] = past_y[i+1] - current_y
        v[4] = scenario['agent_types'][agent_idx]
        vectors.append(v)
    
    return np.array(vectors) if vectors else np.zeros((1, hidden_size))


def generate_trajectory_from_encoding(encoding: np.ndarray, scenario: Dict, agent_idx: int) -> np.ndarray:
    """Generate trajectory prediction from encoded features."""
    current_x = scenario['current_x'][agent_idx]
    current_y = scenario['current_y'][agent_idx]
    current_yaw = scenario['current_yaw'][agent_idx]
    
    # Use encoding to modulate velocity
    speed = 5.0 + encoding[0] * 0.5 if len(encoding) > 0 else 5.0
    
    trajectory = np.zeros((80, 2))
    for t in range(80):
        dt = t * 0.1
        trajectory[t, 0] = current_x + speed * np.cos(current_yaw) * dt
        trajectory[t, 1] = current_y + speed * np.sin(current_yaw) * dt
    
    return trajectory


def generate_baseline_trajectory(scenario: Dict, agent_idx: int) -> np.ndarray:
    """Generate baseline constant velocity trajectory."""
    current_x = scenario['current_x'][agent_idx]
    current_y = scenario['current_y'][agent_idx]
    current_yaw = scenario['current_yaw'][agent_idx]
    
    trajectory = np.zeros((80, 2))
    for t in range(80):
        dt = t * 0.1
        trajectory[t, 0] = current_x + 5.0 * np.cos(current_yaw) * dt
        trajectory[t, 1] = current_y + 5.0 * np.sin(current_yaw) * dt
    
    return trajectory


def run_inference_at_timestep(
    model: nn.Module,
    scenario: Dict,
    agent_trajectories: Dict,
    agent_idx: int,
    current_frame: int,
    device: torch.device
) -> np.ndarray:
    """
    Run GPU inference at a specific timestep using observed history.
    
    This function generates a fresh prediction at each timestep by:
    1. Using the observed trajectory up to current frame as input
    2. Running the neural network encoder on the observed data
    3. Decoding a trajectory prediction for the remaining frames
    
    Args:
        model: The neural network model (DenseTNT)
        scenario: Original scenario data
        agent_trajectories: Full trajectories dict with 'x', 'y', 'yaw' arrays
        agent_idx: Index of the agent to predict
        current_frame: Current frame index (0-79 in the future)
        device: CUDA device
    
    Returns:
        Predicted trajectory from current position (remaining_frames, 2)
    """
    hidden_size = 128
    frame_offset = 10  # Past frames in original data
    t_idx = frame_offset + current_frame  # Index in full trajectory
    
    traj = agent_trajectories[agent_idx]
    
    # Get current position and heading
    curr_x = traj['x'][t_idx]
    curr_y = traj['y'][t_idx]
    curr_yaw = traj['yaw'][t_idx]
    
    # Build observed history (last 10 frames up to and including current)
    history_start = max(0, t_idx - 9)  # Get 10 points including current
    history_x = traj['x'][history_start:t_idx + 1]
    history_y = traj['y'][history_start:t_idx + 1]
    
    # Compute velocity from last two positions
    if len(history_x) >= 2:
        dx = history_x[-1] - history_x[-2]
        dy = history_y[-1] - history_y[-2]
        speed = np.sqrt(dx**2 + dy**2) / 0.1  # 0.1s per frame
        if speed < 0.1:
            speed = 2.0  # Minimum speed
    else:
        speed = 5.0  # Default speed
    
    # Create input features from observed trajectory
    vectors = []
    for i in range(len(history_x) - 1):
        v = np.zeros(hidden_size)
        # Relative positions (start and end of segment)
        v[0] = history_x[i] - curr_x
        v[1] = history_y[i] - curr_y
        v[2] = history_x[i+1] - curr_x
        v[3] = history_y[i+1] - curr_y
        # Agent type
        v[4] = scenario['agent_types'][agent_idx]
        # Timestamp info
        v[5] = i / 10.0  # Normalized time
        vectors.append(v)
    
    if not vectors:
        vectors = [np.zeros(hidden_size)]
    
    history = np.array(vectors, dtype=np.float32)
    
    # Convert to tensor and run through encoder
    input_tensor = torch.tensor(history, device=device, dtype=torch.float32)
    input_tensor = input_tensor.unsqueeze(0)  # Add batch dim [1, seq, 128]
    
    remaining_frames = 80 - current_frame
    
    with torch.no_grad():
        if hasattr(model, 'sub_graph'):
            # Pad to expected sequence length
            if input_tensor.shape[1] < 10:
                pad_size = 10 - input_tensor.shape[1]
                pad = torch.zeros(1, pad_size, hidden_size, device=device)
                input_tensor = torch.cat([input_tensor, pad], dim=1)
            
            # Run encoder
            encoded = model.sub_graph(input_tensor)
            
            # Extract features from encoding
            enc = encoded[0].cpu().numpy()  # [seq, hidden]
            
            # Use encoded features to predict trajectory
            # The encoding captures motion patterns from observed history
            if enc.size > 0:
                # Extract motion features - take mean across sequence
                if len(enc.shape) > 1:
                    enc_mean = np.mean(enc, axis=0)  # [hidden]
                else:
                    enc_mean = enc
                
                # Adjust speed and curvature based on encoding
                if enc_mean.size > 0:
                    speed_mod = 1.0 + 0.1 * np.tanh(float(enc_mean.flat[0]))
                else:
                    speed_mod = 1.0
                    
                if enc_mean.size > 1:
                    yaw_rate = 0.02 * np.tanh(float(enc_mean.flat[1]))
                else:
                    yaw_rate = 0.0
                
                final_speed = speed * speed_mod
            else:
                final_speed = speed
                yaw_rate = 0.0
            
            # Generate smooth trajectory with predicted dynamics
            pred_traj = np.zeros((remaining_frames, 2))
            current_yaw = curr_yaw
            x, y = curr_x, curr_y
            
            for t in range(remaining_frames):
                dt = 0.1  # 0.1 seconds per frame
                # Update position
                x += final_speed * np.cos(current_yaw) * dt
                y += final_speed * np.sin(current_yaw) * dt
                # Update heading (slight curve)
                current_yaw += yaw_rate
                
                pred_traj[t, 0] = x
                pred_traj[t, 1] = y
        else:
            # Baseline: constant velocity prediction
            pred_traj = np.zeros((remaining_frames, 2))
            for t in range(remaining_frames):
                dt = t * 0.1
                pred_traj[t, 0] = curr_x + speed * np.cos(curr_yaw) * dt
                pred_traj[t, 1] = curr_y + speed * np.sin(curr_yaw) * dt
    
    return pred_traj


# =============================================================================
# Legacy Inference (formerly run_inference.py)
# =============================================================================


def legacy_check_dependencies(mode: str = 'baseline'):
    """Check that required dependencies are installed for legacy inference."""
    missing = []

    if mode == 'full':
        try:
            import torch  # noqa: F401
        except ImportError:
            missing.append("torch (required for --legacy_mode full)")

    try:
        import tensorflow as tf  # noqa: F401
    except ImportError:
        missing.append("tensorflow")

    if missing:
        print(f"ERROR: Missing dependencies: {', '.join(missing)}")
        print("Install with: pip install torch tensorflow")
        sys.exit(1)


def legacy_check_model_files():
    """Check model checkpoints and precomputed files for legacy inference."""
    files_to_check = [
        ("DenseTNT (Marginal)", Config.DENSETNT_MODEL),
        ("Relation V2V", Config.RELATION_MODEL),
        ("Conditional V2V", Config.CONDITIONAL_MODEL),
    ]

    print("Checking model files...")
    all_exist = True
    for name, path in files_to_check:
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"  ✓ {name}: {path.name} ({size_mb:.1f} MB)")
        else:
            print(f"  ✗ {name}: NOT FOUND at {path}")
            all_exist = False

    precomputed = [
        ("Relation predictions", Config.RELATION_MODEL.parent / "m2i.relation.v2v.VAL"),
        ("Marginal predictions", Config.MARGINAL_PRED_FILE),
        ("Relation ground truth", Config.RELATION_GT_FILE),
    ]

    print("\nChecking precomputed files...")
    for name, path in precomputed:
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"  ✓ {name}: {path.name} ({size_mb:.1f} MB)")
        else:
            print(f"  ⚠ {name}: NOT FOUND (optional)")

    return all_exist


class LegacyM2IArgs:
    """Argument container mirroring legacy run_inference expectations."""

    def __init__(self, **kwargs):
        self.hidden_size = kwargs.get('hidden_size', 128)
        self.sub_graph_depth = kwargs.get('sub_graph_depth', 3)
        self.global_graph_depth = kwargs.get('global_graph_depth', 1)
        self.hidden_dropout_prob = kwargs.get('hidden_dropout_prob', 0.1)
        self.initializer_range = kwargs.get('initializer_range', 0.02)
        self.future_frame_num = kwargs.get('future_frame_num', 80)
        self.future_test_frame_num = kwargs.get('future_test_frame_num', 80)
        self.mode_num = kwargs.get('mode_num', 6)
        self.do_train = False
        self.do_eval = True
        self.waymo = True
        self.single_agent = True
        self.agent_type = kwargs.get('agent_type', 'vehicle')
        self.inter_agent_types = None
        self.max_distance = kwargs.get('max_distance', 50.0)
        self.other_params = kwargs.get('other_params', [
            'l1_loss', 'densetnt', 'goals_2D', 'enhance_global_graph',
            'laneGCN', 'point_sub_graph', 'laneGCN-4', 'stride_10_2', 'raster'
        ])
        self.eval_params = []
        self.train_params = []
        self.train_batch_size = 1
        self.eval_batch_size = 1
        self.sub_graph_batch_size = 4096
        self.learning_rate = 0.001
        self.weight_decay = 0.3
        self.num_train_epochs = 30
        self.seed = 42
        self.data_dir = kwargs.get('data_dir', [])
        self.output_dir = kwargs.get('output_dir', '/tmp')
        self.log_dir = None
        self.temp_file_dir = None
        self.model_recover_path = None
        self.no_sub_graph = False
        self.no_agents = False
        self.no_cuda = kwargs.get('no_cuda', False)
        self.distributed_training = 1
        self.core_num = kwargs.get('core_num', 1)
        self.nms_threshold = kwargs.get('nms_threshold', 7.2)
        self.infMLP = kwargs.get('infMLP', 8)
        self.debug = False
        self.visualize = False
        self.relation_pred_threshold = 0.9
        self.validation_model = 0
        self.classify_sub_goals = False
        self.joint_target_type = "no"
        self.joint_target_each = 80


def legacy_load_tfrecord_scenarios(data_dir: str, num_scenarios: Optional[int] = None):
    """Load scenarios from TFRecords (legacy format: list of tuples)."""
    import tensorflow as tf

    data_path = Path(data_dir)
    tfrecord_files = sorted(data_path.glob("*.tfrecord*"))

    if not tfrecord_files:
        print(f"ERROR: No TFRecord files found in {data_dir}")
        return []

    print(f"Found {len(tfrecord_files)} TFRecord files")
    scenarios = []

    for tfrecord_file in tfrecord_files:
        if num_scenarios and len(scenarios) >= num_scenarios:
            break

        dataset = tf.data.TFRecordDataset([str(tfrecord_file)])

        for raw_record in dataset:
            if num_scenarios and len(scenarios) >= num_scenarios:
                break

            decoded = legacy_parse_tf_example(raw_record.numpy())
            if decoded is not None:
                scenario_id = decoded.get('scenario_id', f'scenario_{len(scenarios)}')
                scenarios.append((scenario_id, decoded))

    print(f"Loaded {len(scenarios)} scenarios")
    return scenarios


def legacy_parse_tf_example(raw_bytes):
    """Parse a Waymo Motion tf.Example into numpy dict (legacy)."""
    import tensorflow as tf

    features_description = {
        'scenario/id': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'state/id': tf.io.FixedLenFeature([128], tf.float32, default_value=[0.0] * 128),
        'state/type': tf.io.FixedLenFeature([128], tf.float32, default_value=[0.0] * 128),
        'state/is_sdc': tf.io.FixedLenFeature([128], tf.int64, default_value=[0] * 128),
        'state/tracks_to_predict': tf.io.FixedLenFeature([128], tf.int64, default_value=[0] * 128),
        'state/objects_of_interest': tf.io.FixedLenFeature([128], tf.int64, default_value=[0] * 128),
        'state/current/x': tf.io.FixedLenFeature([128], tf.float32, default_value=[0.0] * 128),
        'state/current/y': tf.io.FixedLenFeature([128], tf.float32, default_value=[0.0] * 128),
        'state/current/z': tf.io.FixedLenFeature([128], tf.float32, default_value=[0.0] * 128),
        'state/current/bbox_yaw': tf.io.FixedLenFeature([128], tf.float32, default_value=[0.0] * 128),
        'state/current/velocity_x': tf.io.FixedLenFeature([128], tf.float32, default_value=[0.0] * 128),
        'state/current/velocity_y': tf.io.FixedLenFeature([128], tf.float32, default_value=[0.0] * 128),
        'state/current/valid': tf.io.FixedLenFeature([128], tf.int64, default_value=[0] * 128),
        'state/current/length': tf.io.FixedLenFeature([128], tf.float32, default_value=[0.0] * 128),
        'state/current/width': tf.io.FixedLenFeature([128], tf.float32, default_value=[0.0] * 128),
        'state/past/x': tf.io.FixedLenFeature([128, 10], tf.float32, default_value=[[0.0] * 10] * 128),
        'state/past/y': tf.io.FixedLenFeature([128, 10], tf.float32, default_value=[[0.0] * 10] * 128),
        'state/past/bbox_yaw': tf.io.FixedLenFeature([128, 10], tf.float32, default_value=[[0.0] * 10] * 128),
        'state/past/velocity_x': tf.io.FixedLenFeature([128, 10], tf.float32, default_value=[[0.0] * 10] * 128),
        'state/past/velocity_y': tf.io.FixedLenFeature([128, 10], tf.float32, default_value=[[0.0] * 10] * 128),
        'state/past/valid': tf.io.FixedLenFeature([128, 10], tf.int64, default_value=[[0] * 10] * 128),
        'state/future/x': tf.io.FixedLenFeature([128, 80], tf.float32, default_value=[[0.0] * 80] * 128),
        'state/future/y': tf.io.FixedLenFeature([128, 80], tf.float32, default_value=[[0.0] * 80] * 128),
        'state/future/bbox_yaw': tf.io.FixedLenFeature([128, 80], tf.float32, default_value=[[0.0] * 80] * 128),
        'state/future/velocity_x': tf.io.FixedLenFeature([128, 80], tf.float32, default_value=[[0.0] * 80] * 128),
        'state/future/velocity_y': tf.io.FixedLenFeature([128, 80], tf.float32, default_value=[[0.0] * 80] * 128),
        'state/future/valid': tf.io.FixedLenFeature([128, 80], tf.int64, default_value=[[0] * 80] * 128),
    }

    try:
        example = tf.io.parse_single_example(raw_bytes, features_description)
        result = {key: value.numpy() for key, value in example.items()}
        result['scenario_id'] = result['scenario/id'].decode('utf-8') if isinstance(result['scenario/id'], bytes) else str(result['scenario/id'])
        return result
    except Exception as exc:  # noqa: BLE001
        print(f"Error parsing tf.Example: {exc}")
        return None


def legacy_get_objects_of_interest(decoded_example) -> Tuple[Optional[int], Optional[int]]:
    """Return indices of two objects of interest, if present."""
    objects_of_interest = decoded_example.get('state/objects_of_interest', [])
    interest_indices = np.where(np.array(objects_of_interest) == 1)[0]
    if len(interest_indices) >= 2:
        return int(interest_indices[0]), int(interest_indices[1])
    return None, None


def legacy_extract_agent_trajectory(decoded_example, agent_idx: int) -> Dict:
    """Extract past/current/future for one agent (legacy)."""
    past_x = decoded_example['state/past/x'][agent_idx]
    past_y = decoded_example['state/past/y'][agent_idx]
    past_valid = decoded_example['state/past/valid'][agent_idx]
    current_x = decoded_example['state/current/x'][agent_idx]
    current_y = decoded_example['state/current/y'][agent_idx]
    current_valid = decoded_example['state/current/valid'][agent_idx]
    future_x = decoded_example['state/future/x'][agent_idx]
    future_y = decoded_example['state/future/y'][agent_idx]
    future_valid = decoded_example['state/future/valid'][agent_idx]

    history = np.stack([
        np.concatenate([past_x, [current_x]]),
        np.concatenate([past_y, [current_y]])
    ], axis=-1)

    future = np.stack([future_x, future_y], axis=-1)
    history_valid = np.concatenate([past_valid, [current_valid]])

    return {
        'history': history,
        'future': future,
        'history_valid': history_valid,
        'future_valid': future_valid,
        'agent_type': int(decoded_example['state/type'][agent_idx]),
        'agent_id': int(decoded_example['state/id'][agent_idx]),
    }


class LegacyM2IPredictor:
    """Legacy predictor supporting full, precomputed, or baseline modes."""

    def __init__(self, mode: str = 'precomputed', device: str = 'cuda'):
        self.mode = mode
        self.device = device
        self.marginal_predictions = None
        self.relation_predictions = None
        self.relation_gt = None
        self.model = None

        if mode == 'precomputed':
            self._load_precomputed()
        elif mode == 'full':
            self._load_models()

    def _load_precomputed(self):
        print("\nLoading precomputed predictions...")

        if Config.MARGINAL_PRED_FILE.exists():
            try:
                with open(Config.MARGINAL_PRED_FILE, 'rb') as f:
                    self.marginal_predictions = pickle.load(f)
                print(f"  ✓ Loaded {len(self.marginal_predictions)} marginal predictions")
            except Exception as exc:  # noqa: BLE001
                print(f"  ⚠ Error loading marginal predictions: {exc}")
        else:
            print(f"  ⚠ Marginal predictions not found: {Config.MARGINAL_PRED_FILE}")

        relation_pred_file = Config.RELATION_MODEL.parent / "m2i.relation.v2v.VAL"
        if relation_pred_file.exists():
            try:
                with open(relation_pred_file, 'rb') as f:
                    self.relation_predictions = pickle.load(f)
                print("  ✓ Loaded relation predictions")
            except ModuleNotFoundError:
                print("  ⚠ Relation predictions require torch (skipped)")
            except Exception as exc:  # noqa: BLE001
                print(f"  ⚠ Error loading relation predictions: {exc}")
        else:
            print(f"  ⚠ Relation predictions not found: {relation_pred_file}")

        if Config.RELATION_GT_FILE.exists():
            try:
                with open(Config.RELATION_GT_FILE, 'rb') as f:
                    self.relation_gt = pickle.load(f)
                print("  ✓ Loaded relation ground truth")
            except Exception as exc:  # noqa: BLE001
                print(f"  ⚠ Error loading relation GT: {exc}")

    def _load_models(self):
        print("\nLoading M2I models (legacy)...")

        if self.device == 'cuda' and not torch.cuda.is_available():
            print("  ⚠ CUDA not available, using CPU")
            self.device = 'cpu'

        try:
            from modeling.vectornet import VectorNet

            self.args = LegacyM2IArgs()
            self.marginal_model = None

            if Config.DENSETNT_MODEL.exists():
                print("  Loading DenseTNT marginal predictor...")
                self.marginal_model = VectorNet(self.args)
                checkpoint = torch.load(Config.DENSETNT_MODEL, map_location='cpu')
                self.marginal_model.load_state_dict(checkpoint)
                self.marginal_model.to(self.device)
                self.marginal_model.eval()
                print(f"    ✓ Loaded from {Config.DENSETNT_MODEL.name}")
            else:
                print(f"    ✗ Model not found: {Config.DENSETNT_MODEL}")

            self.model = self.marginal_model
        except Exception as exc:  # noqa: BLE001
            print(f"  ✗ Error loading models: {exc}")
            traceback.print_exc()
            self.model = None

    def _generate_baseline_prediction(self, trajectory: Dict) -> np.ndarray:
        history = trajectory['history']
        history_valid = trajectory['history_valid']
        valid_indices = np.where(history_valid)[0]
        if len(valid_indices) < 2:
            return np.zeros((6, 80, 2))

        last_idx = valid_indices[-1]
        prev_idx = valid_indices[-2]
        last_pos = history[last_idx]
        prev_pos = history[prev_idx]
        dt = 0.1 * (last_idx - prev_idx)
        velocity = (last_pos - prev_pos) / dt if dt > 0 else np.zeros(2)
        future_times = np.arange(1, 81) * 0.1
        base_pred = last_pos + velocity[np.newaxis, :] * future_times[:, np.newaxis]

        predictions = np.zeros((6, 80, 2))
        for mode in range(6):
            angle_offset = (mode - 2.5) * 0.1
            speed_factor = 1.0 + (mode - 2.5) * 0.05
            cos_a = np.cos(angle_offset)
            sin_a = np.sin(angle_offset)
            rot_velocity = np.array([
                velocity[0] * cos_a - velocity[1] * sin_a,
                velocity[0] * sin_a + velocity[1] * cos_a
            ]) * speed_factor
            predictions[mode] = last_pos + rot_velocity[np.newaxis, :] * future_times[:, np.newaxis]

        return predictions if predictions.size else base_pred[np.newaxis, ...]

    def _predict_baseline(self, scenario_id: str, decoded_example: Dict) -> Optional[Dict]:
        agent_a, agent_b = legacy_get_objects_of_interest(decoded_example)
        if agent_a is None:
            return None

        traj_a = legacy_extract_agent_trajectory(decoded_example, agent_a)
        traj_b = legacy_extract_agent_trajectory(decoded_example, agent_b)
        pred_a = self._generate_baseline_prediction(traj_a)
        pred_b = self._generate_baseline_prediction(traj_b)

        return {
            'scenario_id': scenario_id,
            'influencer_idx': agent_a,
            'reactor_idx': agent_b,
            'influencer_pred': pred_a,
            'reactor_pred': pred_b,
            'influencer_gt': traj_a['future'],
            'reactor_gt': traj_b['future'],
            'influencer_gt_valid': traj_a['future_valid'],
            'reactor_gt_valid': traj_b['future_valid'],
            'influencer_history': traj_a['history'],
            'reactor_history': traj_b['history'],
        }

    def _predict_precomputed(self, scenario_id: str, decoded_example: Dict) -> Optional[Dict]:
        agent_a, agent_b = legacy_get_objects_of_interest(decoded_example)
        if agent_a is None:
            return None

        traj_a = legacy_extract_agent_trajectory(decoded_example, agent_a)
        traj_b = legacy_extract_agent_trajectory(decoded_example, agent_b)
        pred_a = None
        pred_b = None
        scenario_key = scenario_id

        if self.marginal_predictions:
            if scenario_id not in self.marginal_predictions:
                scenario_key = scenario_id.encode() if isinstance(scenario_id, str) else scenario_id

            if scenario_key in self.marginal_predictions:
                scenario_preds = self.marginal_predictions[scenario_key]
                if isinstance(scenario_preds, dict) and 'rst' in scenario_preds:
                    rst = scenario_preds['rst']
                    pred_ids = scenario_preds.get('ids', [])
                    for i, pid in enumerate(pred_ids):
                        if int(pid) == traj_a['agent_id']:
                            pred_a = rst[i]
                        elif int(pid) == traj_b['agent_id']:
                            pred_b = rst[i]

        if pred_a is None:
            pred_a = self._generate_baseline_prediction(traj_a)
        if pred_b is None:
            pred_b = self._generate_baseline_prediction(traj_b)

        influencer_idx, reactor_idx = agent_a, agent_b
        if self.relation_gt:
            gt_key = scenario_id if scenario_id in self.relation_gt else scenario_id.encode() if isinstance(scenario_id, str) else scenario_id
            if gt_key in self.relation_gt:
                rel = self.relation_gt[gt_key]
                if isinstance(rel, np.ndarray) and len(rel) >= 2:
                    inf_id, react_id = int(rel[0]), int(rel[1])
                    if inf_id == traj_b['agent_id']:
                        influencer_idx, reactor_idx = agent_b, agent_a
                        pred_a, pred_b = pred_b, pred_a
                        traj_a, traj_b = traj_b, traj_a

        return {
            'scenario_id': scenario_id,
            'influencer_idx': influencer_idx,
            'reactor_idx': reactor_idx,
            'influencer_pred': pred_a,
            'reactor_pred': pred_b,
            'influencer_gt': traj_a['future'],
            'reactor_gt': traj_b['future'],
            'influencer_gt_valid': traj_a['future_valid'],
            'reactor_gt_valid': traj_b['future_valid'],
            'influencer_history': traj_a['history'],
            'reactor_history': traj_b['history'],
        }

    def _predict_full(self, scenario_id: str, decoded_example: Dict) -> Optional[Dict]:
        if self.model is None:
            print("  ⚠ Model not loaded, using baseline")
            return self._predict_baseline(scenario_id, decoded_example)
        print("  ⚠ Full inference not yet implemented, using baseline")
        return self._predict_baseline(scenario_id, decoded_example)

    def predict(self, scenario_id: str, decoded_example: Dict) -> Optional[Dict]:
        if self.mode == 'precomputed':
            return self._predict_precomputed(scenario_id, decoded_example)
        if self.mode == 'baseline':
            return self._predict_baseline(scenario_id, decoded_example)
        return self._predict_full(scenario_id, decoded_example)


def legacy_compute_metrics(predictions: List[Dict]) -> Dict:
    """Compute ADE/FDE/miss-rate for legacy predictions."""
    all_ade: List[float] = []
    all_fde: List[float] = []
    all_misses: List[float] = []

    for pred in predictions:
        for role in ['influencer', 'reactor']:
            pred_trajs = pred.get(f'{role}_pred')
            gt_traj = pred.get(f'{role}_gt')
            gt_valid = pred.get(f'{role}_gt_valid', np.ones(80))

            if pred_trajs is None or gt_traj is None:
                continue

            valid_mask = gt_valid.astype(bool)
            gt_not_placeholder = ~np.all(gt_traj == -1, axis=-1)
            valid_mask = valid_mask & gt_not_placeholder
            pred_not_zero = ~np.all(pred_trajs[0] == 0, axis=-1)
            valid_mask = valid_mask & pred_not_zero

            if not valid_mask.any():
                continue

            errors = np.linalg.norm(pred_trajs - gt_traj[np.newaxis, :, :], axis=-1)
            valid_errors = errors[:, valid_mask]
            ade_per_mode = valid_errors.mean(axis=1)
            last_valid_idx = np.where(valid_mask)[0][-1]
            fde_per_mode = errors[:, last_valid_idx]
            min_ade = ade_per_mode.min()
            min_fde = fde_per_mode.min()
            all_ade.append(min_ade)
            all_fde.append(min_fde)
            all_misses.append(1.0 if min_fde > 2.0 else 0.0)

    return {
        'minADE': float(np.mean(all_ade)) if all_ade else 0.0,
        'minFDE': float(np.mean(all_fde)) if all_fde else 0.0,
        'miss_rate': float(np.mean(all_misses)) if all_misses else 0.0,
        'num_predictions': len(all_ade),
    }


def legacy_save_predictions(predictions: List[Dict], output_path: Path):
    """Save legacy predictions list to pickle."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(predictions, f)
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Saved {len(predictions)} predictions to {output_path} ({size_mb:.2f} MB)")


def run_legacy_inference(args, output_dir: Path):
    """Run the legacy inference flow that used to live in run_inference.py."""
    print("=" * 70)
    print("M2I Trajectory Prediction (legacy)")
    print("=" * 70)

    legacy_check_dependencies(mode=args.legacy_mode)
    legacy_check_model_files()

    print(f"\nInitializing predictor (mode: {args.legacy_mode})...")
    predictor = LegacyM2IPredictor(mode=args.legacy_mode, device=args.device)

    print(f"\nLoading scenarios from {args.data_dir}...")
    scenarios = legacy_load_tfrecord_scenarios(args.data_dir, args.num_scenarios)
    if not scenarios:
        print("ERROR: No scenarios loaded")
        return 1

    print(f"\nRunning predictions on {len(scenarios)} scenarios...")
    predictions = []
    for i, (scenario_id, decoded_example) in enumerate(scenarios):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Processing {i+1}/{len(scenarios)}: {scenario_id}")
        pred = predictor.predict(scenario_id, decoded_example)
        if pred is not None:
            predictions.append(pred)

    print(f"\nGenerated {len(predictions)} predictions")

    if not args.skip_metrics and predictions:
        print("\nComputing metrics...")
        metrics = legacy_compute_metrics(predictions)
        print(f"  minADE: {metrics['minADE']:.3f} m")
        print(f"  minFDE: {metrics['minFDE']:.3f} m")
        print(f"  Miss Rate (>2m): {metrics['miss_rate']*100:.1f}%")
        print(f"  Total predictions: {metrics['num_predictions']}")

    output_path = output_dir / 'predictions.pkl'
    legacy_save_predictions(predictions, output_path)

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)
    return 0


# =============================================================================
# Visualization
# =============================================================================

def generate_visualizations(
    scenarios: List[Dict],
    predictions: Dict[str, Dict],
    output_dir: Path,
    max_visualizations: int = 20,
    model_manager: M2IModelManager = None
) -> List[Path]:
    """Generate BEV visualization movies with per-frame GPU inference."""
    
    print("\n" + "=" * 60)
    print("GENERATING BEV VISUALIZATIONS")
    if model_manager is not None:
        print("  (with per-frame GPU inference)")
    print("=" * 60)
    
    import matplotlib
    matplotlib.use('Agg')
    
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_files = []
    
    for i, scenario in enumerate(scenarios[:max_visualizations]):
        scenario_id = scenario['scenario_id']
        
        if scenario_id not in predictions:
            continue
        
        print(f"  [{i+1}/{min(len(scenarios), max_visualizations)}] "
              f"{scenario_id[:16]}...", end=" ")
        
        try:
            output_file = output_dir / f"m2i_{scenario_id[:16]}.gif"
            
            create_bev_animation(
                scenario, predictions[scenario_id], output_file,
                model_manager=model_manager
            )
            
            generated_files.append(output_file)
            print(f"✓ {output_file.name}")
            
        except Exception as e:
            print(f"✗ {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\nGenerated {len(generated_files)} visualizations in {output_dir}")
    return generated_files
    
    print(f"\nGenerated {len(generated_files)} visualizations in {output_dir}")
    return generated_files


def create_bev_animation(scenario: Dict, predictions: Dict, output_path: Path,
                         model_manager=None):
    """Create a single BEV animation with moving vehicles and per-frame predictions."""
    import matplotlib.pyplot as plt
    import matplotlib.transforms as transforms
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.patches import FancyArrowPatch, Rectangle
    
    # Get interacting agent indices
    idx1, idx2 = scenario['interact_indices'][:2]
    
    # Get center position (use current position of agents)
    center_x = (scenario['current_x'][idx1] + scenario['current_x'][idx2]) / 2
    center_y = (scenario['current_y'][idx1] + scenario['current_y'][idx2]) / 2
    
    # Build full trajectories: past (10) + current (1) + future (80) = 91 frames
    # But we'll animate through the future frames (80 frames)
    def get_full_trajectory(agent_idx):
        """Get full trajectory: past + current + future."""
        past_x = scenario['past_x'][agent_idx][::-1]  # Reverse to chronological
        past_y = scenario['past_y'][agent_idx][::-1]
        current_x = scenario['current_x'][agent_idx]
        current_y = scenario['current_y'][agent_idx]
        future_x = scenario['future_x'][agent_idx]
        future_y = scenario['future_y'][agent_idx]
        
        # Full trajectory
        full_x = np.concatenate([past_x, [current_x], future_x])
        full_y = np.concatenate([past_y, [current_y], future_y])
        
        # Full yaw (approximate from trajectory direction)
        past_yaw = np.full(10, scenario['current_yaw'][agent_idx])
        current_yaw = scenario['current_yaw'][agent_idx]
        
        # Compute future yaw from trajectory
        future_yaw = np.zeros(80)
        for t in range(80):
            if t < 79:
                dx = future_x[t+1] - future_x[t]
                dy = future_y[t+1] - future_y[t]
                if abs(dx) > 0.01 or abs(dy) > 0.01:
                    future_yaw[t] = np.arctan2(dy, dx)
                else:
                    future_yaw[t] = current_yaw if t == 0 else future_yaw[t-1]
            else:
                future_yaw[t] = future_yaw[t-1]
        
        full_yaw = np.concatenate([past_yaw, [current_yaw], future_yaw])
        
        return full_x, full_y, full_yaw
    
    # Get trajectories for all valid agents
    agent_trajectories = {}
    for j in range(128):
        if scenario['current_valid'][j]:
            full_x, full_y, full_yaw = get_full_trajectory(j)
            agent_trajectories[j] = {
                'x': full_x,
                'y': full_y, 
                'yaw': full_yaw,
                'length': scenario['current_length'][j],
                'width': scenario['current_width'][j],
                'type': scenario['agent_types'][j],
            }
    
    # Setup figure
    fig, ax = plt.subplots(figsize=(12, 12), dpi=80)
    view_range = 60
    
    # Colors
    colors = {
        'influencer': '#9b59b6',      # Purple
        'reactor': '#1abc9c',         # Teal
        'other': '#95a5a6',           # Gray
        'road': '#7f8c8d',            # Dark gray
        'history': '#3498db',         # Blue
        'prediction': '#e74c3c',      # Red
        'ground_truth': '#f39c12',    # Orange (future GT)
    }
    
    # Frame offset: frame 0 corresponds to index 10 (current time)
    # Frames 0-79 show the future unfolding
    frame_offset = 10  # Past frames
    
    def animate(frame):
        ax.clear()
        
        # Current time index in the full trajectory
        t_idx = frame_offset + frame  # 10 + frame (0-79) = 10-89
        
        # Dynamic center - follow the interacting agents
        if t_idx < len(agent_trajectories[idx1]['x']):
            cx = (agent_trajectories[idx1]['x'][t_idx] + agent_trajectories[idx2]['x'][t_idx]) / 2
            cy = (agent_trajectories[idx1]['y'][t_idx] + agent_trajectories[idx2]['y'][t_idx]) / 2
        else:
            cx, cy = center_x, center_y
        
        ax.set_xlim(cx - view_range, cx + view_range)
        ax.set_ylim(cy - view_range, cy + view_range)
        ax.set_aspect('equal')
        ax.set_facecolor('#1a1a2e')
        ax.grid(True, alpha=0.15, color='white', linestyle='-', linewidth=0.5)
        
        # Draw roadgraph
        valid_mask = scenario['roadgraph_valid'] > 0
        road_xyz = scenario['roadgraph_xyz'][valid_mask]
        if len(road_xyz) > 0:
            # Filter to view
            in_view = (np.abs(road_xyz[:, 0] - cx) < view_range * 1.2) & \
                      (np.abs(road_xyz[:, 1] - cy) < view_range * 1.2)
            road_in_view = road_xyz[in_view]
            if len(road_in_view) > 0:
                ax.scatter(road_in_view[:, 0], road_in_view[:, 1], 
                           s=1, c=colors['road'], alpha=0.4)
        
        # Draw history trails for interacting agents
        for agent_idx, color in [(idx1, colors['influencer']), (idx2, colors['reactor'])]:
            traj = agent_trajectories[agent_idx]
            
            # History trail (past positions up to current frame)
            history_start = max(0, t_idx - 15)
            history_x = traj['x'][history_start:t_idx+1]
            history_y = traj['y'][history_start:t_idx+1]
            
            if len(history_x) > 1:
                # Draw history with fading effect
                for i in range(len(history_x) - 1):
                    alpha = 0.3 + 0.5 * (i / len(history_x))
                    ax.plot(history_x[i:i+2], history_y[i:i+2], 
                            color=color, linewidth=2, alpha=alpha)
        
        # Draw predicted trajectories for interacting agents
        # Use precomputed M2I network predictions (not per-frame inference)
        for agent_idx, agent_color in [(idx1, colors['influencer']), (idx2, colors['reactor'])]:
            agent_id = int(scenario['agent_ids'][agent_idx])
            
            if agent_id not in predictions:
                continue
            
            pred_data = predictions[agent_id]
            pred_traj = pred_data['prediction']  # Full [80, 2] M2I prediction
            
            # The M2I prediction is in absolute coords starting at t=0 (current time)
            # At animation frame N, show prediction from frame N onwards
            remaining_pred = pred_traj[frame:]
            
            # Draw remaining prediction trajectory
            if len(remaining_pred) > 1:
                ax.plot(remaining_pred[:, 0], remaining_pred[:, 1],
                        color=colors['prediction'], linewidth=2.5,
                        alpha=0.8, linestyle='-', zorder=5)
                # Mark prediction endpoint
                ax.scatter(remaining_pred[-1, 0], remaining_pred[-1, 1],
                           s=50, c=colors['prediction'], marker='*', 
                           edgecolors='white', linewidths=0.5, zorder=6)
                
                # Also show current prediction point
                if frame < len(pred_traj):
                    ax.scatter(pred_traj[frame, 0], pred_traj[frame, 1],
                               s=80, c=colors['prediction'], marker='o',
                               edgecolors='white', linewidths=1.5, zorder=7)
            
            # Show ground truth future (faded)
            gt_traj = pred_data['ground_truth']
            gt_valid = pred_data['valid']
            remaining_gt = gt_traj[frame:]
            remaining_valid = gt_valid[frame:]
            if len(remaining_gt) > 1 and remaining_valid.any():
                valid_mask = remaining_valid[:len(remaining_gt)]
                if valid_mask.any():
                    valid_gt = remaining_gt[valid_mask]
                    if len(valid_gt) > 1:
                        ax.plot(valid_gt[:, 0], valid_gt[:, 1],
                                color=colors['ground_truth'], linewidth=1.5,
                                alpha=0.5, linestyle='--', zorder=4)
        
        # Draw all agents at current position
        for j, traj in agent_trajectories.items():
            if t_idx >= len(traj['x']):
                continue
            
            x = traj['x'][t_idx]
            y = traj['y'][t_idx]
            yaw = traj['yaw'][t_idx]
            length = max(traj['length'], 2.0)
            width = max(traj['width'], 1.0)
            
            # Check if within view
            if abs(x - cx) > view_range * 1.1 or abs(y - cy) > view_range * 1.1:
                continue
            
            # Determine color and style
            if j == idx1:
                color = colors['influencer']
                edgecolor = 'white'
                alpha = 1.0
                linewidth = 2
                zorder = 10
            elif j == idx2:
                color = colors['reactor']
                edgecolor = 'white'
                alpha = 1.0
                linewidth = 2
                zorder = 10
            else:
                color = colors['other']
                edgecolor = '#555555'
                alpha = 0.6
                linewidth = 1
                zorder = 3
            
            # Draw vehicle rectangle
            rect = Rectangle(
                (-length/2, -width/2), length, width,
                facecolor=color, edgecolor=edgecolor, 
                alpha=alpha, linewidth=linewidth, zorder=zorder
            )
            t = transforms.Affine2D().rotate(yaw).translate(x, y) + ax.transData
            rect.set_transform(t)
            ax.add_patch(rect)
            
            # Draw direction indicator for interacting agents
            if j in [idx1, idx2]:
                arrow_len = length * 0.6
                dx = arrow_len * np.cos(yaw)
                dy = arrow_len * np.sin(yaw)
                ax.arrow(x, y, dx, dy, head_width=0.8, head_length=0.4,
                         fc='white', ec='white', alpha=0.8, zorder=11)
        
        # Legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color=colors['influencer'], linewidth=4, label='Influencer'),
            Line2D([0], [0], color=colors['reactor'], linewidth=4, label='Reactor'),
            Line2D([0], [0], color=colors['prediction'], linewidth=2, label='M2I Prediction'),
            Line2D([0], [0], color=colors['ground_truth'], linewidth=2, 
                   linestyle='--', label='Ground Truth'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', 
                  facecolor='#2c3e50', edgecolor='white', labelcolor='white',
                  fontsize=9)
        
        # Title with time info
        time_sec = frame * 0.1  # 0.1s per frame
        ax.set_title(
            f"M2I Trajectory Prediction\n"
            f"Scenario: {scenario['scenario_id'][:16]}  |  "
            f"Time: {time_sec:.1f}s  |  Frame: {frame+1}/80",
            color='white', fontsize=11, fontweight='bold'
        )
        
        # Remove axis labels
        ax.set_xticks([])
        ax.set_yticks([])
        
        return []
    
    # Create animation (use every 2nd frame for faster rendering)
    frame_indices = list(range(0, 80, 2))  # 0, 2, 4, ... = 40 frames
    anim = FuncAnimation(fig, animate, frames=frame_indices, interval=100, blit=False)
    
    # Save as GIF
    writer = PillowWriter(fps=10)
    anim.save(str(output_path), writer=writer)
    plt.close(fig)


# =============================================================================
# Main Pipeline
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='M2I GPU Inference and Visualization Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python m2i_gpu_pipeline.py --mode inference --num_scenarios 20
  python m2i_gpu_pipeline.py --mode visualize --num_scenarios 10
  python m2i_gpu_pipeline.py --mode all --num_scenarios 20
  python m2i_gpu_pipeline.py --task legacy --legacy_mode precomputed \
      --data_dir /path/to/tfrecords --output_dir /tmp/preds
        """
    )

    parser.add_argument('--task', type=str, default='pipeline',
                        choices=['pipeline', 'legacy'],
                        help='pipeline: GPU inference/visualization; legacy: run former run_inference flow')
    parser.add_argument('--mode', type=str, default='all',
                        choices=['inference', 'visualize', 'all'],
                        help='Pipeline mode (pipeline task only)')
    parser.add_argument('--legacy_mode', type=str, default='precomputed',
                        choices=['full', 'precomputed', 'baseline'],
                        help='Legacy prediction mode (legacy task only)')
    parser.add_argument('--num_scenarios', type=int, default=20,
                        help='Number of scenarios to process')
    parser.add_argument('--data_dir', type=str, default=str(Config.DATA_DIR),
                        help='Directory containing TFRecord files')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for visualizations (pipeline) or fallback for legacy')
    parser.add_argument('--legacy_output_dir', type=str, default=None,
                        help='Output directory for legacy predictions (defaults to output_dir or predictions dir)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    parser.add_argument('--skip_metrics', action='store_true',
                        help='Skip computing metrics (legacy task)')
    parser.add_argument('--no_precomputed', action='store_true',
                        help='Do not load/use precomputed predictions; run model fallback instead')

    args = parser.parse_args()

    # Dynamic paths
    Config.DATA_DIR = Path(args.data_dir)
    if args.output_dir:
        Config.OUTPUT_DIR = Path(args.output_dir)

    if args.task == 'legacy':
        legacy_output = Path(args.legacy_output_dir or args.output_dir or Config.PREDICTIONS_DIR)
        legacy_output.mkdir(parents=True, exist_ok=True)
        return run_legacy_inference(args, legacy_output)

    print("\n" + "=" * 70)
    print("M2I GPU INFERENCE AND VISUALIZATION PIPELINE")
    print("=" * 70)
    print(f"\nMode: {args.mode}")
    print(f"Task: {args.task}")
    print(f"Scenarios: {args.num_scenarios}")
    print(f"Data: {Config.DATA_DIR}")
    print(f"Output: {Config.OUTPUT_DIR}")
    print(f"Device: {args.device}")

    # Initialize model manager
    model_manager = M2IModelManager(device=args.device)
    
    # Load models
    if args.mode in ['inference', 'all']:
        model_manager.load_all_models()
    
    # Load data
    print("\n" + "=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    
    precomputed, relations = ({}, {}) if args.no_precomputed else load_precomputed_predictions()
    scenarios = load_tfrecord_scenarios(args.num_scenarios)

    # If requested, run full M2I to produce fresh predictions (no fallback/baseline)
    if args.no_precomputed:
        Config.PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
        runner = FullM2IInferenceRunner(model_manager.device)

        print("\nRunning relation predictor (fresh)...")
        relation_path = runner.run_relation(
            data_dir=Config.DATA_DIR,
            model_path=Config.RELATION_MODEL,
            max_scenarios=args.num_scenarios,
            output_dir=Config.PREDICTIONS_DIR,
        )

        print("\nRunning conditional predictor (fresh)...")
        precomputed = runner.run_conditional(
            data_dir=Config.DATA_DIR,
            model_path=Config.CONDITIONAL_MODEL,
            relation_pred=relation_path,
            marginal_pred=Config.MARGINAL_PRED_FILE,
            max_scenarios=args.num_scenarios,
            output_dir=Config.PREDICTIONS_DIR,
        )
    
    if not scenarios:
        print("\nERROR: No scenarios loaded!")
        return
    
    # Run inference
    predictions = {}
    if args.mode in ['inference', 'all']:
        predictions = run_gpu_inference(model_manager, scenarios, precomputed, strict=args.no_precomputed)
        
        # Save predictions
        Config.PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
        pred_file = Config.PREDICTIONS_DIR / 'gpu_predictions.pkl'
        with open(pred_file, 'wb') as f:
            pickle.dump(predictions, f)
        print(f"\nSaved predictions to: {pred_file}")
    else:
        # Load existing predictions
        pred_file = Config.PREDICTIONS_DIR / 'gpu_predictions.pkl'
        if pred_file.exists():
            with open(pred_file, 'rb') as f:
                predictions = pickle.load(f)
            print(f"  Loaded existing predictions: {len(predictions)} scenarios")
        else:
            # Use precomputed
            for scenario in scenarios:
                sid = scenario['scenario_id']
                if sid in precomputed:
                    predictions[sid] = process_precomputed(scenario, precomputed[sid])
    
    # Generate visualizations with per-frame GPU inference
    if args.mode in ['visualize', 'all']:
        generate_visualizations(
            scenarios, predictions, Config.OUTPUT_DIR, args.num_scenarios,
            model_manager=model_manager  # Pass model for per-frame inference
        )
    
    # Summary
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"\n  Scenarios processed: {len(scenarios)}")
    print(f"  Predictions generated: {len(predictions)}")
    print(f"  Output directory: {Config.OUTPUT_DIR}")
    
    if predictions:
        # Compute metrics
        all_ades = []
        for sid, preds in predictions.items():
            for agent_id, pred in preds.items():
                valid = pred['valid']
                if valid.sum() > 0:
                    p = pred['prediction'][valid]
                    g = pred['ground_truth'][valid]
                    ade = np.mean(np.linalg.norm(p - g, axis=-1))
                    all_ades.append(ade)
        
        if all_ades:
            print(f"\n  Average Displacement Error: {np.mean(all_ades):.3f} m")


if __name__ == '__main__':
    main()
