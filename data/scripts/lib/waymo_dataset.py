"""Data loading and preprocessing for Waymo Open Motion Dataset - ITP Format."""

import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Try importing TensorFlow for tf_example support
try:
    import tensorflow as tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    HAS_TF = True
except ImportError:
    HAS_TF = False
    warnings.warn("TensorFlow not found. TFExample dataset will not be available.")


class WaymoITPDataset(Dataset):
    """
    Dataset for preprocessed Waymo scenarios for ITP training.
    
    Loads preprocessed .npz files created by preprocess_waymo_for_itp.py
    """

    def __init__(
        self,
        data_dir: str,
        relation_file: Optional[str] = None,
        influencer_pred_file: Optional[str] = None,
        split: str = "train",
        short_horizon_frames: int = 80,  # 8s at 10Hz
        long_horizon_frames: int = 160,  # 16s at 10Hz
        history_frames: int = 11,  # 1.1s history
        use_gt_influencer: bool = True,  # For training conditional model
        augment: bool = False,
        max_other_agents: int = 5,  # Max other agents to include
    ):
        """
        Args:
            data_dir: Path to preprocessed .npz files
            relation_file: Path to pre-computed relation labels pickle
            influencer_pred_file: Path to predicted influencer trajectories (for inference)
            split: 'train' or 'val'
            short_horizon_frames: Frames for short horizon (M2I baseline)
            long_horizon_frames: Total frames including extension
            history_frames: Past observation frames
            use_gt_influencer: Use ground truth influencer for training
            augment: Apply data augmentation
            max_other_agents: Maximum number of other agents to include
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.short_horizon = short_horizon_frames
        self.long_horizon = long_horizon_frames
        self.history = history_frames
        self.use_gt = use_gt_influencer
        self.augment = augment and split == "train"
        self.max_other_agents = max_other_agents

        # Load relation labels
        self.relations = None
        if relation_file and Path(relation_file).exists():
            with open(relation_file, "rb") as f:
                self.relations = pickle.load(f)

        # Load influencer predictions (for validation/test)
        self.influencer_preds = None
        if influencer_pred_file and Path(influencer_pred_file).exists():
            with open(influencer_pred_file, "rb") as f:
                self.influencer_preds = pickle.load(f)

        # Load metadata
        metadata_file = self.data_dir / "metadata.pkl"
        if metadata_file.exists():
            with open(metadata_file, "rb") as f:
                self.metadata = pickle.load(f)
                print(f"Loaded metadata: {self.metadata['num_scenarios']} scenarios")
        else:
            self.metadata = {}
            print("Warning: No metadata file found")

        # Load scenarios
        self.scenarios = self._load_scenarios()
        print(f"Loaded {len(self.scenarios)} scenarios from {data_dir}")

    def _load_scenarios(self) -> List[Path]:
        """
        Load list of preprocessed scenario files.
        
        Returns:
            List of paths to .npz files
        """
        scenario_files = sorted(self.data_dir.glob("*.npz"))
        
        if len(scenario_files) == 0:
            print(f"\n⚠️  WARNING: No .npz files found in {self.data_dir}")
            print("   This dataset appears to be empty or not yet preprocessed.")
            print("   Run preprocessing first:")
            print("   ./data/scripts/preprocess_all.sh --test")
            return []  # Return empty list instead of raising error
        
        return scenario_files

    def __len__(self) -> int:
        return len(self.scenarios)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single training sample.
        
        Returns:
            Dictionary containing:
            - 'scenario_id': str
            - 'history': (2, history_frames, 2) past positions for pair
            - 'future_short': (2, short_horizon, 2) short-horizon ground truth
            - 'future_long': (2, long_horizon - short_horizon, 2) long-tail ground truth
            - 'valid_mask': (2, long_horizon) validity mask
            - 'relation': int (0=PASS, 1=YIELD, 2=NONE)
            - 'agent_types': (2,) agent type codes
            - 'map_polylines': (M, P, 7) map features [x, y, dx, dy, type, valid, ...]
            - 'other_agents': (K, long_horizon, 2) other agents' futures
            - 'influencer_pred': (short_horizon, 2) if available
            - 'ego_last_state': (5,) [x, y, heading, v, yaw_rate] at t=short_horizon
        """
        # Load scenario from disk
        scenario_file = self.scenarios[idx]
        data = np.load(scenario_file, allow_pickle=True)
        
        scenario_id = str(data["scenario_id"])
        tracks = data["tracks"].item()  # Convert 0-d array to dict
        map_data = data["map"].item()
        pair_indices = tuple(data["pair_indices"])
        
        # Get interactive pair
        i, j = pair_indices
        
        # Extract trajectories
        T_total = self.history + self.long_horizon
        
        # Check if we have enough frames
        if tracks["pos"].shape[1] < T_total:
            # Pad if necessary
            pad_frames = T_total - tracks["pos"].shape[1]
            for key in ["pos", "vel", "heading", "valid"]:
                if key in ["pos", "vel"]:
                    pad_value = 0.0
                    pad_shape = (tracks[key].shape[0], pad_frames, 2)
                elif key == "heading":
                    pad_value = 0.0
                    pad_shape = (tracks[key].shape[0], pad_frames)
                else:  # valid
                    pad_value = False
                    pad_shape = (tracks[key].shape[0], pad_frames)
                
                tracks[key] = np.concatenate([
                    tracks[key],
                    np.full(pad_shape, pad_value, dtype=tracks[key].dtype)
                ], axis=1)
        
        # History: frames [0:history]
        history = tracks["pos"][[i, j], :self.history]  # (2, history, 2)
        
        # Future: frames [history:history+long_horizon]
        future_all = tracks["pos"][[i, j], self.history:self.history + self.long_horizon]
        future_short = future_all[:, :self.short_horizon]  # (2, short_horizon, 2)
        future_long = future_all[:, self.short_horizon:]  # (2, long-short, 2)
        
        # Valid mask
        valid = tracks["valid"][[i, j], self.history:self.history + self.long_horizon]
        
        # Get relation label
        relation = 2  # Default: NONE
        if self.relations and scenario_id in self.relations:
            relation = self.relations[scenario_id]
        
        # Agent types
        agent_types = tracks["type"][[i, j]]
        
        # Map polylines
        map_polylines = self._encode_map(map_data)
        
        # Other agents (top K by proximity)
        other_agents = self._get_other_agents(tracks, [i, j], K=self.max_other_agents)
        
        # Influencer prediction (if available)
        influencer_pred = None
        if self.influencer_preds and scenario_id in self.influencer_preds:
            influencer_pred = self.influencer_preds[scenario_id][:self.short_horizon]
        elif self.use_gt:
            # Use GT for training (assume agent 0 is influencer)
            influencer_pred = future_short[0]
        
        # Ego last state (for kinematic rollout)
        ego_last_state = self._get_last_state(tracks, i, self.history + self.short_horizon - 1)
        
        # Apply augmentation if enabled
        if self.augment:
            (
                history,
                future_short,
                future_long,
                map_polylines,
                other_agents,
                influencer_pred,
                ego_last_state,
            ) = self._augment(
                history,
                future_short,
                future_long,
                map_polylines,
                other_agents,
                influencer_pred,
                ego_last_state,
            )
        
        # Convert to tensors
        return {
            "scenario_id": scenario_id,
            "history": torch.FloatTensor(history),
            "future_short": torch.FloatTensor(future_short),
            "future_long": torch.FloatTensor(future_long),
            "valid_mask": torch.BoolTensor(valid),
            "relation": torch.LongTensor([relation]),
            "agent_types": torch.LongTensor(agent_types),
            "map_polylines": torch.FloatTensor(map_polylines),
            "other_agents": torch.FloatTensor(other_agents),
            "influencer_pred": torch.FloatTensor(influencer_pred) if influencer_pred is not None else None,
            "ego_last_state": torch.FloatTensor(ego_last_state),
        }

    def _encode_map(self, map_data: Dict) -> np.ndarray:
        """
        Encode map as polylines.
        
        Returns:
            (M, P, 7) array where M is number of polylines, P is max points per polyline
            Features: [x, y, dx, dy, type, valid, speed_limit]
        """
        max_polylines = 256
        max_points = 20
        feature_dim = 7
        
        polylines = np.zeros((max_polylines, max_points, feature_dim), dtype=np.float32)
        
        polyline_idx = 0
        
        # Encode lanes
        if "lanes" in map_data:
            for lane in map_data["lanes"]:
                if polyline_idx >= max_polylines:
                    break
                    
                points = lane["polyline"][:, :2]  # Only x, y (ignore z)
                n_points = min(len(points), max_points)
                
                # Store positions
                polylines[polyline_idx, :n_points, :2] = points[:n_points]
                
                # Compute direction vectors
                if n_points > 1:
                    diffs = points[1:n_points] - points[:n_points-1]
                    polylines[polyline_idx, :n_points-1, 2:4] = diffs
                    polylines[polyline_idx, n_points-1, 2:4] = diffs[-1]  # Repeat last
                
                # Type: 1 for lane
                polylines[polyline_idx, :n_points, 4] = 1
                
                # Valid mask
                polylines[polyline_idx, :n_points, 5] = 1
                
                # Speed limit (normalized by 60 mph)
                speed_limit = lane.get("speed_limit", 30.0) / 60.0
                polylines[polyline_idx, :n_points, 6] = speed_limit
                
                polyline_idx += 1
        
        # Encode road lines
        if "road_lines" in map_data and polyline_idx < max_polylines:
            for road_line in map_data["road_lines"]:
                if polyline_idx >= max_polylines:
                    break
                    
                points = road_line["polyline"][:, :2]
                n_points = min(len(points), max_points)
                
                polylines[polyline_idx, :n_points, :2] = points[:n_points]
                
                if n_points > 1:
                    diffs = points[1:n_points] - points[:n_points-1]
                    polylines[polyline_idx, :n_points-1, 2:4] = diffs
                    polylines[polyline_idx, n_points-1, 2:4] = diffs[-1]
                
                polylines[polyline_idx, :n_points, 4] = 2  # Type: road line
                polylines[polyline_idx, :n_points, 5] = 1
                
                polyline_idx += 1
        
        # Encode road edges
        if "road_edges" in map_data and polyline_idx < max_polylines:
            for road_edge in map_data["road_edges"]:
                if polyline_idx >= max_polylines:
                    break
                    
                points = road_edge["polyline"][:, :2]
                n_points = min(len(points), max_points)
                
                polylines[polyline_idx, :n_points, :2] = points[:n_points]
                
                if n_points > 1:
                    diffs = points[1:n_points] - points[:n_points-1]
                    polylines[polyline_idx, :n_points-1, 2:4] = diffs
                    polylines[polyline_idx, n_points-1, 2:4] = diffs[-1]
                
                polylines[polyline_idx, :n_points, 4] = 3  # Type: road edge
                polylines[polyline_idx, :n_points, 5] = 1
                
                polyline_idx += 1
        
        # Encode crosswalks as polylines (perimeter)
        if "crosswalks" in map_data and polyline_idx < max_polylines:
            for crosswalk in map_data["crosswalks"]:
                if polyline_idx >= max_polylines:
                    break
                    
                # Use polygon perimeter as polyline
                points = crosswalk["polygon"][:, :2]
                n_points = min(len(points), max_points)
                
                polylines[polyline_idx, :n_points, :2] = points[:n_points]
                
                if n_points > 1:
                    # Close the polygon
                    points_closed = np.vstack([points[:n_points], points[0:1]])
                    diffs = points_closed[1:] - points_closed[:-1]
                    polylines[polyline_idx, :n_points, 2:4] = diffs
                
                polylines[polyline_idx, :n_points, 4] = 5  # Type: crosswalk
                polylines[polyline_idx, :n_points, 5] = 1
                
                polyline_idx += 1
        
        return polylines

    def _get_other_agents(
        self, tracks: Dict, exclude_ids: List[int], K: int = 5
    ) -> np.ndarray:
        """
        Get top K other agents' trajectories.
        
        Returns:
            (K, long_horizon, 2) positions
        """
        N = tracks["pos"].shape[0]
        future_start = self.history
        future_end = self.history + self.long_horizon
        
        # Get all other agents
        other_indices = [idx for idx in range(N) if idx not in exclude_ids]
        
        if len(other_indices) == 0:
            return np.zeros((K, self.long_horizon, 2), dtype=np.float32)
        
        # Compute proximity to ego (first agent in pair)
        ego_idx = exclude_ids[0]
        ego_pos = tracks["pos"][ego_idx, future_start:future_end]  # (T, 2)
        ego_valid = tracks["valid"][ego_idx, future_start:future_end]  # (T,)
        
        # Compute average distance for each other agent
        distances = []
        for idx in other_indices:
            other_pos = tracks["pos"][idx, future_start:future_end]
            other_valid = tracks["valid"][idx, future_start:future_end]
            
            # Compute distance where both are valid
            valid_mask = ego_valid & other_valid
            if valid_mask.sum() > 0:
                dists = np.linalg.norm(ego_pos - other_pos, axis=1)
                avg_dist = dists[valid_mask].mean()
            else:
                avg_dist = 1e6  # Large number if never valid together
            
            distances.append(avg_dist)
        
        # Sort by distance
        sorted_indices = np.argsort(distances)
        selected = [other_indices[i] for i in sorted_indices[:K]]
        
        # Pad if needed
        while len(selected) < K:
            selected.append(selected[0] if len(selected) > 0 else 0)
        
        # Extract trajectories
        others = np.zeros((K, self.long_horizon, 2), dtype=np.float32)
        for i, idx in enumerate(selected[:K]):
            others[i] = tracks["pos"][idx, future_start:future_end]
        
        return others

    def _get_last_state(self, tracks: Dict, agent_idx: int, timestep: int) -> np.ndarray:
        """
        Get agent state at timestep for kinematic rollout.
        
        Returns:
            (5,) array [x, y, heading, v, yaw_rate]
        """
        pos = tracks["pos"][agent_idx, timestep]  # (2,)
        heading = tracks["heading"][agent_idx, timestep]
        vel = tracks["vel"][agent_idx, timestep]  # (2,)
        speed = np.linalg.norm(vel)
        
        # Estimate yaw rate
        if timestep > 0:
            prev_heading = tracks["heading"][agent_idx, timestep - 1]
            dt = 0.1  # 10Hz
            yaw_rate = (heading - prev_heading) / dt
        else:
            yaw_rate = 0.0
        
        return np.array([pos[0], pos[1], heading, speed, yaw_rate], dtype=np.float32)

    def _augment(
        self,
        history: np.ndarray,
        future_short: np.ndarray,
        future_long: np.ndarray,
        map_polylines: np.ndarray,
        other_agents: np.ndarray,
        influencer_pred: Optional[np.ndarray],
        ego_last_state: np.ndarray,
    ) -> Tuple:
        """
        Apply data augmentation (rotation, translation).
        """
        # Random rotation [-0.2, 0.2] radians (~11 degrees)
        angle = np.random.uniform(-0.2, 0.2)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        R = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
        
        # Random translation [-2, 2] meters
        translation = np.random.uniform(-2.0, 2.0, size=2).astype(np.float32)
        
        # Apply to trajectories
        history = (history @ R.T) + translation
        future_short = (future_short @ R.T) + translation
        future_long = (future_long @ R.T) + translation
        
        # Apply to map
        map_polylines[..., :2] = (map_polylines[..., :2] @ R.T) + translation
        map_polylines[..., 2:4] = map_polylines[..., 2:4] @ R.T
        
        # Apply to other agents
        other_agents = (other_agents @ R.T) + translation
        
        # Apply to influencer pred
        if influencer_pred is not None:
            influencer_pred = (influencer_pred @ R.T) + translation
        
        # Update ego last state
        ego_last_state[:2] = (ego_last_state[:2] @ R.T) + translation
        ego_last_state[2] += angle  # Heading
        
        return (
            history,
            future_short,
            future_long,
            map_polylines,
            other_agents,
            influencer_pred,
            ego_last_state,
        )


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for batching.
    
    Handles variable-length sequences and None values.
    """
    batched = {}
    
    # Stack tensors
    for key in batch[0].keys():
        if key == "scenario_id":
            batched[key] = [sample[key] for sample in batch]
        elif key == "influencer_pred":
            # Handle None values
            preds = [sample[key] for sample in batch]
            if all(p is not None for p in preds):
                batched[key] = torch.stack(preds)
            else:
                batched[key] = None
        else:
            batched[key] = torch.stack([sample[key] for sample in batch])
    
    return batched


class WaymoTFExampleDataset(Dataset):
    """
    Dataset for loading Waymo tf_example format directly from TFRecords.
    
    This loader reads TFRecord files on-the-fly without preprocessing.
    Suitable for quick experimentation with raw Waymo data.
    
    Uses M2I's feature schema from waymo_tutorial.py.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "training",
        history_frames: int = 11,
        short_horizon_frames: int = 80,
        long_horizon_frames: int = 200,
        max_agents: int = 128,
        max_road_points: int = 20000,
        interaction_threshold: float = 30.0,
        cache_tfrecords: bool = True,
        max_scenarios: Optional[int] = None,
    ):
        """
        Args:
            data_dir: Path to tf_example TFRecord files
            split: Dataset split name
            history_frames: Number of history frames (default 11)
            short_horizon_frames: Short horizon frames (default 80)
            long_horizon_frames: Long horizon frames (default 200)
            max_agents: Maximum agents per scenario
            max_road_points: Maximum road graph points
            interaction_threshold: Distance threshold for interactions
            cache_tfrecords: Cache parsed examples in memory
            max_scenarios: Max scenarios to load (for testing)
        """
        if not HAS_TF:
            raise ImportError(
                "TensorFlow required for TFExample dataset. "
                "Install: pip install tensorflow"
            )

        self.data_dir = Path(data_dir)
        self.split = split
        self.history = history_frames
        self.short_horizon = short_horizon_frames
        self.long_horizon = long_horizon_frames
        self.max_agents = max_agents
        self.max_road_points = max_road_points
        self.interaction_threshold = interaction_threshold
        self.cache_tfrecords = cache_tfrecords

        # Load TFRecord files
        self.tfrecord_files = sorted(self.data_dir.glob("*.tfrecord*"))
        if not self.tfrecord_files:
            raise ValueError(f"No TFRecord files found in {data_dir}")

        print(f"Found {len(self.tfrecord_files)} TFRecord files")

        # Load and cache examples
        self.examples = []
        self.scenario_ids = []
        self._load_examples(max_scenarios)

        print(f"Loaded {len(self.examples)} scenarios from {split}")

    def _load_examples(self, max_scenarios: Optional[int] = None):
        """Load and cache TFRecord examples."""
        from tqdm import tqdm
        
        count = 0
        for tfrecord_file in tqdm(self.tfrecord_files,
                                   desc="Loading TFRecords"):
            dataset = tf.data.TFRecordDataset(
                str(tfrecord_file), compression_type=''
            )

            for raw_record in dataset:
                if max_scenarios and count >= max_scenarios:
                    return

                example = tf.train.Example()
                example.ParseFromString(raw_record.numpy())

                # Extract scenario ID
                features = example.features.feature
                if 'scenario/id' in features:
                    scenario_id = features['scenario/id'].bytes_list.value[
                        0].decode('utf-8')
                else:
                    scenario_id = f"scenario_{count}"

                if self.cache_tfrecords:
                    self.examples.append(example)
                else:
                    self.examples.append((tfrecord_file, count))

                self.scenario_ids.append(scenario_id)
                count += 1

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a training sample with agent data and road graph."""
        if self.cache_tfrecords:
            example = self.examples[idx]
        else:
            raise NotImplementedError("Lazy loading not yet supported")

        scenario_id = self.scenario_ids[idx]
        features = example.features.feature

        # Parse data using M2I's feature schema
        data = self._parse_features(features)

        return {
            "scenario_id": scenario_id,
            "agent_trajectories": torch.FloatTensor(
                data["trajectories"]
            ),  # (N, T, 2)
            "agent_valid": torch.BoolTensor(data["valid"]),  # (N, T)
            "agent_types": torch.LongTensor(data["types"]),  # (N,)
            "agent_velocities": torch.FloatTensor(
                data["velocities"]
            ),  # (N, T)
            "agent_headings": torch.FloatTensor(
                data["headings"]
            ),  # (N, T)
            "roadgraph_xyz": torch.FloatTensor(
                data["roadgraph_xyz"]
            ),  # (P, 3)
            "roadgraph_type": torch.LongTensor(
                data["roadgraph_type"]
            ),  # (P,)
            "roadgraph_valid": torch.BoolTensor(
                data["roadgraph_valid"]
            ),  # (P,)
            "interactive_pairs": torch.LongTensor(
                data["interactive_pairs"]
            ),  # (K, 2)
        }

    def _parse_features(self, features: Dict) -> Dict[str, np.ndarray]:
        """Parse tf_example features using M2I schema."""
        # Get number of agents
        if 'state/id' in features:
            num_agents = len(features['state/id'].float_list.value)
        else:
            num_agents = self.max_agents

        # Extract trajectories (past 10 + current 1 + future 80 = 91 frames)
        total_available = 91
        total_needed = self.history + self.long_horizon

        # Initialize arrays
        trajectories = np.zeros((num_agents, total_needed, 2),
                                dtype=np.float32)
        valid = np.zeros((num_agents, total_needed), dtype=bool)
        velocities = np.zeros((num_agents, total_needed), dtype=np.float32)
        headings = np.zeros((num_agents, total_needed), dtype=np.float32)

        # Extract past (10 frames)
        past_x = self._get_feature(features, 'state/past/x',
                                    num_agents * 10)
        past_y = self._get_feature(features, 'state/past/y',
                                    num_agents * 10)
        past_valid = self._get_feature(features, 'state/past/valid',
                                        num_agents * 10, is_int=True)
        past_yaw = self._get_feature(features, 'state/past/bbox_yaw',
                                      num_agents * 10)

        # Extract current (1 frame)
        curr_x = self._get_feature(features, 'state/current/x', num_agents)
        curr_y = self._get_feature(features, 'state/current/y', num_agents)
        curr_valid = self._get_feature(features, 'state/current/valid',
                                        num_agents, is_int=True)
        curr_yaw = self._get_feature(features, 'state/current/bbox_yaw',
                                      num_agents)

        # Extract future (80 frames)
        future_x = self._get_feature(features, 'state/future/x',
                                      num_agents * 80)
        future_y = self._get_feature(features, 'state/future/y',
                                      num_agents * 80)
        future_valid = self._get_feature(features, 'state/future/valid',
                                          num_agents * 80, is_int=True)
        future_yaw = self._get_feature(features, 'state/future/bbox_yaw',
                                        num_agents * 80)

        # Reshape and concatenate
        if len(past_x) > 0:
            past_x = np.array(past_x).reshape(num_agents, 10)
            past_y = np.array(past_y).reshape(num_agents, 10)
            past_valid = np.array(past_valid).reshape(num_agents, 10)
            past_yaw = np.array(past_yaw).reshape(num_agents, 10)

            curr_x = np.array(curr_x).reshape(num_agents, 1)
            curr_y = np.array(curr_y).reshape(num_agents, 1)
            curr_valid = np.array(curr_valid).reshape(num_agents, 1)
            curr_yaw = np.array(curr_yaw).reshape(num_agents, 1)

            future_x = np.array(future_x).reshape(num_agents, 80)
            future_y = np.array(future_y).reshape(num_agents, 80)
            future_valid = np.array(future_valid).reshape(num_agents, 80)
            future_yaw = np.array(future_yaw).reshape(num_agents, 80)

            # Concatenate: [past 10, current 1, future 80] = 91 frames
            all_x = np.concatenate([past_x, curr_x, future_x], axis=1)
            all_y = np.concatenate([past_y, curr_y, future_y], axis=1)
            all_valid = np.concatenate(
                [past_valid, curr_valid, future_valid], axis=1
            )
            all_yaw = np.concatenate([past_yaw, curr_yaw, future_yaw],
                                      axis=1)

            # Copy available data
            available = min(total_available, total_needed)
            trajectories[:, :available, 0] = all_x[:, :available]
            trajectories[:, :available, 1] = all_y[:, :available]
            valid[:, :available] = all_valid[:, :available].astype(bool)
            headings[:, :available] = all_yaw[:, :available]

            # Compute velocities from positions
            for t in range(1, available):
                dt = 0.1  # 10Hz
                dx = trajectories[:, t, 0] - trajectories[:, t - 1, 0]
                dy = trajectories[:, t, 1] - trajectories[:, t - 1, 1]
                velocities[:, t] = np.sqrt(dx**2 + dy**2) / dt

        # Extract agent types
        types = self._get_feature(features, 'state/type', num_agents)
        types = np.array(types, dtype=np.int64) if types else np.zeros(
            num_agents, dtype=np.int64)

        # Extract road graph
        roadgraph_xyz, roadgraph_type, roadgraph_valid = (
            self._extract_roadgraph(features)
        )

        # Find interactive pairs
        interactive_pairs = self._find_interactive_pairs(trajectories, valid)

        return {
            "trajectories": trajectories,
            "valid": valid,
            "velocities": velocities,
            "headings": headings,
            "types": types,
            "roadgraph_xyz": roadgraph_xyz,
            "roadgraph_type": roadgraph_type,
            "roadgraph_valid": roadgraph_valid,
            "interactive_pairs": interactive_pairs,
        }

    def _get_feature(self, features: Dict, key: str, expected_len: int,
                     is_int: bool = False) -> List:
        """Extract feature from tf.Example."""
        if key not in features:
            return []

        if is_int:
            values = list(features[key].int64_list.value)
        else:
            values = list(features[key].float_list.value)

        return values

    def _extract_roadgraph(
        self, features: Dict
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract road graph features."""
        # Get xyz coordinates (flattened)
        xyz_flat = self._get_feature(features, 'roadgraph_samples/xyz',
                                      self.max_road_points * 3)
        types_flat = self._get_feature(features, 'roadgraph_samples/type',
                                        self.max_road_points, is_int=True)
        valid_flat = self._get_feature(features, 'roadgraph_samples/valid',
                                        self.max_road_points, is_int=True)

        # Reshape
        if len(xyz_flat) > 0:
            num_points = len(xyz_flat) // 3
            xyz = np.array(xyz_flat).reshape(num_points, 3).astype(
                np.float32)
            types = np.array(types_flat[:num_points], dtype=np.int64)
            valid = np.array(valid_flat[:num_points], dtype=bool)

            # Pad or truncate to max_road_points
            if num_points < self.max_road_points:
                pad_size = self.max_road_points - num_points
                xyz = np.vstack(
                    [xyz, np.zeros((pad_size, 3), dtype=np.float32)]
                )
                types = np.concatenate(
                    [types, np.zeros(pad_size, dtype=np.int64)]
                )
                valid = np.concatenate([valid, np.zeros(pad_size, dtype=bool)])
            else:
                xyz = xyz[:self.max_road_points]
                types = types[:self.max_road_points]
                valid = valid[:self.max_road_points]
        else:
            # Empty road graph
            xyz = np.zeros((self.max_road_points, 3), dtype=np.float32)
            types = np.zeros(self.max_road_points, dtype=np.int64)
            valid = np.zeros(self.max_road_points, dtype=bool)

        return xyz, types, valid

    def _find_interactive_pairs(
        self, trajectories: np.ndarray, valid: np.ndarray
    ) -> np.ndarray:
        """Find interactive agent pairs based on proximity."""
        num_agents = trajectories.shape[0]
        pairs = []

        for i in range(num_agents):
            for j in range(i + 1, num_agents):
                both_valid = valid[i] & valid[j]
                if not np.any(both_valid):
                    continue

                valid_times = np.where(both_valid)[0]
                distances = np.linalg.norm(
                    trajectories[i, valid_times] - trajectories[
                        j, valid_times],
                    axis=1
                )

                if np.min(distances) < self.interaction_threshold:
                    pairs.append([i, j])

        if not pairs:
            return np.zeros((0, 2), dtype=np.int64)

        return np.array(pairs, dtype=np.int64)


def collate_tfexample_batch(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for TFExample batches."""
    scenario_ids = [item['scenario_id'] for item in batch]

    batch_dict = {
        'scenario_id': scenario_ids,
        'agent_trajectories': torch.stack(
            [item['agent_trajectories'] for item in batch]
        ),
        'agent_valid': torch.stack([item['agent_valid'] for item in batch]),
        'agent_types': torch.stack([item['agent_types'] for item in batch]),
        'agent_velocities': torch.stack(
            [item['agent_velocities'] for item in batch]
        ),
        'agent_headings': torch.stack(
            [item['agent_headings'] for item in batch]
        ),
        'roadgraph_xyz': torch.stack(
            [item['roadgraph_xyz'] for item in batch]
        ),
        'roadgraph_type': torch.stack(
            [item['roadgraph_type'] for item in batch]
        ),
        'roadgraph_valid': torch.stack(
            [item['roadgraph_valid'] for item in batch]
        ),
    }

    # Handle variable-length interactive pairs
    max_pairs = max(item['interactive_pairs'].shape[0] for item in batch)
    if max_pairs > 0:
        pairs_padded = []
        for item in batch:
            pairs = item['interactive_pairs']
            if pairs.shape[0] < max_pairs:
                padding = torch.full(
                    (max_pairs - pairs.shape[0], 2), -1, dtype=torch.long
                )
                pairs = torch.cat([pairs, padding], dim=0)
            pairs_padded.append(pairs)
        batch_dict['interactive_pairs'] = torch.stack(pairs_padded)
    else:
        batch_dict['interactive_pairs'] = torch.zeros(
            (len(batch), 0, 2), dtype=torch.long
        )

    return batch_dict


def build_tfexample_dataloader(
    data_dir: str,
    split: str = "training",
    batch_size: int = 16,
    num_workers: int = 4,
    shuffle: bool = True,
    **dataset_kwargs
) -> DataLoader:
    """
    Build DataLoader for tf_example data.
    
    Args:
        data_dir: Path to tf_example TFRecord files
        split: Dataset split name
        batch_size: Batch size
        num_workers: Number of workers
        shuffle: Whether to shuffle
        **dataset_kwargs: Additional dataset arguments
    
    Returns:
        DataLoader instance
    """
    dataset = WaymoTFExampleDataset(
        data_dir=data_dir,
        split=split,
        **dataset_kwargs
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_tfexample_batch,
    )


def build_dataloader(
    config: Dict,
    split: str = "train",
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    shuffle: Optional[bool] = None,
) -> DataLoader:
    """
    Build DataLoader from config.
    
    Args:
        config: Configuration dictionary
        split: 'train' or 'val'
        batch_size: Override config batch size
        num_workers: Override config num_workers
        shuffle: Override shuffle setting
    
    Returns:
        DataLoader instance
    """
    data_config = config["data"]
    train_config = config["train"]
    
    # Determine paths
    if split == "train":
        data_dir = data_config["data_dir"]
        relation_file = config["checkpoints"].get("relation_labels_train")
        augment = data_config.get("augment", True)
    else:
        data_dir = data_config.get("val_data_dir", data_config["data_dir"])
        relation_file = config["checkpoints"].get("relation_labels_val")
        augment = False
    
    # Build dataset
    dataset = WaymoITPDataset(
        data_dir=data_dir,
        relation_file=relation_file,
        split=split,
        short_horizon_frames=int(data_config["short_horizon_s"] * data_config["future_hz"]),
        long_horizon_frames=int(data_config["long_horizon_s"] * data_config["future_hz"]),
        history_frames=int(data_config["history_s"] * data_config["future_hz"]),
        augment=augment,
        max_other_agents=data_config.get("max_other_agents", 5),
    )
    
    # DataLoader settings
    batch_size = batch_size or train_config["batch_size"]
    num_workers = num_workers or train_config.get("num_workers", 4)
    shuffle = shuffle if shuffle is not None else (split == "train")
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=train_config.get("pin_memory", True),
        collate_fn=collate_fn,
        drop_last=split == "train",
    )
    
    return dataloader

