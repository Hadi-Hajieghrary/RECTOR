#!/usr/bin/env python3
"""
Waymo Open Motion Dataset Preprocessing for ITP Training

This script converts Waymo TFRecord files into preprocessed PyTorch format
optimized for Interactive Trajectory Planning (ITP) training.

Usage:
    python preprocess_waymo_for_itp.py \\
        --input-dir data/datasets/waymo_open_dataset/motion_v_1_3_0/raw/scenario/training \\
        --output-dir data/datasets/waymo_open_dataset/motion_v_1_3_0/processed/training_interactive \\
        --split training \\
        --interactive-only \\
        --num-workers 8

Features:
    - Converts TFRecord protobuf format to efficient PyTorch tensors
    - Extracts interactive pairs (vehicles that interact)
    - Preprocesses map features (lanes, crosswalks, stop signs, etc.)
    - Computes velocities, headings, and validity masks
    - Handles long-horizon scenarios (20s at 10Hz = 200 frames)
    - Saves as compressed .npz files for fast loading
"""

import argparse
import multiprocessing as mp
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tqdm import tqdm

# Suppress TensorFlow warnings
warnings.filterwarnings("ignore")
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Import Waymo protos
try:
    from waymo_open_dataset.protos import scenario_pb2
except ImportError:
    print("ERROR: waymo_open_dataset not installed!")
    print("Install with: pip install waymo-open-dataset-tf-2-11-0")
    exit(1)


class WaymoPreprocessor:
    """Preprocesses Waymo scenarios for ITP training."""
    
    # Agent types from Waymo dataset
    AGENT_TYPE_VEHICLE = 1
    AGENT_TYPE_PEDESTRIAN = 2
    AGENT_TYPE_CYCLIST = 3
    AGENT_TYPE_OTHER = 4
    
    # Map feature types
    MAP_TYPE_LANE = 1
    MAP_TYPE_ROAD_LINE = 2
    MAP_TYPE_ROAD_EDGE = 3
    MAP_TYPE_STOP_SIGN = 4
    MAP_TYPE_CROSSWALK = 5
    MAP_TYPE_SPEED_BUMP = 6
    
    def __init__(
        self,
        history_frames: int = 11,       # 1.1s history at 10Hz
        short_horizon_frames: int = 80,  # 8s short horizon
        long_horizon_frames: int = 160,  # 16s long horizon (can extend to 200)
        max_agents: int = 128,           # Max agents per scenario
        max_map_polylines: int = 256,    # Max map polylines
        max_polyline_points: int = 20,   # Max points per polyline
        interactive_only: bool = True,   # Only save interactive scenarios
        interaction_threshold: float = 30.0,  # Max distance for interaction (meters)
    ):
        """
        Args:
            history_frames: Number of history frames (default 11 = 1.1s)
            short_horizon_frames: Short horizon prediction frames (default 80 = 8s)
            long_horizon_frames: Long horizon prediction frames (default 160 = 16s)
            max_agents: Maximum number of agents to track
            max_map_polylines: Maximum map polylines to include
            max_polyline_points: Maximum points per polyline
            interactive_only: Only save scenarios with interactive agent pairs
            interaction_threshold: Distance threshold for agent interaction
        """
        self.history_frames = history_frames
        self.short_horizon = short_horizon_frames
        self.long_horizon = long_horizon_frames
        self.max_agents = max_agents
        self.max_map_polylines = max_map_polylines
        self.max_polyline_points = max_polyline_points
        self.interactive_only = interactive_only
        self.interaction_threshold = interaction_threshold
        
        self.total_frames = history_frames + long_horizon_frames
        
    def process_tfrecord(
        self, tfrecord_path: Path
    ) -> List[Dict]:
        """
        Process a single TFRecord file.
        
        Args:
            tfrecord_path: Path to TFRecord file
            
        Returns:
            List of processed scenarios
        """
        scenarios = []
        
        try:
            dataset = tf.data.TFRecordDataset(str(tfrecord_path), compression_type="")
            
            for data in dataset:
                try:
                    # Parse protobuf
                    proto_scenario = scenario_pb2.Scenario()
                    proto_scenario.ParseFromString(data.numpy())
                    
                    # Process scenario
                    processed = self._process_scenario(proto_scenario)
                    
                    if processed is not None:
                        scenarios.append(processed)
                        
                except Exception as e:
                    print(f"Error processing scenario in {tfrecord_path.name}: {e}")
                    continue
                    
        except Exception as e:
            print(f"Error reading {tfrecord_path}: {e}")
            
        return scenarios
    
    def _process_scenario(self, proto_scenario) -> Optional[Dict]:
        """
        Process a single Waymo scenario.
        
        Args:
            proto_scenario: Waymo Scenario protobuf
            
        Returns:
            Processed scenario dictionary or None if invalid
        """
        scenario_id = proto_scenario.scenario_id
        
        # Extract tracks
        tracks = self._extract_tracks(proto_scenario)
        if tracks is None:
            return None
            
        # Check if we have enough frames
        if tracks["pos"].shape[1] < self.total_frames:
            return None
        
        # Extract map features
        map_features = self._extract_map_features(proto_scenario)
        
        # Find interactive pairs
        if self.interactive_only:
            interactive_pairs = self._find_interactive_pairs(tracks)
            if len(interactive_pairs) == 0:
                return None
        else:
            # Use ego vehicle (track 0) with closest vehicle
            interactive_pairs = [(0, 1)] if tracks["pos"].shape[0] >= 2 else []
            if len(interactive_pairs) == 0:
                return None
        
        # For now, use the first interactive pair
        # In production, you might want to save all pairs
        pair_idx = interactive_pairs[0]
        
        return {
            "scenario_id": scenario_id,
            "tracks": tracks,
            "map": map_features,
            "pair_indices": pair_idx,
            "all_pairs": interactive_pairs,  # Save all pairs for reference
        }
    
    def _extract_tracks(self, proto_scenario) -> Optional[Dict]:
        """
        Extract agent tracks from scenario.
        
        Returns:
            Dictionary with:
                - agent_ids: (N,) array of agent IDs
                - valid: (N, T) bool mask
                - pos: (N, T, 2) positions in meters
                - vel: (N, T, 2) velocities in m/s
                - heading: (N, T) heading angles in radians
                - type: (N,) agent types
                - length: (N,) vehicle lengths
                - width: (N,) vehicle widths
        """
        tracks_list = proto_scenario.tracks
        num_tracks = min(len(tracks_list), self.max_agents)
        
        if num_tracks == 0:
            return None
        
        # Get number of timesteps
        num_timesteps = len(tracks_list[0].states)
        
        # Initialize arrays
        agent_ids = np.zeros(num_tracks, dtype=np.int32)
        valid = np.zeros((num_tracks, num_timesteps), dtype=bool)
        pos = np.zeros((num_tracks, num_timesteps, 2), dtype=np.float32)
        vel = np.zeros((num_tracks, num_timesteps, 2), dtype=np.float32)
        heading = np.zeros((num_tracks, num_timesteps), dtype=np.float32)
        agent_types = np.zeros(num_tracks, dtype=np.int32)
        lengths = np.zeros(num_tracks, dtype=np.float32)
        widths = np.zeros(num_tracks, dtype=np.float32)
        
        # Extract data for each track
        for i, track in enumerate(tracks_list[:num_tracks]):
            agent_ids[i] = track.id
            agent_types[i] = track.object_type
            lengths[i] = track.states[0].length if len(track.states) > 0 else 4.0
            widths[i] = track.states[0].width if len(track.states) > 0 else 2.0
            
            for t, state in enumerate(track.states):
                valid[i, t] = state.valid
                pos[i, t] = [state.center_x, state.center_y]
                vel[i, t] = [state.velocity_x, state.velocity_y]
                heading[i, t] = state.heading
        
        return {
            "agent_ids": agent_ids,
            "valid": valid,
            "pos": pos,
            "vel": vel,
            "heading": heading,
            "type": agent_types,
            "length": lengths,
            "width": widths,
        }
    
    def _extract_map_features(self, proto_scenario) -> Dict:
        """
        Extract and encode map features.
        
        Returns:
            Dictionary with:
                - lanes: List of lane polylines
                - road_lines: List of road line polylines
                - road_edges: List of road edge polylines
                - crosswalks: List of crosswalk polygons
                - stop_signs: List of stop sign positions
        """
        map_features = proto_scenario.map_features
        
        lanes = []
        road_lines = []
        road_edges = []
        crosswalks = []
        stop_signs = []
        
        for feature in map_features:
            # Extract polyline/polygon
            if feature.HasField("lane"):
                polyline = np.array([
                    [p.x, p.y, p.z] for p in feature.lane.polyline
                ], dtype=np.float32)
                
                lanes.append({
                    "id": feature.id,
                    "polyline": polyline,
                    "type": feature.lane.type,
                    "speed_limit": feature.lane.speed_limit_mph,
                })
                
            elif feature.HasField("road_line"):
                polyline = np.array([
                    [p.x, p.y, p.z] for p in feature.road_line.polyline
                ], dtype=np.float32)
                
                road_lines.append({
                    "id": feature.id,
                    "polyline": polyline,
                    "type": feature.road_line.type,
                })
                
            elif feature.HasField("road_edge"):
                polyline = np.array([
                    [p.x, p.y, p.z] for p in feature.road_edge.polyline
                ], dtype=np.float32)
                
                road_edges.append({
                    "id": feature.id,
                    "polyline": polyline,
                    "type": feature.road_edge.type,
                })
                
            elif feature.HasField("crosswalk"):
                polygon = np.array([
                    [p.x, p.y, p.z] for p in feature.crosswalk.polygon
                ], dtype=np.float32)
                
                crosswalks.append({
                    "id": feature.id,
                    "polygon": polygon,
                })
                
            elif feature.HasField("stop_sign"):
                stop_signs.append({
                    "id": feature.id,
                    "position": np.array([
                        feature.stop_sign.position.x,
                        feature.stop_sign.position.y,
                        feature.stop_sign.position.z,
                    ], dtype=np.float32),
                })
        
        return {
            "lanes": lanes,
            "road_lines": road_lines,
            "road_edges": road_edges,
            "crosswalks": crosswalks,
            "stop_signs": stop_signs,
        }
    
    def _find_interactive_pairs(self, tracks: Dict) -> List[Tuple[int, int]]:
        """
        Find pairs of agents that interact.
        
        Interaction is defined as:
        1. Both are vehicles
        2. Their trajectories come within threshold distance
        3. Both are valid for sufficient frames
        
        Returns:
            List of (agent_i, agent_j) tuples
        """
        pairs = []
        
        num_agents = tracks["pos"].shape[0]
        pos = tracks["pos"]
        valid = tracks["valid"]
        agent_types = tracks["type"]
        
        # Only consider vehicles
        vehicle_mask = agent_types == self.AGENT_TYPE_VEHICLE
        vehicle_indices = np.where(vehicle_mask)[0]
        
        if len(vehicle_indices) < 2:
            return pairs
        
        # Check each pair of vehicles
        for i in range(len(vehicle_indices)):
            for j in range(i + 1, len(vehicle_indices)):
                idx_i = vehicle_indices[i]
                idx_j = vehicle_indices[j]
                
                # Check if both have sufficient valid frames
                valid_i = valid[idx_i, :self.total_frames]
                valid_j = valid[idx_j, :self.total_frames]
                
                # Require at least 80% valid frames
                min_valid_frames = int(0.8 * self.total_frames)
                if (valid_i.sum() < min_valid_frames or 
                    valid_j.sum() < min_valid_frames):
                    continue
                
                # Check if they come close enough to interact
                pos_i = pos[idx_i, :self.total_frames]
                pos_j = pos[idx_j, :self.total_frames]
                
                # Compute distances over time
                distances = np.linalg.norm(pos_i - pos_j, axis=1)
                
                # Find minimum distance
                valid_distances = distances[valid_i & valid_j]
                if len(valid_distances) > 0:
                    min_distance = valid_distances.min()
                    
                    if min_distance < self.interaction_threshold:
                        pairs.append((idx_i, idx_j))
        
        return pairs


def process_tfrecord_worker(args):
    """Worker function for multiprocessing."""
    tfrecord_path, preprocessor = args
    return preprocessor.process_tfrecord(tfrecord_path)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess Waymo TFRecords for ITP training"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing Waymo TFRecord files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for preprocessed files"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="training",
        choices=["training", "validation", "testing", "training_interactive", "validation_interactive"],
        help="Dataset split to process"
    )
    parser.add_argument(
        "--history-frames",
        type=int,
        default=11,
        help="Number of history frames (default: 11 = 1.1s at 10Hz)"
    )
    parser.add_argument(
        "--short-horizon",
        type=int,
        default=80,
        help="Short horizon frames (default: 80 = 8s at 10Hz)"
    )
    parser.add_argument(
        "--long-horizon",
        type=int,
        default=160,
        help="Long horizon frames (default: 160 = 16s at 10Hz, max 200 = 20s)"
    )
    parser.add_argument(
        "--interactive-only",
        action="store_true",
        help="Only process scenarios with interactive agent pairs"
    )
    parser.add_argument(
        "--interaction-threshold",
        type=float,
        default=30.0,
        help="Distance threshold for agent interaction (meters)"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of parallel workers"
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maximum number of TFRecord files to process (for testing)"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize preprocessor
    preprocessor = WaymoPreprocessor(
        history_frames=args.history_frames,
        short_horizon_frames=args.short_horizon,
        long_horizon_frames=args.long_horizon,
        interactive_only=args.interactive_only,
        interaction_threshold=args.interaction_threshold,
    )
    
    # Find all TFRecord files
    input_dir = Path(args.input_dir)
    tfrecord_files = sorted(input_dir.glob("*.tfrecord*"))
    
    if args.max_files is not None:
        tfrecord_files = tfrecord_files[:args.max_files]
    
    print(f"Found {len(tfrecord_files)} TFRecord files in {input_dir}")
    print(f"Processing split: {args.split}")
    print(f"Output directory: {output_dir}")
    print(f"Interactive only: {args.interactive_only}")
    print(f"History frames: {args.history_frames}")
    print(f"Short horizon: {args.short_horizon}")
    print(f"Long horizon: {args.long_horizon}")
    print(f"Using {args.num_workers} workers")
    print()
    
    # Process files in parallel
    total_scenarios = 0
    
    if args.num_workers > 1:
        # Multiprocessing
        pool = mp.Pool(args.num_workers)
        worker_args = [(f, preprocessor) for f in tfrecord_files]
        
        results = []
        for result in tqdm(
            pool.imap_unordered(process_tfrecord_worker, worker_args),
            total=len(tfrecord_files),
            desc="Processing TFRecords"
        ):
            results.extend(result)
            
        pool.close()
        pool.join()
    else:
        # Single-threaded (easier for debugging)
        results = []
        for tfrecord_file in tqdm(tfrecord_files, desc="Processing TFRecords"):
            scenarios = preprocessor.process_tfrecord(tfrecord_file)
            results.extend(scenarios)
    
    print(f"\nExtracted {len(results)} valid scenarios")
    
    # Save scenarios
    print("Saving preprocessed scenarios...")
    
    for i, scenario in enumerate(tqdm(results, desc="Saving")):
        scenario_id = scenario["scenario_id"]
        output_file = output_dir / f"{scenario_id}.npz"
        
        # Save as compressed numpy archive
        np.savez_compressed(
            output_file,
            scenario_id=scenario_id,
            tracks=scenario["tracks"],
            map=scenario["map"],
            pair_indices=scenario["pair_indices"],
            all_pairs=scenario.get("all_pairs", []),
        )
    
    # Save metadata
    metadata = {
        "split": args.split,
        "num_scenarios": len(results),
        "history_frames": args.history_frames,
        "short_horizon_frames": args.short_horizon,
        "long_horizon_frames": args.long_horizon,
        "interactive_only": args.interactive_only,
        "interaction_threshold": args.interaction_threshold,
    }
    
    metadata_file = output_dir / "metadata.pkl"
    with open(metadata_file, "wb") as f:
        pickle.dump(metadata, f)
    
    print(f"\nPreprocessing complete!")
    print(f"Saved {len(results)} scenarios to {output_dir}")
    print(f"Metadata saved to {metadata_file}")


if __name__ == "__main__":
    main()
