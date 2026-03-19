"""
RECTOR Batched M2I Adapter

This module wraps the M2I conditional model for efficient batched inference.
Instead of running M×K serial forward passes, we batch all queries together.

Key Features:
1. Scene encoding runs ONCE per tick
2. Reactor tensor packs computed ONCE per reactor
3. Conditional inference batched across all (candidate, reactor) pairs
4. Proper coordinate transforms (world ↔ reactor-local)

Performance Target: <50ms for M=16 candidates × K=3 reactors

Phase 3 Implementation - December 2024
"""

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

# Conditional torch import
try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    TORCH_AVAILABLE = False

# Add M2I paths (computed relative to this file so the code is portable)
_WORKSPACE = Path(__file__).resolve().parents[3]
M2I_SRC = _WORKSPACE / "externals" / "M2I" / "src"
M2I_SCRIPTS = _WORKSPACE / "models" / "pretrained" / "m2i" / "scripts" / "lib"
sys.path.insert(0, str(M2I_SRC))
sys.path.insert(0, str(M2I_SCRIPTS))

from data_contracts import (
    PlanningConfig,
    EgoCandidate,
    EgoCandidateBatch,
    ReactorTensorPack,
    SceneEmbeddingCache,
    SinglePrediction,
    PredictionResult,
)

import threading


# Global lock for thread-safe args context
_ARGS_CONTEXT_LOCK = threading.Lock()


class M2IArgsContext:
    """
    Context manager to isolate M2I's global args state.

    M2I uses module-level global variables (utils.args, vectornet.args)
    that get mutated when switching between models. This context manager
    saves and restores the state to prevent contamination.

    Thread Safety:
        Uses a global lock to prevent concurrent modification of args
        in multi-threaded environments.
    """

    def __init__(self):
        self._saved_utils_args = None
        self._saved_vectornet_args = None
        self._lock_acquired = False

    def __enter__(self):
        # Acquire global lock for thread safety
        _ARGS_CONTEXT_LOCK.acquire()
        self._lock_acquired = True

        import utils
        import modeling.vectornet as vectornet_module

        # Save current state
        self._saved_utils_args = getattr(utils, "args", None)
        self._saved_vectornet_args = getattr(vectornet_module, "args", None)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        import utils
        import modeling.vectornet as vectornet_module

        # Restore previous state
        if self._saved_utils_args is not None:
            utils.args = self._saved_utils_args
        if self._saved_vectornet_args is not None:
            vectornet_module.args = self._saved_vectornet_args

        # Release lock
        if self._lock_acquired:
            _ARGS_CONTEXT_LOCK.release()
            self._lock_acquired = False

        return False

    def set_args(self, args):
        """Set M2I args within this context."""
        import utils
        import modeling.vectornet as vectornet_module

        utils.args = args
        vectornet_module.args = args


class BatchedM2IAdapter:
    """
    Adapter for M2I models with true batched inference.

    This class wraps the RecedingHorizonM2I predictor but provides:
    1. Batch-friendly interface accepting EgoCandidateBatch
    2. Scene caching to avoid redundant computation
    3. Proper coordinate transforms for reactor-centric predictions

    Usage:
        adapter = BatchedM2IAdapter(config)
        adapter.load_models()

        # Per-tick: build scene cache
        scene_cache = adapter.build_scene_cache(scenario_data, t=10)

        # Per-tick: build reactor packs (once per reactor)
        reactor_packs = adapter.build_reactor_packs(scene_cache, reactor_ids)

        # Per-tick: batched prediction (all candidates × all reactors)
        predictions = adapter.predict_batched(
            reactor_packs=reactor_packs,
            ego_candidates=candidate_batch,
            scene_cache=scene_cache,
        )
    """

    def __init__(self, config: PlanningConfig = None):
        """
        Initialize adapter.

        Args:
            config: Planning configuration (uses defaults if None)
        """
        self.config = config or PlanningConfig()

        # M2I predictor (lazy loaded)
        self._predictor = None
        self._models_loaded = False

        # Device
        self.device = self.config.device

        # Cached scene data (invalidated each tick)
        self._current_scene_cache: Optional[SceneEmbeddingCache] = None
        self._current_reactor_packs: Dict[int, ReactorTensorPack] = {}

    # =========================================================================
    # Model Loading
    # =========================================================================

    def load_models(self) -> bool:
        """
        Load all M2I models (DenseTNT, Relation, Conditional).

        Returns:
            True if all models loaded successfully
        """
        if self._models_loaded:
            return True

        try:
            from m2i_receding_horizon_full import RecedingHorizonM2I

            # RecedingHorizonM2I discovers model paths internally via MODEL_ROOT
            # It takes device + flags for which model stages to enable
            self._predictor = RecedingHorizonM2I(
                device=self.device,
                enable_relation=True,  # Enable V2V relation model
                enable_conditional=True,  # Enable conditional reactor model
            )

            # Load the base DenseTNT model (marginal predictor)
            self._predictor.load_model()

            self._models_loaded = True
            return True

        except Exception as e:
            print(f"Error loading models: {e}")
            return False

    @property
    def is_loaded(self) -> bool:
        return self._models_loaded

    # =========================================================================
    # Scene Cache Building
    # =========================================================================

    def build_scene_cache(
        self,
        scenario_data: Dict[str, Any],
        timestep: int,
    ) -> SceneEmbeddingCache:
        """
        Build scene embedding cache from parsed scenario data.

        This extracts and caches all scene information that will be
        reused across all M×K inference queries.

        Args:
            scenario_data: Parsed scenario (from TFRecord)
            timestep: Current timestep in scenario

        Returns:
            SceneEmbeddingCache with all scene data
        """
        # Extract key arrays
        trajectories = scenario_data["state/all/x"]  # [N, T]
        valid = scenario_data["state/all/valid"]  # [N, T]
        types = scenario_data["state/type"]  # [N]
        track_ids = scenario_data["state/id"]  # [N]
        objects_of_interest = scenario_data["state/objects_of_interest"]  # [N]

        # Get positions at current timestep
        N = trajectories.shape[0]
        positions = np.stack(
            [
                scenario_data["state/all/x"][:, timestep],
                scenario_data["state/all/y"][:, timestep],
            ],
            axis=1,
        )  # [N, 2]

        headings = scenario_data["state/all/yaw"][:, timestep]  # [N]

        # Build history arrays [N, T_hist, state_dim]
        T_hist = min(11, timestep + 1)
        histories = np.zeros((N, T_hist, 7), dtype=np.float32)
        for t in range(T_hist):
            t_idx = timestep - T_hist + 1 + t
            histories[:, t, 0] = scenario_data["state/all/x"][:, t_idx]
            histories[:, t, 1] = scenario_data["state/all/y"][:, t_idx]
            histories[:, t, 2] = scenario_data["state/all/yaw"][:, t_idx]
            histories[:, t, 3] = scenario_data["state/all/vx"][:, t_idx]
            histories[:, t, 4] = scenario_data["state/all/vy"][:, t_idx]
            histories[:, t, 5] = scenario_data["state/all/length"][:, t_idx]
            histories[:, t, 6] = scenario_data["state/all/width"][:, t_idx]

        # Find ego (usually index 0, but verify)
        ego_idx = 0  # Ego is first agent in Waymo format

        # Build scene cache
        cache = SceneEmbeddingCache(
            scenario_id=str(scenario_data.get("scenario/id", "unknown")),
            timestep=timestep,
            ego_position=positions[ego_idx],
            ego_heading=float(headings[ego_idx]),
            ego_history=histories[ego_idx],
            agent_ids=tuple(int(x) for x in track_ids),
            agent_positions=positions,
            agent_headings=headings,
            agent_types=types.astype(np.int32),
            agent_histories=histories,
            objects_of_interest=objects_of_interest,
            roadgraph_xyz=scenario_data["roadgraph_samples/xyz"].reshape(-1, 3),
            roadgraph_type=scenario_data["roadgraph_samples/type"].astype(np.int32),
            roadgraph_valid=scenario_data["roadgraph_samples/valid"].astype(bool),
            roadgraph_id=scenario_data["roadgraph_samples/id"].astype(np.int32),
        )

        self._current_scene_cache = cache
        return cache

    # =========================================================================
    # Reactor Pack Building
    # =========================================================================

    def build_reactor_packs(
        self,
        scene_cache: SceneEmbeddingCache,
        reactor_ids: List[int],
    ) -> Dict[int, ReactorTensorPack]:
        """
        Build tensor packs for each reactor.

        Each pack contains pre-computed data for a reactor in its
        local coordinate frame. This is computed ONCE per reactor
        and reused for all M ego candidates.

        Args:
            scene_cache: Scene data cache
            reactor_ids: List of reactor agent IDs

        Returns:
            Dict mapping reactor_id -> ReactorTensorPack
        """
        packs = {}

        for reactor_id in reactor_ids:
            try:
                # Find reactor in scene
                reactor_idx = scene_cache.agent_ids.index(reactor_id)
            except ValueError:
                print(f"Warning: reactor_id {reactor_id} not found in scene")
                continue

            # Get reactor's pose
            reactor_pos = scene_cache.agent_positions[reactor_idx].copy()
            reactor_heading = float(scene_cache.agent_headings[reactor_idx])
            reactor_type = int(scene_cache.agent_types[reactor_idx])

            # Create M2I mapping for this reactor
            # This uses the predictor's existing mapping creation logic
            if self._predictor is not None:
                mapping = self._create_reactor_mapping(
                    scene_cache,
                    reactor_idx,
                    reactor_pos,
                    reactor_heading,
                )
            else:
                # Minimal mapping for testing
                mapping = {
                    "matrix": np.zeros((10, 128), dtype=np.float32),
                    "polyline_spans": [],
                    "map_start_polyline_idx": 0,
                }

            # Build tensor pack
            pack = ReactorTensorPack(
                reactor_id=reactor_id,
                position_world=reactor_pos,
                heading_world=reactor_heading,
                origin=reactor_pos,
                rotation=reactor_heading,
                reactor_history_local=scene_cache.agent_histories[reactor_idx],
                reactor_type=reactor_type,
                matrix=mapping.get("matrix", np.zeros((10, 128))),
                polyline_spans=mapping.get("polyline_spans", []),
                map_start_polyline_idx=mapping.get("map_start_polyline_idx", 0),
                goals_2D=mapping.get("goals_2D", np.zeros((100, 2))),
            )

            packs[reactor_id] = pack

        self._current_reactor_packs = packs
        return packs

    def _create_reactor_mapping(
        self,
        scene_cache: SceneEmbeddingCache,
        reactor_idx: int,
        reactor_pos: np.ndarray,
        reactor_heading: float,
    ) -> Dict[str, Any]:
        """
        Create M2I mapping for a reactor using the predictor's logic.

        This wraps the predictor's create_mapping_for_agent method.
        """
        # Build scenario dict for predictor
        scenario_data = {
            "state/all/x": np.zeros((len(scene_cache.agent_ids), 91)),
            "state/all/y": np.zeros((len(scene_cache.agent_ids), 91)),
            "state/all/yaw": np.zeros((len(scene_cache.agent_ids), 91)),
            "state/all/vx": np.zeros((len(scene_cache.agent_ids), 91)),
            "state/all/vy": np.zeros((len(scene_cache.agent_ids), 91)),
            "state/all/valid": np.zeros((len(scene_cache.agent_ids), 91), dtype=bool),
            "state/all/length": np.ones((len(scene_cache.agent_ids), 91)) * 4.5,
            "state/all/width": np.ones((len(scene_cache.agent_ids), 91)) * 2.0,
            "state/type": scene_cache.agent_types,
            "roadgraph_samples/xyz": scene_cache.roadgraph_xyz.flatten(),
            "roadgraph_samples/type": scene_cache.roadgraph_type,
            "roadgraph_samples/valid": scene_cache.roadgraph_valid,
            "roadgraph_samples/id": scene_cache.roadgraph_id,
        }

        # Fill trajectory data
        T_hist = scene_cache.agent_histories.shape[1]
        for t in range(T_hist):
            t_idx = scene_cache.timestep - T_hist + 1 + t
            scenario_data["state/all/x"][:, t_idx] = scene_cache.agent_histories[
                :, t, 0
            ]
            scenario_data["state/all/y"][:, t_idx] = scene_cache.agent_histories[
                :, t, 1
            ]
            scenario_data["state/all/yaw"][:, t_idx] = scene_cache.agent_histories[
                :, t, 2
            ]
            scenario_data["state/all/vx"][:, t_idx] = scene_cache.agent_histories[
                :, t, 3
            ]
            scenario_data["state/all/vy"][:, t_idx] = scene_cache.agent_histories[
                :, t, 4
            ]
            scenario_data["state/all/valid"][:, t_idx] = True

        # Use predictor's mapping creation
        try:
            mapping = self._predictor.create_mapping_for_agent(
                scenario_data,
                agent_idx=reactor_idx,
                current_t=scene_cache.timestep,
            )
            return mapping
        except Exception as e:
            print(f"Warning: Failed to create reactor mapping: {e}")
            return {
                "matrix": np.zeros((10, 128), dtype=np.float32),
                "polyline_spans": [],
                "map_start_polyline_idx": 0,
            }

    # =========================================================================
    # Batched Prediction
    # =========================================================================

    def predict_batched(
        self,
        reactor_packs: Dict[int, ReactorTensorPack],
        ego_candidates: EgoCandidateBatch,
        scene_cache: SceneEmbeddingCache,
    ) -> PredictionResult:
        """
        Run TRUE batched conditional prediction.

        Instead of M×K nested loops with separate forward passes, this method:
        1. Prepares all B=M×K mappings with injected influencer trajectories
        2. Calls model.forward() ONCE with the full batch
        3. Unpacks results back to [M, K, ...] structure

        Performance target: <50ms for M=16, K=3 (B=48)

        Args:
            reactor_packs: Pre-computed reactor tensor packs
            ego_candidates: Batch of ego trajectory candidates
            scene_cache: Cached scene data

        Returns:
            PredictionResult with shape [M, K, N_modes, H, 2]
        """
        M = ego_candidates.num_candidates
        K = len(reactor_packs)
        reactor_ids = list(reactor_packs.keys())
        N_modes = self.config.num_prediction_modes
        H = self.config.candidate_horizon

        # Initialize output arrays
        all_trajectories = np.zeros((M, K, N_modes, H, 2), dtype=np.float32)
        all_scores = np.zeros((M, K, N_modes), dtype=np.float32)

        # If no predictor, use dummy predictions (for testing)
        if self._predictor is None:
            return self._predict_batched_dummy(
                reactor_packs,
                ego_candidates,
                reactor_ids,
                M,
                K,
                N_modes,
                H,
                all_trajectories,
                all_scores,
            )

        # TRUE BATCHED INFERENCE:
        # Step 1: Prepare all B=M×K mappings
        batch_mappings = []
        batch_index_map = []  # [(m, k)] to track which prediction goes where

        for m in range(M):
            candidate = ego_candidates.candidates[m]
            for k, reactor_id in enumerate(reactor_ids):
                pack = reactor_packs[reactor_id]

                # Create mapping with injected influencer trajectory
                mapping = self._prepare_conditional_mapping(
                    pack, candidate.trajectory, scene_cache
                )

                if mapping is not None:
                    batch_mappings.append(mapping)
                    batch_index_map.append((m, k))

        if len(batch_mappings) == 0:
            return PredictionResult(
                trajectories=all_trajectories,
                scores=all_scores,
                ego_candidate_ids=tuple(
                    c.candidate_id for c in ego_candidates.candidates
                ),
                reactor_ids=tuple(reactor_ids),
            )

        # Step 2: Run SINGLE batched forward pass
        try:
            pred_traj, pred_scores = self._run_batched_conditional_inference(
                batch_mappings
            )
        except Exception as e:
            print(f"Batched inference error: {e}")
            # Fall back to serial execution
            return self._predict_batched_serial(
                reactor_packs,
                ego_candidates,
                scene_cache,
                reactor_ids,
                M,
                K,
                N_modes,
                H,
            )

        # Step 3: Unpack results to [M, K, ...] structure
        if pred_traj is not None:
            for idx, (m, k) in enumerate(batch_index_map):
                if idx < len(pred_traj):
                    pack = reactor_packs[reactor_ids[k]]
                    traj_modes = pred_traj[idx]  # [N_modes, H, 2]
                    scores = (
                        pred_scores[idx]
                        if pred_scores is not None
                        else np.ones(N_modes) / N_modes
                    )

                    # Transform predictions to world frame
                    for mode in range(min(N_modes, traj_modes.shape[0])):
                        pred_world = self._transform_trajectory_to_world(
                            traj_modes[mode],
                            pack.origin,
                            pack.rotation,
                        )
                        all_trajectories[m, k, mode] = pred_world
                        all_scores[m, k, mode] = scores[mode]

        return PredictionResult(
            trajectories=all_trajectories,
            scores=all_scores,
            ego_candidate_ids=tuple(c.candidate_id for c in ego_candidates.candidates),
            reactor_ids=tuple(reactor_ids),
        )

    def _predict_batched_dummy(
        self,
        reactor_packs: Dict[int, ReactorTensorPack],
        ego_candidates: EgoCandidateBatch,
        reactor_ids: List[int],
        M: int,
        K: int,
        N_modes: int,
        H: int,
        all_trajectories: np.ndarray,
        all_scores: np.ndarray,
    ) -> PredictionResult:
        """Return dummy predictions for testing without loaded models."""
        for m in range(M):
            for k, reactor_id in enumerate(reactor_ids):
                pack = reactor_packs[reactor_id]
                # Dummy: reactor stays at position
                for mode in range(N_modes):
                    all_trajectories[m, k, mode] = np.tile(pack.position_world, (H, 1))
                    all_scores[m, k, mode] = 1.0 / N_modes

        return PredictionResult(
            trajectories=all_trajectories,
            scores=all_scores,
            ego_candidate_ids=tuple(c.candidate_id for c in ego_candidates.candidates),
            reactor_ids=tuple(reactor_ids),
        )

    def _predict_batched_serial(
        self,
        reactor_packs: Dict[int, ReactorTensorPack],
        ego_candidates: EgoCandidateBatch,
        scene_cache: SceneEmbeddingCache,
        reactor_ids: List[int],
        M: int,
        K: int,
        N_modes: int,
        H: int,
    ) -> PredictionResult:
        """Fallback serial prediction (legacy M×K loop)."""
        all_trajectories = np.zeros((M, K, N_modes, H, 2), dtype=np.float32)
        all_scores = np.zeros((M, K, N_modes), dtype=np.float32)

        for k, reactor_id in enumerate(reactor_ids):
            pack = reactor_packs[reactor_id]
            for m in range(M):
                candidate = ego_candidates.candidates[m]
                ego_traj_local = self._transform_trajectory_to_local(
                    candidate.trajectory, pack.origin, pack.rotation
                )
                pred_traj, pred_scores = self._run_conditional_inference(
                    pack, ego_traj_local, scene_cache
                )
                if pred_traj is not None:
                    for mode in range(min(N_modes, pred_traj.shape[0])):
                        pred_world = self._transform_trajectory_to_world(
                            pred_traj[mode], pack.origin, pack.rotation
                        )
                        all_trajectories[m, k, mode] = pred_world
                        all_scores[m, k, mode] = (
                            pred_scores[mode]
                            if pred_scores is not None
                            else 1.0 / N_modes
                        )

        return PredictionResult(
            trajectories=all_trajectories,
            scores=all_scores,
            ego_candidate_ids=tuple(c.candidate_id for c in ego_candidates.candidates),
            reactor_ids=tuple(reactor_ids),
        )

    def _prepare_conditional_mapping(
        self,
        reactor_pack: ReactorTensorPack,
        ego_trajectory: np.ndarray,
        scene_cache: SceneEmbeddingCache,
    ) -> Optional[Dict[str, Any]]:
        """
        Prepare a single mapping with injected influencer trajectory.

        This creates the mapping dict that the VectorNet model expects,
        with the ego trajectory encoded as the influencer.
        """
        try:
            # Build base reactor mapping from pack
            mapping = {
                "matrix": reactor_pack.matrix.copy(),
                "polyline_spans": list(reactor_pack.polyline_spans),
                "map_start_polyline_idx": reactor_pack.map_start_polyline_idx,
                "goals_2D": (
                    reactor_pack.goals_2D.copy()
                    if reactor_pack.goals_2D is not None
                    else np.zeros((100, 2))
                ),
                "cent_x": float(reactor_pack.origin[0]),
                "cent_y": float(reactor_pack.origin[1]),
                "angle": float(reactor_pack.rotation),
                "scenario_id": scene_cache.scenario_id,
                "object_id": reactor_pack.reactor_id,
            }

            # Add BEV image if available
            if scene_cache.bev_image is not None:
                mapping["image"] = scene_cache.bev_image.copy()

            # Inject ego trajectory as influencer
            # Transform to reactor-local frame for the model
            ego_traj_local = self._transform_trajectory_to_local(
                ego_trajectory,
                reactor_pack.origin,
                reactor_pack.rotation,
            )

            # Store influencer trajectory in mapping
            # The predictor will use this for conditional prediction
            mapping["influencer_traj"] = ego_traj_local  # [H, 2]
            mapping["influencer_traj_world"] = ego_trajectory  # Keep world coords too

            return mapping

        except Exception as e:
            print(f"Error preparing mapping: {e}")
            return None

    def _run_batched_conditional_inference(
        self,
        batch_mappings: List[Dict[str, Any]],
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Run single batched forward pass for all M×K mappings.

        This is the key performance optimization - instead of M×K
        separate forward passes, we run ONE pass with B=M×K batch size.

        Args:
            batch_mappings: List of B=M×K prepared mappings

        Returns:
            pred_traj: [B, N_modes, H, 2] predictions
            pred_scores: [B, N_modes] mode probabilities
        """
        if len(batch_mappings) == 0:
            return None, None

        # Ensure conditional model is loaded
        if self._predictor.conditional_model is None:
            print("  Loading conditional model for batched inference...")
            self._predictor._load_conditional_model()
            if self._predictor.conditional_model is None:
                return None, None

        # Prepare conditional mappings with rasterized influencer trajectories
        conditional_mappings = []
        for mapping in batch_mappings:
            # Get influencer trajectory from mapping
            influencer_traj = mapping.get("influencer_traj_world")
            if influencer_traj is None:
                influencer_traj = mapping.get("influencer_traj")

            if influencer_traj is not None:
                # Wrap in batch dimension if needed
                if influencer_traj.ndim == 2:
                    influencer_traj = influencer_traj[np.newaxis, ...]

                # Use predictor's method to create conditional mapping with rasterization
                cond_mapping = self._predictor._create_conditional_mapping(
                    reactor_mapping=mapping,
                    influencer_trajectory=influencer_traj,
                    influencer_scores=np.array([1.0]),
                )
                conditional_mappings.append(cond_mapping)
            else:
                conditional_mappings.append(mapping)

        # Run SINGLE batched forward pass
        with M2IArgsContext() as ctx:
            import utils
            import modeling.vectornet as vectornet_module

            # Set conditional model args
            ctx.set_args(self._predictor.conditional_args)
            vectornet_module.args = self._predictor.conditional_args

            with torch.no_grad():
                pred_traj, pred_score, _ = self._predictor.conditional_model(
                    conditional_mappings, self.device
                )

            # Convert to numpy arrays
            if pred_traj is not None:
                pred_traj = np.array(pred_traj)  # [B, N_modes, H, 2]
            if pred_score is not None:
                pred_score = np.array(pred_score)  # [B, N_modes]

            return pred_traj, pred_score

    def _run_conditional_inference(
        self,
        reactor_pack: ReactorTensorPack,
        ego_trajectory_local: np.ndarray,
        scene_cache: SceneEmbeddingCache,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Run single conditional inference.

        This wraps the predictor's run_conditional_inference method.
        """
        if self._predictor is None:
            # Return dummy predictions for testing
            N_modes = self.config.num_prediction_modes
            H = self.config.candidate_horizon
            dummy_traj = np.tile(reactor_pack.position_world, (N_modes, H, 1))
            dummy_scores = np.ones(N_modes) / N_modes
            return dummy_traj, dummy_scores

        try:
            # Build reactor mapping from pack
            reactor_mapping = {
                "matrix": reactor_pack.matrix,
                "polyline_spans": reactor_pack.polyline_spans,
                "map_start_polyline_idx": reactor_pack.map_start_polyline_idx,
                "goals_2D": reactor_pack.goals_2D,
                "cent_x": float(reactor_pack.origin[0]),
                "cent_y": float(reactor_pack.origin[1]),
                "angle": float(reactor_pack.rotation),
            }

            # Add image if available
            if scene_cache.bev_image is not None:
                reactor_mapping["image"] = scene_cache.bev_image

            # Ego trajectory as influencer (in reactor-local frame, but we need world)
            # Transform back to world for the predictor
            ego_traj_world = self._transform_trajectory_to_world(
                ego_trajectory_local,
                reactor_pack.origin,
                reactor_pack.rotation,
            )

            # Run conditional inference
            # The predictor's method expects [K, H, 2] for influencer trajectory
            influencer_traj = ego_traj_world[np.newaxis, ...]  # [1, H, 2]
            influencer_scores = np.array([1.0])

            pred_traj, pred_scores = self._predictor.run_conditional_inference(
                reactor_mapping=reactor_mapping,
                influencer_trajectory=influencer_traj,
                influencer_scores=influencer_scores,
            )

            return pred_traj, pred_scores

        except Exception as e:
            print(f"Conditional inference error: {e}")
            return None, None

    # =========================================================================
    # Coordinate Transform Utilities
    # =========================================================================

    def _transform_trajectory_to_local(
        self,
        trajectory: np.ndarray,
        origin: np.ndarray,
        rotation: float,
    ) -> np.ndarray:
        """
        Transform trajectory from world to reactor-local frame.

        Args:
            trajectory: [H, 2] in world coordinates
            origin: [2] reactor position
            rotation: Reactor heading (radians)

        Returns:
            [H, 2] in reactor-local coordinates
        """
        # Translate to origin
        translated = trajectory - origin

        # Rotate to local frame (negative rotation to go world→local)
        cos_a = math.cos(-rotation)
        sin_a = math.sin(-rotation)

        x_local = translated[:, 0] * cos_a - translated[:, 1] * sin_a
        y_local = translated[:, 0] * sin_a + translated[:, 1] * cos_a

        return np.stack([x_local, y_local], axis=1)

    def _transform_trajectory_to_world(
        self,
        trajectory: np.ndarray,
        origin: np.ndarray,
        rotation: float,
    ) -> np.ndarray:
        """
        Transform trajectory from reactor-local to world frame.

        Args:
            trajectory: [H, 2] in reactor-local coordinates
            origin: [2] reactor position
            rotation: Reactor heading (radians)

        Returns:
            [H, 2] in world coordinates
        """
        # Rotate to world frame (positive rotation to go local→world)
        cos_a = math.cos(rotation)
        sin_a = math.sin(rotation)

        x_world = trajectory[:, 0] * cos_a - trajectory[:, 1] * sin_a
        y_world = trajectory[:, 0] * sin_a + trajectory[:, 1] * cos_a

        # Translate to world origin
        result = np.stack([x_world + origin[0], y_world + origin[1]], axis=1)

        return result

    # =========================================================================
    # Convenience Methods
    # =========================================================================

    def predict_single(
        self,
        ego_trajectory: np.ndarray,
        reactor_id: int,
    ) -> SinglePrediction:
        """
        Predict reactor response to a single ego trajectory.

        Convenience method for single (candidate, reactor) pair.

        Args:
            ego_trajectory: [H, 2] ego trajectory in world frame
            reactor_id: Target reactor agent ID

        Returns:
            SinglePrediction with reactor's predicted trajectory
        """
        # Create single-element batch
        candidate = EgoCandidate(
            candidate_id=0,
            trajectory=ego_trajectory.astype(np.float32),
        )
        batch = EgoCandidateBatch.from_candidates([candidate])

        # Check if reactor pack exists
        if reactor_id not in self._current_reactor_packs:
            raise ValueError(f"Reactor {reactor_id} not in current packs")

        reactor_packs = {reactor_id: self._current_reactor_packs[reactor_id]}

        # Run prediction
        result = self.predict_batched(
            reactor_packs=reactor_packs,
            ego_candidates=batch,
            scene_cache=self._current_scene_cache,
        )

        return result.get_prediction(0, 0)

    def select_reactors(
        self,
        scene_cache: SceneEmbeddingCache,
        ego_position: np.ndarray,
        max_reactors: int = None,
    ) -> List[int]:
        """
        Select relevant reactors from scene.

        Currently uses simple distance-based selection.
        Future: Use corridor overlap + reachable sets.

        Args:
            scene_cache: Scene data
            ego_position: Ego's current position [2]
            max_reactors: Maximum number to select (default: config value)

        Returns:
            List of reactor agent IDs
        """
        max_k = max_reactors or self.config.max_reactors

        # Get interactive agents
        interactive_ids = scene_cache.get_interactive_agent_ids()

        if len(interactive_ids) == 0:
            return []

        # Sort by distance to ego
        distances = []
        for agent_id in interactive_ids:
            idx = scene_cache.agent_ids.index(agent_id)
            agent_pos = scene_cache.agent_positions[idx]
            dist = np.linalg.norm(agent_pos - ego_position)
            distances.append((agent_id, dist))

        distances.sort(key=lambda x: x[1])

        # Take closest K
        return [agent_id for agent_id, _ in distances[:max_k]]


def create_adapter(config: PlanningConfig = None) -> BatchedM2IAdapter:
    """
    Factory function to create a BatchedM2IAdapter.

    Args:
        config: Optional planning configuration

    Returns:
        Initialized (but not loaded) adapter
    """
    return BatchedM2IAdapter(config=config)
