#!/usr/bin/env python3
"""
M2I DenseTNT Trajectory Generator for RECTOR.

Uses the full pretrained DenseTNT model to generate 6-mode trajectory
predictions, which are then scored by RECTOR's tiered rule scorer.

This replaces RECTOR's CVAE trajectory head with M2I's proven trajectory
generation pipeline while keeping RECTOR's rulebook evaluation.

Architecture:
    1. Parse Waymo Scenario proto into M2I-compatible format
    2. Render BEV raster + create polyline vectors
    3. Run DenseTNT VectorNet forward pass
    4. Output: 6 multi-modal trajectories [6, 80, 2] + scores [6]

Key Dependencies:
    - M2I VectorNet + DenseTNT decoder (model.24.bin)
    - M2I utils_cython for agent/road vector creation
    - Waymo Open Dataset protos for data parsing

Usage:
    generator = M2ITrajectoryGenerator(device='cuda')
    generator.load_model()
    pred_traj, scores = generator.predict_from_scenario(scenario_proto)
"""

import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

# Add M2I source to path
M2I_SRC = Path("/workspace/externals/M2I/src")
sys.path.insert(0, str(M2I_SRC))

# Suppress TF warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings

warnings.filterwarnings("ignore")

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False


class OtherParams(list):
    """
    Hybrid list/dict for M2I's other_params.

    M2I codebase uses other_params inconsistently — sometimes as a list
    (for 'in' checks) and sometimes as a dict (for .get() calls).
    """

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


class M2ITrajectoryGenerator:
    """
    Full DenseTNT trajectory generator using pretrained M2I model.

    Produces 6-mode trajectory predictions for the ego (SDC) agent,
    ready to be scored by RECTOR's tiered rule scorer.
    """

    # Default model path
    DEFAULT_MODEL_PATH = Path(
        "/workspace/models/pretrained/m2i/models/densetnt/model.24.bin"
    )

    def __init__(
        self,
        model_path: str = None,
        device: str = "cuda",
        future_frames: int = 80,
        mode_num: int = 6,
        hidden_size: int = 128,
    ):
        self.model_path = Path(model_path) if model_path else self.DEFAULT_MODEL_PATH
        self.device_str = device
        self.device = None
        self.future_frames = future_frames
        self.mode_num = mode_num
        self.hidden_size = hidden_size

        self.model = None
        self.args = None
        self._is_loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    # -----------------------------------------------------------------
    # Model Loading
    # -----------------------------------------------------------------

    def _build_args(self):
        """Build args for DenseTNT model matching pretrained config."""
        import utils

        args = utils.Args()

        # Core model parameters
        args.hidden_size = self.hidden_size
        args.future_frame_num = self.future_frames
        args.future_test_frame_num = self.future_frames
        args.mode_num = self.mode_num
        args.sub_graph_batch_size = 4096
        args.train_batch_size = 1
        args.eval_batch_size = 1
        args.core_num = 1
        args.infMLP = 0
        args.nms_threshold = 7.2
        args.agent_type = "vehicle"
        args.inter_agent_types = None
        args.single_agent = True

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
        args.use_centerline = False
        args.autoregression = None
        args.multi = None
        args.placeholder = 0.0
        args.method_span = [0, 1]
        args.train_extra = False
        args.not_use_api = False
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
        args.reuse_temp_file = False
        args.add_prefix = None
        args.eval_params = []
        args.debug_mode = False
        args.debug = False

        # Eval mode
        args.waymo = True
        args.nuscenes = False
        args.argoverse = False
        args.do_train = False
        args.do_eval = True
        args.do_test = False

        # BEV raster image placeholder (populated per-sample)
        args.image = np.zeros([224, 224, 60], dtype=np.int8)

        # Output dirs for logging
        output_dir = Path("/workspace/output/m2i_rector")
        output_dir.mkdir(parents=True, exist_ok=True)
        args.log_dir = str(output_dir / "logs")
        args.output_dir = str(output_dir)
        args.temp_file_dir = str(output_dir / "temp")
        os.makedirs(args.log_dir, exist_ok=True)
        os.makedirs(args.temp_file_dir, exist_ok=True)

        # DenseTNT-specific other_params
        args.other_params = OtherParams(
            [
                "l1_loss",
                "densetnt",
                "goals_2D",
                "enhance_global_graph",
                "laneGCN",
                "point_sub_graph",
                "laneGCN-4",
                "stride_10_2",
                "raster",
            ],
            {"eval_time": 80},
        )

        return args

    def load_model(self):
        """Load pretrained DenseTNT model."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required.")

        # Ensure M2I/src is at front of sys.path before importing utils,
        # because models/RECTOR/scripts/utils/ package can shadow M2I's utils.py
        m2i_src = str(M2I_SRC)
        if sys.path[0] != m2i_src:
            while m2i_src in sys.path:
                sys.path.remove(m2i_src)
            sys.path.insert(0, m2i_src)
        # Remove stale cache if the wrong utils was already imported
        if "utils" in sys.modules:
            cached = getattr(sys.modules["utils"], "__file__", "") or ""
            if "M2I" not in cached:
                del sys.modules["utils"]

        import utils
        from modeling.vectornet import VectorNet

        self.device = torch.device(
            self.device_str if torch.cuda.is_available() else "cpu"
        )
        print(f"[M2ITrajectoryGenerator] Device: {self.device}")

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        self.args = self._build_args()
        utils.args = self.args

        self.model = VectorNet(self.args)

        checkpoint = torch.load(self.model_path, map_location="cpu", weights_only=False)
        model_dict = self.model.state_dict()
        loaded = 0
        for key, value in checkpoint.items():
            clean = key.replace("module.", "") if key.startswith("module.") else key
            if clean in model_dict and model_dict[clean].shape == value.shape:
                model_dict[clean] = value
                loaded += 1
        self.model.load_state_dict(model_dict)
        self.model.to(self.device)
        self.model.eval()

        print(f"  Loaded {loaded}/{len(checkpoint)} DenseTNT weights")
        self._is_loaded = True

    # -----------------------------------------------------------------
    # Data Conversion: Waymo Scenario → M2I mapping
    # -----------------------------------------------------------------

    @staticmethod
    def _render_bev_raster(
        roadgraph_xyz: np.ndarray,
        roadgraph_type: np.ndarray,
        roadgraph_valid: np.ndarray,
        normalizer,
    ) -> np.ndarray:
        """Render BEV raster image (road channels 40-59)."""
        image = np.zeros([224, 224, 60], dtype=np.int8)

        cent_x = normalizer.x
        cent_y = normalizer.y
        angle = normalizer.yaw
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)

        for i in range(len(roadgraph_xyz)):
            if not roadgraph_valid[i]:
                continue
            xw, yw = roadgraph_xyz[i, 0], roadgraph_xyz[i, 1]
            dx, dy = xw - cent_x, yw - cent_y
            xl = dx * cos_a - dy * sin_a
            yl = dx * sin_a + dy * cos_a
            if math.sqrt(xl**2 + (yl - 30) ** 2) > 80.0:
                continue
            xp = int(xl + 112 + 0.5)
            yp = int(yl + 56 + 0.5)
            if 0 <= xp < 224 and 0 <= yp < 224:
                lt = int(roadgraph_type[i])
                if 0 <= lt < 20:
                    image[xp, yp, 40 + lt] = 1
        return image

    @staticmethod
    def _create_road_vectors(
        roadgraph_xyz: np.ndarray,
        roadgraph_type: np.ndarray,
        roadgraph_valid: np.ndarray,
        roadgraph_id: np.ndarray,
        normalizer,
    ) -> Tuple[np.ndarray, list]:
        """Create 128-dim road polyline vectors."""
        cent_x = normalizer.x
        cent_y = normalizer.y
        angle = normalizer.yaw
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        scale = 0.03

        lane_points: Dict[int, list] = {}
        for i in range(len(roadgraph_xyz)):
            if not roadgraph_valid[i]:
                continue
            xw, yw = roadgraph_xyz[i, 0], roadgraph_xyz[i, 1]
            dx, dy = xw - cent_x, yw - cent_y
            xl = dx * cos_a - dy * sin_a
            yl = dx * sin_a + dy * cos_a
            if math.sqrt(xl**2 + (yl - 30) ** 2) > 80.0:
                continue
            lid = int(roadgraph_id[i])
            lt = int(roadgraph_type[i])
            lane_points.setdefault(lid, []).append((xl, yl, lt))

        vectors_list = []
        polyline_spans = []
        stride = 10
        for lid, pts in lane_points.items():
            if len(pts) < 2:
                continue
            start = len(vectors_list)
            for c in range(0, len(pts) - 1, stride):
                p1, p2 = pts[c], pts[min(c + stride, len(pts) - 1)]
                vec = np.zeros(128, dtype=np.float32)
                vec[0] = p1[0] * scale
                vec[1] = p1[1] * scale
                vec[20] = p2[0] * scale
                vec[21] = p2[1] * scale
                lt = p1[2]
                if 0 <= lt < 20:
                    vec[50 + lt] = 1
                vectors_list.append(vec)
            if len(vectors_list) > start:
                polyline_spans.append(slice(start, len(vectors_list)))

        if not vectors_list:
            return np.zeros((0, 128), dtype=np.float32), []
        return np.array(vectors_list, dtype=np.float32), polyline_spans

    @staticmethod
    def _generate_raster_goals_grid() -> np.ndarray:
        """Generate 224×224 pixel grid as candidate goals."""
        goals = []
        for i in range(224):
            xf = float(i - 112)
            for j in range(224):
                yf = float(j - 56)
                goals.append([xf, yf])
        return np.array(goals, dtype=np.float32)

    def scenario_to_mapping(
        self,
        scenario,
        *,
        target_agent_idx: int = 0,
        current_time_index: Optional[int] = None,
    ) -> Optional[Dict]:
        """
        Convert a Waymo Scenario protobuf into an M2I mapping dict.

        Args:
            scenario: waymo Scenario proto (parsed by scenario_pb2)
            target_agent_idx: 0 = SDC (self-driving car)
            current_time_index: Override for current time (default: scenario.current_time_index)

        Returns:
            M2I mapping dict ready for VectorNet.forward(), or None on failure.
        """
        import utils
        import utils_cython
        from utils_cython import get_normalized

        Normalizer = utils.Normalizer

        tracks = list(scenario.tracks)
        if not tracks:
            return None

        current_ts = (
            current_time_index
            if current_time_index is not None
            else scenario.current_time_index
        )
        n_history = 11
        n_future = 80
        n_agents = len(tracks)

        # Determine SDC index
        # NOTE: scenario.sdc_track_index is an array index into tracks[], NOT a track ID.
        sdc_idx = scenario.sdc_track_index
        if sdc_idx < 0 or sdc_idx >= n_agents:
            sdc_idx = 0

        # Select target: 0 = SDC
        if target_agent_idx == 0:
            agent_idx = sdc_idx
        else:
            agent_idx = min(target_agent_idx, n_agents - 1)

        # Build full gt_trajectory [n_agents, 91, 7]
        # columns: x, y, length, width, yaw, vx, vy
        gt_trajectory = np.zeros((n_agents, n_history + n_future, 7), dtype=np.float32)
        gt_is_valid = np.zeros((n_agents, n_history + n_future), dtype=np.float32)
        tracks_type = np.zeros(n_agents, dtype=np.int32)

        for i, track in enumerate(tracks):
            tracks_type[i] = track.object_type
            for t_offset in range(n_history + n_future):
                ts = (current_ts - 10) + t_offset
                if 0 <= ts < len(track.states):
                    s = track.states[ts]
                    if s.valid:
                        gt_trajectory[i, t_offset, 0] = s.center_x
                        gt_trajectory[i, t_offset, 1] = s.center_y
                        gt_trajectory[i, t_offset, 2] = s.length
                        gt_trajectory[i, t_offset, 3] = s.width
                        gt_trajectory[i, t_offset, 4] = s.heading
                        gt_trajectory[i, t_offset, 5] = s.velocity_x
                        gt_trajectory[i, t_offset, 6] = s.velocity_y
                        gt_is_valid[i, t_offset] = 1.0

        # Swap target to index 0
        if agent_idx != 0:
            gt_trajectory[[0, agent_idx]] = gt_trajectory[[agent_idx, 0]]
            gt_is_valid[[0, agent_idx]] = gt_is_valid[[agent_idx, 0]]
            tracks_type[[0, agent_idx]] = tracks_type[[agent_idx, 0]]

        # Check target is valid at current time
        if gt_is_valid[0, n_history - 1] < 0.5:
            return None

        # Normalizer: center on target at current time, rotate to face +Y
        cent_x = gt_trajectory[0, n_history - 1, 0]
        cent_y = gt_trajectory[0, n_history - 1, 1]
        waymo_yaw = float(gt_trajectory[0, n_history - 1, 4])
        angle = -waymo_yaw + math.radians(90)
        normalizer = Normalizer(cent_x, cent_y, angle)

        # Normalize all positions
        gt_trajectory[:, :, :2] = get_normalized(gt_trajectory[:, :, :2], normalizer)

        # Build road graph arrays from scenario map_features
        road_xyz_list, road_type_list, road_valid_list, road_id_list = [], [], [], []
        for mf in scenario.map_features:
            lane_data = None
            feature_type = 0
            if mf.HasField("lane"):
                lane_data = mf.lane.polyline
                feature_type = mf.lane.type
            elif mf.HasField("road_line"):
                lane_data = mf.road_line.polyline
                feature_type = mf.road_line.type
            elif mf.HasField("road_edge"):
                lane_data = mf.road_edge.polyline
                feature_type = mf.road_edge.type
            if lane_data:
                for pt in lane_data:
                    road_xyz_list.append([pt.x, pt.y, 0.0])
                    road_type_list.append(feature_type)
                    road_valid_list.append(1)
                    road_id_list.append(mf.id)

        roadgraph_xyz = (
            np.array(road_xyz_list, dtype=np.float32)
            if road_xyz_list
            else np.zeros((0, 3), dtype=np.float32)
        )
        roadgraph_type = (
            np.array(road_type_list, dtype=np.int32)
            if road_type_list
            else np.zeros(0, dtype=np.int32)
        )
        roadgraph_valid = (
            np.array(road_valid_list, dtype=np.int32)
            if road_valid_list
            else np.zeros(0, dtype=np.int32)
        )
        roadgraph_id = (
            np.array(road_id_list, dtype=np.int32)
            if road_id_list
            else np.zeros(0, dtype=np.int32)
        )

        # Render BEV raster (road channels)
        bev_image = self._render_bev_raster(
            roadgraph_xyz,
            roadgraph_type,
            roadgraph_valid,
            normalizer,
        )

        # Utils args must be set for get_agents
        utils.args = self.args
        self.args.image = bev_image

        # Create agent vectors using M2I cython code
        tracks_type[tracks_type < 0] = 0
        agent_vectors, agent_polyline_spans, _trajs = utils_cython.get_agents(
            gt_trajectory, gt_is_valid, tracks_type, False, self.args
        )

        # Create road vectors (pure Python)
        road_vectors, road_polyline_spans = self._create_road_vectors(
            roadgraph_xyz,
            roadgraph_type,
            roadgraph_valid,
            roadgraph_id,
            normalizer,
        )

        # Combine
        map_start_polyline_idx = len(agent_polyline_spans)
        if len(road_vectors) > 0:
            offset = len(agent_vectors)
            road_spans_adj = [
                slice(s.start + offset, s.stop + offset) for s in road_polyline_spans
            ]
            vectors = np.concatenate([agent_vectors, road_vectors], axis=0)
            polyline_spans = [
                slice(int(s[0]), int(s[1])) for s in agent_polyline_spans
            ] + road_spans_adj
        else:
            vectors = agent_vectors
            polyline_spans = [slice(int(s[0]), int(s[1])) for s in agent_polyline_spans]

        # Labels (future trajectory in normalized coords)
        labels = gt_trajectory[0, n_history:, :2].copy()
        labels_is_valid = gt_is_valid[0, n_history:].copy()

        # Speed and yaw
        vx = float(gt_trajectory[0, n_history - 1, 5])
        vy = float(gt_trajectory[0, n_history - 1, 6])
        speed = math.sqrt(vx**2 + vy**2)

        # Goals grid
        goals_2D = self._generate_raster_goals_grid()

        mapping = {
            "matrix": vectors,
            "polyline_spans": polyline_spans,
            "labels": labels,
            "labels_is_valid": labels_is_valid,
            "normalizer": normalizer,
            "goals_2D": goals_2D,
            "scenario_id": (
                scenario.scenario_id.encode()
                if isinstance(scenario.scenario_id, str)
                else scenario.scenario_id
            ),
            "object_id": tracks[agent_idx].id if agent_idx < len(tracks) else 0,
            "cent_x": float(cent_x),
            "cent_y": float(cent_y),
            "angle": float(angle),
            "eval_time": self.future_frames,
            "map_start_polyline_idx": map_start_polyline_idx,
            "speed": speed,
            "waymo_yaw": waymo_yaw,
            "track_type_int": int(tracks_type[0]),
            "stage_one_label": 0,
            "predict_agent_num": n_agents,
            "image": self.args.image.astype(np.float32),
        }
        return mapping

    # -----------------------------------------------------------------
    # Inference
    # -----------------------------------------------------------------

    def predict(
        self, mapping: Dict
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Run DenseTNT inference on a single M2I mapping.

        Returns:
            pred_traj: [6, 80, 2] trajectory modes in world coordinates
            pred_scores: [6] confidence scores (log-prob from NMS)
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        import utils
        import modeling.vectornet as vectornet_module

        # Restore DenseTNT args (M2I uses module-level globals)
        utils.args = self.args
        vectornet_module.args = self.args

        with torch.no_grad():
            try:
                pred_traj, pred_score, _ = self.model([mapping], self.device)
                pred_traj = np.array(pred_traj[0]) if pred_traj is not None else None
                pred_score = np.array(pred_score[0]) if pred_score is not None else None
                return pred_traj, pred_score
            except Exception as e:
                print(f"[M2ITrajectoryGenerator] Inference error: {e}")
                return None, None

    def predict_from_scenario(
        self,
        scenario,
        target_agent_idx: int = 0,
        current_time_index: Optional[int] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Dict]]:
        """
        End-to-end: Scenario proto → trajectory predictions.

        Args:
            scenario: Waymo Scenario proto
            target_agent_idx: 0 = SDC
            current_time_index: Override for "now" timestep. When used in
                receding-horizon mode, set this to the current simulation
                time so M2I builds its 11-step history ending at this
                timestep and predicts 80 future steps from here.

        Returns:
            pred_traj: [6, 80, 2] in world coordinates
            pred_scores: [6]
            mapping: The M2I mapping dict (for debug / rule evaluation)
        """
        mapping = self.scenario_to_mapping(
            scenario,
            target_agent_idx=target_agent_idx,
            current_time_index=current_time_index,
        )
        if mapping is None:
            return None, None, None

        pred_traj, pred_scores = self.predict(mapping)
        return pred_traj, pred_scores, mapping

    def predict_normalized(
        self,
        scenario,
        target_agent_idx: int = 0,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Dict]]:
        """
        Like predict_from_scenario but keeps trajectories in agent-local
        (normalized) coordinates. Useful for comparison with GT labels.

        Returns:
            pred_traj_local: [6, 80, 2] in local (normalizer) coordinates
            pred_scores: [6]
            mapping: The M2I mapping dict
        """
        mapping = self.scenario_to_mapping(
            scenario,
            target_agent_idx=target_agent_idx,
        )
        if mapping is None:
            return None, None, None

        import utils
        import modeling.vectornet as vectornet_module

        utils.args = self.args
        vectornet_module.args = self.args

        with torch.no_grad():
            try:
                # M2I's goals_2D_eval already calls normalizer(reverse=True)
                # We need to intercept BEFORE that happens.
                # Trick: temporarily replace normalizer with identity
                orig_normalizer = mapping["normalizer"]

                class IdentityNormalizer:
                    """No-op normalizer to keep local coords."""

                    def __init__(self):
                        self.x = 0.0
                        self.y = 0.0
                        self.yaw = 0.0

                    def __call__(self, points, reverse=False):
                        return points

                mapping["normalizer"] = IdentityNormalizer()
                pred_traj, pred_score, _ = self.model([mapping], self.device)
                mapping["normalizer"] = orig_normalizer

                pred_traj = np.array(pred_traj[0]) if pred_traj is not None else None
                pred_score = np.array(pred_score[0]) if pred_score is not None else None
                return pred_traj, pred_score, mapping
            except Exception as e:
                mapping["normalizer"] = orig_normalizer
                print(f"[M2ITrajectoryGenerator] Inference error: {e}")
                return None, None, mapping
