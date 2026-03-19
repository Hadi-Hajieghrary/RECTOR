#!/usr/bin/env python3
"""
RECTOR Training Script.

Features:
1. 5-second prediction horizon (50 steps)
2. Transformer decoder with goal-conditioned prediction
3. Winner-takes-all training with minADE/minFDE optimization
4. Rule applicability supervision from augmented TFRecords
5. Unfrozen M2I encoder (fine-tuned with low LR)

Usage:
    python training/train_rector.py --experiment_name my_run
"""

import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import argparse
import time
import json
import glob
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

import tensorflow as tf

tf.config.set_visible_devices([], "GPU")

_WORKSPACE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(_WORKSPACE / "data" / "WOMD"))
sys.path.insert(0, str(_WORKSPACE / "data"))

from models.rule_aware_generator import RuleAwareGenerator
from models.cvae_head import TRAJECTORY_LENGTH
from training.losses import RECTORLoss

# Constants
TRAJECTORY_SCALE = 50.0
FUTURE_LENGTH = TRAJECTORY_LENGTH  # 50 steps = 5 seconds


def parse_args():
    parser = argparse.ArgumentParser(description="RECTOR Training")

    # Data paths - use augmented data (has rule applicability + violation labels)
    parser.add_argument(
        "--train_dir",
        type=str,
        default="/workspace/data/WOMD/datasets/waymo_open_dataset/motion_v_1_3_0/processed/augmented/scenario/training_interactive",
    )
    parser.add_argument(
        "--val_dir",
        type=str,
        default="/workspace/data/WOMD/datasets/waymo_open_dataset/motion_v_1_3_0/processed/augmented/scenario/validation_interactive",
    )
    parser.add_argument(
        "--output_dir", type=str, default="/workspace/models/RECTOR/output"
    )
    parser.add_argument("--experiment_name", type=str, default="v2_transformer")

    # Model
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--decoder_hidden_dim", type=int, default=256)
    parser.add_argument("--decoder_num_layers", type=int, default=4)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--num_modes", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)

    # M2I
    parser.add_argument("--use_m2i_encoder", action="store_true", default=True)
    parser.add_argument(
        "--m2i_checkpoint",
        type=str,
        default="/workspace/models/pretrained/m2i/models/relation_v2v/model.25.bin",
    )
    parser.add_argument(
        "--freeze_m2i",
        action="store_true",
        default=False,
        help="Freeze M2I (default: False, fine-tune with low LR)",
    )
    parser.add_argument(
        "--pretrained_applicability",
        type=str,
        default=None,
        help="Path to pre-trained applicability head checkpoint (from train_applicability.py)",
    )

    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--val_batches", type=int, default=50)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )

    # Performance
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Pre-cached .pt directory (from preprocess_cache.py). Dramatically faster.",
    )
    parser.add_argument(
        "--grad_accum_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps (effective batch = batch_size * grad_accum_steps)",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        default=False,
        help="Use torch.compile for ~20%% speedup (requires PyTorch 2.0+)",
    )

    return parser.parse_args()


class WaymoDataset(IterableDataset):
    """Dataset for RECTOR with 5-second horizon."""

    def __init__(
        self,
        tfrecord_paths: List[str],
        future_length: int = FUTURE_LENGTH,
        max_agents: int = 32,
        max_lanes: int = 64,
        shuffle_buffer: int = 1000,
        is_training: bool = True,
    ):
        self.tfrecord_paths = tfrecord_paths
        self.future_length = future_length
        self.history_length = 11
        self.max_agents = max_agents
        self.max_lanes = max_lanes
        self.lane_points = 20
        self.shuffle_buffer = shuffle_buffer if is_training else 0
        self._approx_samples = len(tfrecord_paths) * 500

        try:
            from waymo_open_dataset.protos import scenario_pb2

            self.scenario_pb2 = scenario_pb2
        except ImportError:
            raise ImportError("waymo_open_dataset protos required")

    def __len__(self):
        return self._approx_samples

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            files = self.tfrecord_paths
        else:
            per_worker = len(self.tfrecord_paths) // worker_info.num_workers
            worker_id = worker_info.id
            start = worker_id * per_worker
            end = (
                start + per_worker
                if worker_id < worker_info.num_workers - 1
                else len(self.tfrecord_paths)
            )
            files = self.tfrecord_paths[start:end]

        if not files:
            return

        dataset = tf.data.TFRecordDataset(files, num_parallel_reads=4)
        if self.shuffle_buffer > 0:
            dataset = dataset.shuffle(self.shuffle_buffer)

        for raw_record in dataset:
            try:
                sample = self._parse(raw_record)
                if sample is not None:
                    yield sample
            except Exception:
                continue

    def _parse(self, raw_record) -> Optional[Dict[str, torch.Tensor]]:
        raw_bytes = raw_record.numpy()

        # Try augmented Example format first (this is what augmented data uses)
        try:
            example = tf.train.Example()
            example.ParseFromString(raw_bytes)
            features = example.features.feature

            if "scenario/proto" in features:
                scenario_bytes = features["scenario/proto"].bytes_list.value[0]
                scenario = self.scenario_pb2.Scenario()
                scenario.ParseFromString(scenario_bytes)
                sample = self._extract_features(scenario)
                if sample is None:
                    return None

                if "rule/applicability" in features:
                    sample["rule_applicability"] = torch.tensor(
                        list(features["rule/applicability"].int64_list.value),
                        dtype=torch.float32,
                    )
                if "rule/violations" in features:
                    sample["rule_violations"] = torch.tensor(
                        list(features["rule/violations"].int64_list.value),
                        dtype=torch.float32,
                    )
                if "rule/severity" in features:
                    sample["rule_severity"] = torch.tensor(
                        list(features["rule/severity"].float_list.value),
                        dtype=torch.float32,
                    )

                if "scenario/id" in features:
                    sample["scenario_id"] = (
                        features["scenario/id"].bytes_list.value[0].decode("utf-8")
                    )

                return sample
        except Exception:
            pass

        # Fallback: try raw Scenario format (non-augmented data)
        try:
            scenario = self.scenario_pb2.Scenario()
            scenario.ParseFromString(raw_bytes)
            if scenario.scenario_id:
                return self._extract_features(scenario)
        except Exception:
            pass

        return None

    def _extract_features(self, scenario) -> Optional[Dict[str, torch.Tensor]]:
        tracks = list(scenario.tracks)
        if not tracks:
            return None

        # Find SDC: sdc_track_index is an array index, NOT a track object ID
        sdc_idx = scenario.sdc_track_index
        if sdc_idx < 0 or sdc_idx >= len(tracks):
            sdc_idx = 0

        sdc_track = tracks[sdc_idx]
        current_ts = scenario.current_time_index

        # Need enough future
        if len(sdc_track.states) < current_ts + self.future_length:
            return None

        # Reference pose - must be valid
        ref_state = (
            sdc_track.states[current_ts] if current_ts < len(sdc_track.states) else None
        )
        if ref_state is None or not ref_state.valid:
            return None

        # Check for invalid (zero) reference - indicates data issue
        ref_x, ref_y = ref_state.center_x, ref_state.center_y
        if abs(ref_x) < 0.001 and abs(ref_y) < 0.001:
            return None

        ref_h = ref_state.heading
        cos_h, sin_h = np.cos(-ref_h), np.sin(-ref_h)

        def normalize_coords(x, y):
            dx, dy = x - ref_x, y - ref_y
            return dx * cos_h - dy * sin_h, dx * sin_h + dy * cos_h

        def normalize_heading(h):
            rel_h = h - ref_h
            while rel_h > np.pi:
                rel_h -= 2 * np.pi
            while rel_h < -np.pi:
                rel_h += 2 * np.pi
            return rel_h

        # Ego history
        ego_history = np.zeros((self.history_length, 4), dtype=np.float32)
        for t in range(self.history_length):
            if t < len(sdc_track.states) and sdc_track.states[t].valid:
                state = sdc_track.states[t]
                x, y = normalize_coords(state.center_x, state.center_y)
                ego_history[t, 0] = x
                ego_history[t, 1] = y
                ego_history[t, 2] = normalize_heading(state.heading)
                ego_history[t, 3] = np.sqrt(state.velocity_x**2 + state.velocity_y**2)

        # Ego future (5 seconds = 50 steps)
        ego_future = np.zeros((self.future_length, 4), dtype=np.float32)
        valid_count = 0
        for t in range(self.future_length):
            ts_idx = current_ts + 1 + t
            if ts_idx < len(sdc_track.states) and sdc_track.states[ts_idx].valid:
                state = sdc_track.states[ts_idx]
                x, y = normalize_coords(state.center_x, state.center_y)
                ego_future[t, 0] = x
                ego_future[t, 1] = y
                ego_future[t, 2] = normalize_heading(state.heading)
                ego_future[t, 3] = np.sqrt(state.velocity_x**2 + state.velocity_y**2)
                valid_count += 1

        if valid_count < self.future_length // 2:
            return None

        # Agent states
        agent_states = np.zeros(
            (self.max_agents, self.history_length, 4), dtype=np.float32
        )
        agent_count = 0
        for i, track in enumerate(tracks):
            if i == sdc_idx or agent_count >= self.max_agents:
                continue
            has_valid = False
            for t in range(self.history_length):
                if t < len(track.states) and track.states[t].valid:
                    state = track.states[t]
                    x, y = normalize_coords(state.center_x, state.center_y)
                    agent_states[agent_count, t, 0] = x
                    agent_states[agent_count, t, 1] = y
                    agent_states[agent_count, t, 2] = normalize_heading(state.heading)
                    agent_states[agent_count, t, 3] = np.sqrt(
                        state.velocity_x**2 + state.velocity_y**2
                    )
                    has_valid = True
            if has_valid:
                agent_count += 1

        # Lane centers
        lane_centers = np.zeros((self.max_lanes, self.lane_points, 2), dtype=np.float32)
        lane_count = 0
        for map_feature in scenario.map_features:
            if lane_count >= self.max_lanes:
                break
            if map_feature.HasField("lane"):
                polyline = list(map_feature.lane.polyline)
                if polyline:
                    indices = np.linspace(
                        0, len(polyline) - 1, self.lane_points
                    ).astype(int)
                    for p_idx, src_idx in enumerate(indices):
                        x, y = normalize_coords(
                            polyline[src_idx].x, polyline[src_idx].y
                        )
                        lane_centers[lane_count, p_idx, 0] = x
                        lane_centers[lane_count, p_idx, 1] = y
                    lane_count += 1

        # Normalize by scale
        ego_history[:, :2] /= TRAJECTORY_SCALE
        ego_future[:, :2] /= TRAJECTORY_SCALE
        agent_states[:, :, :2] /= TRAJECTORY_SCALE
        lane_centers /= TRAJECTORY_SCALE

        # Sanity check: reject samples with unreasonably large normalized values
        # (e.g., from invalid reference states)
        max_val = max(np.abs(ego_future[:, :2]).max(), np.abs(ego_history[:, :2]).max())
        if max_val > 5.0:  # More than 250m - clearly invalid
            return None

        return {
            "ego_history": torch.from_numpy(ego_history),
            "agent_states": torch.from_numpy(agent_states),
            "lane_centers": torch.from_numpy(lane_centers),
            "traj_gt": torch.from_numpy(ego_future),
            "rule_applicability": torch.zeros(28, dtype=torch.float32),
            "rule_violations": torch.zeros(28, dtype=torch.float32),
            "rule_severity": torch.zeros(28, dtype=torch.float32),
        }


def collate_fn(batch):
    if not batch:
        return {}
    result = {}
    for k in batch[0].keys():
        vals = [s[k] for s in batch]
        if isinstance(vals[0], torch.Tensor):
            result[k] = torch.stack(vals)
        else:
            result[k] = vals  # strings, etc.
    return result


class CachedDataset(Dataset):
    """
    Fast random-access dataset from pre-cached .pt files.

    Created by training/preprocess_cache.py. Eliminates protobuf parsing —
    typically 3-5x faster per epoch than TFRecord-based IterableDataset.
    """

    def __init__(self, cache_dir: str, is_training: bool = True):
        self.cache_dir = Path(cache_dir)
        self.is_training = is_training

        # Discover all .pt files
        self.files = sorted(self.cache_dir.glob("*.pt"))
        if not self.files:
            raise FileNotFoundError(f"No .pt files found in {cache_dir}")
        print(f"  CachedDataset: {len(self.files)} samples from {cache_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return torch.load(self.files[idx], weights_only=True)


class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create output dir
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(args.output_dir) / f"{args.experiment_name}_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self._setup_data()
        self._setup_model()
        self._setup_training()

        # Tracking
        self.best_loss = float("inf")
        self.best_ade = float("inf")
        self.patience_counter = 0
        self.start_epoch = 1

        # Resume from checkpoint if specified
        if args.resume:
            self._resume_from_checkpoint(args.resume)
            # If also fine-tuning with a newer applicability head, load it AFTER resume
            if args.pretrained_applicability:
                print(
                    "Overriding applicability head with fine-tuned weights (post-resume)..."
                )
                self._load_pretrained_applicability(args.pretrained_applicability)

    def _resume_from_checkpoint(self, checkpoint_path):
        """Load model and optimizer state from checkpoint."""
        print(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])

        # Optionally load optimizer state (for exact resume)
        if "optimizer_state_dict" in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                print("  Loaded optimizer state")
            except Exception as e:
                print(f"  Could not load optimizer state: {e}")

        # Restore tracking
        if "best_loss" in checkpoint:
            self.best_loss = checkpoint["best_loss"]
        if "best_ade" in checkpoint:
            self.best_ade = checkpoint["best_ade"]
        if "epoch" in checkpoint:
            self.start_epoch = checkpoint["epoch"] + 1
            print(f"  Resuming from epoch {self.start_epoch}")

        print(f"  Best loss so far: {self.best_loss:.4f}")
        print(f"  Best ADE so far: {self.best_ade:.2f}m")

    def _setup_data(self):
        cache_dir = self.args.cache_dir

        if cache_dir and Path(cache_dir).exists():
            # Fast path: pre-cached .pt files (no protobuf parsing)
            train_cache = Path(cache_dir) / "train_interactive"
            val_cache = Path(cache_dir) / "val_interactive"

            train_dataset = CachedDataset(str(train_cache), is_training=True)
            val_dataset = CachedDataset(str(val_cache), is_training=False)

            self.train_loader = DataLoader(
                train_dataset,
                batch_size=self.args.batch_size,
                shuffle=True,
                num_workers=self.args.num_workers,
                collate_fn=collate_fn,
                pin_memory=True,
                persistent_workers=self.args.num_workers > 0,
                drop_last=True,
            )
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=self.args.batch_size,
                shuffle=False,
                num_workers=self.args.num_workers,
                collate_fn=collate_fn,
                pin_memory=True,
                persistent_workers=self.args.num_workers > 0,
            )
            self._num_train_samples = len(train_dataset)
        else:
            # Slow path: parse TFRecords on-the-fly
            if cache_dir:
                print(
                    f"WARNING: cache_dir {cache_dir} not found, falling back to TFRecord parsing"
                )
            train_files = sorted(
                glob.glob(os.path.join(self.args.train_dir, "*.tfrecord*"))
            )
            val_files = sorted(
                glob.glob(os.path.join(self.args.val_dir, "*.tfrecord*"))
            )

            print(f"Train files: {len(train_files)}, Val files: {len(val_files)}")

            train_dataset = WaymoDataset(train_files, is_training=True)
            val_dataset = WaymoDataset(val_files, is_training=False)

            self.train_loader = DataLoader(
                train_dataset,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                collate_fn=collate_fn,
                prefetch_factor=2,
                pin_memory=True,
            )
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                collate_fn=collate_fn,
                prefetch_factor=2,
                pin_memory=True,
            )
            self._num_train_samples = len(train_files) * 500  # Approximate

    def _setup_model(self):
        self.model = RuleAwareGenerator(
            embed_dim=self.args.embed_dim,
            decoder_hidden_dim=self.args.decoder_hidden_dim,
            decoder_num_layers=self.args.decoder_num_layers,
            latent_dim=self.args.latent_dim,
            num_modes=self.args.num_modes,
            dropout=self.args.dropout,
            use_m2i_encoder=self.args.use_m2i_encoder,
            m2i_checkpoint=self.args.m2i_checkpoint,
            freeze_m2i=self.args.freeze_m2i,
            trajectory_length=FUTURE_LENGTH,
        ).to(self.device)

        # Load pre-trained applicability head if specified
        if self.args.pretrained_applicability:
            self._load_pretrained_applicability(self.args.pretrained_applicability)

        # Print model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print(f"Total parameters: {total_params/1e6:.2f}M")
        print(f"Trainable parameters: {trainable_params/1e6:.2f}M")

        # Loss function
        self.loss_fn = RECTORLoss()

        # Optional torch.compile (PyTorch 2.0+)
        if self.args.compile and hasattr(torch, "compile"):
            print("Compiling model with torch.compile...")
            self.model = torch.compile(self.model)

    def _load_pretrained_applicability(self, checkpoint_path):
        """Load pre-trained scene_encoder + applicability_head from train_applicability.py checkpoint."""
        print(f"Loading pre-trained applicability head from: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        src_state = ckpt["model_state_dict"]

        # Filter to only scene_encoder and applicability_head keys that match
        target_state = self.model.state_dict()
        filtered = {}
        for key, val in src_state.items():
            if key.startswith(("scene_encoder.", "applicability_head.")):
                if key in target_state and target_state[key].shape == val.shape:
                    filtered[key] = val

        self.model.load_state_dict(filtered, strict=False)
        enc_count = sum(1 for k in filtered if k.startswith("scene_encoder."))
        head_count = sum(1 for k in filtered if k.startswith("applicability_head."))
        print(
            f"  Loaded {enc_count} encoder params, {head_count} applicability head params"
        )

    def _setup_training(self):
        # Optimizer with different LR for encoder
        param_groups = self.model.get_parameter_groups(self.args.learning_rate)
        self.optimizer = AdamW(param_groups, weight_decay=self.args.weight_decay)

        # Scheduler — use actual sample count when available
        steps_per_epoch = self._num_train_samples // self.args.batch_size
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=self.args.learning_rate,
            epochs=self.args.epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.1,
        )

        # AMP
        self.scaler = GradScaler()

        # Gradient accumulation
        self.grad_accum_steps = self.args.grad_accum_steps

    def train_step(self, batch, is_accumulating=False):
        self.model.train()

        ego_history = batch["ego_history"].to(self.device)
        agent_states = batch["agent_states"].to(self.device)
        lane_centers = batch["lane_centers"].to(self.device)
        traj_gt = batch["traj_gt"].to(self.device)
        applicability_gt = batch.get(
            "rule_applicability", torch.zeros(traj_gt.shape[0], 28)
        ).to(self.device)

        with autocast():
            outputs = self.model(
                ego_history=ego_history,
                agent_states=agent_states,
                lane_centers=lane_centers,
                traj_gt=traj_gt,
            )

            losses = self.loss_fn(
                outputs, {"traj_gt": traj_gt, "applicability_gt": applicability_gt}
            )
            scaled_loss = losses["total_loss"] / self.grad_accum_steps

        self.scaler.scale(scaled_loss).backward()

        if not is_accumulating:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            self.scheduler.step()

        return losses

    @torch.no_grad()
    def validate(self):
        self.model.eval()

        total_loss = 0
        total_ade = 0
        total_fde = 0
        count = 0

        for batch_idx, batch in enumerate(self.val_loader):
            if batch_idx >= self.args.val_batches:
                break
            if not batch:
                continue

            ego_history = batch["ego_history"].to(self.device)
            agent_states = batch["agent_states"].to(self.device)
            lane_centers = batch["lane_centers"].to(self.device)
            traj_gt = batch["traj_gt"].to(self.device)
            applicability_gt = batch.get(
                "rule_applicability", torch.zeros(traj_gt.shape[0], 28)
            ).to(self.device)

            with autocast():
                outputs = self.model(
                    ego_history=ego_history,
                    agent_states=agent_states,
                    lane_centers=lane_centers,
                    traj_gt=traj_gt,
                )
                losses = self.loss_fn(
                    outputs, {"traj_gt": traj_gt, "applicability_gt": applicability_gt}
                )

            total_loss += losses["total_loss"].item()
            total_ade += losses["minADE"].item()
            total_fde += losses["minFDE"].item()
            count += 1

        return {
            "loss": total_loss / max(count, 1),
            "ade": total_ade / max(count, 1),
            "fde": total_fde / max(count, 1),
        }

    def train(self):
        print(f"\n{'='*60}")
        print(f"Starting RECTOR Training")
        print(f"{'='*60}")
        print(f"  Device: {self.device}")
        print(f"  Epochs: {self.start_epoch} to {self.args.epochs}")
        print(
            f"  Batch Size: {self.args.batch_size}"
            + (
                f" (effective: {self.args.batch_size * self.grad_accum_steps})"
                if self.grad_accum_steps > 1
                else ""
            )
        )
        print(f"  Learning Rate: {self.args.learning_rate}")
        print(
            f"  Prediction Horizon: {FUTURE_LENGTH} steps ({FUTURE_LENGTH/10:.1f} sec)"
        )
        print(f"  Cache: {'enabled' if self.args.cache_dir else 'disabled (TFRecord)'}")
        print(f"  Output: {self.output_dir}")
        print(f"{'='*60}\n")

        for epoch in range(self.start_epoch, self.args.epochs + 1):
            print(f"\n{'='*40}")
            print(f"Epoch {epoch}/{self.args.epochs}")
            print(f"{'='*40}")

            epoch_start = time.time()
            epoch_loss = 0
            step_count = 0

            self.optimizer.zero_grad()
            for batch_idx, batch in enumerate(self.train_loader):
                if not batch:
                    continue

                is_accumulating = (batch_idx + 1) % self.grad_accum_steps != 0
                losses = self.train_step(batch, is_accumulating=is_accumulating)
                epoch_loss += losses["total_loss"].item()
                step_count += 1

                if (batch_idx + 1) % self.args.log_every == 0:
                    lr = self.scheduler.get_last_lr()[0]
                    print(
                        f"  Step {batch_idx+1} | Loss: {losses['total_loss'].item():.4f} | "
                        f"ADE: {losses['minADE'].item():.2f}m | FDE: {losses['minFDE'].item():.2f}m | "
                        f"LR: {lr:.2e}"
                    )

            epoch_time = time.time() - epoch_start
            avg_loss = epoch_loss / max(step_count, 1)

            print(f"\nEpoch {epoch} completed in {epoch_time:.1f}s")
            print(f"  Avg Train Loss: {avg_loss:.4f}")

            # Validation
            print(f"\n  Running validation...")
            val_metrics = self.validate()
            print(
                f"  [VAL] Loss: {val_metrics['loss']:.4f} | "
                f"ADE: {val_metrics['ade']:.2f}m | FDE: {val_metrics['fde']:.2f}m"
            )

            # Save best
            if val_metrics["loss"] < self.best_loss:
                self.best_loss = val_metrics["loss"]
                self.best_ade = val_metrics["ade"]
                self.patience_counter = 0

                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "best_loss": self.best_loss,
                        "best_ade": self.best_ade,
                    },
                    self.output_dir / "best.pt",
                )

                print(
                    f"  ✓ New best! Loss: {self.best_loss:.4f}, ADE: {self.best_ade:.2f}m"
                )
            else:
                self.patience_counter += 1
                print(
                    f"  No improvement ({self.patience_counter}/{self.args.patience})"
                )

            # Early stopping
            if self.patience_counter >= self.args.patience:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                break

        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"  Best Loss: {self.best_loss:.4f}")
        print(f"  Best ADE: {self.best_ade:.2f}m")
        print(f"  Checkpoints: {self.output_dir}")
        print(f"{'='*60}")


def main():
    args = parse_args()

    print("RECTOR Training")
    print("=" * 40)
    print(f"  Experiment: {args.experiment_name}")
    print(f"  Horizon: 5 seconds ({FUTURE_LENGTH} steps)")
    print(f"  Decoder: Transformer ({args.decoder_num_layers} layers)")
    print(f"  M2I Encoder: {args.use_m2i_encoder} (freeze={args.freeze_m2i})")
    print("=" * 40)

    trainer = Trainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
