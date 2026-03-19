#!/usr/bin/env python3
"""
Train RECTOR Applicability Model: SceneEncoder + RuleApplicabilityHead.

Trains ONLY the scene understanding (which rules apply to this scenario)
without trajectory generation. M2I handles trajectory generation at inference.

Architecture trained:
    M2ISceneEncoder(ego_history, agent_states, lane_centers) → scene_embed [B, 256]
    RuleApplicabilityHead(scene_embed) → applicability logits [B, 28]

Loss: Binary cross-entropy on 28 rule applicability labels from augmented TFRecords.

Usage:
    python training/train_applicability.py --experiment_name app_head_v1
    python training/train_applicability.py --cache_dir /workspace/output/cache
"""

import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import argparse
import glob
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset, IterableDataset

import tensorflow as tf

tf.config.set_visible_devices([], "GPU")

_WORKSPACE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(_WORKSPACE / "data" / "WOMD"))
sys.path.insert(0, str(_WORKSPACE / "data"))

from lib.rector_pipeline import RECTORApplicabilityModel
from models.applicability_head import RuleApplicabilityLoss

# Reuse data loading from train_rector
from training.train_rector import (
    WaymoDataset,
    CachedDataset,
    collate_fn,
    TRAJECTORY_SCALE,
)

try:
    from waymo_rule_eval.rules.rule_constants import (
        NUM_RULES,
        RULE_IDS,
        get_tier_mask,
        TIERS,
    )

    TIER_MASKS = {
        name: torch.tensor(get_tier_mask(name), dtype=torch.bool) for name in TIERS
    }
except ImportError:
    NUM_RULES = 28
    RULE_IDS = []
    TIERS = ["safety", "legal", "road", "comfort"]
    TIER_MASKS = {}


def parse_args():
    parser = argparse.ArgumentParser(description="Train RECTOR Applicability Head")

    # Data
    parser.add_argument(
        "--train_dir",
        type=str,
        default="/workspace/data/WOMD/datasets/waymo_open_dataset/"
        "motion_v_1_3_0/processed/augmented/scenario/training_interactive",
    )
    parser.add_argument(
        "--val_dir",
        type=str,
        default="/workspace/data/WOMD/datasets/waymo_open_dataset/"
        "motion_v_1_3_0/processed/augmented/scenario/validation_interactive",
    )
    parser.add_argument(
        "--output_dir", type=str, default="/workspace/models/RECTOR/output"
    )
    parser.add_argument("--experiment_name", type=str, default="applicability_head")
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Pre-cached .pt directory (from preprocess_cache.py)",
    )

    # Model
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument(
        "--m2i_checkpoint",
        type=str,
        default="/workspace/models/pretrained/m2i/models/relation_v2v/model.25.bin",
    )
    parser.add_argument(
        "--freeze_m2i",
        action="store_true",
        default=True,
        help="Freeze M2I backbone (default: True). Use --no_freeze_m2i to unfreeze.",
    )
    parser.add_argument(
        "--no_freeze_m2i",
        dest="freeze_m2i",
        action="store_false",
        help="Unfreeze M2I backbone for end-to-end fine-tuning.",
    )

    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--val_batches", type=int, default=80)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--resume", type=str, default=None)

    # Performance
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--compile", action="store_true", default=False)

    return parser.parse_args()


class ApplicabilityTrainer:
    """Trains RECTORApplicabilityModel (SceneEncoder + ApplicabilityHead)."""

    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(args.output_dir) / f"{args.experiment_name}_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._setup_data()
        self._setup_model()
        self._setup_training()

        self.best_loss = float("inf")
        self.best_f1 = 0.0
        self.patience_counter = 0
        self.start_epoch = 1

        if args.resume:
            self._resume_from_checkpoint(args.resume)

    def _setup_data(self):
        cache_dir = self.args.cache_dir
        if cache_dir and Path(cache_dir).exists():
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
            if cache_dir:
                print(f"WARNING: cache_dir {cache_dir} not found, using TFRecords")
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
            self._num_train_samples = len(train_files) * 500

    def _setup_model(self):
        self.model = RECTORApplicabilityModel(
            embed_dim=self.args.embed_dim,
            m2i_checkpoint=self.args.m2i_checkpoint,
            freeze_m2i=self.args.freeze_m2i,
        ).to(self.device)

        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total / 1e6:.2f}M")
        print(f"Trainable parameters: {trainable / 1e6:.2f}M")

        if self.args.compile and hasattr(torch, "compile"):
            print("Compiling model with torch.compile...")
            self.model = torch.compile(self.model)

    def _compute_pos_weight(
        self, max_batches: int = 50, safety_multiplier: float = 3.0
    ) -> torch.Tensor:
        """Compute per-rule positive class weight from training data.

        Scans a subset of training batches to estimate the positive/negative
        ratio for each rule, then applies an extra multiplier for safety-tier
        rules so the loss upweights rare but critical safety labels.
        """
        print("Computing per-rule pos_weight from training data...")
        pos_counts = torch.zeros(NUM_RULES)
        total_samples = 0

        for batch_idx, batch in enumerate(self.train_loader):
            if batch_idx >= max_batches or not batch:
                break
            labels = batch.get("rule_applicability")
            if labels is None:
                continue
            pos_counts += labels.sum(dim=0)
            total_samples += labels.shape[0]

        if total_samples == 0:
            print("  WARNING: no samples found, using uniform pos_weight=1.0")
            return torch.ones(NUM_RULES)

        neg_counts = total_samples - pos_counts
        # pos_weight = neg / pos (higher weight for rarer positives), clamped
        pos_weight = (neg_counts / pos_counts.clamp(min=1)).clamp(min=1.0, max=20.0)

        # Apply safety-tier multiplier
        if "safety" in TIER_MASKS:
            safety_mask = TIER_MASKS["safety"]
            pos_weight[safety_mask] *= safety_multiplier
            print(
                f"  Safety tier ({safety_mask.sum().item()} rules) upweighted by {safety_multiplier}x"
            )

        print(
            f"  Computed from {total_samples} samples over {min(batch_idx + 1, max_batches)} batches"
        )
        print(f"  pos_weight range: [{pos_weight.min():.1f}, {pos_weight.max():.1f}]")
        return pos_weight

    def _setup_training(self):
        param_groups = self.model.get_parameter_groups(self.args.learning_rate)
        self.optimizer = AdamW(param_groups, weight_decay=self.args.weight_decay)

        steps_per_epoch = max(self._num_train_samples // self.args.batch_size, 1)
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=self.args.learning_rate,
            epochs=self.args.epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.1,
        )
        self.scaler = GradScaler()
        self.grad_accum_steps = self.args.grad_accum_steps

        # Compute per-rule positive class weight from training data
        pos_weight = self._compute_pos_weight()

        # Focal loss for class imbalance (gamma=2.0) with per-rule weighting
        self.loss_fn = RuleApplicabilityLoss(focal_gamma=2.0, pos_weight=pos_weight)

    def _resume_from_checkpoint(self, path):
        print(f"Resuming from {path}")
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            try:
                self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except Exception as e:
                print(f"  Could not load optimizer state: {e}")
        if "best_loss" in ckpt:
            self.best_loss = ckpt["best_loss"]
        if "epoch" in ckpt:
            self.start_epoch = ckpt["epoch"] + 1

    def train_step(self, batch, is_accumulating=False):
        self.model.train()

        ego_history = batch["ego_history"].to(self.device)
        agent_states = batch["agent_states"].to(self.device)
        lane_centers = batch["lane_centers"].to(self.device)
        app_gt = batch.get(
            "rule_applicability", torch.zeros(ego_history.shape[0], NUM_RULES)
        ).to(self.device)

        with autocast():
            outputs = self.model(ego_history, agent_states, lane_centers)
            logits = outputs["applicability"]  # [B, 28]

            loss = self.loss_fn(logits, app_gt)
            scaled_loss = loss / self.grad_accum_steps

        self.scaler.scale(scaled_loss).backward()

        if not is_accumulating:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            self.scheduler.step()

        # Metrics
        with torch.no_grad():
            probs = outputs["applicability_prob"]
            preds = (probs > 0.5).float()
            tp = (preds * app_gt).sum()
            fp = (preds * (1 - app_gt)).sum()
            fn = ((1 - preds) * app_gt).sum()
            precision = tp / (tp + fp).clamp(min=1)
            recall = tp / (tp + fn).clamp(min=1)
            f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-8)

            # Per-tier accuracy
            tier_acc = {}
            for tier_name, tier_mask in TIER_MASKS.items():
                tmask = tier_mask.to(self.device)
                tier_preds = preds[:, tmask]
                tier_gt = app_gt[:, tmask]
                if tier_gt.numel() > 0:
                    tier_acc[tier_name] = (tier_preds == tier_gt).float().mean().item()

        metrics = {
            "loss": loss.item(),
            "f1": f1.item(),
            "precision": precision.item(),
            "recall": recall.item(),
        }
        metrics.update({f"acc_{k}": v for k, v in tier_acc.items()})
        return metrics

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0
        total_tp = 0
        total_fp = 0
        total_fn = 0
        count = 0

        for batch_idx, batch in enumerate(self.val_loader):
            if batch_idx >= self.args.val_batches:
                break
            if not batch:
                continue

            ego_history = batch["ego_history"].to(self.device)
            agent_states = batch["agent_states"].to(self.device)
            lane_centers = batch["lane_centers"].to(self.device)
            app_gt = batch.get(
                "rule_applicability", torch.zeros(ego_history.shape[0], NUM_RULES)
            ).to(self.device)

            with autocast():
                outputs = self.model(ego_history, agent_states, lane_centers)
                logits = outputs["applicability"]
                loss = F.binary_cross_entropy_with_logits(logits, app_gt)

            probs = outputs["applicability_prob"]
            preds = (probs > 0.5).float()
            total_tp += (preds * app_gt).sum().item()
            total_fp += (preds * (1 - app_gt)).sum().item()
            total_fn += ((1 - preds) * app_gt).sum().item()
            total_loss += loss.item()
            count += 1

        avg_loss = total_loss / max(count, 1)
        precision = total_tp / max(total_tp + total_fp, 1)
        recall = total_tp / max(total_tp + total_fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)

        return {
            "loss": avg_loss,
            "f1": f1,
            "precision": precision,
            "recall": recall,
        }

    def train(self):
        print(f"\n{'=' * 60}")
        print(f"RECTOR Applicability Head Training")
        print(f"{'=' * 60}")
        print(f"  Device: {self.device}")
        print(f"  Epochs: {self.start_epoch} to {self.args.epochs}")
        eff_bs = self.args.batch_size * self.grad_accum_steps
        print(
            f"  Batch: {self.args.batch_size}"
            + (f" (effective: {eff_bs})" if self.grad_accum_steps > 1 else "")
        )
        print(f"  LR: {self.args.learning_rate} (encoder @ 0.1x)")
        print(f"  Output: {self.output_dir}")
        print(f"{'=' * 60}\n")

        # Save config
        with open(self.output_dir / "config.json", "w") as f:
            json.dump(vars(self.args), f, indent=2)

        # Data sanity check: verify rule labels are not all-zeros
        print("Checking data labels...")
        for check_batch in self.train_loader:
            if check_batch:
                app_labels = check_batch.get("rule_applicability")
                if app_labels is not None and app_labels.sum().item() > 0:
                    nonzero_rules = (app_labels.sum(0) > 0).sum().item()
                    print(
                        f"  OK: {nonzero_rules}/{NUM_RULES} rules have positive labels in first batch"
                    )
                else:
                    print("  WARNING: rule_applicability labels are ALL ZEROS!")
                    print("  The applicability head will learn nothing useful.")
                    print("  Run the augmentation pipeline first to generate labels.")
                    print("  Aborting training.")
                    return
                break

        for epoch in range(self.start_epoch, self.args.epochs + 1):
            print(f"\nEpoch {epoch}/{self.args.epochs}")
            print("-" * 40)

            epoch_start = time.time()
            epoch_loss = 0
            epoch_f1 = 0
            step_count = 0

            self.optimizer.zero_grad()
            for batch_idx, batch in enumerate(self.train_loader):
                if not batch:
                    continue

                is_accumulating = (batch_idx + 1) % self.grad_accum_steps != 0
                metrics = self.train_step(batch, is_accumulating=is_accumulating)
                epoch_loss += metrics["loss"]
                epoch_f1 += metrics["f1"]
                step_count += 1

                if (batch_idx + 1) % self.args.log_every == 0:
                    lr = self.scheduler.get_last_lr()[0]
                    tier_str = (
                        " | ".join(
                            f"{t[:3]}:{metrics.get(f'acc_{t}', 0):.2f}" for t in TIERS
                        )
                        if TIER_MASKS
                        else ""
                    )
                    print(
                        f"  Step {batch_idx + 1} | Loss: {metrics['loss']:.4f} | "
                        f"F1: {metrics['f1']:.3f} | P: {metrics['precision']:.3f} | "
                        f"R: {metrics['recall']:.3f} | LR: {lr:.2e}"
                        + (f" | {tier_str}" if tier_str else "")
                    )

            epoch_time = time.time() - epoch_start
            avg_loss = epoch_loss / max(step_count, 1)
            avg_f1 = epoch_f1 / max(step_count, 1)

            print(
                f"\n  Train — Loss: {avg_loss:.4f}, F1: {avg_f1:.3f} ({epoch_time:.1f}s)"
            )

            # Validation
            val = self.validate()
            print(
                f"  Val   — Loss: {val['loss']:.4f}, F1: {val['f1']:.3f}, "
                f"P: {val['precision']:.3f}, R: {val['recall']:.3f}"
            )

            # Save best (by val loss)
            improved = False
            if val["loss"] < self.best_loss:
                self.best_loss = val["loss"]
                self.best_f1 = val["f1"]
                self.patience_counter = 0
                improved = True

                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "best_loss": self.best_loss,
                        "best_f1": self.best_f1,
                        "val_metrics": val,
                    },
                    self.output_dir / "best.pt",
                )

                print(
                    f"  >> New best! Loss: {self.best_loss:.4f}, F1: {self.best_f1:.3f}"
                )
            else:
                self.patience_counter += 1
                print(
                    f"  No improvement ({self.patience_counter}/{self.args.patience})"
                )

            # Periodic checkpoint
            if epoch % 5 == 0:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "best_loss": self.best_loss,
                        "best_f1": self.best_f1,
                    },
                    self.output_dir / f"epoch_{epoch}.pt",
                )

            if self.patience_counter >= self.args.patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

        print(f"\n{'=' * 60}")
        print(f"Training Complete!")
        print(f"  Best Loss: {self.best_loss:.4f}")
        print(f"  Best F1: {self.best_f1:.3f}")
        print(f"  Checkpoints: {self.output_dir}")
        print(f"{'=' * 60}")


def main():
    args = parse_args()
    trainer = ApplicabilityTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
