#!/usr/bin/env python3
"""
RECTOR Evaluation Script.

Uses EXACT same data loading as training to ensure metrics match.

Usage:
    python evaluation/evaluate_rector.py --checkpoint /path/to/best.pt
"""

import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import argparse
import glob
import time
from pathlib import Path
import numpy as np

import torch
from torch.utils.data import DataLoader

# Add model paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, "/workspace/data/WOMD")
sys.path.insert(0, "/workspace/data")

# Import EXACT training code - this is critical for matching metrics
from training.train_rector import WaymoDataset, collate_fn, TRAJECTORY_SCALE
from models.rule_aware_generator import RuleAwareGenerator
from training.losses import RECTORLoss


def parse_args():
    parser = argparse.ArgumentParser(description="RECTOR V2 Evaluation")

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained checkpoint (best.pt)",
    )
    parser.add_argument(
        "--val_dir",
        type=str,
        default="/workspace/data/WOMD/datasets/waymo_open_dataset/motion_v_1_3_0/processed/augmented/scenario/validation_interactive",
        help="Path to validation data",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/workspace/output/evaluation",
        help="Output directory for results",
    )

    # Model config (must match training)
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

    # Evaluation
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--max_batches",
        type=int,
        default=None,
        help="Maximum batches to evaluate (None = all)",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Output dir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load Model (same as training) ---
    print("=" * 60)
    print("RECTOR V2 Evaluation")
    print("=" * 60)
    print("\nLoading model...")

    model = RuleAwareGenerator(
        embed_dim=args.embed_dim,
        decoder_hidden_dim=args.decoder_hidden_dim,
        decoder_num_layers=args.decoder_num_layers,
        latent_dim=args.latent_dim,
        num_modes=args.num_modes,
        dropout=args.dropout,
        use_m2i_encoder=args.use_m2i_encoder,
        m2i_checkpoint=args.m2i_checkpoint,
        freeze_m2i=True,
        trajectory_length=50,  # 5 seconds
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    epoch = checkpoint.get("epoch", "unknown")
    best_ade = checkpoint.get("best_ade", "unknown")
    print(f"  Loaded checkpoint from epoch {epoch} (best ADE: {best_ade})")

    # --- Load Data (EXACT same as training) ---
    print("\nLoading validation data...")
    val_files = sorted(glob.glob(os.path.join(args.val_dir, "*")))

    if not val_files:
        print(f"ERROR: No files found in {args.val_dir}")
        return

    print(f"  Found {len(val_files)} files")

    # Use EXACT training dataset class and collate function
    dataset = WaymoDataset(val_files, is_training=False)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Use EXACT training loss function for metrics
    loss_fn = RECTORLoss()

    # --- Evaluation Loop ---
    print("\nRunning evaluation...")
    start_time = time.time()

    all_ade = []
    all_fde = []
    all_miss = []
    batch_count = 0
    sample_count = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if not batch:
                continue

            if args.max_batches and batch_idx >= args.max_batches:
                break

            # Move to device
            ego_history = batch["ego_history"].to(device)
            agent_states = batch["agent_states"].to(device)
            lane_centers = batch["lane_centers"].to(device)
            traj_gt = batch["traj_gt"].to(device)

            batch_size = ego_history.shape[0]

            # Forward pass (same as training)
            outputs = model(
                ego_history=ego_history,
                agent_states=agent_states,
                lane_centers=lane_centers,
                traj_gt=traj_gt,
            )

            # Compute metrics using EXACT training loss function
            losses = loss_fn(outputs, {"traj_gt": traj_gt})

            # Collect batch metrics
            all_ade.append(losses["minADE"].item())
            all_fde.append(losses["minFDE"].item())

            # Compute miss rate (per sample in batch)
            pred_traj = outputs["trajectories"][
                :, :, :, :2
            ]  # [B, M, T, 2] - use 'trajectories' key
            # Get best mode per sample
            traj_gt_pos = traj_gt[:, :, :2]  # [B, T, 2]
            B, M, T, _ = pred_traj.shape

            # ADE per mode
            ade_per_mode = (
                (pred_traj - traj_gt_pos.unsqueeze(1)).norm(dim=-1).mean(dim=-1)
            )  # [B, M]
            best_mode_idx = ade_per_mode.argmin(dim=1)  # [B]

            # FDE of best mode (scaled)
            best_pred = pred_traj[
                torch.arange(B, device=device), best_mode_idx
            ]  # [B, T, 2]
            fde_per_sample = (best_pred[:, -1] - traj_gt_pos[:, -1]).norm(
                dim=-1
            ) * TRAJECTORY_SCALE  # [B]
            miss_per_sample = (fde_per_sample > 2.0).float()
            all_miss.extend(miss_per_sample.cpu().tolist())

            batch_count += 1
            sample_count += batch_size

            if (batch_idx + 1) % 20 == 0:
                elapsed = time.time() - start_time
                print(
                    f"  Batch {batch_idx + 1}: {sample_count} samples, "
                    f"ADE={np.mean(all_ade):.3f}m, "
                    f"FDE={np.mean(all_fde):.3f}m, "
                    f"({elapsed:.1f}s)"
                )

    # --- Final Metrics ---
    elapsed = time.time() - start_time

    mean_ade = np.mean(all_ade)
    std_ade = np.std(all_ade)
    mean_fde = np.mean(all_fde)
    std_fde = np.std(all_fde)
    miss_rate = np.mean(all_miss) * 100

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"\n  Checkpoint: {args.checkpoint}")
    print(f"  Training epoch: {epoch}")
    print(f"  Checkpoint best ADE: {best_ade}")
    print(f"\n  Batches evaluated: {batch_count}")
    print(f"  Samples evaluated: {sample_count}")
    print(f"  Time elapsed: {elapsed:.1f}s")
    print(f"\n  --- Metrics (using EXACT training code) ---")
    print(f"  minADE: {mean_ade:.3f}m ± {std_ade:.3f}m")
    print(f"  minFDE: {mean_fde:.3f}m ± {std_fde:.3f}m")
    print(f"  Miss Rate (@2.0m): {miss_rate:.2f}%")
    print("=" * 60)

    # Save results
    results = {
        "checkpoint": args.checkpoint,
        "epoch": epoch,
        "checkpoint_best_ade": (
            float(best_ade) if isinstance(best_ade, (int, float)) else str(best_ade)
        ),
        "batches_evaluated": batch_count,
        "samples_evaluated": sample_count,
        "minADE_mean": mean_ade,
        "minADE_std": std_ade,
        "minFDE_mean": mean_fde,
        "minFDE_std": std_fde,
        "miss_rate": miss_rate,
        "elapsed_time": elapsed,
    }

    import json

    results_path = output_dir / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
