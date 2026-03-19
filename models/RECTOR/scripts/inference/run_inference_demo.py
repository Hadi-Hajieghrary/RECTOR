#!/usr/bin/env python3
"""
RECTOR V2 Inference Demo.

Runs inference on random scenarios and displays detailed predictions.

Usage:
    python inference/run_inference_demo.py
    python inference/run_inference_demo.py --num_samples 20
"""

import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import argparse
import glob
import random
from pathlib import Path
import numpy as np

import torch
from torch.utils.data import DataLoader

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, "/workspace/data/WOMD")
sys.path.insert(0, "/workspace/data")

from training.train_rector import WaymoDataset, collate_fn, TRAJECTORY_SCALE
from models.rule_aware_generator import RuleAwareGenerator
from training.losses import RECTORLoss


def parse_args():
    parser = argparse.ArgumentParser(description="RECTOR V2 Inference Demo")

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/workspace/models/RECTOR/models/best.pt",
        help="Path to checkpoint",
    )
    parser.add_argument(
        "--val_dir",
        type=str,
        default="/workspace/data/WOMD/datasets/waymo_open_dataset/motion_v_1_3_0/processed/augmented/scenario/validation_interactive",
        help="Path to validation data",
    )
    parser.add_argument(
        "--num_samples", type=int, default=20, help="Number of samples to process"
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size for inference"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("=" * 70)
    print("RECTOR V2 INFERENCE DEMO")
    print("=" * 70)

    # Load model
    print("\n[1/3] Loading Model...")
    print("-" * 70)

    model = RuleAwareGenerator(
        embed_dim=256,
        decoder_hidden_dim=256,
        decoder_num_layers=4,
        latent_dim=64,
        num_modes=6,
        use_m2i_encoder=True,
        m2i_checkpoint="/workspace/models/pretrained/m2i/models/relation_v2v/model.25.bin",
        freeze_m2i=True,
        trajectory_length=50,
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"  Checkpoint: {Path(args.checkpoint).name}")
    print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"  Best ADE: {checkpoint.get('best_ade', 'unknown'):.4f}m")
    print(f"  Total Parameters: {total_params:,}")
    print(f"  Trainable Parameters: {trainable_params:,}")

    # Load data
    print("\n[2/3] Loading Data...")
    print("-" * 70)

    val_files = sorted(glob.glob(os.path.join(args.val_dir, "*")))
    sample_files = random.sample(val_files, min(5, len(val_files)))

    print(f"  Data directory: {args.val_dir}")
    print(f"  Files sampled: {len(sample_files)}")

    dataset = WaymoDataset(sample_files, is_training=False)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        num_workers=0,
    )

    loss_fn = RECTORLoss()

    # Run inference
    print("\n[3/3] Running Inference...")
    print("-" * 70)

    all_ade = []
    all_fde = []
    all_miss = []
    sample_count = 0

    print(
        f"\n{'Sample':<8} {'minADE':<10} {'minFDE':<10} {'Conf Best':<12} {'Miss':<6}"
    )
    print("-" * 50)

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if not batch or sample_count >= args.num_samples:
                break

            # Move to device
            ego_history = batch["ego_history"].to(device)
            agent_states = batch["agent_states"].to(device)
            lane_centers = batch["lane_centers"].to(device)
            traj_gt = batch["traj_gt"].to(device)

            # Forward pass
            outputs = model(
                ego_history=ego_history,
                agent_states=agent_states,
                lane_centers=lane_centers,
                traj_gt=traj_gt,
            )

            # Compute metrics
            losses = loss_fn(outputs, {"traj_gt": traj_gt})

            # Per-sample analysis
            trajectories = outputs["trajectories"]  # [B, 6, 50, 4]
            confidence = outputs["confidence"]  # [B, 6]

            B = ego_history.shape[0]
            for i in range(B):
                if sample_count >= args.num_samples:
                    break

                pred = trajectories[i, :, :, :2]  # [6, 50, 2]
                gt = traj_gt[i, :, :2]  # [50, 2]
                conf = torch.softmax(confidence[i], dim=0)

                # Compute ADE/FDE for best mode
                ade_per_mode = (pred - gt.unsqueeze(0)).norm(dim=-1).mean(dim=-1)  # [6]
                best_idx = ade_per_mode.argmin()

                minADE = ade_per_mode[best_idx].item() * TRAJECTORY_SCALE
                minFDE = (pred[best_idx, -1] - gt[-1]).norm().item() * TRAJECTORY_SCALE
                best_conf = conf[best_idx].item()
                miss = minFDE > 2.0

                all_ade.append(minADE)
                all_fde.append(minFDE)
                all_miss.append(miss)

                miss_str = "MISS" if miss else "OK"
                print(
                    f"{sample_count:<8} {minADE:<10.3f} {minFDE:<10.3f} {best_conf:<12.3f} {miss_str:<6}"
                )

                sample_count += 1

    # Summary statistics
    print("\n" + "=" * 70)
    print("INFERENCE SUMMARY")
    print("=" * 70)

    ade_arr = np.array(all_ade)
    fde_arr = np.array(all_fde)

    print(f"\n  Samples Processed: {len(all_ade)}")
    print(f"\n  minADE Statistics:")
    print(f"    Mean:   {ade_arr.mean():.3f}m")
    print(f"    Std:    {ade_arr.std():.3f}m")
    print(f"    Median: {np.median(ade_arr):.3f}m")
    print(f"    Min:    {ade_arr.min():.3f}m")
    print(f"    Max:    {ade_arr.max():.3f}m")

    print(f"\n  minFDE Statistics:")
    print(f"    Mean:   {fde_arr.mean():.3f}m")
    print(f"    Std:    {fde_arr.std():.3f}m")
    print(f"    Median: {np.median(fde_arr):.3f}m")
    print(f"    Min:    {fde_arr.min():.3f}m")
    print(f"    Max:    {fde_arr.max():.3f}m")

    print(f"\n  Miss Rate (@2.0m): {np.mean(all_miss)*100:.2f}%")

    # Percentiles
    print(f"\n  ADE Percentiles:")
    for p in [50, 75, 90, 95]:
        print(f"    P{p}: {np.percentile(ade_arr, p):.3f}m")

    print("\n" + "=" * 70)
    print("Model Output Structure")
    print("=" * 70)
    print("\n  outputs = model(ego_history, agent_states, lane_centers)")
    print("\n  Output keys:")
    for key, val in outputs.items():
        if isinstance(val, torch.Tensor):
            print(f"    '{key}': {list(val.shape)}")
        else:
            print(f"    '{key}': {type(val).__name__}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
