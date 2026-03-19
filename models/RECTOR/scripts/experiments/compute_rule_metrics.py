#!/usr/bin/env python3
"""
Compute REAL rule-related metrics from RECTOR predictions.

Since kinematic-based rule evaluation (acceleration limits) is noisy for
neural predictions, we focus on:
1. Miss rate (proxy for goal-reaching / safety)
2. Collision proxy (minimum distance to other agents)
3. Lane deviation (proxy for road compliance)

These can be computed reliably from model outputs.

Output:
    /workspace/models/RECTOR/output/artifacts/real_rule_metrics.json
"""

import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import argparse
import glob
import json
import time
from pathlib import Path
from collections import defaultdict
import numpy as np

import torch
from torch.utils.data import DataLoader

# Add paths
sys.path.insert(0, "/workspace/models/RECTOR/scripts")
sys.path.insert(0, "/workspace/data/WOMD")
sys.path.insert(0, "/workspace/data")

# Import model and data loading
from training.train_rector import WaymoDataset, collate_fn, TRAJECTORY_SCALE
from models.rule_aware_generator import RuleAwareGenerator

# Import rule constants
from waymo_rule_eval.rules.rule_constants import (
    TIER_0_SAFETY,
    TIER_1_LEGAL,
    TIER_2_ROAD,
    TIER_3_COMFORT,
    NUM_RULES,
    RULE_IDS,
)


def parse_args():
    parser = argparse.ArgumentParser(description="RECTOR Rule Metrics")

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/workspace/models/RECTOR/models/best.pt",
        help="Path to trained checkpoint",
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
        default="/workspace/models/RECTOR/output/artifacts",
        help="Output directory",
    )
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=128)

    return parser.parse_args()


def compute_rule_metrics(outputs, batch, device):
    """
    Compute rule-related metrics that can be reliably measured:

    1. Miss rate (FDE > 2m) - Safety tier proxy
    2. Agent proximity (min distance to other agents) - Safety tier
    3. Lane deviation (max distance from lane centers) - Road tier

    Notes on agent_states shape:
        The dataset stores HISTORY states only: [B, A, 11, 4].
        The predicted future has T=50 steps, so we cannot align
        timesteps directly.  Instead we use each agent's last known
        (most-recent history) position as a static reference and
        check whether any future predicted ego step comes within 2 m.
        This is a conservative proximity proxy.

    Notes on lane_centers shape:
        The dataset stores polylines: [B, 64, 20, 2].
        We use the midpoint of each polyline (index 10 of 20 points)
        as a representative lane-center coordinate, giving [B, 64, 2].
        We check proximity across the full predicted trajectory
        (not just the final position) to catch mid-trajectory departures.

    Returns metrics dict.
    """
    pred_traj = outputs["trajectories"]  # [B, M, T, 4]
    B, M, T, _ = pred_traj.shape

    traj_gt = batch["traj_gt"].to(device)
    # agent_states shape: [B, A, H, 4]  where H = 11 (history steps)
    agent_states = batch["agent_states"].to(device)
    # lane_centers shape: [B, L, P, 2]  where L = 64 lanes, P = 20 polyline points
    lane_centers = batch["lane_centers"].to(device)

    # Get best trajectory per sample (oracle best-of-K by ADE)
    ade_per_mode = (
        (pred_traj[:, :, :, :2] - traj_gt[:, :, :2].unsqueeze(1))
        .norm(dim=-1)
        .mean(dim=-1)
    )
    best_mode_idx = ade_per_mode.argmin(dim=1)
    best_traj = pred_traj[torch.arange(B, device=device), best_mode_idx]  # [B, T, 4]

    # Metrics accumulators
    miss_count = 0
    close_agent_count = 0
    off_lane_count = 0

    for b in range(B):
        pred = best_traj[b, :, :2] * TRAJECTORY_SCALE  # [T, 2] in meters
        gt = traj_gt[b, :, :2] * TRAJECTORY_SCALE

        # 1. Miss rate (FDE > 2m)
        fde = (pred[-1] - gt[-1]).norm().item()
        if fde > 2.0:
            miss_count += 1

        # 2. Agent proximity — Safety tier
        # agent_states[b] is [A, H, 4]; H=11 history steps, NOT future steps.
        # Use the last history position of each agent as a static reference.
        A_states = agent_states[b]  # [A, H, 4]
        if A_states.shape[0] > 0:
            # Last valid history position for each agent: [A, 2] in meters
            agent_last_pos = A_states[:, -1, :2] * TRAJECTORY_SCALE  # [A, 2]
            # Broadcast: pred [T, 2] vs agent_last_pos [A, 2]
            # dist[a, t] = distance from ego at future step t to agent a
            dist = (pred.unsqueeze(0) - agent_last_pos.unsqueeze(1)).norm(
                dim=-1
            )  # [A, T]
            if dist.min().item() < 2.0:
                close_agent_count += 1

        # 3. Lane deviation — Road tier
        # lane_centers[b] is [L, P, 2]; use midpoint of each polyline as
        # representative lane-center: shape [L, 2] in meters.
        lane_mid = lane_centers[b, :, 10, :] * TRAJECTORY_SCALE  # [L, 2]
        if lane_mid.shape[0] > 0:
            # Check proximity across the full predicted trajectory [T, 2]
            # dist_lane[t, l] = distance from ego at step t to lane l
            dist_lane = (pred.unsqueeze(1) - lane_mid.unsqueeze(0)).norm(
                dim=-1
            )  # [T, L]
            # Scenario is off-lane if the closest lane at ANY step is > 3 m
            min_lane_dist_per_step = dist_lane.min(dim=-1).values  # [T]
            if min_lane_dist_per_step.max().item() > 3.0:
                off_lane_count += 1

    return {
        "total": B,
        "miss_count": miss_count,
        "close_agent_count": close_agent_count,
        "off_lane_count": off_lane_count,
    }


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("RECTOR Rule-Related Metrics")
    print("=" * 60)

    # Load model
    print("\nLoading model...")
    model = RuleAwareGenerator(
        embed_dim=256,
        decoder_hidden_dim=256,
        decoder_num_layers=4,
        latent_dim=64,
        num_modes=6,
        dropout=0.1,
        use_m2i_encoder=True,
        m2i_checkpoint="/workspace/models/pretrained/m2i/models/relation_v2v/model.25.bin",
        freeze_m2i=True,
        trajectory_length=50,
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"  Loaded checkpoint: {args.checkpoint}")

    # Load data
    print("\nLoading validation data...")
    val_files = sorted(glob.glob(os.path.join(args.val_dir, "*")))

    if not val_files:
        print(f"ERROR: No files found in {args.val_dir}")
        return

    print(f"  Found {len(val_files)} files")

    dataset = WaymoDataset(val_files, is_training=False)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    # Metrics loop
    print("\nComputing metrics...")
    start_time = time.time()

    totals = {
        "total": 0,
        "miss_count": 0,
        "close_agent_count": 0,
        "off_lane_count": 0,
    }
    batch_count = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if not batch:
                continue

            if args.max_batches and batch_idx >= args.max_batches:
                break

            ego_history = batch["ego_history"].to(device)
            agent_states = batch["agent_states"].to(device)
            lane_centers = batch["lane_centers"].to(device)
            traj_gt = batch["traj_gt"].to(device)

            outputs = model(
                ego_history=ego_history,
                agent_states=agent_states,
                lane_centers=lane_centers,
                traj_gt=traj_gt,
            )

            batch_metrics = compute_rule_metrics(outputs, batch, device)

            for key in totals:
                totals[key] += batch_metrics[key]

            batch_count += 1

            if (batch_idx + 1) % 50 == 0:
                elapsed = time.time() - start_time
                print(
                    f"  Batch {batch_idx + 1}: {totals['total']} samples ({elapsed:.1f}s)"
                )

    elapsed = time.time() - start_time

    # Compute rates
    N = totals["total"]
    miss_rate = 100.0 * totals["miss_count"] / N
    close_agent_rate = 100.0 * totals["close_agent_count"] / N
    off_lane_rate = 100.0 * totals["off_lane_count"] / N

    print("\n" + "=" * 60)
    print("RULE-RELATED METRICS (REAL DATA)")
    print("=" * 60)
    print(f"\nTotal samples: {N}")
    print(f"Time elapsed: {elapsed:.1f}s")
    print(f"\n--- Safety Tier Proxies ---")
    print(f"  Miss Rate (FDE > 2m):        {miss_rate:.2f}%")
    print(f"  Close Agent (< 2m):          {close_agent_rate:.2f}%")
    print(f"\n--- Road Tier Proxies ---")
    print(f"  Off-Lane (> 3m from lane):   {off_lane_rate:.2f}%")

    # Map to tier violations (using these as proxies)
    # close_agent_rate is a Safety proxy (agent proximity), not Comfort.
    tier_violation_rates = {
        "Safety": max(
            miss_rate, close_agent_rate
        ),  # Miss rate + proximity both safety concerns
        "Legal": 0.0,  # Would need traffic signals - not available
        "Road": off_lane_rate,  # Lane deviation
        "Comfort": 0.0,  # No comfort-specific proxy computed here
    }

    print(f"\n--- Mapped to Tier Violation Rates ---")
    for tier, rate in tier_violation_rates.items():
        print(f"  {tier:10s}: {rate:.2f}%")

    # Save results
    results = {
        "checkpoint": args.checkpoint,
        "total_samples": N,
        "batches_evaluated": batch_count,
        "elapsed_time": elapsed,
        "raw_metrics": {
            "miss_rate": miss_rate,
            "close_agent_rate": close_agent_rate,
            "off_lane_rate": off_lane_rate,
        },
        "tier_violation_rates": tier_violation_rates,
        "note": (
            "Proxy metrics from trajectory predictions. "
            "miss_rate and close_agent_rate are Safety-tier proxies "
            "(close_agent uses last history position of each agent as static reference). "
            "off_lane_rate is a Road-tier proxy (midpoint of each lane polyline, "
            "checked across the full predicted trajectory). "
            "Legal and Comfort tiers require traffic-signal/kinematic data not available here."
        ),
    }

    results_path = output_dir / "real_rule_metrics.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
