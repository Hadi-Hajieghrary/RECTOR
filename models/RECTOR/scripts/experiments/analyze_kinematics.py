#!/usr/bin/env python3
"""
Debug script to analyze trajectory kinematics from RECTOR predictions.
"""

import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import glob
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, "/workspace/models/RECTOR/scripts")
sys.path.insert(0, "/workspace/data/WOMD")
sys.path.insert(0, "/workspace/data")

from training.train_rector import WaymoDataset, collate_fn, TRAJECTORY_SCALE
from models.rule_aware_generator import RuleAwareGenerator


def analyze_kinematics():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    print("Loading model...")
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

    checkpoint = torch.load(
        "/workspace/models/RECTOR/models/best.pt", map_location=device
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load data
    val_dir = "/workspace/data/WOMD/datasets/waymo_open_dataset/motion_v_1_3_0/processed/augmented/scenario/validation_interactive"
    val_files = sorted(glob.glob(os.path.join(val_dir, "*")))[:5]
    dataset = WaymoDataset(val_files, is_training=False)
    loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn, num_workers=2)

    all_max_accels = []
    all_max_speeds = []
    all_max_jerks = []

    print("\nAnalyzing trajectory kinematics...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if not batch or batch_idx >= 5:
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

            pred_traj = outputs["trajectories"]  # [B, M, T, 4]
            B, M, T, _ = pred_traj.shape

            # Get best trajectory
            ade_per_mode = (
                (pred_traj[:, :, :, :2] - traj_gt[:, :, :2].unsqueeze(1))
                .norm(dim=-1)
                .mean(dim=-1)
            )
            best_mode_idx = ade_per_mode.argmin(dim=1)
            best_traj = pred_traj[
                torch.arange(B, device=device), best_mode_idx
            ]  # [B, T, 4]

            for b in range(B):
                traj = best_traj[b]  # [T, 4]

                # Method 1: Raw normalized coordinates
                positions_norm = traj[:, :2]  # Already normalized

                # Method 2: Scale to meters
                positions_meters = traj[:, :2] * TRAJECTORY_SCALE  # Convert to meters

                # Compute velocities using normalized positions first
                # Then scale to real units
                # Velocity: dx/dt where dt = 0.1s (10Hz)
                # In normalized: d(pos/50)/dt = d(pos)/dt / 50
                # So: vel_meters = (pos_norm[t+1] - pos_norm[t]) / 0.1 * 50
                #                = delta_norm * 10 * 50 = delta_norm * 500

                delta_norm = (
                    positions_norm[1:] - positions_norm[:-1]
                )  # In normalized units
                velocities = delta_norm * 500  # m/s (10Hz * scale)
                speeds = velocities.norm(dim=-1)

                # Accelerations
                if velocities.shape[0] > 1:
                    accel = (velocities[1:] - velocities[:-1]) * 10  # m/s^2
                    accel_mags = accel.norm(dim=-1)

                    if accel_mags.shape[0] > 1:
                        jerks = (accel[1:] - accel[:-1]) * 10
                        jerk_mags = jerks.norm(dim=-1)
                        all_max_jerks.append(jerk_mags.max().item())

                    all_max_accels.append(accel_mags.max().item())

                all_max_speeds.append(speeds.max().item())

    print("\n" + "=" * 60)
    print("TRAJECTORY KINEMATICS ANALYSIS")
    print("=" * 60)
    print(f"TRAJECTORY_SCALE: {TRAJECTORY_SCALE}")
    print(f"Samples analyzed: {len(all_max_speeds)}")

    print("\n--- Max Speed (m/s) ---")
    print(f"  Mean: {np.mean(all_max_speeds):.2f}")
    print(f"  Std:  {np.std(all_max_speeds):.2f}")
    print(f"  Min:  {np.min(all_max_speeds):.2f}")
    print(f"  Max:  {np.max(all_max_speeds):.2f}")
    print(f"  P95:  {np.percentile(all_max_speeds, 95):.2f}")

    print("\n--- Max Acceleration (m/s^2) ---")
    print(f"  Mean: {np.mean(all_max_accels):.2f}")
    print(f"  Std:  {np.std(all_max_accels):.2f}")
    print(f"  Min:  {np.min(all_max_accels):.2f}")
    print(f"  Max:  {np.max(all_max_accels):.2f}")
    print(f"  P95:  {np.percentile(all_max_accels, 95):.2f}")
    print(f"  P99:  {np.percentile(all_max_accels, 99):.2f}")

    if all_max_jerks:
        print("\n--- Max Jerk (m/s^3) ---")
        print(f"  Mean: {np.mean(all_max_jerks):.2f}")
        print(f"  P95:  {np.percentile(all_max_jerks, 95):.2f}")

    # Suggest thresholds
    print("\n--- Suggested Thresholds ---")
    print(
        f"  Comfort accel threshold (P90): {np.percentile(all_max_accels, 90):.2f} m/s^2"
    )
    print(
        f"  Safety accel threshold (P99):  {np.percentile(all_max_accels, 99):.2f} m/s^2"
    )


if __name__ == "__main__":
    analyze_kinematics()
