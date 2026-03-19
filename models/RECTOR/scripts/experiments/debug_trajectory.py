#!/usr/bin/env python3
"""
Debug: Check trajectory output format
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


def debug():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
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

    # Load one batch
    val_dir = "/workspace/data/WOMD/datasets/waymo_open_dataset/motion_v_1_3_0/processed/augmented/scenario/validation_interactive"
    val_files = sorted(glob.glob(os.path.join(val_dir, "*")))[:1]
    dataset = WaymoDataset(val_files, is_training=False)
    loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn, num_workers=0)

    with torch.no_grad():
        for batch in loader:
            if not batch:
                continue

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
            B, M, T, D = pred_traj.shape

            print(f"Predicted trajectory shape: {pred_traj.shape}")
            print(f"GT trajectory shape: {traj_gt.shape}")

            # Sample values
            sample = pred_traj[0, 0]  # First sample, first mode
            gt_sample = traj_gt[0]

            print(f"\n--- Predicted trajectory (first mode, first 5 steps) ---")
            print(f"Normalized: {sample[:5, :2]}")
            print(f"In meters:  {sample[:5, :2] * TRAJECTORY_SCALE}")

            print(f"\n--- GT trajectory (first 5 steps) ---")
            print(f"Normalized: {gt_sample[:5, :2]}")
            print(f"In meters:  {gt_sample[:5, :2] * TRAJECTORY_SCALE}")

            # Compute displacements
            pred_disp = sample[1:, :2] - sample[:-1, :2]
            gt_disp = gt_sample[1:, :2] - gt_sample[:-1, :2]

            print(f"\n--- Displacements (first 5 steps, meters) ---")
            print(f"Pred: {pred_disp[:5] * TRAJECTORY_SCALE}")
            print(f"GT:   {gt_disp[:5] * TRAJECTORY_SCALE}")

            # Compute accelerations
            pred_vel = pred_disp * TRAJECTORY_SCALE * 10  # m/s
            gt_vel = gt_disp * TRAJECTORY_SCALE * 10

            pred_accel = (pred_vel[1:] - pred_vel[:-1]) * 10  # m/s^2
            gt_accel = (gt_vel[1:] - gt_vel[:-1]) * 10

            pred_accel_mag = pred_accel.norm(dim=-1)
            gt_accel_mag = gt_accel.norm(dim=-1)

            print(f"\n--- Acceleration (m/s^2) ---")
            print(
                f"Pred mean: {pred_accel_mag.mean().item():.2f}, max: {pred_accel_mag.max().item():.2f}"
            )
            print(
                f"GT mean:   {gt_accel_mag.mean().item():.2f}, max: {gt_accel_mag.max().item():.2f}"
            )
            print(f"GT P90:    {torch.quantile(gt_accel_mag, 0.9).item():.2f}")
            print(f"Pred P90:  {torch.quantile(pred_accel_mag, 0.9).item():.2f}")

            # Check all samples in batch
            all_pred_accels = []
            all_gt_accels = []
            for bidx in range(B):
                for midx in range(M):
                    pt = pred_traj[bidx, midx, :, :2]
                    pd = (pt[1:] - pt[:-1]) * TRAJECTORY_SCALE * 10
                    pa = (pd[1:] - pd[:-1]) * 10
                    all_pred_accels.append(pa.norm(dim=-1).max().item())

                gt_t = traj_gt[bidx, :, :2]
                gd = (gt_t[1:] - gt_t[:-1]) * TRAJECTORY_SCALE * 10
                ga = (gd[1:] - gd[:-1]) * 10
                all_gt_accels.append(ga.norm(dim=-1).max().item())

            print(f"\n--- Max accel across batch ---")
            print(
                f"GT:   mean={np.mean(all_gt_accels):.2f}, max={np.max(all_gt_accels):.2f}"
            )
            print(
                f"Pred: mean={np.mean(all_pred_accels):.2f}, max={np.max(all_pred_accels):.2f}"
            )

            break


if __name__ == "__main__":
    debug()
