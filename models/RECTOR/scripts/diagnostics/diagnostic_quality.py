#!/usr/bin/env python3
"""
Diagnostic: Check raw model trajectory quality.
Shows whether model output is inherently zig-zag.
"""
import os, sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
sys.path.insert(0, str(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, "/workspace/data/WOMD")
sys.path.insert(0, "/workspace/data")

import glob, random
import numpy as np
import torch

from training.train_rector import WaymoDataset, collate_fn, TRAJECTORY_SCALE
from models.rule_aware_generator import RuleAwareGenerator


def trajectory_smoothness(traj_m):
    """Compute lateral jerk/oscillation metric for a trajectory in meters."""
    # Compute forward direction from first-to-last displacement
    dx_total = traj_m[-1, 0] - traj_m[0, 0]
    dy_total = traj_m[-1, 1] - traj_m[0, 1]
    fwd_angle = np.arctan2(dy_total, dx_total)

    # Rotate trajectory so forward is along x-axis
    cos_a, sin_a = np.cos(-fwd_angle), np.sin(-fwd_angle)
    rx = (traj_m[:, 0] - traj_m[0, 0]) * cos_a - (traj_m[:, 1] - traj_m[0, 1]) * sin_a
    ry = (traj_m[:, 0] - traj_m[0, 0]) * sin_a + (traj_m[:, 1] - traj_m[0, 1]) * cos_a

    # Lateral displacement changes
    lat_diff = np.diff(ry)

    # Count sign changes (zig-zag indicator)
    sign_changes = np.sum(np.diff(np.sign(lat_diff)) != 0)

    # Max lateral displacement
    max_lat = np.max(np.abs(ry))

    # RMS lateral velocity (m/s at 10Hz)
    lat_vel = np.abs(lat_diff) / 0.1
    rms_lat_vel = np.sqrt(np.mean(lat_vel**2))

    return {
        "sign_changes": sign_changes,
        "max_lateral_m": max_lat,
        "rms_lateral_vel": rms_lat_vel,
        "total_fwd_m": rx[-1],
    }


def main():
    device = torch.device("cpu")
    data_dir = "/workspace/data/WOMD/datasets/waymo_open_dataset/motion_v_1_3_0/processed/augmented/scenario/validation_interactive"

    # Load model
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
    ckpt = torch.load("/workspace/models/RECTOR/models/best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', 'unknown')}")

    # Load data
    random.seed(42)
    val_files = sorted(glob.glob(os.path.join(data_dir, "*")))
    sample_files = random.sample(val_files, min(5, len(val_files)))
    dataset = WaymoDataset(sample_files, is_training=False)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, collate_fn=collate_fn, num_workers=0
    )

    print(f"\n{'='*80}")
    print(
        f"{'Scenario':>8} {'Mode':>4} {'ADE(m)':>8} {'FDE(m)':>8} {'ZigZag':>8} {'MaxLat':>8} {'RMSLatV':>8} {'FwdDist':>8}"
    )
    print(f"{'='*80}")

    scenario_count = 0
    all_metrics = []

    with torch.no_grad():
        for batch in loader:
            if not batch or scenario_count >= 10:
                break

            ego_history = batch["ego_history"].to(device)
            agent_states = batch["agent_states"].to(device)
            lane_centers = batch["lane_centers"].to(device)
            traj_gt = batch["traj_gt"].to(device)

            outputs = model(
                ego_history=ego_history,
                agent_states=agent_states,
                lane_centers=lane_centers,
            )

            trajectories = outputs["trajectories"][0].cpu().numpy()  # [6, 50, 4]
            confidence = outputs["confidence"][0].cpu().numpy()
            confidence = np.exp(confidence) / np.exp(confidence).sum()
            best_mode = confidence.argmax()

            gt = traj_gt[0].cpu().numpy()  # [50, 4]
            gt_m = gt.copy()
            gt_m[:, :2] *= TRAJECTORY_SCALE

            # Analyze each mode
            for m in range(6):
                traj = trajectories[m].copy()
                traj_m = traj.copy()
                traj_m[:, :2] *= TRAJECTORY_SCALE

                ade = np.linalg.norm(traj_m[:, :2] - gt_m[:, :2], axis=1).mean()
                fde = np.linalg.norm(traj_m[-1, :2] - gt_m[-1, :2])

                smooth = trajectory_smoothness(traj_m)

                marker = " <-- BEST" if m == best_mode else ""
                print(
                    f"{scenario_count:>8} {m:>4} {ade:>8.2f} {fde:>8.2f} "
                    f"{smooth['sign_changes']:>8} {smooth['max_lateral_m']:>8.2f} "
                    f"{smooth['rms_lateral_vel']:>8.2f} {smooth['total_fwd_m']:>8.1f}{marker}"
                )

                if m == best_mode:
                    all_metrics.append({"ade": ade, "fde": fde, **smooth})

            # Also show GT smoothness
            gt_smooth = trajectory_smoothness(gt_m)
            print(
                f"{'GT':>8} {'':>4} {'':>8} {'':>8} "
                f"{gt_smooth['sign_changes']:>8} {gt_smooth['max_lateral_m']:>8.2f} "
                f"{gt_smooth['rms_lateral_vel']:>8.2f} {gt_smooth['total_fwd_m']:>8.1f}"
            )
            print()

            # Show first 10 waypoints of best mode vs GT in meters
            if scenario_count == 0:
                print(
                    f"\n  First 10 waypoints (meters) - Scenario 0, Mode {best_mode}:"
                )
                print(
                    f"  {'Step':>4} {'Pred_X':>8} {'Pred_Y':>8} {'GT_X':>8} {'GT_Y':>8}"
                )
                for t in range(10):
                    print(
                        f"  {t:>4} {traj_m[t,0]:>8.2f} {traj_m[t,1]:>8.2f} "
                        f"{gt_m[t,0]:>8.2f} {gt_m[t,1]:>8.2f}"
                    )
                print()

            scenario_count += 1

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY (best mode per scenario):")
    print(f"  Avg ADE: {np.mean([m['ade'] for m in all_metrics]):.2f}m")
    print(f"  Avg FDE: {np.mean([m['fde'] for m in all_metrics]):.2f}m")
    print(
        f"  Avg lateral sign changes: {np.mean([m['sign_changes'] for m in all_metrics]):.1f}"
    )
    print(
        f"  Avg max lateral: {np.mean([m['max_lateral_m'] for m in all_metrics]):.2f}m"
    )
    print(
        f"  Avg RMS lateral velocity: {np.mean([m['rms_lateral_vel'] for m in all_metrics]):.2f} m/s"
    )
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
