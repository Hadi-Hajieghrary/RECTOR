#!/usr/bin/env python3
"""
RECTOR V2 Visualization Script.

Generates bird's-eye-view visualizations of trajectory predictions.

Usage:
    python visualization/visualize_predictions.py --checkpoint /path/to/best.pt --num_scenarios 5
"""

import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import argparse
import glob
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, "/workspace/data/WOMD")
sys.path.insert(0, "/workspace/data")

from training.train_rector import WaymoDataset, collate_fn, TRAJECTORY_SCALE
from models.rule_aware_generator import RuleAwareGenerator


# Visualization colors
COLORS = {
    "ego_history": "#3498db",  # Blue
    "ego_gt": "#2ecc71",  # Green
    "pred_best": "#e74c3c",  # Red
    "pred_other": "#f39c12",  # Orange
    "agents": "#9b59b6",  # Purple
    "lanes": "#7f8c8d",  # Gray
}


def parse_args():
    parser = argparse.ArgumentParser(description="RECTOR V2 Visualization")

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
        "--output_dir",
        type=str,
        default="/workspace/output/visualizations/rector_v2",
        help="Output directory",
    )
    parser.add_argument(
        "--num_scenarios", type=int, default=10, help="Number of scenarios to visualize"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for scenario selection"
    )

    # Model config
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--decoder_hidden_dim", type=int, default=256)
    parser.add_argument("--decoder_num_layers", type=int, default=4)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--num_modes", type=int, default=6)
    parser.add_argument(
        "--m2i_checkpoint",
        type=str,
        default="/workspace/models/pretrained/m2i/models/relation_v2v/model.25.bin",
    )

    return parser.parse_args()


def visualize_scenario(
    ego_history: np.ndarray,  # [11, 4]
    agent_states: np.ndarray,  # [32, 11, 4]
    lane_centers: np.ndarray,  # [64, 20, 2]
    traj_gt: np.ndarray,  # [50, 4]
    trajectories: np.ndarray,  # [6, 50, 4]
    confidence: np.ndarray,  # [6]
    scenario_idx: int,
    output_path: str,
    trajectory_scale: float = TRAJECTORY_SCALE,
):
    """Create a bird's-eye-view visualization."""

    # Convert from normalized to meters
    ego_hist_m = ego_history[:, :2] * trajectory_scale
    traj_gt_m = traj_gt[:, :2] * trajectory_scale
    trajs_m = trajectories[:, :, :2] * trajectory_scale
    agents_m = agent_states[:, :, :2] * trajectory_scale
    lanes_m = lane_centers * trajectory_scale

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 12), dpi=100)

    # Plot lanes
    for lane_idx in range(lanes_m.shape[0]):
        lane = lanes_m[lane_idx]
        # Check if lane is valid (non-zero)
        if np.abs(lane).max() > 0.01:
            ax.plot(
                lane[:, 0], lane[:, 1], color=COLORS["lanes"], linewidth=1, alpha=0.5
            )

    # Plot agent histories
    for agent_idx in range(min(10, agents_m.shape[0])):  # Show first 10 agents
        agent = agents_m[agent_idx]
        # Check if agent is valid
        if np.abs(agent).max() > 0.01:
            ax.plot(
                agent[:, 0],
                agent[:, 1],
                color=COLORS["agents"],
                linewidth=1.5,
                alpha=0.6,
                marker="o",
                markersize=2,
            )
            # Mark current position
            ax.scatter(
                agent[-1, 0], agent[-1, 1], color=COLORS["agents"], s=50, zorder=5
            )

    # Plot ego history
    ax.plot(
        ego_hist_m[:, 0],
        ego_hist_m[:, 1],
        color=COLORS["ego_history"],
        linewidth=2.5,
        marker="o",
        markersize=4,
        label="Ego History",
        zorder=10,
    )

    # Plot ground truth future
    ax.plot(
        traj_gt_m[:, 0],
        traj_gt_m[:, 1],
        color=COLORS["ego_gt"],
        linewidth=2.5,
        linestyle="--",
        marker="s",
        markersize=3,
        label="Ground Truth",
        zorder=11,
    )

    # Plot all predicted modes
    best_mode = confidence.argmax()

    for mode_idx in range(trajectories.shape[0]):
        traj = trajs_m[mode_idx]
        conf = confidence[mode_idx]

        if mode_idx == best_mode:
            color = COLORS["pred_best"]
            linewidth = 3
            alpha = 1.0
            label = f"Best Mode (conf={conf:.2f})"
            zorder = 15
        else:
            color = COLORS["pred_other"]
            linewidth = 1.5
            alpha = 0.4 + 0.3 * (conf / confidence.max())
            label = None
            zorder = 12

        ax.plot(
            traj[:, 0],
            traj[:, 1],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            label=label,
            zorder=zorder,
        )

        # Mark endpoint
        ax.scatter(
            traj[-1, 0],
            traj[-1, 1],
            color=color,
            s=80,
            marker="*",
            alpha=alpha,
            zorder=zorder + 1,
        )

    # Compute metrics for best mode
    best_traj = trajs_m[best_mode]
    ade = np.linalg.norm(best_traj - traj_gt_m, axis=1).mean()
    fde = np.linalg.norm(best_traj[-1] - traj_gt_m[-1])

    # Set axis properties
    all_x = np.concatenate(
        [ego_hist_m[:, 0], traj_gt_m[:, 0], trajs_m.reshape(-1, 2)[:, 0]]
    )
    all_y = np.concatenate(
        [ego_hist_m[:, 1], traj_gt_m[:, 1], trajs_m.reshape(-1, 2)[:, 1]]
    )

    margin = 10
    x_min, x_max = all_x.min() - margin, all_x.max() + margin
    y_min, y_max = all_y.min() - margin, all_y.max() + margin

    # Make square
    x_range = x_max - x_min
    y_range = y_max - y_min
    max_range = max(x_range, y_range)

    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2

    ax.set_xlim(x_center - max_range / 2, x_center + max_range / 2)
    ax.set_ylim(y_center - max_range / 2, y_center + max_range / 2)

    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=10)

    ax.set_xlabel("X (meters)", fontsize=12)
    ax.set_ylabel("Y (meters)", fontsize=12)
    ax.set_title(
        f"RECTOR V2 Prediction - Scenario {scenario_idx}\n"
        f"Best Mode ADE: {ade:.2f}m | FDE: {fde:.2f}m | Confidence: {confidence[best_mode]:.2f}",
        fontsize=14,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return ade, fde


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("RECTOR V2 Visualization")
    print("=" * 60)

    # Load model
    print("\nLoading model...")
    model = RuleAwareGenerator(
        embed_dim=args.embed_dim,
        decoder_hidden_dim=args.decoder_hidden_dim,
        decoder_num_layers=args.decoder_num_layers,
        latent_dim=args.latent_dim,
        num_modes=args.num_modes,
        use_m2i_encoder=True,
        m2i_checkpoint=args.m2i_checkpoint,
        freeze_m2i=True,
        trajectory_length=50,
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"  Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")

    # Load data
    print("\nLoading data...")
    val_files = sorted(glob.glob(os.path.join(args.val_dir, "*")))

    if not val_files:
        print(f"ERROR: No files found in {args.val_dir}")
        return

    # Sample random files
    sample_files = random.sample(val_files, min(10, len(val_files)))
    print(f"  Using {len(sample_files)} random files")

    dataset = WaymoDataset(sample_files, is_training=False)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, collate_fn=collate_fn, num_workers=0
    )

    # Generate visualizations
    print(f"\nGenerating {args.num_scenarios} visualizations...")

    all_ade = []
    all_fde = []
    scenario_idx = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if not batch or scenario_idx >= args.num_scenarios:
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
            )

            # Get predictions
            trajectories = outputs["trajectories"][0].cpu().numpy()  # [6, 50, 4]
            confidence = outputs["confidence"][0].cpu().numpy()  # [6]

            # Apply softmax to confidence
            confidence = np.exp(confidence) / np.exp(confidence).sum()

            # Visualize
            output_path = output_dir / f"scenario_{scenario_idx:03d}.png"

            ade, fde = visualize_scenario(
                ego_history=ego_history[0].cpu().numpy(),
                agent_states=agent_states[0].cpu().numpy(),
                lane_centers=lane_centers[0].cpu().numpy(),
                traj_gt=traj_gt[0].cpu().numpy(),
                trajectories=trajectories,
                confidence=confidence,
                scenario_idx=scenario_idx,
                output_path=str(output_path),
            )

            all_ade.append(ade)
            all_fde.append(fde)

            print(
                f"  Scenario {scenario_idx}: ADE={ade:.2f}m, FDE={fde:.2f}m → {output_path.name}"
            )
            scenario_idx += 1

    # Summary
    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETE")
    print("=" * 60)
    print(f"\n  Scenarios visualized: {len(all_ade)}")
    print(f"  Average ADE: {np.mean(all_ade):.3f}m")
    print(f"  Average FDE: {np.mean(all_fde):.3f}m")
    print(f"\n  Output directory: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
