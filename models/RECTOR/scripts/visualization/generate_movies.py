#!/usr/bin/env python3
"""
RECTOR V2 Movie Generator.

Generates animated bird's-eye-view movies showing trajectory predictions.

Usage:
    python visualization/generate_movies.py --num_scenarios 5
    python visualization/generate_movies.py --checkpoint /path/to/best.pt --output_dir /path/to/output
"""

import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import argparse
import glob
import random
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from matplotlib.patches import FancyArrowPatch

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, "/workspace/data/WOMD")
sys.path.insert(0, "/workspace/data")

from training.train_rector import WaymoDataset, collate_fn, TRAJECTORY_SCALE
from models.rule_aware_generator import RuleAwareGenerator


# Colors
COLORS = {
    "ego": "#e74c3c",  # Red
    "ego_history": "#3498db",  # Blue
    "ego_gt": "#2ecc71",  # Green
    "pred_best": "#e74c3c",  # Red
    "pred_other": "#f39c12",  # Orange
    "agents": "#9b59b6",  # Purple
    "lanes": "#7f8c8d",  # Gray
    "road_edge": "#2c3e50",  # Dark
}


def parse_args():
    parser = argparse.ArgumentParser(description="RECTOR V2 Movie Generator")

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
        default="/workspace/models/RECTOR/movies",
        help="Output directory for movies",
    )
    parser.add_argument(
        "--num_scenarios", type=int, default=5, help="Number of scenarios to generate"
    )
    parser.add_argument("--fps", type=int, default=10, help="Frames per second")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--format",
        type=str,
        default="mp4",
        choices=["mp4", "gif", "both"],
        help="Output format",
    )

    # Model config
    parser.add_argument(
        "--m2i_checkpoint",
        type=str,
        default="/workspace/models/pretrained/m2i/models/relation_v2v/model.25.bin",
    )

    return parser.parse_args()


def draw_vehicle(
    ax, x, y, heading, length=4.5, width=2.0, color="red", alpha=1.0, zorder=10
):
    """Draw a vehicle as a rotated rectangle with heading arrow."""
    # Create rectangle centered at origin
    rect = patches.FancyBboxPatch(
        (-length / 2, -width / 2),
        length,
        width,
        boxstyle="round,pad=0.05,rounding_size=0.3",
        facecolor=color,
        edgecolor="black",
        linewidth=1,
        alpha=alpha,
        zorder=zorder,
    )

    # Apply rotation and translation
    t = (
        plt.matplotlib.transforms.Affine2D().rotate(heading).translate(x, y)
        + ax.transData
    )
    rect.set_transform(t)
    ax.add_patch(rect)

    # Draw heading arrow
    arrow_length = length * 0.6
    dx = arrow_length * np.cos(heading)
    dy = arrow_length * np.sin(heading)
    ax.arrow(
        x,
        y,
        dx,
        dy,
        head_width=0.5,
        head_length=0.3,
        fc="white",
        ec="black",
        zorder=zorder + 1,
        alpha=alpha,
    )


def generate_scenario_movie(
    ego_history: np.ndarray,  # [11, 4] - x, y, heading, speed
    agent_states: np.ndarray,  # [32, 11, 4]
    lane_centers: np.ndarray,  # [64, 20, 2]
    traj_gt: np.ndarray,  # [50, 4]
    trajectories: np.ndarray,  # [6, 50, 4]
    confidence: np.ndarray,  # [6]
    scenario_idx: int,
    output_path: str,
    fps: int = 10,
    save_gif: bool = False,
):
    """Generate an animated movie for a scenario."""

    # Convert to meters
    ego_hist_m = ego_history.copy()
    ego_hist_m[:, :2] *= TRAJECTORY_SCALE

    traj_gt_m = traj_gt.copy()
    traj_gt_m[:, :2] *= TRAJECTORY_SCALE

    trajs_m = trajectories.copy()
    trajs_m[:, :, :2] *= TRAJECTORY_SCALE

    agents_m = agent_states.copy()
    agents_m[:, :, :2] *= TRAJECTORY_SCALE

    lanes_m = lane_centers * TRAJECTORY_SCALE

    # Best mode
    best_mode = confidence.argmax()
    best_traj = trajs_m[best_mode]

    # Compute metrics
    ade = np.linalg.norm(best_traj[:, :2] - traj_gt_m[:, :2], axis=1).mean()
    fde = np.linalg.norm(best_traj[-1, :2] - traj_gt_m[-1, :2])

    # Determine plot bounds
    all_x = np.concatenate(
        [ego_hist_m[:, 0], traj_gt_m[:, 0], trajs_m.reshape(-1, 4)[:, 0]]
    )
    all_y = np.concatenate(
        [ego_hist_m[:, 1], traj_gt_m[:, 1], trajs_m.reshape(-1, 4)[:, 1]]
    )

    margin = 15
    x_center = (all_x.min() + all_x.max()) / 2
    y_center = (all_y.min() + all_y.max()) / 2
    max_range = max(all_x.max() - all_x.min(), all_y.max() - all_y.min()) + 2 * margin
    max_range = max(max_range, 60)  # Minimum 60m view

    # Total frames: history (11) + future (50) = 61
    history_frames = 11
    future_frames = 50
    total_frames = history_frames + future_frames

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=100)

    def init():
        ax.clear()
        return []

    def animate(frame):
        ax.clear()

        # Determine current time phase
        if frame < history_frames:
            # Still in history phase
            current_hist_idx = frame
            current_future_idx = -1
            phase = "History"
            time_label = f"t = {(frame - 10) * 0.1:.1f}s"
        else:
            # In prediction phase
            current_hist_idx = history_frames - 1  # Show all history
            current_future_idx = frame - history_frames
            phase = "Prediction"
            time_label = f"t = +{(current_future_idx + 1) * 0.1:.1f}s"

        # Draw lanes (static)
        for lane_idx in range(lanes_m.shape[0]):
            lane = lanes_m[lane_idx]
            if np.abs(lane).max() > 0.01:
                ax.plot(
                    lane[:, 0],
                    lane[:, 1],
                    color=COLORS["lanes"],
                    linewidth=1.5,
                    alpha=0.4,
                )

        # Draw agents up to current time
        for agent_idx in range(min(15, agents_m.shape[0])):
            agent = agents_m[agent_idx]
            if np.abs(agent).max() > 0.01:
                # Show trail
                trail_end = min(current_hist_idx + 1, history_frames)
                ax.plot(
                    agent[:trail_end, 0],
                    agent[:trail_end, 1],
                    color=COLORS["agents"],
                    linewidth=1,
                    alpha=0.3,
                )
                # Current position
                if current_hist_idx < history_frames:
                    ax.scatter(
                        agent[current_hist_idx, 0],
                        agent[current_hist_idx, 1],
                        color=COLORS["agents"],
                        s=30,
                        alpha=0.6,
                        zorder=5,
                    )

        # Draw ego history trail
        trail_end = min(current_hist_idx + 1, history_frames)
        ax.plot(
            ego_hist_m[:trail_end, 0],
            ego_hist_m[:trail_end, 1],
            color=COLORS["ego_history"],
            linewidth=2.5,
            alpha=0.7,
            marker="o",
            markersize=3,
            label="Ego History",
        )

        # Draw predictions (show all modes faded, best mode solid)
        if current_future_idx >= 0:
            # Show other modes (faded)
            for mode_idx in range(trajs_m.shape[0]):
                if mode_idx != best_mode:
                    traj = trajs_m[mode_idx]
                    ax.plot(
                        traj[: current_future_idx + 1, 0],
                        traj[: current_future_idx + 1, 1],
                        color=COLORS["pred_other"],
                        linewidth=1.5,
                        alpha=0.3,
                    )

            # Show best mode prediction up to current frame
            ax.plot(
                best_traj[: current_future_idx + 1, 0],
                best_traj[: current_future_idx + 1, 1],
                color=COLORS["pred_best"],
                linewidth=3,
                alpha=0.8,
                label=f"Prediction (ADE={ade:.2f}m)",
            )

            # Show ground truth up to current frame
            ax.plot(
                traj_gt_m[: current_future_idx + 1, 0],
                traj_gt_m[: current_future_idx + 1, 1],
                color=COLORS["ego_gt"],
                linewidth=2.5,
                linestyle="--",
                alpha=0.8,
                label="Ground Truth",
            )

        # Draw ego vehicle at current position
        if current_future_idx >= 0:
            # In prediction phase - show predicted position
            ego_x = best_traj[current_future_idx, 0]
            ego_y = best_traj[current_future_idx, 1]
            ego_h = best_traj[current_future_idx, 2]

            # Also show GT position
            gt_x = traj_gt_m[current_future_idx, 0]
            gt_y = traj_gt_m[current_future_idx, 1]
            gt_h = traj_gt_m[current_future_idx, 2]

            draw_vehicle(
                ax, gt_x, gt_y, gt_h, color=COLORS["ego_gt"], alpha=0.5, zorder=8
            )
            draw_vehicle(
                ax, ego_x, ego_y, ego_h, color=COLORS["ego"], alpha=0.9, zorder=10
            )
        else:
            # In history phase
            ego_x = ego_hist_m[current_hist_idx, 0]
            ego_y = ego_hist_m[current_hist_idx, 1]
            ego_h = ego_hist_m[current_hist_idx, 2]
            draw_vehicle(
                ax, ego_x, ego_y, ego_h, color=COLORS["ego"], alpha=0.9, zorder=10
            )

        # Set axis properties
        ax.set_xlim(x_center - max_range / 2, x_center + max_range / 2)
        ax.set_ylim(y_center - max_range / 2, y_center + max_range / 2)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=9)

        ax.set_xlabel("X (meters)", fontsize=11)
        ax.set_ylabel("Y (meters)", fontsize=11)
        ax.set_title(
            f"RECTOR V2 Trajectory Prediction - Scenario {scenario_idx}\n"
            f"{phase}: {time_label} | FDE: {fde:.2f}m",
            fontsize=12,
        )

        # Add frame counter
        ax.text(
            0.98,
            0.02,
            f"Frame {frame+1}/{total_frames}",
            transform=ax.transAxes,
            fontsize=9,
            ha="right",
            va="bottom",
            alpha=0.7,
        )

        return []

    # Create animation
    anim = animation.FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames=total_frames,
        interval=1000 // fps,
        blit=False,
    )

    # Save as MP4
    if output_path.endswith(".mp4"):
        writer = animation.FFMpegWriter(fps=fps, bitrate=2000)
        anim.save(output_path, writer=writer)
        print(f"    Saved: {output_path}")

    # Optionally save as GIF
    if save_gif:
        gif_path = output_path.replace(".mp4", ".gif")
        writer = animation.PillowWriter(fps=fps)
        anim.save(gif_path, writer=writer)
        print(f"    Saved: {gif_path}")

    plt.close(fig)

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
    print("RECTOR V2 Movie Generator")
    print("=" * 60)

    # Load model
    print("\nLoading model...")
    model = RuleAwareGenerator(
        embed_dim=256,
        decoder_hidden_dim=256,
        decoder_num_layers=4,
        latent_dim=64,
        num_modes=6,
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
    sample_files = random.sample(val_files, min(10, len(val_files)))
    print(f"  Using {len(sample_files)} random files")

    dataset = WaymoDataset(sample_files, is_training=False)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, collate_fn=collate_fn, num_workers=0
    )

    # Generate movies
    print(f"\nGenerating {args.num_scenarios} movies...")
    print(f"  FPS: {args.fps}")
    print(f"  Format: {args.format}")
    print(f"  Output: {output_dir}")

    save_gif = args.format in ["gif", "both"]
    scenario_idx = 0
    all_ade = []
    all_fde = []

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
            trajectories = outputs["trajectories"][0].cpu().numpy()
            confidence = outputs["confidence"][0].cpu().numpy()
            confidence = np.exp(confidence) / np.exp(confidence).sum()  # softmax

            # Generate movie
            output_path = str(output_dir / f"scenario_{scenario_idx:03d}.mp4")

            print(f"\n  Scenario {scenario_idx}:")
            ade, fde = generate_scenario_movie(
                ego_history=ego_history[0].cpu().numpy(),
                agent_states=agent_states[0].cpu().numpy(),
                lane_centers=lane_centers[0].cpu().numpy(),
                traj_gt=traj_gt[0].cpu().numpy(),
                trajectories=trajectories,
                confidence=confidence,
                scenario_idx=scenario_idx,
                output_path=output_path,
                fps=args.fps,
                save_gif=save_gif,
            )

            all_ade.append(ade)
            all_fde.append(fde)
            print(f"    Metrics: ADE={ade:.2f}m, FDE={fde:.2f}m")

            scenario_idx += 1

    # Summary
    print("\n" + "=" * 60)
    print("MOVIE GENERATION COMPLETE")
    print("=" * 60)
    print(f"\n  Movies generated: {len(all_ade)}")
    print(f"  Average ADE: {np.mean(all_ade):.3f}m")
    print(f"  Average FDE: {np.mean(all_fde):.3f}m")
    print(f"\n  Output directory: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
