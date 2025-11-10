#!/usr/bin/env python3
"""
Simple Trajectory Visualization for Preprocessed Data

This script visualizes trajectories from preprocessed numpy/pickle files,
useful for quickly checking data or predictions without full Waymo API.

Usage:
    python scripts/visualize_simple.py --data_file scenario.npz --output viz.png
    python scripts/visualize_simple.py --data_file scenario.npz --animate --output movie.mp4
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
import numpy as np


def plot_trajectory_static(
    history: np.ndarray,
    future_gt: np.ndarray,
    future_pred: Optional[np.ndarray] = None,
    map_lanes: Optional[np.ndarray] = None,
    other_agents: Optional[np.ndarray] = None,
    output_path: Optional[str] = None,
    title: str = "Trajectory Visualization",
):
    """
    Create static trajectory plot.
    
    Args:
        history: (T_history, 2) past trajectory
        future_gt: (T_future, 2) ground truth future
        future_pred: (T_future, 2) predicted future (optional)
        map_lanes: List of lane polylines (optional)
        other_agents: (N, T, 2) other agent trajectories (optional)
        output_path: Path to save figure
        title: Plot title
    """
    
    fig, ax = plt.subplots(figsize=(12, 12), dpi=100)
    
    # Plot map lanes if available
    if map_lanes is not None:
        for lane in map_lanes:
            if len(lane) > 1:
                ax.plot(lane[:, 0], lane[:, 1], 'k-', linewidth=0.5, alpha=0.3, zorder=1)
    
    # Plot other agents if available
    if other_agents is not None:
        for agent_traj in other_agents:
            ax.plot(agent_traj[:, 0], agent_traj[:, 1], 
                   color='gray', linewidth=1.0, alpha=0.3, zorder=2)
    
    # Plot history
    ax.plot(history[:, 0], history[:, 1], 'b-', linewidth=2.5, label='History', zorder=5)
    ax.scatter(history[0, 0], history[0, 1], c='blue', s=150, marker='o', 
              label='Start', zorder=6, edgecolor='white', linewidth=2)
    
    # Plot ground truth future
    ax.plot(future_gt[:, 0], future_gt[:, 1], 'g--', linewidth=2.5, label='Ground Truth', zorder=5)
    ax.scatter(future_gt[-1, 0], future_gt[-1, 1], c='green', s=150, marker='*',
              label='GT End', zorder=6, edgecolor='white', linewidth=2)
    
    # Plot prediction if available
    if future_pred is not None:
        ax.plot(future_pred[:, 0], future_pred[:, 1], 'r:', linewidth=3.0, 
               label='Prediction', zorder=7)
        ax.scatter(future_pred[-1, 0], future_pred[-1, 1], c='red', s=150, marker='X',
                  label='Pred End', zorder=8, edgecolor='white', linewidth=2)
    
    # Styling
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlabel('X (meters)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Y (meters)', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=12, framealpha=0.9)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()
    
    plt.close()


def create_trajectory_animation(
    history: np.ndarray,
    future_gt: np.ndarray,
    future_pred: Optional[np.ndarray] = None,
    output_path: str = "trajectory_animation.mp4",
    fps: int = 10,
    title: str = "Trajectory Animation",
):
    """
    Create animated trajectory visualization.
    
    Args:
        history: (T_history, 2) past trajectory
        future_gt: (T_future, 2) ground truth future
        future_pred: (T_future, 2) predicted future (optional)
        output_path: Output file path
        fps: Frames per second
        title: Animation title
    """
    
    # Combine history and future
    T_history = len(history)
    T_future = len(future_gt)
    T_total = T_history + T_future
    
    full_traj_gt = np.concatenate([history, future_gt], axis=0)
    if future_pred is not None:
        full_traj_pred = np.concatenate([history, future_pred], axis=0)
    else:
        full_traj_pred = None
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 12), dpi=100)
    
    # Set limits
    all_points = full_traj_gt
    if full_traj_pred is not None:
        all_points = np.concatenate([all_points, full_traj_pred], axis=0)
    
    margin = 10
    x_min, x_max = all_points[:, 0].min() - margin, all_points[:, 0].max() + margin
    y_min, y_max = all_points[:, 1].min() - margin, all_points[:, 1].max() + margin
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    # Initialize plot elements
    hist_line, = ax.plot([], [], 'b-', linewidth=2.5, label='History', zorder=5)
    future_gt_line, = ax.plot([], [], 'g--', linewidth=2.5, label='Ground Truth', zorder=5)
    
    if full_traj_pred is not None:
        future_pred_line, = ax.plot([], [], 'r:', linewidth=3.0, label='Prediction', zorder=7)
    else:
        future_pred_line = None
    
    current_pos, = ax.plot([], [], 'ko', markersize=12, zorder=10, 
                           markeredgecolor='white', markeredgewidth=2)
    
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=14,
                       verticalalignment='top', bbox=dict(boxstyle='round', 
                       facecolor='wheat', alpha=0.8))
    
    # Styling
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlabel('X (meters)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Y (meters)', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', fontsize=12, framealpha=0.9)
    
    def update(frame):
        """Update function for animation."""
        
        if frame < T_history:
            # Still in history phase
            hist_line.set_data(history[:frame+1, 0], history[:frame+1, 1])
            future_gt_line.set_data([], [])
            if future_pred_line:
                future_pred_line.set_data([], [])
            current_pos.set_data([history[frame, 0]], [history[frame, 1]])
            time_text.set_text(f'History: {frame}/{T_history} | Future: 0/{T_future}')
        else:
            # In future phase
            future_frame = frame - T_history
            hist_line.set_data(history[:, 0], history[:, 1])
            future_gt_line.set_data(future_gt[:future_frame+1, 0], future_gt[:future_frame+1, 1])
            if future_pred_line and full_traj_pred is not None:
                future_pred_line.set_data(full_traj_pred[T_history:frame+1, 0], 
                                         full_traj_pred[T_history:frame+1, 1])
            current_pos.set_data([full_traj_gt[frame, 0]], [full_traj_gt[frame, 1]])
            time_text.set_text(f'History: {T_history}/{T_history} | Future: {future_frame+1}/{T_future}')
        
        return hist_line, future_gt_line, future_pred_line, current_pos, time_text
    
    # Create animation
    print(f"Creating animation with {T_total} frames...")
    anim = FuncAnimation(fig, update, frames=T_total, interval=1000/fps, blit=True)
    
    # Save
    writer_type = "pillow" if output_path.endswith('.gif') else "ffmpeg"
    if writer_type == "ffmpeg":
        writer = FFMpegWriter(fps=fps, bitrate=5000)
    else:
        writer = PillowWriter(fps=fps)
    
    anim.save(output_path, writer=writer)
    print(f"Animation saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize trajectories from preprocessed data")
    parser.add_argument("--data_file", required=True, help="Path to .npz or .pkl file")
    parser.add_argument("--output", default="visualization.png", help="Output file")
    parser.add_argument("--animate", action="store_true", help="Create animation instead of static plot")
    parser.add_argument("--fps", type=int, default=10, help="FPS for animation")
    parser.add_argument("--title", default="Trajectory Visualization", help="Plot title")
    
    args = parser.parse_args()
    
    # Load data
    data_path = Path(args.data_file)
    if not data_path.exists():
        print(f"File not found: {data_path}")
        return 1
    
    print(f"Loading: {data_path}")
    
    if data_path.suffix == '.npz':
        data = np.load(data_path, allow_pickle=True)
    elif data_path.suffix == '.pkl':
        import pickle
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
    else:
        print(f"Unsupported file type: {data_path.suffix}")
        return 1
    
    # Extract trajectories
    # Expected keys: 'history', 'future', 'prediction' (optional)
    if 'history' not in data or 'future' not in data:
        print("Data file must contain 'history' and 'future' keys")
        print(f"Available keys: {list(data.keys())}")
        return 1
    
    history = data['history']
    future_gt = data['future']
    future_pred = data.get('prediction', None)
    
    print(f"History shape: {history.shape}")
    print(f"Future GT shape: {future_gt.shape}")
    if future_pred is not None:
        print(f"Future pred shape: {future_pred.shape}")
    
    # Create visualization
    if args.animate:
        create_trajectory_animation(
            history=history,
            future_gt=future_gt,
            future_pred=future_pred,
            output_path=args.output,
            fps=args.fps,
            title=args.title,
        )
    else:
        plot_trajectory_static(
            history=history,
            future_gt=future_gt,
            future_pred=future_pred,
            output_path=args.output,
            title=args.title,
        )
    
    return 0


if __name__ == "__main__":
    sys.exit(main())