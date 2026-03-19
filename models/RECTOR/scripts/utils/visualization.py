"""
RECTOR Training Visualization Utilities.

Provides real-time and post-training visualization of:
- Loss curves (per tier, per loss component)
- Metric tracking (applicability accuracy, compliance rates)
- Trajectory quality visualization
- Training progress dashboard
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
from dataclasses import dataclass, field, asdict
import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.gridspec import GridSpec

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available, visualization disabled")


@dataclass
class TrainingMetrics:
    """Container for training metrics over time."""

    steps: List[int] = field(default_factory=list)
    epochs: List[int] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)

    # Loss components
    total_loss: List[float] = field(default_factory=list)
    applicability_loss: List[float] = field(default_factory=list)
    compliance_loss: List[float] = field(default_factory=list)
    recon_loss: List[float] = field(default_factory=list)
    kl_loss: List[float] = field(default_factory=list)
    temporal_loss: List[float] = field(default_factory=list)

    # Per-tier losses
    safety_loss: List[float] = field(default_factory=list)
    legal_loss: List[float] = field(default_factory=list)
    road_loss: List[float] = field(default_factory=list)
    comfort_loss: List[float] = field(default_factory=list)

    # Accuracy metrics
    applicability_accuracy: List[float] = field(default_factory=list)
    safety_compliance_rate: List[float] = field(default_factory=list)
    overall_compliance_rate: List[float] = field(default_factory=list)

    # Trajectory metrics
    ade: List[float] = field(default_factory=list)  # Average Displacement Error
    fde: List[float] = field(default_factory=list)  # Final Displacement Error

    # Learning rate
    learning_rate: List[float] = field(default_factory=list)

    def add_step(
        self,
        step: int,
        epoch: int,
        losses: Dict[str, float],
        metrics: Optional[Dict[str, float]] = None,
        lr: Optional[float] = None,
    ):
        """Add metrics for a single training step."""
        self.steps.append(step)
        self.epochs.append(epoch)
        self.timestamps.append(time.time())

        # Extract losses
        self.total_loss.append(losses.get("total_loss", 0.0))
        self.applicability_loss.append(losses.get("applicability_loss", 0.0))
        self.compliance_loss.append(losses.get("compliance_loss", 0.0))
        self.recon_loss.append(losses.get("recon_loss", 0.0))
        self.kl_loss.append(losses.get("kl_loss", 0.0))
        self.temporal_loss.append(losses.get("temporal_loss", 0.0))

        # Per-tier
        self.safety_loss.append(
            losses.get("safety_compliance_loss", losses.get("safety_app_loss", 0.0))
        )
        self.legal_loss.append(
            losses.get("legal_compliance_loss", losses.get("legal_app_loss", 0.0))
        )
        self.road_loss.append(
            losses.get("road_compliance_loss", losses.get("road_app_loss", 0.0))
        )
        self.comfort_loss.append(
            losses.get("comfort_compliance_loss", losses.get("comfort_app_loss", 0.0))
        )

        # Metrics
        if metrics:
            self.applicability_accuracy.append(metrics.get("app_accuracy", 0.0))
            self.safety_compliance_rate.append(metrics.get("safety_compliance", 0.0))
            self.overall_compliance_rate.append(metrics.get("overall_compliance", 0.0))
            self.ade.append(metrics.get("ade", 0.0))
            self.fde.append(metrics.get("fde", 0.0))
        else:
            # Pad with last value or 0
            self.applicability_accuracy.append(
                self.applicability_accuracy[-1] if self.applicability_accuracy else 0.0
            )
            self.safety_compliance_rate.append(
                self.safety_compliance_rate[-1] if self.safety_compliance_rate else 0.0
            )
            self.overall_compliance_rate.append(
                self.overall_compliance_rate[-1]
                if self.overall_compliance_rate
                else 0.0
            )
            self.ade.append(self.ade[-1] if self.ade else 0.0)
            self.fde.append(self.fde[-1] if self.fde else 0.0)

        self.learning_rate.append(
            lr if lr else (self.learning_rate[-1] if self.learning_rate else 0.0)
        )

    def save(self, path: str):
        """Save metrics to JSON."""
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "TrainingMetrics":
        """Load metrics from JSON."""
        with open(path, "r") as f:
            data = json.load(f)
        metrics = cls()
        for key, value in data.items():
            setattr(metrics, key, value)
        return metrics

    def get_smoothed(self, key: str, window: int = 100) -> np.ndarray:
        """Get smoothed values for a metric."""
        values = getattr(self, key, [])
        if len(values) < window:
            return np.array(values)
        kernel = np.ones(window) / window
        return np.convolve(values, kernel, mode="valid")


class TrainingVisualizer:
    """Real-time training visualization."""

    def __init__(
        self,
        log_dir: str,
        experiment_name: str = "rector_training",
        save_frequency: int = 100,
    ):
        """
        Initialize visualizer.

        Args:
            log_dir: Directory to save plots and metrics
            experiment_name: Name for this experiment
            save_frequency: How often to save plots (in steps)
        """
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        self.save_frequency = save_frequency

        # Create directories
        self.plots_dir = self.log_dir / "plots"
        self.metrics_dir = self.log_dir / "metrics"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        # Metrics containers
        self.train_metrics = TrainingMetrics()
        self.val_metrics = TrainingMetrics()
        self.test_metrics = TrainingMetrics()

        # Timing
        self.start_time = time.time()
        self.step_times: List[float] = []

    def log_train_step(
        self,
        step: int,
        epoch: int,
        losses: Dict[str, float],
        metrics: Optional[Dict[str, float]] = None,
        lr: Optional[float] = None,
    ):
        """Log training step."""
        self.train_metrics.add_step(step, epoch, losses, metrics, lr)

        # Track step time
        self.step_times.append(time.time())

        # Save periodically
        if step % self.save_frequency == 0:
            self._save_metrics()
            if MATPLOTLIB_AVAILABLE:
                self._save_plots()

    def log_validation(
        self,
        step: int,
        epoch: int,
        losses: Dict[str, float],
        metrics: Optional[Dict[str, float]] = None,
    ):
        """Log validation metrics."""
        self.val_metrics.add_step(step, epoch, losses, metrics)

    def log_test(
        self,
        step: int,
        epoch: int,
        losses: Dict[str, float],
        metrics: Optional[Dict[str, float]] = None,
    ):
        """Log test metrics."""
        self.test_metrics.add_step(step, epoch, losses, metrics)

    def _save_metrics(self):
        """Save all metrics to disk."""
        self.train_metrics.save(self.metrics_dir / "train_metrics.json")
        if self.val_metrics.steps:
            self.val_metrics.save(self.metrics_dir / "val_metrics.json")
        if self.test_metrics.steps:
            self.test_metrics.save(self.metrics_dir / "test_metrics.json")

    def _save_plots(self):
        """Generate and save all plots."""
        self.plot_loss_curves()
        self.plot_tier_losses()
        self.plot_metrics()
        self.plot_learning_rate()

    def plot_loss_curves(self, save: bool = True) -> Optional[Figure]:
        """Plot main loss curves."""
        if not MATPLOTLIB_AVAILABLE or not self.train_metrics.steps:
            return None

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f"{self.experiment_name} - Loss Curves", fontsize=14)

        steps = self.train_metrics.steps

        # Total loss
        ax = axes[0, 0]
        ax.plot(steps, self.train_metrics.total_loss, "b-", alpha=0.3, label="Train")
        if len(steps) > 50:
            smoothed = self.train_metrics.get_smoothed("total_loss", 50)
            ax.plot(steps[49:], smoothed, "b-", linewidth=2, label="Train (smoothed)")
        if self.val_metrics.steps:
            ax.plot(
                self.val_metrics.steps,
                self.val_metrics.total_loss,
                "r-",
                linewidth=2,
                label="Val",
            )
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title("Total Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Applicability loss
        ax = axes[0, 1]
        ax.plot(steps, self.train_metrics.applicability_loss, "g-", alpha=0.3)
        if len(steps) > 50:
            smoothed = self.train_metrics.get_smoothed("applicability_loss", 50)
            ax.plot(steps[49:], smoothed, "g-", linewidth=2)
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title("Applicability Loss")
        ax.grid(True, alpha=0.3)

        # Compliance loss
        ax = axes[0, 2]
        ax.plot(steps, self.train_metrics.compliance_loss, "m-", alpha=0.3)
        if len(steps) > 50:
            smoothed = self.train_metrics.get_smoothed("compliance_loss", 50)
            ax.plot(steps[49:], smoothed, "m-", linewidth=2)
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title("Compliance Loss")
        ax.grid(True, alpha=0.3)

        # Reconstruction loss
        ax = axes[1, 0]
        ax.plot(steps, self.train_metrics.recon_loss, "c-", alpha=0.3)
        if len(steps) > 50:
            smoothed = self.train_metrics.get_smoothed("recon_loss", 50)
            ax.plot(steps[49:], smoothed, "c-", linewidth=2)
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title("Reconstruction Loss")
        ax.grid(True, alpha=0.3)

        # KL loss
        ax = axes[1, 1]
        ax.plot(steps, self.train_metrics.kl_loss, "y-", alpha=0.3)
        if len(steps) > 50:
            smoothed = self.train_metrics.get_smoothed("kl_loss", 50)
            ax.plot(steps[49:], smoothed, "y-", linewidth=2)
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title("KL Loss")
        ax.grid(True, alpha=0.3)

        # Temporal loss
        ax = axes[1, 2]
        ax.plot(steps, self.train_metrics.temporal_loss, "orange", alpha=0.3)
        if len(steps) > 50:
            smoothed = self.train_metrics.get_smoothed("temporal_loss", 50)
            ax.plot(steps[49:], smoothed, "orange", linewidth=2)
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title("Temporal Loss")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            fig.savefig(
                self.plots_dir / "loss_curves.png", dpi=150, bbox_inches="tight"
            )
            plt.close(fig)

        return fig

    def plot_tier_losses(self, save: bool = True) -> Optional[Figure]:
        """Plot per-tier loss curves."""
        if not MATPLOTLIB_AVAILABLE or not self.train_metrics.steps:
            return None

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f"{self.experiment_name} - Per-Tier Losses", fontsize=14)

        steps = self.train_metrics.steps
        tiers = [
            ("safety_loss", "Safety (Tier 0)", "red", axes[0, 0]),
            ("legal_loss", "Legal (Tier 1)", "orange", axes[0, 1]),
            ("road_loss", "Road (Tier 2)", "blue", axes[1, 0]),
            ("comfort_loss", "Comfort (Tier 3)", "green", axes[1, 1]),
        ]

        for key, title, color, ax in tiers:
            values = getattr(self.train_metrics, key)
            ax.plot(steps, values, color=color, alpha=0.3)
            if len(steps) > 50:
                smoothed = self.train_metrics.get_smoothed(key, 50)
                ax.plot(steps[49:], smoothed, color=color, linewidth=2)
            ax.set_xlabel("Step")
            ax.set_ylabel("Loss")
            ax.set_title(title)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            fig.savefig(
                self.plots_dir / "tier_losses.png", dpi=150, bbox_inches="tight"
            )
            plt.close(fig)

        return fig

    def plot_metrics(self, save: bool = True) -> Optional[Figure]:
        """Plot accuracy and compliance metrics."""
        if not MATPLOTLIB_AVAILABLE or not self.train_metrics.steps:
            return None

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f"{self.experiment_name} - Training Metrics", fontsize=14)

        steps = self.train_metrics.steps

        # Applicability accuracy
        ax = axes[0, 0]
        if self.train_metrics.applicability_accuracy:
            ax.plot(
                steps,
                self.train_metrics.applicability_accuracy,
                "b-",
                alpha=0.5,
                label="Train",
            )
            if self.val_metrics.steps and self.val_metrics.applicability_accuracy:
                ax.plot(
                    self.val_metrics.steps,
                    self.val_metrics.applicability_accuracy,
                    "r-",
                    linewidth=2,
                    label="Val",
                )
        ax.set_xlabel("Step")
        ax.set_ylabel("Accuracy")
        ax.set_title("Applicability Accuracy")
        ax.set_ylim([0, 1])
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Safety compliance rate
        ax = axes[0, 1]
        if self.train_metrics.safety_compliance_rate:
            ax.plot(steps, self.train_metrics.safety_compliance_rate, "g-", alpha=0.5)
        ax.set_xlabel("Step")
        ax.set_ylabel("Rate")
        ax.set_title("Safety Compliance Rate")
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)

        # ADE
        ax = axes[1, 0]
        if self.train_metrics.ade:
            ax.plot(steps, self.train_metrics.ade, "c-", alpha=0.5, label="Train")
            if self.val_metrics.steps and self.val_metrics.ade:
                ax.plot(
                    self.val_metrics.steps,
                    self.val_metrics.ade,
                    "r-",
                    linewidth=2,
                    label="Val",
                )
        ax.set_xlabel("Step")
        ax.set_ylabel("ADE (m)")
        ax.set_title("Average Displacement Error")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # FDE
        ax = axes[1, 1]
        if self.train_metrics.fde:
            ax.plot(steps, self.train_metrics.fde, "m-", alpha=0.5, label="Train")
            if self.val_metrics.steps and self.val_metrics.fde:
                ax.plot(
                    self.val_metrics.steps,
                    self.val_metrics.fde,
                    "r-",
                    linewidth=2,
                    label="Val",
                )
        ax.set_xlabel("Step")
        ax.set_ylabel("FDE (m)")
        ax.set_title("Final Displacement Error")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            fig.savefig(self.plots_dir / "metrics.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

        return fig

    def plot_learning_rate(self, save: bool = True) -> Optional[Figure]:
        """Plot learning rate schedule."""
        if not MATPLOTLIB_AVAILABLE or not self.train_metrics.learning_rate:
            return None

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(self.train_metrics.steps, self.train_metrics.learning_rate, "b-")
        ax.set_xlabel("Step")
        ax.set_ylabel("Learning Rate")
        ax.set_title(f"{self.experiment_name} - Learning Rate Schedule")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            fig.savefig(
                self.plots_dir / "learning_rate.png", dpi=150, bbox_inches="tight"
            )
            plt.close(fig)

        return fig

    def get_summary(self) -> Dict[str, Any]:
        """Get training summary statistics."""
        elapsed = time.time() - self.start_time

        summary = {
            "experiment_name": self.experiment_name,
            "total_steps": len(self.train_metrics.steps),
            "total_epochs": (
                max(self.train_metrics.epochs) if self.train_metrics.epochs else 0
            ),
            "elapsed_time_s": elapsed,
            "elapsed_time_str": (
                f"{elapsed/3600:.1f}h" if elapsed > 3600 else f"{elapsed/60:.1f}m"
            ),
            "steps_per_second": (
                len(self.train_metrics.steps) / elapsed if elapsed > 0 else 0
            ),
        }

        # Best metrics
        if self.train_metrics.total_loss:
            summary["best_train_loss"] = min(self.train_metrics.total_loss)
            summary["final_train_loss"] = self.train_metrics.total_loss[-1]

        if self.val_metrics.total_loss:
            summary["best_val_loss"] = min(self.val_metrics.total_loss)
            summary["final_val_loss"] = self.val_metrics.total_loss[-1]

        return summary

    def print_summary(self):
        """Print training summary to console."""
        summary = self.get_summary()

        print("\n" + "=" * 60)
        print(f"Training Summary: {summary['experiment_name']}")
        print("=" * 60)
        print(f"  Total Steps: {summary['total_steps']}")
        print(f"  Total Epochs: {summary['total_epochs']}")
        print(f"  Elapsed Time: {summary['elapsed_time_str']}")
        print(f"  Steps/Second: {summary['steps_per_second']:.2f}")

        if "best_train_loss" in summary:
            print(
                f"\n  Train Loss: {summary['final_train_loss']:.4f} (best: {summary['best_train_loss']:.4f})"
            )
        if "best_val_loss" in summary:
            print(
                f"  Val Loss: {summary['final_val_loss']:.4f} (best: {summary['best_val_loss']:.4f})"
            )

        print("=" * 60 + "\n")


def plot_trajectory_comparison(
    pred_traj: np.ndarray,
    gt_traj: np.ndarray,
    lane_centers: Optional[np.ndarray] = None,
    agent_positions: Optional[np.ndarray] = None,
    title: str = "Trajectory Comparison",
    save_path: Optional[str] = None,
) -> Optional[Figure]:
    """
    Plot predicted vs ground truth trajectory.

    Args:
        pred_traj: [T, 2] predicted trajectory
        gt_traj: [T, 2] ground truth trajectory
        lane_centers: [L, P, 2] lane centerlines
        agent_positions: [N, T, 2] other agent positions
        title: Plot title
        save_path: Path to save figure

    Returns:
        matplotlib Figure or None
    """
    if not MATPLOTLIB_AVAILABLE:
        return None

    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot lanes
    if lane_centers is not None:
        for lane in lane_centers:
            ax.plot(lane[:, 0], lane[:, 1], "gray", alpha=0.3, linewidth=1)

    # Plot other agents
    if agent_positions is not None:
        for i, agent in enumerate(agent_positions):
            ax.plot(agent[:, 0], agent[:, 1], "orange", alpha=0.5, linewidth=1)
            ax.scatter(agent[-1, 0], agent[-1, 1], c="orange", s=30, marker="s")

    # Plot ground truth
    ax.plot(gt_traj[:, 0], gt_traj[:, 1], "g-", linewidth=2, label="Ground Truth")
    ax.scatter(gt_traj[0, 0], gt_traj[0, 1], c="g", s=100, marker="o", zorder=5)
    ax.scatter(gt_traj[-1, 0], gt_traj[-1, 1], c="g", s=100, marker="*", zorder=5)

    # Plot prediction
    ax.plot(pred_traj[:, 0], pred_traj[:, 1], "b-", linewidth=2, label="Prediction")
    ax.scatter(pred_traj[-1, 0], pred_traj[-1, 1], c="b", s=100, marker="*", zorder=5)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(title)
    ax.legend()
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig


def compute_trajectory_metrics(
    pred_traj: np.ndarray,
    gt_traj: np.ndarray,
) -> Dict[str, float]:
    """
    Compute trajectory quality metrics.

    Args:
        pred_traj: [T, 2+] predicted trajectory
        gt_traj: [T, 2+] ground truth trajectory

    Returns:
        Dict with ADE, FDE, and other metrics
    """
    # Ensure same length
    T = min(len(pred_traj), len(gt_traj))
    pred = pred_traj[:T, :2]
    gt = gt_traj[:T, :2]

    # Displacement errors
    displacements = np.linalg.norm(pred - gt, axis=1)

    metrics = {
        "ade": float(np.mean(displacements)),
        "fde": float(displacements[-1]),
        "max_de": float(np.max(displacements)),
        "min_de": float(np.min(displacements)),
    }

    # Miss rate (threshold-based)
    for threshold in [1.0, 2.0, 5.0]:
        miss_rate = float(np.mean(displacements > threshold))
        metrics[f"miss_rate_{threshold}m"] = miss_rate

    return metrics
