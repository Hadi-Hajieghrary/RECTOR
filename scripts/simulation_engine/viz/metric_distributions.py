#!/usr/bin/env python3
"""
Metric Distribution Plots
==========================

Generate publication-quality box-and-violin plots from the 50-scenario
closed-loop validation results.

Produces:
  • Per-metric violin plots (one row, 6 key metrics)
  • Full 11-metric grid (3×4 subplot grid)

Usage:
    python -m simulation_engine.viz.metric_distributions \
        --results /workspace/output/closedloop/validate_50_results.json \
        --outdir  /workspace/output/closedloop/figures
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


IEEE_COL_W = 3.5  # inches (single-column width)
IEEE_FULL_W = 7.16  # inches (double-column)
IEEE_FONTSIZE = 8

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": IEEE_FONTSIZE,
        "axes.labelsize": IEEE_FONTSIZE,
        "axes.titlesize": IEEE_FONTSIZE + 1,
        "xtick.labelsize": IEEE_FONTSIZE - 1,
        "ytick.labelsize": IEEE_FONTSIZE - 1,
        "legend.fontsize": IEEE_FONTSIZE - 1,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
    }
)


METRIC_META = {
    "overlap/rate": {
        "label": "Overlap Rate",
        "unit": "",
        "fmt": ".2f",
        "color": "#d62728",
    },
    "log_divergence/mean": {
        "label": "Log Divergence (mean)",
        "unit": "m",
        "fmt": ".2f",
        "color": "#1f77b4",
    },
    "log_divergence/max": {
        "label": "Log Divergence (max)",
        "unit": "m",
        "fmt": ".2f",
        "color": "#1f77b4",
    },
    "log_divergence/final": {
        "label": "Log Divergence (final)",
        "unit": "m",
        "fmt": ".2f",
        "color": "#1f77b4",
    },
    "kinematic_infeasibility/rate": {
        "label": "Kin. Infeasibility Rate",
        "unit": "",
        "fmt": ".3f",
        "color": "#ff7f0e",
    },
    "jerk/mean": {
        "label": "Jerk (mean)",
        "unit": "m/s³",
        "fmt": ".1f",
        "color": "#2ca02c",
    },
    "jerk/max": {
        "label": "Jerk (max)",
        "unit": "m/s³",
        "fmt": ".0f",
        "color": "#2ca02c",
    },
    "min_clearance/min": {
        "label": "Min Clearance",
        "unit": "m",
        "fmt": ".2f",
        "color": "#9467bd",
    },
    "min_clearance/mean": {
        "label": "Mean Clearance",
        "unit": "m",
        "fmt": ".2f",
        "color": "#9467bd",
    },
    "ttc/min": {"label": "TTC (min)", "unit": "s", "fmt": ".2f", "color": "#8c564b"},
    "ttc/mean": {"label": "TTC (mean)", "unit": "s", "fmt": ".2f", "color": "#8c564b"},
}

# The 6 "hero" metrics used in the paper Table XXII
HERO_METRICS = [
    "overlap/rate",
    "log_divergence/mean",
    "kinematic_infeasibility/rate",
    "min_clearance/min",
    "jerk/mean",
    "jerk/max",
]


def load_results(path: str) -> Dict[str, List[Dict[str, float]]]:
    """Load validate_50_results.json → {selector_name: [scenario_dicts]}."""
    with open(path) as f:
        return json.load(f)


def extract_metric_array(
    results: Dict[str, List[Dict]],
    metric_key: str,
    selector: str = "Confidence",
) -> np.ndarray:
    """Pull one metric as a flat array from one selector's results."""
    vals = []
    for d in results[selector]:
        if metric_key in d:
            vals.append(d[metric_key])
    return np.array(vals, dtype=np.float64)


def plot_hero_violin(
    results: Dict[str, List[Dict]],
    outpath: str,
    selector: str = "Confidence",
):
    """Six-panel violin plot of the hero metrics."""
    fig, axes = plt.subplots(1, 6, figsize=(IEEE_FULL_W, 1.8))

    for ax, mkey in zip(axes, HERO_METRICS):
        meta = METRIC_META[mkey]
        data = extract_metric_array(results, mkey, selector)
        if len(data) == 0:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(meta["label"])
            continue

        parts = ax.violinplot(data, positions=[0], showmedians=True, showextrema=True)
        for pc in parts["bodies"]:
            pc.set_facecolor(meta["color"])
            pc.set_alpha(0.6)
        parts["cmedians"].set_color("black")

        # Overlay individual points
        jitter = np.random.default_rng(42).uniform(-0.08, 0.08, len(data))
        ax.scatter(jitter, data, s=6, alpha=0.5, color=meta["color"], zorder=5)

        unit = f" ({meta['unit']})" if meta["unit"] else ""
        ax.set_title(f"{meta['label']}{unit}", fontsize=IEEE_FONTSIZE - 0.5)
        ax.set_xticks([])
        ax.yaxis.set_major_locator(mticker.MaxNLocator(5))

        # Annotate median
        med = np.median(data)
        ax.axhline(med, color="gray", lw=0.5, ls="--")
        ax.text(
            0.25, med, f"{med:{meta['fmt']}}", fontsize=6, color="gray", va="bottom"
        )

    fig.suptitle(
        "50-Scenario Closed-Loop Metric Distributions (Waymax Pipeline Validation)",
        fontsize=IEEE_FONTSIZE + 1,
        y=1.02,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.savefig(outpath)
    plt.close(fig)
    print(f"[metric_distributions] Saved hero violin → {outpath}")


def plot_full_grid(
    results: Dict[str, List[Dict]],
    outpath: str,
    selector: str = "Confidence",
):
    """3×4 grid of box-and-strip plots for all 11 metrics."""
    all_keys = list(METRIC_META.keys())
    nrows, ncols = 3, 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(IEEE_FULL_W, 5.5))
    axes = axes.flat

    for i, mkey in enumerate(all_keys):
        ax = axes[i]
        meta = METRIC_META[mkey]
        data = extract_metric_array(results, mkey, selector)

        if len(data) == 0:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(meta["label"])
            continue

        bp = ax.boxplot(
            data,
            vert=True,
            patch_artist=True,
            widths=0.5,
            boxprops=dict(facecolor=meta["color"], alpha=0.3),
            medianprops=dict(color="black", lw=1.5),
            flierprops=dict(marker=".", markersize=3, alpha=0.5),
        )
        # Strip overlay
        jitter = np.random.default_rng(42).uniform(0.85, 1.15, len(data))
        ax.scatter(jitter, data, s=5, alpha=0.45, color=meta["color"], zorder=5)

        unit = f" [{meta['unit']}]" if meta["unit"] else ""
        ax.set_title(f"{meta['label']}{unit}", fontsize=IEEE_FONTSIZE - 0.5)
        ax.set_xticks([])
        ax.yaxis.set_major_locator(mticker.MaxNLocator(5))

        # Stats annotation
        mean = np.mean(data)
        std = np.std(data)
        ax.text(
            0.98,
            0.95,
            f"μ={mean:{meta['fmt']}}\nσ={std:{meta['fmt']}}",
            transform=ax.transAxes,
            fontsize=6,
            ha="right",
            va="top",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8),
        )

    # Hide extra subplot(s)
    for j in range(len(all_keys), nrows * ncols):
        axes[j].axis("off")

    fig.suptitle(
        "Closed-Loop Metric Summary — 50 Interactive Scenarios (Waymax)",
        fontsize=IEEE_FONTSIZE + 1,
    )
    plt.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
    print(f"[metric_distributions] Saved full grid → {outpath}")


def main():
    parser = argparse.ArgumentParser(description="Metric distribution figures")
    parser.add_argument(
        "--results", default="/workspace/output/closedloop/validate_50_results.json"
    )
    parser.add_argument("--outdir", default="/workspace/output/closedloop/figures")
    parser.add_argument("--selector", default="Confidence")
    args = parser.parse_args()

    results = load_results(args.results)

    plot_hero_violin(
        results, os.path.join(args.outdir, "metric_hero_violin.pdf"), args.selector
    )
    plot_hero_violin(
        results, os.path.join(args.outdir, "metric_hero_violin.png"), args.selector
    )

    plot_full_grid(
        results, os.path.join(args.outdir, "metric_full_grid.pdf"), args.selector
    )
    plot_full_grid(
        results, os.path.join(args.outdir, "metric_full_grid.png"), args.selector
    )


if __name__ == "__main__":
    main()
