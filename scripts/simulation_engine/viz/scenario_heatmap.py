#!/usr/bin/env python3
"""
Scenario × Metric Heatmap
===========================

Generate a heatmap where rows are scenarios and columns are metrics,
providing a compact visual overview of per-scenario performance.

Usage:
    python -m simulation_engine.viz.scenario_heatmap \
        --results /workspace/output/closedloop/validate_50_results.json \
        --outdir  /workspace/output/closedloop/figures
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


IEEE_FONTSIZE = 8
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": IEEE_FONTSIZE,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    }
)


HEATMAP_METRICS = [
    ("overlap/rate", "Overlap\nRate", True),  # higher=worse
    ("kinematic_infeasibility/rate", "Kin.\nInfeas.", True),
    ("log_divergence/mean", "Log Div.\n(mean)", True),
    ("jerk/mean", "Jerk\n(mean)", True),
    ("jerk/max", "Jerk\n(max)", True),
    ("min_clearance/min", "Min\nClear.", False),  # higher=better
    ("min_clearance/mean", "Mean\nClear.", False),
]


def load_results(path: str) -> Dict[str, List[Dict[str, float]]]:
    with open(path) as f:
        return json.load(f)


def plot_heatmap(
    results: Dict[str, List[Dict]],
    outpath: str,
    selector: str = "Confidence",
):
    data = results[selector]
    n = len(data)
    m = len(HEATMAP_METRICS)

    # Build 2-D matrix [n_scenarios × n_metrics]
    matrix = np.full((n, m), np.nan)
    for i, d in enumerate(data):
        for j, (mkey, _, _) in enumerate(HEATMAP_METRICS):
            if mkey in d:
                matrix[i, j] = d[mkey]

    # normalize each column to [0,1] for visual comparability
    norm_matrix = np.copy(matrix)
    col_min = np.nanmin(matrix, axis=0)
    col_max = np.nanmax(matrix, axis=0)
    col_range = col_max - col_min
    col_range[col_range == 0] = 1.0
    norm_matrix = (matrix - col_min) / col_range

    # For "lower is better" metrics, invert so red = bad
    for j, (_, _, higher_is_worse) in enumerate(HEATMAP_METRICS):
        if not higher_is_worse:
            norm_matrix[:, j] = 1.0 - norm_matrix[:, j]

    # Sort scenarios by worst aggregate score
    agg = np.nanmean(norm_matrix, axis=1)
    order = np.argsort(agg)[::-1]
    matrix = matrix[order]
    norm_matrix = norm_matrix[order]

    # Plot
    fig, ax = plt.subplots(figsize=(7.16, 6.0))

    # Custom colourmap: green(good) → yellow → red(bad)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "safety",
        ["#2ca02c", "#ffffb2", "#d62728"],
    )

    im = ax.imshow(
        norm_matrix,
        aspect="auto",
        cmap=cmap,
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )

    # Annotate cells with actual values
    for i in range(n):
        for j in range(m):
            val = matrix[i, j]
            if np.isnan(val):
                continue
            txt = f"{val:.2f}" if val < 10 else f"{val:.0f}"
            text_col = (
                "white"
                if norm_matrix[i, j] > 0.7 or norm_matrix[i, j] < 0.2
                else "black"
            )
            ax.text(j, i, txt, ha="center", va="center", fontsize=5, color=text_col)

    # Labels
    ax.set_xticks(range(m))
    ax.set_xticklabels([lbl for _, lbl, _ in HEATMAP_METRICS], fontsize=6)
    ax.set_yticks(range(n))
    ax.set_yticklabels([f"S{order[i]+1:02d}" for i in range(n)], fontsize=5)
    ax.set_xlabel("Metric", fontsize=IEEE_FONTSIZE)
    ax.set_ylabel("Scenario (sorted by aggregate risk)", fontsize=IEEE_FONTSIZE)

    # Colourbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("normalized Risk (0=good, 1=bad)", fontsize=IEEE_FONTSIZE - 1)
    cbar.ax.tick_params(labelsize=6)

    ax.set_title(
        "Scenario × Metric Heatmap — 50 Interactive Scenarios (Waymax)",
        fontsize=IEEE_FONTSIZE + 1,
    )

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.savefig(outpath)
    plt.close(fig)
    print(f"[scenario_heatmap] Saved → {outpath}")


def main():
    parser = argparse.ArgumentParser(description="Scenario × Metric heatmap")
    parser.add_argument(
        "--results", default="/workspace/output/closedloop/validate_50_results.json"
    )
    parser.add_argument("--outdir", default="/workspace/output/closedloop/figures")
    parser.add_argument("--selector", default="Confidence")
    args = parser.parse_args()

    results = load_results(args.results)
    od = args.outdir

    plot_heatmap(
        results, os.path.join(od, "scenario_metric_heatmap.pdf"), args.selector
    )
    plot_heatmap(
        results, os.path.join(od, "scenario_metric_heatmap.png"), args.selector
    )


if __name__ == "__main__":
    main()
