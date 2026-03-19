#!/usr/bin/env python3
"""
Scenario Safety Profile
========================

Horizontal-bar chart colour-coded by safety outcome:
  • Green  – collision-free (overlap_rate = 0)
  • Amber  – minor overlap  (0 < overlap_rate < 0.10)
  • Red    – significant overlap (overlap_rate ≥ 0.10)

Plus a side annotation showing log-divergence.

Usage:
    python -m simulation_engine.viz.scenario_safety_profile \
        --results /workspace/output/closedloop/validate_50_results.json \
        --outdir  /workspace/output/closedloop/figures
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


IEEE_FONTSIZE = 8
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": IEEE_FONTSIZE,
        "axes.labelsize": IEEE_FONTSIZE,
        "axes.titlesize": IEEE_FONTSIZE + 1,
        "xtick.labelsize": IEEE_FONTSIZE - 1,
        "ytick.labelsize": IEEE_FONTSIZE - 1,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    }
)


def _classify_scenario(overlap_rate: float) -> Tuple[str, str]:
    """Return (label, colour) for a given overlap rate."""
    if overlap_rate <= 0.0:
        return ("Collision-free", "#2ca02c")  # green
    elif overlap_rate < 0.10:
        return ("Minor overlap", "#ff7f0e")  # amber
    else:
        return ("Significant overlap", "#d62728")  # red


def load_results(path: str) -> Dict[str, List[Dict[str, float]]]:
    with open(path) as f:
        return json.load(f)


def plot_safety_profile(
    results: Dict[str, List[Dict]],
    outpath: str,
    selector: str = "Confidence",
):
    """
    Dual-panel figure:
      Left  – Horizontal bar of overlap_rate per scenario, colour-coded
      Right – Horizontal bar of log_divergence/mean, grey-scale
    """
    scenario_data = results[selector]
    n = len(scenario_data)
    indices = np.arange(n)

    overlap_rates = np.array([d.get("overlap/rate", 0.0) for d in scenario_data])
    log_divs = np.array([d.get("log_divergence/mean", 0.0) for d in scenario_data])
    min_clear = np.array([d.get("min_clearance/min", 0.0) for d in scenario_data])

    # Sort by overlap rate (worst at top)
    order = np.argsort(overlap_rates)[::-1]
    overlap_rates = overlap_rates[order]
    log_divs = log_divs[order]
    min_clear = min_clear[order]

    colours = [_classify_scenario(ov)[1] for ov in overlap_rates]

    fig, (ax1, ax2, ax3) = plt.subplots(
        1,
        3,
        figsize=(7.16, 5.0),
        sharey=True,
        gridspec_kw={"width_ratios": [2, 2, 2], "wspace": 0.12},
    )

    # --- Panel 1: Overlap rate ---
    ax1.barh(indices, overlap_rates * 100, color=colours, height=0.8, edgecolor="none")
    ax1.set_xlabel("Overlap Rate [%]")
    ax1.set_title("Collision Safety")
    ax1.set_yticks(indices)
    ax1.set_yticklabels([f"S{order[i]+1:02d}" for i in range(n)], fontsize=5)
    ax1.axvline(0, color="black", lw=0.5)
    ax1.invert_yaxis()

    # Legend
    patches = [
        mpatches.Patch(color="#2ca02c", label="Collision-free"),
        mpatches.Patch(color="#ff7f0e", label="Minor (<10%)"),
        mpatches.Patch(color="#d62728", label="Significant (≥10%)"),
    ]
    ax1.legend(handles=patches, loc="lower right", fontsize=6)

    # --- Panel 2: Log divergence ---
    ax2.barh(indices, log_divs, color="#1f77b4", alpha=0.7, height=0.8)
    ax2.set_xlabel("Log Divergence [m]")
    ax2.set_title("Trajectory Accuracy")
    ax2.axvline(0, color="black", lw=0.5)

    # --- Panel 3: Min clearance ---
    # Colour: low clearance → red, high → green
    max_clear = max(min_clear.max(), 1.0)
    clear_colours = [plt.cm.RdYlGn(min(c / max_clear, 1.0)) for c in min_clear]
    ax3.barh(indices, min_clear, color=clear_colours, height=0.8, edgecolor="none")
    ax3.set_xlabel("Min Clearance [m]")
    ax3.set_title("Proximity Safety")
    ax3.axvline(0, color="black", lw=0.5)

    fig.suptitle(
        "Per-Scenario Safety Profile — 50 Interactive Scenarios (Waymax)",
        fontsize=IEEE_FONTSIZE + 1,
    )

    # Summary annotation
    n_free = int(np.sum(overlap_rates <= 0))
    fig.text(
        0.5,
        -0.02,
        f"{n_free}/{n} ({100*n_free/n:.0f}%) collision-free  |  "
        f"Mean log-divergence: {log_divs.mean():.2f} m  |  "
        f"Mean min-clearance: {min_clear.mean():.2f} m",
        ha="center",
        fontsize=IEEE_FONTSIZE - 1,
        style="italic",
    )

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.savefig(outpath)
    plt.close(fig)
    print(f"[safety_profile] Saved → {outpath}")


def plot_safety_pie(
    results: Dict[str, List[Dict]],
    outpath: str,
    selector: str = "Confidence",
):
    """Simple pie chart: collision-free vs. overlap scenarios."""
    scenario_data = results[selector]
    overlaps = [d.get("overlap/rate", 0.0) for d in scenario_data]
    n_free = sum(1 for o in overlaps if o <= 0)
    n_minor = sum(1 for o in overlaps if 0 < o < 0.10)
    n_sig = sum(1 for o in overlaps if o >= 0.10)

    sizes = [n_free, n_minor, n_sig]
    labels = [
        f"Collision-free\n({n_free})",
        f"Minor overlap\n({n_minor})",
        f"Significant\n({n_sig})",
    ]
    colors = ["#2ca02c", "#ff7f0e", "#d62728"]
    explode = [0.03, 0.03, 0.05]

    # Remove zero slices
    active = [
        (s, l, c, e) for s, l, c, e in zip(sizes, labels, colors, explode) if s > 0
    ]
    sizes, labels, colors, explode = zip(*active)

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        colors=colors,
        explode=explode,
        autopct="%1.0f%%",
        pctdistance=0.7,
        startangle=90,
        textprops={"fontsize": IEEE_FONTSIZE - 1},
    )
    for at in autotexts:
        at.set_fontsize(IEEE_FONTSIZE)
        at.set_fontweight("bold")

    ax.set_title(
        "Scenario Safety Outcome Distribution",
        fontsize=IEEE_FONTSIZE + 1,
        pad=10,
    )
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.savefig(outpath)
    plt.close(fig)
    print(f"[safety_pie] Saved → {outpath}")


def main():
    parser = argparse.ArgumentParser(description="Scenario safety profile figures")
    parser.add_argument(
        "--results", default="/workspace/output/closedloop/validate_50_results.json"
    )
    parser.add_argument("--outdir", default="/workspace/output/closedloop/figures")
    parser.add_argument("--selector", default="Confidence")
    args = parser.parse_args()

    results = load_results(args.results)
    od = args.outdir

    plot_safety_profile(
        results, os.path.join(od, "scenario_safety_profile.pdf"), args.selector
    )
    plot_safety_profile(
        results, os.path.join(od, "scenario_safety_profile.png"), args.selector
    )

    plot_safety_pie(results, os.path.join(od, "scenario_safety_pie.pdf"), args.selector)
    plot_safety_pie(results, os.path.join(od, "scenario_safety_pie.png"), args.selector)


if __name__ == "__main__":
    main()
