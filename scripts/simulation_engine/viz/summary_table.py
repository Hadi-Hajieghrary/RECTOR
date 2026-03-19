#!/usr/bin/env python3
"""
Summary Statistics Table
=========================

Generate a LaTeX-ready and Markdown summary table from the 50-scenario
closed-loop validation results.

Matches Table XXII in the paper but with fuller statistics
(mean ± std, median, Q1, Q3, min, max).

Usage:
    python -m simulation_engine.viz.summary_table \
        --results /workspace/output/closedloop/validate_50_results.json \
        --outdir  /workspace/output/closedloop/figures
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import numpy as np


METRIC_SPEC = [
    ("overlap/rate", "Overlap Rate", "%", 100.0),
    ("log_divergence/mean", "Log Divergence (mean)", "m", 1.0),
    ("log_divergence/max", "Log Divergence (max)", "m", 1.0),
    ("kinematic_infeasibility/rate", "Kin. Infeasibility", "%", 100.0),
    ("jerk/mean", "Jerk (mean)", "m/s³", 1.0),
    ("jerk/max", "Jerk (max)", "m/s³", 1.0),
    ("min_clearance/min", "Min Clearance", "m", 1.0),
    ("min_clearance/mean", "Mean Clearance", "m", 1.0),
    ("ttc/min", "TTC (min)", "s", 1.0),
    ("ttc/mean", "TTC (mean)", "s", 1.0),
]


def load_results(path: str) -> Dict[str, List[Dict[str, float]]]:
    with open(path) as f:
        return json.load(f)


def compute_stats(values: np.ndarray) -> Dict[str, float]:
    """Compute descriptive statistics for an array."""
    if len(values) == 0:
        return {
            k: float("nan")
            for k in ["n", "mean", "std", "med", "q1", "q3", "min", "max"]
        }
    return {
        "n": len(values),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "med": float(np.median(values)),
        "q1": float(np.percentile(values, 25)),
        "q3": float(np.percentile(values, 75)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def generate_latex_table(
    results: Dict[str, List[Dict]],
    selector: str = "Confidence",
) -> str:
    """Generate a LaTeX tabular environment string."""
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Closed-Loop Pipeline Validation: Full Statistics (50 scenarios, Waymax)}"
    )
    lines.append(r"\label{tab:closedloop_full_stats}")
    lines.append(r"\footnotesize")
    lines.append(r"\begin{tabular}{l c c c c c c c}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Metric} & \textbf{Unit} & \textbf{Mean} & \textbf{Std} & \textbf{Median} & \textbf{Q1} & \textbf{Q3} & \textbf{Range} \\"
    )
    lines.append(r"\midrule")

    data = results[selector]
    for mkey, label, unit, scale in METRIC_SPEC:
        vals = np.array([d[mkey] * scale for d in data if mkey in d])
        if len(vals) == 0:
            continue
        s = compute_stats(vals)
        # Smart formatting
        if s["max"] < 1:
            fmt = ".3f"
        elif s["max"] < 10:
            fmt = ".2f"
        elif s["max"] < 100:
            fmt = ".1f"
        else:
            fmt = ".0f"

        row = (
            f"  {label} & {unit} & "
            f"${s['mean']:{fmt}}$ & ${s['std']:{fmt}}$ & "
            f"${s['med']:{fmt}}$ & ${s['q1']:{fmt}}$ & ${s['q3']:{fmt}}$ & "
            f"$[{s['min']:{fmt}},\\,{s['max']:{fmt}}]$ \\\\"
        )
        lines.append(row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def generate_markdown_table(
    results: Dict[str, List[Dict]],
    selector: str = "Confidence",
) -> str:
    """Generate a Markdown table string."""
    lines = []
    lines.append(
        "## Closed-Loop Pipeline Validation: Full Statistics (50 scenarios, Waymax)"
    )
    lines.append("")
    lines.append("| Metric | Unit | Mean | Std | Median | Q1 | Q3 | Min | Max |")
    lines.append("|--------|------|------|-----|--------|----|----|-----|-----|")

    data = results[selector]
    for mkey, label, unit, scale in METRIC_SPEC:
        vals = np.array([d[mkey] * scale for d in data if mkey in d])
        if len(vals) == 0:
            continue
        s = compute_stats(vals)
        fmt = ".2f"
        row = (
            f"| {label} | {unit} | "
            f"{s['mean']:{fmt}} | {s['std']:{fmt}} | "
            f"{s['med']:{fmt}} | {s['q1']:{fmt}} | {s['q3']:{fmt}} | "
            f"{s['min']:{fmt}} | {s['max']:{fmt}} |"
        )
        lines.append(row)

    return "\n".join(lines)


def generate_summary_text(
    results: Dict[str, List[Dict]],
    selector: str = "Confidence",
) -> str:
    data = results[selector]
    n = len(data)
    ov = [d.get("overlap/rate", 0.0) for d in data]
    n_free = sum(1 for o in ov if o <= 0)
    ld = [d.get("log_divergence/mean", 0.0) for d in data]
    jm = [d.get("jerk/mean", 0.0) for d in data]
    mc = [d.get("min_clearance/min", 0.0) for d in data]

    return (
        f"CLOSED-LOOP VALIDATION SUMMARY\n"
        f"{'='*50}\n"
        f"Scenarios tested:       {n}\n"
        f"Collision-free:         {n_free}/{n} ({100*n_free/n:.0f}%)\n"
        f"Mean overlap rate:      {100*np.mean(ov):.1f}%\n"
        f"Mean log divergence:    {np.mean(ld):.3f} m\n"
        f"Mean jerk:              {np.mean(jm):.1f} m/s³\n"
        f"Mean min-clearance:     {np.mean(mc):.2f} m\n"
    )


def main():
    parser = argparse.ArgumentParser(description="Summary statistics table")
    parser.add_argument(
        "--results", default="/workspace/output/closedloop/validate_50_results.json"
    )
    parser.add_argument("--outdir", default="/workspace/output/closedloop/figures")
    parser.add_argument("--selector", default="Confidence")
    args = parser.parse_args()

    results = load_results(args.results)
    od = args.outdir
    os.makedirs(od, exist_ok=True)

    # LaTeX
    latex = generate_latex_table(results, args.selector)
    latex_path = os.path.join(od, "closedloop_stats_table.tex")
    with open(latex_path, "w") as f:
        f.write(latex)
    print(f"[summary_table] LaTeX → {latex_path}")

    # Markdown
    md = generate_markdown_table(results, args.selector)
    md_path = os.path.join(od, "closedloop_stats_table.md")
    with open(md_path, "w") as f:
        f.write(md)
    print(f"[summary_table] Markdown → {md_path}")

    # Summary text
    txt = generate_summary_text(results, args.selector)
    txt_path = os.path.join(od, "closedloop_summary.txt")
    with open(txt_path, "w") as f:
        f.write(txt)
    print(f"[summary_table] Summary → {txt_path}")

    # Also print
    print()
    print(latex)
    print()
    print(txt)


if __name__ == "__main__":
    main()
