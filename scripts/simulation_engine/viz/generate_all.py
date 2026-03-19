#!/usr/bin/env python3
"""
Generate All Artifacts
=======================

Master script that regenerates all closed-loop validation figures
and tables in one invocation.

Usage:
    python -m simulation_engine.viz.generate_all \
        --results /workspace/output/closedloop/validate_50_results.json \
        --outdir  /workspace/output/closedloop/figures
"""

from __future__ import annotations

import argparse
import os
import sys
import time


def main():
    parser = argparse.ArgumentParser(description="Generate all closed-loop artifacts")
    parser.add_argument(
        "--results", default="/workspace/output/closedloop/validate_50_results.json"
    )
    parser.add_argument("--outdir", default="/workspace/output/closedloop/figures")
    parser.add_argument("--selector", default="Confidence")
    parser.add_argument(
        "--bev-scenarios",
        type=int,
        nargs="*",
        default=[0, 4],
        help="Scenario indices for BEV rendering (empty to skip)",
    )
    parser.add_argument(
        "--skip-bev",
        action="store_true",
        help="Skip BEV rendering (faster, doesn't require JAX)",
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    t0 = time.time()

    # ---- 1. Metric distributions ----
    print("=" * 60)
    print("[1/5] Metric distribution plots")
    print("=" * 60)
    from .metric_distributions import load_results, plot_hero_violin, plot_full_grid

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

    # ---- 2. Safety profile ----
    print("=" * 60)
    print("[2/5] Scenario safety profile")
    print("=" * 60)
    from .scenario_safety_profile import plot_safety_profile, plot_safety_pie

    plot_safety_profile(
        results, os.path.join(args.outdir, "scenario_safety_profile.pdf"), args.selector
    )
    plot_safety_profile(
        results, os.path.join(args.outdir, "scenario_safety_profile.png"), args.selector
    )
    plot_safety_pie(
        results, os.path.join(args.outdir, "scenario_safety_pie.pdf"), args.selector
    )
    plot_safety_pie(
        results, os.path.join(args.outdir, "scenario_safety_pie.png"), args.selector
    )

    # ---- 3. Summary table ----
    print("=" * 60)
    print("[3/5] Summary statistics table")
    print("=" * 60)
    from .summary_table import (
        generate_latex_table,
        generate_markdown_table,
        generate_summary_text,
    )

    latex = generate_latex_table(results, args.selector)
    with open(os.path.join(args.outdir, "closedloop_stats_table.tex"), "w") as f:
        f.write(latex)
    md = generate_markdown_table(results, args.selector)
    with open(os.path.join(args.outdir, "closedloop_stats_table.md"), "w") as f:
        f.write(md)
    txt = generate_summary_text(results, args.selector)
    with open(os.path.join(args.outdir, "closedloop_summary.txt"), "w") as f:
        f.write(txt)
    print(txt)

    # ---- 4. Heatmap ----
    print("=" * 60)
    print("[4/5] Scenario × metric heatmap")
    print("=" * 60)
    from .scenario_heatmap import plot_heatmap

    plot_heatmap(
        results, os.path.join(args.outdir, "scenario_metric_heatmap.pdf"), args.selector
    )
    plot_heatmap(
        results, os.path.join(args.outdir, "scenario_metric_heatmap.png"), args.selector
    )

    # ---- 5. BEV rollouts ----
    if not args.skip_bev and args.bev_scenarios:
        print("=" * 60)
        print(f"[5/5] BEV rollouts (scenarios: {args.bev_scenarios})")
        print("=" * 60)
        from .bev_rollout import render_scenario_rollout, BEVConfig

        bev_cfg = BEVConfig()
        for si in args.bev_scenarios:
            bev_outdir = os.path.join(
                os.path.dirname(args.outdir), "bev_frames", f"scenario_{si:02d}"
            )
            render_scenario_rollout(
                scenario_index=si,
                max_steps=80,
                outdir=bev_outdir,
                cfg=bev_cfg,
                make_video=True,
            )
    else:
        print("[5/5] BEV rollouts — skipped")

    elapsed = time.time() - t0
    print()
    print("=" * 60)
    print(f"All artifacts generated in {elapsed:.1f}s")
    print(f"Output directory: {args.outdir}")
    print("=" * 60)

    # List outputs
    for root, dirs, files in os.walk(args.outdir):
        for f in sorted(files):
            fpath = os.path.join(root, f)
            sz = os.path.getsize(fpath)
            print(f"  {os.path.relpath(fpath, args.outdir):50s}  {sz/1024:7.1f} KB")


if __name__ == "__main__":
    main()
