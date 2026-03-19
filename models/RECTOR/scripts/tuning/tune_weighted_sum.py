#!/usr/bin/env python3
"""
Tune the weighted-sum baseline via grid search.

Searches over per-tier weights to find the combination that minimizes
S+L (Safety + Legal) union violation rate on a held-out tuning set.

Usage:
    python tuning/tune_weighted_sum.py --input /workspace/output/evaluation/canonical_results.json

Note: This script operates on pre-computed per-mode violation data from
evaluate_canonical.py. For a full implementation, it needs per-mode tier
scores saved in the canonical JSON (extend evaluate_canonical.py to save
per-mode data).

For now, this script provides the infrastructure and can be run once
evaluate_canonical.py is extended to save per-scenario, per-mode tier scores.
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np


def grid_search_weights(
    tier_scores_all,  # [N, M, 4] per-scenario, per-mode tier scores
    gt_positions,  # [N, T, 2] ground truth positions (for selADE)
    pred_positions,  # [N, M, T, 2] predicted positions
    tune_fraction=0.2,
    seed=42,
):
    """
    Grid search over per-tier weights.

    Args:
        tier_scores_all: [N, M, 4] tier scores for all scenarios
        gt_positions: [N, T, 2] ground truth
        pred_positions: [N, M, T, 2] predictions
        tune_fraction: fraction held out for tuning
        seed: random seed for split

    Returns:
        best_weights, results_table
    """
    np.random.seed(seed)
    N = tier_scores_all.shape[0]

    # Split into tune/eval
    indices = np.random.permutation(N)
    n_tune = int(N * tune_fraction)
    tune_idx = indices[:n_tune]
    eval_idx = indices[n_tune:]

    tune_scores = tier_scores_all[tune_idx]
    eval_scores = tier_scores_all[eval_idx]

    # Weight grid
    w_safety_range = [1, 10, 100, 500, 1000, 5000, 10000]
    w_legal_range = [1, 10, 50, 100, 500, 1000]
    w_road_range = [1, 5, 10, 50, 100]
    w_comfort_range = [1]  # Fix comfort at 1

    best_sl = float("inf")
    best_weights = None
    results = []

    print(
        f"Grid search: {len(w_safety_range)} × {len(w_legal_range)} × "
        f"{len(w_road_range)} × {len(w_comfort_range)} = "
        f"{len(w_safety_range)*len(w_legal_range)*len(w_road_range)*len(w_comfort_range)} combinations"
    )

    for w_s, w_l, w_r, w_c in itertools.product(
        w_safety_range, w_legal_range, w_road_range, w_comfort_range
    ):
        weights = np.array([w_s, w_l, w_r, w_c], dtype=np.float64)

        # Select on tuning set
        weighted = (tune_scores * weights[None, None, :]).sum(axis=-1)  # [N_tune, M]
        selected = weighted.argmin(axis=1)  # [N_tune]

        # Compute S+L violation rate
        sel_scores = tune_scores[np.arange(len(tune_idx)), selected]  # [N_tune, 4]
        safety_violated = sel_scores[:, 0] > 0.01
        legal_violated = sel_scores[:, 1] > 0.01
        sl_violated = safety_violated | legal_violated
        sl_rate = sl_violated.mean() * 100

        total_violated = (sel_scores > 0.01).any(axis=1)
        total_rate = total_violated.mean() * 100

        results.append(
            {
                "weights": [int(w_s), int(w_l), int(w_r), int(w_c)],
                "tune_SL_pct": float(sl_rate),
                "tune_Total_pct": float(total_rate),
            }
        )

        if sl_rate < best_sl:
            best_sl = sl_rate
            best_weights = weights

    # Evaluate best weights on eval set
    print(f"\nBest weights (tuning): {best_weights.tolist()}")
    print(f"  Tuning S+L: {best_sl:.2f}%")

    weighted_eval = (eval_scores * best_weights[None, None, :]).sum(axis=-1)
    selected_eval = weighted_eval.argmin(axis=1)
    sel_eval_scores = eval_scores[np.arange(len(eval_idx)), selected_eval]

    eval_safety = (sel_eval_scores[:, 0] > 0.01).mean() * 100
    eval_legal = (sel_eval_scores[:, 1] > 0.01).mean() * 100
    eval_sl = (
        (sel_eval_scores[:, 0] > 0.01) | (sel_eval_scores[:, 1] > 0.01)
    ).mean() * 100
    eval_total = (sel_eval_scores > 0.01).any(axis=1).mean() * 100

    print(f"  Eval S+L: {eval_sl:.2f}%")
    print(f"  Eval Total: {eval_total:.2f}%")
    print(f"  Eval Safety: {eval_safety:.2f}%")
    print(f"  Eval Legal: {eval_legal:.2f}%")

    # Also evaluate default weights (1000/100/10/1)
    default_weights = np.array([1000, 100, 10, 1], dtype=np.float64)
    weighted_default = (eval_scores * default_weights[None, None, :]).sum(axis=-1)
    selected_default = weighted_default.argmin(axis=1)
    sel_default_scores = eval_scores[np.arange(len(eval_idx)), selected_default]
    default_sl = (
        (sel_default_scores[:, 0] > 0.01) | (sel_default_scores[:, 1] > 0.01)
    ).mean() * 100
    default_total = (sel_default_scores > 0.01).any(axis=1).mean() * 100

    print(f"\nDefault weights (1000/100/10/1):")
    print(f"  Eval S+L: {default_sl:.2f}%")
    print(f"  Eval Total: {default_total:.2f}%")

    return {
        "best_weights": best_weights.tolist(),
        "n_tune": int(n_tune),
        "n_eval": int(len(eval_idx)),
        "tuned": {
            "SL_pct": float(eval_sl),
            "Total_pct": float(eval_total),
            "Safety_pct": float(eval_safety),
            "Legal_pct": float(eval_legal),
        },
        "default": {
            "SL_pct": float(default_sl),
            "Total_pct": float(default_total),
        },
        "gap_closed": float(default_sl - eval_sl),
        "top_10_configs": sorted(results, key=lambda x: x["tune_SL_pct"])[:10],
    }


def main():
    parser = argparse.ArgumentParser(description="Tune weighted-sum baseline")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Canonical results JSON (must contain per-mode tier scores)",
    )
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--tune_fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 60)
    print("WEIGHTED-SUM BASELINE TUNING")
    print("=" * 60)

    # Load data
    with open(args.input) as f:
        data = json.load(f)

    # Check if per-mode tier scores are available
    if "per_scenario_per_mode" not in data:
        print(
            "\nERROR: The canonical results JSON does not contain per-scenario, per-mode"
        )
        print("tier scores. To enable this, extend evaluate_canonical.py to save:")
        print("  per_scenario_per_mode: [N, M, 4] array of tier scores")
        print("\nFor now, generating a placeholder with the infrastructure ready.")
        print("Re-run after extending evaluate_canonical.py.")

        # Save placeholder
        output_path = args.output or str(
            Path(args.input).parent / "weighted_sum_tuning.json"
        )
        with open(output_path, "w") as f:
            json.dump(
                {
                    "status": "pending",
                    "reason": "Per-mode tier scores not yet saved in canonical JSON",
                    "instruction": "Extend evaluate_canonical.py to save per_scenario_per_mode data",
                },
                f,
                indent=2,
            )
        print(f"Placeholder saved to: {output_path}")
        return

    tier_scores = np.array(data["per_scenario_per_mode"]["tier_scores"])

    results = grid_search_weights(
        tier_scores,
        gt_positions=None,
        pred_positions=None,
        tune_fraction=args.tune_fraction,
        seed=args.seed,
    )

    output_path = args.output or str(
        Path(args.input).parent / "weighted_sum_tuning.json"
    )
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
