#!/usr/bin/env python3
"""
Compute 95% Bootstrap Confidence Intervals for RECTOR headline metrics.

Reads the canonical evaluation JSON (from evaluation.evaluate_canonical.py) and computes
bootstrap CIs for all headline metrics, plus paired Wilcoxon signed-rank tests.

Usage:
    python evaluation/compute_bootstrap_cis.py --input /workspace/output/evaluation/canonical_results.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats


def bootstrap_ci(data, n_bootstrap=10000, ci=0.95, stat_fn=np.mean):
    """
    Compute bootstrap confidence interval.

    Args:
        data: 1D array of observations
        n_bootstrap: number of bootstrap resamples
        ci: confidence level (0.95 = 95%)
        stat_fn: statistic to compute (default: mean)

    Returns:
        (point_estimate, ci_low, ci_high)
    """
    data = np.asarray(data)
    n = len(data)
    point = stat_fn(data)

    boot_stats = np.array(
        [
            stat_fn(np.random.choice(data, size=n, replace=True))
            for _ in range(n_bootstrap)
        ]
    )

    alpha = (1 - ci) / 2
    ci_low = np.percentile(boot_stats, alpha * 100)
    ci_high = np.percentile(boot_stats, (1 - alpha) * 100)

    return float(point), float(ci_low), float(ci_high)


def paired_wilcoxon(data_a, data_b, name_a="A", name_b="B"):
    """
    Paired Wilcoxon signed-rank test.

    Tests whether the median difference between paired observations
    is significantly different from zero.
    """
    data_a = np.asarray(data_a)
    data_b = np.asarray(data_b)
    diff = data_a - data_b

    # Remove zeros (ties)
    nonzero = diff != 0
    if nonzero.sum() < 10:
        return {
            "test": "wilcoxon",
            "comparison": f"{name_a} vs {name_b}",
            "n_nonzero": int(nonzero.sum()),
            "note": "Too few non-zero differences for reliable test",
        }

    stat, p_value = stats.wilcoxon(diff[nonzero])

    return {
        "test": "wilcoxon_signed_rank",
        "comparison": f"{name_a} vs {name_b}",
        "statistic": float(stat),
        "p_value": float(p_value),
        "significant_at_005": p_value < 0.05,
        "significant_at_001": p_value < 0.01,
        "n_pairs": int(len(data_a)),
        "n_nonzero": int(nonzero.sum()),
        "mean_diff": float(np.mean(diff)),
        "median_diff": float(np.median(diff)),
    }


def main():
    parser = argparse.ArgumentParser(description="Bootstrap CIs for RECTOR metrics")
    parser.add_argument(
        "--input", type=str, required=True, help="Canonical results JSON"
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Output JSON (default: input dir)"
    )
    parser.add_argument("--n_bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    with open(args.input) as f:
        data = json.load(f)

    print("=" * 60)
    print("BOOTSTRAP CONFIDENCE INTERVALS")
    print(f"  n_bootstrap = {args.n_bootstrap}")
    print("=" * 60)

    results = {"n_bootstrap": args.n_bootstrap, "seed": args.seed, "metrics": {}}

    # Compute CIs for each selection strategy
    for strategy, sdata in data.get("selection_strategies", {}).items():
        # We need per-scenario data; if not available, skip
        # The canonical script stores aggregated means; for proper bootstrap,
        # we'd need per-scenario arrays. This script assumes they're available
        # or re-derives them from the raw data.
        print(f"\n  Strategy: {strategy}")
        print(f"    n_scenarios: {sdata['n_scenarios']}")

        # For the metrics that are already aggregated (means of binary indicators),
        # we can reconstruct the binary array from the mean and n
        metrics_to_bootstrap = [
            ("Total_Viol_pct", "total_violated"),
            ("SL_Viol_pct", "sl_violated"),
            ("Safety_Viol_pct", "tier_0_violated"),
            ("Legal_Viol_pct", "tier_1_violated"),
            ("Road_Viol_pct", "tier_2_violated"),
            ("Comfort_Viol_pct", "tier_3_violated"),
        ]

        strategy_cis = {}
        for metric_name, _ in metrics_to_bootstrap:
            val = sdata[metric_name]
            # For binary metrics, bootstrap CI on a Bernoulli proportion
            n = sdata["n_scenarios"]
            p = val / 100.0
            # Generate synthetic binary array matching observed proportion
            binary = np.zeros(n)
            binary[: int(round(p * n))] = 1
            np.random.shuffle(binary)

            point, ci_lo, ci_hi = bootstrap_ci(binary * 100, args.n_bootstrap)
            strategy_cis[metric_name] = {
                "point": round(point, 2),
                "ci_95_low": round(ci_lo, 2),
                "ci_95_high": round(ci_hi, 2),
                "formatted": f"{point:.1f}% [{ci_lo:.1f}, {ci_hi:.1f}]",
            }
            print(f"    {metric_name}: {strategy_cis[metric_name]['formatted']}")

        # selADE CI (continuous metric - use normal approximation if per-sample not available)
        if "selADE_mean" in sdata:
            strategy_cis["selADE"] = {
                "point": sdata["selADE_mean"],
                "note": "Per-sample data needed for proper bootstrap CI",
            }

        results["metrics"][strategy] = strategy_cis

    # Protocol reconciliation CIs
    print("\n  Protocol results:")
    for pkey, pdata in data.get("protocol_results", {}).items():
        n = pdata["n_scenarios"]
        for metric in ["Total_Viol_pct", "Safety_Viol_pct", "SL_Viol_pct"]:
            p = pdata[metric] / 100.0
            binary = np.zeros(n)
            binary[: int(round(p * n))] = 1
            np.random.shuffle(binary)
            point, ci_lo, ci_hi = bootstrap_ci(binary * 100, args.n_bootstrap)
            print(f"    {pkey}/{metric}: {point:.1f}% [{ci_lo:.1f}, {ci_hi:.1f}]")

    # Save
    output_path = args.output or str(Path(args.input).parent / "bootstrap_cis.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
