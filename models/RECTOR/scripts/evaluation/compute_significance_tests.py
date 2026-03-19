#!/usr/bin/env python3
"""
Scenario-level significance tests for RECTOR selection strategy comparisons.

Reads per-scenario CSVs (43,219 rows each) and computes:
  1. McNemar's test on binary Safety/Total violation indicators (lex vs confidence)
  2. Wilcoxon signed-rank on continuous selADE (lex vs confidence)
  3. Bootstrap 95% CIs from actual per-scenario data
  4. Confirms lex vs WS produces p >> 0.05 (identical selections)

Usage:
    python evaluation/compute_significance_tests.py
"""

import argparse
import json
import csv
from pathlib import Path

import numpy as np
from scipy import stats


def load_per_scenario_csv(path):
    """Load per-scenario CSV into dict of numpy arrays."""
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    n = len(rows)
    result = {
        "selADE": np.array([float(r["selADE"]) for r in rows]),
        "selFDE": np.array([float(r["selFDE"]) for r in rows]),
        "total_violated": np.array([r["total_violated"] == "True" for r in rows]),
        "sl_violated": np.array([r["sl_violated"] == "True" for r in rows]),
        "tier_0_violated": np.array([r["tier_0_violated"] == "True" for r in rows]),
        "tier_1_violated": np.array([r["tier_1_violated"] == "True" for r in rows]),
        "tier_2_violated": np.array([r["tier_2_violated"] == "True" for r in rows]),
        "tier_3_violated": np.array([r["tier_3_violated"] == "True" for r in rows]),
    }
    return result, n


def mcnemar_test(a_violated, b_violated, name_a="A", name_b="B"):
    """McNemar's test for paired binary outcomes.

    Tests whether the marginal frequencies of violations differ between two strategies.
    More powerful than chi-squared for paired data.
    """
    a = np.asarray(a_violated, dtype=bool)
    b = np.asarray(b_violated, dtype=bool)

    # Discordant pairs
    b_count = int((a & ~b).sum())  # a violated, b did not (a worse)
    c_count = int((~a & b).sum())  # b violated, a did not (b worse)
    # Concordant pairs
    a_count = int((a & b).sum())  # both violated
    d_count = int((~a & ~b).sum())  # neither violated

    n = len(a)
    total_discordant = b_count + c_count

    if total_discordant == 0:
        return {
            "test": "mcnemar",
            "comparison": f"{name_a} vs {name_b}",
            "n": n,
            "both_violated": a_count,
            "neither_violated": d_count,
            "a_only": b_count,
            "b_only": c_count,
            "note": "No discordant pairs — strategies produce identical violations",
            "p_value": 1.0,
        }

    # McNemar's with continuity correction
    chi2_stat = (abs(b_count - c_count) - 1) ** 2 / (b_count + c_count)
    p_value = 1 - stats.chi2.cdf(chi2_stat, df=1)

    # Exact binomial test (more accurate for small discordant counts)
    if total_discordant < 25:
        p_exact = stats.binom_test(min(b_count, c_count), total_discordant, 0.5)
    else:
        p_exact = None

    return {
        "test": "mcnemar",
        "comparison": f"{name_a} vs {name_b}",
        "n": n,
        "both_violated": a_count,
        "neither_violated": d_count,
        "a_only_violated": b_count,
        "b_only_violated": c_count,
        "chi2_statistic": float(chi2_stat),
        "p_value": float(p_value),
        "p_exact": float(p_exact) if p_exact is not None else None,
        "significant_005": p_value < 0.05,
        "significant_001": p_value < 0.01,
        "direction": f'{name_a} has {"fewer" if b_count < c_count else "more"} unique violations',
    }


def paired_wilcoxon(a_vals, b_vals, name_a="A", name_b="B"):
    """Paired Wilcoxon signed-rank test on continuous values."""
    a = np.asarray(a_vals)
    b = np.asarray(b_vals)
    diff = a - b
    nonzero = diff != 0

    if nonzero.sum() < 10:
        return {
            "test": "wilcoxon_signed_rank",
            "comparison": f"{name_a} vs {name_b}",
            "n_pairs": int(len(a)),
            "n_nonzero": int(nonzero.sum()),
            "note": "Too few non-zero differences",
        }

    stat, p_value = stats.wilcoxon(diff[nonzero])
    return {
        "test": "wilcoxon_signed_rank",
        "comparison": f"{name_a} vs {name_b}",
        "statistic": float(stat),
        "p_value": float(p_value),
        "significant_005": p_value < 0.05,
        "significant_001": p_value < 0.01,
        "n_pairs": int(len(a)),
        "n_nonzero": int(nonzero.sum()),
        "mean_diff": float(np.mean(diff)),
        "median_diff": float(np.median(diff)),
    }


def bootstrap_ci(data, n_bootstrap=10000, ci=0.95):
    """Bootstrap CI on the mean of data."""
    data = np.asarray(data)
    n = len(data)
    point = float(np.mean(data))
    boot_means = np.array(
        [
            np.mean(np.random.choice(data, size=n, replace=True))
            for _ in range(n_bootstrap)
        ]
    )
    alpha = (1 - ci) / 2
    lo = float(np.percentile(boot_means, alpha * 100))
    hi = float(np.percentile(boot_means, (1 - alpha) * 100))
    return point, lo, hi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dir", default="/workspace/output/evaluation")
    parser.add_argument(
        "--output", default="/workspace/output/evaluation/significance_tests.json"
    )
    parser.add_argument("--n_bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    eval_dir = Path(args.eval_dir)

    print("=" * 70)
    print("SCENARIO-LEVEL SIGNIFICANCE TESTS")
    print("=" * 70)

    # Load per-scenario data
    print("\nLoading per-scenario CSVs...")
    conf_data, n_conf = load_per_scenario_csv(eval_dir / "per_scenario_confidence.csv")
    lex_data, n_lex = load_per_scenario_csv(eval_dir / "per_scenario_lexicographic.csv")
    ws_data, n_ws = load_per_scenario_csv(eval_dir / "per_scenario_weighted_sum.csv")
    print(f"  Loaded: confidence={n_conf}, lexicographic={n_lex}, weighted_sum={n_ws}")

    assert n_conf == n_lex == n_ws, "Row counts must match across strategies"

    results = {
        "metadata": {"n_scenarios": n_conf, "n_bootstrap": args.n_bootstrap},
        "mcnemar_tests": {},
        "wilcoxon_tests": {},
        "bootstrap_cis": {},
    }

    # --- McNemar's tests on binary violations ---
    print("\n--- McNemar's Tests (binary violations) ---")
    for tier_name, tier_key in [
        ("Safety", "tier_0_violated"),
        ("Total", "total_violated"),
        ("S+L", "sl_violated"),
    ]:
        # Lex vs Confidence
        test = mcnemar_test(
            lex_data[tier_key], conf_data[tier_key], "Lexicographic", "Confidence"
        )
        results["mcnemar_tests"][f"{tier_name}_lex_vs_conf"] = test
        print(
            f"  {tier_name} (Lex vs Conf): p={test['p_value']:.2e}, "
            f"discordant=({test.get('a_only_violated', 0)}, {test.get('b_only_violated', 0)})"
        )

        # Lex vs WS
        test = mcnemar_test(
            lex_data[tier_key], ws_data[tier_key], "Lexicographic", "WeightedSum"
        )
        results["mcnemar_tests"][f"{tier_name}_lex_vs_ws"] = test
        print(f"  {tier_name} (Lex vs WS):   p={test['p_value']:.2e}")

    # --- Wilcoxon tests on continuous metrics ---
    print("\n--- Wilcoxon Signed-Rank Tests (continuous metrics) ---")
    for metric_name, metric_key in [("selADE", "selADE"), ("selFDE", "selFDE")]:
        test = paired_wilcoxon(
            lex_data[metric_key], conf_data[metric_key], "Lexicographic", "Confidence"
        )
        results["wilcoxon_tests"][f"{metric_name}_lex_vs_conf"] = test
        print(
            f"  {metric_name} (Lex vs Conf): p={test.get('p_value', 'N/A')}, "
            f"mean_diff={test.get('mean_diff', 'N/A')}"
        )

        test = paired_wilcoxon(
            lex_data[metric_key], ws_data[metric_key], "Lexicographic", "WeightedSum"
        )
        results["wilcoxon_tests"][f"{metric_name}_lex_vs_ws"] = test
        print(f"  {metric_name} (Lex vs WS):   p={test.get('p_value', 'N/A')}")

    # --- Bootstrap CIs from actual per-scenario data ---
    print(f"\n--- Bootstrap CIs (n={args.n_bootstrap} resamples) ---")
    for strategy_name, strategy_data in [
        ("confidence", conf_data),
        ("lexicographic", lex_data),
        ("weighted_sum", ws_data),
    ]:
        cis = {}
        for metric_name, metric_key in [
            ("Safety_Viol_pct", "tier_0_violated"),
            ("Total_Viol_pct", "total_violated"),
            ("SL_Viol_pct", "sl_violated"),
            ("selADE", "selADE"),
        ]:
            data = strategy_data[metric_key]
            if metric_key.endswith("_violated"):
                data = data.astype(float) * 100  # convert to percentage
            point, lo, hi = bootstrap_ci(data, args.n_bootstrap)
            cis[metric_name] = {
                "point": round(point, 3),
                "ci_95_low": round(lo, 3),
                "ci_95_high": round(hi, 3),
            }
            unit = "%" if metric_key.endswith("_violated") else "m"
            print(
                f"  {strategy_name}/{metric_name}: "
                f"{point:.2f}{unit} [{lo:.2f}, {hi:.2f}]"
            )

        results["bootstrap_cis"][strategy_name] = cis

    # Save (convert numpy types to Python natives for JSON)
    def to_native(obj):
        if isinstance(obj, (np.bool_, np.integer)):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, dict):
            return {k: to_native(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_native(v) for v in obj]
        return obj

    with open(args.output, "w") as f:
        json.dump(to_native(results), f, indent=2)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
