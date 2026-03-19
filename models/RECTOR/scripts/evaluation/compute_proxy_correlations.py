#!/usr/bin/env python3
"""
Compute proxy vs. full evaluator correlations for all 24 proxied rules.

Runs RECTOR inference via ScenarioContextBuilder from full_rule_evaluation.py,
which handles ego-to-world coordinate alignment correctly. For each scenario:
  1. Run RECTOR inference to get K=6 candidate trajectories (ego-centric)
  2. Convert trajectories to world coordinates using the scenario reference frame
  3. Compute proxy violations on ego-centric trajectories
  4. Build ScenarioContext with world-coordinate trajectories
  5. Run full Waymo RuleExecutor on each candidate
  6. Compute per-rule Spearman rho, FPR, FNR, F1, and pairwise ranking accuracy

Usage:
    python evaluation/compute_proxy_correlations.py --max_scenarios 1000
"""

import os
import sys
import json
import glob
import time
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, "/workspace/data/WOMD")
sys.path.insert(0, "/workspace/data")
sys.path.insert(0, "/workspace/models/RECTOR/scripts")

import torch

import tensorflow as tf

tf.config.set_visible_devices([], "GPU")

from waymo_rule_eval.rules.rule_constants import (
    RULE_IDS,
    NUM_RULES,
    RULE_INDEX_MAP,
    TIER_0_SAFETY,
    TIER_1_LEGAL,
    TIER_2_ROAD,
    TIER_3_COMFORT,
    TIER_DEFINITIONS,
)
from waymo_rule_eval.pipeline.rule_executor import RuleExecutor

from scipy.stats import spearmanr

# Reuse the ScenarioContextBuilder from full_rule_evaluation
sys.path.insert(0, str(Path(__file__).parent.parent / "experiments"))
from full_rule_evaluation import (
    RECTORInference,
    ScenarioContextBuilder,
    get_tier_for_rule,
)

RULES_WITHOUT_PROXIES = {"L5.R2", "L5.R3", "L5.R4", "L5.R5"}
TRAJECTORY_SCALE = 50.0
TIER_NAMES = {0: "Safety", 1: "Legal", 2: "Road", 3: "Comfort"}

RULE_DESCRIPTIONS = {
    "L0.R2": "Longitudinal safe distance",
    "L0.R3": "Collision/overlap",
    "L0.R4": "Crosswalk clearance",
    "L10.R1": "VRU longitudinal distance",
    "L10.R2": "VRU lateral clearance",
    "L1.R1": "Longitudinal accel",
    "L1.R2": "Lateral acceleration",
    "L1.R3": "Combined acceleration",
    "L1.R4": "Longitudinal jerk",
    "L1.R5": "Lateral jerk",
    "L3.R3": "Lane keeping",
    "L4.R3": "Left turn gap",
    "L5.R1": "Traffic signal compliance",
    "L6.R1": "Following distance",
    "L6.R2": "Lateral gap to neighbors",
    "L6.R3": "Passing clearance",
    "L6.R4": "VRU lateral buffer",
    "L6.R5": "VRU longitudinal buffer",
    "L7.R3": "Drivable surface",
    "L7.R4": "Speed limit compliance",
    "L8.R1": "Signal stop line",
    "L8.R2": "Stop sign compliance",
    "L8.R3": "Crosswalk yield",
    "L8.R5": "Wrong-way driving",
}


def get_tier_idx(rule_id):
    for tier_idx, rules in TIER_DEFINITIONS.items():
        if rule_id in rules:
            return tier_idx
    return -1


def pairwise_concordance(proxy_scores, full_scores):
    """Compute pairwise ranking concordance between proxy and full evaluator.

    For K candidates, there are C(K,2) pairs. For each pair (i,j), check
    if proxy ranks them in the same order as the full evaluator.

    Args:
        proxy_scores: [K] proxy violation scores (lower = better)
        full_scores: [K] full evaluator violation scores (lower = better)

    Returns:
        concordance_rate: float in [0, 1]
        n_pairs: int
    """
    K = len(proxy_scores)
    concordant = 0
    discordant = 0
    tied = 0
    for i in range(K):
        for j in range(i + 1, K):
            p_diff = proxy_scores[i] - proxy_scores[j]
            f_diff = full_scores[i] - full_scores[j]
            if abs(p_diff) < 1e-8 or abs(f_diff) < 1e-8:
                tied += 1
            elif (p_diff > 0) == (f_diff > 0):
                concordant += 1
            else:
                discordant += 1
    total = concordant + discordant
    if total == 0:
        return 1.0, concordant + discordant + tied  # all tied = perfect agreement
    return concordant / total, concordant + discordant + tied


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="/workspace/models/RECTOR/models/best.pt")
    p.add_argument(
        "--val_dir",
        default="/workspace/data/WOMD/datasets/waymo_open_dataset/"
        "motion_v_1_3_0/processed/augmented/scenario/validation_interactive",
    )
    p.add_argument(
        "--output",
        default="/workspace/output/evaluation/proxy_correlations_all24_aligned.json",
    )
    p.add_argument("--max_scenarios", type=int, default=1000)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(42)
    device = args.device if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("PROXY vs FULL EVALUATOR CORRELATION — Aligned Coordinates")
    print("=" * 70)

    # Use the RECTORInference class which handles coordinate alignment correctly
    print("\n[1/4] Loading RECTOR model...")
    inference = RECTORInference(args.checkpoint, device=device)

    print("\n[2/4] Initializing Waymo RuleExecutor...")
    rule_executor = RuleExecutor()
    rule_executor.register_all_rules()
    print(f"  Registered {len(rule_executor.rules)} rules")

    context_builder = ScenarioContextBuilder()

    # Load scenario protos directly from TFRecords
    print(f"\n[3/4] Loading scenarios (max {args.max_scenarios})...")
    tfrecord_files = sorted(glob.glob(os.path.join(args.val_dir, "*")))

    # Per-rule accumulators: lists of (proxy_severity, full_severity, full_violated_bool)
    # across ALL K=6 candidates (not just selected)
    per_rule_proxy = defaultdict(list)
    per_rule_full_sev = defaultdict(list)
    per_rule_full_viol = defaultdict(list)

    # Pairwise ranking: per scenario, per rule
    per_rule_pairwise = defaultdict(list)  # rule -> list of concordance rates

    # Overall pairwise (aggregate across all rules in each tier)
    tier_pairwise = defaultdict(list)

    print(f"\n[4/4] Evaluating scenarios...")
    start = time.time()
    n_scenarios = 0
    n_failed = 0

    for tfr_path in tfrecord_files:
        if n_scenarios >= args.max_scenarios:
            break
        try:
            raw_dataset = tf.data.TFRecordDataset(tfr_path)
        except Exception:
            continue

        for record in raw_dataset:
            if n_scenarios >= args.max_scenarios:
                break

            try:
                # Parse TFRecord
                example = tf.train.Example.FromString(record.numpy())
                features = example.features.feature
                if "scenario/proto" not in features:
                    continue

                from waymo_open_dataset.protos import scenario_pb2

                scenario = scenario_pb2.Scenario()
                scenario.ParseFromString(features["scenario/proto"].bytes_list.value[0])

                # Run RECTOR inference → world-coordinate trajectories [K, T, 2]
                traj_world, confidences, applicability = (
                    inference.predict_from_scenario(scenario)
                )
                if traj_world is None:
                    n_failed += 1
                    continue

                K = traj_world.shape[0]  # 6

                # Also get ego-centric trajectories for proxy evaluation
                feats = inference._extract_features(scenario)
                if feats is None:
                    n_failed += 1
                    continue

                # Run proxy evaluation on ego-centric trajectories
                # Proxies expect [B, M, T, 4] in ego-centric METERS
                ego_traj_norm = (
                    torch.from_numpy(feats["ego_history"])
                    .unsqueeze(0)
                    .to(inference.device)
                )
                agent_states = (
                    torch.from_numpy(feats["agent_states"])
                    .unsqueeze(0)
                    .to(inference.device)
                )
                lane_centers = (
                    torch.from_numpy(feats["lane_centers"])
                    .unsqueeze(0)
                    .to(inference.device)
                )

                with torch.no_grad():
                    outputs = inference.model(
                        ego_history=ego_traj_norm,
                        agent_states=agent_states,
                        lane_centers=lane_centers,
                    )
                    trajectories = outputs["trajectories"]  # [1, K, T, 4]

                    # Compute proxy violations
                    from evaluation.evaluate_canonical import evaluate_proxy_violations

                    batch = {
                        "agent_states": agent_states.cpu(),
                        "lane_centers": lane_centers.cpu(),
                    }
                    proxy_violations = evaluate_proxy_violations(
                        inference.model, trajectories, batch, inference.device
                    )  # [1, K, NUM_RULES]
                    proxy_viols = proxy_violations[0].cpu().numpy()  # [K, NUM_RULES]

                # Run full evaluator on each of K candidates
                full_viols_per_mode = (
                    []
                )  # list of K dicts: {rule_id: (violated, severity)}
                for k in range(K):
                    try:
                        ctx = context_builder.build(scenario, traj_world[k])
                        if ctx is None:
                            full_viols_per_mode.append(None)
                            continue
                        result = rule_executor.evaluate(ctx)
                        viols = {}
                        for rr in result.rule_results:
                            sev = (
                                rr.severity
                                if hasattr(rr, "severity")
                                else (1.0 if rr.has_violation else 0.0)
                            )
                            viols[rr.rule_id] = (rr.has_violation, sev)
                        full_viols_per_mode.append(viols)
                    except Exception:
                        full_viols_per_mode.append(None)

                # Skip if too many modes failed
                valid_modes = [
                    i for i, v in enumerate(full_viols_per_mode) if v is not None
                ]
                if len(valid_modes) < 3:
                    n_failed += 1
                    continue

                # Collect per-rule data across all valid modes
                for rule_idx, rule_id in enumerate(RULE_IDS):
                    if rule_id in RULES_WITHOUT_PROXIES:
                        continue

                    proxy_vals = []
                    full_vals = []

                    for k in valid_modes:
                        p_sev = proxy_viols[k, rule_idx]
                        full_dict = full_viols_per_mode[k]
                        if rule_id in full_dict:
                            f_viol, f_sev = full_dict[rule_id]
                        else:
                            f_viol, f_sev = False, 0.0

                        per_rule_proxy[rule_id].append(float(p_sev))
                        per_rule_full_sev[rule_id].append(float(f_sev))
                        per_rule_full_viol[rule_id].append(bool(f_viol))

                        proxy_vals.append(float(p_sev))
                        full_vals.append(float(f_sev))

                    # Pairwise ranking for this rule in this scenario
                    if len(proxy_vals) >= 2:
                        conc, n_pairs = pairwise_concordance(
                            np.array(proxy_vals), np.array(full_vals)
                        )
                        per_rule_pairwise[rule_id].append(conc)

                n_scenarios += 1
                if n_scenarios % 50 == 0:
                    elapsed = time.time() - start
                    print(
                        f"  {n_scenarios} scenarios ({elapsed:.1f}s, "
                        f"{n_failed} failed)"
                    )

            except Exception as e:
                n_failed += 1
                continue

    elapsed = time.time() - start
    print(f"\n  Total: {n_scenarios} scenarios, {n_failed} failed ({elapsed:.1f}s)")

    # Compute per-rule statistics
    print("\n" + "=" * 70)
    print("PER-RULE PROXY-FULL CORRELATIONS (Aligned Coordinates)")
    print("=" * 70)

    results = {
        "metadata": {
            "n_scenarios": n_scenarios,
            "n_failed": n_failed,
            "elapsed_s": elapsed,
            "checkpoint": args.checkpoint,
        },
        "per_rule": {},
    }

    print(
        f"\n  {'Rule':<10} {'Tier':<8} {'rho':>6} {'FPR%':>6} {'FNR%':>6} "
        f"{'F1':>6} {'PairConc':>8} {'N':>6}"
    )
    print("  " + "-" * 60)

    all_rhos = []
    all_f1s = []
    all_conc = []

    for rule_id in RULE_IDS:
        if rule_id in RULES_WITHOUT_PROXIES:
            continue

        proxy_arr = np.array(per_rule_proxy.get(rule_id, []))
        full_sev_arr = np.array(per_rule_full_sev.get(rule_id, []))
        full_viol_arr = np.array(per_rule_full_viol.get(rule_id, []))

        n = len(proxy_arr)
        tier_idx = get_tier_idx(rule_id)
        tier = TIER_NAMES.get(tier_idx, "?")

        # Spearman correlation
        rho = np.nan
        if n > 10 and np.std(proxy_arr) > 1e-8 and np.std(full_sev_arr) > 1e-8:
            rho, _ = spearmanr(proxy_arr, full_sev_arr)
        elif n > 10 and np.std(proxy_arr) > 1e-8 and np.std(full_viol_arr) > 0:
            rho, _ = spearmanr(proxy_arr, full_viol_arr.astype(float))

        # Binary classification metrics
        proxy_binary = proxy_arr > 0.01
        tp = ((proxy_binary) & (full_viol_arr)).sum() if n > 0 else 0
        fp = ((proxy_binary) & (~full_viol_arr)).sum() if n > 0 else 0
        fn = ((~proxy_binary) & (full_viol_arr)).sum() if n > 0 else 0
        tn = ((~proxy_binary) & (~full_viol_arr)).sum() if n > 0 else 0
        fpr = 100.0 * fp / max(fp + tn, 1)
        fnr = 100.0 * fn / max(fn + tp, 1)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)

        # Pairwise concordance
        conc_list = per_rule_pairwise.get(rule_id, [])
        mean_conc = float(np.mean(conc_list)) if conc_list else np.nan

        rho_str = f"{rho:6.3f}" if not np.isnan(rho) else "   nan"
        conc_str = f"{mean_conc:7.3f}" if not np.isnan(mean_conc) else "    nan"
        print(
            f"  {rule_id:<10} {tier:<8} {rho_str} {fpr:5.1f}% {fnr:5.1f}% "
            f"{f1:5.3f} {conc_str} {n:>6}"
        )

        if not np.isnan(rho):
            all_rhos.append(rho)
        if f1 > 0:
            all_f1s.append(f1)
        if not np.isnan(mean_conc):
            all_conc.append(mean_conc)

        results["per_rule"][rule_id] = {
            "tier": tier,
            "tier_idx": tier_idx,
            "description": RULE_DESCRIPTIONS.get(rule_id, ""),
            "spearman_rho": float(rho) if not np.isnan(rho) else None,
            "fpr_pct": float(fpr),
            "fnr_pct": float(fnr),
            "f1": float(f1),
            "pairwise_concordance": (
                float(mean_conc) if not np.isnan(mean_conc) else None
            ),
            "n_samples": n,
            "proxy_violation_rate": float(proxy_binary.mean() * 100) if n > 0 else 0,
            "full_violation_rate": float(full_viol_arr.mean() * 100) if n > 0 else 0,
        }

    # Summary
    print(
        f"\n  Mean Spearman rho: {np.mean(all_rhos):.3f} "
        f"(range: {min(all_rhos):.3f} to {max(all_rhos):.3f}, "
        f"n={len(all_rhos)} rules with variance)"
    )
    print(f"  Mean F1: {np.mean(all_f1s):.3f} (n={len(all_f1s)} rules with >0)")
    print(
        f"  Mean pairwise concordance: {np.mean(all_conc):.3f} "
        f"(n={len(all_conc)} rules)"
    )

    results["summary"] = {
        "mean_spearman_rho": float(np.mean(all_rhos)) if all_rhos else None,
        "min_spearman_rho": float(min(all_rhos)) if all_rhos else None,
        "max_spearman_rho": float(max(all_rhos)) if all_rhos else None,
        "mean_f1": float(np.mean(all_f1s)) if all_f1s else None,
        "mean_pairwise_concordance": float(np.mean(all_conc)) if all_conc else None,
        "n_rules_with_rho": len(all_rhos),
        "n_rules_with_f1": len(all_f1s),
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
