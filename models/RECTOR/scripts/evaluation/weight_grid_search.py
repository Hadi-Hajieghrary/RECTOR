#!/usr/bin/env python3
"""
Tier-weight grid search for weighted-sum selection baseline.

Sweeps tier-level weight configurations to find the optimal weighted-sum
parameters, providing a properly-tuned weighted-sum baseline for fair
comparison against lexicographic selection.

This addresses reviewer concern: "Is the weighted-sum baseline properly
tuned, or is RECTOR only winning because weighted-sum uses default weights?"

Usage:
    python evaluation/weight_grid_search.py --checkpoint /path/to/best.pt
    python evaluation/weight_grid_search.py --checkpoint /path/to/best.pt --max_batches 10  # Quick test
    python evaluation/weight_grid_search.py --checkpoint /path/to/best.pt --metric Total_Viol_pct  # Optimize for total violations
"""

import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import argparse
import glob
import json
import time
import itertools
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# Add model paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, "/workspace/data/WOMD")
sys.path.insert(0, "/workspace/data")
sys.path.insert(0, "/workspace/models/RECTOR/scripts")

from training.train_rector import WaymoDataset, collate_fn, TRAJECTORY_SCALE
from models.rule_aware_generator import RuleAwareGenerator
from waymo_rule_eval.rules.rule_constants import (
    TIER_DEFINITIONS,
    NUM_RULES,
    RULE_IDS,
    RULE_INDEX_MAP,
    TIER_WEIGHTS,
)

# Reuse core functions from evaluate_canonical
from evaluation.evaluate_canonical import (
    compute_tier_scores,
    compute_violation_rates,
    evaluate_proxy_violations,
    select_by_weighted_sum,
    select_by_lexicographic,
    PROXY_RULE_MASK,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Weight grid search for weighted-sum selection"
    )
    parser.add_argument(
        "--checkpoint", type=str, default="/workspace/models/RECTOR/models/best.pt"
    )
    parser.add_argument(
        "--val_dir",
        type=str,
        default="/workspace/data/WOMD/datasets/waymo_open_dataset/motion_v_1_3_0/processed/augmented/scenario/validation_interactive",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/workspace/output/evaluation/weight_grid_search_results.json",
    )
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--decoder_hidden_dim", type=int, default=256)
    parser.add_argument("--decoder_num_layers", type=int, default=4)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--num_modes", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--use_m2i_encoder", action="store_true", default=True)
    parser.add_argument(
        "--m2i_checkpoint",
        type=str,
        default="/workspace/models/pretrained/m2i/models/relation_v2v/model.25.bin",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epsilon", type=float, default=1e-3)

    # Grid search config
    parser.add_argument(
        "--metric",
        type=str,
        default="SL_Viol_pct",
        choices=[
            "Total_Viol_pct",
            "SL_Viol_pct",
            "Safety_Viol_pct",
            "selADE_mean",
            "pareto",
        ],
        help="Metric to optimize (pareto = multi-objective frontier)",
    )
    parser.add_argument(
        "--safety_weights",
        nargs="+",
        type=float,
        default=[100, 500, 1000, 2000, 5000],
        help="Safety tier weight candidates",
    )
    parser.add_argument(
        "--legal_weights",
        nargs="+",
        type=float,
        default=[10, 50, 100, 200, 500],
        help="Legal tier weight candidates",
    )
    parser.add_argument(
        "--road_weights",
        nargs="+",
        type=float,
        default=[1, 5, 10, 20, 50],
        help="Road tier weight candidates",
    )
    parser.add_argument(
        "--comfort_weight",
        type=float,
        default=1.0,
        help="Comfort tier weight (fixed reference)",
    )

    # Misspecification sweep: add flat/low safety weights to test where WS
    # begins to diverge from lexicographic selection.  The standard grid
    # uses safety_weights >= 100 (all "lex-equivalent" configurations).
    # This flag prepends [1, 5, 10, 20, 50] to safety_weights so that the
    # sweep also covers the misspecification regime where WS may trade Safety
    # violations for Comfort gains.
    parser.add_argument(
        "--include_misspecification",
        action="store_true",
        default=False,
        help="Prepend misspecification safety weights [1,5,10,20,50] "
        "to the safety_weights list to test WS failure regime",
    )
    return parser.parse_args()


def evaluate_weight_config(
    tier_weights,
    cached_data,
    proxy_rule_mask,
    device,
):
    """
    Evaluate a single weight configuration on cached model outputs.

    Args:
        tier_weights: [4] tensor of tier weights
        cached_data: list of dicts with precomputed violations, applicability, etc.
        proxy_rule_mask: boolean mask for proxy rules
        device: torch device

    Returns:
        dict with aggregated metrics for this weight configuration
    """
    all_selADE = []
    all_selFDE = []
    all_violated = []
    all_sl_violated = []
    tier_violated = {t: [] for t in range(4)}

    for batch_data in cached_data:
        violations = batch_data["proxy_violations"]
        app = batch_data["applicability"]
        pred_pos = batch_data["pred_pos"]
        gt_pos = batch_data["gt_pos"]
        B = violations.shape[0]

        tier_scores = compute_tier_scores(violations, app, proxy_rule_mask)

        # Weighted-sum selection with custom weights
        sel = select_by_weighted_sum(tier_scores, tier_weights)

        # Metrics
        sel_pred = pred_pos[torch.arange(B, device=device), sel]
        selADE = (sel_pred - gt_pos).norm(dim=-1).mean(dim=-1) * TRAJECTORY_SCALE
        selFDE = (sel_pred[:, -1] - gt_pos[:, -1]).norm(dim=-1) * TRAJECTORY_SCALE

        sel_viols = violations[torch.arange(B, device=device), sel]
        rates = compute_violation_rates(sel_viols, app, proxy_rule_mask)

        all_selADE.extend(selADE.cpu().tolist())
        all_selFDE.extend(selFDE.cpu().tolist())
        all_violated.extend(rates["total_union"].tolist())
        all_sl_violated.extend(rates["sl_union"].tolist())
        for t in range(4):
            tier_violated[t].extend(rates[f"tier_{t}_union"].tolist())

    return {
        "selADE_mean": float(np.mean(all_selADE)),
        "selFDE_mean": float(np.mean(all_selFDE)),
        "Total_Viol_pct": float(np.mean(all_violated) * 100),
        "SL_Viol_pct": float(np.mean(all_sl_violated) * 100),
        "Safety_Viol_pct": float(np.mean(tier_violated[0]) * 100),
        "Legal_Viol_pct": float(np.mean(tier_violated[1]) * 100),
        "Road_Viol_pct": float(np.mean(tier_violated[2]) * 100),
        "Comfort_Viol_pct": float(np.mean(tier_violated[3]) * 100),
    }


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("WEIGHT GRID SEARCH — Weighted-Sum Baseline Tuning")
    print("=" * 70)

    # Optionally prepend misspecification safety weights
    safety_weights = list(args.safety_weights)
    if args.include_misspecification:
        misspec_weights = [w for w in [1, 5, 10, 20, 50] if w not in safety_weights]
        safety_weights = misspec_weights + safety_weights
        print("\n[MISSPECIFICATION MODE] Extended safety weights: " f"{safety_weights}")

    # Build weight grid
    weight_grid = list(
        itertools.product(safety_weights, args.legal_weights, args.road_weights)
    )
    print(
        f"\nGrid: {len(safety_weights)} × {len(args.legal_weights)} × "
        f"{len(args.road_weights)} = {len(weight_grid)} configurations"
    )

    # Load model
    print("\n[1/4] Loading model...")
    model = RuleAwareGenerator(
        embed_dim=args.embed_dim,
        decoder_hidden_dim=args.decoder_hidden_dim,
        decoder_num_layers=args.decoder_num_layers,
        latent_dim=args.latent_dim,
        num_modes=args.num_modes,
        dropout=args.dropout,
        use_m2i_encoder=args.use_m2i_encoder,
        m2i_checkpoint=args.m2i_checkpoint,
        freeze_m2i=True,
        trajectory_length=50,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Load data
    print("\n[2/4] Loading validation data...")
    val_files = sorted(glob.glob(os.path.join(args.val_dir, "*")))
    if not val_files:
        print(f"ERROR: No files in {args.val_dir}")
        return

    dataset = WaymoDataset(val_files, is_training=False)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Cache model outputs (run inference once, sweep weights without re-inference)
    print("\n[3/4] Caching model outputs (single forward pass)...")
    cached_data = []
    sample_count = 0
    start_time = time.time()

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if not batch:
                continue
            if args.max_batches and batch_idx >= args.max_batches:
                break

            ego_history = batch["ego_history"].to(device)
            agent_states = batch["agent_states"].to(device)
            lane_centers = batch["lane_centers"].to(device)
            traj_gt = batch["traj_gt"].to(device)

            outputs = model(
                ego_history=ego_history,
                agent_states=agent_states,
                lane_centers=lane_centers,
                traj_gt=traj_gt,
            )

            applicability_pred = outputs["applicability_prob"]
            applicability_binary = (applicability_pred > 0.5).float()
            proxy_violations = evaluate_proxy_violations(
                model, outputs["trajectories"], batch, device
            )

            cached_data.append(
                {
                    "proxy_violations": proxy_violations,
                    "applicability": applicability_binary,
                    "pred_pos": outputs["trajectories"][:, :, :, :2],
                    "gt_pos": traj_gt[:, :, :2],
                    "confidence": outputs[
                        "confidence"
                    ],  # [B, M] needed for lex tiebreaker
                }
            )
            sample_count += ego_history.shape[0]
            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx+1}: {sample_count} samples")

    cache_time = time.time() - start_time
    print(f"  Cached {sample_count} samples in {cache_time:.1f}s")

    # Also compute lexicographic baseline for comparison
    lex_result = None
    lex_data = {
        "all_selADE": [],
        "all_selFDE": [],
        "all_violated": [],
        "all_sl": [],
        "tier": {t: [] for t in range(4)},
    }
    with torch.no_grad():
        for batch_data in cached_data:
            violations = batch_data["proxy_violations"]
            app = batch_data["applicability"]
            pred_pos = batch_data["pred_pos"]
            gt_pos = batch_data["gt_pos"]
            B = violations.shape[0]
            ts = compute_tier_scores(violations, app, PROXY_RULE_MASK)
            conf = batch_data["confidence"]
            sel, _ = select_by_lexicographic(ts, args.epsilon, confidence=conf)
            sel_pred = pred_pos[torch.arange(B, device=device), sel]
            selADE = (sel_pred - gt_pos).norm(dim=-1).mean(dim=-1) * TRAJECTORY_SCALE
            selFDE = (sel_pred[:, -1] - gt_pos[:, -1]).norm(dim=-1) * TRAJECTORY_SCALE
            sel_v = violations[torch.arange(B, device=device), sel]
            rates = compute_violation_rates(sel_v, app, PROXY_RULE_MASK)
            lex_data["all_selADE"].extend(selADE.cpu().tolist())
            lex_data["all_selFDE"].extend(selFDE.cpu().tolist())
            lex_data["all_violated"].extend(rates["total_union"].tolist())
            lex_data["all_sl"].extend(rates["sl_union"].tolist())
            for t in range(4):
                lex_data["tier"][t].extend(rates[f"tier_{t}_union"].tolist())
    lex_result = {
        "selADE_mean": float(np.mean(lex_data["all_selADE"])),
        "selFDE_mean": float(np.mean(lex_data["all_selFDE"])),
        "Total_Viol_pct": float(np.mean(lex_data["all_violated"]) * 100),
        "SL_Viol_pct": float(np.mean(lex_data["all_sl"]) * 100),
    }

    # Sweep weight configurations
    print(f"\n[4/4] Sweeping {len(weight_grid)} weight configurations...")
    results = []
    sweep_start = time.time()

    for i, (w_safety, w_legal, w_road) in enumerate(weight_grid):
        tier_weights = torch.tensor(
            [w_safety, w_legal, w_road, args.comfort_weight], dtype=torch.float32
        ).to(device)

        metrics = evaluate_weight_config(
            tier_weights,
            cached_data,
            PROXY_RULE_MASK,
            device,
        )
        metrics["weights"] = {
            "safety": w_safety,
            "legal": w_legal,
            "road": w_road,
            "comfort": args.comfort_weight,
        }
        results.append(metrics)

        if (i + 1) % 25 == 0:
            elapsed = time.time() - sweep_start
            print(f"  Config {i+1}/{len(weight_grid)} ({elapsed:.1f}s)")

    sweep_time = time.time() - sweep_start
    print(f"  Sweep complete in {sweep_time:.1f}s")

    # Find best configuration
    if args.metric == "pareto":
        # Pareto frontier: minimize both violation rate and selADE
        pareto_front = []
        for r in results:
            dominated = False
            for other in results:
                if (
                    other["SL_Viol_pct"] <= r["SL_Viol_pct"]
                    and other["selADE_mean"] <= r["selADE_mean"]
                    and (
                        other["SL_Viol_pct"] < r["SL_Viol_pct"]
                        or other["selADE_mean"] < r["selADE_mean"]
                    )
                ):
                    dominated = True
                    break
            if not dominated:
                pareto_front.append(r)
        best = min(pareto_front, key=lambda r: r["SL_Viol_pct"])
        best_label = "Pareto-optimal (best S+L)"
    elif args.metric == "selADE_mean":
        best = min(results, key=lambda r: r["selADE_mean"])
        best_label = "Best selADE"
    else:
        best = min(results, key=lambda r: r[args.metric])
        best_label = f"Best {args.metric}"

    # Also find default weights result
    default_result = None
    for r in results:
        w = r["weights"]
        if (
            w["safety"] == 1000
            and w["legal"] == 100
            and w["road"] == 10
            and w["comfort"] == 1
        ):
            default_result = r
            break

    # Print summary
    print("\n" + "=" * 70)
    print("GRID SEARCH RESULTS")
    print("=" * 70)
    print(f"\n{best_label}:")
    print(
        f"  Weights: Safety={best['weights']['safety']}, Legal={best['weights']['legal']}, "
        f"Road={best['weights']['road']}, Comfort={best['weights']['comfort']}"
    )
    print(
        f"  S+L: {best['SL_Viol_pct']:.1f}%  Total: {best['Total_Viol_pct']:.1f}%  "
        f"selADE: {best['selADE_mean']:.3f}m"
    )

    if default_result:
        print(f"\nDefault weights [1000, 100, 10, 1]:")
        print(
            f"  S+L: {default_result['SL_Viol_pct']:.1f}%  "
            f"Total: {default_result['Total_Viol_pct']:.1f}%  "
            f"selADE: {default_result['selADE_mean']:.3f}m"
        )

    print(f"\nLexicographic (RECTOR):")
    print(
        f"  S+L: {lex_result['SL_Viol_pct']:.1f}%  "
        f"Total: {lex_result['Total_Viol_pct']:.1f}%  "
        f"selADE: {lex_result['selADE_mean']:.3f}m"
    )

    improvement = (
        default_result["SL_Viol_pct"] - best["SL_Viol_pct"] if default_result else 0
    )
    print(f"\nBest vs Default improvement: {improvement:+.1f} pp S+L")
    print("=" * 70)

    # Save
    output_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "sample_count": sample_count,
            "grid_size": len(weight_grid),
            "optimize_metric": args.metric,
            "cache_time_s": cache_time,
            "sweep_time_s": sweep_time,
            "include_misspecification": args.include_misspecification,
            "safety_weights_used": safety_weights,
        },
        "best_config": best,
        "default_config": default_result,
        "lexicographic_baseline": lex_result,
        "all_results": results,
    }
    if args.metric == "pareto":
        output_data["pareto_front"] = pareto_front

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
