#!/usr/bin/env python3
"""
Mine WOMD validation scenarios for natural lex-WS selection divergence.

For each scenario, checks whether ANY of 125 weight-sum configurations
would select a different candidate than lexicographic ordering. If so,
characterizes the trade-off (does WS sacrifice Safety for Comfort?).

This is a data analysis on cached model outputs — no GPU required.

Usage:
    python evaluation/mine_natural_divergence.py
"""

import os
import sys
import json
import glob
import time
import argparse
from pathlib import Path

import numpy as np
import torch

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, "/workspace/data/WOMD")
sys.path.insert(0, "/workspace/data")
sys.path.insert(0, "/workspace/models/RECTOR/scripts")

from torch.utils.data import DataLoader
from training.train_rector import WaymoDataset, collate_fn, TRAJECTORY_SCALE
from models.rule_aware_generator import RuleAwareGenerator
from evaluation.evaluate_canonical import (
    evaluate_proxy_violations,
    compute_tier_scores,
    PROXY_RULE_MASK,
    select_by_lexicographic,
    select_by_weighted_sum,
    select_by_confidence,
)

import itertools


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
        default="/workspace/output/evaluation/natural_divergence_analysis.json",
    )
    p.add_argument(
        "--max_scenarios", type=int, default=None, help="Max scenarios (None = all)"
    )
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epsilon", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("NATURAL LEX-WS DIVERGENCE MINING")
    print("=" * 70)

    # Weight grid (same as weight_grid_search.py)
    safety_weights = [100, 500, 1000, 2000, 5000]
    legal_weights = [10, 50, 100, 200, 500]
    road_weights = [1, 5, 10, 20, 50]
    comfort_weight = 1.0
    weight_grid = list(itertools.product(safety_weights, legal_weights, road_weights))
    print(f"Weight grid: {len(weight_grid)} configurations")

    # Load model
    print("\n[1/3] Loading model...")
    model = RuleAwareGenerator(
        embed_dim=256,
        decoder_hidden_dim=256,
        decoder_num_layers=4,
        latent_dim=64,
        num_modes=6,
        dropout=0.1,
        use_m2i_encoder=True,
        m2i_checkpoint="/workspace/models/pretrained/m2i/models/relation_v2v/model.25.bin",
        freeze_m2i=True,
        trajectory_length=50,
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Load data
    print("\n[2/3] Loading data...")
    val_files = sorted(glob.glob(os.path.join(args.val_dir, "*")))
    dataset = WaymoDataset(val_files, is_training=False)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    # Track divergences
    n_scenarios = 0
    n_any_divergent = 0  # scenarios where ANY WS config differs from lex
    divergence_details = []  # detailed info for divergent scenarios

    # Per-weight-class tracking
    weight_classes = {
        "flat": [],  # w_S < 10 * w_C
        "moderate": [],  # 10*w_C <= w_S < 100*w_L
        "lex_like": [],  # w_S >= 100 * w_L
    }
    for w_s, w_l, w_r in weight_grid:
        if w_s < 10 * comfort_weight:
            weight_classes["flat"].append((w_s, w_l, w_r))
        elif w_s >= 100 * w_l:
            weight_classes["lex_like"].append((w_s, w_l, w_r))
        else:
            weight_classes["moderate"].append((w_s, w_l, w_r))

    per_class_divergent = {cls: 0 for cls in weight_classes}
    per_class_cross_tier = {
        cls: 0 for cls in weight_classes
    }  # Safety↔Comfort trade-offs

    print(f"\n[3/3] Mining divergence...")
    start = time.time()

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if not batch:
                continue
            if args.max_scenarios and n_scenarios >= args.max_scenarios:
                break

            ego_history = batch["ego_history"].to(device)
            agent_states = batch["agent_states"].to(device)
            lane_centers = batch["lane_centers"].to(device)
            traj_gt = batch["traj_gt"].to(device)
            B = ego_history.shape[0]

            outputs = model(
                ego_history=ego_history,
                agent_states=agent_states,
                lane_centers=lane_centers,
                traj_gt=traj_gt,
            )

            trajectories = outputs["trajectories"]
            confidence = outputs["confidence"]
            app_pred = (outputs["applicability_prob"] > 0.5).float()

            proxy_violations = evaluate_proxy_violations(
                model, trajectories, batch, device
            )

            tier_scores = compute_tier_scores(
                proxy_violations, app_pred, PROXY_RULE_MASK
            )  # [B, M, 4]

            # Lex selection
            lex_sel, _ = select_by_lexicographic(
                tier_scores, args.epsilon, confidence=confidence
            )

            # Check each scenario
            for b in range(B):
                if args.max_scenarios and n_scenarios >= args.max_scenarios:
                    break

                lex_idx = lex_sel[b].item()
                lex_tier = tier_scores[b, lex_idx].cpu().numpy()  # [4]
                scenario_divergent = False

                for cls_name, cls_weights in weight_classes.items():
                    for w_s, w_l, w_r in cls_weights:
                        weights = torch.tensor(
                            [w_s, w_l, w_r, comfort_weight],
                            dtype=torch.float32,
                            device=device,
                        )
                        ws_sel = select_by_weighted_sum(
                            tier_scores[b : b + 1], weights
                        )[0].item()

                        if ws_sel != lex_idx:
                            scenario_divergent = True
                            per_class_divergent[cls_name] += 1

                            # Check cross-tier trade-off
                            ws_tier = tier_scores[b, ws_sel].cpu().numpy()
                            # WS chose candidate with higher Safety but lower Comfort
                            safety_worse = ws_tier[0] > lex_tier[0] + 1e-6
                            comfort_better = ws_tier[3] < lex_tier[3] - 1e-6
                            if safety_worse and comfort_better:
                                per_class_cross_tier[cls_name] += 1

                            # Save first few detailed examples
                            if len(divergence_details) < 100:
                                divergence_details.append(
                                    {
                                        "scenario_idx": n_scenarios,
                                        "lex_mode": lex_idx,
                                        "ws_mode": ws_sel,
                                        "weight_class": cls_name,
                                        "weights": [w_s, w_l, w_r, comfort_weight],
                                        "lex_tier_scores": lex_tier.tolist(),
                                        "ws_tier_scores": ws_tier.tolist(),
                                        "safety_worse_for_ws": bool(safety_worse),
                                        "comfort_better_for_ws": bool(comfort_better),
                                    }
                                )
                            break  # Found divergence for this class, move to next
                    else:
                        continue
                    break  # Found divergence, stop checking classes

                if scenario_divergent:
                    n_any_divergent += 1
                n_scenarios += 1

            if (batch_idx + 1) % 50 == 0:
                elapsed = time.time() - start
                print(
                    f"  {n_scenarios} scenarios, {n_any_divergent} divergent "
                    f"({100*n_any_divergent/max(n_scenarios,1):.1f}%) ({elapsed:.1f}s)"
                )

    elapsed = time.time() - start

    # Report
    print(f"\n{'='*70}")
    print(f"RESULTS ({n_scenarios} scenarios, {elapsed:.1f}s)")
    print(f"{'='*70}")
    print(
        f"\nScenarios with ANY lex-WS divergence: {n_any_divergent} "
        f"({100*n_any_divergent/max(n_scenarios,1):.2f}%)"
    )

    print(f"\nPer weight class:")
    for cls in ["flat", "moderate", "lex_like"]:
        n_weights = len(weight_classes[cls])
        div = per_class_divergent[cls]
        cross = per_class_cross_tier[cls]
        print(
            f"  {cls:12s}: {n_weights:3d} configs, "
            f"{div:6d} divergent scenario-config pairs, "
            f"{cross:6d} with Safety↔Comfort trade-off"
        )

    if n_any_divergent == 0:
        print("\n  FINDING: On the current WOMD evaluation population, NO weight")
        print("  configuration in the 125-config grid produces a different selection")
        print("  than lexicographic ordering. Safety-tier proxy magnitudes dominate")
        print("  mode discrimination across all plausible weight parameterizations.")
    else:
        print(
            f"\n  FINDING: {n_any_divergent} of {n_scenarios} scenarios "
            f"({100*n_any_divergent/n_scenarios:.2f}%) show lex-WS divergence."
        )

    results = {
        "metadata": {
            "n_scenarios": n_scenarios,
            "n_weight_configs": len(weight_grid),
            "elapsed_s": elapsed,
        },
        "n_any_divergent": n_any_divergent,
        "divergence_rate": n_any_divergent / max(n_scenarios, 1),
        "per_weight_class": {
            cls: {
                "n_configs": len(weight_classes[cls]),
                "n_divergent_pairs": per_class_divergent[cls],
                "n_cross_tier_tradeoffs": per_class_cross_tier[cls],
            }
            for cls in weight_classes
        },
        "examples": divergence_details[:20],
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
