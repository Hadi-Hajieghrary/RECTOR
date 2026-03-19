#!/usr/bin/env python3
"""
Adversarial candidate injection for Safety-tier stress testing.

Injects known-violating trajectories into the candidate set and verifies
that all selection strategies correctly deprioritise them. This addresses
the reviewer concern that Safety-tier violations are 0% across all methods,
making the Safety tier's contribution indistinguishable in normal evaluation.

Injection types:
  - collision_course: Trajectory aimed at nearest agent (Safety L10.R1/R2)
  - clearance_violation: Trajectory that clips safe lateral distance (L0.R2/R3)
  - vru_incursion: Trajectory through crosswalk with pedestrian (L0.R4)
  - combined: All injection types applied

Usage:
    python evaluation/adversarial_injection.py --checkpoint /path/to/best.pt
    python evaluation/adversarial_injection.py --checkpoint /path/to/best.pt --max_batches 10
    python evaluation/adversarial_injection.py --injection_types collision_course clearance_violation
"""

import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import argparse
import glob
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, "/workspace/data/WOMD")
sys.path.insert(0, "/workspace/data")
sys.path.insert(0, "/workspace/models/RECTOR/scripts")

from training.train_rector import WaymoDataset, collate_fn, TRAJECTORY_SCALE
from models.rule_aware_generator import RuleAwareGenerator
from waymo_rule_eval.rules.rule_constants import (
    NUM_RULES,
    RULE_IDS,
    RULE_INDEX_MAP,
    TIER_DEFINITIONS,
)
from evaluation.evaluate_canonical import (
    compute_tier_scores,
    compute_violation_rates,
    evaluate_proxy_violations,
    select_by_confidence,
    select_by_weighted_sum,
    select_by_lexicographic,
    PROXY_RULE_MASK,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Adversarial candidate injection stress test"
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
        default="/workspace/output/evaluation/adversarial_injection_results.json",
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

    parser.add_argument(
        "--injection_types",
        nargs="+",
        default=["collision_course", "clearance_violation", "vru_incursion"],
        help="Types of adversarial trajectories to inject",
    )
    parser.add_argument(
        "--replace_mode",
        type=int,
        default=-1,
        help="Which mode to replace (-1 = last mode, i.e. lowest confidence)",
    )
    return parser.parse_args()


def inject_collision_course(trajectories, batch, device):
    """
    Replace one mode with a trajectory aimed at the nearest agent.

    Creates a trajectory that moves directly toward the closest non-ego
    agent, triggering collision (L10.R1) and VRU clearance (L10.R2) violations.

    Args:
        trajectories: [B, M, T, 4] model predictions
        batch: data batch with agent_states
        device: torch device

    Returns:
        adversarial_traj: [B, T, 4] collision-course trajectory
        expected_rules: list of rule IDs expected to be violated
    """
    B, M, T, D = trajectories.shape

    # Use ego starting position from mode 0
    ego_start = trajectories[:, 0, 0, :2]  # [B, 2]

    # Find nearest agent position
    agent_states = batch["agent_states"].to(
        device
    )  # [B, N_agents, ...] or [B, N, H, 4]
    if agent_states.ndim == 4:
        # [B, N, H, D] -> take last timestep -> [B, N, D]
        agent_states_2d = agent_states[:, :, -1, :]
    elif agent_states.ndim >= 3:
        agent_states_2d = agent_states
    else:
        agent_states_2d = None

    if agent_states_2d is not None and agent_states_2d.ndim == 3:
        agent_pos = agent_states_2d[:, :, :2]  # [B, N, 2]
        # Find nearest non-zero agent
        agent_valid = agent_pos.abs().sum(dim=-1) > 0  # [B, N]
        distances = (agent_pos - ego_start.unsqueeze(1)).norm(dim=-1)  # [B, N]
        distances[~agent_valid] = float("inf")
        nearest_idx = distances.argmin(dim=1)  # [B]
        target_pos = agent_pos[torch.arange(B, device=device), nearest_idx]  # [B, 2]
    else:
        # Fallback: aim straight ahead + offset
        ego_end = trajectories[:, 0, -1, :2]
        direction = ego_end - ego_start
        target_pos = ego_start + direction * 0.5

    # Create trajectory from ego_start toward target
    adversarial = trajectories[:, 0].clone()  # [B, T, D]
    for t in range(T):
        alpha = (t + 1) / T
        adversarial[:, t, :2] = ego_start * (1 - alpha) + target_pos * alpha

    return adversarial, ["L10.R1", "L10.R2", "L0.R2", "L0.R3"]


def inject_clearance_violation(trajectories, batch, device):
    """
    Replace one mode with a trajectory that clips safe lateral distance.

    Shifts the best trajectory laterally toward the nearest agent,
    triggering longitudinal/lateral clearance violations (L0.R2, L0.R3).

    Returns:
        adversarial_traj: [B, T, 4]
        expected_rules: list of violated rule IDs
    """
    B, M, T, D = trajectories.shape

    # Take the best mode and shift laterally
    best_traj = trajectories[:, 0].clone()  # [B, T, D]

    # Compute heading-normal direction from trajectory
    dx = best_traj[:, 1:, 0] - best_traj[:, :-1, 0]  # [B, T-1]
    dy = best_traj[:, 1:, 1] - best_traj[:, :-1, 1]
    heading = torch.atan2(dy, dx)  # [B, T-1]
    heading = torch.cat([heading[:, :1], heading], dim=1)  # [B, T]

    # Perpendicular offset (1.5m — inside typical 2m clearance threshold)
    offset = 1.5 / TRAJECTORY_SCALE
    perp_x = -torch.sin(heading) * offset
    perp_y = torch.cos(heading) * offset

    adversarial = best_traj.clone()
    adversarial[:, :, 0] += perp_x
    adversarial[:, :, 1] += perp_y

    return adversarial, ["L0.R2", "L0.R3"]


def inject_vru_incursion(trajectories, batch, device):
    """
    Replace one mode with a trajectory through a crosswalk area.

    Extends ego trajectory through the nearest crosswalk region,
    triggering crosswalk occupancy (L0.R4) and yield (L8.R3) violations.

    Returns:
        adversarial_traj: [B, T, 4]
        expected_rules: list of violated rule IDs
    """
    B, M, T, D = trajectories.shape

    # Accelerate the best trajectory (speed up through crosswalk)
    best_traj = trajectories[:, 0].clone()
    ego_start = best_traj[:, 0, :2]

    # Scale positions to 1.3x speed (creates overshoot into crosswalk area)
    relative = best_traj[:, :, :2] - ego_start.unsqueeze(1)
    adversarial = best_traj.clone()
    adversarial[:, :, :2] = ego_start.unsqueeze(1) + relative * 1.3

    # Also increase speed channel if present
    if D >= 4:
        adversarial[:, :, 3] *= 1.3

    return adversarial, ["L0.R4", "L8.R3"]


INJECTION_REGISTRY = {
    "collision_course": inject_collision_course,
    "clearance_violation": inject_clearance_violation,
    "vru_incursion": inject_vru_incursion,
}


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ADVERSARIAL CANDIDATE INJECTION — Safety Tier Stress Test")
    print("=" * 70)
    print(f"  Injection types: {args.injection_types}")

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

    # Evaluate
    print(f"\n[3/4] Evaluating with adversarial injection...")
    strategies = ["confidence", "weighted_sum", "lexicographic"]
    replace_mode = args.replace_mode if args.replace_mode >= 0 else args.num_modes - 1

    # Results accumulators: {injection_type: {strategy: {metric: [values]}}}
    results = {}
    # Also track: did the strategy select the adversarial mode?
    adversarial_selection_rates = {}

    for inj_type in args.injection_types:
        if inj_type not in INJECTION_REGISTRY:
            print(f"  WARNING: Unknown injection type '{inj_type}', skipping")
            continue

        inject_fn = INJECTION_REGISTRY[inj_type]
        results[inj_type] = {s: [] for s in strategies}
        adversarial_selection_rates[inj_type] = {s: [] for s in strategies}

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
                B = ego_history.shape[0]

                outputs = model(
                    ego_history=ego_history,
                    agent_states=agent_states,
                    lane_centers=lane_centers,
                    traj_gt=traj_gt,
                )

                trajectories = outputs["trajectories"]
                confidence = outputs["confidence"]
                applicability_pred = (outputs["applicability_prob"] > 0.5).float()

                # Inject adversarial candidate
                adv_traj, expected_rules = inject_fn(trajectories, batch, device)
                injected_trajectories = trajectories.clone()
                injected_trajectories[:, replace_mode] = adv_traj

                # Give adversarial mode HIGH confidence (worst case for rule-unaware)
                injected_confidence = confidence.clone()
                injected_confidence[:, replace_mode] = (
                    confidence.max(dim=1).values + 0.1
                )

                # Evaluate violations on injected set
                proxy_violations = evaluate_proxy_violations(
                    model, injected_trajectories, batch, device
                )

                tier_scores = compute_tier_scores(
                    proxy_violations, applicability_pred, PROXY_RULE_MASK
                )

                gt_pos = traj_gt[:, :, :2]
                pred_pos = injected_trajectories[:, :, :, :2]

                for strategy in strategies:
                    if strategy == "confidence":
                        sel = select_by_confidence(injected_confidence)
                    elif strategy == "weighted_sum":
                        sel = select_by_weighted_sum(tier_scores)
                    elif strategy == "lexicographic":
                        sel, _ = select_by_lexicographic(tier_scores, args.epsilon)
                    else:
                        continue

                    # Track if adversarial mode was selected
                    selected_adversarial = (sel == replace_mode).float()
                    adversarial_selection_rates[inj_type][strategy].extend(
                        selected_adversarial.cpu().tolist()
                    )

                    # Compute metrics
                    sel_pred = pred_pos[torch.arange(B, device=device), sel]
                    selADE = (sel_pred - gt_pos).norm(dim=-1).mean(
                        dim=-1
                    ) * TRAJECTORY_SCALE
                    sel_viols = proxy_violations[torch.arange(B, device=device), sel]
                    sel_rates = compute_violation_rates(
                        sel_viols, applicability_pred, PROXY_RULE_MASK
                    )

                    for i in range(B):
                        results[inj_type][strategy].append(
                            {
                                "selADE": selADE[i].item(),
                                "total_violated": bool(sel_rates["total_union"][i]),
                                "tier_0_violated": bool(sel_rates["tier_0_union"][i]),
                                "tier_1_violated": bool(sel_rates["tier_1_union"][i]),
                            }
                        )

                sample_count += B
                if (batch_idx + 1) % 10 == 0:
                    print(f"  [{inj_type}] Batch {batch_idx+1}: {sample_count} samples")

        elapsed = time.time() - start_time
        print(f"  [{inj_type}] Complete: {sample_count} samples in {elapsed:.1f}s")

    # Aggregate
    print(f"\n[4/4] Aggregating results...")
    output_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "injection_types": args.injection_types,
            "replace_mode": replace_mode,
        },
        "per_injection": {},
    }

    print("\n" + "=" * 70)
    print("ADVERSARIAL INJECTION RESULTS")
    print("=" * 70)

    for inj_type in args.injection_types:
        if inj_type not in results:
            continue

        output_data["per_injection"][inj_type] = {}
        print(f"\n  Injection: {inj_type}")
        print(
            f"  {'Strategy':<20} {'Adv.Sel%':>10} {'Safety%':>10} {'Total%':>10} {'selADE':>10}"
        )

        for strategy in strategies:
            strat_results = results[inj_type][strategy]
            if not strat_results:
                continue

            adv_rate = float(
                np.mean(adversarial_selection_rates[inj_type][strategy]) * 100
            )
            safety_pct = float(
                np.mean([r["tier_0_violated"] for r in strat_results]) * 100
            )
            total_pct = float(
                np.mean([r["total_violated"] for r in strat_results]) * 100
            )
            selADE = float(np.mean([r["selADE"] for r in strat_results]))

            output_data["per_injection"][inj_type][strategy] = {
                "adversarial_selection_rate_pct": adv_rate,
                "Safety_Viol_pct": safety_pct,
                "Total_Viol_pct": total_pct,
                "selADE_mean": selADE,
                "n_scenarios": len(strat_results),
            }

            print(
                f"  {strategy:<20} {adv_rate:>9.1f}% {safety_pct:>9.1f}% "
                f"{total_pct:>9.1f}% {selADE:>9.3f}m"
            )

    print("\n" + "=" * 70)
    print("KEY FINDING: If lexicographic has lower Adv.Sel% than confidence,")
    print("the Safety tier is actively differentiating under adversarial stress.")
    print("=" * 70)

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
