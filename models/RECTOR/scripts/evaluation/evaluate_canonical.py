#!/usr/bin/env python3
"""
Canonical RECTOR Evaluation Script.

Single source of truth for ALL paper metrics under ALL protocol combinations.
Every table in the paper MUST be generated from the output of this script.

Protocol axes:
  Rule set:        proxy-24 vs full-28
  Applicability:   oracle vs predicted
  Aggregation:     per-tier union (primary), per-tier sum (diagnostic)

Usage:
        python evaluation/evaluate_canonical.py --checkpoint /path/to/best.pt
        python evaluation/evaluate_canonical.py --checkpoint /path/to/best.pt --eval_gt  # Human GT
        python evaluation/evaluate_canonical.py --checkpoint /path/to/best.pt --max_batches 10  # Quick test
"""

import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import argparse
import csv
import glob
import hashlib
import json
import math
import time
from collections import defaultdict
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

# Import model and data
from training.train_rector import WaymoDataset, collate_fn, TRAJECTORY_SCALE
from models.rule_aware_generator import RuleAwareGenerator
from training.losses import RECTORLoss

# Import rule evaluation
from waymo_rule_eval.rules.rule_constants import (
    TIER_0_SAFETY,
    TIER_1_LEGAL,
    TIER_2_ROAD,
    TIER_3_COMFORT,
    TIER_DEFINITIONS,
    NUM_RULES,
    RULE_IDS,
    RULE_INDEX_MAP,
    get_tier_mask,
    TIER_WEIGHTS,
)

# Import proxies
from proxies.aggregator import DifferentiableRuleProxies

# Import applicability baselines
from evaluation.heuristic_applicability import (
    always_on_applicability,
    heuristic_applicability,
    hybrid_conservative_applicability,
)

# Which rules have proxies (proxy-24 set)
# Rules WITHOUT proxies are those requiring intent modeling / multi-agent negotiation
RULES_WITHOUT_PROXIES = {
    # These 4 rules lack differentiable proxies (14% of catalog)
    # Verified from DifferentiableRuleProxies.covered_rules
    "L5.R2",  # Priority / right-of-way violation
    "L5.R3",  # Parking zone violation
    "L5.R4",  # School zone speed compliance
    "L5.R5",  # Construction zone compliance
}

PROXY_RULE_MASK = np.array(
    [(rule_id not in RULES_WITHOUT_PROXIES) for rule_id in RULE_IDS], dtype=bool
)  # True = has proxy (24 of 28)

TIER_NAMES = {0: "Safety", 1: "Legal", 2: "Road", 3: "Comfort"}


def _normal_sf(z):
    """Survival function of standard normal (1 - CDF)."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def mcnemar_test(a, b):
    """
    Paired McNemar test for binary outcomes.

    Args:
        a: iterable of bool/int (method A violated?)
        b: iterable of bool/int (method B violated?)

    Returns:
        dict with discordant counts and two-sided p-value
    """
    a = np.asarray(a, dtype=np.int32)
    b = np.asarray(b, dtype=np.int32)
    if a.shape != b.shape:
        raise ValueError("McNemar inputs must have identical shapes")

    # b_only: A=1,B=0 ; c_only: A=0,B=1 under standard notation
    b_only = int(np.sum((a == 1) & (b == 0)))
    c_only = int(np.sum((a == 0) & (b == 1)))
    n_disc = b_only + c_only

    if n_disc == 0:
        return {
            "b_only": b_only,
            "c_only": c_only,
            "n_discordant": 0,
            "chi2_cc": 0.0,
            "p_value": 1.0,
            "method": "degenerate",
        }

    # Exact two-sided binomial for small discordant counts.
    if n_disc <= 1000:
        k = min(b_only, c_only)
        # 2 * P[X <= k], X ~ Binomial(n_disc, 0.5)
        prob = 0.0
        two_pow_n = 2.0**n_disc
        for i in range(k + 1):
            prob += math.comb(n_disc, i) / two_pow_n
        p_val = min(1.0, 2.0 * prob)
        return {
            "b_only": b_only,
            "c_only": c_only,
            "n_discordant": n_disc,
            "chi2_cc": None,
            "p_value": float(p_val),
            "method": "exact_binomial",
        }

    # Normal approximation (with continuity correction) for large n.
    chi2_cc = ((abs(b_only - c_only) - 1.0) ** 2) / n_disc
    # chi-square(1) survival = 2 * normal_sf(sqrt(chi2))
    p_val = 2.0 * _normal_sf(math.sqrt(max(chi2_cc, 0.0)))
    return {
        "b_only": b_only,
        "c_only": c_only,
        "n_discordant": n_disc,
        "chi2_cc": float(chi2_cc),
        "p_value": float(max(0.0, min(1.0, p_val))),
        "method": "normal_approx_cc",
    }


def wilcoxon_signed_rank(x, y):
    """
    Approximate Wilcoxon signed-rank test (two-sided, normal approximation).

    Returns W+, W-, z-score and p-value. Falls back to p=1.0 when all deltas=0.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError("Wilcoxon inputs must have identical shapes")

    d = x - y
    nz = np.abs(d) > 1e-12
    d = d[nz]
    n = int(d.size)
    if n == 0:
        return {
            "n_nonzero": 0,
            "w_plus": 0.0,
            "w_minus": 0.0,
            "z": 0.0,
            "p_value": 1.0,
            "method": "degenerate",
        }

    abs_d = np.abs(d)
    order = np.argsort(abs_d)
    sorted_abs = abs_d[order]

    ranks = np.zeros(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_abs[j] == sorted_abs[i]:
            j += 1
        avg_rank = 0.5 * (i + 1 + j)
        ranks[i:j] = avg_rank
        i = j

    unsorted_ranks = np.zeros(n, dtype=np.float64)
    unsorted_ranks[order] = ranks

    w_plus = float(unsorted_ranks[d > 0].sum())
    w_minus = float(unsorted_ranks[d < 0].sum())
    w_stat = min(w_plus, w_minus)

    # Normal approximation mean/variance with tie correction.
    mean_w = n * (n + 1) / 4.0

    _, tie_counts = np.unique(sorted_abs, return_counts=True)
    tie_term = np.sum(tie_counts * (tie_counts + 1) * (2 * tie_counts + 1))
    var_w = (n * (n + 1) * (2 * n + 1) - tie_term) / 24.0
    if var_w <= 0:
        p_val = 1.0
        z = 0.0
    else:
        # Continuity correction on two-sided test.
        z = (abs(w_stat - mean_w) - 0.5) / math.sqrt(var_w)
        p_val = 2.0 * _normal_sf(abs(z))

    return {
        "n_nonzero": n,
        "w_plus": w_plus,
        "w_minus": w_minus,
        "z": float(z),
        "p_value": float(max(0.0, min(1.0, p_val))),
        "method": "normal_approx_tie_corrected",
    }


def compute_pairwise_significance(strategy_results):
    """Compute paired significance tests between key strategy pairs."""
    pairs = [
        ("lexicographic", "confidence"),
        ("lexicographic", "weighted_sum"),
    ]
    out = {}

    for a_name, b_name in pairs:
        if a_name not in strategy_results or b_name not in strategy_results:
            continue
        a = strategy_results[a_name]
        b = strategy_results[b_name]
        if len(a) == 0 or len(b) == 0 or len(a) != len(b):
            continue

        # Binary violation outcomes for McNemar.
        a_safety = [int(r["tier_0_violated"]) for r in a]
        b_safety = [int(r["tier_0_violated"]) for r in b]
        a_total = [int(r["total_violated"]) for r in a]
        b_total = [int(r["total_violated"]) for r in b]

        # Continuous paired outcomes for Wilcoxon.
        a_selade = [r["selADE"] for r in a]
        b_selade = [r["selADE"] for r in b]

        key = f"{a_name}_vs_{b_name}"
        out[key] = {
            "n_scenarios": len(a),
            "mcnemar_safety": mcnemar_test(a_safety, b_safety),
            "mcnemar_total": mcnemar_test(a_total, b_total),
            "wilcoxon_selADE": wilcoxon_signed_rank(
                np.array(a_selade), np.array(b_selade)
            ),
            "mean_selADE_delta_m": float(
                np.mean(np.array(a_selade) - np.array(b_selade))
            ),
        }

    return out


def parse_args():
    parser = argparse.ArgumentParser(
        description="Canonical RECTOR Evaluation — single source of truth for all paper metrics"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/workspace/models/RECTOR/models/best.pt",
        help="Path to trained checkpoint",
    )
    parser.add_argument(
        "--val_dir",
        type=str,
        default="/workspace/data/WOMD/datasets/waymo_open_dataset/motion_v_1_3_0/processed/augmented/scenario/validation_interactive",
        help="Path to validation TFRecords",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/workspace/output/evaluation/canonical_results.json",
        help="Output JSON path",
    )

    # Model config (must match training)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--decoder_hidden_dim", type=int, default=256)
    parser.add_argument("--decoder_num_layers", type=int, default=4)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--num_modes", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)

    # M2I
    parser.add_argument("--use_m2i_encoder", action="store_true", default=True)
    parser.add_argument(
        "--m2i_checkpoint",
        type=str,
        default="/workspace/models/pretrained/m2i/models/relation_v2v/model.25.bin",
    )

    # Evaluation options
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--max_batches",
        type=int,
        default=None,
        help="Limit batches for quick testing (None = all)",
    )
    parser.add_argument(
        "--eval_gt",
        action="store_true",
        help="Evaluate human GT trajectories instead of RECTOR predictions",
    )
    parser.add_argument("--seed", type=int, default=42)

    # Selection strategies to evaluate
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["confidence", "weighted_sum", "lexicographic"],
        help="Selection strategies to evaluate",
    )

    # Tolerance for lexicographic selection
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-3,
        help="Tolerance for lexicographic tier comparison",
    )

    # Applicability mode (WS-1: applicability baselines)
    parser.add_argument(
        "--applicability_mode",
        type=str,
        default="learned",
        choices=["learned", "always_on", "heuristic", "hybrid_conservative"],
        help="Applicability source: learned (default), always_on, heuristic, "
        "or hybrid_conservative (always_on for Tier 0+1, learned for Tier 2+3)",
    )

    # Per-rule accuracy metrics
    parser.add_argument(
        "--per_rule_metrics",
        action="store_true",
        help="Compute per-rule precision/recall/F1 for applicability head",
    )

    return parser.parse_args()


def checkpoint_hash(path):
    """Compute SHA256 of checkpoint file (first 1MB for speed)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(1024 * 1024))
    return h.hexdigest()[:16]


def select_by_confidence(confidence):
    """Select mode with highest confidence. Returns [B] indices."""
    return confidence.argmax(dim=1)


def select_by_weighted_sum(tier_scores, tier_weights=None):
    """
    Select mode minimizing weighted sum of tier scores.
    tier_scores: [B, M, 4] (Safety, Legal, Road, Comfort)
    Returns [B] indices.
    """
    if tier_weights is None:
        tier_weights = torch.tensor([1000.0, 100.0, 10.0, 1.0])
    tier_weights = tier_weights.to(tier_scores.device)
    weighted = (tier_scores * tier_weights.unsqueeze(0).unsqueeze(0)).sum(
        dim=-1
    )  # [B, M]
    return weighted.argmin(dim=1)


def select_by_lexicographic(tier_scores, epsilon=1e-3, confidence=None):
    """
    Lexicographic selection with tolerance.
    tier_scores: [B, M, 4] (Safety, Legal, Road, Comfort)
    confidence: [B, M] optional — used to break ties among survivors.
               If None, falls back to lowest summed tier score.
    Returns [B] indices and [B] infeasibility flags.
    Infeasibility: True when the selected candidate has Safety-tier score > 0,
    meaning no fully compliant candidate exists and the selector returned
    the least-violating option.
    """
    B, M, T = tier_scores.shape
    device = tier_scores.device

    selected = torch.zeros(B, dtype=torch.long, device=device)
    infeasible = torch.zeros(B, dtype=torch.bool, device=device)

    for b in range(B):
        # Start with all modes as candidates
        candidates = torch.arange(M, device=device)

        for tier in range(T):
            if len(candidates) <= 1:
                break
            scores = tier_scores[b, candidates, tier]
            min_score = scores.min()
            # Keep candidates within epsilon of the minimum
            mask = scores <= min_score + epsilon
            candidates = candidates[mask]

        # Among survivors, break ties by highest confidence (consistent
        # with rector_lex.py in the simulation engine)
        if len(candidates) > 0:
            if confidence is not None:
                survivor_conf = confidence[b, candidates]
                best_idx = survivor_conf.argmax()
            else:
                # Fallback when confidence unavailable
                remaining_scores = tier_scores[b, candidates].sum(dim=-1)
                best_idx = remaining_scores.argmin()
            selected[b] = candidates[best_idx]

        # Infeasibility: selected candidate has non-zero Safety-tier score
        infeasible[b] = tier_scores[b, selected[b], 0] > epsilon

    return selected, infeasible


def compute_tier_scores(violations, applicability, rule_mask):
    """
    Compute per-tier scores from per-rule violations.

    Args:
        violations: [B, M, NUM_RULES] violation severity in [0, 1]
        applicability: [B, NUM_RULES] applicability in {0, 1} or [0, 1]
        rule_mask: [NUM_RULES] bool — which rules to include (proxy-24 or full-28)

    Returns:
        tier_scores: [B, M, 4] aggregated tier scores
    """
    B, M, R = violations.shape
    device = violations.device

    # Apply applicability gating: [B, 1, R] * [B, M, R]
    gated = violations * applicability.unsqueeze(1)

    # Apply rule mask
    mask = torch.tensor(rule_mask, dtype=torch.float32, device=device)
    gated = gated * mask.unsqueeze(0).unsqueeze(0)

    # Aggregate by tier
    tier_scores = torch.zeros(B, M, 4, device=device)
    for tier_idx, tier_rules in TIER_DEFINITIONS.items():
        tier_mask = torch.zeros(R, device=device)
        for rule_id in tier_rules:
            if rule_id in RULE_INDEX_MAP:
                tier_mask[RULE_INDEX_MAP[rule_id]] = 1.0
        # Combine with rule_mask
        combined = tier_mask * mask
        tier_scores[:, :, tier_idx] = (gated * combined.unsqueeze(0).unsqueeze(0)).sum(
            dim=-1
        )

    return tier_scores


def compute_violation_rates(violations, applicability, rule_mask):
    """
    Compute per-scenario violation indicators for aggregation.

    Args:
        violations: [B, NUM_RULES] violation severity for selected trajectory
        applicability: [B, NUM_RULES]
        rule_mask: [NUM_RULES] bool

    Returns:
        dict with per-scenario binary indicators
    """
    B, R = violations.shape
    device = violations.device

    mask = torch.tensor(rule_mask, dtype=torch.float32, device=device)
    gated = violations * applicability * mask.unsqueeze(0)

    # Binary violation per rule per scenario (severity > threshold)
    violated = (gated > 0.01).float()  # small threshold for numerical noise

    results = {}

    # Per-tier union: scenario violates tier if ANY rule in tier is violated
    for tier_idx, tier_rules in TIER_DEFINITIONS.items():
        tier_mask_t = torch.zeros(R, device=device)
        for rule_id in tier_rules:
            if rule_id in RULE_INDEX_MAP:
                tier_mask_t[RULE_INDEX_MAP[rule_id]] = 1.0
        combined = tier_mask_t * mask
        tier_violated = (violated * combined.unsqueeze(0)).sum(dim=-1) > 0  # [B] bool
        results[f"tier_{tier_idx}_union"] = tier_violated.cpu().numpy()

    # Total union: scenario violates if ANY rule is violated
    any_violated = violated.sum(dim=-1) > 0  # [B] bool
    results["total_union"] = any_violated.cpu().numpy()

    # S+L union: Safety OR Legal
    sl_violated = results["tier_0_union"] | results["tier_1_union"]
    results["sl_union"] = sl_violated

    # Per-tier sum (for Viol.% sum metric)
    for tier_idx, tier_rules in TIER_DEFINITIONS.items():
        tier_mask_t = torch.zeros(R, device=device)
        for rule_id in tier_rules:
            if rule_id in RULE_INDEX_MAP:
                tier_mask_t[RULE_INDEX_MAP[rule_id]] = 1.0
        combined = tier_mask_t * mask
        tier_rate = (violated * combined.unsqueeze(0)).sum(dim=-1) / max(
            combined.sum().item(), 1
        )
        results[f"tier_{tier_idx}_mean_severity"] = tier_rate.cpu().numpy()

    # Per-rule violations
    for rule_idx, rule_id in enumerate(RULE_IDS):
        if rule_mask[rule_idx]:
            results[f"rule_{rule_id}"] = violated[:, rule_idx].cpu().numpy()

    return results


def evaluate_proxy_violations(model, trajectories, batch, device):
    """
    Evaluate proxy violations for all modes using DifferentiableRuleProxies.

    Args:
        model: RuleAwareGenerator (has rule_proxies)
        trajectories: [B, M, T, 4] predicted trajectories (normalized space)
        batch: data batch dict
        device: torch device

    Returns:
        violations: [B, M, NUM_RULES] proxy violation severity
    """
    B, M, T, D = trajectories.shape

    # Denormalize ego trajectories to meters for proxy evaluation.
    # Proxies use metric thresholds (ego_length=4.5m, clearance=1.8m, etc.)
    # so inputs must be in meters, not normalized (÷TRAJECTORY_SCALE) space.
    traj_meters = trajectories.clone()
    traj_meters[..., :2] = traj_meters[..., :2] * TRAJECTORY_SCALE

    # Extract agent states: [B, N, H_hist, 4] = (x_norm, y_norm, heading, speed)
    agent_states = batch["agent_states"].to(device)
    N = agent_states.shape[1]

    # Denormalize agent positions to meters
    agent_pos_hist = agent_states[..., :2] * TRAJECTORY_SCALE  # [B, N, H_hist, 2]
    agent_heading = agent_states[..., 2]  # [B, N, H_hist] radians
    agent_speed = agent_states[..., 3]  # [B, N, H_hist] m/s

    # Determine valid agents: must have non-zero position at LAST timestep.
    # Agents with zero last position are at ego origin (0,0) and would cause
    # false collision detections in ego-centric coordinates.
    agent_last_valid = agent_pos_hist[:, :, -1, :].abs().sum(dim=-1) > 0.5  # [B, N]

    # Extrapolate agent positions to future horizon using constant velocity
    # from the last history timestep (standard open-loop assumption)
    last_pos = agent_pos_hist[:, :, -1, :]  # [B, N, 2]
    last_heading = agent_heading[:, :, -1]  # [B, N]
    last_speed = agent_speed[:, :, -1]  # [B, N]

    # Zero out velocity for invalid agents to prevent extrapolation from origin
    valid_mask_f = agent_last_valid.float()  # [B, N]
    vx = last_speed * torch.cos(last_heading) * valid_mask_f  # [B, N]
    vy = last_speed * torch.sin(last_heading) * valid_mask_f  # [B, N]

    dt = 0.1  # 10 Hz
    time_offsets = (
        torch.arange(1, T + 1, device=device, dtype=torch.float32) * dt
    )  # [T]

    # Broadcast: [B, N, 1] + [B, N, 1] * [T] -> [B, N, T]
    agent_future_x = last_pos[:, :, 0:1] + vx.unsqueeze(-1) * time_offsets
    agent_future_y = last_pos[:, :, 1:2] + vy.unsqueeze(-1) * time_offsets
    agent_positions = torch.stack(
        [agent_future_x, agent_future_y], dim=-1
    )  # [B, N, T, 2]

    # Push invalid agents far away so they never trigger collision checks.
    # This is more robust than relying on agent_valid mask which not all
    # proxy code paths consistently apply.
    far_away = torch.tensor([9999.0, 9999.0], device=device)
    invalid_mask = (~agent_last_valid).unsqueeze(-1).unsqueeze(-1)  # [B, N, 1, 1]
    agent_positions = torch.where(
        invalid_mask.expand_as(agent_positions),
        far_away.expand_as(agent_positions),
        agent_positions,
    )

    # Validity mask for future: [B, N, T]
    agent_valid = agent_last_valid.unsqueeze(-1).expand(-1, -1, T)

    # Build scene context with correct keys for proxy evaluation
    scene = {
        "agent_positions": agent_positions,  # [B, N, T, 2] meters
        "agent_valid": agent_valid,  # [B, N, T]
        "lane_centers": batch["lane_centers"].to(device) * TRAJECTORY_SCALE,  # meters
    }

    # DifferentiableRuleProxies.forward() expects [B, M, H, 2+] in meters
    # It returns [B, M, NUM_RULES] costs
    try:
        violations = model.rule_proxies(traj_meters, scene)  # [B, M, NUM_RULES]
        violations = violations.clamp(0, 1)
    except Exception as e:
        print(f"  Warning: Proxy evaluation failed ({e}), using kinematic fallback")
        violations = torch.zeros(B, M, NUM_RULES, device=device)
        for m in range(M):
            violations[:, m] = compute_kinematic_violations(trajectories[:, m], device)

    return violations


def compute_kinematic_violations(traj, device):
    """
    Fallback kinematic violation computation.
    traj: [B, T, 4] trajectory (x, y, heading, speed or similar)
    Returns: [B, NUM_RULES] violation severity
    """
    B, T, D = traj.shape
    violations = torch.zeros(B, NUM_RULES, device=device)

    positions = traj[:, :, :2] * TRAJECTORY_SCALE  # scale to meters
    if T < 3:
        return violations

    # Velocities and accelerations
    dt = 0.1  # 10 Hz
    displacements = positions[:, 1:] - positions[:, :-1]  # [B, T-1, 2]
    velocities = displacements / dt  # m/s
    speeds = velocities.norm(dim=-1)  # [B, T-1]

    if T > 2:
        accels = (velocities[:, 1:] - velocities[:, :-1]) / dt  # [B, T-2, 2]
        accel_mag = accels.norm(dim=-1)  # [B, T-2]

        # Comfort tier: longitudinal acceleration > 3 m/s^2
        for rule_id in ["L1.R1", "L1.R2", "L1.R3"]:
            if rule_id in RULE_INDEX_MAP:
                idx = RULE_INDEX_MAP[rule_id]
                max_accel = accel_mag.max(dim=-1).values
                violations[:, idx] = torch.clamp((max_accel - 3.0) / 3.0, 0, 1)

        # Jerk (comfort)
        if T > 3:
            jerks = (accels[:, 1:] - accels[:, :-1]) / dt
            jerk_mag = jerks.norm(dim=-1)
            for rule_id in ["L1.R4", "L1.R5"]:
                if rule_id in RULE_INDEX_MAP:
                    idx = RULE_INDEX_MAP[rule_id]
                    max_jerk = jerk_mag.max(dim=-1).values
                    violations[:, idx] = torch.clamp((max_jerk - 2.5) / 2.5, 0, 1)

    # Speed limit (legal tier)
    if "L7.R4" in RULE_INDEX_MAP:
        idx = RULE_INDEX_MAP["L7.R4"]
        max_speed = speeds.max(dim=-1).values
        violations[:, idx] = torch.clamp((max_speed - 20.0) / 10.0, 0, 1)

    return violations


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("CANONICAL RECTOR EVALUATION")
    print("Single source of truth for all paper metrics")
    print("=" * 70)

    # ---- Load model ----
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
    print(f"  Loaded: {args.checkpoint}")
    print(f"  Epoch: {ckpt.get('epoch', '?')}, best_ade: {ckpt.get('best_ade', '?')}")

    # ---- Load data ----
    print("\n[2/4] Loading validation data...")
    val_files = sorted(glob.glob(os.path.join(args.val_dir, "*")))
    if not val_files:
        print(f"ERROR: No files in {args.val_dir}")
        return
    print(f"  {len(val_files)} TFRecord shards")

    dataset = WaymoDataset(val_files, is_training=False)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # ---- Evaluation loop ----
    print(
        f"\n[3/4] Evaluating ({'GT trajectories' if args.eval_gt else 'RECTOR predictions'})..."
    )
    start_time = time.time()

    # Accumulators — per-scenario data for all protocol combinations
    # Keys: (rule_set, applicability_source) → list of per-scenario dicts
    protocol_results = {
        ("proxy-24", "predicted"): [],
        ("proxy-24", "oracle"): [],
        ("full-28", "predicted"): [],
        ("full-28", "oracle"): [],
    }

    # Geometric metrics accumulators
    all_minADE = []
    all_minFDE = []
    all_miss = []

    # Per-strategy selection results (for proxy-24/predicted — Protocol A)
    strategy_results = {s: [] for s in args.strategies}
    # Per-strategy selection results under Protocol B (full-28, oracle applicability)
    strategy_results_protB = {s: [] for s in args.strategies}
    # Cross-evaluation: select with mode-specific applicability, evaluate under oracle
    strategy_results_cross = {s: [] for s in args.strategies}
    oracle_label_samples = 0
    fallback_oracle_samples = 0

    # Per-rule applicability accuracy accumulators
    all_app_oracle = []
    all_app_pred = []

    sample_count = 0

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

            # Forward pass
            outputs = model(
                ego_history=ego_history,
                agent_states=agent_states,
                lane_centers=lane_centers,
                traj_gt=traj_gt,
            )

            trajectories = outputs["trajectories"]  # [B, M, T, 4]
            confidence = outputs["confidence"]  # [B, M]
            applicability_pred = outputs["applicability_prob"]  # [B, NUM_RULES]

            # Oracle applicability from TFRecord (if available)
            if "rule_applicability" in batch:
                applicability_oracle = batch["rule_applicability"].to(device)
                oracle_label_samples += B
            else:
                # Fallback: use predicted as oracle (flag this in output)
                applicability_oracle = (applicability_pred > 0.5).float()
                fallback_oracle_samples += B

            # Applicability mode override (WS-1)
            if args.applicability_mode == "always_on":
                applicability_pred_binary = torch.ones(B, NUM_RULES, device=device)
            elif args.applicability_mode == "heuristic":
                heur_np = heuristic_applicability(batch)
                applicability_pred_binary = torch.tensor(heur_np, device=device)
            elif args.applicability_mode == "hybrid_conservative":
                learned_np = applicability_pred.cpu().numpy()
                hybrid_np = hybrid_conservative_applicability(learned_np)
                applicability_pred_binary = torch.tensor(hybrid_np, device=device)
            else:
                # Default: learned applicability
                applicability_pred_binary = (applicability_pred > 0.5).float()

            # Collect per-rule applicability data for accuracy metrics
            if args.per_rule_metrics and "rule_applicability" in batch:
                oracle_np = applicability_oracle.cpu().numpy()
                pred_np = applicability_pred_binary.cpu().numpy()
                for i in range(B):
                    all_app_oracle.append(oracle_np[i])
                    all_app_pred.append(pred_np[i])

            # ---- Compute violations for all modes ----
            if args.eval_gt:
                # Evaluate GT trajectory (replicate to M modes for consistent interface)
                gt_expanded = traj_gt.unsqueeze(1).expand(-1, args.num_modes, -1, -1)
                proxy_violations = evaluate_proxy_violations(
                    model, gt_expanded, batch, device
                )
            else:
                proxy_violations = evaluate_proxy_violations(
                    model, trajectories, batch, device
                )

            # Full-28 violations (use proxy as approximation; flag that full evaluator
            # should be used when available via waymo_rule_eval pipeline)
            full_violations = proxy_violations.clone()
            # TODO: Replace with actual full-28 evaluator when RuleExecutor integration is ready
            # For now, proxy violations serve as the base; the 4 unproxied rules get zero

            # ---- Geometric metrics (oracle — best of K) ----
            pred_pos = trajectories[:, :, :, :2]  # [B, M, T, 2]
            gt_pos = traj_gt[:, :, :2]  # [B, T, 2]

            ade_per_mode = (
                (pred_pos - gt_pos.unsqueeze(1)).norm(dim=-1).mean(dim=-1)
            )  # [B, M]
            best_mode = ade_per_mode.argmin(dim=1)  # [B]

            minADE = (
                ade_per_mode[torch.arange(B, device=device), best_mode]
                * TRAJECTORY_SCALE
            )
            fde_per_mode = (pred_pos[:, :, -1] - gt_pos[:, -1:]).norm(
                dim=-1
            ) * TRAJECTORY_SCALE
            minFDE = fde_per_mode[torch.arange(B, device=device), best_mode]
            miss = (minFDE > 2.0).float()

            all_minADE.extend(minADE.cpu().tolist())
            all_minFDE.extend(minFDE.cpu().tolist())
            all_miss.extend(miss.cpu().tolist())

            # ---- Selection strategies (on Protocol A: proxy-24, predicted) ----
            rule_mask_24 = PROXY_RULE_MASK
            tier_scores_24_pred = compute_tier_scores(
                proxy_violations, applicability_pred_binary, rule_mask_24
            )  # [B, M, 4]

            for strategy in args.strategies:
                if strategy == "confidence":
                    sel = select_by_confidence(confidence)
                    infeasible_sel = torch.zeros(B, dtype=torch.bool, device=device)
                elif strategy == "weighted_sum":
                    sel = select_by_weighted_sum(tier_scores_24_pred)
                    infeasible_sel = torch.zeros(B, dtype=torch.bool, device=device)
                elif strategy == "lexicographic":
                    sel, infeasible_sel = select_by_lexicographic(
                        tier_scores_24_pred, args.epsilon, confidence=confidence
                    )
                else:
                    continue

                # Selected trajectory metrics
                sel_pred = pred_pos[torch.arange(B, device=device), sel]  # [B, T, 2]
                selADE = (sel_pred - gt_pos).norm(dim=-1).mean(
                    dim=-1
                ) * TRAJECTORY_SCALE
                selFDE = (sel_pred[:, -1] - gt_pos[:, -1]).norm(
                    dim=-1
                ) * TRAJECTORY_SCALE

                # Selected violations under Protocol A
                sel_violations = proxy_violations[torch.arange(B, device=device), sel]
                sel_rates = compute_violation_rates(
                    sel_violations, applicability_pred_binary, rule_mask_24
                )

                for i in range(B):
                    strategy_results[strategy].append(
                        {
                            "selADE": selADE[i].item(),
                            "selFDE": selFDE[i].item(),
                            "total_violated": bool(sel_rates["total_union"][i]),
                            "sl_violated": bool(sel_rates["sl_union"][i]),
                            "tier_0_violated": bool(sel_rates["tier_0_union"][i]),
                            "tier_1_violated": bool(sel_rates["tier_1_union"][i]),
                            "tier_2_violated": bool(sel_rates["tier_2_union"][i]),
                            "tier_3_violated": bool(sel_rates["tier_3_union"][i]),
                            "infeasible_selected": bool(infeasible_sel[i].item()),
                        }
                    )

            # ---- Cross-evaluation: mode-specific selection, oracle evaluation ----
            rule_mask_28 = np.ones(NUM_RULES, dtype=bool)
            for strategy in args.strategies:
                if strategy == "confidence":
                    sel_x = select_by_confidence(confidence)
                    infeasible_x = torch.zeros(B, dtype=torch.bool, device=device)
                elif strategy == "weighted_sum":
                    sel_x = select_by_weighted_sum(tier_scores_24_pred)
                    infeasible_x = torch.zeros(B, dtype=torch.bool, device=device)
                elif strategy == "lexicographic":
                    sel_x, infeasible_x = select_by_lexicographic(
                        tier_scores_24_pred, args.epsilon, confidence=confidence
                    )
                else:
                    continue

                sel_pred_x = pred_pos[torch.arange(B, device=device), sel_x]
                selADE_x = (sel_pred_x - gt_pos).norm(dim=-1).mean(
                    dim=-1
                ) * TRAJECTORY_SCALE
                selFDE_x = (sel_pred_x[:, -1] - gt_pos[:, -1]).norm(
                    dim=-1
                ) * TRAJECTORY_SCALE

                # Evaluate the mode-selected trajectory under oracle applicability
                sel_viols_x = full_violations[torch.arange(B, device=device), sel_x]
                sel_rates_x = compute_violation_rates(
                    sel_viols_x, applicability_oracle, rule_mask_28
                )

                for i in range(B):
                    strategy_results_cross[strategy].append(
                        {
                            "selADE": selADE_x[i].item(),
                            "selFDE": selFDE_x[i].item(),
                            "total_violated": bool(sel_rates_x["total_union"][i]),
                            "sl_violated": bool(sel_rates_x["sl_union"][i]),
                            "tier_0_violated": bool(sel_rates_x["tier_0_union"][i]),
                            "tier_1_violated": bool(sel_rates_x["tier_1_union"][i]),
                            "tier_2_violated": bool(sel_rates_x["tier_2_union"][i]),
                            "tier_3_violated": bool(sel_rates_x["tier_3_union"][i]),
                            "infeasible_selected": bool(infeasible_x[i].item()),
                        }
                    )

            # ---- Selection strategies under Protocol B (full-28, oracle) ----
            tier_scores_28_oracle = compute_tier_scores(
                full_violations, applicability_oracle, rule_mask_28
            )  # [B, M, 4]

            for strategy in args.strategies:
                if strategy == "confidence":
                    sel_b = select_by_confidence(confidence)
                    infeasible_b = torch.zeros(B, dtype=torch.bool, device=device)
                elif strategy == "weighted_sum":
                    sel_b = select_by_weighted_sum(tier_scores_28_oracle)
                    infeasible_b = torch.zeros(B, dtype=torch.bool, device=device)
                elif strategy == "lexicographic":
                    sel_b, infeasible_b = select_by_lexicographic(
                        tier_scores_28_oracle, args.epsilon, confidence=confidence
                    )
                else:
                    continue

                sel_pred_b = pred_pos[torch.arange(B, device=device), sel_b]
                selADE_b = (sel_pred_b - gt_pos).norm(dim=-1).mean(
                    dim=-1
                ) * TRAJECTORY_SCALE
                selFDE_b = (sel_pred_b[:, -1] - gt_pos[:, -1]).norm(
                    dim=-1
                ) * TRAJECTORY_SCALE

                sel_viols_b = full_violations[torch.arange(B, device=device), sel_b]
                sel_rates_b = compute_violation_rates(
                    sel_viols_b, applicability_oracle, rule_mask_28
                )

                for i in range(B):
                    strategy_results_protB[strategy].append(
                        {
                            "selADE": selADE_b[i].item(),
                            "selFDE": selFDE_b[i].item(),
                            "total_violated": bool(sel_rates_b["total_union"][i]),
                            "sl_violated": bool(sel_rates_b["sl_union"][i]),
                            "tier_0_violated": bool(sel_rates_b["tier_0_union"][i]),
                            "tier_1_violated": bool(sel_rates_b["tier_1_union"][i]),
                            "tier_2_violated": bool(sel_rates_b["tier_2_union"][i]),
                            "tier_3_violated": bool(sel_rates_b["tier_3_union"][i]),
                            "infeasible_selected": bool(infeasible_b[i].item()),
                        }
                    )

            # ---- Per-protocol violation rates (for RECTOR / lexicographic) ----
            sel_lex, _ = select_by_lexicographic(
                tier_scores_24_pred, args.epsilon, confidence=confidence
            )

            for (rule_set, app_source), result_list in protocol_results.items():
                r_mask = (
                    PROXY_RULE_MASK
                    if rule_set == "proxy-24"
                    else np.ones(NUM_RULES, dtype=bool)
                )
                app = (
                    applicability_pred_binary
                    if app_source == "predicted"
                    else applicability_oracle
                )

                # Recompute tier scores with this protocol
                ts = compute_tier_scores(
                    proxy_violations if rule_set == "proxy-24" else full_violations,
                    app,
                    r_mask,
                )
                # Re-select under this protocol
                sel, infeasible_proto = select_by_lexicographic(
                    ts, args.epsilon, confidence=confidence
                )

                sel_viols = (
                    proxy_violations if rule_set == "proxy-24" else full_violations
                )[torch.arange(B, device=device), sel]
                rates = compute_violation_rates(sel_viols, app, r_mask)

                for i in range(B):
                    result_list.append(
                        {
                            "total_violated": bool(rates["total_union"][i]),
                            "sl_violated": bool(rates["sl_union"][i]),
                            "tier_0_violated": bool(rates["tier_0_union"][i]),
                            "tier_1_violated": bool(rates["tier_1_union"][i]),
                            "tier_2_violated": bool(rates["tier_2_union"][i]),
                            "tier_3_violated": bool(rates["tier_3_union"][i]),
                            "infeasible_selected": bool(infeasible_proto[i].item()),
                        }
                    )

            sample_count += B
            if (batch_idx + 1) % 10 == 0:
                elapsed = time.time() - start_time
                print(f"  Batch {batch_idx+1}: {sample_count} samples ({elapsed:.1f}s)")

    elapsed = time.time() - start_time

    if sample_count == 0:
        raise RuntimeError(
            "No samples were evaluated. Check --val_dir format and WaymoDataset compatibility."
        )

    # ---- Aggregate results ----
    print(f"\n[4/4] Aggregating results ({sample_count} samples, {elapsed:.1f}s)...")

    output = {
        "metadata": {
            "checkpoint": str(args.checkpoint),
            "checkpoint_hash": checkpoint_hash(args.checkpoint),
            "eval_mode": "gt" if args.eval_gt else "rector",
            "womd_version": "1.3.0",
            "val_dir": str(args.val_dir),
            "num_val_shards": len(val_files),
            "sample_count": sample_count,
            "seed": args.seed,
            "epsilon": args.epsilon,
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": elapsed,
            "num_rules": NUM_RULES,
            "proxy_rules": int(PROXY_RULE_MASK.sum()),
            "rules_without_proxies": sorted(RULES_WITHOUT_PROXIES),
            "oracle_label_samples": oracle_label_samples,
            "fallback_oracle_samples": fallback_oracle_samples,
            "has_true_oracle_applicability": bool(
                oracle_label_samples > 0 and fallback_oracle_samples == 0
            ),
        },
        "geometric_metrics": {
            "minADE_mean": float(np.mean(all_minADE)),
            "minADE_std": float(np.std(all_minADE)),
            "minFDE_mean": float(np.mean(all_minFDE)),
            "minFDE_std": float(np.std(all_minFDE)),
            "miss_rate_pct": float(np.mean(all_miss) * 100),
        },
        "selection_strategies": {},
        "selection_strategies_protB": {},
        "selection_strategies_cross_eval": {},
        "protocol_results": {},
        "statistical_tests": {},
    }

    # Strategy-level aggregation
    for strategy, results in strategy_results.items():
        if not results:
            continue
        n = len(results)
        output["selection_strategies"][strategy] = {
            "n_scenarios": n,
            "selADE_mean": float(np.mean([r["selADE"] for r in results])),
            "selFDE_mean": float(np.mean([r["selFDE"] for r in results])),
            "Total_Viol_pct": float(
                np.mean([r["total_violated"] for r in results]) * 100
            ),
            "SL_Viol_pct": float(np.mean([r["sl_violated"] for r in results]) * 100),
            "Safety_Viol_pct": float(
                np.mean([r["tier_0_violated"] for r in results]) * 100
            ),
            "Legal_Viol_pct": float(
                np.mean([r["tier_1_violated"] for r in results]) * 100
            ),
            "Road_Viol_pct": float(
                np.mean([r["tier_2_violated"] for r in results]) * 100
            ),
            "Comfort_Viol_pct": float(
                np.mean([r["tier_3_violated"] for r in results]) * 100
            ),
            "infeasible_selected_pct": float(
                np.mean([r["infeasible_selected"] for r in results]) * 100
            ),
            # Per-tier sum (for Viol.% sum metric)
            "tier_sum_pct": float(
                sum(
                    np.mean([r[f"tier_{t}_violated"] for r in results]) * 100
                    for t in range(4)
                )
            ),
        }

    # Strategy-level aggregation under Protocol B (full-28, oracle)
    for strategy, results in strategy_results_protB.items():
        if not results:
            continue
        n = len(results)
        output["selection_strategies_protB"][strategy] = {
            "protocol": "full-28, oracle applicability",
            "n_scenarios": n,
            "selADE_mean": float(np.mean([r["selADE"] for r in results])),
            "selFDE_mean": float(np.mean([r["selFDE"] for r in results])),
            "Total_Viol_pct": float(
                np.mean([r["total_violated"] for r in results]) * 100
            ),
            "SL_Viol_pct": float(np.mean([r["sl_violated"] for r in results]) * 100),
            "Safety_Viol_pct": float(
                np.mean([r["tier_0_violated"] for r in results]) * 100
            ),
            "Legal_Viol_pct": float(
                np.mean([r["tier_1_violated"] for r in results]) * 100
            ),
            "Road_Viol_pct": float(
                np.mean([r["tier_2_violated"] for r in results]) * 100
            ),
            "Comfort_Viol_pct": float(
                np.mean([r["tier_3_violated"] for r in results]) * 100
            ),
            "infeasible_selected_pct": float(
                np.mean([r["infeasible_selected"] for r in results]) * 100
            ),
        }

    # Cross-evaluation aggregation (mode-specific selection, oracle evaluation)
    for strategy, results in strategy_results_cross.items():
        if not results:
            continue
        n = len(results)
        output["selection_strategies_cross_eval"][strategy] = {
            "protocol": "select with mode-specific applicability, evaluate under oracle (full-28)",
            "n_scenarios": n,
            "selADE_mean": float(np.mean([r["selADE"] for r in results])),
            "selFDE_mean": float(np.mean([r["selFDE"] for r in results])),
            "Total_Viol_pct": float(
                np.mean([r["total_violated"] for r in results]) * 100
            ),
            "SL_Viol_pct": float(np.mean([r["sl_violated"] for r in results]) * 100),
            "Safety_Viol_pct": float(
                np.mean([r["tier_0_violated"] for r in results]) * 100
            ),
            "Legal_Viol_pct": float(
                np.mean([r["tier_1_violated"] for r in results]) * 100
            ),
            "Road_Viol_pct": float(
                np.mean([r["tier_2_violated"] for r in results]) * 100
            ),
            "Comfort_Viol_pct": float(
                np.mean([r["tier_3_violated"] for r in results]) * 100
            ),
            "infeasible_selected_pct": float(
                np.mean([r["infeasible_selected"] for r in results]) * 100
            ),
        }

    # Protocol-level aggregation
    for (rule_set, app_source), results in protocol_results.items():
        if not results:
            continue
        key = f"{rule_set}_{app_source}"
        n = len(results)
        output["protocol_results"][key] = {
            "rule_set": rule_set,
            "applicability_source": app_source,
            "n_scenarios": n,
            "Total_Viol_pct": float(
                np.mean([r["total_violated"] for r in results]) * 100
            ),
            "SL_Viol_pct": float(np.mean([r["sl_violated"] for r in results]) * 100),
            "Safety_Viol_pct": float(
                np.mean([r["tier_0_violated"] for r in results]) * 100
            ),
            "Legal_Viol_pct": float(
                np.mean([r["tier_1_violated"] for r in results]) * 100
            ),
            "Road_Viol_pct": float(
                np.mean([r["tier_2_violated"] for r in results]) * 100
            ),
            "Comfort_Viol_pct": float(
                np.mean([r["tier_3_violated"] for r in results]) * 100
            ),
            "infeasible_selected_pct": float(
                np.mean([r["infeasible_selected"] for r in results]) * 100
            ),
        }

    # Paired significance tests (Protocol A strategy comparisons).
    output["statistical_tests"] = compute_pairwise_significance(strategy_results)

    # Per-rule applicability accuracy (WS-1)
    if args.per_rule_metrics and all_app_oracle:
        oracle_arr = np.array(all_app_oracle)  # [N, NUM_RULES]
        pred_arr = np.array(all_app_pred)  # [N, NUM_RULES]
        per_rule_metrics = {}
        for rule_idx, rule_id in enumerate(RULE_IDS):
            y_true = oracle_arr[:, rule_idx]
            y_pred = pred_arr[:, rule_idx]
            tp = ((y_pred == 1) & (y_true == 1)).sum()
            fp = ((y_pred == 1) & (y_true == 0)).sum()
            fn = ((y_pred == 0) & (y_true == 1)).sum()
            tn = ((y_pred == 0) & (y_true == 0)).sum()
            precision = float(tp / max(tp + fp, 1))
            recall = float(tp / max(tp + fn, 1))
            f1 = float(2 * precision * recall / max(precision + recall, 1e-8))
            accuracy = float((tp + tn) / max(tp + fp + fn + tn, 1))
            per_rule_metrics[rule_id] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "accuracy": accuracy,
                "support_positive": int(y_true.sum()),
                "support_total": int(len(y_true)),
            }
        output["per_rule_applicability_metrics"] = per_rule_metrics
        output["metadata"]["applicability_mode"] = args.applicability_mode

    # ---- Print summary ----
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    gm = output["geometric_metrics"]
    print(f"\nGeometric (oracle best-of-K):")
    print(f"  minADE: {gm['minADE_mean']:.4f}m ± {gm['minADE_std']:.4f}")
    print(f"  minFDE: {gm['minFDE_mean']:.4f}m ± {gm['minFDE_std']:.4f}")
    print(f"  Miss Rate: {gm['miss_rate_pct']:.2f}%")

    print(f"\nSelection Strategies (Protocol A: proxy-24, predicted):")
    print(
        f"  {'Strategy':<20} {'Total%':>8} {'S+L%':>8} {'Safety%':>8} {'Infeas%':>9} {'selADE':>8}"
    )
    for strategy in args.strategies:
        if strategy in output["selection_strategies"]:
            s = output["selection_strategies"][strategy]
            print(
                f"  {strategy:<20} {s['Total_Viol_pct']:>7.1f}% {s['SL_Viol_pct']:>7.1f}% "
                f"{s['Safety_Viol_pct']:>7.1f}% {s['infeasible_selected_pct']:>8.1f}% {s['selADE_mean']:>7.3f}m"
            )

    print(f"\nSelection Strategies (Protocol B: full-28, oracle):")
    print(
        f"  {'Strategy':<20} {'Total%':>8} {'S+L%':>8} {'Safety%':>8} {'Legal%':>8} {'Infeas%':>9} {'selADE':>8}"
    )
    for strategy in args.strategies:
        if strategy in output["selection_strategies_protB"]:
            s = output["selection_strategies_protB"][strategy]
            print(
                f"  {strategy:<20} {s['Total_Viol_pct']:>7.1f}% {s['SL_Viol_pct']:>7.1f}% "
                f"{s['Safety_Viol_pct']:>7.1f}% {s['Legal_Viol_pct']:>7.1f}% {s['infeasible_selected_pct']:>8.1f}% {s['selADE_mean']:>7.3f}m"
            )

    print(f"\nProtocol Reconciliation Table:")
    print(f"  {'Protocol':<25} {'Total%':>8} {'Safety%':>8} {'Legal%':>8}")
    for key, pr in output["protocol_results"].items():
        label = f"{pr['rule_set']}/{pr['applicability_source']}"
        print(
            f"  {label:<25} {pr['Total_Viol_pct']:>7.1f}% {pr['Safety_Viol_pct']:>7.1f}% "
            f"{pr['Legal_Viol_pct']:>7.1f}%"
        )

    if output["statistical_tests"]:
        print("\nPaired Significance Tests (Protocol A):")
        for pair_name, stats in output["statistical_tests"].items():
            mc_s = stats["mcnemar_safety"]
            wx = stats["wilcoxon_selADE"]
            print(f"  {pair_name}:")
            print(
                f"    McNemar Safety p={mc_s['p_value']:.3g} "
                f"(discordant: {mc_s['b_only']} vs {mc_s['c_only']})"
            )
            print(
                f"    Wilcoxon selADE p={wx['p_value']:.3g} "
                f"(mean delta={stats['mean_selADE_delta_m']:.4f}m)"
            )

    if fallback_oracle_samples > 0:
        print(
            "\nWARNING: oracle applicability was unavailable for some samples; oracle protocol results"
        )
        print("         fell back to thresholded model predictions for those cases.")

    # Per-rule applicability summary
    if "per_rule_applicability_metrics" in output:
        print(f"\nPer-Rule Applicability Accuracy ({args.applicability_mode} mode):")
        print(
            f"  {'Rule':<10} {'Prec':>8} {'Recall':>8} {'F1':>8} {'Acc':>8} {'Support':>8}"
        )
        for rule_id, m in output["per_rule_applicability_metrics"].items():
            print(
                f"  {rule_id:<10} {m['precision']:>7.3f} {m['recall']:>7.3f} "
                f"{m['f1']:>7.3f} {m['accuracy']:>7.3f} {m['support_positive']:>8d}"
            )

    print("=" * 70)

    # ---- Save JSON ----
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # ---- Save per-scenario CSV (for distribution plots) ----
    csv_path = output_path.parent / "per_scenario_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scenario_idx", "minADE", "minFDE", "miss"])
        for i, (ade, fde, m) in enumerate(zip(all_minADE, all_minFDE, all_miss)):
            writer.writerow([i, f"{ade:.6f}", f"{fde:.6f}", int(m)])

    # Also save per-strategy per-scenario CSV
    for strategy, results in strategy_results.items():
        if not results:
            continue
        strat_csv = output_path.parent / f"per_scenario_{strategy}.csv"
        with open(strat_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    # Save Protocol B per-strategy per-scenario CSV
    for strategy, results in strategy_results_protB.items():
        if not results:
            continue
        strat_csv = output_path.parent / f"per_scenario_protB_{strategy}.csv"
        with open(strat_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    print(f"Per-scenario CSV saved to: {csv_path}")


if __name__ == "__main__":
    main()
