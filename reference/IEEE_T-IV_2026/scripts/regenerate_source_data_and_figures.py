#!/usr/bin/env python3
"""
Regenerate source JSON files and all figures from the corrected experiment cache.

Reads:  /workspace/output/experiments/cache/per_mode_data.npz
Writes: /workspace/Source_Reference/output/evaluation/canonical_results.json
        /workspace/Source_Reference/output/evaluation/bootstrap_cis.json
        /workspace/Source_Reference/output/evaluation/epsilon_*.json
        /workspace/IEEE_T-IV_2026/Figures/*.{pdf,png}
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────
CACHE_PATH = Path('/workspace/output/experiments/cache/per_mode_data.npz')
EVAL_DIR = Path('/workspace/Source_Reference/output/evaluation')
FIG_DIR = Path('/workspace/IEEE_T-IV_2026/Figures')

# ── Load cache ─────────────────────────────────────────────────────────────────
print("Loading cached data...")
data = np.load(CACHE_PATH, allow_pickle=True)
N = data['proxy_violations'].shape[0]
print(f"  {N:,} scenarios loaded")

proxy_violations = data['proxy_violations']      # [N, 6, 28]
confidence = data['confidence']                    # [N, 6]
pred_positions = data['pred_positions']            # [N, 6, T, 2]
gt_positions = data['gt_positions']                # [N, T, 2]
tier_scores_learned = data['tier_scores_learned']  # [N, 6, 4]
proxy_rule_mask = data['proxy_rule_mask']           # [28]

# ── Tier indices ───────────────────────────────────────────────────────────────
# Tier assignments from the rule catalog (0=Safety, 1=Legal, 2=Road, 3=Comfort)
RULE_TIERS = np.array([
    0, 0, 0, 0, 0,       # L0.R1-R4, L10.R1 (Safety, indices 0-4)
    1, 1, 1, 1, 1, 1, 1,  # Legal rules (indices 5-11)
    2, 2, 2, 2, 2, 2, 2,  # Road rules (indices 12-18)
    3, 3, 3, 3, 3, 3, 3, 3, 3  # Comfort rules (indices 19-27)
])

# ── Selection functions ────────────────────────────────────────────────────────
def select_confidence(confidence_scores):
    """Select mode with highest confidence."""
    return np.argmax(confidence_scores, axis=1)  # [N]

def select_weighted_sum(tier_scores, weights=np.array([1000, 100, 10, 1])):
    """Select mode with lowest weighted-sum score."""
    weighted = np.sum(tier_scores * weights[None, None, :], axis=2)  # [N, K]
    return np.argmin(weighted, axis=1)  # [N]

def select_lexicographic(tier_scores, eps=1e-3):
    """Lexicographic selection: Safety > Legal > Road > Comfort."""
    N, K, T = tier_scores.shape
    alive = np.ones((N, K), dtype=bool)

    for t in range(T):
        scores_t = tier_scores[:, :, t]  # [N, K]
        # Mask dead candidates with inf
        masked = np.where(alive, scores_t, np.inf)
        best_t = np.min(masked, axis=1, keepdims=True)  # [N, 1]
        # Keep candidates within eps of best
        new_alive = alive & (scores_t <= best_t + eps)
        # Only update if at least one survives
        any_alive = np.any(new_alive, axis=1, keepdims=True)
        alive = np.where(any_alive, new_alive, alive)

    # Among survivors, pick lowest total score (then lowest index)
    total = np.sum(tier_scores, axis=2)  # [N, K]
    masked_total = np.where(alive, total, np.inf)
    return np.argmin(masked_total, axis=1)  # [N]

def compute_metrics(sel_indices, pred_positions, gt_positions, proxy_violations,
                    proxy_rule_mask, tier_scores):
    """Compute full metrics for selected modes."""
    N = sel_indices.shape[0]
    idx = sel_indices

    # Geometric: selected ADE/FDE
    sel_pos = pred_positions[np.arange(N), idx]  # [N, T, 2]
    errors = np.sqrt(np.sum((sel_pos - gt_positions) ** 2, axis=2))  # [N, T]
    sel_ade = np.mean(errors, axis=1)  # [N]
    sel_fde = errors[:, -1]  # [N]

    # Best-of-K (oracle): use ADE-best mode
    all_errors = np.sqrt(np.sum(
        (pred_positions - gt_positions[:, None, :, :]) ** 2, axis=3))  # [N, K, T]
    ade_per_mode = np.mean(all_errors, axis=2)  # [N, K]
    best_mode = np.argmin(ade_per_mode, axis=1)  # [N]
    min_ade = ade_per_mode[np.arange(N), best_mode]
    fde_per_mode = all_errors[:, :, -1]
    min_fde = fde_per_mode[np.arange(N), best_mode]
    miss = (min_fde > 2.0).astype(float)

    # Per-tier violations (from tier_scores: a tier is violated if score > 0)
    sel_tier = tier_scores[np.arange(N), idx]  # [N, 4]
    safety_viol = (sel_tier[:, 0] > 0).mean() * 100
    legal_viol = (sel_tier[:, 1] > 0).mean() * 100
    road_viol = (sel_tier[:, 2] > 0).mean() * 100
    comfort_viol = (sel_tier[:, 3] > 0).mean() * 100

    # S+L and Total
    sl_viol = ((sel_tier[:, 0] > 0) | (sel_tier[:, 1] > 0)).mean() * 100
    total_viol = (np.any(sel_tier > 0, axis=1)).mean() * 100

    return {
        'selADE_mean': float(np.mean(sel_ade)),
        'selFDE_mean': float(np.mean(sel_fde)),
        'miss_rate': float(miss.mean() * 100),
        'Total_Viol_pct': float(total_viol),
        'Safety_Viol_pct': float(safety_viol),
        'Legal_Viol_pct': float(legal_viol),
        'Road_Viol_pct': float(road_viol),
        'Comfort_Viol_pct': float(comfort_viol),
        'SL_Viol_pct': float(sl_viol),
        # Percentiles for selADE
        'selADE_p50': float(np.median(sel_ade)),
        'selADE_p90': float(np.percentile(sel_ade, 90)),
        'selADE_p95': float(np.percentile(sel_ade, 95)),
        # Oracle metrics
        'minADE_mean': float(np.mean(min_ade)),
        'minFDE_mean': float(np.mean(min_fde)),
        'miss_rate_pct': float(miss.mean() * 100),
    }


def bootstrap_ci(tier_scores, sel_func, n_bootstrap=10000, seed=42):
    """Compute 95% bootstrap CIs for per-tier violation rates."""
    rng = np.random.RandomState(seed)
    N = tier_scores.shape[0]

    metrics_keys = ['Total_Viol_pct', 'Safety_Viol_pct', 'Legal_Viol_pct',
                    'Road_Viol_pct', 'Comfort_Viol_pct']

    # Pre-compute selection
    sel_idx = sel_func()
    sel_tier = tier_scores[np.arange(N), sel_idx]  # [N, 4]

    # Per-scenario binary flags
    safety_flag = (sel_tier[:, 0] > 0).astype(float)
    legal_flag = (sel_tier[:, 1] > 0).astype(float)
    road_flag = (sel_tier[:, 2] > 0).astype(float)
    comfort_flag = (sel_tier[:, 3] > 0).astype(float)
    total_flag = np.any(sel_tier > 0, axis=1).astype(float)

    flags = np.stack([total_flag, safety_flag, legal_flag, road_flag, comfort_flag], axis=1)  # [N, 5]

    # Bootstrap
    boot_means = np.zeros((n_bootstrap, 5))
    for b in range(n_bootstrap):
        idx = rng.randint(0, N, size=N)
        boot_means[b] = flags[idx].mean(axis=0) * 100

    result = {}
    point_vals = flags.mean(axis=0) * 100
    for i, key in enumerate(metrics_keys):
        lo = np.percentile(boot_means[:, i], 2.5)
        hi = np.percentile(boot_means[:, i], 97.5)
        result[key] = {
            'point': float(point_vals[i]),
            'ci_95_low': float(lo),
            'ci_95_high': float(hi),
        }
    return result


# ── Step 1: Compute metrics for all three strategies ──────────────────────────
print("\nComputing selection strategies...")
sel_conf = select_confidence(confidence)
sel_ws = select_weighted_sum(tier_scores_learned)
sel_lex = select_lexicographic(tier_scores_learned)

metrics_conf = compute_metrics(sel_conf, pred_positions, gt_positions,
                                proxy_violations, proxy_rule_mask, tier_scores_learned)
metrics_ws = compute_metrics(sel_ws, pred_positions, gt_positions,
                              proxy_violations, proxy_rule_mask, tier_scores_learned)
metrics_lex = compute_metrics(sel_lex, pred_positions, gt_positions,
                               proxy_violations, proxy_rule_mask, tier_scores_learned)

print(f"  Confidence: selADE={metrics_conf['selADE_mean']:.3f}, "
      f"Safety={metrics_conf['Safety_Viol_pct']:.1f}%, "
      f"S+L={metrics_conf['SL_Viol_pct']:.1f}%, "
      f"Total={metrics_conf['Total_Viol_pct']:.1f}%")
print(f"  WS:         selADE={metrics_ws['selADE_mean']:.3f}, "
      f"Safety={metrics_ws['Safety_Viol_pct']:.1f}%, "
      f"S+L={metrics_ws['SL_Viol_pct']:.1f}%, "
      f"Total={metrics_ws['Total_Viol_pct']:.1f}%")
print(f"  RECTOR:     selADE={metrics_lex['selADE_mean']:.3f}, "
      f"Safety={metrics_lex['Safety_Viol_pct']:.1f}%, "
      f"S+L={metrics_lex['SL_Viol_pct']:.1f}%, "
      f"Total={metrics_lex['Total_Viol_pct']:.1f}%")

# ── Step 2: Write canonical_results.json ──────────────────────────────────────
canonical = {
    'n_scenarios': int(N),
    'n_bootstrap': 10000,
    'geometric_metrics': {
        'minADE_mean': metrics_lex['minADE_mean'],
        'minFDE_mean': metrics_lex['minFDE_mean'],
        'miss_rate_pct': metrics_lex['miss_rate_pct'],
    },
    'selection_strategies': {
        'confidence': {k: metrics_conf[k] for k in
                       ['selADE_mean', 'selFDE_mean', 'miss_rate',
                        'Total_Viol_pct', 'Safety_Viol_pct', 'Legal_Viol_pct',
                        'Road_Viol_pct', 'Comfort_Viol_pct']},
        'weighted_sum': {k: metrics_ws[k] for k in
                         ['selADE_mean', 'selFDE_mean', 'miss_rate',
                          'Total_Viol_pct', 'Safety_Viol_pct', 'Legal_Viol_pct',
                          'Road_Viol_pct', 'Comfort_Viol_pct']},
        'lexicographic': {k: metrics_lex[k] for k in
                          ['selADE_mean', 'selFDE_mean', 'miss_rate',
                           'Total_Viol_pct', 'Safety_Viol_pct', 'Legal_Viol_pct',
                           'Road_Viol_pct', 'Comfort_Viol_pct']},
    }
}

os.makedirs(EVAL_DIR, exist_ok=True)
with open(EVAL_DIR / 'canonical_results.json', 'w') as f:
    json.dump(canonical, f, indent=2)
print(f"\n✓ Wrote {EVAL_DIR / 'canonical_results.json'}")

# ── Step 3: Bootstrap CIs ────────────────────────────────────────────────────
print("\nComputing bootstrap CIs (10,000 resamples)...")
ci_conf = bootstrap_ci(tier_scores_learned, lambda: sel_conf)
ci_ws = bootstrap_ci(tier_scores_learned, lambda: sel_ws)
ci_lex = bootstrap_ci(tier_scores_learned, lambda: sel_lex)

bootstrap_data = {
    'n_bootstrap': 10000,
    'metrics': {
        'confidence': ci_conf,
        'weighted_sum': ci_ws,
        'lexicographic': ci_lex,
    }
}

with open(EVAL_DIR / 'bootstrap_cis.json', 'w') as f:
    json.dump(bootstrap_data, f, indent=2)
print(f"✓ Wrote {EVAL_DIR / 'bootstrap_cis.json'}")

# ── Step 4: Epsilon sensitivity JSONs ─────────────────────────────────────────
print("\nComputing epsilon sensitivity...")
epsilons = [0.0, 0.0001, 0.001, 0.005, 0.01, 0.05, 0.1]
for eps in epsilons:
    sel_eps = select_lexicographic(tier_scores_learned, eps=eps)
    m = compute_metrics(sel_eps, pred_positions, gt_positions,
                        proxy_violations, proxy_rule_mask, tier_scores_learned)
    eps_data = {
        'epsilon': eps,
        'n_scenarios': int(N),
        'selection_strategies': {
            'lexicographic': {k: m[k] for k in
                              ['selADE_mean', 'selFDE_mean',
                               'Total_Viol_pct', 'Safety_Viol_pct', 'Legal_Viol_pct',
                               'Road_Viol_pct', 'Comfort_Viol_pct']}
        }
    }
    with open(EVAL_DIR / f'epsilon_{eps}.json', 'w') as f:
        json.dump(eps_data, f, indent=2)
    print(f"  ε={eps}: Total={m['Total_Viol_pct']:.1f}%, selADE={m['selADE_mean']:.3f}")
print(f"✓ Wrote epsilon_*.json files")

# ── Step 5: Print summary ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Updated source data summary:")
print(f"  Oracle: minADE={metrics_lex['minADE_mean']:.3f}, "
      f"minFDE={metrics_lex['minFDE_mean']:.3f}, "
      f"Miss={metrics_lex['miss_rate_pct']:.1f}%")
print(f"  Confidence CIs: Total={ci_conf['Total_Viol_pct']['point']:.1f}% "
      f"[{ci_conf['Total_Viol_pct']['ci_95_low']:.1f}, "
      f"{ci_conf['Total_Viol_pct']['ci_95_high']:.1f}]")
print(f"  RECTOR CIs:     Total={ci_lex['Total_Viol_pct']['point']:.1f}% "
      f"[{ci_lex['Total_Viol_pct']['ci_95_low']:.1f}, "
      f"{ci_lex['Total_Viol_pct']['ci_95_high']:.1f}]")
print("=" * 60)
