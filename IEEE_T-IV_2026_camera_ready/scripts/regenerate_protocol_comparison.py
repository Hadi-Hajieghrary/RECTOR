#!/usr/bin/env python3
"""
Regenerate protocol_comparison_heatmap.pdf from experiment cache.

Shows 4 protocol variants: Proxy-24/Full-28 x Predicted/Oracle applicability.
Since 4 non-proxy rules have 0 cost, Proxy-24 ≡ Full-28 per applicability type.
The figure now includes the Safety column (post-fix, Safety ≠ 0%).

Data source: /workspace/output/experiments/cache/per_mode_data.npz
             /workspace/output/experiments/exp2/applicability_ablation.json
             /workspace/Source_Reference/output/evaluation/canonical_results.json
"""

import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────
FIG_DIR = '/workspace/IEEE_T-IV_2026/Figures'
CACHE_PATH = '/workspace/output/experiments/cache/per_mode_data.npz'
EXP2_PATH = '/workspace/output/experiments/exp2/applicability_ablation.json'
CANON_PATH = '/workspace/Source_Reference/output/evaluation/canonical_results.json'

# ── IEEE Style ─────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'legend.fontsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.01,
    'axes.grid': False,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.5,
    'lines.markersize': 5,
})

COL_W = 3.5
DBL_W = 7.16


def main():
    # Load data
    with open(EXP2_PATH) as f:
        exp2 = json.load(f)
    with open(CANON_PATH) as f:
        canon = json.load(f)

    # Protocol A (Proxy-24, Predicted) = exp2 "learned" condition
    # Protocol A (Proxy-24, Oracle)    = exp2 "oracle" condition
    # Protocol B (Full-28, Predicted)  = same as A-predicted (4 non-proxy rules have 0 cost)
    # Protocol B (Full-28, Oracle)     = same as A-oracle

    predicted = exp2['conditions']['learned']
    oracle = exp2['conditions']['oracle']

    # Tiers to show (now includes Safety)
    tier_keys = ['Safety_pct', 'Legal_pct', 'Road_pct', 'Comfort_pct', 'Total_pct']
    tier_labels = ['Safety', 'Legal', 'Road', 'Comfort', 'Total']

    # 4 protocol rows
    proto_labels = ['Proxy-24\nPredicted', 'Proxy-24\nOracle', 'Full-28\nPredicted', 'Full-28\nOracle']
    proto_data = [predicted, oracle, predicted, oracle]

    # Build RECTOR violation matrix
    rector_matrix = np.zeros((len(proto_labels), len(tier_labels)))
    for i, pdata in enumerate(proto_data):
        for j, tk in enumerate(tier_keys):
            rector_matrix[i, j] = pdata[tk]

    # Baseline = confidence strategy from canonical results
    conf = canon['selection_strategies']['confidence']
    conf_values = {
        'Safety_pct': conf['Safety_Viol_pct'],
        'Legal_pct': conf['Legal_Viol_pct'],
        'Road_pct': conf['Road_Viol_pct'],
        'Comfort_pct': conf['Comfort_Viol_pct'],
        'Total_pct': conf['Total_Viol_pct'],
    }

    # Build baseline matrix (same for all rows)
    baseline_matrix = np.zeros_like(rector_matrix)
    for i in range(len(proto_labels)):
        for j, tk in enumerate(tier_keys):
            baseline_matrix[i, j] = conf_values[tk]

    # Reduction matrix
    reduction_matrix = baseline_matrix - rector_matrix

    # ── Panel (a): RECTOR violation rates ────────────────────────────────────────
    vmax_left = min(100, rector_matrix.max() * 1.15)
    cmap1 = LinearSegmentedColormap.from_list('viol',
        ['#E8F5E9', '#FFF9C4', '#FFCDD2', '#C62828'], N=256)

    fig_a, ax1 = plt.subplots(figsize=(COL_W, 2.0))
    im1 = ax1.imshow(rector_matrix, cmap=cmap1, aspect='auto', vmin=0, vmax=vmax_left)
    ax1.set_xticks(range(len(tier_labels)))
    ax1.set_xticklabels(tier_labels, fontsize=7)
    ax1.set_yticks(range(len(proto_labels)))
    ax1.set_yticklabels(proto_labels, fontsize=7)
    for i in range(len(proto_labels)):
        for j in range(len(tier_labels)):
            val = rector_matrix[i, j]
            color = 'white' if val > 70 else 'black'
            ax1.text(j, i, f'{val:.1f}%', ha='center', va='center',
                     fontsize=7, fontweight='bold', color=color)
    fig_a.colorbar(im1, ax=ax1, shrink=0.7, pad=0.02, aspect=20).ax.tick_params(labelsize=7)
    fig_a.subplots_adjust(left=0.22, right=0.92, top=0.96, bottom=0.06)

    for ext in ['png', 'pdf']:
        path = os.path.join(FIG_DIR, f'protocol_comparison_heatmap_a.{ext}')
        fig_a.savefig(path, format=ext, dpi=300, bbox_inches='tight')
        print(f"  Saved: {path}")
    plt.close(fig_a)

    # ── Panel (b): Reduction from confidence baseline ─────────────────────────
    cmap2 = LinearSegmentedColormap.from_list('red',
        ['#FFEBEE', '#A5D6A7', '#1B5E20'], N=256)

    fig_b, ax2 = plt.subplots(figsize=(COL_W, 2.0))
    im2 = ax2.imshow(reduction_matrix, cmap=cmap2, aspect='auto',
                     vmin=0, vmax=max(75, reduction_matrix.max() * 1.1))
    ax2.set_xticks(range(len(tier_labels)))
    ax2.set_xticklabels(tier_labels, fontsize=7)
    ax2.set_yticks(range(len(proto_labels)))
    ax2.set_yticklabels(proto_labels, fontsize=7)
    for i in range(len(proto_labels)):
        for j in range(len(tier_labels)):
            val = reduction_matrix[i, j]
            color = 'white' if val > 40 else 'black'
            ax2.text(j, i, f'{val:+.1f}pp', ha='center', va='center',
                     fontsize=7, fontweight='bold', color=color)
    fig_b.colorbar(im2, ax=ax2, shrink=0.7, pad=0.02, aspect=20).ax.tick_params(labelsize=7)
    fig_b.subplots_adjust(left=0.22, right=0.92, top=0.96, bottom=0.06)

    for ext in ['png', 'pdf']:
        path = os.path.join(FIG_DIR, f'protocol_comparison_heatmap_b.{ext}')
        fig_b.savefig(path, format=ext, dpi=300, bbox_inches='tight')
        print(f"  Saved: {path}")
    plt.close(fig_b)


if __name__ == '__main__':
    main()
