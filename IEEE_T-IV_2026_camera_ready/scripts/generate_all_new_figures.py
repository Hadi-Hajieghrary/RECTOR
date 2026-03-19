#!/usr/bin/env python3
"""
Generate all new figures for IEEE T-IV 2026 RECTOR paper.
All data sourced from real evaluation results.

Figures generated:
  1. GT-vs-RECTOR Waterfall Chart (violation reduction breakdown)
  2. Epsilon Sensitivity Dual-Axis Curve
  3. Tail-Risk Survival Curve (CCDF of per-scenario ADE)
  4. Pareto Front with Bootstrap CIs (selADE vs Total Violations)
  5. Per-Rule Violation Heatmap (RECTOR predictions vs Ground Truth)
  6. Training Loss Decomposition (stacked area)
  7. Protocol Comparison Heatmap (Protocol A vs B)
  8. Test-Set Generalization (val vs test with percentile whiskers)

Usage:
    python generate_all_new_figures.py
"""

import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR = os.path.join(os.path.dirname(BASE), "Source_Reference", "output", "evaluation")
RULE_DIR = os.path.join(os.path.dirname(BASE), "Source_Reference", "output", "rule_evaluation")
METRICS_DIR = os.path.join(os.path.dirname(BASE), "Source_Reference", "models", "RECTOR",
                           "output", "delta_prediction_v2", "metrics")
FIG_DIR = os.path.join(BASE, "Figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ── IEEE Style ───────────────────────────────────────────────────────────────
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
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.5,
    'lines.markersize': 5,
})

# IEEE column widths
COL_W = 3.5   # single column (inches)
DBL_W = 7.16  # double column

# Color palette
COLORS = {
    'rector': '#2166AC',
    'gt': '#B2182B',
    'confidence': '#4DAF4A',
    'weighted': '#FF7F00',
    'lexicographic': '#2166AC',
    'safety': '#D32F2F',
    'legal': '#FF9800',
    'road': '#4CAF50',
    'comfort': '#2196F3',
    'total': '#333333',
}

TIER_COLORS = [COLORS['safety'], COLORS['legal'], COLORS['road'], COLORS['comfort']]
TIER_NAMES = ['Safety', 'Legal', 'Road', 'Comfort']


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_fig(fig, name):
    """Save figure as both PNG and PDF."""
    for ext in ['png', 'pdf']:
        path = os.path.join(FIG_DIR, f"{name}.{ext}")
        fig.savefig(path, format=ext)
    print(f"  Saved: {name}.png + .pdf")
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 1: GT-vs-RECTOR Waterfall Chart
# ═════════════════════════════════════════════════════════════════════════════
def fig_waterfall():
    print("[1/8] GT-vs-RECTOR Waterfall Chart")
    gt = load_json(os.path.join(EVAL_DIR, "canonical_gt_results.json"))
    rector = load_json(os.path.join(EVAL_DIR, "canonical_results.json"))

    gt_conf = gt['selection_strategies']['confidence']
    rec_lex = rector['selection_strategies']['lexicographic']

    # Categories and values
    tiers = ['Total', 'Legal', 'Road', 'Comfort']
    keys = ['Total_Viol_pct', 'Legal_Viol_pct', 'Road_Viol_pct', 'Comfort_Viol_pct']
    gt_vals = [gt_conf[k] for k in keys]
    rec_vals = [rec_lex[k] for k in keys]
    deltas = [g - r for g, r in zip(gt_vals, rec_vals)]

    fig, ax = plt.subplots(figsize=(COL_W, 2.6))

    x = np.arange(len(tiers))
    width = 0.32

    # GT bars (starting from 0)
    bars_gt = ax.bar(x - width/2, gt_vals, width, color=COLORS['gt'],
                     label='M2I (conf.)', alpha=0.85, edgecolor='white', linewidth=0.5)
    # RECTOR bars
    bars_rec = ax.bar(x + width/2, rec_vals, width, color=COLORS['rector'],
                      label='RECTOR (lex.)', alpha=0.85, edgecolor='white', linewidth=0.5)

    # Delta annotations
    for i in range(len(tiers)):
        pct_change = (deltas[i] / gt_vals[i]) * 100 if gt_vals[i] > 0 else 0
        mid_x = x[i] + width/2
        top_y = max(gt_vals[i], rec_vals[i])
        if deltas[i] > 0.5:
            ax.annotate(f'$\\downarrow${deltas[i]:.1f}pp\n({pct_change:.0f}%)',
                        xy=(x[i], top_y + 1.5),
                        fontsize=7, ha='center', va='bottom',
                        color='#1B5E20', fontweight='bold')

    ax.set_ylabel('Violation Rate (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(tiers)
    ax.legend(loc='upper right', framealpha=0.9, edgecolor='#ccc')
    ax.set_ylim(0, max(gt_vals) * 1.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add value labels on bars
    for bars in [bars_gt, bars_rec]:
        for bar in bars:
            h = bar.get_height()
            if h > 0.5:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.3,
                        f'{h:.1f}', ha='center', va='bottom', fontsize=7)

    fig.tight_layout(pad=0.5)
    save_fig(fig, 'gt_vs_rector_waterfall')


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 2: Epsilon Sensitivity Dual-Axis Curve
# ═════════════════════════════════════════════════════════════════════════════
def fig_epsilon_sensitivity():
    print("[2/8] Epsilon Sensitivity Curve")
    epsilons = [0.0, 0.0001, 0.001, 0.005, 0.01, 0.05, 0.1]
    comfort_viol = []
    sel_ade = []
    legal_viol = []
    total_viol = []

    for eps in epsilons:
        d = load_json(os.path.join(EVAL_DIR, f"epsilon_{eps}.json"))
        lex = d['selection_strategies']['lexicographic']
        comfort_viol.append(lex['Comfort_Viol_pct'])
        sel_ade.append(lex['selADE_mean'])
        legal_viol.append(lex['Legal_Viol_pct'])
        total_viol.append(lex['Total_Viol_pct'])

    fig, ax1 = plt.subplots(figsize=(COL_W, 2.4))

    # Left axis: Violation rates
    ax1.set_xlabel('Comfort Tolerance $\\varepsilon$')
    ax1.set_ylabel('Violation Rate (%)')

    l1, = ax1.plot(range(len(epsilons)), comfort_viol, 's-',
                   color=COLORS['comfort'], label='Comfort Viol.', markersize=5)
    l2, = ax1.plot(range(len(epsilons)), legal_viol, '^-',
                   color=COLORS['legal'], label='Legal Viol.', markersize=5)
    l3, = ax1.plot(range(len(epsilons)), total_viol, 'D-',
                   color=COLORS['total'], label='Total Viol.', markersize=4)

    ax1.set_xticks(range(len(epsilons)))
    ax1.set_xticklabels([str(e) for e in epsilons], fontsize=7)

    # Right axis: selADE
    ax2 = ax1.twinx()
    ax2.set_ylabel('selADE (m)', color=COLORS['rector'])
    l4, = ax2.plot(range(len(epsilons)), sel_ade, 'o--',
                   color=COLORS['rector'], label='selADE', markersize=5, alpha=0.8)
    ax2.tick_params(axis='y', labelcolor=COLORS['rector'])

    # Highlight operating point (ε=0.001)
    op_idx = 2  # ε=0.001
    ax1.axvline(x=op_idx, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    ax1.annotate('default\n$\\varepsilon$=0.001',
                 xy=(op_idx, comfort_viol[op_idx]),
                 xytext=(op_idx + 1.2, comfort_viol[op_idx] + 0.6),
                 fontsize=7, ha='left',
                 arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))

    # Combined legend
    lines = [l1, l2, l3, l4]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right', fontsize=7, framealpha=0.9)

    ax1.spines['top'].set_visible(False)
    fig.tight_layout(pad=0.5)
    save_fig(fig, 'epsilon_sensitivity')


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 3: Tail-Risk Survival Curve (CCDF)
# ═════════════════════════════════════════════════════════════════════════════
def fig_tail_risk():
    print("[3/8] Tail-Risk Survival Curve (CCDF)")
    df = pd.read_csv(os.path.join(EVAL_DIR, "per_scenario_metrics.csv"))

    ade = np.sort(df['minADE'].values)
    fde = np.sort(df['minFDE'].values)
    n = len(ade)
    ccdf = 1.0 - np.arange(1, n + 1) / n

    fig, ax = plt.subplots(figsize=(COL_W, 2.4))

    ax.semilogy(ade, ccdf, '-', color=COLORS['rector'], label='minADE', linewidth=1.2)
    ax.semilogy(fde, ccdf, '-', color=COLORS['gt'], label='minFDE', linewidth=1.2, alpha=0.8)

    # Mark percentiles
    for pct, ls in [(50, ':'), (90, '--'), (95, '-.')]:
        val_ade = np.percentile(ade, pct)
        val_fde = np.percentile(fde, pct)
        surv = 1.0 - pct / 100.0
        ax.axhline(y=surv, color='gray', linestyle=ls, alpha=0.3, linewidth=0.7)
        ax.plot(val_ade, surv, 'o', color=COLORS['rector'], markersize=4)
        ax.plot(val_fde, surv, 's', color=COLORS['gt'], markersize=4)
        ax.text(max(ade) * 0.7, surv * 1.2, f'p{pct}', fontsize=7, color='gray')

    ax.set_xlabel('Error (m)')
    ax.set_ylabel('P(Error > x)')
    ax.set_xlim(0, np.percentile(ade, 99) * 1.1)
    ax.set_ylim(1e-4, 1.0)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Annotate tail region
    p95_ade = np.percentile(ade, 95)
    ax.axvspan(p95_ade, ax.get_xlim()[1], alpha=0.08, color='red')
    ax.text(p95_ade + 0.2, 0.3, 'Tail\n(top 5%)', fontsize=7, color='#B71C1C', fontstyle='italic')

    fig.tight_layout(pad=0.5)
    save_fig(fig, 'tail_risk_survival')


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 4: Pareto Front with Bootstrap CIs
# ═════════════════════════════════════════════════════════════════════════════
def fig_pareto_front():
    print("[4/8] Pareto Front with Bootstrap CIs")
    canonical = load_json(os.path.join(EVAL_DIR, "canonical_results.json"))
    bootstrap = load_json(os.path.join(EVAL_DIR, "bootstrap_cis.json"))
    gt = load_json(os.path.join(EVAL_DIR, "canonical_gt_results.json"))

    strategies = ['confidence', 'weighted_sum', 'lexicographic']
    labels = ['Confidence', 'Weighted Sum', 'Lexicographic\n(RECTOR)']
    colors = [COLORS['confidence'], COLORS['weighted'], COLORS['lexicographic']]
    markers = ['o', 's', 'D']

    fig, ax = plt.subplots(figsize=(COL_W, 2.8))

    for i, (strat, label, color, marker) in enumerate(zip(strategies, labels, colors, markers)):
        s = canonical['selection_strategies'][strat]
        b = bootstrap['metrics'][strat]

        x = s['selADE_mean']
        y = s['Total_Viol_pct']
        y_lo = b['Total_Viol_pct']['ci_95_low']
        y_hi = b['Total_Viol_pct']['ci_95_high']

        ax.errorbar(x, y, yerr=[[y - y_lo], [y_hi - y]],
                    fmt=marker, color=color, markersize=7,
                    capsize=3, capthick=1.2, linewidth=1.2,
                    label=label, zorder=5)

    # GT baseline
    gt_x = gt['selection_strategies']['confidence']['selADE_mean']
    gt_y = gt['selection_strategies']['confidence']['Total_Viol_pct']
    ax.plot(gt_x, gt_y, '*', color=COLORS['gt'], markersize=12,
            label='M2I Baseline (GT)', zorder=4, markeredgecolor='black', markeredgewidth=0.5)

    # Pareto arrow from GT → RECTOR
    rec_s = canonical['selection_strategies']['lexicographic']
    ax.annotate('',
                xy=(rec_s['selADE_mean'], rec_s['Total_Viol_pct']),
                xytext=(gt_x, gt_y),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.0,
                                linestyle='--', connectionstyle='arc3,rad=0.2'))
    ax.text((gt_x + rec_s['selADE_mean'])/2 + 0.05,
            (gt_y + rec_s['Total_Viol_pct'])/2 + 3,
            'Rule-aware\nselection', fontsize=7, ha='center', color='gray',
            fontstyle='italic')

    ax.set_xlabel('selADE (m)')
    ax.set_ylabel('Total Violation Rate (%)')
    ax.legend(loc='upper left', fontsize=7, framealpha=0.9, edgecolor='#ccc')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Shade Pareto-optimal region
    ax.axhspan(0, rec_s['Total_Viol_pct'], xmin=0, xmax=1, alpha=0.04, color='green')

    fig.tight_layout(pad=0.5)
    save_fig(fig, 'pareto_front_ci')


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 5: Per-Rule Violation Heatmap (RECTOR vs GT)
# ═════════════════════════════════════════════════════════════════════════════
def fig_per_rule_heatmap():
    print("[5/8] Per-Rule Violation Heatmap")
    rector = load_json(os.path.join(RULE_DIR, "rector_predictions.json"))
    gt = load_json(os.path.join(RULE_DIR, "ground_truth.json"))

    # Collect all rules in tier order
    all_rules = []
    rule_tiers = []
    for tier in TIER_NAMES:
        rules_in_tier = sorted(rector['per_rule_violations'][tier].keys())
        all_rules.extend(rules_in_tier)
        rule_tiers.extend([tier] * len(rules_in_tier))

    # Compute violation rates
    rector_rates = []
    gt_rates = []
    for tier in TIER_NAMES:
        for rule in sorted(rector['per_rule_violations'][tier].keys()):
            r_viol = rector['per_rule_violations'][tier][rule]
            r_appl = rector['per_rule_applicable'][tier][rule]
            g_viol = gt['per_rule_violations'][tier].get(rule, 0)
            g_appl = gt['per_rule_applicable'][tier].get(rule, 1)
            rector_rates.append(r_viol / r_appl * 100 if r_appl > 0 else 0)
            gt_rates.append(g_viol / g_appl * 100 if g_appl > 0 else 0)

    # Create matrix: rows = rules, cols = [GT, RECTOR]
    matrix = np.array([gt_rates, rector_rates]).T  # shape: (n_rules, 2)

    n_rules = len(all_rules)
    fig_h = max(3.0, 0.18 * n_rules + 0.6)
    fig, ax = plt.subplots(figsize=(COL_W, fig_h))

    # Custom colormap: green (low) → yellow (mid) → red (high)
    cmap = LinearSegmentedColormap.from_list('viol',
        ['#E8F5E9', '#FFF9C4', '#FFCDD2', '#C62828'], N=256)

    im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=0, vmax=100)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Ground Truth', 'RECTOR'], fontsize=8)
    ax.set_yticks(range(len(all_rules)))
    ax.set_yticklabels(all_rules, fontsize=7)

    # Add text annotations — use luminance for readability
    for i in range(len(all_rules)):
        for j in range(2):
            val = matrix[i, j]
            rgba = cmap(val / 100.0)
            lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            color = 'white' if lum < 0.55 else 'black'
            ax.text(j, i, f'{val:.0f}%', ha='center', va='center',
                    fontsize=7, color=color, fontweight='bold')

    # Add tier separators and labels
    tier_boundaries = []
    current_count = 0
    for tier in TIER_NAMES:
        n = len(rector['per_rule_violations'][tier])
        if current_count > 0:
            ax.axhline(y=current_count - 0.5, color='white', linewidth=2)
        tier_boundaries.append((current_count, current_count + n, tier))
        current_count += n

    # Tier labels on the left
    for start, end, tier in tier_boundaries:
        mid = (start + end - 1) / 2
        color = COLORS[tier.lower()]
        ax.annotate(tier, xy=(-0.12, mid), xycoords=('axes fraction', 'data'),
                    fontsize=7, fontweight='bold', color=color,
                    ha='right', va='center')

    cbar = fig.colorbar(im, ax=ax, shrink=0.5, pad=0.02, aspect=30)
    cbar.set_label('Violation Rate (%)', fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    fig.subplots_adjust(left=0.22, right=0.92, top=0.97, bottom=0.04)
    save_fig(fig, 'per_rule_violation_heatmap')


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 6: Training Loss Decomposition
# ═════════════════════════════════════════════════════════════════════════════
def fig_training_loss():
    print("[6/8] Training Loss Decomposition")
    metrics = load_json(os.path.join(METRICS_DIR, "val_metrics.json"))

    epochs = np.array(metrics['epochs'])
    recon = np.array(metrics['recon_loss'])
    appl = np.array(metrics['applicability_loss'])
    temporal = np.array(metrics['temporal_loss'])
    total = np.array(metrics['total_loss'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DBL_W, 2.6))

    # Left panel: Total loss + components (log scale)
    ax1.semilogy(epochs, total, '-', color=COLORS['total'], label='Total Loss', linewidth=2)
    ax1.semilogy(epochs, recon, '--', color=COLORS['rector'], label='Reconstruction', linewidth=1.2)
    ax1.semilogy(epochs, temporal, '-.', color=COLORS['comfort'], label='Temporal', linewidth=1.2)
    ax1.semilogy(epochs, appl, ':', color=COLORS['legal'], label='Applicability', linewidth=1.2)

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (log scale)')
    ax1.legend(loc='upper right', fontsize=7, framealpha=0.9)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_title('(a) Loss Components', fontsize=9)

    # Right panel: ADE and FDE convergence
    ade = np.array(metrics['ade'])
    fde = np.array(metrics['fde'])

    ax2.plot(epochs, ade, 'o-', color=COLORS['rector'], label='ADE', markersize=3, linewidth=1.2)
    ax2.plot(epochs, fde, 's-', color=COLORS['gt'], label='FDE', markersize=3, linewidth=1.2)

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Error (m)')
    ax2.legend(loc='upper right', fontsize=7, framealpha=0.9)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_title('(b) Geometric Error Convergence', fontsize=9)

    # Highlight best epoch
    best_ade_epoch = epochs[np.argmin(ade)]
    best_ade_val = np.min(ade)
    ax2.annotate(f'Best ADE: {best_ade_val:.1f}m\n(epoch {best_ade_epoch})',
                 xy=(best_ade_epoch, best_ade_val),
                 xytext=(best_ade_epoch + 3, best_ade_val + 5),
                 fontsize=7, ha='left',
                 arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))

    fig.tight_layout(pad=0.5)
    save_fig(fig, 'training_loss_decomposition')


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 7: Protocol Comparison Heatmap
# ═════════════════════════════════════════════════════════════════════════════
def fig_protocol_comparison():
    print("[7/8] Protocol Comparison Heatmap")
    canonical = load_json(os.path.join(EVAL_DIR, "canonical_results.json"))
    gt = load_json(os.path.join(EVAL_DIR, "canonical_gt_results.json"))

    # Protocols
    protocols = ['proxy-24_predicted', 'proxy-24_oracle', 'full-28_predicted', 'full-28_oracle']
    proto_labels = ['Proxy-24\nPredicted', 'Proxy-24\nOracle', 'Full-28\nPredicted', 'Full-28\nOracle']
    tiers = ['Total_Viol_pct', 'Legal_Viol_pct', 'Road_Viol_pct', 'Comfort_Viol_pct']
    tier_labels = ['Total', 'Legal', 'Road', 'Comfort']

    # Build RECTOR matrix
    rector_matrix = np.zeros((len(protocols), len(tiers)))
    for i, proto in enumerate(protocols):
        for j, tier in enumerate(tiers):
            rector_matrix[i, j] = canonical['protocol_results'][proto][tier]

    # Build GT matrix (GT doesn't change with protocol, but show for reference)
    gt_matrix = np.zeros((len(protocols), len(tiers)))
    for i in range(len(protocols)):
        for j, tier in enumerate(tiers):
            gt_matrix[i, j] = gt['selection_strategies']['confidence'][tier]

    # Compute reduction matrix
    reduction_matrix = gt_matrix - rector_matrix

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DBL_W, 2.4))

    # Left: RECTOR violation rates
    cmap1 = LinearSegmentedColormap.from_list('viol', ['#E8F5E9', '#FFF9C4', '#FFCDD2'], N=256)
    im1 = ax1.imshow(rector_matrix, cmap=cmap1, aspect='auto', vmin=0, vmax=15)
    ax1.set_xticks(range(len(tier_labels)))
    ax1.set_xticklabels(tier_labels, fontsize=8)
    ax1.set_yticks(range(len(proto_labels)))
    ax1.set_yticklabels(proto_labels, fontsize=7)
    for i in range(len(protocols)):
        for j in range(len(tiers)):
            val = rector_matrix[i,j]
            rgba = cmap1(val / 15.0)
            lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            ax1.text(j, i, f'{val:.1f}%', ha='center', va='center',
                     fontsize=7, fontweight='bold',
                     color='white' if lum < 0.55 else 'black')
    ax1.set_title('(a) RECTOR Violation Rates (%)', fontsize=9)
    fig.colorbar(im1, ax=ax1, shrink=0.8, pad=0.02).ax.tick_params(labelsize=7)

    # Right: Reduction from GT
    cmap2 = LinearSegmentedColormap.from_list('red', ['#FFEBEE', '#A5D6A7', '#1B5E20'], N=256)
    im2 = ax2.imshow(reduction_matrix, cmap=cmap2, aspect='auto', vmin=0, vmax=60)
    ax2.set_xticks(range(len(tier_labels)))
    ax2.set_xticklabels(tier_labels, fontsize=8)
    ax2.set_yticks(range(len(proto_labels)))
    ax2.set_yticklabels(proto_labels, fontsize=7)
    for i in range(len(protocols)):
        for j in range(len(tiers)):
            val = reduction_matrix[i, j]
            ax2.text(j, i, f'{val:+.1f}pp', ha='center', va='center',
                     fontsize=7, fontweight='bold',
                     color='white' if val > 30 else 'black')
    ax2.set_title('(b) Reduction vs. M2I Baseline (pp)', fontsize=9)
    fig.colorbar(im2, ax=ax2, shrink=0.8, pad=0.02).ax.tick_params(labelsize=7)

    fig.tight_layout(pad=0.5)
    save_fig(fig, 'protocol_comparison_heatmap')


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 8: Test-Set Generalization
# ═════════════════════════════════════════════════════════════════════════════
def fig_test_generalization():
    print("[8/8] Test-Set Generalization")
    canonical = load_json(os.path.join(EVAL_DIR, "canonical_results.json"))
    test = load_json(os.path.join(EVAL_DIR, "test_results.json"))

    # Validation metrics
    val_ade = canonical['geometric_metrics']['minADE_mean']
    val_fde = canonical['geometric_metrics']['minFDE_mean']
    val_miss = canonical['geometric_metrics']['miss_rate_pct']

    # Test metrics
    test_ade = test['metrics']['minADE']
    test_fde = test['metrics']['minFDE']
    test_miss = test['metrics']['miss_rate'] * 100  # convert to %

    # Test percentiles
    test_p50_ade = test['percentiles']['ade_50']
    test_p90_ade = test['percentiles']['ade_90']
    test_p95_ade = test['percentiles']['ade_95']
    test_p50_fde = test['percentiles']['fde_50']
    test_p90_fde = test['percentiles']['fde_90']
    test_p95_fde = test['percentiles']['fde_95']

    # Panel (a): Val vs Test comparison
    metrics = ['minADE\n(m)', 'minFDE\n(m)', 'Miss Rate\n(%)']
    val_vals = [val_ade, val_fde, val_miss]
    test_vals = [test_ade, test_fde, test_miss]

    fig_a, ax1 = plt.subplots(figsize=(COL_W, 2.2))
    x = np.arange(len(metrics))
    width = 0.32
    bars_val = ax1.bar(x - width/2, val_vals, width, color=COLORS['rector'],
                       label='Validation', alpha=0.85, edgecolor='white')
    bars_test = ax1.bar(x + width/2, test_vals, width, color=COLORS['gt'],
                        label='Test', alpha=0.85, edgecolor='white')

    for bars in [bars_val, bars_test]:
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2, h + 0.2,
                     f'{h:.2f}' if h < 10 else f'{h:.1f}%',
                     ha='center', va='bottom', fontsize=7)

    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics, fontsize=7)
    ax1.set_ylabel('Value')
    ax1.legend(fontsize=7, framealpha=0.9)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    fig_a.subplots_adjust(left=0.15, right=0.97, top=0.95, bottom=0.18)
    save_fig(fig_a, 'test_set_generalization_a')

    # Panel (b): Test percentile distribution
    percentiles = ['p50', 'p90', 'p95']
    ade_pcts = [test_p50_ade, test_p90_ade, test_p95_ade]
    fde_pcts = [test_p50_fde, test_p90_fde, test_p95_fde]

    fig_b, ax2 = plt.subplots(figsize=(COL_W, 2.2))
    x2 = np.arange(len(percentiles))
    bars_ade = ax2.bar(x2 - width/2, ade_pcts, width, color=COLORS['rector'],
                       label='ADE', alpha=0.85, edgecolor='white')
    bars_fde = ax2.bar(x2 + width/2, fde_pcts, width, color=COLORS['gt'],
                       label='FDE', alpha=0.85, edgecolor='white')

    for bars in [bars_ade, bars_fde]:
        for bar in bars:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2, h + 0.2,
                     f'{h:.1f}', ha='center', va='bottom', fontsize=7)

    ax2.set_xticks(x2)
    ax2.set_xticklabels(percentiles, fontsize=7)
    ax2.set_ylabel('Error (m)')
    ax2.legend(fontsize=7, framealpha=0.9)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    fig_b.subplots_adjust(left=0.15, right=0.97, top=0.95, bottom=0.18)
    save_fig(fig_b, 'test_set_generalization_b')


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("RECTOR Paper — New Figure Generation")
    print(f"Output directory: {FIG_DIR}")
    print("=" * 60)

    fig_waterfall()
    fig_epsilon_sensitivity()
    fig_tail_risk()
    fig_pareto_front()
    fig_per_rule_heatmap()
    fig_training_loss()
    fig_protocol_comparison()
    fig_test_generalization()

    print("\n" + "=" * 60)
    print("All 8 figures generated successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()
