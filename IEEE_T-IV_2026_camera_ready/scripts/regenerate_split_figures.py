#!/usr/bin/env python3
"""
Regenerate split-panel figures for IEEE T-IV 2026 RECTOR paper.

Reads real data from canonical_results.json and generates separate panel files
for multi-panel figures that had axis-label overlap when combined.

Figures generated:
  1. protocol_comparison_heatmap_a.pdf  — Violation rates
  2. protocol_comparison_heatmap_b.pdf  — Reduction from Confidence
  3. test_set_generalization_a.pdf      — Val vs Test aggregate
  4. test_set_generalization_b.pdf      — Test error percentiles
  5. selection_comparison_a.pdf         — Pareto scatter (selADE vs Total)
  6. selection_comparison_b.pdf         — Per-tier violation bars
  7. per_rule_violation_heatmap.pdf     — Fixed tier labels (single panel)

Usage:
    python regenerate_split_figures.py
"""

import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────
FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Figures')
CANON_PATH = '/workspace/output/evaluation/canonical_results.json'
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
    'axes.grid': False,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.5,
    'lines.markersize': 5,
})

COL_W = 3.5   # single column
DBL_W = 7.16  # double column

COLORS = {
    'rector': '#2166AC',
    'confidence': '#e53935',
    'weighted': '#f57c00',
    'lexicographic': '#1565c0',
    'safety': '#D32F2F',
    'legal': '#FF9800',
    'road': '#4CAF50',
    'comfort': '#2196F3',
    'total': '#333333',
}


def save_fig(fig, name):
    for ext in ['png', 'pdf']:
        path = os.path.join(FIG_DIR, f"{name}.{ext}")
        fig.savefig(path, format=ext, dpi=300, bbox_inches='tight')
    print(f"  Saved: {name}.pdf + .png")
    plt.close(fig)


def load_canonical():
    with open(CANON_PATH) as f:
        return json.load(f)


# ═════════════════════════════════════════════════════════════════════════════
# 1. Protocol Comparison Heatmap (split into _a and _b)
# ═════════════════════════════════════════════════════════════════════════════
def fig_protocol_comparison(canon):
    print("\n[1/4] Protocol Comparison Heatmap (split)")

    pr = canon['protocol_results']
    conf = canon['selection_strategies']['confidence']

    tier_labels = ['Total', 'Safety', 'Legal', 'Road', 'Comfort']
    tier_keys = ['Total_Viol_pct', 'Safety_Viol_pct', 'Legal_Viol_pct',
                 'Road_Viol_pct', 'Comfort_Viol_pct']

    proto_labels = ['Proxy-24\nPredicted', 'Proxy-24\nOracle',
                    'Full-28\nPredicted', 'Full-28\nOracle']
    proto_keys = ['proxy-24_predicted', 'proxy-24_oracle',
                  'full-28_predicted', 'full-28_oracle']

    # Build RECTOR violation matrix
    rector_matrix = np.array([
        [pr[pk][tk] for tk in tier_keys] for pk in proto_keys
    ])

    # Baseline (confidence) values
    conf_row = np.array([conf[tk] for tk in tier_keys])
    baseline_matrix = np.tile(conf_row, (len(proto_keys), 1))
    reduction_matrix = baseline_matrix - rector_matrix

    # ── Panel (a): Violation rates ───────────────────────────────────────
    cmap1 = LinearSegmentedColormap.from_list(
        'viol', ['#E8F5E9', '#FFF9C4', '#FFCDD2', '#C62828'], N=256)

    fig_a, ax1 = plt.subplots(figsize=(COL_W, 2.2))
    vmax_a = max(40, rector_matrix.max() * 1.15)
    im1 = ax1.imshow(rector_matrix, cmap=cmap1, aspect='auto', vmin=0, vmax=vmax_a)
    ax1.set_xticks(range(len(tier_labels)))
    ax1.set_xticklabels(tier_labels, fontsize=8)
    ax1.xaxis.set_ticks_position('bottom')
    ax1.set_xlabel('Violation Tier', fontsize=9)
    ax1.set_yticks(range(len(proto_labels)))
    ax1.set_yticklabels(proto_labels, fontsize=7)

    for i in range(len(proto_keys)):
        for j in range(len(tier_labels)):
            val = rector_matrix[i, j]
            color = 'white' if val > vmax_a * 0.7 else 'black'
            ax1.text(j, i, f'{val:.1f}', ha='center', va='center',
                     fontsize=7, fontweight='bold', color=color)

    cbar1 = fig_a.colorbar(im1, ax=ax1, shrink=0.75, pad=0.03, aspect=20)
    cbar1.set_label('Violation %', fontsize=7)
    cbar1.ax.tick_params(labelsize=7)
    ax1.set_title('(a) Violation Rate (%)', fontsize=10, pad=8)
    fig_a.subplots_adjust(left=0.24, right=0.88, top=0.90, bottom=0.14)
    save_fig(fig_a, 'protocol_comparison_heatmap_a')

    # ── Panel (b): Reduction from confidence ─────────────────────────────
    cmap2_colors = ['#C62828', '#FFCDD2', '#FFFDE7', '#A5D6A7', '#1B5E20']
    cmap2 = LinearSegmentedColormap.from_list('rdgn', cmap2_colors, N=256)
    abs_max = max(abs(reduction_matrix.min()), abs(reduction_matrix.max()), 1)

    fig_b, ax2 = plt.subplots(figsize=(COL_W, 2.2))
    im2 = ax2.imshow(reduction_matrix, cmap=cmap2, aspect='auto',
                     vmin=-abs_max, vmax=abs_max)
    ax2.set_xticks(range(len(tier_labels)))
    ax2.set_xticklabels(tier_labels, fontsize=8)
    ax2.xaxis.set_ticks_position('bottom')
    ax2.set_xlabel('Violation Tier', fontsize=9)
    ax2.set_yticks(range(len(proto_labels)))
    ax2.set_yticklabels(proto_labels, fontsize=7)

    for i in range(len(proto_keys)):
        for j in range(len(tier_labels)):
            val = reduction_matrix[i, j]
            color = 'white' if abs(val) > abs_max * 0.65 else 'black'
            ax2.text(j, i, f'{val:+.1f}', ha='center', va='center',
                     fontsize=7, fontweight='bold', color=color)

    cbar2 = fig_b.colorbar(im2, ax=ax2, shrink=0.75, pad=0.03, aspect=20)
    cbar2.set_label('pp reduction', fontsize=7)
    cbar2.ax.tick_params(labelsize=7)
    ax2.set_title('(b) Reduction from Confidence (pp)', fontsize=10, pad=8)
    fig_b.subplots_adjust(left=0.24, right=0.88, top=0.90, bottom=0.14)
    save_fig(fig_b, 'protocol_comparison_heatmap_b')


# ═════════════════════════════════════════════════════════════════════════════
# 2. Test-Set Generalization (split into _a and _b)
# ═════════════════════════════════════════════════════════════════════════════
def fig_test_generalization(canon):
    print("\n[2/4] Test-Set Generalization (split)")

    # Validation metrics from canonical
    val_ade = canon['geometric_metrics']['minADE_mean']
    val_fde = canon['geometric_metrics']['minFDE_mean']
    val_miss = canon['geometric_metrics']['miss_rate_pct']

    # Test metrics (from existing figure — hardcoded from rendered values)
    test_ade = 2.86
    test_fde = 5.63
    test_miss = 52.2

    # Validation percentiles (from percentile_analysis figure)
    val_pcts = {'p50': 0.46, 'p75': 0.87, 'p90': 1.59, 'p95': 2.16, 'p99': 3.62}
    # Test percentiles (from test_set_generalization figure)
    test_pcts = {'p50': 1.05, 'p75': 3.85, 'p90': 8.36, 'p95': 10.5, 'p99': 16.2}

    # ── Panel (a): Aggregate val vs test bars ────────────────────────────
    fig_a, ax1 = plt.subplots(figsize=(COL_W, 2.4))
    metrics = ['minADE\n(m)', 'minFDE\n(m)', 'Miss Rate\n(%)']
    val_vals = [val_ade, val_fde, val_miss]
    test_vals = [test_ade, test_fde, test_miss]

    x = np.arange(len(metrics))
    width = 0.32
    bars_val = ax1.bar(x - width / 2, val_vals, width,
                       color=COLORS['rector'], alpha=0.85,
                       edgecolor='white', label='Validation')
    bars_test = ax1.bar(x + width / 2, test_vals, width,
                        color=COLORS['confidence'], alpha=0.85,
                        edgecolor='white', label='Test')

    for bars in [bars_val, bars_test]:
        for bar in bars:
            h = bar.get_height()
            fmt = f'{h:.2f}' if h < 10 else f'{h:.1f}%'
            ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                     fmt, ha='center', va='bottom', fontsize=7)

    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics, fontsize=8)
    ax1.set_ylabel('Value', fontsize=9)
    ax1.legend(fontsize=7, framealpha=0.9, loc='upper left')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_ylim(0, max(test_vals) * 1.15)
    ax1.set_title('(a) Aggregate Metrics', fontsize=10, fontweight='bold', pad=8)
    fig_a.subplots_adjust(left=0.15, right=0.97, top=0.88, bottom=0.18)
    save_fig(fig_a, 'test_set_generalization_a')

    # ── Panel (b): Percentile comparison ─────────────────────────────────
    fig_b, ax2 = plt.subplots(figsize=(COL_W, 2.4))
    pct_labels = list(val_pcts.keys())
    val_p = list(val_pcts.values())
    test_p = list(test_pcts.values())

    x2 = np.arange(len(pct_labels))
    ax2.plot(x2, val_p, 'o-', color=COLORS['rector'], linewidth=1.8,
             markersize=6, label='Validation', zorder=5)
    ax2.plot(x2, test_p, 's-', color=COLORS['confidence'], linewidth=1.8,
             markersize=6, label='Test', zorder=5)

    # Shade gap
    ax2.fill_between(x2, val_p, test_p, alpha=0.12, color=COLORS['confidence'])

    for i in range(len(pct_labels)):
        ax2.text(x2[i], val_p[i] - 0.5, f'{val_p[i]:.2f}',
                 ha='center', va='top', fontsize=7, color=COLORS['rector'])
        ax2.text(x2[i], test_p[i] + 0.3, f'{test_p[i]:.1f}',
                 ha='center', va='bottom', fontsize=7, color=COLORS['confidence'])

    ax2.set_xticks(x2)
    ax2.set_xticklabels(pct_labels, fontsize=8)
    ax2.set_xlabel('Percentile', fontsize=9)
    ax2.set_ylabel('minADE (m)', fontsize=9)
    ax2.legend(fontsize=7, framealpha=0.9)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_title('(b) Error Percentiles', fontsize=10, fontweight='bold', pad=8)
    fig_b.subplots_adjust(left=0.15, right=0.97, top=0.88, bottom=0.18)
    save_fig(fig_b, 'test_set_generalization_b')


# ═════════════════════════════════════════════════════════════════════════════
# 3. Selection Comparison (split into _a and _b)
# ═════════════════════════════════════════════════════════════════════════════
def fig_selection_comparison(canon):
    print("\n[3/4] Selection Comparison (split)")

    ss = canon['selection_strategies']
    strats = ['confidence', 'weighted_sum', 'lexicographic']
    strat_labels = {
        'confidence': 'Confidence',
        'weighted_sum': 'Weighted Sum',
        'lexicographic': 'Lexicographic\n(RECTOR)',
    }
    strat_colors = {
        'confidence': COLORS['confidence'],
        'weighted_sum': COLORS['weighted'],
        'lexicographic': COLORS['lexicographic'],
    }
    strat_markers = {'confidence': 'o', 'weighted_sum': 's', 'lexicographic': 'D'}

    # ── Panel (a): Pareto scatter ────────────────────────────────────────
    fig_a, ax1 = plt.subplots(figsize=(COL_W, 2.6))

    for s in strats:
        x = ss[s]['selADE_mean']
        y = ss[s]['SL_Viol_pct']
        ax1.plot(x, y, strat_markers[s], color=strat_colors[s],
                 markersize=10, markeredgecolor='white', markeredgewidth=0.8,
                 label=strat_labels[s], zorder=5)

    # Arrow from Confidence → RECTOR
    x_conf = ss['confidence']['selADE_mean']
    y_conf = ss['confidence']['SL_Viol_pct']
    x_lex = ss['lexicographic']['selADE_mean']
    y_lex = ss['lexicographic']['SL_Viol_pct']

    ax1.annotate('',
                 xy=(x_lex, y_lex), xytext=(x_conf, y_conf),
                 arrowprops=dict(arrowstyle='->', color='gray', lw=1.0,
                                 linestyle='--', connectionstyle='arc3,rad=0.2'))

    ax1.set_xlabel('selADE (m)', fontsize=9)
    ax1.set_ylabel('S+L Violation Rate (%)', fontsize=9)
    ax1.legend(fontsize=7, framealpha=0.9, loc='upper right')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(True, alpha=0.3, linestyle='--')

    # Dynamic limits
    all_x = [ss[s]['selADE_mean'] for s in strats]
    all_y = [ss[s]['SL_Viol_pct'] for s in strats]
    xm = max(0.15, (max(all_x) - min(all_x)) * 0.25)
    ym = max(1.0, (max(all_y) - min(all_y)) * 0.25)
    ax1.set_xlim(min(all_x) - xm, max(all_x) + xm)
    ax1.set_ylim(min(all_y) - ym, max(all_y) + ym)
    fig_a.subplots_adjust(left=0.16, right=0.97, top=0.95, bottom=0.16)
    save_fig(fig_a, 'selection_comparison_a')

    # ── Panel (b): Per-tier violation bars ────────────────────────────────
    fig_b, ax2 = plt.subplots(figsize=(COL_W, 2.6))
    tier_keys = ['Safety_Viol_pct', 'Legal_Viol_pct',
                 'Road_Viol_pct', 'Comfort_Viol_pct']
    tier_labels = ['Safety', 'Legal', 'Road', 'Comfort']

    x = np.arange(len(tier_labels))
    width = 0.24

    # Collect all bar data to detect overlapping labels
    all_bar_data = []
    for i, s in enumerate(strats):
        vals = [ss[s][tk] for tk in tier_keys]
        offset = (i - 1) * width
        bars = ax2.bar(x + offset, vals, width,
                       color=strat_colors[s], alpha=0.85,
                       label=strat_labels[s], edgecolor='white',
                       linewidth=0.5, zorder=3)
        all_bar_data.append((bars, vals, s))

    # Add value labels, skipping duplicates at the same tier
    for tier_idx in range(len(tier_labels)):
        seen_vals = {}
        for i, (bars, vals, s) in enumerate(all_bar_data):
            val = vals[tier_idx]
            if val < 0.3:
                continue
            val_str = f'{val:.1f}%'
            if val_str in seen_vals:
                continue  # skip duplicate label
            seen_vals[val_str] = True
            bar = bars[tier_idx]
            ax2.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.3,
                     val_str, ha='center', va='bottom',
                     fontsize=7, fontweight='bold')

    ax2.set_xticks(x)
    ax2.set_xticklabels(tier_labels, fontsize=8)
    ax2.set_xlabel('Rule Tier', fontsize=9)
    ax2.set_ylabel('Violation Rate (%)', fontsize=9)
    ax2.legend(fontsize=7, framealpha=0.9, loc='upper right')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
    max_val = max(ss[s][tk] for s in strats for tk in tier_keys)
    ax2.set_ylim(0, max_val * 1.2)

    # Priority annotation
    ax2.annotate(r'$\leftarrow$ Higher Priority', xy=(0.02, 0.95),
                 xycoords='axes fraction', fontsize=7, color='gray',
                 style='italic')

    fig_b.subplots_adjust(left=0.15, right=0.97, top=0.95, bottom=0.16)
    save_fig(fig_b, 'selection_comparison_b')


# ═════════════════════════════════════════════════════════════════════════════
# 4. Per-Rule Violation Heatmap (fixed tier labels — single panel)
# ═════════════════════════════════════════════════════════════════════════════
def fig_per_rule_heatmap(canon):
    print("\n[4/4] Per-Rule Applicability Heatmap (fixed tier labels)")

    prm = canon['per_rule_applicability_metrics']

    # Tier definitions in display order
    tiers = [
        ('Safety\n(Tier 0)', ['L0.R2', 'L0.R3', 'L0.R4', 'L10.R1', 'L10.R2'],
         COLORS['safety']),
        ('Legal\n(Tier 1)', ['L1.R1', 'L1.R2', 'L1.R3', 'L1.R4', 'L1.R5'],
         COLORS['legal']),
        ('Road\n(Tier 2)', ['L3.R3', 'L4.R3'],
         COLORS['road']),
        ('Comfort\n(Tier 3)', ['L5.R1', 'L5.R2', 'L5.R3', 'L5.R4', 'L5.R5',
                                'L6.R1', 'L6.R2', 'L6.R3', 'L6.R4', 'L6.R5',
                                'L7.R3', 'L7.R4', 'L8.R1', 'L8.R2', 'L8.R3', 'L8.R5'],
         COLORS['comfort']),
    ]

    # Collect data
    all_rules = []
    rule_tiers = []
    metric_keys = ['precision', 'recall', 'f1', 'accuracy']
    metric_labels = ['Precision', 'Recall', 'F1', 'Accuracy']

    for tier_name, rules, _ in tiers:
        for rule in rules:
            if rule in prm:
                all_rules.append(rule)
                rule_tiers.append(tier_name)

    n_rules = len(all_rules)
    matrix = np.zeros((n_rules, len(metric_keys)))
    for i, rule in enumerate(all_rules):
        for j, mk in enumerate(metric_keys):
            matrix[i, j] = prm[rule][mk]

    # Figure
    fig_h = max(4.0, 0.22 * n_rules + 1.0)
    fig, ax = plt.subplots(figsize=(COL_W + 0.3, fig_h))

    cmap = LinearSegmentedColormap.from_list(
        'perf', ['#C62828', '#FFCDD2', '#FFF9C4', '#A5D6A7', '#1B5E20'], N=256)

    im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(range(len(metric_labels)))
    ax.set_xticklabels(metric_labels, fontsize=8, fontweight='bold')
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')

    ax.set_yticks(range(n_rules))
    ax.set_yticklabels(all_rules, fontsize=7, family='monospace')

    # Cell annotations — use luminance for text color
    for i in range(n_rules):
        for j in range(len(metric_keys)):
            val = matrix[i, j]
            rgba = cmap(val)
            lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            color = 'white' if lum < 0.55 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=7, color=color, fontweight='bold')

    # Tier separators and labels — positioned OUTSIDE the plot area
    tier_boundaries = []
    current_idx = 0
    for tier_name, rules, tier_color in tiers:
        n = sum(1 for r in rules if r in prm)
        if n == 0:
            continue
        if current_idx > 0:
            ax.axhline(y=current_idx - 0.5, color='white', linewidth=2.5)
        tier_boundaries.append((current_idx, current_idx + n, tier_name, tier_color))
        current_idx += n

    # Draw colored tier bars on the left margin
    for start, end, tier_name, tier_color in tier_boundaries:
        mid = (start + end - 1) / 2
        # Colored rectangle bar
        rect = plt.Rectangle((-0.7, start - 0.5), 0.2, end - start,
                              transform=ax.get_yaxis_transform(),
                              color=tier_color, clip_on=False)
        ax.add_patch(rect)
        # Tier label further left
        ax.text(-0.85, mid, tier_name, transform=ax.get_yaxis_transform(),
                fontsize=7, fontweight='bold', color=tier_color,
                ha='right', va='center')

    cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.03, aspect=25)
    cbar.ax.tick_params(labelsize=7)

    fig.subplots_adjust(left=0.30, right=0.88, top=0.96, bottom=0.02)
    save_fig(fig, 'per_rule_violation_heatmap')


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("RECTOR Paper — Split Figure Regeneration")
    print(f"Output: {FIG_DIR}")
    print("=" * 60)

    canon = load_canonical()

    fig_protocol_comparison(canon)
    fig_test_generalization(canon)
    fig_selection_comparison(canon)
    fig_per_rule_heatmap(canon)

    print("\n" + "=" * 60)
    print("All split figures generated successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()
