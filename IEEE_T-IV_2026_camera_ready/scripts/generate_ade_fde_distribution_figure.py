#!/usr/bin/env python3
"""
Generate the minADE / minFDE distribution figure (ade_fde_distribution.png).

Reads per-scenario metrics from the canonical evaluation CSV and produces
a two-panel histogram with mean and median annotations.

Verifies that computed means match the paper's reported values before saving.
"""

import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Paths ──────────────────────────────────────────────────────────────────────
CSV_PATH = Path('/workspace/output/evaluation/per_scenario_metrics.csv')
FIG_DIR = Path('/workspace/IEEE_T-IV_2026_extFigures')
OUT_PATH = FIG_DIR / 'ade_fde_distribution.png'

# ── Expected values (sanity check) ────────────────────────────────────────────
EXPECTED_MINADE_MEAN = 0.684
EXPECTED_MINFDE_MEAN = 1.270
TOLERANCE = 0.01  # allow 10mm tolerance

# ── Style ──────────────────────────────────────────────────────────────────────
COL_W = 3.5  # IEEE single-column width in inches
plt.rcParams.update({
    'font.size': 9,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7,
    'font.family': 'serif',
})

COLOR_ADE = '#1565C0'
COLOR_FDE = '#2E7D32'
COLOR_MEAN = '#B71C1C'
COLOR_MEDIAN = '#E65100'


def main():
    # ── Load data ──────────────────────────────────────────────────────────────
    if not CSV_PATH.exists():
        print(f"ERROR: {CSV_PATH} not found", file=sys.stderr)
        sys.exit(1)

    with open(CSV_PATH) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    min_ade = np.array([float(r['minADE']) for r in rows])
    min_fde = np.array([float(r['minFDE']) for r in rows])

    print(f"Loaded {len(min_ade)} scenarios from {CSV_PATH}")
    print(f"  minADE: mean={min_ade.mean():.4f}, median={np.median(min_ade):.4f}")
    print(f"  minFDE: mean={min_fde.mean():.4f}, median={np.median(min_fde):.4f}")

    # ── Sanity check ───────────────────────────────────────────────────────────
    if abs(min_ade.mean() - EXPECTED_MINADE_MEAN) > TOLERANCE:
        print(f"ERROR: minADE mean {min_ade.mean():.4f} != expected {EXPECTED_MINADE_MEAN} "
              f"(tolerance {TOLERANCE}). Aborting to prevent saving wrong figure.",
              file=sys.stderr)
        sys.exit(1)

    if abs(min_fde.mean() - EXPECTED_MINFDE_MEAN) > TOLERANCE:
        print(f"ERROR: minFDE mean {min_fde.mean():.4f} != expected {EXPECTED_MINFDE_MEAN} "
              f"(tolerance {TOLERANCE}). Aborting to prevent saving wrong figure.",
              file=sys.stderr)
        sys.exit(1)

    print("  Sanity check passed.")

    # ── Create figure ──────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 2.4))

    # Panel (a): minADE
    ax1.hist(min_ade, bins=80, color=COLOR_ADE, alpha=0.7,
             edgecolor='white', linewidth=0.3)
    ax1.axvline(min_ade.mean(), color=COLOR_MEAN, linestyle='--', linewidth=1.5,
                label=f'Mean: {min_ade.mean():.3f}m')
    ax1.axvline(np.median(min_ade), color=COLOR_MEDIAN, linestyle=':', linewidth=1.5,
                label=f'Median: {np.median(min_ade):.3f}m')
    ax1.set_xlabel('minADE (m)')
    ax1.set_ylabel('Count')
    ax1.set_title('(a) minADE Distribution', fontsize=10, fontweight='bold')
    ax1.legend(framealpha=0.9)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_xlim(0, np.percentile(min_ade, 99))

    # Panel (b): minFDE
    ax2.hist(min_fde, bins=80, color=COLOR_FDE, alpha=0.7,
             edgecolor='white', linewidth=0.3)
    ax2.axvline(min_fde.mean(), color=COLOR_MEAN, linestyle='--', linewidth=1.5,
                label=f'Mean: {min_fde.mean():.3f}m')
    ax2.axvline(np.median(min_fde), color=COLOR_MEDIAN, linestyle=':', linewidth=1.5,
                label=f'Median: {np.median(min_fde):.3f}m')
    ax2.set_xlabel('minFDE (m)')
    ax2.set_ylabel('Count')
    ax2.set_title('(b) minFDE Distribution', fontsize=10, fontweight='bold')
    ax2.legend(framealpha=0.9)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_xlim(0, np.percentile(min_fde, 99))

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"\nSaved: {OUT_PATH}")
    print("Done.")


if __name__ == '__main__':
    main()
