#!/usr/bin/env python3
"""
Generate RECTOR architecture diagram — clean, legible, publication-ready.

Output: Figures/rector_architecture_diagram.{pdf,png}
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

BASE    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(BASE, "Figures")
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'sans-serif'],
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.01,
})

FW, FH = 10.5, 18.5
fig, ax = plt.subplots(figsize=(FW, FH))
ax.set_xlim(0, FW)
ax.set_ylim(0, FH)
ax.axis('off')

# ── colour palette ─────────────────────────────────────────────────────────────
C = {
    'input':   ('#E3F2FD', '#1565C0'),
    'enc':     ('#BBDEFB', '#1565C0'),
    'emb':     ('#3F51B5', '#1A237E'),
    'app':     ('#FFF8E1', '#E65100'),
    'tier':    ('#FFE0B2', '#BF360C'),
    'cvae':    ('#E8F5E9', '#2E7D32'),
    'cblock':  ('#C8E6C9', '#2E7D32'),
    'proxy':   ('#F3E5F5', '#6A1B9A'),
    'mask':    ('#EDE7F6', '#4527A0'),
    'scorer':  ('#FFFDE7', '#F57F17'),
    'out':     ('#DCEDC8', '#33691E'),
    'out2':    ('#E8F5E9', '#2E7D32'),
}

# ── helpers ────────────────────────────────────────────────────────────────────
def box(ax, x, y, w, h, title, subtitle='', face='#F5F5F5', edge='#455A64',
        tsz=10.5, ssz=9, bold=False, dashed=False):
    """Draw a rounded box with an optional two-line label."""
    ls = '--' if dashed else '-'
    r = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.08',
                       linewidth=1.5, linestyle=ls,
                       edgecolor=edge, facecolor=face, zorder=2)
    ax.add_patch(r)
    fw = 'bold' if bold else 'normal'
    ty = y + h * 0.65 if subtitle else y + h / 2
    ax.text(x + w/2, ty, title, ha='center', va='center',
            fontsize=tsz, fontweight=fw, color='#1A1A1A', zorder=3,
            clip_on=False)
    if subtitle:
        ax.text(x + w/2, y + h * 0.28, subtitle, ha='center', va='center',
                fontsize=ssz, color='#424242', zorder=3,
                linespacing=1.3, style='italic', clip_on=False)

def panel(ax, x, y, w, h, title, params, face, edge):
    """Outer container with a bold header title, param count, and a divider."""
    HDR = 0.68
    r = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.08',
                       linewidth=1.8, edgecolor=edge, facecolor=face, zorder=2)
    ax.add_patch(r)
    sep_y = y + h - HDR
    # title (upper part of header)
    ax.text(x + w/2, sep_y + HDR * 0.62,
            title, ha='center', va='center',
            fontsize=12, fontweight='bold', color='#1A1A1A', zorder=4)
    # params (lower part of header)
    ax.text(x + w/2, sep_y + HDR * 0.22,
            params, ha='center', va='center',
            fontsize=10, color='#424242', zorder=4, style='italic')
    # divider line
    ax.plot([x + 0.12, x + w - 0.12], [sep_y, sep_y],
            color=edge, lw=0.9, alpha=0.5, zorder=4)
    return sep_y

def arr(ax, x0, y0, x1, y1, label='', lw=1.5, dashed=False):
    ls = (0, (5, 3)) if dashed else 'solid'
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='->', color='#37474F',
                                lw=lw, linestyle=ls), zorder=5)
    if label:
        mx = (x0 + x1)/2 + 0.12
        my = (y0 + y1)/2
        ax.text(mx, my, label, fontsize=8.5, color='#263238',
                ha='left', va='center', zorder=6,
                bbox=dict(boxstyle='round,pad=0.18', fc='white',
                          ec='#90A4AE', lw=0.7, alpha=0.95))

def hconn(ax, x0, x1, y, lw=1.4):
    ax.plot([x0, x1], [y, y], color='#37474F', lw=lw, zorder=4)

def vconn(ax, x, y0, y1, lw=1.4):
    ax.plot([x, x], [y0, y1], color='#37474F', lw=lw, zorder=4)

def tip_down(ax, x, y, lw=1.5):
    ax.annotate('', xy=(x, y), xytext=(x, y + 0.20),
                arrowprops=dict(arrowstyle='->', color='#37474F', lw=lw), zorder=5)

# =============================================================================
# COORDINATES  (all computed once, top-to-bottom)
# =============================================================================
M   = 0.50
CX  = FW / 2
BW  = FW - 2 * M          # 9.50
GAP = 0.45
HW  = (BW - GAP) / 2      # 4.525
LX  = M                   # 0.50
RX  = LX + HW + GAP       # 5.475
LCX = LX + HW / 2         # 2.7625
RCX = RX + HW / 2         # 7.7375
IBW = HW - 0.28           # inner block width  4.245
LIBX = LX + 0.14          # inner block x (left column)
RIBX = RX + 0.14          # inner block x (right column)
IBH = 0.74                # inner block height — generous
IBG = 0.14                # gap between inner blocks

# ── TITLE ─────────────────────────────────────────────────────────────────────
ax.text(CX, 18.20, 'RECTOR  —  Rule-Enforced Constrained Trajectory Orchestrator',
        ha='center', va='center', fontsize=13.5, fontweight='bold', color='#0D1B4B')
ax.axhline(17.92, color='#B0BEC5', lw=1.0)

# ── INPUTS  y = 17.24…17.84 ──────────────────────────────────────────────────
IY, IH = 17.24, 0.60
IW = (BW - 0.50) / 3
for i, (t, s) in enumerate([
    ('Ego History',  '[B, 11, 4]'),
    ('Agent States', '[B, 32, 11, 4]'),
    ('Lane Centers', '[B, 64, 20, 2]'),
]):
    bx = LX + i * (IW + 0.25)
    box(ax, bx, IY, IW, IH, t, s, *C['input'], tsz=10.5, ssz=9.5)

arr(ax, CX, IY, CX, IY - 0.32)

# ── M2I SCENE ENCODER  y = 16.25…16.92 ───────────────────────────────────────
EY, EH = 16.08, 0.84
box(ax, LX, EY, BW, EH,
    'M2I Scene Encoder   (324 K params  ·  fine-tuned @ 0.1× LR)',
    'SubGraph → GlobalGraphRes → LaneGCN (A2L: CrossAttn  ·  L2L: GlobalGraphRes  ·  L2A: CrossAttn)\n→ Projection  [B, 128] → [B, 256]',
    *C['enc'], tsz=11, ssz=9.5, bold=True)

arr(ax, CX, EY, CX, EY - 0.32)

# ── SCENE EMBEDDING  y = 15.51…15.93 ─────────────────────────────────────────
SEY, SEH, SEW = 15.34, 0.42, 3.50
box(ax, CX - SEW/2, SEY, SEW, SEH, 'Scene Embedding   [B, 256]', '',
    *C['emb'], tsz=12, bold=True)
# white text on dark indigo
for t in ax.texts[-1:]:
    t.set_color('white')

# Fork: stem down → horizontal rail → two vertical drops with arrowheads
PY, PH = 10.00, 4.61          # parallel panel: y=10.00, top=14.61
PANEL_TOP = PY + PH            # 14.61
FORK_Y    = SEY - 0.38        # 14.96  (stem 0.38 in, drops 0.35 in each)

vconn(ax, CX, SEY, FORK_Y)
hconn(ax, LCX, RCX, FORK_Y)
vconn(ax, LCX, FORK_Y, PANEL_TOP + 0.01)
vconn(ax, RCX, FORK_Y, PANEL_TOP + 0.01)
tip_down(ax, LCX, PANEL_TOP)
tip_down(ax, RCX, PANEL_TOP)

# ── PARALLEL PANELS  y = 10.50…15.28 ─────────────────────────────────────────

# --- Left: Rule Applicability Head -------------------------------------------
sep_app = panel(ax, LX, PY, HW, PH,
                'Rule Applicability Head', '3.33 M params',
                *C['app'])

tier_rows = [
    ('Safety Tier   (5 rules)',
     'queries [5,256]  →  self-attn  →  cross-attn  →  sigmoid'),
    ('Legal Tier   (7 rules)',
     'queries [7,256]  →  self-attn  →  cross-attn  →  sigmoid'),
    ('Road Tier   (2 rules)',
     'queries [2,256]  →  self-attn  →  cross-attn  →  sigmoid'),
    ('Comfort Tier   (14 rules)',
     'queries [14,256]  →  self-attn  →  cross-attn  →  sigmoid'),
]
# Stack blocks downward from separator, evenly spaced with bottom margin
INNER_TOP = sep_app - 0.16
for i, (tt, ts) in enumerate(tier_rows):
    cy = INNER_TOP - (i + 1) * IBH - i * IBG
    box(ax, LIBX, cy, IBW, IBH, tt, ts, *C['tier'], tsz=10, ssz=8)

# --- Right: CVAE Trajectory Head ---------------------------------------------
sep_cv = panel(ax, RX, PY, HW, PH,
               'CVAE Trajectory Head', '5.18 M params',
               *C['cvae'])

cvae_rows = [
    ('Prior + Posterior Encoder',
     'p(z|scene)  /  q(z|scene,y) [train-only]  →  z [B, 64]  (×6 modes)'),
    ('Goal Head',
     'learnable queries  →  cross-attn  →  goals [B,6,2]'),
    ('Transformer Decoder   (4 layers · 8 heads)',
     'traj queries + PosEnc  →  cross-attend  →  Δ-pred'),
    ('Trajectory Refiner   (residual MLP)',
     'Δ + cumsum  →  [B,6,50,4]  +  confidences [B,6]'),
]
INNER_TOP_CV = sep_cv - 0.16
for i, (ct, cs) in enumerate(cvae_rows):
    cy = INNER_TOP_CV - (i + 1) * IBH - i * IBG
    box(ax, RIBX, cy, IBW, IBH, ct, cs, *C['cblock'], tsz=10, ssz=8)

# ── DIFFERENTIABLE RULE PROXIES  y = 8.50…9.60 ───────────────────────────────
# (defined before panel arrows so arrow endpoints can reference PRY+PRH)
PRY, PRH = 8.30, 1.10

# Arrows out of panels — land directly on proxy top (length ≈ 0.90, labels visible)
arr(ax, LCX, PY, LCX, PRY + PRH, label='app  [B, 28]')
arr(ax, RCX, PY, RCX, PRY + PRH, label='traj  [B,6,50,4]')

box(ax, LX, PRY, BW, PRH,
    'Differentiable Rule Proxies   (24 / 28 rules  ·  no learned parameters)',
    'Collision (SAT-OBB)  ·  Smoothness  ·  Lane-keeping  ·  Speed-limit\nTraffic-signal  ·  Interaction   →   violations [B, 6, 28]',
    *C['proxy'], tsz=11, ssz=9, bold=True)

arr(ax, CX, PRY, CX, PRY - 0.45)

# ── APPLICABILITY MASK  y = 7.61…8.05 ────────────────────────────────────────
MY, MH = 7.41, 0.44          # top = PRY - 0.45 = 7.85
box(ax, LX, MY, BW, MH,
    'Apply Applicability Mask   →   masked violations  [B, 6, 28]', '',
    *C['mask'], tsz=11)

arr(ax, CX, MY, CX, MY - 0.45)

# ── TIERED RULE SCORER  y = 6.32…7.16 ────────────────────────────────────────
SCY, SCH = 6.12, 0.84        # top = MY - 0.45 = 6.96
box(ax, LX, SCY, BW, SCH,
    'Tiered Rule Scorer   (140 fixed constants  ·  no learned parameters)',
    'Score = 10¹²·S_safety + 10⁹·S_legal + 10⁶·S_road + 10³·S_comfort    (B = 1000, lexicographic)',
    *C['scorer'], tsz=11, ssz=9.5, bold=True)

# Fork to two output boxes — long stems (≈ 0.40 in) so arrows are clearly visible
OY, OH   = 4.31, 0.86        # output boxes
OBJ_TOP  = OY + OH           # 5.17
FORK2_Y  = SCY - 0.55        # 5.57  (stem = 0.55 in)
vconn(ax, CX, SCY, FORK2_Y)
hconn(ax, LCX, RCX, FORK2_Y)
vconn(ax, LCX, FORK2_Y, OBJ_TOP + 0.01)
vconn(ax, RCX, FORK2_Y, OBJ_TOP + 0.01)
tip_down(ax, LCX, OBJ_TOP)
tip_down(ax, RCX, OBJ_TOP)

# ── OUTPUT BOXES  y = 4.51…5.37 ──────────────────────────────────────────────
box(ax, LX, OY, HW, OH,
    'Best Trajectory',
    '[B, 50, 4]  ·  rule-compliant lexicographic selection',
    *C['out'], tsz=11, ssz=9.5, bold=True)

box(ax, RX, OY, HW, OH,
    'All K Trajectories  +  Confidences',
    '[B, 6, 50, 4]  +  [B, 6]  ·  ensemble / reranking',
    *C['out2'], tsz=11, ssz=9.5, bold=True)

# ── LEGEND ────────────────────────────────────────────────────────────────────
ax.axhline(4.10, color='#B0BEC5', lw=0.9)
ax.text(CX, 3.90, 'Dashed border = train-only path (posterior)',
        ha='center', va='center', fontsize=9, color='#546E7A', style='italic')

legend_items = [
    (C['input'],  'Inputs'),
    (C['enc'],    'Scene Encoder'),
    (C['app'],    'App. Head'),
    (C['cvae'],   'CVAE Head'),
    (C['proxy'],  'Rule Proxies'),
    (C['mask'],   'App. Mask'),
    (C['scorer'], 'Tiered Scorer'),
    (C['out'],    'Output'),
]
n = len(legend_items)
lw_item = BW / n
for i, ((fc, ec), lbl) in enumerate(legend_items):
    lx = LX + i * lw_item
    r = FancyBboxPatch((lx + 0.04, 3.10), lw_item - 0.08, 0.40,
                       boxstyle='round,pad=0.05',
                       linewidth=1.0, edgecolor=ec, facecolor=fc, zorder=2)
    ax.add_patch(r)
    ax.text(lx + lw_item/2, 3.30, lbl,
            ha='center', va='center', fontsize=8,
            color='#1A1A1A', zorder=3)

# =============================================================================
# SAVE
# =============================================================================
for ext in ('pdf', 'png'):
    path = os.path.join(FIG_DIR, f'rector_architecture_diagram.{ext}')
    fig.savefig(path)
    print(f'  Saved: {path}')
plt.close(fig)
print('Done.')
