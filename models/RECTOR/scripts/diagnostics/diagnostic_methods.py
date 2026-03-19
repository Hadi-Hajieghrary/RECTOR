#!/usr/bin/env python3
"""
Diagnostic: Understand model output structure and test different
trajectory reconstruction methods.
"""
import os, sys, numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
sys.path.insert(0, "..")
sys.path.insert(0, "/workspace/data/WOMD")
sys.path.insert(0, "/workspace/data")
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d
from training.train_rector import WaymoDataset, collate_fn, TRAJECTORY_SCALE
from models.rule_aware_generator import RuleAwareGenerator
import torch, glob, random

DT = 0.1


def reconstruct_from_anchors(traj_m, n_anchors=8):
    """
    Take a few anchor points from the raw trajectory and fit a smooth
    cubic spline through them.  Preserves endpoints and overall direction
    while removing per-step noise.
    """
    T = len(traj_m)
    if T < 4:
        return traj_m.copy()

    # Pick anchor indices (including first and last)
    anchor_idx = np.unique(np.linspace(0, T - 1, n_anchors).astype(int))
    t_anchor = anchor_idx * DT
    t_full = np.arange(T) * DT

    out = traj_m.copy()
    sp_x = CubicSpline(t_anchor, traj_m[anchor_idx, 0], bc_type="clamped")
    sp_y = CubicSpline(t_anchor, traj_m[anchor_idx, 1], bc_type="clamped")
    out[:, 0] = sp_x(t_full)
    out[:, 1] = sp_y(t_full)

    # Recompute heading from tangent
    dx = np.gradient(out[:, 0])
    dy = np.gradient(out[:, 1])
    disp = np.sqrt(dx**2 + dy**2)
    mask = disp > 1e-3
    if mask.any():
        out[mask, 2] = np.arctan2(dy[mask], dx[mask])
        for t in range(1, T):
            if not mask[t]:
                out[t, 2] = out[t - 1, 2]

    # Recompute speed
    dx_dt = np.diff(out[:, 0]) / DT
    dy_dt = np.diff(out[:, 1]) / DT
    out[1:, 3] = np.sqrt(dx_dt**2 + dy_dt**2)
    out[0, 3] = out[1, 3]
    return out


def metrics(t):
    dx = t[-1, 0] - t[0, 0]
    dy = t[-1, 1] - t[0, 1]
    a = np.arctan2(dy, dx)
    c, s = np.cos(-a), np.sin(-a)
    ry = (t[:, 0] - t[0, 0]) * s + (t[:, 1] - t[0, 1]) * c
    lat_diff = np.diff(ry)
    sign_ch = int(np.sum(np.diff(np.sign(lat_diff)) != 0))
    fwd = np.sqrt(dx**2 + dy**2)
    return sign_ch, float(np.max(np.abs(ry))), fwd


random.seed(42)
device = torch.device("cpu")
model = RuleAwareGenerator(
    embed_dim=256,
    decoder_hidden_dim=256,
    decoder_num_layers=4,
    latent_dim=64,
    num_modes=6,
    use_m2i_encoder=True,
    m2i_checkpoint="/workspace/models/pretrained/m2i/models/relation_v2v/model.25.bin",
    freeze_m2i=True,
    trajectory_length=50,
).to(device)
ckpt = torch.load("/workspace/models/RECTOR/models/best.pt", map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

data_dir = "/workspace/data/WOMD/datasets/waymo_open_dataset/motion_v_1_3_0/processed/augmented/scenario/validation_interactive"
val_files = sorted(glob.glob(os.path.join(data_dir, "*")))
sample_files = random.sample(val_files, min(5, len(val_files)))
dataset = WaymoDataset(sample_files, is_training=False)
loader = torch.utils.data.DataLoader(
    dataset, batch_size=1, collate_fn=collate_fn, num_workers=0
)

header = f"{'#':>3} │ {'Method':>12} │ {'ZigZag':>6} │ {'MaxLat':>7} │ {'FwdDist':>8} │ {'Endpoint':>20}"
print(header)
print("─" * len(header))

count = 0
with torch.no_grad():
    for batch in loader:
        if not batch or count >= 8:
            break
        out = model(
            ego_history=batch["ego_history"],
            agent_states=batch["agent_states"],
            lane_centers=batch["lane_centers"],
        )
        trajs = out["trajectories"][0].cpu().numpy()
        gt = batch["traj_gt"][0].cpu().numpy() if "traj_gt" in batch else None
        conf = np.exp(out["confidence"][0].cpu().numpy())
        conf /= conf.sum()
        best = conf.argmax()

        raw = trajs[best].copy()
        raw[:, :2] *= TRAJECTORY_SCALE

        # Method 1: Gaussian smooth (current)
        gauss = raw.copy()
        gauss[:, 0] = gaussian_filter1d(raw[:, 0], sigma=3.0, mode="nearest")
        gauss[:, 1] = gaussian_filter1d(raw[:, 1], sigma=3.0, mode="nearest")

        # Method 2: Anchor-based cubic spline (8 anchors)
        anchor8 = reconstruct_from_anchors(raw, n_anchors=8)

        # Method 3: Anchor-based cubic spline (5 anchors)
        anchor5 = reconstruct_from_anchors(raw, n_anchors=5)

        for label, t in [
            ("Raw", raw),
            ("Gauss σ=3", gauss),
            ("Anchor-8", anchor8),
            ("Anchor-5", anchor5),
        ]:
            zz, ml, fwd = metrics(t)
            ep = f"({t[-1,0]:+.1f}, {t[-1,1]:+.1f})"
            print(
                f"{count:>3} │ {label:>12} │ {zz:>6} │ {ml:>7.2f} │ {fwd:>8.1f} │ {ep:>20}"
            )

        if gt is not None:
            gt_m = gt.copy()
            gt_m[:, :2] *= TRAJECTORY_SCALE
            zz, ml, fwd = metrics(gt_m)
            ep = f"({gt_m[-1,0]:+.1f}, {gt_m[-1,1]:+.1f})"
            print(
                f"{count:>3} │ {'GT':>12} │ {zz:>6} │ {ml:>7.2f} │ {fwd:>8.1f} │ {ep:>20}"
            )
        print()
        count += 1
