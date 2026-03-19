#!/usr/bin/env python3
"""Quick comparison of raw vs smoothed trajectory quality."""
import os, sys, numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
sys.path.insert(0, "..")
sys.path.insert(0, "/workspace/data/WOMD")
sys.path.insert(0, "/workspace/data")
from training.train_rector import WaymoDataset, collate_fn, TRAJECTORY_SCALE
from models.rule_aware_generator import RuleAwareGenerator
from visualization.generate_receding_movies import smooth_trajectory
import torch, glob, random

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
sample_files = random.sample(val_files, min(3, len(val_files)))
dataset = WaymoDataset(sample_files, is_training=False)
loader = torch.utils.data.DataLoader(
    dataset, batch_size=1, collate_fn=collate_fn, num_workers=0
)

hdr = f"{'#':>4}  {'Raw_ZZ':>6} {'Smo_ZZ':>6}  {'Raw_ML':>7} {'Smo_ML':>7}  {'Raw_RMS':>8} {'Smo_RMS':>8}"
print(hdr)
print("-" * len(hdr))


def metrics(t):
    dx = t[-1, 0] - t[0, 0]
    dy = t[-1, 1] - t[0, 1]
    a = np.arctan2(dy, dx)
    c, s = np.cos(-a), np.sin(-a)
    ry = (t[:, 0] - t[0, 0]) * s + (t[:, 1] - t[0, 1]) * c
    lat_diff = np.diff(ry)
    sign_ch = int(np.sum(np.diff(np.sign(lat_diff)) != 0))
    return (
        sign_ch,
        float(np.max(np.abs(ry))),
        float(np.sqrt(np.mean((np.abs(lat_diff) / 0.1) ** 2))),
    )


count = 0
with torch.no_grad():
    for batch in loader:
        if not batch or count >= 5:
            break
        out = model(
            ego_history=batch["ego_history"],
            agent_states=batch["agent_states"],
            lane_centers=batch["lane_centers"],
        )
        trajs = out["trajectories"][0].cpu().numpy()
        conf = np.exp(out["confidence"][0].cpu().numpy())
        conf /= conf.sum()
        best = conf.argmax()
        raw = trajs[best].copy()
        raw[:, :2] *= TRAJECTORY_SCALE
        smoothed = smooth_trajectory(raw.copy(), sigma=3.0)
        r_zz, r_ml, r_rms = metrics(raw)
        s_zz, s_ml, s_rms = metrics(smoothed)
        print(
            f"{count:>4}  {r_zz:>6} {s_zz:>6}  {r_ml:>7.2f} {s_ml:>7.2f}  {r_rms:>8.2f} {s_rms:>8.2f}"
        )
        count += 1

print("\nZZ=lateral sign changes, ML=max lateral(m), RMS=RMS lateral vel(m/s)")
