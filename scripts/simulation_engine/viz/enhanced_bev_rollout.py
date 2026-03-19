#!/usr/bin/env python3
"""
Enhanced BEV Closed-Loop Rollout Renderer
==========================================

Dual-panel layout:
  LEFT:  Bird's-eye view with road, agents, ego, all 6 candidate trajectories
         (thin colored), selected trajectory (thick), executed trail, logged trail
  RIGHT: Rule violation panel showing per-tier scores for each candidate,
         selected candidate highlighted, applicability mask, legend

Usage:
    cd /workspace/scripts
    XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.25 \
    python -m simulation_engine.viz.enhanced_bev_rollout \
        --target-movies 50 \
        --outdir /workspace/output/closedloop/videos
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.transforms as mtrans
import matplotlib.gridspec as gridspec
import numpy as np

from .bev_rollout import _extract_road_polylines, _ROAD_STYLE


# ── Candidate trajectory colors (6 modes) ──────────────────────────────
CAND_COLORS = [
    "#e41a1c",  # red
    "#377eb8",  # blue
    "#4daf4a",  # green
    "#984ea3",  # purple
    "#ff7f00",  # orange
    "#a65628",  # brown
]

TIER_NAMES = ["Safety", "Legal", "Road", "Comfort"]
TIER_COLORS = ["#d32f2f", "#1976d2", "#388e3c", "#f57c00"]

TRAJECTORY_SCALE = 50.0  # Must match observation_extractor


# ── Transform ego-local trajectories to world coords ────────────────────


def candidates_to_world(
    trajectories: np.ndarray,  # [K, T, 4] ego-local normalized
    ego_pos: np.ndarray,  # [2]
    ego_yaw: float,
) -> np.ndarray:
    """Convert K candidate trajectories from ego-local normalized to world XY.

    Returns [K, T, 2] world coordinates.
    """
    K, T, _ = trajectories.shape
    world = np.zeros((K, T, 2), dtype=np.float32)

    cos_r = np.cos(ego_yaw)
    sin_r = np.sin(ego_yaw)

    for k in range(K):
        for t in range(T):
            # Denormalize
            ex = trajectories[k, t, 0] * TRAJECTORY_SCALE
            ey = trajectories[k, t, 1] * TRAJECTORY_SCALE
            # Rotate back to global frame
            wx = ex * cos_r - ey * sin_r + ego_pos[0]
            wy = ex * sin_r + ey * cos_r + ego_pos[1]
            world[k, t, 0] = wx
            world[k, t, 1] = wy

    return world


# ── Diverse mock generator ──────────────────────────────────────────────


class DiverseMockGenerator:
    """
    Mock generator that creates visually distinct candidates
    with synthetic rule violations for demo purposes.

    Candidate 0: clean logged trajectory (low violations)
    Candidates 1-5: perturbed with lateral offsets + synthetic violations
    """

    def __init__(self, num_modes: int = 6, horizon: int = 50):
        self.num_modes = num_modes
        self.horizon = horizon

    def generate(self, obs, state):
        import jax.numpy as jnp

        sdc_idx = int(jnp.argmax(state.object_metadata.is_sdc))
        t = int(state.timestep)
        T = self.horizon
        K = self.num_modes

        ego_pos = obs["ego_pos"]
        ego_yaw = obs["ego_yaw"]
        cos_r = np.cos(-ego_yaw)
        sin_r = np.sin(-ego_yaw)

        log_x = np.asarray(state.log_trajectory.x[sdc_idx])
        log_y = np.asarray(state.log_trajectory.y[sdc_idx])
        log_vx = np.asarray(state.log_trajectory.vel_x[sdc_idx])
        log_vy = np.asarray(state.log_trajectory.vel_y[sdc_idx])
        total_t = len(log_x)

        # Base trajectory (ego-local normalized)
        traj = np.zeros((T, 4), dtype=np.float32)
        for i in range(T):
            ti = min(t + 1 + i, total_t - 1)
            dx = log_x[ti] - ego_pos[0]
            dy = log_y[ti] - ego_pos[1]
            ex = dx * cos_r - dy * sin_r
            ey = dx * sin_r + dy * cos_r
            traj[i, 0] = ex / TRAJECTORY_SCALE
            traj[i, 1] = ey / TRAJECTORY_SCALE
            traj[i, 2] = log_vx[ti] * cos_r - log_vy[ti] * sin_r
            traj[i, 3] = log_vx[ti] * sin_r + log_vy[ti] * cos_r

        # Create K diverse candidates
        trajectories = np.tile(traj[np.newaxis], (K, 1, 1))
        rng = np.random.RandomState(int(t) * 7 + 42)

        # Add structured perturbations for candidates 1-5
        for k in range(1, K):
            # Lateral offset that grows over time (lane change / swerve)
            lateral_offset = rng.uniform(-0.15, 0.15)  # in normalized units
            ramp = np.linspace(0, 1, T).astype(np.float32)
            trajectories[k, :, 1] += lateral_offset * ramp

            # Longitudinal offset (speed variation)
            speed_factor = rng.uniform(-0.05, 0.05)
            trajectories[k, :, 0] += speed_factor * ramp

            # Small random noise
            trajectories[k] += rng.randn(T, 4).astype(np.float32) * 0.005

        # Confidence: mode 0 is highest
        confidence = np.array([0.35, 0.20, 0.15, 0.12, 0.10, 0.08], dtype=np.float32)[
            :K
        ]
        confidence /= confidence.sum()

        # Synthetic rule violations: create realistic-looking patterns
        rule_costs = np.zeros((K, 28), dtype=np.float32)
        applicability = np.ones(28, dtype=np.float32)

        # Some rules are inapplicable
        inapplicable = rng.choice(28, size=rng.randint(5, 15), replace=False)
        applicability[inapplicable] = 0.0

        # Mode 0 (clean): very low violations
        rule_costs[0] = rng.uniform(0, 0.05, 28).astype(np.float32) * applicability

        for k in range(1, K):
            base = rng.uniform(0, 0.3, 28).astype(np.float32)
            # Some candidates have safety violations
            if rng.random() < 0.4:
                safety_rules = [0, 1, 2, 3, 4]
                for r in safety_rules[: rng.randint(1, 4)]:
                    base[r] = rng.uniform(0.5, 1.0)
            # Some have legal violations
            if rng.random() < 0.3:
                legal_rules = [5, 6, 7, 8, 9, 10, 11]
                for r in legal_rules[: rng.randint(1, 3)]:
                    base[r] = rng.uniform(0.3, 0.8)
            rule_costs[k] = base * applicability

        return {
            "trajectories": trajectories,
            "confidence": confidence,
            "rule_costs": rule_costs,
            "applicability": applicability,
        }


# ── Enhanced frame renderer ─────────────────────────────────────────────


def render_enhanced_frame(
    fig,
    ax_bev: plt.Axes,
    ax_panel: plt.Axes,
    ego_x: float,
    ego_y: float,
    ego_yaw: float,
    agent_positions: np.ndarray,
    agent_valid: np.ndarray,
    road_polylines: List[np.ndarray],
    road_types: List[int],
    logged_traj_xy: Optional[np.ndarray],
    exec_traj_xy: Optional[np.ndarray],
    candidate_world: Optional[np.ndarray],  # [K, T, 2] world coords
    selected_idx: int,
    confidence: Optional[np.ndarray],  # [K]
    rule_costs: Optional[np.ndarray],  # [K, 28]
    applicability: Optional[np.ndarray],  # [28]
    overlap: float,
    step_idx: int,
    total_steps: int,
    scenario_id: str,
    view_range: float = 60.0,
):
    """Render one enhanced BEV frame with candidate trajectories + rule panel."""

    # ── LEFT: BEV ────────────────────────────────────────────────────
    ax = ax_bev
    ax.clear()
    ax.set_facecolor("#f5f5f0")

    r = view_range
    ax.set_xlim(ego_x - r, ego_x + r)
    ax.set_ylim(ego_y - r, ego_y + r)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=5)
    ax.set_xlabel("x [m]", fontsize=6)
    ax.set_ylabel("y [m]", fontsize=6)

    # Road polylines
    xlo, xhi = ego_x - r - 20, ego_x + r + 20
    ylo, yhi = ego_y - r - 20, ego_y + r + 20
    for i, poly in enumerate(road_polylines):
        px, py = poly[:, 0], poly[:, 1]
        if px.max() < xlo or px.min() > xhi or py.max() < ylo or py.min() > yhi:
            continue
        rtype = road_types[i] if road_types else 0
        color, lw_scale, ls = _ROAD_STYLE.get(rtype, ("#cccccc", 1.0, "-"))
        ax.plot(
            px,
            py,
            color=color,
            lw=0.8 * lw_scale,
            ls=ls,
            zorder=1,
            solid_capstyle="round",
        )

    # Logged trajectory (full future as thin dashed)
    if logged_traj_xy is not None and len(logged_traj_xy) > 1:
        ax.plot(
            logged_traj_xy[:, 0],
            logged_traj_xy[:, 1],
            color="#17becf",
            lw=1.0,
            ls="--",
            alpha=0.4,
            zorder=2,
            label="Logged",
        )

    # Executed trajectory (trail behind ego)
    if exec_traj_xy is not None and len(exec_traj_xy) > 1:
        ax.plot(
            exec_traj_xy[:, 0],
            exec_traj_xy[:, 1],
            color="#2ca02c",
            lw=1.8,
            alpha=0.7,
            zorder=3,
            label="Executed",
        )

    # ── Candidate trajectories (AHEAD of ego) ──
    if candidate_world is not None:
        K = candidate_world.shape[0]
        # Draw non-selected first (thin, semi-transparent)
        for k in range(K):
            if k == selected_idx:
                continue
            color = CAND_COLORS[k % len(CAND_COLORS)]
            ax.plot(
                candidate_world[k, :, 0],
                candidate_world[k, :, 1],
                color=color,
                lw=0.8,
                alpha=0.4,
                zorder=5,
                label=f"Mode {k}" if step_idx <= 1 else None,
            )

        # Draw selected (thick, opaque, with arrow)
        sel = candidate_world[selected_idx]
        sel_color = CAND_COLORS[selected_idx % len(CAND_COLORS)]
        ax.plot(
            sel[:, 0],
            sel[:, 1],
            color=sel_color,
            lw=2.5,
            alpha=0.9,
            zorder=6,
            label=f"Selected (mode {selected_idx})",
        )
        # Arrow at end of selected trajectory
        if len(sel) > 2:
            dx = sel[-1, 0] - sel[-3, 0]
            dy = sel[-1, 1] - sel[-3, 1]
            ax.annotate(
                "",
                xy=(sel[-1, 0], sel[-1, 1]),
                xytext=(sel[-1, 0] - dx * 0.3, sel[-1, 1] - dy * 0.3),
                arrowprops=dict(arrowstyle="->", color=sel_color, lw=2),
                zorder=7,
            )

    # Other agents
    n_draw = min(int(agent_valid.sum()), 32)
    drawn = 0
    for ai in range(len(agent_valid)):
        if not agent_valid[ai]:
            continue
        if drawn >= n_draw:
            break
        ax_pos, ay_pos, a_yaw = agent_positions[ai]
        rect = plt.Rectangle(
            (-2.25, -0.95),
            4.5,
            1.9,
            facecolor="#1f77b4",
            edgecolor="black",
            linewidth=0.3,
            alpha=0.5,
        )
        t = mtrans.Affine2D().rotate(a_yaw).translate(ax_pos, ay_pos) + ax.transData
        rect.set_transform(t)
        ax.add_patch(rect)
        drawn += 1

    # Ego vehicle
    ego_color = "#d62728" if overlap > 0 else "#2ca02c"
    ego_rect = plt.Rectangle(
        (-2.4, -1.0),
        4.8,
        2.0,
        facecolor=ego_color,
        edgecolor="black",
        linewidth=0.8,
        alpha=0.85,
    )
    ego_t = mtrans.Affine2D().rotate(ego_yaw).translate(ego_x, ego_y) + ax.transData
    ego_rect.set_transform(ego_t)
    ax.add_patch(ego_rect)

    # Heading arrow
    hx = ego_x + 3.5 * np.cos(ego_yaw)
    hy = ego_y + 3.5 * np.sin(ego_yaw)
    ax.annotate(
        "",
        xy=(hx, hy),
        xytext=(ego_x, ego_y),
        arrowprops=dict(arrowstyle="->", color="white", lw=1.5),
        zorder=10,
    )

    # Title
    t_sec = step_idx * 0.1
    title = f"Step {step_idx}/{total_steps} (t={t_sec:.1f}s)"
    if overlap > 0:
        title += "  [OVERLAP]"
    ax.set_title(title, fontsize=8, fontweight="bold")
    ax.legend(loc="upper right", fontsize=5, framealpha=0.8, ncol=2)

    # ── RIGHT: Violation panel ───────────────────────────────────────
    ax_p = ax_panel
    ax_p.clear()
    ax_p.set_facecolor("white")

    if rule_costs is not None and confidence is not None:
        K = rule_costs.shape[0]

        # Compute per-tier aggregated violations for each candidate
        # Tier indices (matching rule_constants.py convention)
        tier_slices = {
            "Safety": slice(0, 5),
            "Legal": slice(5, 12),
            "Road": slice(12, 14),
            "Comfort": slice(14, 28),
        }

        tier_viols = np.zeros((K, 4), dtype=np.float32)
        for ti, (tname, tslice) in enumerate(tier_slices.items()):
            tier_viols[:, ti] = rule_costs[:, tslice].sum(axis=1)

        # Bar chart: grouped bars (K candidates × 4 tiers)
        x_tiers = np.arange(4)
        bar_width = 0.12
        offsets = np.arange(K) - (K - 1) / 2

        for k in range(K):
            color = CAND_COLORS[k % len(CAND_COLORS)]
            alpha = 0.9 if k == selected_idx else 0.35
            edgecolor = "black" if k == selected_idx else "none"
            lw = 1.5 if k == selected_idx else 0
            bars = ax_p.bar(
                x_tiers + offsets[k] * bar_width,
                tier_viols[k],
                width=bar_width * 0.85,
                color=color,
                alpha=alpha,
                edgecolor=edgecolor,
                linewidth=lw,
                label=f"M{k} ({confidence[k]:.0%})" if k == selected_idx else f"M{k}",
            )

        ax_p.set_xticks(x_tiers)
        ax_p.set_xticklabels(TIER_NAMES, fontsize=7, fontweight="bold")
        for i, label in enumerate(ax_p.get_xticklabels()):
            label.set_color(TIER_COLORS[i])
        ax_p.set_ylabel("Violation Score", fontsize=7)
        ax_p.set_title("Per-Tier Rule Violations", fontsize=8, fontweight="bold")
        ax_p.legend(fontsize=5, loc="upper right", ncol=2, framealpha=0.9)
        ax_p.set_ylim(0, max(tier_viols.max() * 1.3, 0.5))
        ax_p.spines["top"].set_visible(False)
        ax_p.spines["right"].set_visible(False)
        ax_p.tick_params(labelsize=6)

        # Applicability bar below
        if applicability is not None:
            # Small text showing applicable rule count
            n_app = int((applicability > 0.5).sum())
            ax_p.text(
                0.02,
                0.98,
                f"Active rules: {n_app}/28",
                transform=ax_p.transAxes,
                fontsize=5,
                va="top",
                bbox=dict(boxstyle="round,pad=0.2", fc="#e8e8e8", alpha=0.8),
            )

        # Scenario info
        info = f"Scenario: {scenario_id[:20]}\nSelected: Mode {selected_idx}"
        ax_p.text(
            0.02,
            0.02,
            info,
            transform=ax_p.transAxes,
            fontsize=5,
            va="bottom",
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.9),
        )


# ── Scenario mining for interesting ego behavior ─────────────────────────


def compute_ego_behavior_score(proto) -> Dict[str, float]:
    """
    Score a scenario by how interesting the ego trajectory is,
    and classify it into a maneuver category.

    Categories:
      - TURN:       sharp turns, U-turns (high heading change)
      - LANE_CHANGE: lateral shift with moderate heading (lane change / overtaking)
      - EXIT_RAMP:   moderate turn with path diverging from straight
      - COMPLEX:     stop-and-go, multi-phase maneuvers, speed changes
    """
    sdc_idx = proto.sdc_track_index
    track = proto.tracks[sdc_idx]

    xs, ys, headings, speeds = [], [], [], []
    for st in track.states:
        if st.valid:
            xs.append(st.center_x)
            ys.append(st.center_y)
            headings.append(st.heading)
            speeds.append(np.sqrt(st.velocity_x**2 + st.velocity_y**2))

    null_result = {
        "composite": 0,
        "category": "NONE",
        "heading_change": 0,
        "max_heading_change_deg": 0,
        "lateral_disp": 0,
        "speed_var": 0,
        "total_displacement": 0,
        "path_length": 0,
        "transitions": 0,
        "infra_score": 0,
        "crosswalks": 0,
        "stop_signs": 0,
        "agent_count": 0,
        "lat_reversals": 0,
        "net_heading_change_deg": 0,
    }

    if len(xs) < 20:
        return null_result

    xs = np.array(xs)
    ys = np.array(ys)
    headings = np.array(headings)
    speeds = np.array(speeds)

    # Cumulative path length
    dxy = np.sqrt(np.diff(xs) ** 2 + np.diff(ys) ** 2)
    path_length = float(dxy.sum())
    if path_length < 20.0:
        return null_result

    total_disp = float(np.sqrt((xs[-1] - xs[0]) ** 2 + (ys[-1] - ys[0]) ** 2))

    # Heading changes — only while moving (speed > 1 m/s)
    moving_mask = speeds[:-1] > 1.0
    dh = np.diff(headings)
    dh = np.arctan2(np.sin(dh), np.cos(dh))
    moving_dh = dh[moving_mask] if moving_mask.any() else dh
    total_heading_change = float(np.abs(moving_dh).sum())
    max_heading_change = float(np.abs(moving_dh).max()) if len(moving_dh) > 0 else 0.0
    # Net heading change (signed) — how much ego actually turned end-to-end
    net_heading_change = float(
        np.arctan2(
            np.sin(headings[-1] - headings[0]),
            np.cos(headings[-1] - headings[0]),
        )
    )

    # Lateral displacement from start→end line
    lateral_disp = 0.0
    lat_profile = np.zeros(len(xs))
    if len(xs) > 2:
        start = np.array([xs[0], ys[0]])
        end = np.array([xs[-1], ys[-1]])
        direction = end - start
        dir_len = np.linalg.norm(direction)
        if dir_len > 5.0:
            direction /= dir_len
            normal = np.array([-direction[1], direction[0]])
            pts = np.stack([xs, ys], axis=1) - start
            lat_profile = pts @ normal  # signed lateral distance
            lateral_disp = float(np.abs(lat_profile).max())

    # Lateral reversals — number of times lateral direction changes
    # (indicates lane change + return, or weaving / overtaking)
    lat_reversals = 0
    if len(lat_profile) > 10:
        # Smooth to avoid noise
        kernel = np.ones(5) / 5.0
        smooth_lat = np.convolve(lat_profile, kernel, mode="valid")
        if len(smooth_lat) > 2:
            lat_diff = np.diff(smooth_lat)
            sign_changes = np.diff(np.sign(lat_diff))
            lat_reversals = int(np.sum(np.abs(sign_changes) > 0.5))

    speed_var = float(np.std(speeds))

    # Stop-and-go
    moving = speeds > 1.0
    transitions = int(np.abs(np.diff(moving.astype(int))).sum())

    # Infrastructure
    crosswalks = sum(1 for mf in proto.map_features if mf.HasField("crosswalk"))
    stop_signs = sum(1 for mf in proto.map_features if mf.HasField("stop_sign"))
    tl_count = sum(len(dms.lane_states) for dms in proto.dynamic_map_states)
    tl_avg = tl_count / max(len(proto.dynamic_map_states), 1)
    lanes = sum(1 for mf in proto.map_features if mf.HasField("lane"))
    infra_score = crosswalks * 5 + stop_signs * 4 + tl_avg * 2 + min(lanes, 100) * 0.2

    agent_count = len(proto.tracks)

    # ── Maneuver classification ──
    abs_net_hdg = abs(net_heading_change)

    if total_heading_change > 1.5 and abs_net_hdg > 0.8:
        # Large heading change with net directional turn → TURN or U-turn
        category = "TURN"
        cat_score = (
            abs_net_hdg * 50.0
            + max_heading_change * 100.0
            + infra_score * 0.5
            + min(agent_count, 30) * 1.0
        )
    elif lateral_disp > 2.0 and total_heading_change < 1.5:
        # Lateral shift without much turning → LANE_CHANGE
        category = "LANE_CHANGE"
        cat_score = (
            lateral_disp * 30.0
            + lat_reversals * 20.0
            + speed_var * 5.0
            + min(agent_count, 30) * 2.0  # more agents = more interesting
            + min(path_length, 200) * 0.5
        )
    elif 0.3 < abs_net_hdg < 1.5 and lateral_disp > 1.0:
        # Moderate turn + lateral displacement → EXIT_RAMP or merge
        category = "EXIT_RAMP"
        cat_score = (
            abs_net_hdg * 80.0
            + lateral_disp * 10.0
            + speed_var * 5.0
            + infra_score * 1.0
            + min(path_length, 200) * 0.3
        )
    elif transitions >= 2 or (speed_var > 3.0 and lat_reversals >= 1):
        # Stop-and-go, speed changes, multi-phase → COMPLEX
        category = "COMPLEX"
        cat_score = (
            transitions * 15.0
            + speed_var * 8.0
            + lat_reversals * 15.0
            + lateral_disp * 5.0
            + total_heading_change * 10.0
            + min(agent_count, 30) * 1.5
        )
    else:
        # Fallback — score by general interestingness
        category = "OTHER"
        cat_score = (
            total_heading_change * 20.0
            + lateral_disp * 10.0
            + speed_var * 5.0
            + transitions * 8.0
            + infra_score * 0.5
        )

    return {
        "composite": round(cat_score, 1),
        "category": category,
        "heading_change": round(total_heading_change, 2),
        "max_heading_change_deg": round(np.degrees(max_heading_change), 1),
        "net_heading_change_deg": round(np.degrees(net_heading_change), 1),
        "lateral_disp": round(lateral_disp, 1),
        "lat_reversals": lat_reversals,
        "speed_var": round(speed_var, 1),
        "total_displacement": round(total_disp, 1),
        "path_length": round(path_length, 1),
        "transitions": transitions,
        "infra_score": round(infra_score, 1),
        "crosswalks": crosswalks,
        "stop_signs": stop_signs,
        "agent_count": agent_count,
    }


def mine_interesting_scenarios(
    data_root: str,
    split_dir: str,
    target_count: int = 50,
    shards_to_scan: int = 40,
    scenarios_per_shard: int = 40,
) -> List[Tuple[str, object, Dict]]:
    """
    Mine scenarios with DIVERSE ego behavior.

    Samples proportionally from maneuver categories:
      TURN        ~30%  (sharp turns, U-turns, intersections)
      LANE_CHANGE ~30%  (lane shifts, overtaking)
      EXIT_RAMP   ~20%  (merges, exits, moderate curves)
      COMPLEX     ~15%  (stop-and-go, multi-phase, speed variation)
      OTHER       ~5%   (fallback — interesting but uncategorized)
    """
    import tensorflow as tf
    from waymo_open_dataset.protos import scenario_pb2

    pattern = os.path.join(data_root, split_dir, "*.tfrecord*")
    shard_paths = sorted(glob.glob(pattern))
    if not shard_paths:
        raise FileNotFoundError(f"No TFRecords at {pattern}")

    print(f"  Found {len(shard_paths)} shards")
    n_shards = min(shards_to_scan, len(shard_paths))
    step = max(1, len(shard_paths) // n_shards)
    selected_shards = shard_paths[::step][:n_shards]
    print(f"  Scanning {len(selected_shards)} shards (step={step})")

    # Buckets by maneuver category
    buckets: Dict[str, List] = {
        "TURN": [],
        "LANE_CHANGE": [],
        "EXIT_RAMP": [],
        "COMPLEX": [],
        "OTHER": [],
    }
    total_scanned = 0

    for si, sp in enumerate(selected_shards):
        ds = tf.data.TFRecordDataset(sp)
        count = 0
        for raw_record in ds:
            if count >= scenarios_per_shard:
                break
            raw_bytes = raw_record.numpy()
            example = tf.train.Example()
            example.ParseFromString(raw_bytes)
            if "scenario/proto" not in example.features.feature:
                continue
            proto_bytes = example.features.feature["scenario/proto"].bytes_list.value[0]
            scenario = scenario_pb2.Scenario()
            scenario.ParseFromString(proto_bytes)

            scores = compute_ego_behavior_score(scenario)
            cat = scores.get("category", "OTHER")
            if cat == "NONE":
                count += 1
                continue
            if cat not in buckets:
                cat = "OTHER"
            buckets[cat].append((scenario.scenario_id, scenario, scores))
            count += 1
            total_scanned += 1

        if (si + 1) % 10 == 0:
            cat_counts = {k: len(v) for k, v in buckets.items()}
            print(
                f"    [{si+1}/{len(selected_shards)}] scanned {total_scanned}: {cat_counts}"
            )

    # Sort each bucket by category-specific score
    for cat in buckets:
        buckets[cat].sort(key=lambda x: x[2]["composite"], reverse=True)

    # Print category stats
    print(f"\n  Category distribution (from {total_scanned} valid scenarios):")
    for cat, items in buckets.items():
        print(f"    {cat:>12s}: {len(items)} scenarios")

    # Target allocation per category
    allocations = {
        "TURN": int(target_count * 0.30),
        "LANE_CHANGE": int(target_count * 0.30),
        "EXIT_RAMP": int(target_count * 0.20),
        "COMPLEX": int(target_count * 0.15),
        "OTHER": int(target_count * 0.05),
    }
    # Ensure we hit target_count
    allocated = sum(allocations.values())
    if allocated < target_count:
        allocations["LANE_CHANGE"] += target_count - allocated

    # Select from each bucket, overflow to other buckets if short
    selected = []
    overflow = 0
    for cat in ["TURN", "LANE_CHANGE", "EXIT_RAMP", "COMPLEX", "OTHER"]:
        want = allocations[cat] + overflow
        got = buckets[cat][:want]
        selected.extend(got)
        overflow = want - len(got)
        if overflow < 0:
            overflow = 0

    # If still short, grab more from largest bucket
    if len(selected) < target_count:
        seen_ids = {s[0] for s in selected}
        for cat in ["TURN", "LANE_CHANGE", "EXIT_RAMP", "COMPLEX", "OTHER"]:
            for item in buckets[cat]:
                if item[0] not in seen_ids:
                    selected.append(item)
                    seen_ids.add(item[0])
                    if len(selected) >= target_count:
                        break
            if len(selected) >= target_count:
                break

    # Print selections per category
    print(f"\n  Selected {len(selected)} scenarios:")
    cat_selected = {}
    for sid, _, sc in selected:
        c = sc["category"]
        cat_selected[c] = cat_selected.get(c, 0) + 1
    for cat, cnt in sorted(cat_selected.items()):
        print(f"    {cat:>12s}: {cnt}")

    # Print top examples per category
    for cat in ["TURN", "LANE_CHANGE", "EXIT_RAMP", "COMPLEX"]:
        cat_items = [(s, p, sc) for s, p, sc in selected if sc["category"] == cat]
        if not cat_items:
            continue
        print(f"\n  Top {cat} scenarios:")
        print(
            f"  {'ID':>20s}  {'Score':>7s}  {'NetHdg':>6s}  {'Lat':>5s}  "
            f"{'LatRev':>6s}  {'SpdVar':>6s}  {'Path':>5s}  {'Trans':>5s}  {'Agents':>6s}"
        )
        for sid, _, sc in cat_items[:5]:
            print(
                f"  {sid[:20]:>20s}  {sc['composite']:>7.1f}  "
                f"{sc['net_heading_change_deg']:>5.1f}°  "
                f"{sc['lateral_disp']:>5.1f}  "
                f"{sc['lat_reversals']:>6d}  "
                f"{sc['speed_var']:>6.1f}  "
                f"{sc.get('path_length',0):>5.0f}  "
                f"{sc.get('transitions',0):>5d}  "
                f"{sc['agent_count']:>6d}"
            )

    return selected


# ── Video stitching ──────────────────────────────────────────────────────


def stitch_video(frame_dir: str, outpath: str, fps: int = 10) -> bool:
    pattern = os.path.join(frame_dir, "frame_%04d.png")
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        pattern,
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "23",
        outpath,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=120)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


# ── Render one scenario ──────────────────────────────────────────────────


def render_scenario(
    scenario_id: str,
    proto,
    idx: int,
    max_steps: int,
    max_objects: int,
    max_rg_points: int,
    outdir: str,
    view_range: float = 60.0,
    dpi: int = 150,
    fps: int = 10,
) -> Dict:
    """Render one scenario with enhanced dual-panel BEV."""
    import jax.numpy as jnp
    from ..waymax_bridge.scenario_loader import scenario_proto_to_simulator_state
    from ..waymax_bridge.env_factory import make_env
    from ..waymax_bridge.observation_extractor import (
        extract_observation,
        ObservationExtractorConfig,
    )
    from ..waymax_bridge.action_converter import trajectory_to_action
    from ..waymax_bridge.metric_collector import MetricCollector, MetricCollectorConfig
    from ..selectors.confidence import ConfidenceSelector
    from ..config import WaymaxBridgeConfig

    t0 = time.time()

    sim_state = scenario_proto_to_simulator_state(
        proto, max_objects=max_objects, max_rg_points=max_rg_points
    )
    bridge_cfg = WaymaxBridgeConfig(dynamics_model="delta", steps_per_replan=20)
    env, state = make_env(sim_state, dynamics_model=bridge_cfg.dynamics_model)

    obs_cfg = ObservationExtractorConfig()
    mc = MetricCollector(
        MetricCollectorConfig(waymax_metrics=("log_divergence", "overlap"))
    )
    mc.reset()
    generator = DiverseMockGenerator(num_modes=6, horizon=50)
    selector = ConfidenceSelector()

    road_polylines, road_types = _extract_road_polylines(state)
    sdc_idx = int(jnp.argmax(state.object_metadata.is_sdc))
    log_x = np.asarray(state.log_trajectory.x[sdc_idx])
    log_y = np.asarray(state.log_trajectory.y[sdc_idx])
    logged_xy = np.stack([log_x, log_y], axis=-1)

    frame_dir = os.path.join(outdir, f"frames_{idx:03d}")
    os.makedirs(frame_dir, exist_ok=True)

    exec_traj = []
    steps_per_replan = bridge_cfg.steps_per_replan
    total_steps = min(max_steps, 91 - int(state.timestep) - 1)

    selected_traj = None
    obs = None
    step_in_plan = 0
    overlap_events = 0

    # Current candidate data for rendering
    cur_trajectories = None
    cur_confidence = None
    cur_rule_costs = None
    cur_applicability = None
    cur_sel_idx = 0
    cur_candidate_world = None

    # Figure with dual panel: BEV (left, larger) + violations (right)
    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1], wspace=0.08)
    ax_bev = fig.add_subplot(gs[0])
    ax_panel = fig.add_subplot(gs[1])

    actual_steps = 0
    for step_i in range(total_steps):
        # Replan
        if step_i % steps_per_replan == 0 or selected_traj is None:
            obs = extract_observation(state, obs_cfg)
            gen_out = generator.generate(obs, state)
            cur_trajectories = gen_out["trajectories"]  # [K, T, 4]
            cur_confidence = gen_out["confidence"]  # [K]
            cur_rule_costs = gen_out["rule_costs"]  # [K, 28]
            cur_applicability = gen_out["applicability"]  # [28]

            K = cur_rule_costs.shape[0]
            tier_scores = np.zeros((K, 4), dtype=np.float32)
            cur_sel_idx, trace = selector.select(
                candidates=cur_trajectories,
                probs=cur_confidence,
                tier_scores=tier_scores,
                rule_scores=cur_rule_costs,
                rule_applicability=cur_applicability,
            )
            selected_traj = cur_trajectories[cur_sel_idx]
            step_in_plan = 0

            # Convert candidates to world coords for plotting
            cur_candidate_world = candidates_to_world(
                cur_trajectories, obs["ego_pos"], obs["ego_yaw"]
            )

        # Step
        action = trajectory_to_action(
            trajectory=selected_traj,
            ego_pos=obs["ego_pos"],
            ego_yaw=obs["ego_yaw"],
            state=state,
            steps=step_in_plan + 1,
        )
        state = env.step(state, action)
        step_in_plan += 1
        mc.step(state)

        t_now = int(state.timestep)
        ego_x = float(np.asarray(state.sim_trajectory.x[sdc_idx, t_now]))
        ego_y = float(np.asarray(state.sim_trajectory.y[sdc_idx, t_now]))
        ego_yaw = float(np.asarray(state.sim_trajectory.yaw[sdc_idx, t_now]))
        exec_traj.append([ego_x, ego_y])

        n_obj = state.sim_trajectory.x.shape[0]
        agent_pos = np.zeros((n_obj, 3), dtype=np.float32)
        agent_valid = np.zeros(n_obj, dtype=bool)
        for ai in range(n_obj):
            if ai == sdc_idx:
                continue
            v = bool(np.asarray(state.sim_trajectory.valid[ai, t_now]))
            if v:
                agent_pos[ai] = [
                    float(np.asarray(state.sim_trajectory.x[ai, t_now])),
                    float(np.asarray(state.sim_trajectory.y[ai, t_now])),
                    float(np.asarray(state.sim_trajectory.yaw[ai, t_now])),
                ]
                agent_valid[ai] = True

        overlap_val = 0.0
        try:
            from waymax import metrics as wm_metrics

            ov = wm_metrics.overlap(state)
            overlap_val = float(np.asarray(ov.value[sdc_idx]))
        except Exception:
            pass
        if overlap_val > 0:
            overlap_events += 1

        # Render
        render_enhanced_frame(
            fig=fig,
            ax_bev=ax_bev,
            ax_panel=ax_panel,
            ego_x=ego_x,
            ego_y=ego_y,
            ego_yaw=ego_yaw,
            agent_positions=agent_pos,
            agent_valid=agent_valid,
            road_polylines=road_polylines,
            road_types=road_types,
            logged_traj_xy=logged_xy,
            exec_traj_xy=np.array(exec_traj) if exec_traj else None,
            candidate_world=cur_candidate_world,
            selected_idx=cur_sel_idx,
            confidence=cur_confidence,
            rule_costs=cur_rule_costs,
            applicability=cur_applicability,
            overlap=overlap_val,
            step_idx=step_i + 1,
            total_steps=total_steps,
            scenario_id=scenario_id,
            view_range=view_range,
        )

        fig.savefig(os.path.join(frame_dir, f"frame_{step_i:04d}.png"), dpi=dpi)
        actual_steps = step_i + 1

        if bool(state.is_done):
            break

    plt.close(fig)

    try:
        metrics = mc.finalise()
    except Exception:
        metrics = {}

    elapsed = time.time() - t0
    safe_id = scenario_id.replace("/", "_")[:32]
    video_name = f"scenario_{idx:03d}_{safe_id}.mp4"
    video_path = os.path.join(outdir, video_name)
    ok = stitch_video(frame_dir, video_path, fps=fps)
    if ok:
        shutil.rmtree(frame_dir, ignore_errors=True)

    return {
        "scenario_idx": idx,
        "scenario_id": scenario_id,
        "steps": actual_steps,
        "overlaps": overlap_events,
        "video": video_name if ok else None,
        "elapsed": round(elapsed, 1),
        "road_segments": len(road_polylines),
    }


# ── Main ────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Enhanced BEV movies with candidate trajectories + violation panel"
    )
    parser.add_argument("--target-movies", type=int, default=50)
    parser.add_argument("--outdir", default="/workspace/output/closedloop/videos")
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--max-objects", type=int, default=64)
    parser.add_argument("--max-rg-points", type=int, default=20000)
    parser.add_argument("--shards-to-scan", type=int, default=40)
    parser.add_argument("--scenarios-per-shard", type=int, default=40)
    parser.add_argument("--view-range", type=float, default=60.0)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    data_root = (
        "/workspace/data/WOMD/datasets/waymo_open_dataset"
        "/motion_v_1_3_0/processed/augmented/scenario"
    )

    print("=" * 70)
    print("Enhanced BEV Movie Generator")
    print("  - 6 candidate trajectories shown ahead of ego")
    print("  - Selected trajectory highlighted (thick + arrow)")
    print("  - Right panel: per-tier rule violation scores")
    print("  - Mined for turns, lane changes, stops")
    print("=" * 70)
    print(f"  Target: {args.target_movies} movies")
    print(f"  Output: {args.outdir}")
    print()

    # Phase 1: Mine interesting scenarios
    print("Phase 1: Mining scenarios with interesting ego behavior...")
    t_mine = time.time()
    candidates = mine_interesting_scenarios(
        data_root,
        "validation_interactive",
        target_count=args.target_movies,
        shards_to_scan=args.shards_to_scan,
        scenarios_per_shard=args.scenarios_per_shard,
    )
    print(f"Mining done: {len(candidates)} scenarios in {time.time()-t_mine:.1f}s\n")

    # Phase 2: Render
    print(f"Phase 2: Rendering {len(candidates)} enhanced BEV movies...")
    results = []
    success = 0
    t_render = time.time()

    for i, (sid, proto, scores) in enumerate(candidates):
        sc = scores["composite"]
        cat = scores.get("category", "?")
        lat = scores["lateral_disp"]
        net_hdg = scores.get("net_heading_change_deg", 0)
        lr = scores.get("lat_reversals", 0)
        print(
            f"\n[{i+1}/{len(candidates)}] {sid}  "
            f"({cat}, score={sc:.0f}, netHdg={net_hdg:.0f}°, lat={lat:.1f}m, latRev={lr})"
        )

        try:
            r = render_scenario(
                scenario_id=sid,
                proto=proto,
                idx=i,
                max_steps=args.max_steps,
                max_objects=args.max_objects,
                max_rg_points=args.max_rg_points,
                outdir=args.outdir,
                view_range=args.view_range,
                dpi=args.dpi,
                fps=args.fps,
            )
            r["behavior_scores"] = scores
            results.append(r)

            if r["video"]:
                success += 1
                sz = os.path.getsize(os.path.join(args.outdir, r["video"])) / (
                    1024 * 1024
                )
                print(
                    f"    OK: {r['video']} ({r['steps']} steps, {r['elapsed']:.0f}s, {sz:.1f}MB)"
                )
            else:
                print("    WARN: video stitching failed")

            done = i + 1
            remaining = len(candidates) - done
            if remaining > 0:
                eta = (time.time() - t_render) / done * remaining
                print(f"    ETA: {eta:.0f}s ({eta/60:.1f}min)")

        except Exception as e:
            results.append({"scenario_idx": i, "scenario_id": sid, "error": str(e)})
            print(f"    FAIL: {e}")
            import traceback

            traceback.print_exc()

    total = time.time() - t_render
    print("\n" + "=" * 70)
    print(
        f"DONE: {success}/{len(candidates)} videos in {total:.0f}s ({total/60:.1f}min)"
    )
    print(f"Output: {args.outdir}")
    print("=" * 70)

    # Save summary
    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "totals": {
            "videos": success,
            "failed": len(candidates) - success,
            "time_seconds": round(total, 1),
        },
        "results": results,
    }
    with open(os.path.join(args.outdir, "enhanced_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)


if __name__ == "__main__":
    main()
