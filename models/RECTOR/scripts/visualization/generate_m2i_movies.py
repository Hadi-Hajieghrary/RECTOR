#!/usr/bin/env python3
"""
M2I DenseTNT + RECTOR Rule Scoring — Simulation Movie Generator.

Generates BEV (bird's-eye-view) videos showing:
  - Map (lanes, road edges, crosswalks)
  - Ego vehicle with GT history trail
  - M2I DenseTNT 6-mode trajectory predictions (8-second horizon)
  - Rule-selected best trajectory (highlighted)
  - GT future trajectory (dashed)
  - Other agents (vehicles, pedestrians, cyclists)
  - Rule violation panel on the right

Operates in "receding horizon" style: at each timestep t, re-runs M2I
for a fresh 8s prediction, and the rule scorer picks the best mode.

Usage:
    python visualization/generate_m2i_movies.py --num_scenarios 5
    python visualization/generate_m2i_movies.py --num_scenarios 3 --seed 42
    python visualization/generate_m2i_movies.py --data_dir /path/to/data --output_dir /path/to/movies
"""

import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import argparse
import glob
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec

import tensorflow as tf

tf.config.set_visible_devices([], "GPU")

# Paths
SCRIPT_DIR = Path(__file__).parent
RECTOR_SCRIPTS = SCRIPT_DIR.parent
sys.path.insert(0, str(RECTOR_SCRIPTS))
sys.path.insert(0, str(RECTOR_SCRIPTS / "lib"))
sys.path.insert(0, "/workspace/externals/M2I/src")
sys.path.insert(0, "/workspace/data/WOMD")

from waymo_open_dataset.protos import scenario_pb2

# RECTOR rule constants
try:
    from waymo_rule_eval.rules.rule_constants import (
        RULE_IDS,
        NUM_RULES,
        RULE_INDEX_MAP,
        TIER_0_SAFETY,
        TIER_1_LEGAL,
        TIER_2_ROAD,
        TIER_3_COMFORT,
        TIERS,
        TIER_BY_NAME,
        get_tier_mask,
    )

    RULES_AVAILABLE = True
except ImportError:
    RULES_AVAILABLE = False
    RULE_IDS = []
    NUM_RULES = 28
    RULE_INDEX_MAP = {}
    TIER_0_SAFETY = TIER_1_LEGAL = TIER_2_ROAD = TIER_3_COMFORT = []
    TIERS = ["safety", "legal", "road", "comfort"]

from lib.m2i_trajectory_generator import M2ITrajectoryGenerator
from lib.m2i_rule_selector import KinematicRuleEvaluator

TOTAL_TIMESTEPS = 91
DT = 0.1
HISTORY_LENGTH = 11
M2I_FUTURE_LENGTH = 80  # 8 seconds @ 10Hz

COLORS = {
    "ego": "#e74c3c",
    "ego_history": "#c0392b",
    "gt_future": "#27ae60",
    "planned_best": "#3498db",
    "other_modes": "#f39c12",
    "agents_vehicle": "#9b59b6",
    "agents_ped": "#2ecc71",
    "agents_cyclist": "#e67e22",
    "lane": "#95a5a6",
    "road_edge": "#2c3e50",
    "crosswalk": "#f1c40f",
}

TIER_COLORS = {
    "safety": "#e74c3c",
    "legal": "#f39c12",
    "road": "#3498db",
    "comfort": "#2ecc71",
}

RULE_SHORT_NAMES = {
    "L0.R2": "Long. Distance",
    "L0.R3": "Lat. Clearance",
    "L0.R4": "Crosswalk Occ.",
    "L10.R1": "Veh. Collision",
    "L10.R2": "VRU Clearance",
    "L5.R1": "Traffic Signal",
    "L5.R2": "Priority",
    "L7.R4": "Speed Limit",
    "L8.R1": "Red Light",
    "L8.R2": "Stop Sign",
    "L8.R3": "Crosswalk Yield",
    "L8.R5": "Wrong Way",
    "L3.R3": "Drivable Surf.",
    "L7.R3": "Lane Departure",
    "L1.R1": "Accel Comfort",
    "L1.R2": "Brake Comfort",
    "L1.R3": "Steer Comfort",
    "L1.R4": "Speed Consist.",
    "L1.R5": "Lane Change",
    "L4.R3": "Left Turn Gap",
    "L5.R3": "Parking",
    "L5.R4": "School Zone",
    "L5.R5": "Construction",
    "L6.R1": "Coop. Lane Chg",
    "L6.R2": "Following Dist",
    "L6.R3": "Intersection",
    "L6.R4": "Ped. Interact",
    "L6.R5": "Cyclist Inter.",
}


def load_scenarios_from_tfrecords(
    data_dir: str,
    max_files: int = 20,
    seed: Optional[int] = None,
) -> List[Dict]:
    """Load scenarios from augmented TFRecords."""
    tfrecord_files = sorted(glob.glob(os.path.join(data_dir, "*.tfrecord*")))
    if not tfrecord_files:
        print(f"ERROR: No TFRecord files in {data_dir}")
        return []

    if seed is not None:
        random.seed(seed)
    sample_files = random.sample(tfrecord_files, min(max_files, len(tfrecord_files)))

    scenarios = []
    for fpath in sample_files:
        try:
            dataset = tf.data.TFRecordDataset(fpath)
            for record in dataset:
                raw = record.numpy()
                scenario = _parse_scenario(raw)
                if scenario is not None:
                    scenario_data = _extract_scenario_data(scenario)
                    if scenario_data is not None:
                        scenarios.append(scenario_data)
        except Exception as e:
            print(f"  Warning: Failed {fpath}: {e}")
    return scenarios


def _parse_scenario(raw_bytes):
    """Parse raw TFRecord bytes into Scenario proto."""
    try:
        example = tf.train.Example.FromString(raw_bytes)
        features = example.features.feature
        if "scenario/proto" in features:
            proto_bytes = features["scenario/proto"].bytes_list.value[0]
            scenario = scenario_pb2.Scenario()
            scenario.ParseFromString(proto_bytes)
            return scenario
    except Exception:
        pass
    try:
        scenario = scenario_pb2.Scenario()
        scenario.ParseFromString(raw_bytes)
        if scenario.scenario_id:
            return scenario
    except Exception:
        pass
    return None


def _extract_scenario_data(scenario) -> Optional[Dict]:
    """Extract all data needed for visualization from a Scenario proto."""
    tracks = list(scenario.tracks)
    if not tracks or len(tracks) <= scenario.sdc_track_index:
        return None

    sdc_idx = scenario.sdc_track_index
    sdc_track = tracks[sdc_idx]
    current_ts = scenario.current_time_index

    if len(sdc_track.states) < TOTAL_TIMESTEPS:
        return None

    ref_state = sdc_track.states[current_ts]
    if not ref_state.valid:
        return None

    ref_x, ref_y = ref_state.center_x, ref_state.center_y
    ref_h = ref_state.heading
    cos_h, sin_h = math.cos(-ref_h), math.sin(-ref_h)

    def to_local(wx, wy):
        dx, dy = wx - ref_x, wy - ref_y
        return dx * cos_h - dy * sin_h, dx * sin_h + dy * cos_h

    # Ego trajectory (all 91 timesteps in local frame)
    ego_traj = np.zeros((TOTAL_TIMESTEPS, 5), dtype=np.float32)
    for t in range(min(TOTAL_TIMESTEPS, len(sdc_track.states))):
        s = sdc_track.states[t]
        if s.valid:
            x, y = to_local(s.center_x, s.center_y)
            ego_traj[t, 0] = x
            ego_traj[t, 1] = y
            ego_traj[t, 2] = s.heading - ref_h
            ego_traj[t, 3] = math.sqrt(s.velocity_x**2 + s.velocity_y**2)
            ego_traj[t, 4] = 1.0

    # Other agents
    max_agents = 48
    agent_trajs = np.zeros((max_agents, TOTAL_TIMESTEPS, 6), dtype=np.float32)
    agent_count = 0
    for i, track in enumerate(tracks):
        if i == sdc_idx or agent_count >= max_agents:
            continue
        has_valid = any(
            track.states[t].valid for t in range(min(HISTORY_LENGTH, len(track.states)))
        )
        if not has_valid:
            continue
        for t in range(min(TOTAL_TIMESTEPS, len(track.states))):
            s = track.states[t]
            if s.valid:
                x, y = to_local(s.center_x, s.center_y)
                agent_trajs[agent_count, t, 0] = x
                agent_trajs[agent_count, t, 1] = y
                agent_trajs[agent_count, t, 2] = s.heading - ref_h
                agent_trajs[agent_count, t, 3] = math.sqrt(
                    s.velocity_x**2 + s.velocity_y**2
                )
                agent_trajs[agent_count, t, 4] = 1.0
                agent_trajs[agent_count, t, 5] = track.object_type
        agent_count += 1

    # Map features
    lanes, road_edges, crosswalks = [], [], []
    for mf in scenario.map_features:
        if mf.HasField("lane"):
            pts = [(p.x, p.y) for p in mf.lane.polyline]
            if pts:
                lanes.append(np.array([to_local(p[0], p[1]) for p in pts]))
        elif mf.HasField("road_edge"):
            pts = [(p.x, p.y) for p in mf.road_edge.polyline]
            if pts:
                road_edges.append(np.array([to_local(p[0], p[1]) for p in pts]))
        elif mf.HasField("crosswalk"):
            pts = [(p.x, p.y) for p in mf.crosswalk.polygon]
            if len(pts) >= 3:
                crosswalks.append(np.array([to_local(p[0], p[1]) for p in pts]))

    # Ego speed stats
    valid_mask = ego_traj[:, 4] > 0
    if valid_mask.sum() > 1:
        pos = ego_traj[valid_mask, :2]
        total_dist = np.sqrt(np.sum(np.diff(pos, axis=0) ** 2, axis=1)).sum()
        duration = (valid_mask.sum() - 1) * DT
        avg_speed = total_dist / max(duration, 1e-6)
    else:
        total_dist, avg_speed = 0.0, 0.0

    return {
        "scenario_proto": scenario,
        "scenario_id": scenario.scenario_id,
        "ego_traj": ego_traj,
        "agent_trajs": agent_trajs,
        "agent_count": agent_count,
        "lanes": lanes,
        "road_edges": road_edges,
        "crosswalks": crosswalks,
        "ref_x": ref_x,
        "ref_y": ref_y,
        "ref_h": ref_h,
        "ego_total_dist": total_dist,
        "ego_avg_speed": avg_speed,
    }


def draw_vehicle(
    ax,
    x,
    y,
    heading,
    length=4.5,
    width=2.0,
    color="red",
    alpha=1.0,
    zorder=10,
    label=None,
):
    corners = np.array(
        [
            [-length / 2, -width / 2],
            [length / 2, -width / 2],
            [length / 2, width / 2],
            [-length / 2, width / 2],
        ]
    )
    c, s = np.cos(heading), np.sin(heading)
    rot = np.array([[c, -s], [s, c]])
    corners = corners @ rot.T + np.array([x, y])
    poly = plt.Polygon(
        corners,
        facecolor=color,
        edgecolor="black",
        linewidth=1,
        alpha=alpha,
        zorder=zorder,
        label=label,
    )
    ax.add_patch(poly)
    arrow_len = length * 0.4
    ax.arrow(
        x,
        y,
        arrow_len * np.cos(heading),
        arrow_len * np.sin(heading),
        head_width=0.4,
        head_length=0.2,
        fc="white",
        ec="black",
        zorder=zorder + 1,
        alpha=alpha,
    )


def draw_rule_panel(
    ax, violations, tier_scores, composite_scores, best_mode_idx, current_t
):
    """Draw rule violation info panel."""
    ax.clear()
    ax.axis("off")

    time_sec = (current_t - 10) * DT
    ax.set_title(
        f"M2I + RECTOR Rules (t={time_sec:+.1f}s)\nBest mode: {best_mode_idx}",
        fontsize=10,
        fontweight="bold",
        loc="left",
    )

    y_pos = 0
    tier_rules_map = {
        "safety": TIER_0_SAFETY,
        "legal": TIER_1_LEGAL,
        "road": TIER_2_ROAD,
        "comfort": TIER_3_COMFORT,
    }
    tier_labels = {
        "safety": "SAFETY (Tier 0)",
        "legal": "LEGAL (Tier 1)",
        "road": "ROAD (Tier 2)",
        "comfort": "COMFORT (Tier 3)",
    }

    for tier_name in TIERS:
        rules = tier_rules_map.get(tier_name, [])
        if not rules:
            continue
        color = TIER_COLORS.get(tier_name, "gray")
        score = tier_scores.get(tier_name, np.zeros(6))
        best_score = score[best_mode_idx] if len(score) > best_mode_idx else 0

        ax.text(
            0.02,
            y_pos,
            f"{tier_labels[tier_name]}",
            fontsize=8,
            fontweight="bold",
            color=color,
            va="center",
            transform=ax.transData,
        )
        ax.text(
            0.85,
            y_pos,
            f"{best_score:.3f}",
            fontsize=7,
            va="center",
            color=color,
            ha="right",
            transform=ax.transData,
        )
        y_pos += 1

        for rule_id in rules:
            if rule_id not in RULE_INDEX_MAP:
                continue
            idx = RULE_INDEX_MAP[rule_id]
            viol = violations[best_mode_idx, idx] if idx < violations.shape[1] else 0
            name = RULE_SHORT_NAMES.get(rule_id, rule_id)
            name_alpha = 1.0 if viol > 0.01 else 0.5

            ax.text(
                0.04,
                y_pos,
                name,
                fontsize=6.5,
                va="center",
                alpha=name_alpha,
                transform=ax.transData,
            )

            # Violation bar
            bar_x, bar_w = 0.5, 0.3
            ax.barh(
                y_pos,
                bar_w,
                left=bar_x,
                height=0.5,
                color="#e0e0e0",
                alpha=0.4,
                transform=ax.transData,
            )
            if viol > 0.005:
                if viol < 0.2:
                    bar_c = "#27ae60"
                elif viol < 0.5:
                    bar_c = "#f1c40f"
                elif viol < 0.7:
                    bar_c = "#e67e22"
                else:
                    bar_c = "#c0392b"
                ax.barh(
                    y_pos,
                    min(viol, 1.0) * bar_w,
                    left=bar_x,
                    height=0.5,
                    color=bar_c,
                    alpha=0.9,
                    transform=ax.transData,
                )
            ax.text(
                bar_x + bar_w + 0.02,
                y_pos,
                f"{viol:.3f}" if viol > 0 else "0",
                fontsize=6,
                va="center",
                alpha=0.8,
                transform=ax.transData,
            )
            y_pos += 1
        y_pos += 0.5

    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.5, y_pos + 0.5)
    ax.invert_yaxis()


def generate_movie(
    m2i_gen: M2ITrajectoryGenerator,
    rule_eval: KinematicRuleEvaluator,
    scenario_data: Dict,
    output_path: str,
    fps: int = 10,
    view_range: float = 80,
    predict_interval: int = 5,
):
    """
    Generate a receding-horizon movie for one scenario.

    Args:
        m2i_gen: Loaded M2I trajectory generator
        rule_eval: Kinematic rule evaluator
        scenario_data: Dict from _extract_scenario_data
        output_path: Path for the .mp4 file
        fps: Frames per second
        view_range: BEV view range in meters
        predict_interval: Re-predict every N frames (10 = every second)
    """
    scenario = scenario_data["scenario_proto"]
    ego_traj = scenario_data["ego_traj"]
    agent_trajs = scenario_data["agent_trajs"]
    lanes = scenario_data["lanes"]
    road_edges = scenario_data["road_edges"]
    crosswalks = scenario_data["crosswalks"]

    # Scene center (follow ego)
    all_ego_pts = ego_traj[ego_traj[:, 4] > 0, :2]
    if len(all_ego_pts) == 0:
        return

    # Cache predictions
    pred_cache = {}
    rule_cache = {}

    fig = plt.figure(figsize=(18, 10), dpi=100)
    gs = GridSpec(1, 2, width_ratios=[2.8, 1], wspace=0.05)
    ax_scene = fig.add_subplot(gs[0])
    ax_rules = fig.add_subplot(gs[1])

    def animate(frame):
        ax_scene.clear()
        current_t = frame

        # Ego center for view
        if ego_traj[current_t, 4] > 0:
            cx, cy = ego_traj[current_t, 0], ego_traj[current_t, 1]
        else:
            cx, cy = 0, 0

        # Draw map
        for edge in road_edges:
            ax_scene.plot(
                edge[:, 0],
                edge[:, 1],
                color=COLORS["road_edge"],
                linewidth=2,
                alpha=0.7,
                zorder=1,
            )
        for lane in lanes:
            ax_scene.plot(
                lane[:, 0],
                lane[:, 1],
                color=COLORS["lane"],
                linewidth=0.8,
                alpha=0.4,
                zorder=2,
            )
        for cw in crosswalks:
            if len(cw) >= 3:
                poly = plt.Polygon(
                    cw, facecolor=COLORS["crosswalk"], alpha=0.25, zorder=1
                )
                ax_scene.add_patch(poly)

        # Draw other agents
        for a in range(scenario_data["agent_count"]):
            if agent_trajs[a, current_t, 4] > 0:
                ax_, ay_ = agent_trajs[a, current_t, 0], agent_trajs[a, current_t, 1]
                ah_ = agent_trajs[a, current_t, 2]
                atype = int(agent_trajs[a, current_t, 5])
                if atype == 2:
                    ax_scene.scatter(
                        ax_, ay_, c=COLORS["agents_ped"], s=60, zorder=5, marker="o"
                    )
                elif atype == 3:
                    ax_scene.scatter(
                        ax_, ay_, c=COLORS["agents_cyclist"], s=80, zorder=5, marker="^"
                    )
                else:
                    draw_vehicle(
                        ax_scene,
                        ax_,
                        ay_,
                        ah_,
                        length=4.0,
                        width=1.8,
                        color=COLORS["agents_vehicle"],
                        alpha=0.6,
                        zorder=5,
                    )

        # Ego history trail
        hist_start = max(0, current_t - 10)
        hist_pts = [
            ego_traj[t, :2]
            for t in range(hist_start, current_t + 1)
            if ego_traj[t, 4] > 0
        ]
        if hist_pts:
            hp = np.array(hist_pts)
            ax_scene.plot(
                hp[:, 0],
                hp[:, 1],
                color=COLORS["ego_history"],
                linewidth=2,
                alpha=0.5,
                linestyle="--",
                zorder=8,
            )

        # GT future
        future_end = min(current_t + M2I_FUTURE_LENGTH, TOTAL_TIMESTEPS)
        gt_pts = [
            ego_traj[t, :2] for t in range(current_t, future_end) if ego_traj[t, 4] > 0
        ]
        if len(gt_pts) > 1:
            gp = np.array(gt_pts)
            ax_scene.plot(
                gp[:, 0],
                gp[:, 1],
                color=COLORS["gt_future"],
                linewidth=2,
                alpha=0.4,
                linestyle=":",
                zorder=7,
                label="GT Future",
            )

        # M2I prediction (re-run periodically)
        violations = np.zeros((6, NUM_RULES), dtype=np.float32)
        tier_scores = {t: np.zeros(6) for t in TIERS}
        composite = np.zeros(6)
        best_mode = 0

        if current_t >= 10 and current_t < TOTAL_TIMESTEPS - 1:
            cache_key = current_t // predict_interval
            if cache_key not in pred_cache:
                try:
                    traj_world, scores, mapping = m2i_gen.predict_from_scenario(
                        scenario,
                        target_agent_idx=0,
                        current_time_index=current_t,
                    )
                    if traj_world is not None:
                        # Convert world → local frame for display
                        ref_x = scenario_data["ref_x"]
                        ref_y = scenario_data["ref_y"]
                        ref_h = scenario_data["ref_h"]
                        cos_h = math.cos(-ref_h)
                        sin_h = math.sin(-ref_h)

                        traj_local = np.zeros_like(traj_world)
                        for m in range(traj_world.shape[0]):
                            for t in range(traj_world.shape[1]):
                                dx = traj_world[m, t, 0] - ref_x
                                dy = traj_world[m, t, 1] - ref_y
                                traj_local[m, t, 0] = dx * cos_h - dy * sin_h
                                traj_local[m, t, 1] = dx * sin_h + dy * cos_h

                        # Rule evaluation
                        rv = rule_eval.evaluate(traj_local)
                        best_idx = int(np.argmin(rv.composite_scores))
                        if np.all(rv.composite_scores == rv.composite_scores[0]):
                            best_idx = (
                                int(np.argmax(scores)) if scores is not None else 0
                            )

                        pred_cache[cache_key] = {
                            "traj_local": traj_local,
                            "scores": scores,
                            "best_mode": best_idx,
                        }
                        rule_cache[cache_key] = rv
                except Exception as e:
                    pass

            # Use cached prediction
            if cache_key in pred_cache:
                pc = pred_cache[cache_key]
                traj_local = pc["traj_local"]
                best_mode = pc["best_mode"]

                if cache_key in rule_cache:
                    rv = rule_cache[cache_key]
                    violations = rv.violations
                    tier_scores = rv.tier_scores
                    composite = rv.composite_scores

                # Draw all modes
                for m in range(traj_local.shape[0]):
                    if m == best_mode:
                        ax_scene.plot(
                            traj_local[m, :, 0],
                            traj_local[m, :, 1],
                            color=COLORS["planned_best"],
                            linewidth=3,
                            alpha=0.9,
                            zorder=9,
                            label=f"Rule-Best (mode {m})",
                        )
                        ax_scene.scatter(
                            traj_local[m, ::10, 0],
                            traj_local[m, ::10, 1],
                            c=COLORS["planned_best"],
                            s=25,
                            marker="o",
                            zorder=9,
                            alpha=0.7,
                            edgecolors="white",
                            linewidths=0.5,
                        )
                        ax_scene.scatter(
                            traj_local[m, -1, 0],
                            traj_local[m, -1, 1],
                            c=COLORS["planned_best"],
                            s=120,
                            marker="*",
                            zorder=10,
                            edgecolors="white",
                            linewidths=1,
                        )
                    else:
                        ax_scene.plot(
                            traj_local[m, :, 0],
                            traj_local[m, :, 1],
                            color=COLORS["other_modes"],
                            linewidth=1,
                            alpha=0.25,
                            zorder=8,
                        )

        # Ego vehicle
        if ego_traj[current_t, 4] > 0:
            draw_vehicle(
                ax_scene,
                ego_traj[current_t, 0],
                ego_traj[current_t, 1],
                ego_traj[current_t, 2],
                length=4.8,
                width=2.1,
                color=COLORS["ego"],
                alpha=1.0,
                zorder=15,
                label="Ego (SDC)",
            )

        ax_scene.set_xlim(cx - view_range / 2, cx + view_range / 2)
        ax_scene.set_ylim(cy - view_range / 2, cy + view_range / 2)
        ax_scene.set_aspect("equal")
        ax_scene.grid(True, alpha=0.15)
        ax_scene.legend(loc="upper left", fontsize=8)

        time_sec = (current_t - 10) * DT
        if current_t < 10:
            phase = "History"
        else:
            cycle_num = (current_t - 10) // predict_interval + 1
            cycle_dt = predict_interval * DT
            is_replan = current_t >= 10 and (current_t - 10) % predict_interval == 0
            phase = (
                f"Plan cycle {cycle_num} "
                f"(every {cycle_dt:.1f}s)"
                f"{' ★ REPLAN' if is_replan else ''}"
            )
        ax_scene.set_xlabel("X (m)", fontsize=10)
        ax_scene.set_ylabel("Y (m)", fontsize=10)
        ax_scene.set_title(
            f"M2I DenseTNT + RECTOR Rule Scoring — "
            f'{scenario_data["scenario_id"][:12]}\n'
            f"{phase} | t = {time_sec:+.1f}s | "
            f"Frame {current_t + 1}/{TOTAL_TIMESTEPS}",
            fontsize=11,
        )

        # Rule panel
        draw_rule_panel(
            ax_rules, violations, tier_scores, composite, best_mode, current_t
        )

        return []

    print(f"    Rendering {TOTAL_TIMESTEPS} frames...")
    anim = animation.FuncAnimation(
        fig,
        animate,
        frames=TOTAL_TIMESTEPS,
        interval=1000 // fps,
        blit=False,
    )

    writer = animation.FFMpegWriter(fps=fps, bitrate=4000)
    anim.save(output_path, writer=writer)
    print(f"    Saved: {output_path}")
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(
        description="M2I DenseTNT + RECTOR Rule Scoring — Movie Generator",
    )
    p.add_argument(
        "--data_dir",
        type=str,
        default="/workspace/data/WOMD/datasets/waymo_open_dataset/"
        "motion_v_1_3_0/processed/augmented/scenario/"
        "validation_interactive",
        help="TFRecord data directory",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="/workspace/models/RECTOR/movies/m2i_rector",
        help="Output directory for videos",
    )
    p.add_argument(
        "--m2i_model",
        type=str,
        default="/workspace/models/pretrained/m2i/models/densetnt/model.24.bin",
    )
    p.add_argument("--num_scenarios", type=int, default=5)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--view_range", type=float, default=80, help="View range in meters")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--min_ego_speed",
        type=float,
        default=2.0,
        help="Min avg ego speed (m/s) to select scenario",
    )
    p.add_argument(
        "--predict_interval",
        type=int,
        default=20,
        help="Re-predict every N frames (20=2s planning cycle)",
    )
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main():
    args = parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("M2I DenseTNT + RECTOR Rule Scoring — Movie Generator")
    print("=" * 70)
    print(f"  Prediction: M2I DenseTNT (8s, 6 modes)")
    print(f"  Scoring: RECTOR Kinematic Rule Evaluator (28 rules, 4 tiers)")
    print(f"  Duration: {TOTAL_TIMESTEPS * DT:.1f}s per scenario")
    print(
        f"  Planning cycle: every {args.predict_interval} frames "
        f"({args.predict_interval * DT:.1f}s)"
    )

    # Load M2I model
    print("\n[1/3] Loading M2I DenseTNT model...")
    m2i_gen = M2ITrajectoryGenerator(
        model_path=args.m2i_model,
        device=args.device,
    )
    m2i_gen.load_model()

    rule_eval = KinematicRuleEvaluator()
    print("  Rule evaluator ready (28 kinematic rules)")

    # Load scenarios
    print(f"\n[2/3] Loading scenarios from {args.data_dir}...")
    all_scenarios = load_scenarios_from_tfrecords(
        args.data_dir,
        max_files=20,
        seed=args.seed,
    )
    print(f"  Loaded {len(all_scenarios)} scenarios")

    if not all_scenarios:
        print("ERROR: No valid scenarios!")
        return

    # Filter for moving scenarios
    moving = [s for s in all_scenarios if s["ego_avg_speed"] >= args.min_ego_speed]
    print(
        f"  Moving scenarios (speed >= {args.min_ego_speed} m/s): "
        f"{len(moving)}/{len(all_scenarios)}"
    )

    if not moving:
        print("  WARNING: No moving scenarios, using all.")
        moving = all_scenarios

    n_gen = min(args.num_scenarios, len(moving))
    selected = random.sample(moving, n_gen)

    # Generate movies
    print(f"\n[3/3] Generating {n_gen} movies...")
    t_start = time.time()

    for i, scenario_data in enumerate(selected):
        sid = scenario_data["scenario_id"][:12]
        output_path = str(output_dir / f"m2i_rector_{i:03d}_{sid}.mp4")
        print(
            f"\n  [{i + 1}/{n_gen}] Scenario {sid}... "
            f"(dist={scenario_data['ego_total_dist']:.0f}m, "
            f"speed={scenario_data['ego_avg_speed']:.1f}m/s)"
        )

        generate_movie(
            m2i_gen=m2i_gen,
            rule_eval=rule_eval,
            scenario_data=scenario_data,
            output_path=output_path,
            fps=args.fps,
            view_range=args.view_range,
            predict_interval=args.predict_interval,
        )

    elapsed = time.time() - t_start
    print("\n" + "=" * 70)
    print("MOVIE GENERATION COMPLETE")
    print("=" * 70)
    print(f"  Videos: {n_gen}")
    print(f"  Time: {elapsed:.1f}s ({elapsed / max(n_gen, 1):.1f}s each)")
    print(f"  Output: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
