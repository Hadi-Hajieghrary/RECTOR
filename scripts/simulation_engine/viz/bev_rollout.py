#!/usr/bin/env python3
"""
BEV (Bird's-Eye View) Rollout Renderer
========================================

Renders a 2-D top-down animation of a closed-loop scenario rollout.
Draws:
  • Lane geometry (grey polylines)
  • Agent bounding boxes (blue)
  • Ego vehicle (green / red if overlapping)
  • Logged ego trajectory (dashed cyan)
  • Executed ego trajectory (solid green)
  • Per-step metrics annotation overlay

Outputs either a sequence of PNG frames or an MP4 video (if ffmpeg
is available).

Usage:
    python -m simulation_engine.viz.bev_rollout \
        --scenario-index 0 \
        --outdir /workspace/output/closedloop/bev_frames
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np


@dataclass
class BEVConfig:
    """Rendering configuration."""

    view_range_m: float = 60.0  # Half-side of the BEV window (metres)
    fig_size: Tuple[float, float] = (6, 6)
    dpi: int = 150
    ego_length_m: float = 4.8
    ego_width_m: float = 2.0
    agent_length_m: float = 4.5
    agent_width_m: float = 1.9
    lane_color: str = "#cccccc"
    lane_width: float = 0.8
    ego_color_ok: str = "#2ca02c"
    ego_color_collision: str = "#d62728"
    agent_color: str = "#1f77b4"
    logged_traj_color: str = "#17becf"
    exec_traj_color: str = "#2ca02c"
    bg_color: str = "#f5f5f0"
    max_agents_draw: int = 32
    annotate_metrics: bool = True


def _extract_road_polylines(state) -> Tuple[List[np.ndarray], List[int]]:
    """Extract road polylines from Waymax SimulatorState.roadgraph_points.

    Groups roadgraph points by segment ID and returns each polyline as
    a variable-length [N, 2] array in global coordinates.

    Returns
    -------
    polylines : list of np.ndarray, each [N, 2]
    poly_types : list of int – road type ID for each polyline
        Common types: 1=freeway, 2=surface_street, 3=bike_lane,
        6=stop_sign, 7=crosswalk, 15=road_edge_boundary, ...
    """
    rg = state.roadgraph_points
    x = np.asarray(rg.x)
    y = np.asarray(rg.y)
    valid = np.asarray(rg.valid)
    ids = np.asarray(rg.ids)
    types = np.asarray(rg.types)

    polylines: List[np.ndarray] = []
    poly_types: List[int] = []

    unique_ids = np.unique(ids[valid])
    for rid in unique_ids:
        mask = (ids == rid) & valid
        if mask.sum() < 2:
            continue
        indices = np.where(mask)[0]  # order preserved from data
        poly = np.stack([x[indices], y[indices]], axis=-1)  # [N, 2]
        polylines.append(poly)
        poly_types.append(int(types[indices[0]]))

    return polylines, poly_types


def _rotated_rect(
    cx: float,
    cy: float,
    length: float,
    width: float,
    yaw: float,
    color: str,
    alpha: float = 0.7,
) -> mpatches.FancyBboxPatch:
    """Create a rotated rectangle patch."""
    import matplotlib.transforms as mtrans

    rect = plt.Rectangle(
        (-length / 2, -width / 2),
        length,
        width,
        facecolor=color,
        edgecolor="black",
        linewidth=0.5,
        alpha=alpha,
    )
    t = mtrans.Affine2D().rotate(yaw).translate(cx, cy)
    rect.set_transform(t)
    return rect


# Road-type styling constants
# Types match scenario_loader._MAP_TYPE: lane=1, road_line=2, road_edge=3,
# stop_sign=4, crosswalk=5, speed_bump=6
_ROAD_STYLE: Dict[int, Tuple[str, float, str]] = {
    # type_id: (color, linewidth_scale, linestyle)
    1: ("#888888", 1.0, "-"),  # lane (center polyline)
    2: ("#aaaaaa", 0.8, "--"),  # road_line (lane boundaries)
    3: ("#555555", 1.6, "-"),  # road_edge (boundary)
    4: ("#cc0000", 1.8, "-"),  # stop_sign
    5: ("#ddbb44", 1.5, "-"),  # crosswalk
    6: ("#cc8800", 1.5, "-"),  # speed_bump
    # Legacy Waymax raw types (in case data uses them)
    7: ("#ddbb44", 1.5, "-"),  # crosswalk (alt)
    8: ("#cc8800", 1.5, "-"),  # speed bump (alt)
    15: ("#555555", 1.6, "-"),  # road edge boundary (alt)
    16: ("#555555", 1.4, "--"),  # road edge median (alt)
    17: ("#999999", 1.0, "-"),  # lane center freeway (alt)
    18: ("#888888", 1.0, "-"),  # lane center surface street (alt)
    19: ("#aabb99", 0.7, "--"),  # bike lane (alt)
}


def render_frame(
    ax: plt.Axes,
    cfg: BEVConfig,
    ego_x: float,
    ego_y: float,
    ego_yaw: float,
    agent_positions: np.ndarray,  # [A, 3] (x, y, yaw)
    agent_valid: np.ndarray,  # [A] bool
    lane_segments: Optional[np.ndarray] = None,  # [L, P, 2] (legacy)
    lane_valid: Optional[np.ndarray] = None,  # [L] (legacy)
    road_polylines: Optional[List[np.ndarray]] = None,  # list of [N, 2]
    road_types: Optional[List[int]] = None,  # type per polyline
    logged_traj_xy: Optional[np.ndarray] = None,  # [T, 2]
    exec_traj_xy: Optional[np.ndarray] = None,  # [steps, 2]
    overlap: float = 0.0,
    step_idx: int = 0,
    metrics_text: str = "",
):
    """Render one BEV frame onto *ax*."""
    ax.clear()
    ax.set_facecolor(cfg.bg_color)

    # Centre view on ego
    r = cfg.view_range_m
    ax.set_xlim(ego_x - r, ego_x + r)
    ax.set_ylim(ego_y - r, ego_y + r)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]", fontsize=7)
    ax.set_ylabel("y [m]", fontsize=7)
    ax.tick_params(labelsize=6)

    # --- Road polylines (new path) ---
    if road_polylines is not None:
        xlo, xhi = ego_x - r - 20, ego_x + r + 20
        ylo, yhi = ego_y - r - 20, ego_y + r + 20
        for i, poly in enumerate(road_polylines):
            # Cull polylines entirely outside the view (with margin)
            px, py = poly[:, 0], poly[:, 1]
            if px.max() < xlo or px.min() > xhi or py.max() < ylo or py.min() > yhi:
                continue
            rtype = road_types[i] if road_types is not None else 0
            color, lw_scale, ls = _ROAD_STYLE.get(rtype, (cfg.lane_color, 1.0, "-"))
            ax.plot(
                px,
                py,
                color=color,
                lw=cfg.lane_width * lw_scale,
                ls=ls,
                zorder=1,
                solid_capstyle="round",
            )

    # --- Lanes (legacy fixed-size path) ---
    elif lane_segments is not None and lane_valid is not None:
        n_lanes = lane_segments.shape[0]
        for li in range(min(n_lanes, 200)):
            if not lane_valid[li]:
                continue
            pts = lane_segments[li]
            valid_pts = pts[np.linalg.norm(pts, axis=-1) > 0]
            if len(valid_pts) > 1:
                ax.plot(
                    valid_pts[:, 0],
                    valid_pts[:, 1],
                    color=cfg.lane_color,
                    lw=cfg.lane_width,
                    zorder=1,
                )

    # --- Logged trajectory ---
    if logged_traj_xy is not None and len(logged_traj_xy) > 1:
        ax.plot(
            logged_traj_xy[:, 0],
            logged_traj_xy[:, 1],
            color=cfg.logged_traj_color,
            lw=1.2,
            ls="--",
            alpha=0.6,
            zorder=3,
            label="Logged",
        )

    # --- Executed trajectory ---
    if exec_traj_xy is not None and len(exec_traj_xy) > 1:
        ax.plot(
            exec_traj_xy[:, 0],
            exec_traj_xy[:, 1],
            color=cfg.exec_traj_color,
            lw=1.5,
            alpha=0.8,
            zorder=4,
            label="Executed",
        )

    # --- Other agents ---
    import matplotlib.transforms as mtrans

    n_draw = min(int(agent_valid.sum()), cfg.max_agents_draw)
    drawn = 0
    for ai in range(len(agent_valid)):
        if not agent_valid[ai]:
            continue
        if drawn >= n_draw:
            break
        ax_pos, ay_pos, a_yaw = agent_positions[ai]
        rect = plt.Rectangle(
            (-cfg.agent_length_m / 2, -cfg.agent_width_m / 2),
            cfg.agent_length_m,
            cfg.agent_width_m,
            facecolor=cfg.agent_color,
            edgecolor="black",
            linewidth=0.3,
            alpha=0.5,
        )
        t = mtrans.Affine2D().rotate(a_yaw).translate(ax_pos, ay_pos) + ax.transData
        rect.set_transform(t)
        ax.add_patch(rect)
        drawn += 1

    # --- Ego vehicle ---
    ego_color = cfg.ego_color_collision if overlap > 0 else cfg.ego_color_ok
    ego_rect = plt.Rectangle(
        (-cfg.ego_length_m / 2, -cfg.ego_width_m / 2),
        cfg.ego_length_m,
        cfg.ego_width_m,
        facecolor=ego_color,
        edgecolor="black",
        linewidth=0.8,
        alpha=0.85,
    )
    ego_t = mtrans.Affine2D().rotate(ego_yaw).translate(ego_x, ego_y) + ax.transData
    ego_rect.set_transform(ego_t)
    ax.add_patch(ego_rect)

    # Heading indicator
    hx = ego_x + cfg.ego_length_m * 0.6 * np.cos(ego_yaw)
    hy = ego_y + cfg.ego_length_m * 0.6 * np.sin(ego_yaw)
    ax.annotate(
        "",
        xy=(hx, hy),
        xytext=(ego_x, ego_y),
        arrowprops=dict(arrowstyle="->", color="white", lw=1.2),
        zorder=10,
    )

    # --- Step & metrics annotation ---
    title = f"Step {step_idx} (t={step_idx*0.1:.1f}s)"
    if overlap > 0:
        title += "  [OVERLAP]"
    ax.set_title(title, fontsize=8, fontweight="bold")

    if cfg.annotate_metrics and metrics_text:
        ax.text(
            0.02,
            0.02,
            metrics_text,
            transform=ax.transAxes,
            fontsize=5,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85),
            family="monospace",
        )

    ax.legend(loc="upper right", fontsize=6, framealpha=0.8)


def render_scenario_rollout(
    scenario_index: int = 0,
    max_steps: int = 80,
    outdir: str = "/workspace/output/closedloop/bev_frames",
    cfg: Optional[BEVConfig] = None,
    make_video: bool = True,
):
    """
    Load scenario, run simulation, render frame-by-frame.

    This reuses the existing simulation loop infrastructure.
    """
    if cfg is None:
        cfg = BEVConfig()

    # Lazy imports to avoid loading JAX until needed
    from ..waymax_bridge.scenario_loader import load_scenarios, ScenarioLoaderConfig
    from ..waymax_bridge.env_factory import make_env
    from ..waymax_bridge.observation_extractor import (
        extract_observation,
        ObservationExtractorConfig,
    )
    from ..waymax_bridge.action_converter import trajectory_to_action
    from ..waymax_bridge.metric_collector import MetricCollector, MetricCollectorConfig
    from ..waymax_bridge.simulation_loop import MockLogReplayGenerator
    from ..selectors.confidence import ConfidenceSelector
    from ..config import WaymaxBridgeConfig, AgentConfig

    loader_cfg = ScenarioLoaderConfig(max_scenarios=scenario_index + 1)
    scenarios = list(load_scenarios(loader_cfg))
    if scenario_index >= len(scenarios):
        print(
            f"Only {len(scenarios)} scenarios available, requested index {scenario_index}"
        )
        return

    sid, sim_state = scenarios[scenario_index]
    print(f"Rendering scenario {sid} (index {scenario_index})")

    # Setup
    bridge_cfg = WaymaxBridgeConfig()
    agent_cfg = AgentConfig()
    generator = MockLogReplayGenerator()
    selector = ConfidenceSelector()
    obs_cfg = ObservationExtractorConfig()
    metric_collector = MetricCollector()

    env, state = make_env(sim_state, dynamics_model=bridge_cfg.dynamics_model)
    metric_collector.reset()

    # Extract road geometry (static — do once)
    road_polylines, road_types = _extract_road_polylines(state)
    print(f"  Road geometry: {len(road_polylines)} polylines")

    # Extract logged trajectory for reference
    import jax.numpy as jnp

    sdc_idx = int(jnp.argmax(state.object_metadata.is_sdc))
    log_x = np.asarray(state.log_trajectory.x[sdc_idx])
    log_y = np.asarray(state.log_trajectory.y[sdc_idx])
    logged_xy = np.stack([log_x, log_y], axis=-1)  # [91, 2]

    # Prepare frame output
    os.makedirs(outdir, exist_ok=True)
    exec_traj = []  # Accumulate executed positions

    steps_per_replan = bridge_cfg.steps_per_replan
    total_steps = min(max_steps, 91 - int(state.timestep) - 1)

    selected_traj = None
    obs = None
    step_in_plan = 0

    fig, ax = plt.subplots(figsize=cfg.fig_size)

    for step_i in range(total_steps):
        # Replan
        if step_i % steps_per_replan == 0 or selected_traj is None:
            obs = extract_observation(state, obs_cfg)
            gen_out = generator.generate(obs, state)
            trajectories = gen_out["trajectories"]
            confidence = gen_out["confidence"]
            rule_costs = gen_out["rule_costs"]
            applicability = gen_out["applicability"]

            K = rule_costs.shape[0]
            tier_scores = np.zeros((K, 4), dtype=np.float32)
            sel_idx, trace = selector.select(
                candidates=trajectories,
                probs=confidence,
                tier_scores=tier_scores,
                rule_scores=rule_costs,
                rule_applicability=applicability,
            )
            selected_traj = trajectories[sel_idx]
            step_in_plan = 0

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
        metric_collector.step(state)

        # Record ego position
        ego_x_now = float(
            np.asarray(state.sim_trajectory.x[sdc_idx, int(state.timestep)])
        )
        ego_y_now = float(
            np.asarray(state.sim_trajectory.y[sdc_idx, int(state.timestep)])
        )
        ego_yaw_now = float(
            np.asarray(state.sim_trajectory.yaw[sdc_idx, int(state.timestep)])
        )
        exec_traj.append([ego_x_now, ego_y_now])

        # Agent positions
        t_now = int(state.timestep)
        n_obj = state.sim_trajectory.x.shape[0]
        agent_pos = np.zeros((n_obj, 3), dtype=np.float32)
        agent_valid = np.zeros(n_obj, dtype=bool)
        for ai in range(n_obj):
            if ai == sdc_idx:
                continue
            v = bool(np.asarray(state.sim_trajectory.valid[ai, t_now]))
            if v:
                agent_pos[ai, 0] = float(np.asarray(state.sim_trajectory.x[ai, t_now]))
                agent_pos[ai, 1] = float(np.asarray(state.sim_trajectory.y[ai, t_now]))
                agent_pos[ai, 2] = float(
                    np.asarray(state.sim_trajectory.yaw[ai, t_now])
                )
                agent_valid[ai] = True

        # Check overlap (approximate)
        overlap_val = 0.0
        try:
            from waymax import metrics as wm_metrics

            ov = wm_metrics.overlap(state)
            overlap_val = float(np.asarray(ov.value[sdc_idx]))
        except Exception:
            pass

        # Metrics text
        metrics_text = f"step={step_i+1}/{total_steps}\n" f"overlap={overlap_val:.3f}"

        # Render frame
        render_frame(
            ax=ax,
            cfg=cfg,
            ego_x=ego_x_now,
            ego_y=ego_y_now,
            ego_yaw=ego_yaw_now,
            agent_positions=agent_pos,
            agent_valid=agent_valid,
            road_polylines=road_polylines,
            road_types=road_types,
            logged_traj_xy=logged_xy[: t_now + 1],
            exec_traj_xy=np.array(exec_traj) if exec_traj else None,
            overlap=overlap_val,
            step_idx=step_i + 1,
            metrics_text=metrics_text,
        )

        frame_path = os.path.join(outdir, f"frame_{step_i:04d}.png")
        fig.savefig(frame_path, dpi=cfg.dpi)

        if bool(state.is_done):
            break

    plt.close(fig)
    print(f"Rendered {step_i+1} frames to {outdir}")

    # Final metrics
    summary = metric_collector.finalise()
    print("Final metrics:")
    for k, v in sorted(summary.items()):
        print(f"  {k}: {v:.4f}")

    # Try to make video
    if make_video:
        _stitch_video(outdir, os.path.join(outdir, f"scenario_{sid}.mp4"))


def _stitch_video(frame_dir: str, outpath: str, fps: int = 10):
    """Stitch PNG frames into an MP4 using ffmpeg."""
    import subprocess

    pattern = os.path.join(frame_dir, "frame_%04d.png")
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        pattern,
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
        print(f"Video saved → {outpath}")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"ffmpeg not available or failed: {e}")
        print(f"Frames are still available in {frame_dir}")


def main():
    parser = argparse.ArgumentParser(description="BEV rollout renderer")
    parser.add_argument(
        "--scenario-index",
        type=int,
        default=0,
        help="Index of the scenario to render (0-based)",
    )
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--outdir", default="/workspace/output/closedloop/bev_frames")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument(
        "--view-range", type=float, default=60.0, help="BEV window half-side in metres"
    )
    args = parser.parse_args()

    cfg = BEVConfig(view_range_m=args.view_range)
    render_scenario_rollout(
        scenario_index=args.scenario_index,
        max_steps=args.max_steps,
        outdir=args.outdir,
        cfg=cfg,
        make_video=not args.no_video,
    )


if __name__ == "__main__":
    main()
