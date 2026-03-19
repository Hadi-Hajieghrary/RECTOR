#!/usr/bin/env python3
"""
Batch BEV Closed-Loop Movie Generator
======================================

Generates closed-loop BEV rollout movies for multiple scenarios.
Loads all scenarios once, then renders each one with frame-by-frame
simulation and stitches into MP4 videos.

Usage:
    source /opt/venv/bin/activate
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    export XLA_PYTHON_CLIENT_MEM_FRACTION=0.25
    cd /workspace
    python -m simulation_engine.viz.generate_bev_batch \
        --num-scenarios 50 \
        --outdir /workspace/output/closedloop/videos \
        --max-steps 80

Output:
    /workspace/output/closedloop/videos/
        scenario_000_<id>.mp4
        scenario_001_<id>.mp4
        ...
        summary.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .bev_rollout import BEVConfig, render_frame, _extract_road_polylines


def _stitch_video(frame_dir: str, outpath: str, fps: int = 10) -> bool:
    """Stitch PNG frames into an MP4 using ffmpeg. Returns success."""
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
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"    ffmpeg failed: {e}")
        return False


def render_single_scenario(
    scenario_id: str,
    sim_state,
    env_factory_fn,
    generator,
    selector,
    obs_extractor_fn,
    obs_cfg,
    metric_collector_cls,
    metric_cfg,
    bridge_cfg,
    cfg: BEVConfig,
    max_steps: int,
    outdir: str,
    scenario_idx: int,
    fps: int = 10,
) -> Dict:
    """
    Run closed-loop sim on one scenario and render BEV frames + MP4.

    Returns dict with scenario metadata and metrics.
    """
    import jax.numpy as jnp

    t_start = time.time()

    # Create environment
    env, state = env_factory_fn(
        sim_state,
        dynamics_model=bridge_cfg.dynamics_model,
    )

    # Metric collector
    mc = metric_collector_cls(metric_cfg)
    mc.reset()

    # Extract road geometry (static)
    road_polylines, road_types = _extract_road_polylines(state)

    # Extract logged trajectory for reference
    sdc_idx = int(jnp.argmax(state.object_metadata.is_sdc))
    log_x = np.asarray(state.log_trajectory.x[sdc_idx])
    log_y = np.asarray(state.log_trajectory.y[sdc_idx])
    logged_xy = np.stack([log_x, log_y], axis=-1)

    # Frame output directory
    frame_dir = os.path.join(outdir, f"frames_{scenario_idx:03d}")
    os.makedirs(frame_dir, exist_ok=True)

    exec_traj = []
    steps_per_replan = bridge_cfg.steps_per_replan
    total_steps = min(max_steps, 91 - int(state.timestep) - 1)

    selected_traj = None
    obs = None
    step_in_plan = 0
    overlap_events = 0
    max_overlap = 0.0

    fig, ax = plt.subplots(figsize=cfg.fig_size)

    actual_steps = 0
    for step_i in range(total_steps):
        # Replan
        if step_i % steps_per_replan == 0 or selected_traj is None:
            obs = obs_extractor_fn(state, obs_cfg)
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
        from ..waymax_bridge.action_converter import trajectory_to_action

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

        # Record ego position
        t_now = int(state.timestep)
        ego_x_now = float(np.asarray(state.sim_trajectory.x[sdc_idx, t_now]))
        ego_y_now = float(np.asarray(state.sim_trajectory.y[sdc_idx, t_now]))
        ego_yaw_now = float(np.asarray(state.sim_trajectory.yaw[sdc_idx, t_now]))
        exec_traj.append([ego_x_now, ego_y_now])

        # Agent positions
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

        # Check overlap
        overlap_val = 0.0
        try:
            from waymax import metrics as wm_metrics

            ov = wm_metrics.overlap(state)
            overlap_val = float(np.asarray(ov.value[sdc_idx]))
        except Exception:
            pass

        if overlap_val > 0:
            overlap_events += 1
        max_overlap = max(max_overlap, overlap_val)

        # Metrics text
        metrics_text = (
            f"Scenario: {scenario_id[:16]}\n"
            f"step={step_i+1}/{total_steps}\n"
            f"overlap={overlap_val:.3f}"
        )

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

        frame_path = os.path.join(frame_dir, f"frame_{step_i:04d}.png")
        fig.savefig(frame_path, dpi=cfg.dpi)
        actual_steps = step_i + 1

        if bool(state.is_done):
            break

    plt.close(fig)

    # Finalise metrics
    try:
        summary_metrics = mc.finalise()
    except Exception:
        summary_metrics = {}

    elapsed = time.time() - t_start

    # Stitch video
    safe_id = scenario_id.replace("/", "_").replace(" ", "_")[:32]
    video_name = f"scenario_{scenario_idx:03d}_{safe_id}.mp4"
    video_path = os.path.join(outdir, video_name)
    video_ok = _stitch_video(frame_dir, video_path, fps=fps)

    # Clean up frames to save disk
    if video_ok:
        import shutil

        shutil.rmtree(frame_dir, ignore_errors=True)

    result = {
        "scenario_idx": scenario_idx,
        "scenario_id": scenario_id,
        "steps_rendered": actual_steps,
        "total_steps": total_steps,
        "overlap_events": overlap_events,
        "max_overlap": float(max_overlap),
        "video_path": video_path if video_ok else None,
        "video_name": video_name if video_ok else None,
        "elapsed_seconds": round(elapsed, 1),
        "metrics": {
            k: float(v) if np.isfinite(v) else None for k, v in summary_metrics.items()
        },
        "road_polylines_count": len(road_polylines),
    }

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Batch BEV closed-loop movie generator"
    )
    parser.add_argument(
        "--num-scenarios",
        type=int,
        default=50,
        help="Number of scenarios to render (default: 50)",
    )
    parser.add_argument(
        "--outdir",
        default="/workspace/output/closedloop/videos",
        help="Output directory for MP4 files",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=80,
        help="Max simulation steps per scenario (default: 80 = 8s at 10Hz)",
    )
    parser.add_argument(
        "--max-objects",
        type=int,
        default=32,
        help="Max objects per scenario (default: 32, reduced for speed)",
    )
    parser.add_argument(
        "--max-rg-points",
        type=int,
        default=20000,
        help="Max roadgraph points (default: 20000 for full road geometry)",
    )
    parser.add_argument(
        "--view-range",
        type=float,
        default=60.0,
        help="BEV window half-side in metres (default: 60)",
    )
    parser.add_argument(
        "--fps", type=int, default=10, help="Video frame rate (default: 10)"
    )
    parser.add_argument("--dpi", type=int, default=150, help="Frame DPI (default: 150)")
    parser.add_argument(
        "--start-idx",
        type=int,
        default=0,
        help="Start scenario index (for resuming, default: 0)",
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print("=" * 70)
    print("Batch BEV Closed-Loop Movie Generator")
    print("=" * 70)
    print(f"  Scenarios: {args.num_scenarios}")
    print(f"  Max steps: {args.max_steps}")
    print(f"  Output:    {args.outdir}")
    print(f"  View range: {args.view_range}m")
    print(f"  DPI: {args.dpi}, FPS: {args.fps}")
    print()

    # --- Load dependencies ---
    print("Loading Waymax dependencies...")
    t_load = time.time()

    from ..waymax_bridge.scenario_loader import load_scenarios, ScenarioLoaderConfig
    from ..waymax_bridge.env_factory import make_env
    from ..waymax_bridge.observation_extractor import (
        extract_observation,
        ObservationExtractorConfig,
    )
    from ..waymax_bridge.simulation_loop import MockLogReplayGenerator
    from ..waymax_bridge.metric_collector import MetricCollector, MetricCollectorConfig
    from ..selectors.confidence import ConfidenceSelector
    from ..config import WaymaxBridgeConfig

    print(f"  Dependencies loaded in {time.time() - t_load:.1f}s")

    # --- Configuration ---
    loader_cfg = ScenarioLoaderConfig(
        max_scenarios=args.num_scenarios,
        max_objects=args.max_objects,
        max_rg_points=args.max_rg_points,
    )
    bridge_cfg = WaymaxBridgeConfig(dynamics_model="delta", steps_per_replan=20)
    obs_cfg = ObservationExtractorConfig()
    mc_cfg = MetricCollectorConfig(
        waymax_metrics=("log_divergence", "overlap"),
    )
    bev_cfg = BEVConfig(view_range_m=args.view_range, dpi=args.dpi)
    generator = MockLogReplayGenerator(num_modes=6, horizon=50)
    selector = ConfidenceSelector()

    # --- Load scenarios ---
    print(f"\nLoading up to {args.num_scenarios} scenarios...")
    t_load = time.time()
    scenarios = []
    for sid, sim_state in load_scenarios(loader_cfg):
        scenarios.append((sid, sim_state))
    print(f"  Loaded {len(scenarios)} scenarios in {time.time() - t_load:.1f}s")

    if not scenarios:
        print("ERROR: No scenarios loaded!")
        sys.exit(1)

    # --- Render each scenario ---
    results = []
    success_count = 0
    fail_count = 0
    t_global = time.time()

    for s_idx, (sid, sim_state) in enumerate(scenarios):
        if s_idx < args.start_idx:
            continue

        print(f"\n[{s_idx + 1}/{len(scenarios)}] Rendering scenario: {sid}")

        try:
            result = render_single_scenario(
                scenario_id=sid,
                sim_state=sim_state,
                env_factory_fn=make_env,
                generator=generator,
                selector=selector,
                obs_extractor_fn=extract_observation,
                obs_cfg=obs_cfg,
                metric_collector_cls=MetricCollector,
                metric_cfg=mc_cfg,
                bridge_cfg=bridge_cfg,
                cfg=bev_cfg,
                max_steps=args.max_steps,
                outdir=args.outdir,
                scenario_idx=s_idx,
                fps=args.fps,
            )
            results.append(result)

            if result["video_path"]:
                success_count += 1
                sz_mb = os.path.getsize(result["video_path"]) / (1024 * 1024)
                print(
                    f"    OK: {result['video_name']} "
                    f"({result['steps_rendered']} steps, "
                    f"{result['elapsed_seconds']:.1f}s, "
                    f"{sz_mb:.1f}MB, "
                    f"overlaps={result['overlap_events']})"
                )
            else:
                fail_count += 1
                print(f"    WARN: Frames rendered but video stitching failed")

            # ETA
            elapsed = time.time() - t_global
            done = s_idx - args.start_idx + 1
            remaining = len(scenarios) - s_idx - 1
            if done > 0 and remaining > 0:
                eta = elapsed / done * remaining
                print(
                    f"    ETA: {eta:.0f}s ({eta/60:.1f}min) for {remaining} remaining"
                )

        except Exception as e:
            fail_count += 1
            results.append(
                {
                    "scenario_idx": s_idx,
                    "scenario_id": sid,
                    "error": str(e),
                    "video_path": None,
                }
            )
            print(f"    FAIL: {e}")

    # --- Summary ---
    total_elapsed = time.time() - t_global
    print("\n" + "=" * 70)
    print("GENERATION SUMMARY")
    print("=" * 70)
    print(f"  Scenarios processed: {len(results)}")
    print(f"  Videos generated:    {success_count}")
    print(f"  Failed:              {fail_count}")
    print(f"  Total time:          {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
    if success_count > 0:
        print(f"  Avg time/scenario:   {total_elapsed/max(len(results),1):.1f}s")

    # List overlap events
    overlap_scenarios = [r for r in results if r.get("overlap_events", 0) > 0]
    if overlap_scenarios:
        print(f"\n  Scenarios with overlaps ({len(overlap_scenarios)}):")
        for r in overlap_scenarios:
            print(
                f"    {r['scenario_id']}: {r['overlap_events']} events, max={r['max_overlap']:.3f}"
            )

    # List generated videos
    print(f"\n  Generated videos ({success_count}):")
    for r in results:
        if r.get("video_path"):
            print(f"    {r['video_name']}")

    # Save summary JSON
    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {
            "num_scenarios": args.num_scenarios,
            "max_steps": args.max_steps,
            "max_objects": args.max_objects,
            "view_range": args.view_range,
            "dpi": args.dpi,
            "fps": args.fps,
        },
        "totals": {
            "scenarios_processed": len(results),
            "videos_generated": success_count,
            "failed": fail_count,
            "total_seconds": round(total_elapsed, 1),
        },
        "results": results,
    }
    summary_path = os.path.join(args.outdir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Summary saved: {summary_path}")
    print(f"  Output dir:    {args.outdir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
