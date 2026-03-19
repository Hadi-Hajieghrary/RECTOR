#!/usr/bin/env python3
"""
Challenging BEV Closed-Loop Movie Generator
=============================================

Mines WOMD validation shards for scenarios with complex road infrastructure
(intersections, crosswalks, stop signs, speed bumps, dense lane networks),
then renders closed-loop BEV movies with full road geometry.

Usage:
    cd /workspace/scripts
    XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.25 \
    python -m simulation_engine.viz.generate_bev_challenging \
        --target-movies 50 \
        --outdir /workspace/output/closedloop/videos

Strategy:
    1. Scan multiple shards (spread across the 150-shard validation set)
    2. Score each scenario by road complexity (crosswalks, stop signs,
       intersection density, lane count, agent count)
    3. Pick top-N most complex scenarios
    4. Render with full roadgraph (20K+ points)
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .bev_rollout import BEVConfig, render_frame, _extract_road_polylines


# ── Scenario scoring ────────────────────────────────────────────────────


def score_scenario_complexity(proto) -> Dict[str, float]:
    """
    Score a scenario by infrastructure complexity.

    Returns dict of sub-scores:
      - crosswalks: count of crosswalk features
      - stop_signs: count of stop sign features
      - speed_bumps: count of speed bump features
      - lane_count: total number of lane polylines
      - road_lines: road boundary/marking features
      - total_map_features: total map features
      - agent_count: number of tracks
      - traffic_lights: number of dynamic traffic light states
      - composite: weighted total score
    """
    crosswalks = 0
    stop_signs = 0
    speed_bumps = 0
    lanes = 0
    road_lines = 0
    road_edges = 0

    for mf in proto.map_features:
        if mf.HasField("crosswalk"):
            crosswalks += 1
        elif mf.HasField("stop_sign"):
            stop_signs += 1
        elif mf.HasField("speed_bump"):
            speed_bumps += 1
        elif mf.HasField("lane"):
            lanes += 1
        elif mf.HasField("road_line"):
            road_lines += 1
        elif mf.HasField("road_edge"):
            road_edges += 1

    total_map = len(proto.map_features)
    agent_count = len(proto.tracks)

    # Traffic light states
    tl_count = 0
    for dms in proto.dynamic_map_states:
        tl_count += len(dms.lane_states)
    # Average per timestep
    tl_avg = tl_count / max(len(proto.dynamic_map_states), 1)

    # Composite score: heavily weight crosswalks, stop signs, traffic lights
    composite = (
        crosswalks * 10.0
        + stop_signs * 8.0
        + speed_bumps * 5.0
        + tl_avg * 3.0
        + min(lanes, 100) * 0.5  # cap to avoid highway bias
        + min(agent_count, 30) * 1.0  # more agents = more interesting
        + road_lines * 0.2
        + road_edges * 0.1
    )

    return {
        "crosswalks": crosswalks,
        "stop_signs": stop_signs,
        "speed_bumps": speed_bumps,
        "lane_count": lanes,
        "road_lines": road_lines,
        "road_edges": road_edges,
        "total_map_features": total_map,
        "agent_count": agent_count,
        "traffic_light_avg": round(tl_avg, 1),
        "composite": round(composite, 1),
    }


def mine_complex_scenarios(
    data_root: str,
    split_dir: str,
    target_count: int = 50,
    shards_to_scan: int = 30,
    scenarios_per_shard: int = 20,
) -> List[Tuple[str, "Scenario", Dict]]:
    """
    Mine complex scenarios from WOMD TFRecord shards.

    Scans shards spread evenly across the validation set,
    scores each scenario, and returns the top-N by complexity.
    """
    import tensorflow as tf
    from waymo_open_dataset.protos import scenario_pb2

    pattern = os.path.join(data_root, split_dir, "*.tfrecord*")
    shard_paths = sorted(glob.glob(pattern))

    if not shard_paths:
        raise FileNotFoundError(f"No TFRecords at {pattern}")

    print(f"  Found {len(shard_paths)} shards")

    # Spread shard selection evenly across dataset
    n_shards = min(shards_to_scan, len(shard_paths))
    step = max(1, len(shard_paths) // n_shards)
    selected_shards = shard_paths[::step][:n_shards]
    print(f"  Scanning {len(selected_shards)} shards (step={step})")

    all_candidates = []

    for shard_idx, shard_path in enumerate(selected_shards):
        shard_name = os.path.basename(shard_path)
        ds = tf.data.TFRecordDataset(shard_path)

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

            scores = score_scenario_complexity(scenario)
            all_candidates.append((scenario.scenario_id, scenario, scores))
            count += 1

        print(
            f"    [{shard_idx+1}/{len(selected_shards)}] "
            f"{shard_name}: {count} scenarios scanned"
        )

    # Sort by composite score (descending)
    all_candidates.sort(key=lambda x: x[2]["composite"], reverse=True)

    # Take top-N
    top = all_candidates[:target_count]

    print(f"\n  Top {len(top)} scenarios by complexity:")
    print(
        f"  {'ID':>20s}  {'Score':>7s}  {'XW':>3s}  {'SS':>3s}  {'TL':>4s}  {'Lanes':>5s}  {'Agents':>6s}"
    )
    for sid, _, sc in top[:10]:
        print(
            f"  {sid[:20]:>20s}  {sc['composite']:>7.1f}  "
            f"{sc['crosswalks']:>3d}  {sc['stop_signs']:>3d}  "
            f"{sc['traffic_light_avg']:>4.1f}  {sc['lane_count']:>5d}  "
            f"{sc['agent_count']:>6d}"
        )
    if len(top) > 10:
        print(f"  ... and {len(top)-10} more")

    return top


# ── Rendering ───────────────────────────────────────────────────────────


def _stitch_video(frame_dir: str, outpath: str, fps: int = 10) -> bool:
    """Stitch PNG frames into MP4."""
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


def render_scenario_from_proto(
    scenario_id: str,
    proto,
    scenario_idx: int,
    cfg: BEVConfig,
    max_steps: int,
    max_objects: int,
    max_rg_points: int,
    outdir: str,
    fps: int = 10,
) -> Dict:
    """
    Convert proto → SimulatorState → closed-loop rollout → BEV movie.
    """
    import jax.numpy as jnp
    from ..waymax_bridge.scenario_loader import scenario_proto_to_simulator_state
    from ..waymax_bridge.env_factory import make_env
    from ..waymax_bridge.observation_extractor import (
        extract_observation,
        ObservationExtractorConfig,
    )
    from ..waymax_bridge.simulation_loop import MockLogReplayGenerator
    from ..waymax_bridge.action_converter import trajectory_to_action
    from ..waymax_bridge.metric_collector import MetricCollector, MetricCollectorConfig
    from ..selectors.confidence import ConfidenceSelector
    from ..config import WaymaxBridgeConfig

    t_start = time.time()

    # Convert proto to SimulatorState with full road geometry
    sim_state = scenario_proto_to_simulator_state(
        proto,
        max_objects=max_objects,
        max_rg_points=max_rg_points,
    )

    # Create environment
    bridge_cfg = WaymaxBridgeConfig(dynamics_model="delta", steps_per_replan=20)
    env, state = make_env(sim_state, dynamics_model=bridge_cfg.dynamics_model)

    # Setup
    obs_cfg = ObservationExtractorConfig()
    mc_cfg = MetricCollectorConfig(waymax_metrics=("log_divergence", "overlap"))
    mc = MetricCollector(mc_cfg)
    mc.reset()
    generator = MockLogReplayGenerator(num_modes=6, horizon=50)
    selector = ConfidenceSelector()

    # Extract road geometry
    road_polylines, road_types = _extract_road_polylines(state)

    # Logged trajectory
    sdc_idx = int(jnp.argmax(state.object_metadata.is_sdc))
    log_x = np.asarray(state.log_trajectory.x[sdc_idx])
    log_y = np.asarray(state.log_trajectory.y[sdc_idx])
    logged_xy = np.stack([log_x, log_y], axis=-1)

    # Frame output
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
        mc.step(state)

        # Record ego
        t_now = int(state.timestep)
        ego_x = float(np.asarray(state.sim_trajectory.x[sdc_idx, t_now]))
        ego_y = float(np.asarray(state.sim_trajectory.y[sdc_idx, t_now]))
        ego_yaw_now = float(np.asarray(state.sim_trajectory.yaw[sdc_idx, t_now]))
        exec_traj.append([ego_x, ego_y])

        # Agents
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

        # Overlap
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

        n_agents = int(agent_valid.sum())
        metrics_text = (
            f"Scenario: {scenario_id[:20]}\n"
            f"step={step_i+1}/{total_steps}  agents={n_agents}\n"
            f"roads={len(road_polylines)} segs\n"
            f"overlap={overlap_val:.3f}"
        )

        render_frame(
            ax=ax,
            cfg=cfg,
            ego_x=ego_x,
            ego_y=ego_y,
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

    try:
        final_metrics = mc.finalise()
    except Exception:
        final_metrics = {}

    elapsed = time.time() - t_start

    # Stitch video
    safe_id = scenario_id.replace("/", "_").replace(" ", "_")[:32]
    video_name = f"scenario_{scenario_idx:03d}_{safe_id}.mp4"
    video_path = os.path.join(outdir, video_name)
    video_ok = _stitch_video(frame_dir, video_path, fps=fps)

    if video_ok:
        import shutil

        shutil.rmtree(frame_dir, ignore_errors=True)

    return {
        "scenario_idx": scenario_idx,
        "scenario_id": scenario_id,
        "steps_rendered": actual_steps,
        "total_steps": total_steps,
        "overlap_events": overlap_events,
        "max_overlap": float(max_overlap),
        "video_path": video_path if video_ok else None,
        "video_name": video_name if video_ok else None,
        "elapsed_seconds": round(elapsed, 1),
        "road_segments": len(road_polylines),
        "metrics": {
            k: float(v) if np.isfinite(v) else None for k, v in final_metrics.items()
        },
    }


# ── Main ────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Generate BEV movies for challenging WOMD scenarios"
    )
    parser.add_argument(
        "--target-movies",
        type=int,
        default=50,
        help="Number of movies to generate (default: 50)",
    )
    parser.add_argument(
        "--outdir",
        default="/workspace/output/closedloop/videos",
        help="Output directory",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=80,
        help="Max simulation steps (default: 80 = 8s)",
    )
    parser.add_argument(
        "--max-objects",
        type=int,
        default=64,
        help="Max objects per scenario (default: 64)",
    )
    parser.add_argument(
        "--max-rg-points",
        type=int,
        default=20000,
        help="Max roadgraph points (default: 20000)",
    )
    parser.add_argument(
        "--shards-to-scan",
        type=int,
        default=30,
        help="Number of shards to scan for mining (default: 30)",
    )
    parser.add_argument(
        "--scenarios-per-shard",
        type=int,
        default=30,
        help="Scenarios to examine per shard (default: 30)",
    )
    parser.add_argument(
        "--view-range",
        type=float,
        default=60.0,
        help="BEV half-side in metres (default: 60)",
    )
    parser.add_argument("--dpi", type=int, default=150, help="Frame DPI (default: 150)")
    parser.add_argument("--fps", type=int, default=10, help="Video FPS (default: 10)")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    data_root = (
        "/workspace/data/WOMD/datasets/waymo_open_dataset"
        "/motion_v_1_3_0/processed/augmented/scenario"
    )
    split_dir = "validation_interactive"

    print("=" * 70)
    print("Challenging BEV Closed-Loop Movie Generator")
    print("=" * 70)
    print(f"  Target movies:   {args.target_movies}")
    print(f"  Max objects:     {args.max_objects}")
    print(f"  Max RG points:   {args.max_rg_points}")
    print(f"  View range:      {args.view_range}m")
    print(f"  Shards to scan:  {args.shards_to_scan}")
    print(f"  Scen/shard:      {args.scenarios_per_shard}")
    print(f"  Output:          {args.outdir}")
    print()

    # ── Phase 1: Mine complex scenarios ──────────────────────────────
    print("Phase 1: Mining complex scenarios...")
    t_mine = time.time()
    candidates = mine_complex_scenarios(
        data_root=data_root,
        split_dir=split_dir,
        target_count=args.target_movies,
        shards_to_scan=args.shards_to_scan,
        scenarios_per_shard=args.scenarios_per_shard,
    )
    print(
        f"\nMining complete: {len(candidates)} scenarios in {time.time()-t_mine:.1f}s"
    )

    if len(candidates) == 0:
        print("ERROR: No scenarios found!")
        sys.exit(1)

    # ── Phase 2: Render movies ───────────────────────────────────────
    print(f"\nPhase 2: Rendering {len(candidates)} BEV movies...")
    bev_cfg = BEVConfig(view_range_m=args.view_range, dpi=args.dpi)

    results = []
    success = 0
    fail = 0
    t_render = time.time()

    for idx, (sid, proto, scores) in enumerate(candidates):
        xw = scores["crosswalks"]
        ss = scores["stop_signs"]
        tl = scores["traffic_light_avg"]
        sc = scores["composite"]
        print(
            f"\n[{idx+1}/{len(candidates)}] {sid}  "
            f"(score={sc:.0f}, xwalks={xw}, stops={ss}, TL={tl:.0f})"
        )

        try:
            result = render_scenario_from_proto(
                scenario_id=sid,
                proto=proto,
                scenario_idx=idx,
                cfg=bev_cfg,
                max_steps=args.max_steps,
                max_objects=args.max_objects,
                max_rg_points=args.max_rg_points,
                outdir=args.outdir,
                fps=args.fps,
            )
            result["complexity_scores"] = scores
            results.append(result)

            if result["video_path"]:
                success += 1
                sz_mb = os.path.getsize(result["video_path"]) / (1024 * 1024)
                print(
                    f"    OK: {result['video_name']} "
                    f"({result['steps_rendered']} steps, "
                    f"{result['elapsed_seconds']:.1f}s, "
                    f"{sz_mb:.1f}MB, "
                    f"segs={result['road_segments']}, "
                    f"overlaps={result['overlap_events']})"
                )
            else:
                fail += 1
                print("    WARN: video stitching failed")

            # ETA
            done = idx + 1
            remaining = len(candidates) - done
            if done > 0 and remaining > 0:
                elapsed = time.time() - t_render
                eta = elapsed / done * remaining
                print(f"    ETA: {eta:.0f}s ({eta/60:.1f}min)")

        except Exception as e:
            fail += 1
            results.append(
                {
                    "scenario_idx": idx,
                    "scenario_id": sid,
                    "error": str(e),
                    "video_path": None,
                    "complexity_scores": scores,
                }
            )
            print(f"    FAIL: {e}")
            import traceback

            traceback.print_exc()

    # ── Summary ──────────────────────────────────────────────────────
    total_time = time.time() - t_render
    print("\n" + "=" * 70)
    print("GENERATION SUMMARY")
    print("=" * 70)
    print(f"  Scenarios rendered: {len(results)}")
    print(f"  Videos generated:   {success}")
    print(f"  Failed:             {fail}")
    print(f"  Render time:        {total_time:.1f}s ({total_time/60:.1f}min)")

    # Score distribution
    comp_scores = [
        r.get("complexity_scores", {}).get("composite", 0)
        for r in results
        if r.get("video_path")
    ]
    if comp_scores:
        print(
            f"\n  Complexity scores: min={min(comp_scores):.0f} max={max(comp_scores):.0f} mean={np.mean(comp_scores):.0f}"
        )

    # Infrastructure stats
    xw_total = sum(
        r.get("complexity_scores", {}).get("crosswalks", 0)
        for r in results
        if r.get("video_path")
    )
    ss_total = sum(
        r.get("complexity_scores", {}).get("stop_signs", 0)
        for r in results
        if r.get("video_path")
    )
    print(f"  Total crosswalks across videos: {xw_total}")
    print(f"  Total stop signs across videos: {ss_total}")

    print(f"\n  Generated videos ({success}):")
    for r in results:
        if r.get("video_path"):
            sc = r.get("complexity_scores", {})
            print(f"    {r['video_name']}  (score={sc.get('composite',0):.0f})")

    # Save summary
    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {
            "target_movies": args.target_movies,
            "max_steps": args.max_steps,
            "max_objects": args.max_objects,
            "max_rg_points": args.max_rg_points,
            "shards_scanned": args.shards_to_scan,
        },
        "totals": {
            "videos_generated": success,
            "failed": fail,
            "render_time_seconds": round(total_time, 1),
        },
        "results": results,
    }
    summary_path = os.path.join(args.outdir, "challenging_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Summary: {summary_path}")
    print(f"  Output:  {args.outdir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
