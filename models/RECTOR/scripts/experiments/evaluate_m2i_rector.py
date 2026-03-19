#!/usr/bin/env python3
"""
Evaluate M2I DenseTNT + RECTOR Rule Scoring.

This script:
1. Loads Waymo scenarios from TFRecords (or augmented data)
2. Runs M2I DenseTNT for trajectory generation (6 modes, 80 steps)
3. Scores trajectories with RECTOR's kinematic rule evaluator
4. Optionally runs full rule evaluation via RuleExecutor (28 Waymo rules)
5. Computes ADE/FDE metrics against ground truth
6. Outputs a comparison report

Usage:
    # Basic evaluation (kinematic rules only)
    python evaluate_m2i_rector.py \\
        --data_dir /workspace/data/WOMD/datasets/waymo_open_dataset/motion_v_1_3_0/processed/augmented/scenario/validation_interactive \\
        --output_dir /workspace/output/evaluation/m2i_rector

    # Full rule evaluation (all 28 Waymo rules, slower)
    python evaluate_m2i_rector.py \\
        --data_dir /path/to/data \\
        --full_rule_eval \\
        --output_dir /workspace/output/evaluation/m2i_rector

    # Quick test (limit scenarios)
    python evaluate_m2i_rector.py \\
        --data_dir /path/to/data \\
        --max_scenarios 10 \\
        --output_dir /workspace/output/evaluation/m2i_rector_test
"""

import argparse
import glob
import json
import logging
import os
import sys
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

RECTOR_SCRIPTS = Path(__file__).parent.parent
sys.path.insert(0, str(RECTOR_SCRIPTS))
sys.path.insert(0, str(RECTOR_SCRIPTS / "lib"))
sys.path.insert(0, "/workspace/data/WOMD")

# TensorFlow for TFRecords (CPU only)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

tf.config.set_visible_devices([], "GPU")

# Waymo protos
from waymo_open_dataset.protos import scenario_pb2

# M2I + RECTOR rule selector
from lib.m2i_rule_selector import M2IRuleSelector, GenerationResult
from lib.m2i_trajectory_generator import M2ITrajectoryGenerator

# Rule constants
from waymo_rule_eval.rules.rule_constants import (
    NUM_RULES,
    RULE_IDS,
    TIERS,
    TIER_0_SAFETY,
    TIER_1_LEGAL,
    TIER_2_ROAD,
    TIER_3_COMFORT,
    TIER_BY_NAME,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(message)s",
)
log = logging.getLogger(__name__)

# Try to import full rule evaluation framework (optional)
try:
    from waymo_rule_eval.pipeline.rule_executor import RuleExecutor, ScenarioResult
    from waymo_rule_eval.core.context import (
        Agent,
        EgoState,
        MapContext,
        MapSignals,
        ScenarioContext,
    )
    from waymo_rule_eval.utils.constants import (
        AGENT_TYPE_MAP,
        DEFAULT_DT_S,
        DEFAULT_EGO_LENGTH,
        DEFAULT_EGO_WIDTH,
        SIGNAL_STATE_UNKNOWN,
        SIGNAL_STATE_STOP,
        SIGNAL_STATE_CAUTION,
        SIGNAL_STATE_GO,
    )

    FULL_EVAL_AVAILABLE = True
except ImportError:
    FULL_EVAL_AVAILABLE = False
    log.warning(
        "Full rule evaluation framework not available. Only kinematic rules will run."
    )


def iter_scenarios_from_tfrecords(
    data_dir: str,
    max_scenarios: int = None,
):
    """
    Iterate over Waymo scenarios from TFRecord files.

    Supports two formats:
    1. Augmented TFRecords (with 'scenario/proto' feature)
    2. Raw Waymo motion TFRecords (with 'scenario/id' feature)

    Yields:
        (scenario_id, scenario_proto)
    """
    tfrecord_files = sorted(
        glob.glob(os.path.join(data_dir, "*.tfrecord*"))
        + glob.glob(os.path.join(data_dir, "**/*.tfrecord*"), recursive=True)
    )
    # Deduplicate
    tfrecord_files = sorted(set(tfrecord_files))

    if not tfrecord_files:
        log.error(f"No TFRecord files found in {data_dir}")
        return

    log.info(f"Found {len(tfrecord_files)} TFRecord files")

    count = 0
    for tfpath in tfrecord_files:
        if max_scenarios and count >= max_scenarios:
            break

        dataset = tf.data.TFRecordDataset(tfpath)
        for record in dataset:
            if max_scenarios and count >= max_scenarios:
                break

            try:
                raw = record.numpy()

                # Try augmented format first
                try:
                    example = tf.train.Example.FromString(raw)
                    features = example.features.feature

                    if "scenario/proto" in features:
                        proto_bytes = features["scenario/proto"].bytes_list.value[0]
                        scenario_id = (
                            features["scenario/id"].bytes_list.value[0].decode("utf-8")
                        )
                        scenario = scenario_pb2.Scenario()
                        scenario.ParseFromString(proto_bytes)
                        yield scenario_id, scenario
                        count += 1
                        continue
                except Exception:
                    pass

                # Try raw Waymo format (full Scenario proto)
                scenario = scenario_pb2.Scenario()
                scenario.ParseFromString(raw)
                yield scenario.scenario_id, scenario
                count += 1

            except Exception as e:
                log.debug(f"Failed to parse record: {e}")
                continue

    log.info(f"Yielded {count} scenarios")


def extract_gt_future_trajectory(scenario, agent_idx: int = 0) -> Optional[np.ndarray]:
    """
    Extract ground truth future trajectory for an agent.

    Args:
        scenario: Waymo Scenario proto
        agent_idx: 0 = SDC

    Returns:
        [T, 2] array of future (x, y) positions in world coords,
        or None if not available.
    """
    tracks = list(scenario.tracks)
    if not tracks:
        return None

    # Find the target track
    # NOTE: sdc_track_index is an array index, NOT a track object ID
    if agent_idx == 0:
        sdc_idx = scenario.sdc_track_index
        if 0 <= sdc_idx < len(tracks):
            target_track = tracks[sdc_idx]
        elif tracks:
            target_track = tracks[0]
        else:
            target_track = None
    else:
        if agent_idx < len(tracks):
            target_track = tracks[agent_idx]
        else:
            return None

    current_ts = scenario.current_time_index
    states = list(target_track.states)
    n_future = len(states) - (current_ts + 1)

    if n_future <= 0:
        return None

    future_xy = []
    future_valid = []
    for t in range(current_ts + 1, len(states)):
        state = states[t]
        if state.valid:
            future_xy.append([state.center_x, state.center_y])
            future_valid.append(True)
        else:
            future_xy.append([0.0, 0.0])
            future_valid.append(False)

    future_xy = np.array(future_xy, dtype=np.float32)
    future_valid = np.array(future_valid, dtype=bool)

    if not np.any(future_valid):
        return None

    return future_xy  # [T, 2]


def extract_agent_futures(scenario) -> Optional[np.ndarray]:
    """
    Extract future trajectories for all non-ego agents.

    Returns:
        [A, T, 2] array or None
    """
    tracks = list(scenario.tracks)
    # sdc_track_index is an array index, NOT a track object ID
    sdc_idx = scenario.sdc_track_index
    current_ts = scenario.current_time_index

    agent_trajs = []
    for i, track in enumerate(tracks):
        if i == sdc_idx:
            continue

        states = list(track.states)
        n_future = len(states) - (current_ts + 1)
        if n_future <= 0:
            continue

        xy = []
        any_valid = False
        for t in range(current_ts + 1, len(states)):
            state = states[t]
            if state.valid:
                xy.append([state.center_x, state.center_y])
                any_valid = True
            else:
                # Forward fill
                if xy:
                    xy.append(xy[-1])
                else:
                    xy.append([0.0, 0.0])

        if any_valid:
            agent_trajs.append(np.array(xy, dtype=np.float32))

    if not agent_trajs:
        return None

    # Pad to same length
    max_len = max(len(t) for t in agent_trajs)
    padded = np.zeros((len(agent_trajs), max_len, 2), dtype=np.float32)
    for i, t in enumerate(agent_trajs):
        padded[i, : len(t)] = t

    return padded


@dataclass
class ScenarioMetrics:
    """Metrics for a single scenario."""

    scenario_id: str = ""
    # Trajectory quality (best mode via ADE)
    best_mode_ade: float = float("inf")  # meters
    best_mode_fde: float = float("inf")  # meters
    # All modes
    all_modes_ade: np.ndarray = field(default_factory=lambda: np.array([]))  # [M]
    all_modes_fde: np.ndarray = field(default_factory=lambda: np.array([]))  # [M]
    # M2I picks
    m2i_best_ade: float = float("inf")
    m2i_best_fde: float = float("inf")
    # Rule-reranked picks
    rule_best_ade: float = float("inf")
    rule_best_fde: float = float("inf")
    # Rule violations
    tier_violations: Dict[str, int] = field(default_factory=dict)
    # Timing
    total_ms: float = 0.0
    m2i_ms: float = 0.0
    rule_eval_ms: float = 0.0


def compute_metrics(
    result: GenerationResult,
    gt_future: np.ndarray,  # [T_gt, 2] world coords
) -> ScenarioMetrics:
    """
    Compute ADE/FDE metrics comparing generated trajectories to GT.

    Handles horizon mismatch: M2I produces 80 steps (8s), GT may be different.
    """
    metrics = ScenarioMetrics(scenario_id=result.scenario_id)

    traj_world = result.trajectories_world  # [M, 80, 2]
    M, T_pred, _ = traj_world.shape
    T_gt = len(gt_future)
    T = min(T_pred, T_gt)  # common horizon

    if T == 0:
        return metrics

    # Per-mode ADE and FDE
    ade_per_mode = np.zeros(M)
    fde_per_mode = np.zeros(M)
    for m in range(M):
        dists = np.linalg.norm(traj_world[m, :T] - gt_future[:T], axis=-1)  # [T]
        ade_per_mode[m] = dists.mean()
        fde_per_mode[m] = dists[-1]

    metrics.all_modes_ade = ade_per_mode
    metrics.all_modes_fde = fde_per_mode

    # Oracle best (min ADE)
    best_idx = int(np.argmin(ade_per_mode))
    metrics.best_mode_ade = ade_per_mode[best_idx]
    metrics.best_mode_fde = fde_per_mode[best_idx]

    # M2I top pick
    metrics.m2i_best_ade = ade_per_mode[result.best_m2i_idx]
    metrics.m2i_best_fde = fde_per_mode[result.best_m2i_idx]

    # Rule-reranked pick
    metrics.rule_best_ade = ade_per_mode[result.best_rule_idx]
    metrics.rule_best_fde = fde_per_mode[result.best_rule_idx]

    # Rule tier violations (for rule-selected trajectory)
    if result.rule_violations is not None:
        for tier_name in TIERS:
            violations = result.rule_violations.violations[result.best_rule_idx]
            tier_rules = TIER_BY_NAME.get(tier_name, [])
            from waymo_rule_eval.rules.rule_constants import RULE_INDEX_MAP

            tier_viols = sum(
                violations[RULE_INDEX_MAP[r]] > 0.5
                for r in tier_rules
                if r in RULE_INDEX_MAP
            )
            metrics.tier_violations[tier_name] = int(tier_viols)

    # Timing
    metrics.total_ms = sum(result.timing_ms.values())
    metrics.m2i_ms = result.timing_ms.get("m2i_generate_ms", 0)
    metrics.rule_eval_ms = result.timing_ms.get("rule_eval_ms", 0)

    return metrics


def run_evaluation(args):
    """Main evaluation loop."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 70)
    log.info("M2I DenseTNT + RECTOR Rule Evaluation")
    log.info("=" * 70)

    # --- Load generator ---
    log.info("Loading M2I DenseTNT model...")
    generator = M2IRuleSelector(
        model_path=args.m2i_model,
        device=args.device,
    )
    generator.load()

    # --- Iterate scenarios ---
    all_metrics: List[ScenarioMetrics] = []
    start_time = time.time()

    for scenario_idx, (scenario_id, scenario) in enumerate(
        iter_scenarios_from_tfrecords(args.data_dir, max_scenarios=args.max_scenarios)
    ):
        try:
            # Get GT future for metrics
            gt_future = extract_gt_future_trajectory(scenario, agent_idx=0)
            if gt_future is None:
                log.debug(f"Scenario {scenario_id}: No GT future, skipping")
                continue

            # Get other agent futures for collision evaluation
            agent_futures = (
                extract_agent_futures(scenario) if args.use_agent_futures else None
            )

            # Run M2I + RECTOR
            result = generator.evaluate_scenario(
                scenario,
                target_agent_idx=0,
                agent_future_trajs=agent_futures,
            )

            if result is None:
                log.debug(f"Scenario {scenario_id}: M2I inference failed, skipping")
                continue

            # Compute metrics
            metrics = compute_metrics(result, gt_future)
            all_metrics.append(metrics)

            # Progress
            if (scenario_idx + 1) % args.log_every == 0:
                n = len(all_metrics)
                avg_ade = np.mean([m.best_mode_ade for m in all_metrics])
                avg_fde = np.mean([m.best_mode_fde for m in all_metrics])
                rule_ade = np.mean([m.rule_best_ade for m in all_metrics])
                elapsed = time.time() - start_time
                log.info(
                    f"  [{scenario_idx + 1}] {n} valid | "
                    f"Oracle minADE={avg_ade:.3f}m minFDE={avg_fde:.3f}m | "
                    f"Rule-picked ADE={rule_ade:.3f}m | "
                    f"{elapsed:.1f}s"
                )

        except Exception as e:
            log.warning(f"Scenario {scenario_id}: Error - {e}")
            continue

    elapsed = time.time() - start_time

    if not all_metrics:
        log.error("No valid scenarios processed!")
        return

    # --- Aggregate Results ---
    n = len(all_metrics)

    results = {
        "config": {
            "data_dir": args.data_dir,
            "m2i_model": args.m2i_model,
            "device": args.device,
            "max_scenarios": args.max_scenarios,
            "use_agent_futures": args.use_agent_futures,
        },
        "summary": {
            "total_scenarios": n,
            "elapsed_time_s": elapsed,
            "scenarios_per_second": n / elapsed if elapsed > 0 else 0,
        },
        "metrics": {},
        "rule_violations": {},
        "timing": {},
    }

    # Trajectory quality
    oracle_ade = [m.best_mode_ade for m in all_metrics]
    oracle_fde = [m.best_mode_fde for m in all_metrics]
    m2i_ade = [m.m2i_best_ade for m in all_metrics]
    m2i_fde = [m.m2i_best_fde for m in all_metrics]
    rule_ade = [m.rule_best_ade for m in all_metrics]
    rule_fde = [m.rule_best_fde for m in all_metrics]

    results["metrics"] = {
        "oracle_minADE_mean": float(np.mean(oracle_ade)),
        "oracle_minADE_std": float(np.std(oracle_ade)),
        "oracle_minFDE_mean": float(np.mean(oracle_fde)),
        "oracle_minFDE_std": float(np.std(oracle_fde)),
        "oracle_miss_rate_2m": float(np.mean(np.array(oracle_fde) > 2.0)) * 100,
        "m2i_best_ADE_mean": float(np.mean(m2i_ade)),
        "m2i_best_ADE_std": float(np.std(m2i_ade)),
        "m2i_best_FDE_mean": float(np.mean(m2i_fde)),
        "m2i_best_FDE_std": float(np.std(m2i_fde)),
        "m2i_miss_rate_2m": float(np.mean(np.array(m2i_fde) > 2.0)) * 100,
        "rule_picked_ADE_mean": float(np.mean(rule_ade)),
        "rule_picked_ADE_std": float(np.std(rule_ade)),
        "rule_picked_FDE_mean": float(np.mean(rule_fde)),
        "rule_picked_FDE_std": float(np.std(rule_fde)),
        "rule_miss_rate_2m": float(np.mean(np.array(rule_fde) > 2.0)) * 100,
    }

    # Rule violations (for rule-picked trajectories)
    for tier_name in TIERS:
        viols = [m.tier_violations.get(tier_name, 0) for m in all_metrics]
        results["rule_violations"][tier_name] = {
            "total_violations": int(np.sum(viols)),
            "scenarios_with_violations": int(np.sum(np.array(viols) > 0)),
            "violation_rate_pct": float(np.mean(np.array(viols) > 0)) * 100,
        }

    # Timing
    results["timing"] = {
        "avg_total_ms": float(np.mean([m.total_ms for m in all_metrics])),
        "avg_m2i_ms": float(np.mean([m.m2i_ms for m in all_metrics])),
        "avg_rule_eval_ms": float(np.mean([m.rule_eval_ms for m in all_metrics])),
    }

    # --- Print Report ---
    print("\n" + "=" * 70)
    print("M2I DenseTNT + RECTOR Rule Evaluation Results")
    print("=" * 70)
    print(f"\n  Scenarios evaluated: {n}")
    print(f"  Time elapsed: {elapsed:.1f}s ({n/elapsed:.1f} scen/s)")
    print(f"\n  --- Trajectory Quality (world coords) ---")
    print(
        f"  Oracle minADE:  {results['metrics']['oracle_minADE_mean']:.3f} ± "
        f"{results['metrics']['oracle_minADE_std']:.3f} m"
    )
    print(
        f"  Oracle minFDE:  {results['metrics']['oracle_minFDE_mean']:.3f} ± "
        f"{results['metrics']['oracle_minFDE_std']:.3f} m"
    )
    print(f"  Oracle Miss@2m: {results['metrics']['oracle_miss_rate_2m']:.1f}%")
    print(
        f"  M2I-picked ADE: {results['metrics']['m2i_best_ADE_mean']:.3f} ± "
        f"{results['metrics']['m2i_best_ADE_std']:.3f} m"
    )
    print(
        f"  M2I-picked FDE: {results['metrics']['m2i_best_FDE_mean']:.3f} ± "
        f"{results['metrics']['m2i_best_FDE_std']:.3f} m"
    )
    print(f"  M2I Miss@2m:    {results['metrics']['m2i_miss_rate_2m']:.1f}%")
    print(
        f"  Rule-picked ADE:{results['metrics']['rule_picked_ADE_mean']:.3f} ± "
        f"{results['metrics']['rule_picked_ADE_std']:.3f} m"
    )
    print(
        f"  Rule-picked FDE:{results['metrics']['rule_picked_FDE_mean']:.3f} ± "
        f"{results['metrics']['rule_picked_FDE_std']:.3f} m"
    )
    print(f"  Rule Miss@2m:   {results['metrics']['rule_miss_rate_2m']:.1f}%")
    print(f"\n  --- Rule Violations (rule-picked trajectory) ---")
    for tier_name in TIERS:
        tv = results["rule_violations"][tier_name]
        print(
            f"  {tier_name:>8}: {tv['total_violations']} violations in "
            f"{tv['scenarios_with_violations']}/{n} scenarios "
            f"({tv['violation_rate_pct']:.1f}%)"
        )
    print(f"\n  --- Timing ---")
    print(f"  Avg total:     {results['timing']['avg_total_ms']:.1f} ms/scenario")
    print(f"  Avg M2I:       {results['timing']['avg_m2i_ms']:.1f} ms/scenario")
    print(f"  Avg rule eval: {results['timing']['avg_rule_eval_ms']:.1f} ms/scenario")
    print("=" * 70)

    # --- Save ---
    results_path = output_dir / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Results saved to {results_path}")

    # Save per-scenario details
    per_scenario = []
    for m in all_metrics:
        per_scenario.append(
            {
                "scenario_id": m.scenario_id,
                "oracle_minADE": float(m.best_mode_ade),
                "oracle_minFDE": float(m.best_mode_fde),
                "m2i_best_ADE": float(m.m2i_best_ade),
                "m2i_best_FDE": float(m.m2i_best_fde),
                "rule_best_ADE": float(m.rule_best_ade),
                "rule_best_FDE": float(m.rule_best_fde),
                "tier_violations": m.tier_violations,
                "total_ms": float(m.total_ms),
            }
        )
    details_path = output_dir / "per_scenario_details.json"
    with open(details_path, "w") as f:
        json.dump(per_scenario, f, indent=2)
    log.info(f"Per-scenario details saved to {details_path}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate M2I DenseTNT + RECTOR Rule Scoring",
    )
    p.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing Waymo TFRecord files",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="/workspace/output/evaluation/m2i_rector",
        help="Output directory for results",
    )
    p.add_argument(
        "--m2i_model",
        type=str,
        default="/workspace/models/pretrained/m2i/models/densetnt/model.24.bin",
        help="Path to pretrained DenseTNT model",
    )
    p.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")
    p.add_argument(
        "--max_scenarios",
        type=int,
        default=None,
        help="Max scenarios to evaluate (None = all)",
    )
    p.add_argument(
        "--use_agent_futures",
        action="store_true",
        default=False,
        help="Use GT agent futures for collision evaluation",
    )
    p.add_argument(
        "--full_rule_eval",
        action="store_true",
        default=False,
        help="Run full 28-rule evaluation via RuleExecutor (slower)",
    )
    p.add_argument(
        "--log_every", type=int, default=10, help="Log progress every N scenarios"
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(args)
