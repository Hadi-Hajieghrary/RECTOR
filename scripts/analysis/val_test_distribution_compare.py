#!/usr/bin/env python3
"""
Distributional analysis: compare validation and test splits of WOMD.

Extracts per-scenario metadata (agent count, ego speed, scenario complexity)
from both splits and compares distributions to support or refute the
distributional-shift hypothesis for the test-set generalization gap.

Usage:
    python scripts/analysis/val_test_distribution_compare.py
"""

import os
import sys
import glob
import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def extract_scenario_metadata(
    data_dir, max_scenarios=None, batch_size=64, num_workers=4
):
    """Extract per-scenario metadata directly from TFRecords using proto parsing.

    Works for both val (with GT) and test (without GT) splits.
    """
    from waymo_open_dataset.protos import scenario_pb2

    files = sorted(glob.glob(os.path.join(data_dir, "*")))
    files = [f for f in files if not f.endswith("README.md")]
    if not files:
        print(f"WARNING: No files found in {data_dir}")
        return []

    dataset = tf.data.TFRecordDataset(files, num_parallel_reads=4)

    metadata = []
    count = 0

    for raw_record in dataset:
        if max_scenarios and count >= max_scenarios:
            break

        raw_bytes = raw_record.numpy()

        # Parse the TFRecord — try augmented Example format first
        scenario = None
        try:
            example = tf.train.Example()
            example.ParseFromString(raw_bytes)
            features = example.features.feature
            if "scenario/proto" in features:
                scenario_bytes = features["scenario/proto"].bytes_list.value[0]
                scenario = scenario_pb2.Scenario()
                scenario.ParseFromString(scenario_bytes)
        except Exception:
            pass

        if scenario is None:
            try:
                scenario = scenario_pb2.Scenario()
                scenario.ParseFromString(raw_bytes)
                if not scenario.scenario_id:
                    continue
            except Exception:
                continue

        tracks = list(scenario.tracks)
        if not tracks:
            continue

        sdc_idx = scenario.sdc_track_index
        if sdc_idx < 0 or sdc_idx >= len(tracks):
            sdc_idx = 0
        sdc_track = tracks[sdc_idx]
        current_ts = scenario.current_time_index

        # SDC state at current time
        if (
            current_ts >= len(sdc_track.states)
            or not sdc_track.states[current_ts].valid
        ):
            continue
        ref_state = sdc_track.states[current_ts]
        ref_x, ref_y = ref_state.center_x, ref_state.center_y

        # Ego speed at current time
        ego_speed = np.sqrt(ref_state.velocity_x**2 + ref_state.velocity_y**2)

        # Count valid agents at current time
        n_agents = 0
        n_moving = 0
        n_nearby = 0
        for i, track in enumerate(tracks):
            if i == sdc_idx:
                continue
            if current_ts < len(track.states) and track.states[current_ts].valid:
                st = track.states[current_ts]
                n_agents += 1
                spd = np.sqrt(st.velocity_x**2 + st.velocity_y**2)
                if spd > 0.5:
                    n_moving += 1
                dist = np.sqrt((st.center_x - ref_x) ** 2 + (st.center_y - ref_y) ** 2)
                if dist < 20.0:
                    n_nearby += 1

        # Ego heading change over history
        total_heading_change = 0.0
        history_start = max(0, current_ts - 10)
        headings = []
        for t in range(history_start, current_ts + 1):
            if t < len(sdc_track.states) and sdc_track.states[t].valid:
                headings.append(sdc_track.states[t].heading)
        if len(headings) > 1:
            for j in range(1, len(headings)):
                dh = headings[j] - headings[j - 1]
                dh = abs(np.arctan2(np.sin(dh), np.cos(dh)))
                total_heading_change += dh

        # Count map lane features
        n_lanes = 0
        for mf in scenario.map_features:
            if mf.HasField("lane"):
                n_lanes += 1

        metadata.append(
            {
                "n_agents": n_agents,
                "n_moving_agents": n_moving,
                "n_nearby_agents": n_nearby,
                "ego_speed_mps": float(ego_speed),
                "ego_displacement": 0.0,  # not computed from proto for simplicity
                "ego_heading_change_rad": float(total_heading_change),
                "n_lanes": n_lanes,
            }
        )
        count += 1

        if (count % 1000) == 0:
            print(f"  Processed {count} scenarios...")

    return metadata


def compute_distribution_stats(metadata, name):
    """Compute summary statistics for a split."""
    if not metadata:
        return {}

    stats = {}
    numeric_keys = [
        "n_agents",
        "n_moving_agents",
        "n_nearby_agents",
        "ego_speed_mps",
        "ego_displacement",
        "ego_heading_change_rad",
        "n_lanes",
    ]

    for key in numeric_keys:
        values = np.array([m[key] for m in metadata])
        stats[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "median": float(np.median(values)),
            "p25": float(np.percentile(values, 25)),
            "p75": float(np.percentile(values, 75)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

    # Speed regime breakdown
    speeds = np.array([m["ego_speed_mps"] for m in metadata])
    stats["speed_regime"] = {
        "stopped_pct": float((speeds < 0.5).mean() * 100),
        "low_pct": float(((speeds >= 0.5) & (speeds < 5.0)).mean() * 100),
        "medium_pct": float(((speeds >= 5.0) & (speeds < 10.0)).mean() * 100),
        "high_pct": float(((speeds >= 10.0) & (speeds < 15.0)).mean() * 100),
        "very_high_pct": float((speeds >= 15.0).mean() * 100),
    }

    # Agent count breakdown
    agents = np.array([m["n_agents"] for m in metadata])
    stats["agent_count_regime"] = {
        "1_3_pct": float(((agents >= 1) & (agents <= 3)).mean() * 100),
        "4_6_pct": float(((agents >= 4) & (agents <= 6)).mean() * 100),
        "7_10_pct": float(((agents >= 7) & (agents <= 10)).mean() * 100),
        "10plus_pct": float((agents > 10).mean() * 100),
    }

    # Turning indicator
    heading_changes = np.array([m["ego_heading_change_rad"] for m in metadata])
    stats["maneuver_proxy"] = {
        "straight_pct": float((heading_changes < 0.1).mean() * 100),
        "gentle_turn_pct": float(
            ((heading_changes >= 0.1) & (heading_changes < 0.5)).mean() * 100
        ),
        "sharp_turn_pct": float((heading_changes >= 0.5).mean() * 100),
    }

    return stats


def ks_test(val_metadata, test_metadata):
    """Run two-sample KS test on each numeric dimension."""
    from scipy import stats as scipy_stats

    results = {}
    numeric_keys = [
        "n_agents",
        "n_moving_agents",
        "n_nearby_agents",
        "ego_speed_mps",
        "ego_displacement",
        "ego_heading_change_rad",
        "n_lanes",
    ]

    for key in numeric_keys:
        val_vals = np.array([m[key] for m in val_metadata])
        test_vals = np.array([m[key] for m in test_metadata])
        ks_stat, p_value = scipy_stats.ks_2samp(val_vals, test_vals)
        results[key] = {
            "ks_statistic": float(ks_stat),
            "p_value": float(p_value),
            "val_mean": float(np.mean(val_vals)),
            "test_mean": float(np.mean(test_vals)),
            "val_std": float(np.std(val_vals)),
            "test_std": float(np.std(test_vals)),
            "shift_direction": (
                "test_higher"
                if np.mean(test_vals) > np.mean(val_vals)
                else "test_lower"
            ),
        }

    return results


def print_comparison_table(val_stats, test_stats, ks_results):
    """Print formatted comparison table."""
    print("\n" + "=" * 90)
    print("DISTRIBUTIONAL COMPARISON: VALIDATION vs TEST")
    print("=" * 90)

    dimension_labels = {
        "n_agents": "Agent count",
        "n_moving_agents": "Moving agents",
        "n_nearby_agents": "Nearby agents (<20m)",
        "ego_speed_mps": "Ego speed (m/s)",
        "ego_displacement": "Ego displacement (m)",
        "ego_heading_change_rad": "Heading change (rad)",
        "n_lanes": "Lane count",
    }

    print(
        f"\n{'Dimension':<25} {'Val Mean':>10} {'Val Std':>10} {'Test Mean':>10} "
        f"{'Test Std':>10} {'KS Stat':>10} {'p-value':>12}"
    )
    print("-" * 90)

    for key, label in dimension_labels.items():
        ks = ks_results.get(key, {})
        print(
            f"{label:<25} {ks.get('val_mean', 0):>10.2f} {ks.get('val_std', 0):>10.2f} "
            f"{ks.get('test_mean', 0):>10.2f} {ks.get('test_std', 0):>10.2f} "
            f"{ks.get('ks_statistic', 0):>10.4f} {ks.get('p_value', 1):>12.2e}"
        )

    # Speed regime comparison
    print(f"\n{'Speed Regime':<25} {'Validation':>12} {'Test':>12}")
    print("-" * 50)
    for regime in ["stopped_pct", "low_pct", "medium_pct", "high_pct", "very_high_pct"]:
        label = regime.replace("_pct", "").replace("_", " ").title()
        v = val_stats.get("speed_regime", {}).get(regime, 0)
        t = test_stats.get("speed_regime", {}).get(regime, 0)
        print(f"{label:<25} {v:>11.1f}% {t:>11.1f}%")

    # Agent count regime comparison
    print(f"\n{'Agent Count':<25} {'Validation':>12} {'Test':>12}")
    print("-" * 50)
    for regime in ["1_3_pct", "4_6_pct", "7_10_pct", "10plus_pct"]:
        label = regime.replace("_pct", "").replace("_", "-")
        v = val_stats.get("agent_count_regime", {}).get(regime, 0)
        t = test_stats.get("agent_count_regime", {}).get(regime, 0)
        print(f"{label:<25} {v:>11.1f}% {t:>11.1f}%")

    # Maneuver proxy
    print(f"\n{'Maneuver Type':<25} {'Validation':>12} {'Test':>12}")
    print("-" * 50)
    for regime in ["straight_pct", "gentle_turn_pct", "sharp_turn_pct"]:
        label = regime.replace("_pct", "").replace("_", " ").title()
        v = val_stats.get("maneuver_proxy", {}).get(regime, 0)
        t = test_stats.get("maneuver_proxy", {}).get(regime, 0)
        print(f"{label:<25} {v:>11.1f}% {t:>11.1f}%")

    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(description="Val/Test distributional analysis")
    parser.add_argument(
        "--val_dir",
        type=str,
        default="/workspace/data/WOMD/datasets/waymo_open_dataset/motion_v_1_3_0/processed/augmented/scenario/validation_interactive",
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        default="/workspace/data/WOMD/datasets/waymo_open_dataset/motion_v_1_3_0/processed/augmented/scenario/testing_interactive",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/workspace/output/evaluation/val_test_distribution.json",
    )
    parser.add_argument("--max_scenarios", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("VAL/TEST DISTRIBUTIONAL ANALYSIS")
    print("=" * 70)

    print(f"\n[1/4] Extracting validation metadata from {args.val_dir}...")
    val_metadata = extract_scenario_metadata(
        args.val_dir,
        max_scenarios=args.max_scenarios,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"  Extracted {len(val_metadata)} validation scenarios")

    print(f"\n[2/4] Extracting test metadata from {args.test_dir}...")
    test_metadata = extract_scenario_metadata(
        args.test_dir,
        max_scenarios=args.max_scenarios,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"  Extracted {len(test_metadata)} test scenarios")

    if not val_metadata or not test_metadata:
        print("ERROR: Could not extract metadata from one or both splits.")
        return

    print(f"\n[3/4] Computing statistics...")
    val_stats = compute_distribution_stats(val_metadata, "validation")
    test_stats = compute_distribution_stats(test_metadata, "test")

    print(f"\n[4/4] Running KS tests...")
    ks_results = ks_test(val_metadata, test_metadata)

    print_comparison_table(val_stats, test_stats, ks_results)

    # Save results
    output_data = {
        "validation": {
            "n_scenarios": len(val_metadata),
            "stats": val_stats,
        },
        "test": {
            "n_scenarios": len(test_metadata),
            "stats": test_stats,
        },
        "ks_tests": ks_results,
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
