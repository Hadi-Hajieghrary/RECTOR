#!/usr/bin/env python3
"""
Pre-process TFRecords into cached .pt files for fast training.

Parses protobuf once, saves PyTorch tensors to disk.
Subsequent training loads tensors directly — no protobuf overhead.

Typical speedup: 3-5x per epoch (eliminates ~80% of data loading time).

Usage:
    python training/preprocess_cache.py
    python training/preprocess_cache.py --split val
    python training/preprocess_cache.py --split train --workers 8
"""

import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import argparse
import glob
import time
from pathlib import Path
from multiprocessing import Pool, cpu_count

import torch
import numpy as np
import tensorflow as tf

tf.config.set_visible_devices([], "GPU")

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, "/workspace/data/WOMD")
sys.path.insert(0, "/workspace/data")

from waymo_open_dataset.protos import scenario_pb2


# Match training constants exactly
TRAJECTORY_SCALE = 50.0
FUTURE_LENGTH = 50
HISTORY_LENGTH = 11
MAX_AGENTS = 32
MAX_LANES = 64
LANE_POINTS = 20

DATA_ROOT = "/workspace/data/WOMD/datasets/waymo_open_dataset/motion_v_1_3_0/processed/augmented/scenario"
CACHE_ROOT = "/workspace/data/WOMD/datasets/waymo_open_dataset/motion_v_1_3_0/processed/augmented/cached"

SPLITS = {
    "train": f"{DATA_ROOT}/training_interactive",
    "val": f"{DATA_ROOT}/validation_interactive",
    "test": f"{DATA_ROOT}/testing_interactive",
}


def normalize_coords(x, y, ref_x, ref_y, cos_h, sin_h):
    dx, dy = x - ref_x, y - ref_y
    return dx * cos_h - dy * sin_h, dx * sin_h + dy * cos_h


def normalize_heading(h, ref_h):
    rel_h = h - ref_h
    while rel_h > np.pi:
        rel_h -= 2 * np.pi
    while rel_h < -np.pi:
        rel_h += 2 * np.pi
    return rel_h


def extract_features(scenario):
    """Extract features from a Scenario proto. Returns dict or None."""
    tracks = list(scenario.tracks)
    if not tracks:
        return None

    sdc_idx = scenario.sdc_track_index
    if sdc_idx < 0 or sdc_idx >= len(tracks):
        sdc_idx = 0

    sdc_track = tracks[sdc_idx]
    current_ts = scenario.current_time_index

    if len(sdc_track.states) < current_ts + FUTURE_LENGTH:
        return None

    ref_state = (
        sdc_track.states[current_ts] if current_ts < len(sdc_track.states) else None
    )
    if ref_state is None or not ref_state.valid:
        return None

    ref_x, ref_y = ref_state.center_x, ref_state.center_y
    if abs(ref_x) < 0.001 and abs(ref_y) < 0.001:
        return None

    ref_h = ref_state.heading
    cos_h, sin_h = np.cos(-ref_h), np.sin(-ref_h)

    # Ego history
    ego_history = np.zeros((HISTORY_LENGTH, 4), dtype=np.float32)
    for t in range(HISTORY_LENGTH):
        if t < len(sdc_track.states) and sdc_track.states[t].valid:
            state = sdc_track.states[t]
            x, y = normalize_coords(
                state.center_x, state.center_y, ref_x, ref_y, cos_h, sin_h
            )
            ego_history[t] = [
                x,
                y,
                normalize_heading(state.heading, ref_h),
                np.sqrt(state.velocity_x**2 + state.velocity_y**2),
            ]

    # Ego future
    ego_future = np.zeros((FUTURE_LENGTH, 4), dtype=np.float32)
    valid_count = 0
    for t in range(FUTURE_LENGTH):
        ts_idx = current_ts + 1 + t
        if ts_idx < len(sdc_track.states) and sdc_track.states[ts_idx].valid:
            state = sdc_track.states[ts_idx]
            x, y = normalize_coords(
                state.center_x, state.center_y, ref_x, ref_y, cos_h, sin_h
            )
            ego_future[t] = [
                x,
                y,
                normalize_heading(state.heading, ref_h),
                np.sqrt(state.velocity_x**2 + state.velocity_y**2),
            ]
            valid_count += 1

    if valid_count < FUTURE_LENGTH // 2:
        return None

    # Agent states
    agent_states = np.zeros((MAX_AGENTS, HISTORY_LENGTH, 4), dtype=np.float32)
    agent_count = 0
    for i, track in enumerate(tracks):
        if i == sdc_idx or agent_count >= MAX_AGENTS:
            continue
        has_valid = False
        for t in range(HISTORY_LENGTH):
            if t < len(track.states) and track.states[t].valid:
                state = track.states[t]
                x, y = normalize_coords(
                    state.center_x, state.center_y, ref_x, ref_y, cos_h, sin_h
                )
                agent_states[agent_count, t] = [
                    x,
                    y,
                    normalize_heading(state.heading, ref_h),
                    np.sqrt(state.velocity_x**2 + state.velocity_y**2),
                ]
                has_valid = True
        if has_valid:
            agent_count += 1

    # Lane centers
    lane_centers = np.zeros((MAX_LANES, LANE_POINTS, 2), dtype=np.float32)
    lane_count = 0
    for map_feature in scenario.map_features:
        if lane_count >= MAX_LANES:
            break
        if map_feature.HasField("lane"):
            polyline = list(map_feature.lane.polyline)
            if polyline:
                indices = np.linspace(0, len(polyline) - 1, LANE_POINTS).astype(int)
                for p_idx, src_idx in enumerate(indices):
                    x, y = normalize_coords(
                        polyline[src_idx].x,
                        polyline[src_idx].y,
                        ref_x,
                        ref_y,
                        cos_h,
                        sin_h,
                    )
                    lane_centers[lane_count, p_idx] = [x, y]
                lane_count += 1

    # Normalize
    ego_history[:, :2] /= TRAJECTORY_SCALE
    ego_future[:, :2] /= TRAJECTORY_SCALE
    agent_states[:, :, :2] /= TRAJECTORY_SCALE
    lane_centers /= TRAJECTORY_SCALE

    max_val = max(np.abs(ego_future[:, :2]).max(), np.abs(ego_history[:, :2]).max())
    if max_val > 5.0:
        return None

    return {
        "ego_history": ego_history,
        "agent_states": agent_states,
        "lane_centers": lane_centers,
        "traj_gt": ego_future,
    }


def process_single_file(args):
    """Process one TFRecord file → multiple .pt files. Returns count."""
    tfrecord_path, cache_dir, file_idx = args
    count = 0
    skipped = 0

    dataset = tf.data.TFRecordDataset([tfrecord_path])
    for record_idx, raw_record in enumerate(dataset):
        raw_bytes = raw_record.numpy()

        # Parse augmented Example format
        try:
            example = tf.train.Example()
            example.ParseFromString(raw_bytes)
            features = example.features.feature

            if "scenario/proto" not in features:
                skipped += 1
                continue

            scenario_bytes = features["scenario/proto"].bytes_list.value[0]
            scenario = scenario_pb2.Scenario()
            scenario.ParseFromString(scenario_bytes)

            sample = extract_features(scenario)
            if sample is None:
                skipped += 1
                continue

            # Add rule labels
            if "rule/applicability" in features:
                sample["rule_applicability"] = np.array(
                    list(features["rule/applicability"].int64_list.value),
                    dtype=np.float32,
                )
            else:
                sample["rule_applicability"] = np.zeros(28, dtype=np.float32)

            if "rule/violations" in features:
                sample["rule_violations"] = np.array(
                    list(features["rule/violations"].int64_list.value), dtype=np.float32
                )
            else:
                sample["rule_violations"] = np.zeros(28, dtype=np.float32)

            if "rule/severity" in features:
                sample["rule_severity"] = np.array(
                    list(features["rule/severity"].float_list.value), dtype=np.float32
                )
            else:
                sample["rule_severity"] = np.zeros(28, dtype=np.float32)

            # Convert to tensors and save
            pt_sample = {k: torch.from_numpy(v) for k, v in sample.items()}
            out_path = cache_dir / f"{file_idx:05d}_{record_idx:05d}.pt"
            torch.save(pt_sample, out_path)
            count += 1

        except Exception:
            skipped += 1
            continue

    return count, skipped


def main():
    parser = argparse.ArgumentParser(description="Pre-cache TFRecords to .pt files")
    parser.add_argument(
        "--split", type=str, default="train", choices=["train", "val", "test", "all"]
    )
    parser.add_argument("--workers", type=int, default=min(8, cpu_count()))
    args = parser.parse_args()

    splits = ["train", "val", "test"] if args.split == "all" else [args.split]

    for split in splits:
        src_dir = SPLITS[split]
        cache_dir = Path(CACHE_ROOT) / f"{split}_interactive"
        cache_dir.mkdir(parents=True, exist_ok=True)

        tfrecord_files = sorted(glob.glob(os.path.join(src_dir, "*.tfrecord*")))
        if not tfrecord_files:
            print(f"[{split}] No TFRecord files found in {src_dir}")
            continue

        print(f"[{split}] Processing {len(tfrecord_files)} TFRecords → {cache_dir}")
        start = time.time()

        work_items = [(f, cache_dir, i) for i, f in enumerate(tfrecord_files)]

        total_count = 0
        total_skipped = 0

        if args.workers > 1:
            with Pool(args.workers) as pool:
                for i, (count, skipped) in enumerate(
                    pool.imap_unordered(process_single_file, work_items)
                ):
                    total_count += count
                    total_skipped += skipped
                    if (i + 1) % 50 == 0:
                        elapsed = time.time() - start
                        print(
                            f"  {i+1}/{len(tfrecord_files)} files, "
                            f"{total_count} samples, {elapsed:.0f}s"
                        )
        else:
            for item in work_items:
                count, skipped = process_single_file(item)
                total_count += count
                total_skipped += skipped

        elapsed = time.time() - start
        print(
            f"[{split}] Done: {total_count} samples cached, "
            f"{total_skipped} skipped, {elapsed:.0f}s"
        )
        print(f"  Cache: {cache_dir}")

        # Save manifest
        manifest = sorted(cache_dir.glob("*.pt"))
        manifest_path = cache_dir / "manifest.txt"
        with open(manifest_path, "w") as f:
            for p in manifest:
                f.write(f"{p.name}\n")
        print(f"  Manifest: {manifest_path} ({len(manifest)} files)")


if __name__ == "__main__":
    main()
