#!/usr/bin/env python3
"""
Filter Interactive Scenarios from Waymo Training Data (Scenario Format)

This script identifies and extracts interactive scenarios directly from
scenario format TFRecords using Waymo's official definition.

Interactive scenarios (Waymo's official definition):
- Contains exactly 2 objects_of_interest (primary interacting agents)
- Used for the joint prediction task in models like M2I

Features:
- Filter by interaction type: v2v (vehicle-vehicle), v2p (vehicle-pedestrian),
  v2c (vehicle-cyclist), or others
- Count agent types in interactive pairs
- Parallel processing with multiprocessing

Usage:
    # Basic filtering (all interactive scenarios):
    python filter_interactive_scenario.py \\
        --input-dir /path/to/training \\
        --output-dir /path/to/training_interactive

    # Filter only vehicle-to-vehicle interactions:
    python filter_interactive_scenario.py \\
        --input-dir /path/to/training \\
        --output-dir /path/to/training_interactive_v2v \\
        --type v2v

    # Count types without filtering:
    python filter_interactive_scenario.py \\
        --input-dir /path/to/training \\
        --output-dir /tmp/dummy \\
        --skip-filtering --count-type
"""

import argparse
import multiprocessing as mp
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import tensorflow as tf
from tqdm import tqdm
from waymo_open_dataset.protos import scenario_pb2

# Object type constants (from Waymo)
# vehicle = 1, pedestrian = 2, cyclist = 3
OBJECT_TYPE_DICT = {
    "v2v": [1, 1],
    "v2p": [1, 2],
    "v2c": [1, 3],
}
TYPE_NAMES = {1: "vehicle", 2: "pedestrian", 3: "cyclist"}


@dataclass
class FilterStats:
    """Statistics from filtering a single file."""

    total: int = 0
    interactive: int = 0
    interactive_typed: int = 0
    type_counts: Dict[str, int] = field(
        default_factory=lambda: {"v": 0, "p": 0, "c": 0, "o": 0}
    )
    pair_counts: Dict[str, int] = field(
        default_factory=lambda: {
            "v2v": 0,
            "v2p": 0,
            "v2c": 0,
            "p2p": 0,
            "p2c": 0,
            "c2c": 0,
            "other": 0,
        }
    )


class InteractiveScenarioFilter:
    """Filter interactive scenarios from Waymo scenario format."""

    def __init__(
        self,
        interaction_type: Optional[str] = None,
        count_type: bool = False,
        skip_filtering: bool = False,
    ):
        """
        Initialize the filter.

        Args:
            interaction_type: Filter by type (v2v, v2p, v2c, others, or None for all)
            count_type: Whether to count agent types
            skip_filtering: If True, only count without writing output
        """
        self.interaction_type = interaction_type
        self.count_type = count_type
        self.skip_filtering = skip_filtering

    def _get_object_types(self, scenario: scenario_pb2.Scenario) -> List[int]:
        """Get the object types for objects_of_interest."""
        types = []
        obj_ids = set(scenario.objects_of_interest)
        for track in scenario.tracks:
            if track.id in obj_ids:
                types.append(track.object_type)
        return sorted(types)

    def _get_pair_key(self, types: List[int]) -> str:
        """Get the pair type key from sorted object types."""
        if types == [1, 1]:
            return "v2v"
        elif types == [1, 2]:
            return "v2p"
        elif types == [1, 3]:
            return "v2c"
        elif types == [2, 2]:
            return "p2p"
        elif types == [2, 3]:
            return "p2c"
        elif types == [3, 3]:
            return "c2c"
        else:
            return "other"

    def _matches_type_filter(self, types: List[int]) -> bool:
        """Check if the object types match the filter."""
        if self.interaction_type is None:
            return True
        if self.interaction_type == "others":
            return types not in [
                OBJECT_TYPE_DICT["v2v"],
                OBJECT_TYPE_DICT["v2p"],
                OBJECT_TYPE_DICT["v2c"],
            ]
        return types == OBJECT_TYPE_DICT.get(self.interaction_type, [])

    def is_interactive(self, scenario: scenario_pb2.Scenario) -> bool:
        """Check if a scenario contains exactly 2 interactive agents."""
        if hasattr(scenario, "objects_of_interest") and scenario.objects_of_interest:
            return len(scenario.objects_of_interest) == 2
        return False

    def process_tfrecord(
        self,
        input_path: Path,
        output_path: Path,
        non_interactive_path: Optional[Path] = None,
        move: bool = False,
    ) -> FilterStats:
        """
        Process a single TFRecord file and save interactive scenarios.

        Args:
            input_path: Path to input TFRecord file
            output_path: Path to output TFRecord file
            non_interactive_path: Path to save non-interactive scenarios (optional)
            move: If True, delete original file after copying

        Returns:
            FilterStats with counts
        """
        stats = FilterStats()
        interactive_scenarios = []
        non_interactive_scenarios = []

        try:
            dataset = tf.data.TFRecordDataset(str(input_path), compression_type="")

            for data in dataset:
                stats.total += 1
                scenario = scenario_pb2.Scenario()
                scenario.ParseFromString(data.numpy())

                if not self.is_interactive(scenario):
                    # Save non-interactive scenarios if path provided
                    if non_interactive_path is not None and not self.skip_filtering:
                        non_interactive_scenarios.append(data.numpy())
                    continue

                stats.interactive += 1
                object_types = self._get_object_types(scenario)

                # Count types if requested
                if self.count_type:
                    for obj_type in object_types:
                        if obj_type == 1:
                            stats.type_counts["v"] += 1
                        elif obj_type == 2:
                            stats.type_counts["p"] += 1
                        elif obj_type == 3:
                            stats.type_counts["c"] += 1
                        else:
                            stats.type_counts["o"] += 1

                # Count pair types
                pair_key = self._get_pair_key(object_types)
                stats.pair_counts[pair_key] += 1

                # Skip filtering if requested (just counting)
                if self.skip_filtering:
                    continue

                # Check type filter
                if self._matches_type_filter(object_types):
                    stats.interactive_typed += 1
                    interactive_scenarios.append(data.numpy())

            # Write filtered scenarios to output
            if interactive_scenarios and output_path is not None:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with tf.io.TFRecordWriter(str(output_path)) as writer:
                    for scenario_bytes in interactive_scenarios:
                        writer.write(scenario_bytes)

            # Write non-interactive scenarios if path provided
            if non_interactive_scenarios and non_interactive_path is not None:
                non_interactive_path.parent.mkdir(parents=True, exist_ok=True)
                with tf.io.TFRecordWriter(str(non_interactive_path)) as writer:
                    for scenario_bytes in non_interactive_scenarios:
                        writer.write(scenario_bytes)

            # Delete original file if move flag is set
            if move and (interactive_scenarios or non_interactive_scenarios):
                input_path.unlink()

        except Exception as e:
            print(f"Error processing {input_path}: {e}")
            return FilterStats()

        return stats


def process_file_worker(args):
    """Worker function for parallel processing."""
    input_file, output_file, non_interactive_file, filter_obj, move = args
    return filter_obj.process_tfrecord(
        input_file, output_file, non_interactive_file, move
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--input-dir",
        "-i",
        type=str,
        required=True,
        help="Input directory with scenario TFRecords",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        required=True,
        help="Output directory for filtered TFRecords",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of parallel workers (default: 8)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maximum number of files to process (for testing)",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move files by deleting originals after filtering",
    )
    parser.add_argument(
        "--type",
        type=str,
        choices=["v2v", "v2p", "v2c", "others"],
        default=None,
        help="Filter by interaction type (v2v, v2p, v2c, others)",
    )
    parser.add_argument(
        "--count-type",
        action="store_true",
        help="Count agent types in interactive pairs",
    )
    parser.add_argument(
        "--skip-filtering",
        action="store_true",
        help="Skip filtering (only count scenarios)",
    )
    parser.add_argument(
        "--non-interactive-dir",
        type=str,
        default=None,
        help="Output directory for non-interactive scenarios (optional)",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")

    non_interactive_dir = None
    if args.non_interactive_dir:
        non_interactive_dir = Path(args.non_interactive_dir)
        if not args.skip_filtering:
            non_interactive_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_filtering:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Get input files
    input_files = sorted(input_dir.glob("*.tfrecord*"))
    if args.max_files:
        input_files = input_files[: args.max_files]

    print("=" * 80)
    print("Interactive Scenario Filtering (Scenario Format)")
    print("=" * 80)
    print(f"Input directory:  {input_dir}")
    print(f"Output directory: {output_dir}")
    if non_interactive_dir:
        print(f"Non-interactive:  {non_interactive_dir}")
    print(f"Files to process: {len(input_files)}")
    print(f"Workers:          {args.num_workers}")
    print(f"Type filter:      {args.type or 'all'}")
    print(f"Count types:      {args.count_type}")
    print(f"Skip filtering:   {args.skip_filtering}")
    print(f"Delete originals: {args.move}")
    print("=" * 80)
    print()

    # Create filter
    filter_obj = InteractiveScenarioFilter(
        interaction_type=args.type,
        count_type=args.count_type,
        skip_filtering=args.skip_filtering,
    )

    # Prepare work items
    work_items = []
    for input_file in input_files:
        output_file = output_dir / input_file.name
        non_interactive_file = None
        if non_interactive_dir:
            non_interactive_file = non_interactive_dir / input_file.name
        work_items.append(
            (input_file, output_file, non_interactive_file, filter_obj, args.move)
        )

    # Process files in parallel
    total_stats = FilterStats()

    if args.num_workers > 1:
        with mp.Pool(args.num_workers) as pool:
            results = list(
                tqdm(
                    pool.imap(process_file_worker, work_items),
                    total=len(work_items),
                    desc="Processing files",
                )
            )
    else:
        results = []
        for item in tqdm(work_items, desc="Processing files"):
            results.append(process_file_worker(item))

    # Aggregate results
    for stats in results:
        total_stats.total += stats.total
        total_stats.interactive += stats.interactive
        total_stats.interactive_typed += stats.interactive_typed
        for k, v in stats.type_counts.items():
            total_stats.type_counts[k] += v
        for k, v in stats.pair_counts.items():
            total_stats.pair_counts[k] += v

    print()
    print("=" * 80)
    print("Filtering Complete!")
    print("=" * 80)
    print(f"Total scenarios:       {total_stats.total:,}")
    print(f"Interactive scenarios: {total_stats.interactive:,}")
    if args.type:
        print(f"Type-filtered ({args.type}):  {total_stats.interactive_typed:,}")
    if total_stats.total > 0:
        rate = 100.0 * total_stats.interactive / total_stats.total
        print(f"Interactive rate:      {rate:.2f}%")

    # Print pair type breakdown
    print()
    print("Interaction pair breakdown:")
    for pair_type, count in sorted(
        total_stats.pair_counts.items(), key=lambda x: -x[1]
    ):
        if count > 0:
            pct = (
                100.0 * count / total_stats.interactive
                if total_stats.interactive > 0
                else 0
            )
            print(f"  {pair_type}: {count:,} ({pct:.1f}%)")

    if args.count_type:
        print()
        print("Agent type counts (in interactive pairs):")
        for type_key, count in sorted(
            total_stats.type_counts.items(), key=lambda x: -x[1]
        ):
            type_name = {
                "v": "vehicle",
                "p": "pedestrian",
                "c": "cyclist",
                "o": "other",
            }[type_key]
            print(f"  {type_name}: {count:,}")

    if not args.skip_filtering:
        output_file_count = len(list(output_dir.glob("*.tfrecord*")))
        print()
        print(f"Output files: {output_file_count}")

    print("=" * 80)


if __name__ == "__main__":
    main()
