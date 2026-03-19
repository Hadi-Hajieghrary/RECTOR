#!/usr/bin/env python3
"""
Process Waymo scenarios and augment with rule evaluation data.

Usage:
    python -m waymo_rule_eval.augmentation.process_scenarios \
        --input /path/to/scenarios.tfrecord \
        --output /path/to/output.jsonl

    # Process directory
    python -m waymo_rule_eval.augmentation.process_scenarios \
        --input "/path/to/*.tfrecord-*" \
        --output /path/to/output_dir/
"""

import argparse
import glob
import json
import logging
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

warnings.filterwarnings("ignore")

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from waymo_rule_eval.augmentation import ScenarioAugmenter
from waymo_rule_eval.data_access.adapter_motion_scenario import MotionScenarioReader

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)


def process_single_file(
    input_path: str,
    output_path: str,
    window_size_s: float = 8.0,
    stride_s: float = 1.0,
    max_scenarios: Optional[int] = None,
) -> dict:
    """
    Process a single TFRecord file.

    Returns:
        dict with processing statistics
    """
    stats = {
        "input_file": input_path,
        "output_file": output_path,
        "scenarios_processed": 0,
        "windows_generated": 0,
        "total_violations": 0,
        "errors": 0,
    }

    try:
        # Initialize
        augmenter = ScenarioAugmenter(
            window_size_s=window_size_s,
            stride_s=stride_s,
        )
        reader = MotionScenarioReader()

        # Open output file
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        with open(output_path, "w") as f:
            count = 0
            for ctx in reader.read_tfrecord(input_path):
                if max_scenarios and count >= max_scenarios:
                    break

                try:
                    aug = augmenter.augment_scenario(ctx)

                    # Write as JSON line
                    f.write(json.dumps(aug.to_dict()) + "\n")

                    stats["scenarios_processed"] += 1
                    stats["windows_generated"] += aug.num_windows
                    stats["total_violations"] += aug.total_violations
                    count += 1

                except Exception as e:
                    log.warning(f"Error processing scenario: {e}")
                    stats["errors"] += 1

        log.info(
            f"Processed {stats['scenarios_processed']} scenarios from {input_path}"
        )

    except Exception as e:
        log.error(f"Failed to process {input_path}: {e}")
        stats["errors"] += 1

    return stats


def process_files_parallel(
    input_files: List[str],
    output_dir: str,
    window_size_s: float = 8.0,
    stride_s: float = 1.0,
    max_scenarios_per_file: Optional[int] = None,
    workers: int = 1,
) -> List[dict]:
    """Process multiple files in parallel."""

    os.makedirs(output_dir, exist_ok=True)
    all_stats = []

    def get_output_path(input_path: str) -> str:
        basename = os.path.basename(input_path)
        name = basename.replace(".tfrecord", "").replace("-", "_")
        return os.path.join(output_dir, f"{name}_augmented.jsonl")

    if workers == 1:
        # Sequential processing
        for input_path in input_files:
            output_path = get_output_path(input_path)
            stats = process_single_file(
                input_path, output_path, window_size_s, stride_s, max_scenarios_per_file
            )
            all_stats.append(stats)
    else:
        # Parallel processing
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {}
            for input_path in input_files:
                output_path = get_output_path(input_path)
                future = executor.submit(
                    process_single_file,
                    input_path,
                    output_path,
                    window_size_s,
                    stride_s,
                    max_scenarios_per_file,
                )
                futures[future] = input_path

            for future in as_completed(futures):
                input_path = futures[future]
                try:
                    stats = future.result()
                    all_stats.append(stats)
                except Exception as e:
                    log.error(f"Failed to process {input_path}: {e}")
                    all_stats.append(
                        {
                            "input_file": input_path,
                            "errors": 1,
                        }
                    )

    return all_stats


def main():
    parser = argparse.ArgumentParser(
        description="Augment Waymo scenarios with rule evaluation data"
    )
    parser.add_argument(
        "--input", "-i", required=True, help="Input TFRecord file or glob pattern"
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output file (for single input) or directory (for multiple)",
    )
    parser.add_argument(
        "--window-size",
        "-w",
        type=float,
        default=8.0,
        help="Window size in seconds (default: 8.0)",
    )
    parser.add_argument(
        "--stride",
        "-s",
        type=float,
        default=1.0,
        help="Stride between windows in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--max-scenarios",
        "-m",
        type=int,
        default=None,
        help="Maximum scenarios per file (default: all)",
    )
    parser.add_argument(
        "--workers",
        "-j",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Find input files
    if "*" in args.input or "?" in args.input:
        input_files = sorted(glob.glob(args.input, recursive=True))
    else:
        input_files = [args.input]

    if not input_files:
        log.error(f"No files found matching: {args.input}")
        sys.exit(1)

    log.info(f"Found {len(input_files)} input files")

    # Process
    if len(input_files) == 1 and not os.path.isdir(args.output):
        # Single file
        stats = process_single_file(
            input_files[0],
            args.output,
            args.window_size,
            args.stride,
            args.max_scenarios,
        )
        all_stats = [stats]
    else:
        # Multiple files
        all_stats = process_files_parallel(
            input_files,
            args.output,
            args.window_size,
            args.stride,
            args.max_scenarios,
            args.workers,
        )

    # Print summary
    total_scenarios = sum(s.get("scenarios_processed", 0) for s in all_stats)
    total_windows = sum(s.get("windows_generated", 0) for s in all_stats)
    total_violations = sum(s.get("total_violations", 0) for s in all_stats)
    total_errors = sum(s.get("errors", 0) for s in all_stats)

    print("\n" + "=" * 60)
    print("AUGMENTATION SUMMARY")
    print("=" * 60)
    print(f"Files processed:     {len(input_files)}")
    print(f"Scenarios processed: {total_scenarios}")
    print(f"Windows generated:   {total_windows}")
    print(f"Total violations:    {total_violations}")
    print(f"Errors:              {total_errors}")
    print("=" * 60)


if __name__ == "__main__":
    main()
