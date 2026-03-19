#!/usr/bin/env python3
"""
TFRecord Writer for Augmented Scenarios

Writes augmented scenarios back to TFRecord format with rule evaluation features
added as additional fields. This preserves compatibility with existing Waymo
data pipelines while adding rule evaluation data.

Output format adds these features to each scenario:
- rule_applicability: (NUM_RULES,) bool array - which rules are applicable
- rule_violations: (NUM_RULES,) bool array - which rules were violated  
- rule_severity: (NUM_RULES,) float array - severity scores per rule
- rule_ids: (NUM_RULES,) string array - rule ID labels

Usage:
    python -m waymo_rule_eval.augmentation.tfrecord_writer \
        --input /path/to/scenarios/*.tfrecord* \
        --output /path/to/augmented/
"""

import argparse
import glob
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

# Suppress TF warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf

try:
    from ..core.context import ScenarioContext
    from ..data_access.adapter_motion_scenario import MotionScenarioReader
    from ..pipeline.rule_executor import RuleExecutor
    from ..rules.rule_constants import RULE_IDS, NUM_RULES
except ImportError:
    from core.context import ScenarioContext
    from data_access.adapter_motion_scenario import MotionScenarioReader
    from pipeline.rule_executor import RuleExecutor
    from rules.rule_constants import RULE_IDS, NUM_RULES

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)


# Import waymo protos
try:
    from waymo_open_dataset.protos import scenario_pb2

    WAYMO_AVAILABLE = True
except ImportError:
    WAYMO_AVAILABLE = False
    log.warning("Waymo Open Dataset protos not available")


class TFRecordAugmenter:
    """
    Augments Waymo TFRecord files with rule evaluation data.

    Reads original TFRecord scenarios, evaluates rules, and writes
    augmented TFRecords with additional features.
    """

    def __init__(self):
        """Initialize the augmenter with rule executor."""
        self.executor = RuleExecutor()
        self.executor.register_all_rules()
        self.reader = MotionScenarioReader()

        # Build rule ID to index mapping from canonical list
        self.rule_ids = RULE_IDS

        self.n_rules = NUM_RULES
        log.info(f"Initialized with {self.n_rules} rules: {self.rule_ids}")

    def _evaluate_scenario(self, ctx: ScenarioContext) -> Dict[str, np.ndarray]:
        """
        Evaluate all rules on a scenario.

        Returns:
            Dict with:
                - applicability: (n_rules,) bool array
                - violations: (n_rules,) bool array
                - severity: (n_rules,) float array
        """
        applicability = np.zeros(self.n_rules, dtype=bool)
        violations = np.zeros(self.n_rules, dtype=bool)
        severity = np.zeros(self.n_rules, dtype=np.float32)

        try:
            result = self.executor.evaluate(ctx)

            for i, rule_id in enumerate(self.rule_ids):
                # Find matching result
                for rr in result.rule_results:
                    if rr.rule_id == rule_id:
                        applicability[i] = rr.applies
                        if rr.applies and rr.violation is not None:
                            violations[i] = rr.violation.severity > 0
                            severity[i] = rr.violation.severity_normalized
                        break

        except Exception as e:
            log.warning(f"Error evaluating scenario {ctx.scenario_id}: {e}")

        return {
            "applicability": applicability,
            "violations": violations,
            "severity": severity,
        }

    def _serialize_augmented_example(
        self,
        original_bytes: bytes,
        rule_data: Dict[str, np.ndarray],
    ) -> bytes:
        """
        Create augmented tf.Example with rule features added.

        This approach embeds rule data as additional features in the Example.
        """
        # Parse original scenario to get ID
        scenario = scenario_pb2.Scenario()
        scenario.ParseFromString(original_bytes)

        # Create tf.Example with rule features
        feature_dict = {
            # Store original scenario as bytes
            "scenario/proto": tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[original_bytes])
            ),
            # Scenario metadata
            "scenario/id": tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[scenario.scenario_id.encode()])
            ),
            # Rule evaluation results
            "rule/applicability": tf.train.Feature(
                int64_list=tf.train.Int64List(
                    value=rule_data["applicability"].astype(int)
                )
            ),
            "rule/violations": tf.train.Feature(
                int64_list=tf.train.Int64List(value=rule_data["violations"].astype(int))
            ),
            "rule/severity": tf.train.Feature(
                float_list=tf.train.FloatList(value=rule_data["severity"])
            ),
            "rule/ids": tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[r.encode() for r in self.rule_ids])
            ),
            "rule/n_rules": tf.train.Feature(
                int64_list=tf.train.Int64List(value=[self.n_rules])
            ),
            # Summary stats
            "rule/n_applicable": tf.train.Feature(
                int64_list=tf.train.Int64List(
                    value=[int(rule_data["applicability"].sum())]
                )
            ),
            "rule/n_violations": tf.train.Feature(
                int64_list=tf.train.Int64List(
                    value=[int(rule_data["violations"].sum())]
                )
            ),
            "rule/total_severity": tf.train.Feature(
                float_list=tf.train.FloatList(
                    value=[float(rule_data["severity"].sum())]
                )
            ),
        }

        example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
        return example.SerializeToString()

    def process_file(
        self,
        input_path: str,
        output_path: str,
    ) -> Dict[str, int]:
        """
        Process a single TFRecord file.

        Args:
            input_path: Path to input TFRecord
            output_path: Path to output augmented TFRecord

        Returns:
            Stats dict with counts
        """
        stats = {
            "scenarios": 0,
            "errors": 0,
            "total_applicable": 0,
            "total_violations": 0,
        }

        # Read original file as raw bytes and parse scenarios
        # Detect compression from file extension
        compression = "GZIP" if input_path.endswith(".gz") else ""
        dataset = tf.data.TFRecordDataset(input_path, compression_type=compression)

        # Stream ScenarioContext alongside raw records to avoid loading all into memory
        context_iter = self.reader.read_tfrecord(input_path)

        # Write augmented records
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        with tf.io.TFRecordWriter(output_path) as writer:
            for i, raw_record in enumerate(dataset):
                try:
                    ctx = next(context_iter)
                except StopIteration:
                    break

                original_bytes = raw_record.numpy()

                try:
                    # Evaluate rules
                    rule_data = self._evaluate_scenario(ctx)

                    # Create augmented example
                    augmented_bytes = self._serialize_augmented_example(
                        original_bytes, rule_data
                    )

                    writer.write(augmented_bytes)

                    stats["scenarios"] += 1
                    stats["total_applicable"] += int(rule_data["applicability"].sum())
                    stats["total_violations"] += int(rule_data["violations"].sum())

                except Exception as e:
                    log.warning(f"Error processing scenario {i}: {e}")
                    stats["errors"] += 1

        return stats

    def process_directory(
        self,
        input_pattern: str,
        output_dir: str,
        max_files: Optional[int] = None,
        skip_existing: bool = True,
    ) -> Dict[str, int]:
        """
        Process all TFRecord files matching pattern.

        Args:
            input_pattern: Glob pattern for input files
            output_dir: Output directory for augmented files
            max_files: Optional limit on files to process
            skip_existing: Skip files whose output already exists (default True)

        Returns:
            Aggregate stats
        """
        input_files = sorted(glob.glob(input_pattern))
        if max_files:
            input_files = input_files[:max_files]

        log.info(f"Processing {len(input_files)} files to {output_dir}")

        os.makedirs(output_dir, exist_ok=True)

        total_stats = {
            "files": 0,
            "scenarios": 0,
            "errors": 0,
            "total_applicable": 0,
            "total_violations": 0,
        }

        for input_path in input_files:
            # Generate output filename
            basename = os.path.basename(input_path)
            # Add _augmented before extension
            if ".tfrecord" in basename:
                parts = basename.split(".tfrecord")
                output_name = (
                    parts[0] + "_augmented.tfrecord" + parts[1]
                    if len(parts) > 1
                    else parts[0] + "_augmented.tfrecord"
                )
            else:
                output_name = basename + "_augmented"

            output_path = os.path.join(output_dir, output_name)

            if skip_existing and os.path.exists(output_path):
                log.info(f"Skipping {basename} (output exists)")
                continue

            log.info(f"Processing {basename}...")
            stats = self.process_file(input_path, output_path)

            total_stats["files"] += 1
            total_stats["scenarios"] += stats["scenarios"]
            total_stats["errors"] += stats["errors"]
            total_stats["total_applicable"] += stats["total_applicable"]
            total_stats["total_violations"] += stats["total_violations"]

            log.info(
                f"  -> {stats['scenarios']} scenarios, {stats['total_violations']} violations"
            )

        return total_stats


def read_augmented_tfrecord(path: str) -> Iterator[Tuple[Any, Dict[str, np.ndarray]]]:
    """
    Read an augmented TFRecord file.

    Yields:
        Tuple of (scenario_proto, rule_data_dict)
    """
    if not WAYMO_AVAILABLE:
        raise ImportError("Waymo Open Dataset not installed")

    feature_description = {
        "scenario/proto": tf.io.FixedLenFeature([], tf.string),
        "scenario/id": tf.io.FixedLenFeature([], tf.string),
        "rule/applicability": tf.io.VarLenFeature(tf.int64),
        "rule/violations": tf.io.VarLenFeature(tf.int64),
        "rule/severity": tf.io.VarLenFeature(tf.float32),
        "rule/ids": tf.io.VarLenFeature(tf.string),
        "rule/n_rules": tf.io.FixedLenFeature([], tf.int64),
        "rule/n_applicable": tf.io.FixedLenFeature([], tf.int64),
        "rule/n_violations": tf.io.FixedLenFeature([], tf.int64),
        "rule/total_severity": tf.io.FixedLenFeature([], tf.float32),
    }

    dataset = tf.data.TFRecordDataset(path)

    for raw_record in dataset:
        example = tf.io.parse_single_example(raw_record, feature_description)

        # Parse original scenario
        scenario = scenario_pb2.Scenario()
        scenario.ParseFromString(example["scenario/proto"].numpy())

        # Extract rule data
        rule_data = {
            "applicability": tf.sparse.to_dense(example["rule/applicability"])
            .numpy()
            .astype(bool),
            "violations": tf.sparse.to_dense(example["rule/violations"])
            .numpy()
            .astype(bool),
            "severity": tf.sparse.to_dense(example["rule/severity"]).numpy(),
            "rule_ids": [
                r.decode() for r in tf.sparse.to_dense(example["rule/ids"]).numpy()
            ],
            "n_applicable": example["rule/n_applicable"].numpy(),
            "n_violations": example["rule/n_violations"].numpy(),
            "total_severity": example["rule/total_severity"].numpy(),
        }

        yield scenario, rule_data


def main():
    parser = argparse.ArgumentParser(
        description="Augment Waymo TFRecord files with rule evaluation data"
    )
    parser.add_argument(
        "--input", "-i", required=True, help="Input TFRecord file or glob pattern"
    )
    parser.add_argument(
        "--output", "-o", required=True, help="Output directory for augmented TFRecords"
    )
    parser.add_argument(
        "--max-files", type=int, default=None, help="Maximum number of files to process"
    )

    args = parser.parse_args()

    if not WAYMO_AVAILABLE:
        log.error("Waymo Open Dataset not installed")
        sys.exit(1)

    augmenter = TFRecordAugmenter()

    if "*" in args.input:
        # Directory/pattern mode
        stats = augmenter.process_directory(
            args.input,
            args.output,
            max_files=args.max_files,
        )
    else:
        # Single file mode
        output_path = os.path.join(args.output, os.path.basename(args.input))
        stats = augmenter.process_file(args.input, output_path)
        stats["files"] = 1

    print("\n" + "=" * 60)
    print("AUGMENTATION SUMMARY")
    print("=" * 60)
    print(f"Files processed:     {stats.get('files', 1)}")
    print(f"Scenarios processed: {stats['scenarios']}")
    print(f"Total applicable:    {stats['total_applicable']}")
    print(f"Total violations:    {stats['total_violations']}")
    print(f"Errors:              {stats['errors']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
