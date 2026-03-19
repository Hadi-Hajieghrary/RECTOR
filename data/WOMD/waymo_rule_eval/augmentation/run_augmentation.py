#!/usr/bin/env python3
"""
Standalone Augmentation Script for Waymo Scenarios

Run from /workspace/data/waymo_rule_eval directory:
    python augmentation/run_augmentation.py \
        --input /path/to/raw/scenario/testing_interactive \
        --output /path/to/augmented/scenario/testing_interactive
"""

import argparse
import glob
import os
import sys
import warnings

warnings.filterwarnings("ignore")

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
from data_access.adapter_motion_scenario import MotionScenarioReader
from rules.rule_constants import RULE_IDS, NUM_RULES

# Now import from package
from pipeline.rule_executor import RuleExecutor
from tqdm import tqdm
from waymo_open_dataset.protos import scenario_pb2


class TFRecordWriter:
    """Simple TFRecord writer for augmented scenarios."""

    RULE_ORDER = RULE_IDS
    NUM_RULES = NUM_RULES

    def __init__(self, output_path: str):
        self.output_path = output_path
        self.writer = tf.io.TFRecordWriter(output_path)
        self.rule_id_to_idx = {r: i for i, r in enumerate(self.RULE_ORDER)}

    def write(self, scenario: scenario_pb2.Scenario, results: list):
        """Write augmented scenario to TFRecord."""
        # Build rule evaluation arrays
        applicability = np.zeros(self.NUM_RULES, dtype=np.int64)
        violations = np.zeros(self.NUM_RULES, dtype=np.int64)
        severity = np.zeros(self.NUM_RULES, dtype=np.float32)

        for r in results:
            rule_id = r.get("rule_id", "")
            if rule_id in self.rule_id_to_idx:
                idx = self.rule_id_to_idx[rule_id]
                if r.get("applicable", False):
                    applicability[idx] = 1
                    if r.get("violated", False):
                        violations[idx] = 1
                        severity[idx] = float(r.get("severity", 0.0))

        # Create feature dict
        feature = {
            "scenario/proto": tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[scenario.SerializeToString()])
            ),
            "scenario/id": tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[scenario.scenario_id.encode()])
            ),
            "rule/applicability": tf.train.Feature(
                int64_list=tf.train.Int64List(value=applicability)
            ),
            "rule/violations": tf.train.Feature(
                int64_list=tf.train.Int64List(value=violations)
            ),
            "rule/severity": tf.train.Feature(
                float_list=tf.train.FloatList(value=severity)
            ),
            "rule/ids": tf.train.Feature(
                bytes_list=tf.train.BytesList(
                    value=[r.encode() for r in self.RULE_ORDER]
                )
            ),
        }

        example = tf.train.Example(features=tf.train.Features(feature=feature))
        self.writer.write(example.SerializeToString())

    def close(self):
        self.writer.close()


def augment_dataset(input_dir: str, output_dir: str):
    """Augment all TFRecord files in input directory."""
    # Get all input files
    input_files = sorted(glob.glob(os.path.join(input_dir, "*.tfrecord*")))
    print(f"Found {len(input_files)} input files")

    if not input_files:
        print(f"No TFRecord files found in {input_dir}")
        return

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize components
    executor = RuleExecutor()
    executor.register_all_rules()
    reader = MotionScenarioReader()

    total_scenarios = 0
    total_violations = 0
    total_applicable = 0

    # Process each file
    for input_file in tqdm(input_files, desc="Processing files"):
        basename = os.path.basename(input_file)
        output_file = os.path.join(output_dir, basename)

        writer = TFRecordWriter(output_file)

        dataset = tf.data.TFRecordDataset(input_file)
        for record in dataset:
            scenario = scenario_pb2.Scenario()
            scenario.ParseFromString(record.numpy())

            try:
                ctx = reader.parse_scenario(scenario)
                result = executor.evaluate(ctx)
                # Convert ScenarioResult to list of dicts for writer
                results = []
                for rule_result in result.rule_results:
                    results.append(
                        {
                            "rule_id": rule_result.rule_id,
                            "applicable": rule_result.applies,
                            "violated": rule_result.has_violation,
                            "severity": rule_result.severity,
                        }
                    )
                writer.write(scenario, results)

                total_scenarios += 1
                for r in results:
                    if r.get("applicable", False):
                        total_applicable += 1
                        if r.get("violated", False):
                            total_violations += 1
            except Exception as e:
                # Still write scenario even if rule evaluation fails
                writer.write(scenario, [])
                total_scenarios += 1

        writer.close()

    print(f"\nCompleted!")
    print(f"Total scenarios: {total_scenarios}")
    print(f"Total applicable evaluations: {total_applicable}")
    print(f"Total violations: {total_violations}")


def main():
    parser = argparse.ArgumentParser(
        description="Augment Waymo scenarios with rule evaluation"
    )
    parser.add_argument(
        "--input", required=True, help="Input directory with TFRecord files"
    )
    parser.add_argument(
        "--output", required=True, help="Output directory for augmented files"
    )
    args = parser.parse_args()

    augment_dataset(args.input, args.output)


if __name__ == "__main__":
    main()
