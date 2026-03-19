#!/usr/bin/env python3
"""
Augmentation CLI for Waymo Rule Evaluation

Run as module from /workspace/data:
    python -m waymo_rule_eval.augmentation.augment_cli \
        --input /path/to/raw/scenario/testing_interactive \
        --output /path/to/augmented/scenario/testing_interactive
"""

import argparse
import glob
import os
import sys
import warnings

warnings.filterwarnings("ignore")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
from tqdm import tqdm
from waymo_open_dataset.protos import scenario_pb2

from ..data_access.adapter_motion_scenario import MotionScenarioReader
from ..pipeline.rule_executor import RuleExecutor
from ..rules.rule_constants import RULE_IDS, NUM_RULES


class TFRecordWriter:
    """Simple TFRecord writer for augmented scenarios."""

    # Static rule ordering for TFRecord feature vectors.
    # Must match the canonical ordering in rules/rule_constants.py.
    RULE_ORDER = RULE_IDS
    NUM_RULES = NUM_RULES

    def __init__(self, output_path: str):
        self.output_path = output_path
        self.writer = tf.io.TFRecordWriter(output_path)
        self.rule_id_to_idx = {r: i for i, r in enumerate(self.RULE_ORDER)}

    def write(self, scenario: scenario_pb2.Scenario, results: list):
        """Write augmented scenario to TFRecord."""
        applicability = np.zeros(self.NUM_RULES, dtype=np.int64)
        violations = np.zeros(self.NUM_RULES, dtype=np.int64)
        severity = np.zeros(self.NUM_RULES, dtype=np.float32)

        for r in results:
            rule_id = r.rule_id if hasattr(r, "rule_id") else r.get("rule_id", "")
            if rule_id in self.rule_id_to_idx:
                idx = self.rule_id_to_idx[rule_id]
                applies = (
                    r.applies if hasattr(r, "applies") else r.get("applicable", False)
                )
                if applies:
                    applicability[idx] = 1
                    violated = False
                    sev_value = 0.0
                    if hasattr(r, "violation") and r.violation is not None:
                        # ViolationResult uses severity > 0 to indicate violation
                        raw_sev = (
                            r.violation.severity
                            if r.violation.severity is not None
                            else 0.0
                        )
                        sev_value = float(raw_sev) if not np.isnan(raw_sev) else 0.0
                        violated = sev_value > 0
                    elif isinstance(r, dict) and r.get("violated", False):
                        violated = True
                        sev_value = float(r.get("severity", 0.0))
                    severity[idx] = sev_value
                    if violated:
                        violations[idx] = 1

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
    input_files = sorted(glob.glob(os.path.join(input_dir, "*.tfrecord*")))
    print(f"Found {len(input_files)} input files")

    if not input_files:
        print(f"No TFRecord files found in {input_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)

    executor = RuleExecutor()
    executor.register_all_rules()  # Register all rules before evaluation
    adapter = MotionScenarioReader()

    total_scenarios = 0
    total_violations = 0
    total_applicable = 0
    skipped_files = []

    for input_file in tqdm(input_files, desc="Processing files"):
        basename = os.path.basename(input_file)
        output_file = os.path.join(output_dir, basename)

        # Skip if output file already exists
        if os.path.exists(output_file):
            continue

        writer = TFRecordWriter(output_file)

        try:
            dataset = tf.data.TFRecordDataset(input_file)
            for record in dataset:
                scenario = scenario_pb2.Scenario()
                scenario.ParseFromString(record.numpy())

                try:
                    ctx = adapter.parse_scenario(scenario)
                    scenario_result = executor.evaluate(ctx)

                    # Convert ScenarioResult to list of rule results for writer
                    results = scenario_result.rule_results
                    writer.write(scenario, results)

                    total_scenarios += 1
                    for r in results:
                        applies = r.applies if hasattr(r, "applies") else False
                        if applies:
                            total_applicable += 1
                            # Check for violation using severity > 0
                            if hasattr(r, "violation") and r.violation is not None:
                                sev = r.violation.severity
                                if sev is not None and not np.isnan(sev) and sev > 0:
                                    total_violations += 1
                except Exception as e:
                    import warnings

                    warnings.warn(
                        f"Error evaluating scenario {scenario.scenario_id}: {e}"
                    )
                    writer.write(scenario, [])
                    total_scenarios += 1
        except tf.errors.DataLossError as e:
            print(f"\nSkipping corrupted file: {basename} - {e}")
            skipped_files.append(basename)
        finally:
            writer.close()

    print("\nCompleted!")
    print(f"Total scenarios: {total_scenarios}")
    print(f"Total applicable evaluations: {total_applicable}")
    print(f"Total violations: {total_violations}")
    if skipped_files:
        print(f"Skipped {len(skipped_files)} corrupted files: {skipped_files}")


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
