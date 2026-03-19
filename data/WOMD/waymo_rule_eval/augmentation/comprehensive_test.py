#!/usr/bin/env python3
"""
Comprehensive Testing Strategy for Augmented Waymo Scenarios

This module randomly samples at least 20% of the augmented data and generates
a comprehensive report on rule application, violation rates, and statistics.

Run as module from /workspace/data:
    python -m waymo_rule_eval.augmentation.comprehensive_test \
        --augmented-dir /path/to/processed/augmented/scenario \
        --sample-ratio 0.2 \
        --output /path/to/report.md
"""

import argparse
import glob
import json
import os
import random
import sys
import warnings
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Tuple

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
from tqdm import tqdm

try:
    from ..rules.rule_constants import RULE_IDS, NUM_RULES
except ImportError:
    from rules.rule_constants import RULE_IDS, NUM_RULES


@dataclass
class RuleStatistics:
    """Statistics for a single rule."""

    rule_id: str
    total_scenarios: int = 0
    applicable_count: int = 0
    violation_count: int = 0
    severity_sum: float = 0.0
    severity_values: List[float] = field(default_factory=list)

    @property
    def applicability_rate(self) -> float:
        if self.total_scenarios == 0:
            return 0.0
        return self.applicable_count / self.total_scenarios * 100

    @property
    def violation_rate(self) -> float:
        if self.applicable_count == 0:
            return 0.0
        return self.violation_count / self.applicable_count * 100

    @property
    def mean_severity(self) -> float:
        if not self.severity_values:
            return 0.0
        return np.mean(self.severity_values)

    @property
    def max_severity(self) -> float:
        if not self.severity_values:
            return 0.0
        return np.max(self.severity_values)

    @property
    def std_severity(self) -> float:
        if len(self.severity_values) < 2:
            return 0.0
        return np.std(self.severity_values)


@dataclass
class DatasetStatistics:
    """Statistics for an entire dataset."""

    dataset_name: str
    total_files: int = 0
    sampled_files: int = 0
    total_scenarios: int = 0
    scenarios_with_violations: int = 0
    scenarios_with_multiple_violations: int = 0
    rule_stats: Dict[str, RuleStatistics] = field(default_factory=dict)
    violation_distribution: Dict[int, int] = field(default_factory=dict)

    @property
    def violation_rate(self) -> float:
        if self.total_scenarios == 0:
            return 0.0
        return self.scenarios_with_violations / self.total_scenarios * 100


class AugmentedDataTester:
    """Comprehensive tester for augmented Waymo scenario data."""

    # Rule ordering matching the canonical ordering from rule_constants.py
    RULE_ORDER = RULE_IDS

    RULE_NAMES = {
        "L0.R2": "Safe Following Distance",
        "L0.R3": "Safe Lateral Clearance",
        "L0.R4": "Crosswalk Occupancy",
        "L1.R1": "Smooth Acceleration",
        "L1.R2": "Smooth Braking",
        "L1.R3": "Smooth Steering",
        "L1.R4": "Speed Consistency",
        "L1.R5": "Lane Change Smoothness",
        "L3.R3": "Drivable Surface",
        "L4.R3": "Left Turn Gap",
        "L5.R1": "Traffic Signal Compliance",
        "L5.R2": "Priority Violation",
        "L5.R3": "Parking Violation",
        "L5.R4": "School Zone Compliance",
        "L5.R5": "Construction Zone Compliance",
        "L6.R1": "Cooperative Lane Change",
        "L6.R2": "Following Distance",
        "L6.R3": "Intersection Negotiation",
        "L6.R4": "Pedestrian Interaction",
        "L6.R5": "Cyclist Interaction",
        "L7.R3": "Lane Departure",
        "L7.R4": "Speed Limit",
        "L8.R1": "Red Light",
        "L8.R2": "Stop Sign",
        "L8.R3": "Crosswalk Yield",
        "L8.R5": "Wrong-Way",
        "L10.R1": "Collision Detection",
        "L10.R2": "VRU Clearance",
    }

    RULE_LEVELS = {
        "L0": "Safety Critical",
        "L1": "Comfort",
        "L3": "Surface/Road",
        "L4": "Maneuver",
        "L5": "Priority",
        "L6": "Interaction",
        "L7": "Lane/Speed",
        "L8": "Traffic Control",
        "L10": "Collision",
    }

    def __init__(self, sample_ratio: float = 0.2, seed: int = 42):
        self.sample_ratio = max(0.2, min(1.0, sample_ratio))
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

    def parse_augmented_record(self, record: bytes) -> Dict[str, Any]:
        """Parse a single augmented TFRecord example."""
        example = tf.train.Example()
        example.ParseFromString(record)

        features = example.features.feature

        # Use rule/ids from TFRecord as source of truth for ordering
        n_rules = len(self.RULE_ORDER)
        result = {
            "scenario_id": "",
            "applicability": np.zeros(n_rules, dtype=bool),
            "violations": np.zeros(n_rules, dtype=bool),
            "severity": np.zeros(n_rules, dtype=np.float32),
            "rule_ids": self.RULE_ORDER[:],
        }

        if "scenario/id" in features:
            result["scenario_id"] = features["scenario/id"].bytes_list.value[0].decode()

        if "rule/applicability" in features:
            vals = list(features["rule/applicability"].int64_list.value)
            result["applicability"] = np.array(vals, dtype=bool)

        if "rule/violations" in features:
            vals = list(features["rule/violations"].int64_list.value)
            result["violations"] = np.array(vals, dtype=bool)

        if "rule/severity" in features:
            vals = list(features["rule/severity"].float_list.value)
            result["severity"] = np.array(vals, dtype=np.float32)

        if "rule/ids" in features:
            vals = [v.decode() for v in features["rule/ids"].bytes_list.value]
            result["rule_ids"] = vals

        return result

    def test_dataset(self, dataset_dir: str, dataset_name: str) -> DatasetStatistics:
        """Test a single augmented dataset directory."""
        stats = DatasetStatistics(dataset_name=dataset_name)

        # Initialize rule statistics
        for rule_id in self.RULE_ORDER:
            stats.rule_stats[rule_id] = RuleStatistics(rule_id=rule_id)

        # Find all TFRecord files
        pattern = os.path.join(dataset_dir, "*.tfrecord*")
        all_files = sorted(glob.glob(pattern))
        stats.total_files = len(all_files)

        if not all_files:
            print(f"  No TFRecord files found in {dataset_dir}")
            return stats

        # Sample files (at least 20%)
        sample_size = max(1, int(len(all_files) * self.sample_ratio))
        sampled_files = random.sample(all_files, sample_size)
        stats.sampled_files = len(sampled_files)

        print(
            f"  Sampling {len(sampled_files)}/{len(all_files)} files "
            f"({len(sampled_files)/len(all_files)*100:.1f}%)"
        )

        # Process sampled files
        for filepath in tqdm(sampled_files, desc=f"  Testing {dataset_name}"):
            dataset = tf.data.TFRecordDataset(filepath)

            for record in dataset:
                try:
                    parsed = self.parse_augmented_record(record.numpy())
                    stats.total_scenarios += 1

                    # Count violations in this scenario
                    num_violations = int(np.sum(parsed["violations"]))

                    if num_violations > 0:
                        stats.scenarios_with_violations += 1
                    if num_violations > 1:
                        stats.scenarios_with_multiple_violations += 1

                    # Track violation distribution
                    if num_violations not in stats.violation_distribution:
                        stats.violation_distribution[num_violations] = 0
                    stats.violation_distribution[num_violations] += 1

                    # Update per-rule statistics using rule_ids from the record
                    record_rule_ids = parsed["rule_ids"]
                    for i, rule_id in enumerate(record_rule_ids):
                        if rule_id not in stats.rule_stats:
                            stats.rule_stats[rule_id] = RuleStatistics(rule_id=rule_id)
                        if i < len(parsed["applicability"]):
                            rule_stat = stats.rule_stats[rule_id]
                            rule_stat.total_scenarios += 1

                            if parsed["applicability"][i]:
                                rule_stat.applicable_count += 1

                                if parsed["violations"][i]:
                                    rule_stat.violation_count += 1
                                    severity = float(parsed["severity"][i])
                                    rule_stat.severity_sum += severity
                                    rule_stat.severity_values.append(severity)

                except Exception as e:
                    # Skip malformed records
                    continue

        return stats

    def test_all_datasets(self, augmented_dir: str) -> Dict[str, DatasetStatistics]:
        """Test all augmented datasets in the directory."""
        results = {}

        # Find dataset directories
        for entry in os.listdir(augmented_dir):
            dataset_path = os.path.join(augmented_dir, entry)
            if os.path.isdir(dataset_path):
                print(f"\nTesting dataset: {entry}")
                stats = self.test_dataset(dataset_path, entry)
                results[entry] = stats

        return results

    def generate_report(
        self, results: Dict[str, DatasetStatistics], output_path: str
    ) -> str:
        """Generate a comprehensive markdown report."""

        lines = []
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Header
        lines.append("# Comprehensive Rule Evaluation Report")
        lines.append("")
        lines.append(f"**Generated:** {timestamp}")
        lines.append(f"**Sample Ratio:** {self.sample_ratio*100:.0f}%")
        lines.append(f"**Random Seed:** {self.seed}")
        lines.append("")

        # Executive Summary
        lines.append("## Executive Summary")
        lines.append("")

        total_scenarios = sum(s.total_scenarios for s in results.values())
        total_with_violations = sum(
            s.scenarios_with_violations for s in results.values()
        )
        total_files = sum(s.total_files for s in results.values())
        sampled_files = sum(s.sampled_files for s in results.values())

        lines.append(
            f"- **Total Files Analyzed:** {sampled_files} of {total_files} "
            f"({sampled_files/total_files*100:.1f}%)"
            if total_files > 0
            else "N/A"
        )
        lines.append(f"- **Total Scenarios Tested:** {total_scenarios:,}")
        lines.append(
            f"- **Scenarios with Violations:** {total_with_violations:,} "
            f"({total_with_violations/total_scenarios*100:.1f}%)"
            if total_scenarios > 0
            else "N/A"
        )
        lines.append("")

        # Dataset-specific summaries
        lines.append("### Dataset Summary")
        lines.append("")
        lines.append(
            "| Dataset | Files (sampled/total) | Scenarios | With Violations | Violation Rate |"
        )
        lines.append(
            "|---------|----------------------|-----------|-----------------|----------------|"
        )

        for name, stats in sorted(results.items()):
            viol_rate = (
                f"{stats.violation_rate:.2f}%" if stats.total_scenarios > 0 else "N/A"
            )
            lines.append(
                f"| {name} | {stats.sampled_files}/{stats.total_files} | "
                f"{stats.total_scenarios:,} | {stats.scenarios_with_violations:,} | {viol_rate} |"
            )
        lines.append("")

        # Detailed Rule Analysis
        lines.append("## Detailed Rule Analysis")
        lines.append("")

        # Aggregate rule statistics across all datasets
        agg_rules = {
            rule_id: RuleStatistics(rule_id=rule_id) for rule_id in self.RULE_ORDER
        }

        for stats in results.values():
            for rule_id, rule_stat in stats.rule_stats.items():
                agg = agg_rules[rule_id]
                agg.total_scenarios += rule_stat.total_scenarios
                agg.applicable_count += rule_stat.applicable_count
                agg.violation_count += rule_stat.violation_count
                agg.severity_values.extend(rule_stat.severity_values)

        # Group by level
        levels = defaultdict(list)
        for rule_id in self.RULE_ORDER:
            level = rule_id.split(".")[0]
            levels[level].append(rule_id)

        for level in sorted(levels.keys(), key=lambda x: int(x[1:])):
            level_name = self.RULE_LEVELS.get(level, level)
            lines.append(f"### {level}: {level_name}")
            lines.append("")
            lines.append(
                "| Rule ID | Rule Name | Applicable | Violated | App. Rate | Viol. Rate | Mean Sev. | Max Sev. |"
            )
            lines.append(
                "|---------|-----------|------------|----------|-----------|------------|-----------|----------|"
            )

            for rule_id in levels[level]:
                rule = agg_rules[rule_id]
                rule_name = self.RULE_NAMES.get(rule_id, "Unknown")

                app_rate = f"{rule.applicability_rate:.1f}%"
                viol_rate = (
                    f"{rule.violation_rate:.1f}%"
                    if rule.applicable_count > 0
                    else "N/A"
                )
                mean_sev = (
                    f"{rule.mean_severity:.3f}" if rule.severity_values else "N/A"
                )
                max_sev = f"{rule.max_severity:.3f}" if rule.severity_values else "N/A"

                lines.append(
                    f"| {rule_id} | {rule_name} | {rule.applicable_count:,} | "
                    f"{rule.violation_count:,} | {app_rate} | {viol_rate} | {mean_sev} | {max_sev} |"
                )
            lines.append("")

        # Violation Distribution
        lines.append("## Violation Distribution")
        lines.append("")
        lines.append("Distribution of violations per scenario:")
        lines.append("")

        # Aggregate violation distribution
        agg_dist = defaultdict(int)
        for stats in results.values():
            for count, freq in stats.violation_distribution.items():
                agg_dist[count] += freq

        lines.append("| Violations per Scenario | Count | Percentage |")
        lines.append("|------------------------|-------|------------|")

        for count in sorted(agg_dist.keys()):
            freq = agg_dist[count]
            pct = freq / total_scenarios * 100 if total_scenarios > 0 else 0
            lines.append(f"| {count} | {freq:,} | {pct:.2f}% |")
        lines.append("")

        # Top Violated Rules
        lines.append("## Top Violated Rules")
        lines.append("")
        lines.append("Rules ranked by violation count:")
        lines.append("")

        sorted_rules = sorted(
            agg_rules.values(), key=lambda r: r.violation_count, reverse=True
        )

        lines.append("| Rank | Rule ID | Rule Name | Violations | Violation Rate |")
        lines.append("|------|---------|-----------|------------|----------------|")

        for i, rule in enumerate(sorted_rules[:10], 1):
            rule_name = self.RULE_NAMES.get(rule.rule_id, "Unknown")
            viol_rate = (
                f"{rule.violation_rate:.1f}%" if rule.applicable_count > 0 else "N/A"
            )
            lines.append(
                f"| {i} | {rule.rule_id} | {rule_name} | "
                f"{rule.violation_count:,} | {viol_rate} |"
            )
        lines.append("")

        # Most Applicable Rules
        lines.append("## Most Applicable Rules")
        lines.append("")
        lines.append("Rules ranked by applicability rate:")
        lines.append("")

        sorted_by_app = sorted(
            agg_rules.values(), key=lambda r: r.applicability_rate, reverse=True
        )

        lines.append(
            "| Rank | Rule ID | Rule Name | Applicable Scenarios | App. Rate |"
        )
        lines.append("|------|---------|-----------|---------------------|-----------|")

        for i, rule in enumerate(sorted_by_app[:10], 1):
            rule_name = self.RULE_NAMES.get(rule.rule_id, "Unknown")
            lines.append(
                f"| {i} | {rule.rule_id} | {rule_name} | "
                f"{rule.applicable_count:,} | {rule.applicability_rate:.1f}% |"
            )
        lines.append("")

        # Severity Analysis
        lines.append("## Severity Analysis")
        lines.append("")
        lines.append("Severity statistics for rules with violations:")
        lines.append("")

        rules_with_severity = [r for r in agg_rules.values() if r.severity_values]
        sorted_by_sev = sorted(
            rules_with_severity, key=lambda r: r.mean_severity, reverse=True
        )

        if sorted_by_sev:
            lines.append(
                "| Rule ID | Rule Name | Mean Severity | Std Dev | Max Severity | Violations |"
            )
            lines.append(
                "|---------|-----------|---------------|---------|--------------|------------|"
            )

            for rule in sorted_by_sev:
                rule_name = self.RULE_NAMES.get(rule.rule_id, "Unknown")
                lines.append(
                    f"| {rule.rule_id} | {rule_name} | {rule.mean_severity:.4f} | "
                    f"{rule.std_severity:.4f} | {rule.max_severity:.4f} | {rule.violation_count:,} |"
                )
            lines.append("")
        else:
            lines.append("*No severity data available.*")
            lines.append("")

        # Per-Dataset Breakdown
        lines.append("## Per-Dataset Breakdown")
        lines.append("")

        for name, stats in sorted(results.items()):
            lines.append(f"### {name}")
            lines.append("")
            lines.append(
                f"- **Files Sampled:** {stats.sampled_files} of {stats.total_files}"
            )
            lines.append(f"- **Scenarios:** {stats.total_scenarios:,}")
            lines.append(
                f"- **With Violations:** {stats.scenarios_with_violations:,} "
                f"({stats.violation_rate:.2f}%)"
            )
            lines.append(
                f"- **With Multiple Violations:** {stats.scenarios_with_multiple_violations:,}"
            )
            lines.append("")

            # Top 5 violated rules for this dataset
            sorted_dataset_rules = sorted(
                stats.rule_stats.values(), key=lambda r: r.violation_count, reverse=True
            )[:5]

            if any(r.violation_count > 0 for r in sorted_dataset_rules):
                lines.append("**Top Violated Rules:**")
                lines.append("")
                lines.append("| Rule ID | Violations | Rate |")
                lines.append("|---------|------------|------|")
                for rule in sorted_dataset_rules:
                    if rule.violation_count > 0:
                        rate = f"{rule.violation_rate:.1f}%"
                        lines.append(
                            f"| {rule.rule_id} | {rule.violation_count:,} | {rate} |"
                        )
                lines.append("")

        # Methodology
        lines.append("## Methodology")
        lines.append("")
        lines.append("### Sampling Strategy")
        lines.append("")
        lines.append(
            f"- Random sampling of {self.sample_ratio*100:.0f}% of TFRecord files"
        )
        lines.append(f"- Random seed: {self.seed} (for reproducibility)")
        lines.append("- All scenarios within sampled files are analyzed")
        lines.append("")

        lines.append("### Rule Evaluation")
        lines.append("")
        lines.append("Each augmented TFRecord contains:")
        lines.append(
            "- `rule/applicability`: Boolean array indicating if each rule applies"
        )
        lines.append("- `rule/violations`: Boolean array indicating rule violations")
        lines.append("- `rule/severity`: Float array with violation severity scores")
        lines.append("- `rule/ids`: String array with rule identifiers")
        lines.append("")

        lines.append("### Metrics")
        lines.append("")
        lines.append("- **Applicability Rate:** % of scenarios where the rule applies")
        lines.append("- **Violation Rate:** % of applicable scenarios with violations")
        lines.append("- **Severity:** Magnitude of violation (rule-specific)")
        lines.append("")

        # Footer
        lines.append("---")
        lines.append("")
        lines.append(
            f"*Report generated by waymo_rule_eval.augmentation.comprehensive_test*"
        )
        lines.append(f"*Timestamp: {timestamp}*")

        report = "\n".join(lines)

        # Write report
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            f.write(report)

        print(f"\nReport saved to: {output_path}")

        return report


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive testing of augmented Waymo scenarios"
    )
    parser.add_argument(
        "--augmented-dir",
        required=True,
        help="Directory containing augmented scenario datasets",
    )
    parser.add_argument(
        "--sample-ratio",
        type=float,
        default=0.2,
        help="Fraction of files to sample (default: 0.2, minimum: 0.2)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output path for the comprehensive report (markdown)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Comprehensive Testing of Augmented Waymo Scenarios")
    print("=" * 60)
    print(f"Augmented Directory: {args.augmented_dir}")
    print(f"Sample Ratio: {args.sample_ratio*100:.0f}%")
    print(f"Output: {args.output}")
    print(f"Seed: {args.seed}")
    print("=" * 60)

    tester = AugmentedDataTester(sample_ratio=args.sample_ratio, seed=args.seed)

    results = tester.test_all_datasets(args.augmented_dir)

    if not results:
        print("No datasets found to test!")
        return

    tester.generate_report(results, args.output)

    print("\n" + "=" * 60)
    print("Testing Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
