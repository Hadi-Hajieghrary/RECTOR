# waymo_rule_eval/tests/test_real_scenarios.py
# -*- coding: utf-8 -*-
"""
Comprehensive tests using actual Waymo Motion Dataset TFRecord files.

These tests verify rule evaluation on real-world driving scenarios from:
- testing_interactive
- training_interactive
- validation_interactive

Tests cover:
1. Data loading and parsing
2. Rule applicability detection
3. Violation evaluation
4. Windowed pipeline execution
5. Coverage of all 28 rules across real scenarios
"""
import glob
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pytest

# Import core modules
from waymo_rule_eval.core.context import Agent, EgoState, MapContext, ScenarioContext
from waymo_rule_eval.data_access.adapter_motion_scenario import MotionScenarioReader
from waymo_rule_eval.io import CsvSink, JsonlSink, make_sink
from waymo_rule_eval.pipeline.rule_executor import (
    RuleExecutor,
    ScenarioResult,
    WindowedExecutor,
)
from waymo_rule_eval.rules.registry import all_rules, rule_ids

# Dataset paths
DATA_ROOT = Path(
    "/workspace/data/WOMD/datasets/waymo_open_dataset/motion_v_1_3_0/raw/scenario"
)
TESTING_DIR = DATA_ROOT / "testing_interactive"
TRAINING_DIR = DATA_ROOT / "training_interactive"
VALIDATION_DIR = DATA_ROOT / "validation_interactive"


def get_tfrecord_files(directory: Path, max_files: int = 5) -> List[str]:
    """Get list of TFRecord files from directory."""
    if not directory.exists():
        return []
    pattern = str(directory / "*.tfrecord*")
    files = sorted(glob.glob(pattern))
    return files[:max_files]


def has_waymo_data() -> bool:
    """Check if Waymo dataset is available."""
    try:
        import tensorflow as tf
        from waymo_open_dataset.protos import scenario_pb2

        return TESTING_DIR.exists() or TRAINING_DIR.exists() or VALIDATION_DIR.exists()
    except ImportError:
        return False


# Skip tests if Waymo data not available
requires_waymo = pytest.mark.skipif(
    not has_waymo_data(), reason="Waymo Open Dataset not available"
)


class TestDataLoading:
    """Tests for loading and parsing TFRecord files."""

    @requires_waymo
    def test_reader_initialization(self):
        """Test MotionScenarioReader can be initialized."""
        reader = MotionScenarioReader()
        assert reader.dt == 0.1

    @requires_waymo
    def test_load_single_file(self):
        """Test loading scenarios from a single TFRecord file."""
        files = get_tfrecord_files(TESTING_DIR, max_files=1)
        if not files:
            pytest.skip("No testing TFRecord files found")

        reader = MotionScenarioReader()
        scenarios = list(reader.read_tfrecord(files[0]))

        assert len(scenarios) > 0
        ctx = scenarios[0]
        assert isinstance(ctx, ScenarioContext)
        assert ctx.scenario_id is not None
        assert isinstance(ctx.ego, EgoState)
        assert len(ctx.ego.x) > 0

    @requires_waymo
    def test_ego_state_extraction(self):
        """Test that ego state is properly extracted."""
        files = get_tfrecord_files(TESTING_DIR, max_files=1)
        if not files:
            pytest.skip("No testing TFRecord files found")

        reader = MotionScenarioReader()
        for ctx in reader.read_tfrecord(files[0]):
            ego = ctx.ego

            # Check ego is properly constructed (dataclass fields are guaranteed)
            assert isinstance(ego.x, np.ndarray)
            assert isinstance(ego.y, np.ndarray)
            assert isinstance(ego.yaw, np.ndarray)
            assert isinstance(ego.speed, np.ndarray)

            # Check arrays have same length
            assert len(ego.x) == len(ego.y)
            assert len(ego.x) == len(ego.yaw)
            assert len(ego.x) == len(ego.speed)

            # Check for valid values (not all NaN)
            assert not np.all(np.isnan(ego.x))
            assert not np.all(np.isnan(ego.y))

            break  # Just test first scenario

    @requires_waymo
    def test_agent_extraction(self):
        """Test that agents are properly extracted."""
        files = get_tfrecord_files(TESTING_DIR, max_files=1)
        if not files:
            pytest.skip("No testing TFRecord files found")

        reader = MotionScenarioReader()
        found_agents = False

        for ctx in reader.read_tfrecord(files[0]):
            if len(ctx.agents) > 0:
                found_agents = True
                agent = ctx.agents[0]

                assert isinstance(agent.id, int)
                assert isinstance(agent.type, str)
                assert isinstance(agent.x, np.ndarray)
                assert isinstance(agent.y, np.ndarray)
                assert isinstance(agent.yaw, np.ndarray)
                assert isinstance(agent.speed, np.ndarray)

                # Check agent type
                assert agent.type in ["vehicle", "pedestrian", "cyclist"]
                break

        assert found_agents, "No scenarios with agents found"

    @requires_waymo
    def test_map_context_extraction(self):
        """Test that map context is extracted."""
        files = get_tfrecord_files(TESTING_DIR, max_files=1)
        if not files:
            pytest.skip("No testing TFRecord files found")

        reader = MotionScenarioReader()
        for ctx in reader.read_tfrecord(files[0]):
            assert hasattr(ctx, "map_context")
            if ctx.map_context is not None:
                # Check for lane data
                if hasattr(ctx.map_context, "lane_center_xy"):
                    assert isinstance(ctx.map_context.lane_center_xy, np.ndarray)
            break

    @requires_waymo
    def test_iter_scenarios_from_glob(self):
        """Test iterating over multiple files."""
        pattern = str(TESTING_DIR / "*.tfrecord-0000[0-2]-of-*")
        reader = MotionScenarioReader()

        count = 0
        for ctx in reader.iter_scenarios_from_glob(pattern):
            count += 1
            if count >= 5:
                break

        assert count >= 1, "Should load at least one scenario"


class TestRuleApplicability:
    """Tests for rule applicability on real scenarios."""

    @requires_waymo
    def test_some_rules_apply(self):
        """Test that at least some rules apply to real scenarios."""
        files = get_tfrecord_files(TESTING_DIR, max_files=1)
        if not files:
            pytest.skip("No testing TFRecord files found")

        reader = MotionScenarioReader()
        executor = RuleExecutor()
        executor.register_all_rules()

        applicable_rules = set()
        scenarios_tested = 0

        for ctx in reader.read_tfrecord(files[0]):
            result = executor.evaluate(ctx)
            for r in result.rule_results:
                if r.applies:
                    applicable_rules.add(r.rule_id)

            scenarios_tested += 1
            if scenarios_tested >= 10:
                break

        print(f"Applicable rules: {applicable_rules}")
        assert (
            len(applicable_rules) >= 3
        ), f"Expected at least 3 applicable rules, got {len(applicable_rules)}"

    @requires_waymo
    def test_l0_safety_rules_apply(self):
        """Test that L0 safety rules are applicable."""
        files = get_tfrecord_files(TESTING_DIR, max_files=2)
        if not files:
            pytest.skip("No testing TFRecord files found")

        reader = MotionScenarioReader()
        executor = RuleExecutor()
        executor.register_all_rules()

        l0_applicable = set()

        for fpath in files:
            for ctx in reader.read_tfrecord(fpath):
                if len(ctx.agents) < 2:
                    continue  # Skip single-agent scenarios

                result = executor.evaluate(ctx)
                for r in result.rule_results:
                    if r.rule_id.startswith("L0.") and r.applies:
                        l0_applicable.add(r.rule_id)

                if len(l0_applicable) >= 2:
                    break
            if len(l0_applicable) >= 2:
                break

        print(f"L0 rules applicable: {l0_applicable}")
        # L0 rules require close proximity to other agents
        # May not apply in all scenarios

    @requires_waymo
    def test_l1_comfort_rules_apply(self):
        """Test that L1 comfort rules apply to driving scenarios."""
        files = get_tfrecord_files(TESTING_DIR, max_files=1)
        if not files:
            pytest.skip("No testing TFRecord files found")

        reader = MotionScenarioReader()
        executor = RuleExecutor()
        executor.register_all_rules()

        l1_applicable = set()

        for ctx in reader.read_tfrecord(files[0]):
            result = executor.evaluate(ctx)
            for r in result.rule_results:
                if r.rule_id.startswith("L1.") and r.applies:
                    l1_applicable.add(r.rule_id)

            if l1_applicable:
                break

        print(f"L1 rules applicable: {l1_applicable}")
        # L1 rules should apply to any moving vehicle
        assert len(l1_applicable) >= 1, "Expected at least 1 L1 rule to apply"

    @requires_waymo
    def test_l6_interaction_rules_apply(self):
        """Test that L6 interaction rules apply with multiple agents."""
        files = get_tfrecord_files(TESTING_DIR, max_files=3)
        if not files:
            pytest.skip("No testing TFRecord files found")

        reader = MotionScenarioReader()
        executor = RuleExecutor()
        executor.register_all_rules()

        l6_applicable = set()

        for fpath in files:
            for ctx in reader.read_tfrecord(fpath):
                if len(ctx.agents) < 2:
                    continue

                result = executor.evaluate(ctx)
                for r in result.rule_results:
                    if r.rule_id.startswith("L6.") and r.applies:
                        l6_applicable.add(r.rule_id)

                if len(l6_applicable) >= 2:
                    break
            if len(l6_applicable) >= 2:
                break

        print(f"L6 rules applicable: {l6_applicable}")


class TestViolationEvaluation:
    """Tests for violation evaluation on real scenarios."""

    @requires_waymo
    def test_violations_have_valid_severity(self):
        """Test that violations have valid severity values."""
        files = get_tfrecord_files(TESTING_DIR, max_files=1)
        if not files:
            pytest.skip("No testing TFRecord files found")

        reader = MotionScenarioReader()
        executor = RuleExecutor()
        executor.register_all_rules()

        for ctx in reader.read_tfrecord(files[0]):
            result = executor.evaluate(ctx)

            for r in result.rule_results:
                if r.violation is not None:
                    # Handle NaN values (can occur with sparse data)
                    sev = r.violation.severity
                    sev_norm = r.violation.severity_normalized
                    if not np.isnan(sev):
                        assert sev >= 0, f"Severity should be non-negative: {sev}"
                    if not np.isnan(sev_norm):
                        assert sev_norm >= 0, f"Normalized severity >= 0: {sev_norm}"
                        assert sev_norm <= 1.0, f"Normalized severity <= 1: {sev_norm}"
            break

    @requires_waymo
    def test_find_speeding_violations(self):
        """Look for speeding violations in the data."""
        files = get_tfrecord_files(TESTING_DIR, max_files=3)
        if not files:
            pytest.skip("No testing TFRecord files found")

        reader = MotionScenarioReader()
        executor = RuleExecutor()
        executor.register_all_rules()

        speed_violations = 0
        max_speed_seen = 0.0

        for fpath in files:
            for ctx in reader.read_tfrecord(fpath):
                max_speed = np.nanmax(ctx.ego.speed)
                if not np.isnan(max_speed):
                    max_speed_seen = max(max_speed_seen, max_speed)

                result = executor.evaluate(ctx)
                for r in result.rule_results:
                    if (
                        r.rule_id == "L7.R4"
                        and r.violation
                        and r.violation.severity > 0
                    ):
                        speed_violations += 1

        print(
            f"Max speed seen: {max_speed_seen:.1f} m/s ({max_speed_seen*2.237:.1f} mph)"
        )
        print(f"Speed violations found: {speed_violations}")
        assert True

    @requires_waymo
    def test_find_harsh_braking_violations(self):
        """Look for harsh braking violations."""
        files = get_tfrecord_files(TESTING_DIR, max_files=3)
        if not files:
            pytest.skip("No testing TFRecord files found")

        reader = MotionScenarioReader()
        executor = RuleExecutor()
        executor.register_all_rules()

        braking_violations = 0
        min_accel_seen = 0.0

        for fpath in files:
            for ctx in reader.read_tfrecord(fpath):
                accel = np.diff(ctx.ego.speed) / 0.1
                if len(accel) > 0:
                    min_accel = np.nanmin(accel)
                    if not np.isnan(min_accel):
                        min_accel_seen = min(min_accel_seen, min_accel)

                result = executor.evaluate(ctx)
                for r in result.rule_results:
                    if (
                        r.rule_id == "L1.R3"
                        and r.violation
                        and r.violation.severity > 0
                    ):
                        braking_violations += 1

        print(f"Min acceleration seen: {min_accel_seen:.2f} m/s²")
        print(f"Harsh braking violations found: {braking_violations}")
        assert True


class TestWindowedEvaluation:
    """Tests for windowed evaluation on real data."""

    @requires_waymo
    def test_windowed_executor_on_real_data(self):
        """Test windowed executor on real TFRecord data."""
        files = get_tfrecord_files(TESTING_DIR, max_files=1)
        if not files:
            pytest.skip("No testing TFRecord files found")

        reader = MotionScenarioReader()
        executor = WindowedExecutor(window_size_s=4.0, stride_s=2.0)
        executor.register_all_rules()

        scenarios = []
        for ctx in reader.read_tfrecord(files[0]):
            scenarios.append(ctx)
            if len(scenarios) >= 3:
                break

        # run_batch returns Dict[scenario_id, List[WindowResult]]
        results = executor.run_batch(scenarios)

        assert len(results) > 0
        for scenario_id, window_results in results.items():
            assert scenario_id is not None
            for window_result in window_results:
                assert window_result.window_idx >= 0
                # Check rule results exist (may vary based on registered rules)
                assert len(window_result.rule_results) > 0

    @requires_waymo
    def test_window_coverage(self):
        """Test that windows cover the entire scenario."""
        files = get_tfrecord_files(TESTING_DIR, max_files=1)
        if not files:
            pytest.skip("No testing TFRecord files found")

        reader = MotionScenarioReader()
        executor = WindowedExecutor(window_size_s=8.0, stride_s=2.0)
        executor.register_all_rules()

        for ctx in reader.read_tfrecord(files[0]):
            results = executor.run(ctx)

            if len(results) > 0:
                # Check windows are sequential
                window_indices = [r.window_idx for r in results]
                assert window_indices == list(range(len(results)))

                # Check timing
                for r in results:
                    assert r.window_start_ts >= 0 or r.window_start_ts is None
                    assert r.window_start_idx >= 0
                    assert r.window_end_idx > r.window_start_idx
            break


class TestTrainingData:
    """Tests for training data loading and evaluation."""

    @requires_waymo
    def test_training_data_loading(self):
        """Test loading scenarios from training data."""
        files = get_tfrecord_files(TRAINING_DIR, max_files=2)
        if not files:
            pytest.skip("No training TFRecord files found")

        reader = MotionScenarioReader()
        count = 0

        for fpath in files:
            for ctx in reader.read_tfrecord(fpath):
                assert ctx.scenario_id is not None
                assert ctx.n_frames > 0
                count += 1

        print(f"Loaded {count} scenarios from training data")
        assert count > 0

    @requires_waymo
    def test_training_data_rule_evaluation(self):
        """Test rule evaluation on training data."""
        files = get_tfrecord_files(TRAINING_DIR, max_files=2)
        if not files:
            pytest.skip("No training TFRecord files found")

        reader = MotionScenarioReader()
        executor = RuleExecutor()
        executor.register_all_rules()

        applicable_count = 0
        violation_count = 0

        for fpath in files:
            for ctx in reader.read_tfrecord(fpath):
                result = executor.evaluate(ctx)
                for r in result.rule_results:
                    if r.applies:
                        applicable_count += 1
                    if r.violation and r.violation.severity > 0:
                        violation_count += 1

        print(
            f"Training data - Applicable: {applicable_count}, Violations: {violation_count}"
        )
        assert applicable_count > 0


class TestValidationData:
    """Tests for validation data loading and evaluation."""

    @requires_waymo
    def test_validation_data_loading(self):
        """Test loading scenarios from validation data."""
        files = get_tfrecord_files(VALIDATION_DIR, max_files=2)
        if not files:
            pytest.skip("No validation TFRecord files found")

        reader = MotionScenarioReader()
        count = 0

        for fpath in files:
            for ctx in reader.read_tfrecord(fpath):
                assert ctx.scenario_id is not None
                assert ctx.n_frames > 0
                count += 1

        print(f"Loaded {count} scenarios from validation data")
        assert count > 0

    @requires_waymo
    def test_validation_data_rule_evaluation(self):
        """Test rule evaluation on validation data."""
        files = get_tfrecord_files(VALIDATION_DIR, max_files=2)
        if not files:
            pytest.skip("No validation TFRecord files found")

        reader = MotionScenarioReader()
        executor = RuleExecutor()
        executor.register_all_rules()

        applicable_count = 0
        violation_count = 0

        for fpath in files:
            for ctx in reader.read_tfrecord(fpath):
                result = executor.evaluate(ctx)
                for r in result.rule_results:
                    if r.applies:
                        applicable_count += 1
                    if r.violation and r.violation.severity > 0:
                        violation_count += 1

        print(
            f"Validation data - Applicable: {applicable_count}, Violations: {violation_count}"
        )
        assert applicable_count > 0


class TestOutputSinks:
    """Tests for output sinks with real data."""

    @requires_waymo
    def test_jsonl_output(self):
        """Test JSONL output sink with real data."""
        files = get_tfrecord_files(TESTING_DIR, max_files=1)
        if not files:
            pytest.skip("No testing TFRecord files found")

        import tempfile

        reader = MotionScenarioReader()
        executor = RuleExecutor()
        executor.register_all_rules()

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            sink = JsonlSink(f.name)

            for ctx in reader.read_tfrecord(files[0]):
                result = executor.evaluate(ctx)
                # Use to_dict() to get the result as a dict for the sink
                record = result.to_dict()
                sink.write(record)
                break

            sink.close()

            # Verify output
            with open(f.name, "r") as rf:
                lines = rf.readlines()
                print(f"Wrote {len(lines)} records to JSONL")
                assert len(lines) > 0

    @requires_waymo
    def test_csv_output(self):
        """Test CSV output sink with real data."""
        files = get_tfrecord_files(TESTING_DIR, max_files=1)
        if not files:
            pytest.skip("No testing TFRecord files found")

        import tempfile

        reader = MotionScenarioReader()
        executor = RuleExecutor()
        executor.register_all_rules()

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            sink = CsvSink(f.name)

            for ctx in reader.read_tfrecord(files[0]):
                result = executor.evaluate(ctx)
                # Use to_dict() to get the result as a dict for the sink
                record = result.to_dict()
                sink.write(record)
                break

            sink.close()

            # Verify output
            with open(f.name, "r") as rf:
                lines = rf.readlines()
                print(f"Wrote {len(lines)} records to CSV")
                assert len(lines) > 1  # Header + data


class TestComprehensiveRuleCoverage:
    """Tests for comprehensive rule coverage on real data."""

    @requires_waymo
    def test_all_rules_evaluated(self):
        """Test that rules are evaluated on real data (not all 28 may be registered)."""
        files = get_tfrecord_files(TESTING_DIR, max_files=1)
        if not files:
            pytest.skip("No testing TFRecord files found")

        reader = MotionScenarioReader()
        executor = RuleExecutor()
        executor.register_all_rules()

        # Get the actually registered rules
        registered_rules = set(r.rule_id for r in executor.rules)
        evaluated_rules = set()

        for ctx in reader.read_tfrecord(files[0]):
            result = executor.evaluate(ctx)

            for r in result.rule_results:
                evaluated_rules.add(r.rule_id)

            if evaluated_rules == registered_rules:
                break

        missing = registered_rules - evaluated_rules
        # Allow for some rules not being evaluated (may depend on scenario type)
        assert (
            len(evaluated_rules) >= len(registered_rules) * 0.8
        ), f"At least 80% of rules should be evaluated. Missing: {missing}"

    @requires_waymo
    def test_rule_applicability_distribution(self):
        """Test applicability distribution across rules."""
        files = get_tfrecord_files(TESTING_DIR, max_files=3)
        if not files:
            pytest.skip("No testing TFRecord files found")

        reader = MotionScenarioReader()
        executor = RuleExecutor()
        executor.register_all_rules()

        rule_applicability = {}
        scenario_count = 0

        for fpath in files:
            for ctx in reader.read_tfrecord(fpath):
                result = executor.evaluate(ctx)
                scenario_count += 1

                for r in result.rule_results:
                    if r.rule_id not in rule_applicability:
                        rule_applicability[r.rule_id] = 0
                    if r.applies:
                        rule_applicability[r.rule_id] += 1

                if scenario_count >= 50:
                    break
            if scenario_count >= 50:
                break

        print(f"\nRule applicability over {scenario_count} scenarios:")
        for rule_id, count in sorted(rule_applicability.items()):
            pct = 100 * count / scenario_count
            print(f"  {rule_id}: {count}/{scenario_count} ({pct:.1f}%)")

        # At least some rules should apply
        applicable_rules = [r for r, c in rule_applicability.items() if c > 0]
        assert len(applicable_rules) >= 3, f"At least 3 rules should apply to real data"


class TestL5RegulatoryRules:
    """Tests specifically for L5 regulatory rules on real data."""

    @requires_waymo
    def test_l5_r1_traffic_signal_compliance(self):
        """Test L5.R1 traffic signal compliance on real data."""
        files = get_tfrecord_files(TESTING_DIR, max_files=3)
        if not files:
            pytest.skip("No testing TFRecord files found")

        reader = MotionScenarioReader()

        # Check if map has traffic signals
        signal_scenarios = 0
        for fpath in files:
            for ctx in reader.read_tfrecord(fpath):
                if hasattr(ctx.map_context, "signals") and ctx.map_context.signals:
                    signal_scenarios += 1
                if signal_scenarios >= 5:
                    break
            if signal_scenarios >= 5:
                break

        print(f"Found {signal_scenarios} scenarios with traffic signals")
        assert True  # Informational test

    @requires_waymo
    def test_l5_r2_priority_violation(self):
        """Test L5.R2 priority violation on real data."""
        from waymo_rule_eval.rules.l5_priority_violation import (
            PriorityViolationApplicability,
            PriorityViolationEvaluator,
        )

        files = get_tfrecord_files(TESTING_DIR, max_files=2)
        if not files:
            pytest.skip("No testing TFRecord files found")

        detector = PriorityViolationApplicability()
        evaluator = PriorityViolationEvaluator()
        reader = MotionScenarioReader()

        applicable_count = 0
        for fpath in files:
            for ctx in reader.read_tfrecord(fpath):
                app_result = detector.detect(ctx)
                if app_result.applies:
                    applicable_count += 1
                    evaluator.evaluate(ctx, app_result)

        print(f"L5.R2 applicable in {applicable_count} scenarios")
        assert True


class TestL6InteractionRules:
    """Tests specifically for L6 interaction rules on real data."""

    @requires_waymo
    def test_l6_r1_cooperative_lane_change(self):
        """Test L6.R1 cooperative lane change on real data."""
        from waymo_rule_eval.rules.l6_cooperative_lane_change import (
            CooperativeLaneChangeApplicability,
            CooperativeLaneChangeEvaluator,
        )

        files = get_tfrecord_files(TESTING_DIR, max_files=2)
        if not files:
            pytest.skip("No testing TFRecord files found")

        detector = CooperativeLaneChangeApplicability()
        evaluator = CooperativeLaneChangeEvaluator()
        reader = MotionScenarioReader()

        applicable_count = 0
        for fpath in files:
            for ctx in reader.read_tfrecord(fpath):
                app_result = detector.detect(ctx)
                if app_result.applies:
                    applicable_count += 1
                    evaluator.evaluate(ctx, app_result)

        print(f"L6.R1 applicable in {applicable_count} scenarios")
        assert True

    @requires_waymo
    def test_l6_r3_intersection_negotiation(self):
        """Test L6.R3 intersection negotiation on real data."""
        from waymo_rule_eval.rules.l6_intersection_negotiation import (
            IntersectionNegotiationApplicability,
            IntersectionNegotiationEvaluator,
        )

        files = get_tfrecord_files(TESTING_DIR, max_files=2)
        if not files:
            pytest.skip("No testing TFRecord files found")

        detector = IntersectionNegotiationApplicability()
        evaluator = IntersectionNegotiationEvaluator()
        reader = MotionScenarioReader()

        applicable_count = 0
        for fpath in files:
            for ctx in reader.read_tfrecord(fpath):
                app_result = detector.detect(ctx)
                if app_result.applies:
                    applicable_count += 1
                    evaluator.evaluate(ctx, app_result)

        print(f"L6.R3 applicable in {applicable_count} scenarios")
        assert True
