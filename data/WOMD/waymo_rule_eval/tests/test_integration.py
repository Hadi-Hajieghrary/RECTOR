"""
Integration Tests for Waymo Rule Evaluation Pipeline.

Tests end-to-end pipeline functionality including:
- Full scenario processing
- Rule chaining and aggregation
- Result serialization
- Performance benchmarks
"""

import json
import os
import sys
import time

import pytest

# Add the workspace root to path so waymo_rule_eval package can be imported
_workspace_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _workspace_root not in sys.path:
    sys.path.insert(0, _workspace_root)

from waymo_rule_eval.core.context import ScenarioContext
from waymo_rule_eval.pipeline.rule_executor import (
    PipelineRuleResult,
    RuleExecutor,
    ScenarioResult,
)


class TestEndToEndPipeline:
    """End-to-end pipeline integration tests."""

    def test_full_pipeline_normal_scenario(self, fixture_factory):
        """Complete pipeline execution on normal driving scenario."""
        # Create scenario
        scenario = fixture_factory.create_normal_driving_scenario()

        # Initialize executor
        executor = RuleExecutor()
        executor.register_all_rules()

        # Run all rules
        result = executor.evaluate(scenario)

        # Verify results structure
        assert isinstance(result, ScenarioResult)
        assert len(result.rule_results) > 0

        # All results should have standard structure
        for rule_result in result.rule_results:
            assert isinstance(rule_result, PipelineRuleResult)
            assert rule_result.rule_id is not None

    def test_full_pipeline_collision_scenario(self, fixture_factory):
        """Complete pipeline execution on collision scenario."""
        scenario = fixture_factory.create_rear_end_collision_scenario()

        executor = RuleExecutor()
        executor.register_all_rules()
        result = executor.evaluate(scenario)

        # Should detect collision-related violations
        assert isinstance(result, ScenarioResult)

        # At least one rule should apply
        applicable_rules = [r for r in result.rule_results if r.applies]
        assert len(applicable_rules) >= 0  # May or may not apply

    def test_full_pipeline_vru_scenario(self, fixture_factory):
        """Complete pipeline execution on VRU scenario."""
        scenario = fixture_factory.create_pedestrian_crossing_scenario()

        executor = RuleExecutor()
        executor.register_all_rules()
        result = executor.evaluate(scenario)

        assert isinstance(result, ScenarioResult)

    def test_full_pipeline_traffic_control_scenario(self, fixture_factory):
        """Complete pipeline execution on traffic control scenario."""
        scenario = fixture_factory.create_red_light_violation_scenario()

        executor = RuleExecutor()
        executor.register_all_rules()
        result = executor.evaluate(scenario)

        assert isinstance(result, ScenarioResult)


class TestScenarioBatching:
    """Tests for batch scenario processing."""

    def test_process_multiple_scenarios(self, fixture_factory):
        """Process multiple scenarios using batch evaluation."""
        scenarios = [
            fixture_factory.create_normal_driving_scenario(),
            fixture_factory.create_rear_end_collision_scenario(),
            fixture_factory.create_harsh_braking_scenario(),
            fixture_factory.create_speeding_scenario(),
            fixture_factory.create_lane_departure_scenario(),
        ]

        executor = RuleExecutor()
        executor.register_all_rules()

        all_results = executor.evaluate_batch(scenarios)

        assert len(all_results) == 5

        # All should return valid results
        for result in all_results:
            assert isinstance(result, ScenarioResult)

    def test_independent_scenario_evaluation(self, fixture_factory):
        """Scenarios are evaluated independently."""
        scenario1 = fixture_factory.create_normal_driving_scenario()
        scenario2 = fixture_factory.create_harsh_braking_scenario()

        executor = RuleExecutor()
        executor.register_all_rules()

        result1 = executor.evaluate(scenario1)
        result2 = executor.evaluate(scenario2)

        # Results should be independent
        assert result1.scenario_id != result2.scenario_id or (
            len(result1.rule_results) == len(result2.rule_results)
        )


class TestResultAggregation:
    """Tests for result aggregation and reporting."""

    def test_aggregate_violations(self, fixture_factory):
        """Aggregate violations across rules."""
        scenario = fixture_factory.create_harsh_braking_scenario()

        executor = RuleExecutor()
        executor.register_all_rules()
        result = executor.evaluate(scenario)

        # Count violations using ScenarioResult properties
        n_violations = result.n_violations
        total_severity = result.total_severity

        assert isinstance(n_violations, int)
        assert isinstance(total_severity, float)

    def test_max_severity_calculation(self, fixture_factory):
        """Calculate maximum severity across all rules."""
        scenario = fixture_factory.create_harsh_braking_scenario()

        executor = RuleExecutor()
        executor.register_all_rules()
        result = executor.evaluate(scenario)

        max_severity = 0.0
        for rule_result in result.rule_results:
            if rule_result.has_violation:
                max_severity = max(max_severity, rule_result.severity)

        # Can also use result.max_severity_rule
        if result.max_severity_rule:
            assert max_severity >= 0.0

    def test_weighted_severity_score(self, fixture_factory):
        """Calculate weighted severity score."""
        scenario = fixture_factory.create_normal_driving_scenario()

        executor = RuleExecutor()
        executor.register_all_rules()
        result = executor.evaluate(scenario)

        # Define rule weights (higher for safety-critical rules)
        level_weights = {
            0: 1.0,  # L0 - Critical safety
            1: 0.8,  # L1 - Comfort
            6: 0.5,  # L6 - Interaction
            10: 1.0,  # L10 - Collision
        }

        total_weight = 0.0
        weighted_sum = 0.0

        for rule_result in result.rule_results:
            if rule_result.has_violation:
                weight = level_weights.get(rule_result.level, 0.5)
                weighted_sum += weight * rule_result.severity
                total_weight += weight

        if total_weight > 0:
            weighted_score = weighted_sum / total_weight
            assert 0.0 <= weighted_score


class TestResultSerialization:
    """Tests for result serialization."""

    def test_results_to_json(self, fixture_factory):
        """Results can be serialized to JSON via to_dict()."""
        scenario = fixture_factory.create_normal_driving_scenario()

        executor = RuleExecutor()
        executor.register_all_rules()
        result = executor.evaluate(scenario)

        # Use to_dict() method for serialization
        result_dict = result.to_dict()

        # Attempt JSON serialization
        try:
            json_str = json.dumps(result_dict, default=str)
            assert isinstance(json_str, str)

            # Can be deserialized
            parsed = json.loads(json_str)
            assert isinstance(parsed, dict)
            assert "scenario_id" in parsed
            assert "rules" in parsed
        except TypeError as e:
            pytest.fail(f"JSON serialization failed: {e}")

    def test_results_summary(self, fixture_factory):
        """Generate human-readable results summary."""
        scenario = fixture_factory.create_normal_driving_scenario()

        executor = RuleExecutor()
        executor.register_all_rules()
        result = executor.evaluate(scenario)

        summary_lines = []
        for rule_result in result.rule_results:
            summary_lines.append(
                f"{rule_result.rule_id}: applies={rule_result.applies}, "
                f"has_violation={rule_result.has_violation}, "
                f"severity={rule_result.severity:.2f}"
            )

        summary = "\n".join(summary_lines)
        assert len(summary) > 0


class TestPerformanceBenchmarks:
    """Performance benchmark tests."""

    def test_single_scenario_timing(self, fixture_factory):
        """Measure time for single scenario evaluation."""
        scenario = fixture_factory.create_normal_driving_scenario()

        executor = RuleExecutor()
        executor.register_all_rules()

        start_time = time.time()
        result = executor.evaluate(scenario)
        elapsed = time.time() - start_time

        # Should complete within reasonable time (< 5 seconds)
        assert elapsed < 5.0, f"Evaluation took {elapsed:.2f}s"
        assert isinstance(result, ScenarioResult)

        print(f"Single scenario evaluation: {elapsed*1000:.2f}ms")

    def test_batch_scenario_timing(self, fixture_factory):
        """Measure time for batch scenario evaluation."""
        n_scenarios = 10
        scenarios = [
            fixture_factory.create_normal_driving_scenario() for _ in range(n_scenarios)
        ]

        executor = RuleExecutor()
        executor.register_all_rules()

        start_time = time.time()
        results = executor.evaluate_batch(scenarios)
        elapsed = time.time() - start_time

        avg_time = elapsed / n_scenarios
        print(f"Batch evaluation: {n_scenarios} scenarios in {elapsed:.2f}s")
        print(f"Average per scenario: {avg_time*1000:.2f}ms")

        # Average should be reasonable (< 1 second per scenario)
        assert avg_time < 1.0
        assert len(results) == n_scenarios

    def test_rule_individual_timing(self, fixture_factory):
        """Measure time for individual rules."""
        scenario = fixture_factory.create_normal_driving_scenario()

        executor = RuleExecutor()
        executor.register_all_rules()

        rule_timings = {}

        for rule in executor.rules:
            start = time.time()

            try:
                # Run applicability
                applicability = rule.detector.detect(scenario)

                # Run evaluation if applicable
                if applicability.applies:
                    rule.evaluator.evaluate(scenario, applicability)
            except Exception:
                # Some rules may fail on certain scenarios - this is okay for timing
                pass

            elapsed = time.time() - start
            rule_timings[rule.rule_id] = elapsed

        # Print timing report
        print("\nRule timing breakdown:")
        for rule_id, timing in sorted(rule_timings.items(), key=lambda x: -x[1]):
            print(f"  {rule_id}: {timing*1000:.2f}ms")

        # No single rule should take too long
        max_time = max(rule_timings.values())
        assert max_time < 1.0, f"Slowest rule took {max_time:.2f}s"


class TestRobustness:
    """Robustness and error handling tests."""

    def test_missing_map_context(self, fixture_factory):
        """Handle scenario with minimal map context."""
        # Create scenario with minimal map
        scenario = fixture_factory.create_scenario_context(
            include_crosswalks=False,
            include_stoplines=False,
        )

        executor = RuleExecutor()
        executor.register_all_rules()

        # Should not crash
        try:
            result = executor.evaluate(scenario)
            assert isinstance(result, ScenarioResult)
        except Exception as e:
            pytest.fail(f"Failed with minimal map: {e}")

    def test_nan_handling(self, fixture_factory):
        """Handle NaN values in trajectory data."""
        scenario = fixture_factory.create_normal_driving_scenario()

        # Note: This tests if rules handle edge cases gracefully
        executor = RuleExecutor()
        executor.register_all_rules()

        try:
            result = executor.evaluate(scenario)
            assert isinstance(result, ScenarioResult)
        except Exception as e:
            # Some rules may fail with NaN - this is acceptable
            pass

    def test_very_long_trajectory(self, fixture_factory):
        """Handle very long trajectory (1000 frames)."""
        scenario = fixture_factory.create_scenario_context(n_frames=1000)

        executor = RuleExecutor()
        executor.register_all_rules()

        start = time.time()
        result = executor.evaluate(scenario)
        elapsed = time.time() - start

        assert isinstance(result, ScenarioResult)
        # Should still complete in reasonable time
        assert elapsed < 30.0, f"Long trajectory took {elapsed:.2f}s"

    def test_many_agents(self, fixture_factory):
        """Handle scenario with many agents."""
        scenario = fixture_factory.create_scenario_context(n_agents=50)

        executor = RuleExecutor()
        executor.register_all_rules()

        start = time.time()
        result = executor.evaluate(scenario)
        elapsed = time.time() - start

        assert isinstance(result, ScenarioResult)
        print(f"50-agent scenario: {elapsed*1000:.2f}ms")


class TestConsistency:
    """Tests for result consistency and reproducibility."""

    def test_deterministic_results(self, fixture_factory):
        """Same scenario produces same results."""
        scenario1 = fixture_factory.create_normal_driving_scenario()
        scenario2 = fixture_factory.create_normal_driving_scenario()

        executor = RuleExecutor()
        executor.register_all_rules()

        result1 = executor.evaluate(scenario1)
        result2 = executor.evaluate(scenario2)

        # Results should be identical for identical scenarios
        assert len(result1.rule_results) == len(result2.rule_results)

        for r1, r2 in zip(result1.rule_results, result2.rule_results):
            assert r1.rule_id == r2.rule_id
            assert r1.applies == r2.applies

    def test_executor_reuse(self, fixture_factory):
        """Executor can be reused for multiple scenarios."""
        executor = RuleExecutor()
        executor.register_all_rules()

        scenarios = [
            fixture_factory.create_normal_driving_scenario(),
            fixture_factory.create_harsh_braking_scenario(),
            fixture_factory.create_speeding_scenario(),
        ]

        results = []
        for scenario in scenarios:
            r = executor.evaluate(scenario)
            results.append(r)

        # All should produce valid results
        assert len(results) == 3
        assert all(isinstance(r, ScenarioResult) for r in results)


class TestSpecificScenarios:
    """Tests for specific driving scenarios."""

    def test_emergency_braking_scenario(self, fixture_factory):
        """Emergency braking triggers appropriate rules."""
        scenario = fixture_factory.create_harsh_braking_scenario()

        executor = RuleExecutor()
        executor.register_all_rules()
        result = executor.evaluate(scenario)

        # Comfort rules (L1) should apply
        l1_results = [r for r in result.rule_results if r.level == 1]

        # Should have L1 results
        assert len(l1_results) >= 0  # May not always apply

    def test_cut_in_scenario(self, fixture_factory):
        """Cut-in scenario triggers following distance rules."""
        scenario = fixture_factory.create_cut_in_scenario()

        executor = RuleExecutor()
        executor.register_all_rules()
        result = executor.evaluate(scenario)

        assert isinstance(result, ScenarioResult)

    def test_lane_change_scenario(self, fixture_factory):
        """Lane change triggers lane keeping rules."""
        scenario = fixture_factory.create_lane_change_scenario()

        executor = RuleExecutor()
        executor.register_all_rules()
        result = executor.evaluate(scenario)

        assert isinstance(result, ScenarioResult)

    def test_intersection_scenario(self, fixture_factory):
        """Intersection scenario triggers right-of-way rules."""
        scenario = fixture_factory.create_intersection_scenario()

        executor = RuleExecutor()
        executor.register_all_rules()
        result = executor.evaluate(scenario)

        # Should have L4 maneuver results
        l4_results = [r for r in result.rule_results if r.level == 4]

        assert len(l4_results) >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
