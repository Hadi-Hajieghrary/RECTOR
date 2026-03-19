"""
Comprehensive Rule Tests.

Tests for all implemented traffic rules using RuleExecutor covering:
- Normal/compliant scenarios
- Violation scenarios
- Edge cases and boundary conditions
"""

import os
import sys

import numpy as np
import pytest

# Add the workspace root to path so waymo_rule_eval package can be imported
_workspace_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _workspace_root not in sys.path:
    sys.path.insert(0, _workspace_root)

from waymo_rule_eval.pipeline.rule_executor import (
    PipelineRuleResult,
    RuleExecutor,
    ScenarioResult,
)


class TestRuleExecutorIntegration:
    """Tests for RuleExecutor with all rules."""

    def test_executor_initialization(self):
        """RuleExecutor initializes correctly."""
        executor = RuleExecutor()

        # Before registration, no rules
        assert len(executor.rules) == 0

        # After registration, rules are loaded
        executor.register_all_rules()
        assert len(executor.rules) >= 15

    def test_executor_has_expected_rule_levels(self):
        """Executor has rules from different levels."""
        executor = RuleExecutor()
        executor.register_all_rules()

        levels = {r.level for r in executor.rules}
        assert 0 in levels  # L0 - Critical safety
        assert 1 in levels  # L1 - Comfort
        assert 10 in levels  # L10 - Collision

    def test_executor_run_all_rules(self, fixture_factory):
        """Executor runs all rules without errors."""
        executor = RuleExecutor()
        executor.register_all_rules()
        scenario = fixture_factory.create_normal_driving_scenario()

        result = executor.evaluate(scenario)

        assert isinstance(result, ScenarioResult)
        assert len(result.rule_results) > 0


class TestScenarioEvaluation:
    """Tests for evaluating different scenarios."""

    def test_normal_driving_scenario(self, fixture_factory):
        """Evaluate normal driving scenario."""
        executor = RuleExecutor()
        executor.register_all_rules()
        scenario = fixture_factory.create_normal_driving_scenario()

        result = executor.evaluate(scenario)

        assert isinstance(result, ScenarioResult)
        assert result.n_frames > 0

    def test_collision_scenario(self, fixture_factory):
        """Evaluate collision scenario."""
        executor = RuleExecutor()
        executor.register_all_rules()
        scenario = fixture_factory.create_rear_end_collision_scenario()

        result = executor.evaluate(scenario)

        assert isinstance(result, ScenarioResult)

    def test_pedestrian_scenario(self, fixture_factory):
        """Evaluate pedestrian crossing scenario."""
        executor = RuleExecutor()
        executor.register_all_rules()
        scenario = fixture_factory.create_pedestrian_crossing_scenario()

        result = executor.evaluate(scenario)

        assert isinstance(result, ScenarioResult)

    def test_harsh_braking_scenario(self, fixture_factory):
        """Evaluate harsh braking scenario."""
        executor = RuleExecutor()
        executor.register_all_rules()
        scenario = fixture_factory.create_harsh_braking_scenario()

        result = executor.evaluate(scenario)

        assert isinstance(result, ScenarioResult)

    def test_speeding_scenario(self, fixture_factory):
        """Evaluate speeding scenario."""
        executor = RuleExecutor()
        executor.register_all_rules()
        scenario = fixture_factory.create_speeding_scenario()

        result = executor.evaluate(scenario)

        assert isinstance(result, ScenarioResult)

    def test_lane_departure_scenario(self, fixture_factory):
        """Evaluate lane departure scenario."""
        executor = RuleExecutor()
        executor.register_all_rules()
        scenario = fixture_factory.create_lane_departure_scenario()

        result = executor.evaluate(scenario)

        assert isinstance(result, ScenarioResult)


class TestRuleCoverage:
    """Tests to ensure rule coverage."""

    def test_l0_rules_present(self):
        """L0 level rules are registered."""
        executor = RuleExecutor()
        executor.register_all_rules()

        l0_rules = [r for r in executor.rules if r.level == 0]
        assert len(l0_rules) >= 1

    def test_l1_rules_present(self):
        """L1 level rules are registered."""
        executor = RuleExecutor()
        executor.register_all_rules()

        l1_rules = [r for r in executor.rules if r.level == 1]
        assert len(l1_rules) >= 1

    def test_l6_rules_present(self):
        """L6 level rules are registered."""
        executor = RuleExecutor()
        executor.register_all_rules()

        l6_rules = [r for r in executor.rules if r.level == 6]
        assert len(l6_rules) >= 1

    def test_l10_rules_present(self):
        """L10 level rules are registered."""
        executor = RuleExecutor()
        executor.register_all_rules()

        l10_rules = [r for r in executor.rules if r.level == 10]
        assert len(l10_rules) >= 1


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_agent_list(self, fixture_factory):
        """Handle scenario with no other agents."""
        scenario = fixture_factory.create_scenario_context(add_agents=False)

        executor = RuleExecutor()
        executor.register_all_rules()

        # Should not crash with empty agent list
        try:
            result = executor.evaluate(scenario)
            assert isinstance(result, ScenarioResult)
        except Exception as e:
            pytest.fail(f"Failed with empty agent list: {e}")

    def test_single_frame(self, fixture_factory):
        """Handle scenario with single frame."""
        scenario = fixture_factory.create_scenario_context(n_frames=1)

        executor = RuleExecutor()
        executor.register_all_rules()

        try:
            result = executor.evaluate(scenario)
            assert isinstance(result, ScenarioResult)
        except Exception as e:
            pytest.fail(f"Failed with single frame: {e}")

    def test_stationary_ego(self, fixture_factory):
        """Handle stationary ego vehicle."""
        scenario = fixture_factory.create_stationary_scenario()

        executor = RuleExecutor()
        executor.register_all_rules()

        try:
            result = executor.evaluate(scenario)
            assert isinstance(result, ScenarioResult)
        except Exception as e:
            pytest.fail(f"Failed with stationary ego: {e}")

    def test_very_high_speed(self, fixture_factory):
        """Handle very high speeds without overflow."""
        scenario = fixture_factory.create_scenario_with_speed(100.0)

        executor = RuleExecutor()
        executor.register_all_rules()

        try:
            result = executor.evaluate(scenario)
            assert isinstance(result, ScenarioResult)
        except Exception as e:
            pytest.fail(f"Failed with very high speed: {e}")


class TestResultStructure:
    """Tests for result data structures."""

    def test_result_structure(self, fixture_factory):
        """Results have expected structure."""
        executor = RuleExecutor()
        executor.register_all_rules()
        scenario = fixture_factory.create_normal_driving_scenario()

        result = executor.evaluate(scenario)

        assert result.scenario_id is not None
        assert result.n_frames > 0

        for rule_result in result.rule_results:
            assert isinstance(rule_result, PipelineRuleResult)
            assert isinstance(rule_result.rule_id, str)
            assert isinstance(rule_result.applies, bool)

    def test_severity_normalized(self, fixture_factory):
        """Violation severity is normalized to [0, 1]."""
        executor = RuleExecutor()
        executor.register_all_rules()

        scenarios = [
            fixture_factory.create_normal_driving_scenario(),
            fixture_factory.create_harsh_braking_scenario(),
            fixture_factory.create_speeding_scenario(),
        ]

        for scenario in scenarios:
            result = executor.evaluate(scenario)

            for rule_result in result.rule_results:
                if rule_result.violation is not None:
                    sev = rule_result.violation.severity_normalized
                    assert 0 <= sev <= 1, f"Severity {sev} out of range"


class TestBatchEvaluation:
    """Tests for batch evaluation."""

    def test_evaluate_batch(self, fixture_factory):
        """Test batch evaluation of multiple scenarios."""
        executor = RuleExecutor()
        executor.register_all_rules()

        scenarios = [
            fixture_factory.create_normal_driving_scenario(),
            fixture_factory.create_harsh_braking_scenario(),
            fixture_factory.create_speeding_scenario(),
        ]

        results = executor.evaluate_batch(scenarios)

        assert len(results) == 3
        for result in results:
            assert isinstance(result, ScenarioResult)

    def test_evaluate_batch_empty(self):
        """Test batch evaluation with empty list."""
        executor = RuleExecutor()
        executor.register_all_rules()

        results = executor.evaluate_batch([])

        assert results == []


class TestRuleApplicability:
    """Tests for rule applicability detection."""

    def test_some_rules_apply(self, fixture_factory):
        """Some rules should apply to normal scenario."""
        executor = RuleExecutor()
        executor.register_all_rules()
        scenario = fixture_factory.create_normal_driving_scenario()

        result = executor.evaluate(scenario)

        # At least some rules should be applicable
        applicable_count = sum(1 for r in result.rule_results if r.applies)
        assert applicable_count > 0

    def test_pedestrian_rules_apply_when_pedestrian_present(self, fixture_factory):
        """Pedestrian rules should apply when pedestrians present."""
        executor = RuleExecutor()
        executor.register_all_rules()
        scenario = fixture_factory.create_pedestrian_crossing_scenario()

        result = executor.evaluate(scenario)

        # Find pedestrian interaction rule
        ped_rules = [r for r in result.rule_results if "pedestrian" in r.name.lower()]
        if ped_rules:
            # At least one pedestrian rule should apply
            assert any(r.applies for r in ped_rules)


class TestToDict:
    """Tests for to_dict conversion methods."""

    def test_scenario_result_to_dict(self, fixture_factory):
        """ScenarioResult.to_dict() works correctly."""
        executor = RuleExecutor()
        executor.register_all_rules()
        scenario = fixture_factory.create_normal_driving_scenario()

        result = executor.evaluate(scenario)
        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert "scenario_id" in result_dict
        assert "n_frames" in result_dict
        assert "n_agents" in result_dict
        assert "rules" in result_dict
        assert isinstance(result_dict["rules"], list)

    def test_pipeline_rule_result_to_dict(self, fixture_factory):
        """PipelineRuleResult.to_dict() works correctly."""
        executor = RuleExecutor()
        executor.register_all_rules()
        scenario = fixture_factory.create_normal_driving_scenario()

        result = executor.evaluate(scenario)

        for rule_result in result.rule_results:
            rule_dict = rule_result.to_dict()

            assert isinstance(rule_dict, dict)
            assert "rule_id" in rule_dict
            assert "level" in rule_dict
            assert "name" in rule_dict
            assert "applies" in rule_dict


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
