"""
Waymo Rule Evaluation Pipeline

A comprehensive rule evaluation system for autonomous vehicle driving scenarios
based on the Waymo Open Motion Dataset.

This package provides:
- Data loading from Waymo TFRecord files
- Efficient spatial indexing for multi-agent queries
- Rule-based evaluation for driving safety and comfort
- Two-phase evaluation: Applicability detection + Violation evaluation

Example usage:
    from waymo_rule_eval.data_access import MotionScenarioReader
    from waymo_rule_eval.pipeline import RuleExecutor

    reader = MotionScenarioReader()
    executor = RuleExecutor()
    executor.register_all_rules()

    for context in reader.read_tfrecord("path/to/file.tfrecord"):
        result = executor.evaluate(context)
        print(f"Scenario {result.scenario_id}: {result.n_violations} violations")
"""

__version__ = "1.0.0"
__author__ = "Waymo Rule Eval Team"

from .core.context import Agent, EgoState, MapContext, MapSignals, ScenarioContext
from .core.geometry import (
    compute_bumper_distance,
    compute_relative_position,
    oriented_box_corners,
    sat_collision_check,
)
from .core.temporal_spatial import FrameSpatialIndex, TemporalSpatialIndex
from .data_access.adapter_motion_scenario import (
    MotionScenarioReader,
    create_scenario_from_arrays,
)
from .pipeline.rule_executor import PipelineRuleResult, RuleExecutor, ScenarioResult
from .rules.base import ApplicabilityResult, ViolationResult

__all__ = [
    # Version
    "__version__",
    # Core context classes
    "EgoState",
    "Agent",
    "MapContext",
    "MapSignals",
    "ScenarioContext",
    # Geometry utilities
    "compute_relative_position",
    "compute_bumper_distance",
    "oriented_box_corners",
    "sat_collision_check",
    # Spatial indexing
    "TemporalSpatialIndex",
    "FrameSpatialIndex",
    # Data access
    "MotionScenarioReader",
    "create_scenario_from_arrays",
    # Pipeline
    "RuleExecutor",
    "ScenarioResult",
    "PipelineRuleResult",
    # Results
    "ApplicabilityResult",
    "ViolationResult",
]
