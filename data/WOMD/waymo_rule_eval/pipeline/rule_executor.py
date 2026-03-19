"""
Rule Executor Pipeline

Orchestrates running all rules on a scenario and collecting results.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..core.context import ScenarioContext
from ..rules.base import (
    ApplicabilityDetector,
    ApplicabilityResult,
    ViolationEvaluator,
    ViolationResult,
)

log = logging.getLogger(__name__)


@dataclass
class PipelineRuleResult:
    """Result from evaluating a single rule in the pipeline."""

    rule_id: str
    level: int
    name: str
    applies: bool
    applicability: Optional[ApplicabilityResult] = None
    violation: Optional[ViolationResult] = None
    error: Optional[str] = None

    @property
    def has_violation(self) -> bool:
        return self.violation is not None and self.violation.severity > 0

    @property
    def severity(self) -> float:
        return self.violation.severity if self.violation else 0.0

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "rule_id": self.rule_id,
            "level": self.level,
            "name": self.name,
            "applies": self.applies,
            "has_violation": self.has_violation,
            "severity": self.severity,
        }

        if self.applicability:
            result["applicability"] = {
                "confidence": self.applicability.confidence,
                "reasons": self.applicability.reasons,
                "features": self.applicability.features,
            }

        if self.violation:
            result["violation"] = {
                "severity": self.violation.severity,
                "severity_normalized": self.violation.severity_normalized,
                "measurements": self.violation.measurements,
                "explanation": self.violation.explanation,
                "frame_count": len(self.violation.frame_violations),
            }

        if self.error:
            result["error"] = self.error

        return result


@dataclass
class ScenarioResult:
    """Complete result from evaluating all rules on a scenario."""

    scenario_id: str
    n_frames: int
    n_agents: int
    rule_results: List[PipelineRuleResult] = field(default_factory=list)

    @property
    def n_applicable(self) -> int:
        return sum(1 for r in self.rule_results if r.applies)

    @property
    def n_violations(self) -> int:
        return sum(1 for r in self.rule_results if r.has_violation)

    @property
    def total_severity(self) -> float:
        return sum(r.severity for r in self.rule_results)

    @property
    def max_severity_rule(self) -> Optional[PipelineRuleResult]:
        if not self.rule_results:
            return None
        return max(self.rule_results, key=lambda r: r.severity)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "n_frames": self.n_frames,
            "n_agents": self.n_agents,
            "n_applicable": self.n_applicable,
            "n_violations": self.n_violations,
            "total_severity": self.total_severity,
            "rules": [r.to_dict() for r in self.rule_results],
        }


@dataclass
class RuleDefinition:
    """Complete rule definition bundling detector and evaluator."""

    rule_id: str
    level: int
    name: str
    description: str
    detector: ApplicabilityDetector
    evaluator: ViolationEvaluator


class RuleExecutor:
    """
    Executes a set of rules on scenarios.

    Usage:
        executor = RuleExecutor()
        executor.register_all_rules()
        result = executor.evaluate(scenario_context)
        print(f"Violations: {result.n_violations}")
    """

    def __init__(self):
        self._rules: List[RuleDefinition] = []

    def register_rule(
        self,
        detector: ApplicabilityDetector,
        evaluator: ViolationEvaluator,
        description: str = "",
    ) -> None:
        """Register a rule with its detector and evaluator."""
        rule = RuleDefinition(
            rule_id=detector.rule_id,
            level=detector.level,
            name=detector.name,
            description=description,
            detector=detector,
            evaluator=evaluator,
        )
        self._rules.append(rule)
        log.debug(f"Registered rule: {rule.rule_id} - {rule.name}")

    def register_all_rules(self) -> None:
        """Register all available rules from the canonical registry."""
        from ..rules.registry import all_rules

        for entry in all_rules():
            self.register_rule(
                entry.detector,
                entry.evaluator,
                f"Registered from rules.registry ({entry.rule_id})",
            )

        log.info(f"Registered {len(self._rules)} rules from registry")

    @property
    def rules(self) -> List[RuleDefinition]:
        return self._rules

    def evaluate(self, ctx: ScenarioContext) -> ScenarioResult:
        """
        Evaluate all registered rules on a scenario.

        Args:
            ctx: ScenarioContext with ego, agents, and map data

        Returns:
            ScenarioResult with all rule evaluations
        """
        result = ScenarioResult(
            scenario_id=ctx.scenario_id,
            n_frames=ctx.n_frames,
            n_agents=len(ctx.agents),
        )

        for rule in self._rules:
            try:
                # Check applicability
                applicability = rule.detector.detect(ctx)

                rule_result = PipelineRuleResult(
                    rule_id=rule.rule_id,
                    level=rule.level,
                    name=rule.name,
                    applies=applicability.applies,
                    applicability=applicability,
                )

                # Evaluate violation if applicable
                if applicability.applies:
                    violation = rule.evaluator.evaluate(ctx, applicability)
                    rule_result.violation = violation

                result.rule_results.append(rule_result)

            except Exception as e:
                log.error(f"Error evaluating rule {rule.rule_id}: {e}")
                result.rule_results.append(
                    PipelineRuleResult(
                        rule_id=rule.rule_id,
                        level=rule.level,
                        name=rule.name,
                        applies=False,
                        error=str(e),
                    )
                )

        return result

    def evaluate_batch(self, contexts: List[ScenarioContext]) -> List[ScenarioResult]:
        """
        Evaluate all rules on multiple scenarios.

        Args:
            contexts: List of ScenarioContext objects

        Returns:
            List of ScenarioResult objects
        """
        results = []
        for ctx in contexts:
            result = self.evaluate(ctx)
            results.append(result)
        return results


def extract_window(
    ctx: ScenarioContext, start_idx: int, end_idx: int
) -> ScenarioContext:
    """
    Extract a time window from a ScenarioContext.

    Creates a new ScenarioContext containing only the frames
    from start_idx to end_idx (exclusive).

    Args:
        ctx: Full scenario context
        start_idx: Start frame index (0-based)
        end_idx: End frame index (exclusive)

    Returns:
        New ScenarioContext with sliced ego and agent trajectories
    """
    import numpy as np

    from ..core.context import Agent, EgoState

    # Slice ego trajectory
    ego = ctx.ego
    sliced_ego = EgoState(
        x=np.asarray(ego.x)[start_idx:end_idx],
        y=np.asarray(ego.y)[start_idx:end_idx],
        yaw=np.asarray(ego.yaw)[start_idx:end_idx],
        speed=np.asarray(ego.speed)[start_idx:end_idx],
        length=ego.length,
        width=ego.width,
        valid=(
            np.asarray(ego.valid)[start_idx:end_idx] if ego.valid is not None else None
        ),
    )

    # Slice agent trajectories
    sliced_agents = []
    for a in ctx.agents:
        sliced_agents.append(
            Agent(
                id=a.id,
                type=a.type,
                x=np.asarray(a.x)[start_idx:end_idx],
                y=np.asarray(a.y)[start_idx:end_idx],
                yaw=np.asarray(a.yaw)[start_idx:end_idx],
                speed=np.asarray(a.speed)[start_idx:end_idx],
                length=a.length,
                width=a.width,
                valid=(
                    np.asarray(a.valid)[start_idx:end_idx]
                    if a.valid is not None
                    else None
                ),
            )
        )

    # Calculate window start time from index and dt
    dt = ctx.dt or 0.1
    window_start_ts = start_idx * dt

    # Slice signals if present
    sliced_signals = ctx.signals
    if ctx.signals is not None:
        from ..core.context import MapSignals

        sliced_signals = MapSignals(
            signal_state=(
                np.asarray(ctx.signals.signal_state)[start_idx:end_idx]
                if ctx.signals.signal_state is not None
                else None
            ),
            ego_lane_id=ctx.signals.ego_lane_id,
        )

    # Create windowed context
    return ScenarioContext(
        scenario_id=ctx.scenario_id,
        dt=ctx.dt,
        ego=sliced_ego,
        agents=sliced_agents,
        map_context=ctx.map_context,  # Map is shared
        signals=sliced_signals,
        window_start_ts=window_start_ts,
        window_size=end_idx - start_idx,
    )


@dataclass
class WindowResult:
    """Result from evaluating a single time window."""

    scenario_id: str
    window_idx: int
    window_start_ts: Optional[float]
    window_start_idx: int
    window_end_idx: int
    window_size: int
    n_applicable_rules: int
    n_violations: int
    total_severity: float
    rule_results: List[PipelineRuleResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "window_idx": self.window_idx,
            "window_start_ts": self.window_start_ts,
            "window_start_idx": self.window_start_idx,
            "window_end_idx": self.window_end_idx,
            "window_size": self.window_size,
            "n_applicable_rules": self.n_applicable_rules,
            "n_violations": self.n_violations,
            "total_severity": self.total_severity,
            "rules": [r.to_dict() for r in self.rule_results],
        }

    def to_flat_records(self, include_version: bool = True) -> List[Dict[str, Any]]:
        """Convert to flat per-rule records for output sinks."""
        from ..utils.version import get_engine_version, get_rule_version

        records = []
        for r in self.rule_results:
            record = {
                "scenario_id": self.scenario_id,
                "window_idx": self.window_idx,
                "window_start_ts": self.window_start_ts,
                "window_size": self.window_size,
                "rule_id": r.rule_id,
                "rule_level": r.level,
                "rule_name": r.name,
                "applies": r.applies,
                "has_violation": r.has_violation,
                "severity": r.severity,
                "confidence": (r.applicability.confidence if r.applicability else None),
                "reasons": (
                    ";".join(r.applicability.reasons) if r.applicability else None
                ),
                "explanation": (r.violation.explanation if r.violation else None),
                "error": r.error,
            }
            if include_version:
                record["engine_version"] = get_engine_version()
                record["rule_version"] = get_rule_version(r.rule_id)
            records.append(record)
        return records


class WindowedExecutor:
    """
    Executes rule evaluation across sliding time windows.

    This executor:
    1. Generates time windows from the scenario
    2. For each window, determines which rules apply
    3. Evaluates applicable rules
    4. Writes results to a sink

    Usage:
        executor = WindowedExecutor(window_size_s=8.0, stride_s=2.0)
        executor.register_all_rules()
        with JsonlSink("output.jsonl") as sink:
            executor.run(context, sink)
    """

    def __init__(
        self,
        window_size_s: float = 8.0,
        stride_s: float = 2.0,
        dt: float = 0.1,
    ):
        """
        Initialize windowed executor.

        Args:
            window_size_s: Window size in seconds (default 8s)
            stride_s: Stride between windows in seconds (default 2s)
            dt: Time step between frames (default 0.1s = 100ms)
        """
        self.window_size_s = window_size_s
        self.stride_s = stride_s
        self.dt = dt
        self._executor = RuleExecutor()

    def register_all_rules(self) -> None:
        """Register all available rules."""
        self._executor.register_all_rules()

    @property
    def rules(self) -> List[RuleDefinition]:
        return self._executor.rules

    def make_windows(self, ctx: ScenarioContext) -> List[tuple]:
        """
        Generate time windows for the scenario.

        Returns:
            List of (window_idx, start_idx, end_idx) tuples
        """
        from .window_scheduler import make_windows_timed

        n_frames = ctx.n_frames
        dt = ctx.dt or self.dt

        specs = make_windows_timed(
            T=n_frames,
            dt=dt,
            window_size_s=self.window_size_s,
            stride_s=self.stride_s,
        )

        return [(i, s.start_idx, s.end_idx) for i, s in enumerate(specs)]

    def evaluate_window(
        self,
        ctx: ScenarioContext,
        window_idx: int,
        start_idx: int,
        end_idx: int,
    ) -> WindowResult:
        """
        Evaluate all rules on a single time window.

        Args:
            ctx: Full scenario context
            window_idx: Index of this window
            start_idx: Start frame index
            end_idx: End frame index

        Returns:
            WindowResult with all rule evaluations
        """
        # Extract window
        windowed_ctx = extract_window(ctx, start_idx, end_idx)

        # Evaluate rules
        scenario_result = self._executor.evaluate(windowed_ctx)

        # Convert to WindowResult
        return WindowResult(
            scenario_id=ctx.scenario_id,
            window_idx=window_idx,
            window_start_ts=windowed_ctx.window_start_ts,
            window_start_idx=start_idx,
            window_end_idx=end_idx,
            window_size=end_idx - start_idx,
            n_applicable_rules=scenario_result.n_applicable,
            n_violations=scenario_result.n_violations,
            total_severity=scenario_result.total_severity,
            rule_results=scenario_result.rule_results,
        )

    def run(
        self,
        ctx: ScenarioContext,
        sink=None,
        flat_output: bool = True,
    ) -> List[WindowResult]:
        """
        Run windowed rule evaluation on a scenario.

        Args:
            ctx: Scenario context
            sink: Optional output sink (JsonlSink, CsvSink, etc.)
            flat_output: If True, write flat per-rule records

        Returns:
            List of WindowResult objects
        """
        windows = self.make_windows(ctx)
        results = []

        log.info(
            f"Running windowed evaluation on {ctx.scenario_id}: "
            f"{len(windows)} windows, {len(self.rules)} rules"
        )

        for window_idx, start_idx, end_idx in windows:
            try:
                result = self.evaluate_window(ctx, window_idx, start_idx, end_idx)
                results.append(result)

                # Write to sink
                if sink:
                    if flat_output:
                        for record in result.to_flat_records():
                            sink.write(record)
                    else:
                        sink.write(result.to_dict())

                log.debug(
                    f"Window {window_idx}: {result.n_applicable_rules} applicable, "
                    f"{result.n_violations} violations"
                )

            except Exception as e:
                log.error(f"Error evaluating window {window_idx}: {e}")

        return results

    def run_batch(
        self,
        contexts: List[ScenarioContext],
        sink=None,
        flat_output: bool = True,
    ) -> Dict[str, List[WindowResult]]:
        """
        Run windowed evaluation on multiple scenarios.

        Args:
            contexts: List of scenario contexts
            sink: Optional output sink
            flat_output: If True, write flat per-rule records

        Returns:
            Dict mapping scenario_id to list of WindowResult
        """
        all_results = {}

        for ctx in contexts:
            results = self.run(ctx, sink=sink, flat_output=flat_output)
            all_results[ctx.scenario_id] = results

        return all_results
