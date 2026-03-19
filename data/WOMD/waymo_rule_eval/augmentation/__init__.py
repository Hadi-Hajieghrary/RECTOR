"""
Waymo Scenario Rule Augmentation Pipeline

This module augments Waymo Motion Dataset scenarios with rule evaluation data.
For each trajectory window, it computes:
1. Boolean vector of rule applicability
2. Violation metrics for applicable rules
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterator, List, Optional

import numpy as np

try:
    from ..core.context import ScenarioContext
    from ..pipeline.rule_executor import RuleDefinition, RuleExecutor
except ImportError:
    from core.context import ScenarioContext
    from pipeline.rule_executor import RuleExecutor

    RuleDefinition = None  # Not needed for augmentation

log = logging.getLogger(__name__)


@dataclass
class RuleResult:
    """Result of a single rule evaluation for a window."""

    rule_id: str
    rule_name: str
    applicable: bool
    violated: bool = False
    severity: float = 0.0
    confidence: float = 0.0
    features: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "rule_name": self.rule_name,
            "applicable": self.applicable,
            "violated": self.violated,
            "severity": self.severity,
            "confidence": self.confidence,
        }


@dataclass
class WindowAugmentation:
    """Augmentation data for a single trajectory window."""

    window_idx: int
    start_frame: int
    end_frame: int
    start_time_s: float
    end_time_s: float
    rule_results: List[RuleResult] = field(default_factory=list)

    @property
    def applicability_vector(self) -> Dict[str, bool]:
        """Boolean dict of which rules are applicable."""
        return {r.rule_id: r.applicable for r in self.rule_results}

    @property
    def violation_vector(self) -> Dict[str, bool]:
        """Boolean dict of which rules are violated."""
        return {r.rule_id: r.violated for r in self.rule_results}

    @property
    def severity_vector(self) -> Dict[str, float]:
        """Float dict of violation severities."""
        return {r.rule_id: r.severity for r in self.rule_results}

    @property
    def n_applicable(self) -> int:
        return sum(1 for r in self.rule_results if r.applicable)

    @property
    def n_violated(self) -> int:
        return sum(1 for r in self.rule_results if r.violated)

    @property
    def total_severity(self) -> float:
        return sum(r.severity for r in self.rule_results)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "window_idx": self.window_idx,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "start_time_s": self.start_time_s,
            "end_time_s": self.end_time_s,
            "rule_applicability": self.applicability_vector,
            "rule_violations": {
                r.rule_id: {
                    "violated": r.violated,
                    "severity": r.severity,
                    "confidence": r.confidence,
                }
                for r in self.rule_results
                if r.applicable
            },
            "summary": {
                "n_applicable": self.n_applicable,
                "n_violated": self.n_violated,
                "total_severity": self.total_severity,
                "violated_rules": [r.rule_id for r in self.rule_results if r.violated],
            },
        }


@dataclass
class ScenarioAugmentation:
    """Complete augmentation for a scenario."""

    scenario_id: str
    num_frames: int
    dt: float
    windows: List[WindowAugmentation] = field(default_factory=list)

    @property
    def num_windows(self) -> int:
        return len(self.windows)

    @property
    def total_applicable(self) -> int:
        return sum(w.n_applicable for w in self.windows)

    @property
    def total_violations(self) -> int:
        return sum(w.n_violated for w in self.windows)

    @property
    def mean_severity(self) -> float:
        if not self.windows:
            return 0.0
        return sum(w.total_severity for w in self.windows) / len(self.windows)

    @property
    def rules_ever_violated(self) -> List[str]:
        violated = set()
        for w in self.windows:
            for r in w.rule_results:
                if r.violated:
                    violated.add(r.rule_id)
        return sorted(violated)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "num_frames": self.num_frames,
            "dt": self.dt,
            "num_windows": self.num_windows,
            "windows": [w.to_dict() for w in self.windows],
            "scenario_summary": {
                "total_windows": self.num_windows,
                "total_applicable": self.total_applicable,
                "total_violations": self.total_violations,
                "mean_severity": self.mean_severity,
                "rules_ever_violated": self.rules_ever_violated,
            },
        }


class ScenarioAugmenter:
    """
    Augments Waymo scenarios with rule evaluation data.

    For each scenario, creates sliding windows over the trajectory
    and evaluates all rules on each window.
    """

    def __init__(
        self,
        window_size_s: float = 8.0,
        stride_s: float = 1.0,
        min_window_frames: int = 10,
    ):
        """
        Initialize the augmenter.

        Args:
            window_size_s: Size of each evaluation window in seconds
            stride_s: Stride between windows in seconds
            min_window_frames: Minimum frames required for a valid window
        """
        self.window_size_s = window_size_s
        self.stride_s = stride_s
        self.min_window_frames = min_window_frames

        # Initialize rule executor
        self.executor = RuleExecutor()
        self.executor.register_all_rules()
        self.rules = self.executor.rules

        log.info(
            f"Initialized augmenter with {len(self.rules)} rules, "
            f"window={window_size_s}s, stride={stride_s}s"
        )

    @property
    def rule_ids(self) -> List[str]:
        """Get sorted list of all rule IDs."""
        return sorted(set(r.rule_id for r in self.rules))

    def _create_window_context(
        self, ctx: ScenarioContext, start_frame: int, end_frame: int
    ) -> ScenarioContext:
        """Create a context for a specific window of frames."""
        from ..core.context import Agent, EgoState, MapContext, MapSignals

        # Slice ego state
        window_ego = EgoState(
            x=ctx.ego.x[start_frame:end_frame].copy(),
            y=ctx.ego.y[start_frame:end_frame].copy(),
            yaw=ctx.ego.yaw[start_frame:end_frame].copy(),
            speed=ctx.ego.speed[start_frame:end_frame].copy(),
            length=ctx.ego.length,
            width=ctx.ego.width,
        )

        # Slice agents
        window_agents = []
        for agent in ctx.agents:
            window_agent = Agent(
                id=agent.id,
                type=agent.type,
                x=agent.x[start_frame:end_frame].copy(),
                y=agent.y[start_frame:end_frame].copy(),
                yaw=agent.yaw[start_frame:end_frame].copy(),
                speed=agent.speed[start_frame:end_frame].copy(),
                valid=agent.valid[start_frame:end_frame].copy(),
                length=agent.length,
                width=agent.width,
            )
            window_agents.append(window_agent)

        # Slice signals if present
        window_signals = None
        if ctx.signals is not None:
            window_signals = MapSignals(
                signal_state=(
                    ctx.signals.signal_state[start_frame:end_frame].copy()
                    if ctx.signals.signal_state is not None
                    else None
                ),
                ego_lane_id=ctx.signals.ego_lane_id,
            )

        return ScenarioContext(
            scenario_id=f"{ctx.scenario_id}_w{start_frame}_{end_frame}",
            ego=window_ego,
            agents=window_agents,
            map_context=ctx.map_context,  # Map doesn't change
            signals=window_signals,
            dt=ctx.dt,
        )

    def _evaluate_window(
        self,
        ctx: ScenarioContext,
        window_idx: int,
        start_frame: int,
        end_frame: int,
    ) -> WindowAugmentation:
        """Evaluate all rules on a single window."""
        start_time = start_frame * ctx.dt
        end_time = end_frame * ctx.dt

        window_aug = WindowAugmentation(
            window_idx=window_idx,
            start_frame=start_frame,
            end_frame=end_frame,
            start_time_s=start_time,
            end_time_s=end_time,
        )

        # Create window context
        try:
            window_ctx = self._create_window_context(ctx, start_frame, end_frame)
        except Exception as e:
            log.warning(f"Failed to create window context: {e}")
            # Return empty results for all rules
            for rule in self.rules:
                window_aug.rule_results.append(
                    RuleResult(
                        rule_id=rule.rule_id,
                        rule_name=rule.name,
                        applicable=False,
                    )
                )
            return window_aug

        # Evaluate each rule
        for rule in self.rules:
            try:
                # Check applicability
                app_result = rule.detector.detect(window_ctx)

                result = RuleResult(
                    rule_id=rule.rule_id,
                    rule_name=rule.name,
                    applicable=app_result.applies,
                    confidence=app_result.confidence,
                )

                # If applicable, evaluate violation
                if app_result.applies:
                    try:
                        viol_result = rule.evaluator.evaluate(window_ctx, app_result)
                        result.violated = viol_result.severity > 0
                        result.severity = viol_result.severity
                        # Store key features
                        if (
                            hasattr(viol_result, "measurements")
                            and viol_result.measurements
                        ):
                            result.features = {
                                k: v
                                for k, v in viol_result.measurements.items()
                                if isinstance(v, (int, float, bool, str))
                            }
                    except Exception as e:
                        log.debug(f"Rule {rule.rule_id} evaluation error: {e}")
                        result.violated = False
                        result.severity = 0.0

                window_aug.rule_results.append(result)

            except Exception as e:
                log.debug(f"Rule {rule.rule_id} detection error: {e}")
                window_aug.rule_results.append(
                    RuleResult(
                        rule_id=rule.rule_id,
                        rule_name=rule.name,
                        applicable=False,
                    )
                )

        return window_aug

    def augment_scenario(self, ctx: ScenarioContext) -> ScenarioAugmentation:
        """
        Augment a single scenario with rule evaluation data.

        Args:
            ctx: The scenario context to augment

        Returns:
            ScenarioAugmentation with rule data for each window
        """
        n_frames = len(ctx.ego.x)
        dt = ctx.dt

        # Calculate window parameters in frames
        window_frames = int(self.window_size_s / dt)
        stride_frames = int(self.stride_s / dt)

        augmentation = ScenarioAugmentation(
            scenario_id=ctx.scenario_id,
            num_frames=n_frames,
            dt=dt,
        )

        # Generate windows
        window_idx = 0
        start_frame = 0

        while start_frame + self.min_window_frames <= n_frames:
            end_frame = min(start_frame + window_frames, n_frames)

            # Ensure minimum window size
            if end_frame - start_frame >= self.min_window_frames:
                window_aug = self._evaluate_window(
                    ctx, window_idx, start_frame, end_frame
                )
                augmentation.windows.append(window_aug)
                window_idx += 1

            start_frame += stride_frames

            # Break if we've reached the end
            if start_frame >= n_frames:
                break

        return augmentation

    def augment_scenarios(
        self,
        contexts: Iterator[ScenarioContext],
        max_scenarios: Optional[int] = None,
    ) -> Iterator[ScenarioAugmentation]:
        """
        Augment multiple scenarios.

        Args:
            contexts: Iterator of scenario contexts
            max_scenarios: Maximum number of scenarios to process

        Yields:
            ScenarioAugmentation for each scenario
        """
        count = 0
        for ctx in contexts:
            if max_scenarios and count >= max_scenarios:
                break

            try:
                yield self.augment_scenario(ctx)
                count += 1

                if count % 100 == 0:
                    log.info(f"Processed {count} scenarios")

            except Exception as e:
                log.warning(f"Failed to augment scenario {ctx.scenario_id}: {e}")
                continue

        log.info(f"Completed augmentation of {count} scenarios")
