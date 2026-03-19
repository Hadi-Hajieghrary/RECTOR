"""
Main closed-loop simulation loop: WaymaxRECTORLoop.

Orchestrates:
  1. Scenario loading
  2. Environment creation
  3. Observation extraction
  4. Trajectory generation (RECTOR or mock)
  5. Rule evaluation & trajectory selection
  6. Action conversion & stepping
  7. Metric collection

Replanning occurs every ``dt_replan`` seconds (default 2.0s = 20 steps
at 10 Hz).  Between replans the selected trajectory is executed
open-loop (one step at a time).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config import ExperimentConfig, WaymaxBridgeConfig, AgentConfig
from ..selectors.base import BaseSelector, DecisionTrace
from ..selectors.confidence import ConfidenceSelector
from ..selectors.weighted_sum import WeightedSumSelector
from ..selectors.rector_lex import RECTORLexSelector

from .scenario_loader import load_scenarios, ScenarioLoaderConfig
from .env_factory import make_env
from .observation_extractor import extract_observation, ObservationExtractorConfig
from .action_converter import trajectory_to_action
from .metric_collector import MetricCollector, MetricCollectorConfig


class MockLogReplayGenerator:
    """
    Mock generator that returns the logged future trajectory as the
    single candidate.  Useful for validating the pipeline end-to-end
    before a real RECTOR checkpoint is available.
    """

    def __init__(self, num_modes: int = 6, horizon: int = 50):
        self.num_modes = num_modes
        self.horizon = horizon

    def generate(
        self,
        obs: Dict[str, np.ndarray],
        state,
    ) -> Dict[str, np.ndarray]:
        """
        Return mock candidates by replaying logged trajectory.

        Returns
        -------
        dict with keys:
            trajectories : [K, T, 4]  (x, y, vx, vy) in ego-centric normalized
            confidence    : [K]
            rule_costs    : [K, 28]
            applicability : [28]
        """
        import jax.numpy as jnp
        from .observation_extractor import TRAJECTORY_SCALE

        sdc_idx = int(jnp.argmax(state.object_metadata.is_sdc))
        t = int(state.timestep)
        T = self.horizon

        # Extract future from log_trajectory in ego frame
        ego_pos = obs["ego_pos"]
        ego_yaw = obs["ego_yaw"]
        cos_r = np.cos(-ego_yaw)
        sin_r = np.sin(-ego_yaw)

        log_x = np.asarray(state.log_trajectory.x[sdc_idx])
        log_y = np.asarray(state.log_trajectory.y[sdc_idx])
        log_vx = np.asarray(state.log_trajectory.vel_x[sdc_idx])
        log_vy = np.asarray(state.log_trajectory.vel_y[sdc_idx])
        total_t = len(log_x)

        traj = np.zeros((T, 4), dtype=np.float32)
        for i in range(T):
            ti = min(t + 1 + i, total_t - 1)
            dx = log_x[ti] - ego_pos[0]
            dy = log_y[ti] - ego_pos[1]
            ex = dx * cos_r - dy * sin_r
            ey = dx * sin_r + dy * cos_r
            traj[i, 0] = ex / TRAJECTORY_SCALE
            traj[i, 1] = ey / TRAJECTORY_SCALE
            # Rotate velocity
            traj[i, 2] = log_vx[ti] * cos_r - log_vy[ti] * sin_r
            traj[i, 3] = log_vx[ti] * sin_r + log_vy[ti] * cos_r

        # Tile to K candidates with slight noise for multi-modality
        K = self.num_modes
        trajectories = np.tile(traj[np.newaxis], (K, 1, 1))
        rng = np.random.RandomState(int(t))
        trajectories[1:] += rng.randn(K - 1, T, 4).astype(np.float32) * 0.001

        confidence = np.ones(K, dtype=np.float32) / K
        confidence[0] = 0.5  # Boost the "clean" candidate
        confidence /= confidence.sum()

        # Mock rule costs (zeros = no violations)
        rule_costs = np.zeros((K, 28), dtype=np.float32)
        applicability = np.ones(28, dtype=np.float32)

        return {
            "trajectories": trajectories,
            "confidence": confidence,
            "rule_costs": rule_costs,
            "applicability": applicability,
        }


@dataclass
class ScenarioResult:
    """Result of running one scenario with one selector."""

    scenario_id: str
    selector_method: str
    metrics: Dict[str, float] = field(default_factory=dict)
    traces: List[DecisionTrace] = field(default_factory=list)
    wall_time_s: float = 0.0


class WaymaxRECTORLoop:
    """
    Closed-loop simulation loop for one scenario + one selector.

    Parameters
    ----------
    generator : object
        Must have ``generate(obs, state) -> dict`` method.
        Use ``MockLogReplayGenerator`` for testing.
    selector : BaseSelector
        One of the three selectors (confidence, weighted_sum, rector_lex).
    cfg : WaymaxBridgeConfig
    agent_cfg : AgentConfig
    metric_cfg : MetricCollectorConfig, optional
    """

    def __init__(
        self,
        generator,
        selector: BaseSelector,
        bridge_cfg: Optional[WaymaxBridgeConfig] = None,
        agent_cfg: Optional[AgentConfig] = None,
        metric_cfg: Optional[MetricCollectorConfig] = None,
    ):
        self.generator = generator
        self.selector = selector
        self.bridge_cfg = bridge_cfg or WaymaxBridgeConfig()
        self.agent_cfg = agent_cfg or AgentConfig()
        self.metric_collector = MetricCollector(metric_cfg)
        self.obs_cfg = ObservationExtractorConfig()

    def run_scenario(
        self,
        sim_state,
        scenario_id: str = "unknown",
        max_sim_steps: int = 80,
    ) -> ScenarioResult:
        """
        Run a full closed-loop rollout on one scenario.

        Parameters
        ----------
        sim_state : SimulatorState
            From ``scenario_loader.load_scenarios()``.
        scenario_id : str
            For logging/tracing.
        max_sim_steps : int
            Maximum simulation steps (default 80 = 8s at 10Hz).

        Returns
        -------
        ScenarioResult
        """
        t_start = time.time()

        # --- Create env -------------------------------------------------------
        env, state = make_env(
            sim_state,
            dynamics_model=self.bridge_cfg.dynamics_model,
            agent_model=self.agent_cfg.agent_model,
            idm_desired_vel=self.agent_cfg.idm_desired_vel,
            idm_min_spacing=self.agent_cfg.idm_min_spacing,
            idm_safe_time_headway=self.agent_cfg.idm_safe_time_headway,
            idm_max_accel=self.agent_cfg.idm_max_accel,
            idm_max_decel=self.agent_cfg.idm_max_decel,
        )

        self.metric_collector.reset()
        traces: List[DecisionTrace] = []

        steps_per_replan = self.bridge_cfg.steps_per_replan
        total_steps = min(max_sim_steps, 91 - int(state.timestep) - 1)

        selected_traj = None
        obs = None
        step_in_plan = 0

        for step_i in range(total_steps):
            # --- Replan if needed -------------------------------------------
            if step_i % steps_per_replan == 0 or selected_traj is None:
                obs = extract_observation(state, self.obs_cfg)

                gen_out = self.generator.generate(obs, state)
                trajectories = gen_out["trajectories"]  # [K, T, 4]
                confidence = gen_out["confidence"]  # [K]
                rule_costs = gen_out["rule_costs"]  # [K, 28]
                applicability = gen_out["applicability"]  # [28]

                # Compute per-tier scores for selector
                tier_scores = self._compute_tier_scores(rule_costs, applicability)

                sel_idx, trace = self.selector.select(
                    candidates=trajectories,
                    probs=confidence,
                    tier_scores=tier_scores,
                    rule_scores=rule_costs,
                    rule_applicability=applicability,
                )
                traces.append(trace)
                selected_traj = trajectories[sel_idx]  # [T, 4]
                step_in_plan = 0

            # --- Step env ---------------------------------------------------
            action = trajectory_to_action(
                trajectory=selected_traj,
                ego_pos=obs["ego_pos"],
                ego_yaw=obs["ego_yaw"],
                state=state,
                steps=step_in_plan + 1,
            )
            state = env.step(state, action)
            step_in_plan += 1

            # --- Collect metrics -------------------------------------------
            self.metric_collector.step(state)

            # --- Check termination -----------------------------------------
            if bool(state.is_done):
                break

        # --- Finalise -------------------------------------------------------
        summary = self.metric_collector.finalise()
        wall_time = time.time() - t_start

        return ScenarioResult(
            scenario_id=scenario_id,
            selector_method=type(self.selector).__name__,
            metrics=summary,
            traces=traces,
            wall_time_s=wall_time,
        )

    @staticmethod
    def _compute_tier_scores(
        rule_costs: np.ndarray,  # [K, 28]
        applicability: np.ndarray,  # [28]
    ) -> np.ndarray:
        """
        Aggregate per-rule costs into 4 tier scores for selector.

        Uses the tier masks from ``rule_constants``.

        Returns
        -------
        np.ndarray [K, 4]
        """
        try:
            from ...data.WOMD.waymo_rule_eval.rules.rule_constants import (
                TIER_INDICES,
            )

            tier_indices = TIER_INDICES
        except (ImportError, AttributeError):
            # Fallback: split into 4 equal-ish tiers
            tier_indices = {
                0: list(range(0, 7)),
                1: list(range(7, 14)),
                2: list(range(14, 21)),
                3: list(range(21, 28)),
            }

        K = rule_costs.shape[0]
        tier_scores = np.zeros((K, 4), dtype=np.float32)
        for tier, indices in tier_indices.items():
            if tier >= 4:
                continue
            idx = [i for i in indices if i < 28]
            if idx:
                # Mask by applicability
                masked = rule_costs[:, idx] * applicability[idx]
                tier_scores[:, tier] = masked.sum(axis=1)
        return tier_scores


def run_batch(
    loader_cfg: ScenarioLoaderConfig,
    selectors: Sequence[BaseSelector],
    generator=None,
    bridge_cfg: Optional[WaymaxBridgeConfig] = None,
    agent_cfg: Optional[AgentConfig] = None,
    metric_cfg: Optional[MetricCollectorConfig] = None,
    max_sim_steps: int = 80,
    verbose: bool = True,
) -> List[ScenarioResult]:
    """
    Run closed-loop simulation over multiple scenarios and selectors.

    Returns a flat list of ScenarioResult objects.
    """
    if generator is None:
        generator = MockLogReplayGenerator()

    results: List[ScenarioResult] = []
    scenario_gen = load_scenarios(loader_cfg)

    for s_idx, (sid, sim_state) in enumerate(scenario_gen):
        for sel in selectors:
            loop = WaymaxRECTORLoop(
                generator=generator,
                selector=sel,
                bridge_cfg=bridge_cfg,
                agent_cfg=agent_cfg,
                metric_cfg=metric_cfg,
            )
            result = loop.run_scenario(
                sim_state, scenario_id=sid, max_sim_steps=max_sim_steps
            )
            results.append(result)
            if verbose:
                ov = result.metrics.get("overlap/rate", float("nan"))
                ld = result.metrics.get("log_divergence/mean", float("nan"))
                print(
                    f"  [{s_idx+1}] {sid} | {result.selector_method:20s} | "
                    f"overlap={ov:.3f} logdiv={ld:.2f} "
                    f"({result.wall_time_s:.1f}s)"
                )

    return results
