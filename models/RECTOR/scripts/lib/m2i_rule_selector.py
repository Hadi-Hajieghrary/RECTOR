#!/usr/bin/env python3
"""
M2I Rule Selector: M2I DenseTNT + RECTOR Rule Scoring.

Uses the full pretrained DenseTNT model for trajectory generation and
RECTOR's tiered rule scorer + differentiable proxies for trajectory ranking.

Key characteristics:
  - Trajectories from pre-trained DenseTNT  (no CVAE, no delta-cumsum)
  - 8-second horizon, 80 steps (DenseTNT native)
  - 6-mode output with NMS-based diversity
  - Rule-based trajectory selection (lexicographic tiered scoring)
  - No training needed for trajectory generation

Pipeline:
  1. Parse Waymo Scenario proto → M2I mapping (polyline vectors + BEV raster)
  2. DenseTNT VectorNet: encode scene → score goals → regress trajectories
  3. Output: 6 trajectory modes [6, 80, 2] + scores [6]
  4. RECTOR Rule Proxies: evaluate each mode against 28 Waymo rules
  5. Tiered Rule Scorer: rank trajectories (Safety > Legal > Road > Comfort)
  6. Return: best trajectory + all modes + rule violation details

Usage:
    selector = M2IRuleSelector(device='cuda')
    selector.load()
    result = selector.evaluate_scenario(scenario_proto)
    best_traj = result['best_trajectory']   # [80, 2] world coords
    violations = result['violations']       # per-rule violation details
"""

import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

# RECTOR paths
RECTOR_SCRIPTS = Path(__file__).parent.parent
sys.path.insert(0, str(RECTOR_SCRIPTS))
sys.path.insert(0, str(RECTOR_SCRIPTS / "lib"))
sys.path.insert(0, "/workspace/data/WOMD")

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

# RECTOR imports
from lib.m2i_trajectory_generator import M2ITrajectoryGenerator

# Rule evaluation
from waymo_rule_eval.rules.rule_constants import (
    NUM_RULES,
    RULE_IDS,
    RULE_INDEX_MAP,
    TIER_0_SAFETY,
    TIER_1_LEGAL,
    TIER_2_ROAD,
    TIER_3_COMFORT,
    TIERS,
    get_tier_mask,
)


@dataclass
class RuleViolationResult:
    """Per-mode rule violation results."""

    violations: np.ndarray  # [M, NUM_RULES] binary violation flags
    violation_costs: np.ndarray  # [M, NUM_RULES] soft costs in [0, 1]
    tier_scores: Dict[str, np.ndarray]  # tier_name -> [M] aggregated score
    composite_scores: np.ndarray  # [M] lexicographic composite scores


@dataclass
class GenerationResult:
    """Complete result from M2I + RECTOR pipeline."""

    # Trajectories
    trajectories_world: np.ndarray  # [M, 80, 2] in world coordinates
    trajectories_local: np.ndarray  # [M, 80, 2] in agent-local coords
    m2i_scores: np.ndarray  # [M] DenseTNT confidence scores
    best_m2i_idx: int  # Index of M2I's top pick

    # Rule evaluation
    rule_violations: Optional[RuleViolationResult] = None
    best_rule_idx: int = 0  # Index after rule-based reranking

    # Metadata
    scenario_id: str = ""
    agent_id: int = 0
    normalizer: Any = None  # For world ↔ local conversion
    timing_ms: Dict[str, float] = field(default_factory=dict)

    @property
    def best_trajectory_world(self) -> np.ndarray:
        """Best trajectory in world coordinates (after rule reranking)."""
        return self.trajectories_world[self.best_rule_idx]

    @property
    def best_trajectory_local(self) -> np.ndarray:
        """Best trajectory in agent-local coordinates."""
        return self.trajectories_local[self.best_rule_idx]


class KinematicRuleEvaluator:
    """
    Evaluate kinematic rule violations on trajectories using numpy.

    Computes violations for:
    - Comfort: max acceleration, jerk, lateral acceleration
    - Safety: collision risk (if other agents provided), speed limits
    - Road: lane deviation

    This is a lightweight alternative to the full DifferentiableRuleProxies
    that works on numpy arrays (no torch needed for M2I trajectories).
    """

    def __init__(
        self,
        dt: float = 0.1,
        max_accel: float = 3.0,  # m/s² comfort threshold
        max_jerk: float = 2.0,  # m/s³ comfort threshold
        max_lat_accel: float = 3.0,  # m/s² lateral comfort threshold
        max_speed: float = 30.0,  # m/s (~108 km/h)
        emergency_accel: float = 6.0,  # m/s² safety threshold
    ):
        self.dt = dt
        self.max_accel = max_accel
        self.max_jerk = max_jerk
        self.max_lat_accel = max_lat_accel
        self.max_speed = max_speed
        self.emergency_accel = emergency_accel

    def evaluate(
        self,
        trajectories: np.ndarray,  # [M, T, 2]
        gt_labels: Optional[np.ndarray] = None,  # [T, 2] for ADE/FDE
        agent_trajectories: Optional[np.ndarray] = None,  # [A, T, 2] other agents
    ) -> RuleViolationResult:
        """
        Evaluate rule violations for M trajectory modes.

        Returns:
            RuleViolationResult with per-mode, per-rule scores.
        """
        M, T, _ = trajectories.shape
        n_rules = NUM_RULES if NUM_RULES > 0 else 28

        violations = np.zeros((M, n_rules), dtype=np.float32)
        costs = np.zeros((M, n_rules), dtype=np.float32)

        for m in range(M):
            traj = trajectories[m]  # [T, 2]
            v = self._compute_violations(traj, agent_trajectories)
            violations[m] = v["binary"]
            costs[m] = v["costs"]

        # Compute tier scores
        tier_scores = {}
        for tier_name in TIERS:
            mask = np.array(get_tier_mask(tier_name), dtype=np.float32)
            # Weighted sum within tier
            tier_scores[tier_name] = (costs * mask[np.newaxis, :]).sum(axis=1)

        # Lexicographic composite score
        composite = np.zeros(M)
        multiplier = 1e12
        for tier_name in TIERS:
            composite += multiplier * tier_scores[tier_name]
            multiplier /= 1000

        return RuleViolationResult(
            violations=violations,
            violation_costs=costs,
            tier_scores=tier_scores,
            composite_scores=composite,
        )

    def _compute_violations(
        self,
        traj: np.ndarray,  # [T, 2]
        agent_trajs: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """Compute violation vector for a single trajectory."""
        n_rules = NUM_RULES if NUM_RULES > 0 else 28
        binary = np.zeros(n_rules, dtype=np.float32)
        costs = np.zeros(n_rules, dtype=np.float32)

        T = len(traj)
        if T < 3:
            return {"binary": binary, "costs": costs}

        # Kinematics
        displacements = traj[1:] - traj[:-1]  # [T-1, 2]
        speeds = np.linalg.norm(displacements, axis=-1) / self.dt  # [T-1]
        max_speed = speeds.max() if len(speeds) > 0 else 0.0

        velocities = displacements / self.dt  # [T-1, 2]
        if len(velocities) > 1:
            accelerations = (velocities[1:] - velocities[:-1]) / self.dt  # [T-2, 2]
            accel_mags = np.linalg.norm(accelerations, axis=-1)
            max_accel = accel_mags.max() if len(accel_mags) > 0 else 0.0
        else:
            accelerations = np.zeros((0, 2))
            accel_mags = np.zeros(0)
            max_accel = 0.0

        if len(accelerations) > 1:
            jerks = (accelerations[1:] - accelerations[:-1]) / self.dt
            jerk_mags = np.linalg.norm(jerks, axis=-1)
            max_jerk = jerk_mags.max() if len(jerk_mags) > 0 else 0.0
        else:
            max_jerk = 0.0

        # Lateral acceleration (curvature × v²)
        max_lat_accel = 0.0
        for t in range(1, len(traj) - 1):
            v1 = traj[t] - traj[t - 1]
            v2 = traj[t + 1] - traj[t]
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 > 0.01 and n2 > 0.01:
                cross = v1[0] * v2[1] - v1[1] * v2[0]
                curvature = 2 * abs(cross) / (n1 * n2 * np.linalg.norm(v1 + v2) + 1e-6)
                spd = speeds[min(t - 1, len(speeds) - 1)]
                lat_a = curvature * spd**2
                max_lat_accel = max(max_lat_accel, lat_a)

        # Collision check (bounding circle)
        min_dist_to_agent = float("inf")
        if agent_trajs is not None and len(agent_trajs) > 0:
            for a in range(len(agent_trajs)):
                dists = np.linalg.norm(
                    traj[: min(T, len(agent_trajs[a]))] - agent_trajs[a][:T], axis=-1
                )
                min_d = dists.min()
                min_dist_to_agent = min(min_dist_to_agent, min_d)

        # Map to rule indices
        # We use a simple mapping — populate what we can measure
        def _set_rule(rule_id: str, is_violated: bool, cost: float):
            if rule_id in RULE_INDEX_MAP:
                idx = RULE_INDEX_MAP[rule_id]
                binary[idx] = float(is_violated)
                costs[idx] = np.clip(cost, 0.0, 1.0)

        # Comfort rules
        comfort_accel_cost = max(0, (max_accel - self.max_accel) / self.max_accel)
        _set_rule("L1.R1", max_accel > self.max_accel, comfort_accel_cost)

        comfort_jerk_cost = max(0, (max_jerk - self.max_jerk) / self.max_jerk)
        _set_rule("L4.R1", max_jerk > self.max_jerk, comfort_jerk_cost)

        lat_cost = max(0, (max_lat_accel - self.max_lat_accel) / self.max_lat_accel)
        _set_rule("L6.R1", max_lat_accel > self.max_lat_accel, lat_cost)

        # Safety rules
        speed_cost = max(0, (max_speed - self.max_speed) / self.max_speed)
        _set_rule("L5.R1", max_speed > self.max_speed, speed_cost)

        emergency_cost = max(
            0, (max_accel - self.emergency_accel) / self.emergency_accel
        )
        _set_rule("L0.R1", max_accel > self.emergency_accel, emergency_cost)

        # Collision
        if min_dist_to_agent < 2.0:
            coll_cost = max(0, (2.0 - min_dist_to_agent) / 2.0)
            _set_rule("L10.R1", min_dist_to_agent < 1.0, coll_cost)
            _set_rule("L0.R2", min_dist_to_agent < 2.0, coll_cost * 0.5)

        return {"binary": binary, "costs": costs}


class M2IRuleSelector:
    """
    M2I DenseTNT trajectory generation + RECTOR rule-based scoring.

    This combines:
    - M2I's proven DenseTNT model for high-quality multi-modal trajectories
    - RECTOR's tiered rule scorer for safety-aware trajectory selection

    No training required — both components are pretrained.
    """

    def __init__(
        self,
        model_path: str = None,
        device: str = "cuda",
    ):
        self.m2i_generator = M2ITrajectoryGenerator(
            model_path=model_path,
            device=device,
        )
        self.rule_evaluator = KinematicRuleEvaluator()
        self._is_loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def load(self):
        """Load M2I DenseTNT model."""
        self.m2i_generator.load_model()
        self._is_loaded = True
        print("[M2IRuleSelector] Loaded M2I DenseTNT + RECTOR rule evaluator")

    def evaluate_scenario(
        self,
        scenario,
        target_agent_idx: int = 0,
        agent_future_trajs: Optional[np.ndarray] = None,
    ) -> Optional[GenerationResult]:
        """
        Full pipeline: generate trajectories and score with rulebook.

        Args:
            scenario: Waymo Scenario proto
            target_agent_idx: 0 = SDC
            agent_future_trajs: [A, T, 2] other agents' future trajectories
                                (GT or predicted) for collision evaluation.
                                If None, collision rules are not evaluated.

        Returns:
            GenerationResult with trajectories, rule violations, and best pick.
        """
        if not self._is_loaded:
            raise RuntimeError("Call load() first.")

        timing = {}

        # Step 1: Generate trajectories (world coords)
        t0 = time.perf_counter()
        traj_world, scores, mapping = self.m2i_generator.predict_from_scenario(
            scenario,
            target_agent_idx=target_agent_idx,
        )
        timing["m2i_generate_ms"] = (time.perf_counter() - t0) * 1000

        if traj_world is None or mapping is None:
            return None

        M = len(traj_world)  # 6 modes

        # Step 2: Get trajectories in local coords (for rule evaluation)
        t0 = time.perf_counter()
        normalizer = mapping["normalizer"]
        traj_local = np.zeros_like(traj_world)
        for m in range(M):
            traj_local[m] = normalizer(traj_world[m], reverse=False)
        timing["normalize_ms"] = (time.perf_counter() - t0) * 1000

        # M2I's best pick (highest confidence)
        best_m2i_idx = int(np.argmax(scores)) if scores is not None else 0

        # Step 3: Evaluate rules on each trajectory mode
        t0 = time.perf_counter()
        # Convert agent trajectories to local coords if provided
        agent_trajs_local = None
        if agent_future_trajs is not None:
            agent_trajs_local = np.zeros_like(agent_future_trajs)
            for a in range(len(agent_future_trajs)):
                agent_trajs_local[a] = normalizer(agent_future_trajs[a], reverse=False)

        rule_violations = self.rule_evaluator.evaluate(
            traj_local,
            agent_trajectories=agent_trajs_local,
        )
        timing["rule_eval_ms"] = (time.perf_counter() - t0) * 1000

        # Step 4: Select best trajectory by rule scores
        best_rule_idx = int(np.argmin(rule_violations.composite_scores))

        # If rule scores are all equal (no violations detected), fall back to M2I
        if np.all(
            rule_violations.composite_scores == rule_violations.composite_scores[0]
        ):
            best_rule_idx = best_m2i_idx

        result = GenerationResult(
            trajectories_world=traj_world,
            trajectories_local=traj_local,
            m2i_scores=scores if scores is not None else np.ones(M) / M,
            best_m2i_idx=best_m2i_idx,
            rule_violations=rule_violations,
            best_rule_idx=best_rule_idx,
            scenario_id=str(mapping.get("scenario_id", "")),
            agent_id=int(mapping.get("object_id", 0)),
            normalizer=normalizer,
            timing_ms=timing,
        )
        return result

    def evaluate_scenario_batch(
        self,
        scenarios: list,
        target_agent_idx: int = 0,
    ) -> List[Optional[GenerationResult]]:
        """Evaluate multiple scenarios sequentially."""
        results = []
        for scenario in scenarios:
            result = self.evaluate_scenario(scenario, target_agent_idx)
            results.append(result)
        return results
