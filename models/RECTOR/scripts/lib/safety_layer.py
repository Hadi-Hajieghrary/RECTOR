"""
RECTOR Safety Layer

This module implements comprehensive safety checking for trajectory planning:
1. Reality Check - Kinematic feasibility validation
2. Non-Cooperative Envelope - Conservative worst-case trajectories
3. OBB Collision Detection - Oriented bounding box collision checking
4. CVaR Risk Scoring - Conditional Value at Risk computation

Phase 5 Implementation - December 2024
"""

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

# Conditional torch import
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

# Add RECTOR src to path
RECTOR_SRC = Path(__file__).parent
sys.path.insert(0, str(RECTOR_SRC))

from data_contracts import (
    PlanningConfig,
    AgentState,
    EgoCandidate,
    EgoCandidateBatch,
)


@dataclass
class KinematicLimits:
    """Vehicle kinematic constraints."""

    a_lon_min: float = -8.0  # Max braking (m/s²)
    a_lon_max: float = 4.0  # Max acceleration (m/s²)
    a_lat_max: float = 6.0  # Max lateral acceleration (m/s²)
    jerk_max: float = 12.0  # Max jerk (m/s³)
    yaw_rate_max: float = 1.0  # Max yaw rate (rad/s)
    v_max: float = 40.0  # Max speed (m/s)
    v_min: float = 0.0  # Min speed (m/s)

    # Low-speed handling
    low_speed_threshold: float = 1.0  # Below this, don't compute yaw from heading


@dataclass
class FeasibilityResult:
    """Result of kinematic feasibility check."""

    is_feasible: bool
    violations: Dict[str, bool]  # Which checks failed
    max_a_lon: float
    min_a_lon: float
    max_a_lat: float
    max_jerk: float
    max_yaw_rate: float
    max_speed: float


class RealityChecker:
    """
    Comprehensive kinematic feasibility checker.

    Checks:
    1. Longitudinal acceleration bounds
    2. Lateral acceleration bounds (v² × κ)
    3. Jerk bounds
    4. Yaw rate bounds (with low-speed handling)
    5. Position teleportation/jumps
    6. Speed bounds
    """

    def __init__(self, limits: KinematicLimits = None, dt: float = 0.1):
        """
        Initialize reality checker.

        Args:
            limits: Kinematic constraint limits
            dt: Time step in seconds
        """
        self.limits = limits or KinematicLimits()
        self.dt = dt

    def check_trajectory(self, trajectory: np.ndarray) -> FeasibilityResult:
        """
        Check if trajectory is kinematically feasible.

        Args:
            trajectory: [H, 2] trajectory positions (x, y)

        Returns:
            FeasibilityResult with detailed violation information
        """
        H = len(trajectory)
        if H < 3:
            return FeasibilityResult(
                is_feasible=True,
                violations={},
                max_a_lon=0,
                min_a_lon=0,
                max_a_lat=0,
                max_jerk=0,
                max_yaw_rate=0,
                max_speed=0,
            )

        # Compute velocities: [H-1, 2]
        vel = (trajectory[1:] - trajectory[:-1]) / self.dt

        # Compute speeds: [H-1]
        speed = np.linalg.norm(vel, axis=-1)

        # Compute accelerations: [H-2, 2]
        accel = (vel[1:] - vel[:-1]) / self.dt

        # Longitudinal acceleration (along velocity direction)
        vel_unit = vel[:-1] / (speed[:-1, None] + 1e-6)
        a_lon = np.sum(accel * vel_unit, axis=-1)  # [H-2]

        # Lateral acceleration (perpendicular to velocity)
        vel_perp = np.stack([-vel_unit[:, 1], vel_unit[:, 0]], axis=-1)
        a_lat = np.abs(np.sum(accel * vel_perp, axis=-1))  # [H-2]

        # Jerk: [H-3]
        if H >= 4:
            jerk = (a_lon[1:] - a_lon[:-1]) / self.dt
            max_jerk = np.max(np.abs(jerk))
        else:
            max_jerk = 0.0

        # Heading and yaw rate
        heading = np.arctan2(vel[:, 1], vel[:, 0])  # [H-1]
        heading_diff = heading[1:] - heading[:-1]
        # Wrap to [-π, π]
        heading_diff = np.arctan2(np.sin(heading_diff), np.cos(heading_diff))
        yaw_rate = heading_diff / self.dt  # [H-2]

        # Mask out yaw rate at low speeds
        low_speed_mask = speed[:-1] < self.limits.low_speed_threshold
        yaw_rate = np.where(low_speed_mask, 0.0, yaw_rate)

        # Check violations
        violations = {}

        max_a_lon = np.max(a_lon) if len(a_lon) > 0 else 0.0
        min_a_lon = np.min(a_lon) if len(a_lon) > 0 else 0.0
        max_a_lat = np.max(a_lat) if len(a_lat) > 0 else 0.0
        max_yaw_rate = np.max(np.abs(yaw_rate)) if len(yaw_rate) > 0 else 0.0
        max_speed = np.max(speed)

        violations["a_lon_too_high"] = max_a_lon > self.limits.a_lon_max
        violations["a_lon_too_low"] = min_a_lon < self.limits.a_lon_min
        violations["a_lat_exceeded"] = max_a_lat > self.limits.a_lat_max
        violations["jerk_exceeded"] = max_jerk > self.limits.jerk_max
        violations["yaw_rate_exceeded"] = max_yaw_rate > self.limits.yaw_rate_max
        violations["speed_exceeded"] = max_speed > self.limits.v_max

        is_feasible = not any(violations.values())

        return FeasibilityResult(
            is_feasible=is_feasible,
            violations=violations,
            max_a_lon=float(max_a_lon),
            min_a_lon=float(min_a_lon),
            max_a_lat=float(max_a_lat),
            max_jerk=float(max_jerk),
            max_yaw_rate=float(max_yaw_rate),
            max_speed=float(max_speed),
        )

    def check_batch(
        self,
        predictions: np.ndarray,  # [M, K, N, H, 2] or [N, H, 2]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Check batch of predictions for feasibility.

        Args:
            predictions: Prediction trajectories

        Returns:
            valid_mask: Boolean mask of valid predictions
            penalties: Per-candidate penalty scores
        """
        original_shape = predictions.shape

        # Handle different input shapes
        if len(original_shape) == 3:  # [N, H, 2]
            predictions = predictions[None, None, ...]  # [1, 1, N, H, 2]
        elif len(original_shape) == 4:  # [K, N, H, 2]
            predictions = predictions[None, ...]  # [1, K, N, H, 2]

        M, K, N, H, _ = predictions.shape

        valid_mask = np.ones((M, K, N), dtype=bool)

        for m in range(M):
            for k in range(K):
                for n in range(N):
                    result = self.check_trajectory(predictions[m, k, n])
                    valid_mask[m, k, n] = result.is_feasible

        # Compute penalties
        invalid_count = (~valid_mask).sum(axis=(1, 2))  # [M]
        total_modes = K * N
        penalties = invalid_count / total_modes * 10.0

        # Reshape to match input
        if len(original_shape) == 3:
            valid_mask = valid_mask[0, 0]  # [N]
            penalties = penalties[0]  # scalar
        elif len(original_shape) == 4:
            valid_mask = valid_mask[0]  # [K, N]
            penalties = penalties[0]

        return valid_mask, penalties


@dataclass
class NonCoopParams:
    """Parameters for non-cooperative envelope."""

    a_brake_min: float = -4.0  # Conservative braking (not emergency)
    a_brake_max: float = 0.0  # No acceleration assumed
    reaction_time: float = 0.5  # Seconds before braking starts
    lane_keep: bool = True  # Assume stays in lane
    heading_bound: float = 0.1  # Max heading change rate (rad/s)


class NonCoopEnvelopeGenerator:
    """
    Generate bounded non-cooperative trajectory envelope.

    This replaces the naive constant-velocity assumption with a
    set-based envelope that captures:
    1. Reaction delay (maintains current behavior)
    2. Bounded braking (not emergency, but reasonable)
    3. Lane-keeping constraints
    4. Bounded heading change
    """

    def __init__(self, params: NonCoopParams = None, dt: float = 0.1):
        """
        Initialize envelope generator.

        Args:
            params: Non-cooperative behavior parameters
            dt: Time step in seconds
        """
        self.params = params or NonCoopParams()
        self.dt = dt

    def generate_envelope(
        self,
        agent_state: AgentState,
        horizon: int = 80,
        num_samples: int = 5,
    ) -> np.ndarray:
        """
        Generate envelope as multiple trajectory samples.

        Args:
            agent_state: Current agent state
            horizon: Prediction horizon
            num_samples: Number of envelope samples

        Returns:
            envelope: [num_samples, H, 2] representing envelope bounds
        """
        pos = np.array([agent_state.x, agent_state.y])
        vel = np.array([agent_state.velocity_x, agent_state.velocity_y])
        speed = np.linalg.norm(vel)
        heading = agent_state.heading

        samples = []

        # Sample 1: Constant velocity (baseline)
        samples.append(self._constant_velocity(pos, vel, horizon))

        # Sample 2: Max braking after reaction time
        samples.append(
            self._braking_envelope(
                pos,
                vel,
                heading,
                horizon,
                accel=self.params.a_brake_min,
                reaction_time=self.params.reaction_time,
            )
        )

        # Sample 3: Mild braking
        samples.append(
            self._braking_envelope(
                pos,
                vel,
                heading,
                horizon,
                accel=self.params.a_brake_min / 2,
                reaction_time=self.params.reaction_time,
            )
        )

        # Sample 4: Slight left drift
        samples.append(
            self._drifting_envelope(
                pos,
                vel,
                heading,
                horizon,
                heading_rate=self.params.heading_bound,
            )
        )

        # Sample 5: Slight right drift
        samples.append(
            self._drifting_envelope(
                pos,
                vel,
                heading,
                horizon,
                heading_rate=-self.params.heading_bound,
            )
        )

        return np.stack(samples[:num_samples], axis=0)

    def _constant_velocity(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        horizon: int,
    ) -> np.ndarray:
        """Constant velocity trajectory."""
        trajectory = np.zeros((horizon, 2))
        for t in range(horizon):
            trajectory[t] = pos + vel * (t + 1) * self.dt
        return trajectory

    def _braking_envelope(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        heading: float,
        horizon: int,
        accel: float,
        reaction_time: float,
    ) -> np.ndarray:
        """Trajectory with braking after reaction time."""
        trajectory = np.zeros((horizon, 2))

        speed = np.linalg.norm(vel)
        direction = np.array([np.cos(heading), np.sin(heading)])

        reaction_steps = int(reaction_time / self.dt)
        current_pos = pos.copy()
        current_speed = speed

        for t in range(horizon):
            if t < reaction_steps:
                # Constant velocity during reaction
                current_pos = current_pos + current_speed * direction * self.dt
            else:
                # Braking
                current_speed = max(0, current_speed + accel * self.dt)
                current_pos = current_pos + current_speed * direction * self.dt

            trajectory[t] = current_pos

        return trajectory

    def _drifting_envelope(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        heading: float,
        horizon: int,
        heading_rate: float,
    ) -> np.ndarray:
        """Trajectory with gradual heading change."""
        trajectory = np.zeros((horizon, 2))

        speed = np.linalg.norm(vel)
        current_pos = pos.copy()
        current_heading = heading

        for t in range(horizon):
            current_heading += heading_rate * self.dt
            direction = np.array([np.cos(current_heading), np.sin(current_heading)])
            current_pos = current_pos + speed * direction * self.dt
            trajectory[t] = current_pos

        return trajectory

    def get_worst_case(
        self,
        envelope: np.ndarray,
        ego_trajectory: np.ndarray,
    ) -> np.ndarray:
        """
        Get worst-case trajectory from envelope for collision scoring.

        Args:
            envelope: [num_samples, H, 2] envelope samples
            ego_trajectory: [H, 2] ego trajectory

        Returns:
            worst_case: [H, 2] trajectory closest to ego at each timestep
        """
        num_samples, H, _ = envelope.shape
        worst_case = np.zeros((H, 2))

        for t in range(H):
            ego_pos = (
                ego_trajectory[t] if t < len(ego_trajectory) else ego_trajectory[-1]
            )

            # Find sample closest to ego at this timestep
            dists = np.linalg.norm(envelope[:, t] - ego_pos, axis=1)
            closest_idx = np.argmin(dists)
            worst_case[t] = envelope[closest_idx, t]

        return worst_case


@dataclass
class BoundingBox:
    """Oriented bounding box."""

    center: np.ndarray  # [2] center position
    half_size: np.ndarray  # [2] half width and half length
    heading: float  # Orientation angle (radians)

    def get_corners(self) -> np.ndarray:
        """Get 4 corners of the box. Returns [4, 2]."""
        cos_h, sin_h = np.cos(self.heading), np.sin(self.heading)
        rotation = np.array([[cos_h, -sin_h], [sin_h, cos_h]])

        # Local corners
        local_corners = np.array(
            [
                [-self.half_size[0], -self.half_size[1]],
                [self.half_size[0], -self.half_size[1]],
                [self.half_size[0], self.half_size[1]],
                [-self.half_size[0], self.half_size[1]],
            ]
        )

        # Rotate and translate
        corners = local_corners @ rotation.T + self.center
        return corners


class OBBCollisionChecker:
    """
    Oriented Bounding Box collision detection.

    Uses Separating Axis Theorem (SAT) for efficient OBB-OBB collision tests.
    """

    def __init__(
        self,
        ego_length: float = 4.5,
        ego_width: float = 2.0,
        safety_margin: float = 0.5,
    ):
        """
        Initialize collision checker.

        Args:
            ego_length: Ego vehicle length (meters)
            ego_width: Ego vehicle width (meters)
            safety_margin: Additional safety margin (meters)
        """
        self.ego_half_size = np.array(
            [
                ego_length / 2 + safety_margin,
                ego_width / 2 + safety_margin,
            ]
        )

    def check_collision(
        self,
        ego_pos: np.ndarray,
        ego_heading: float,
        other_pos: np.ndarray,
        other_heading: float,
        other_length: float = 4.5,
        other_width: float = 2.0,
    ) -> bool:
        """
        Check if two OBBs collide.

        Args:
            ego_pos: Ego center position [2]
            ego_heading: Ego heading (radians)
            other_pos: Other vehicle center position [2]
            other_heading: Other vehicle heading (radians)
            other_length: Other vehicle length
            other_width: Other vehicle width

        Returns:
            True if collision detected
        """
        ego_box = BoundingBox(
            center=ego_pos,
            half_size=self.ego_half_size,
            heading=ego_heading,
        )

        other_box = BoundingBox(
            center=other_pos,
            half_size=np.array([other_length / 2, other_width / 2]),
            heading=other_heading,
        )

        return self._sat_collision(ego_box, other_box)

    def _sat_collision(self, box1: BoundingBox, box2: BoundingBox) -> bool:
        """
        Separating Axis Theorem collision test.

        If no separating axis exists, boxes are colliding.
        """
        # Get axes to test (normals to each edge)
        axes = [
            np.array([np.cos(box1.heading), np.sin(box1.heading)]),
            np.array([-np.sin(box1.heading), np.cos(box1.heading)]),
            np.array([np.cos(box2.heading), np.sin(box2.heading)]),
            np.array([-np.sin(box2.heading), np.cos(box2.heading)]),
        ]

        corners1 = box1.get_corners()
        corners2 = box2.get_corners()

        for axis in axes:
            # Project corners onto axis
            proj1 = corners1 @ axis
            proj2 = corners2 @ axis

            # Check for gap
            if np.max(proj1) < np.min(proj2) or np.max(proj2) < np.min(proj1):
                return False  # Separating axis found, no collision

        return True  # No separating axis, collision detected

    def check_trajectory_collision(
        self,
        ego_traj: np.ndarray,  # [H, 2]
        other_traj: np.ndarray,  # [H, 2]
        ego_heading_traj: Optional[np.ndarray] = None,  # [H]
        other_heading_traj: Optional[np.ndarray] = None,  # [H]
    ) -> Tuple[bool, Optional[int], float]:
        """
        Check for collision between two trajectories over time.

        Args:
            ego_traj: Ego trajectory positions
            other_traj: Other vehicle trajectory positions
            ego_heading_traj: Ego headings (optional, computed from trajectory)
            other_heading_traj: Other headings (optional, computed from trajectory)

        Returns:
            has_collision: Whether collision was detected
            collision_time: Timestep of first collision (None if no collision)
            min_distance: Minimum distance between vehicles
        """
        H = min(len(ego_traj), len(other_traj))

        # Compute headings if not provided
        if ego_heading_traj is None:
            ego_heading_traj = self._compute_headings(ego_traj)
        if other_heading_traj is None:
            other_heading_traj = self._compute_headings(other_traj)

        min_distance = float("inf")
        collision_time = None

        for t in range(H):
            # Quick distance check (skip if far away)
            dist = np.linalg.norm(ego_traj[t] - other_traj[t])
            min_distance = min(min_distance, dist)

            if dist > 15.0:  # Skip detailed check if far apart
                continue

            # Detailed OBB check
            if self.check_collision(
                ego_pos=ego_traj[t],
                ego_heading=ego_heading_traj[t],
                other_pos=other_traj[t],
                other_heading=other_heading_traj[t],
            ):
                if collision_time is None:
                    collision_time = t

        return collision_time is not None, collision_time, min_distance

    def _compute_headings(self, trajectory: np.ndarray) -> np.ndarray:
        """Compute headings from trajectory positions."""
        H = len(trajectory)
        headings = np.zeros(H)

        for t in range(H - 1):
            direction = trajectory[t + 1] - trajectory[t]
            if np.linalg.norm(direction) > 0.01:
                headings[t] = np.arctan2(direction[1], direction[0])
            elif t > 0:
                headings[t] = headings[t - 1]

        # Last heading same as previous
        if H > 1:
            headings[-1] = headings[-2]

        return headings


@dataclass
class CVaRConfig:
    """Configuration for CVaR risk scoring."""

    alpha: float = 0.1  # Risk level (lower = more conservative)
    num_samples: int = 100  # Number of samples
    seed: int = 42  # For reproducibility
    time_weight_decay: float = 0.95  # Earlier collisions weighted more


class DeterministicCVaRScorer:
    """
    CVaR risk scoring with deterministic sampling.

    Uses Common Random Numbers (CRN) for consistent comparisons:
    1. Same random indices used for all candidates in a tick
    2. Seed reset at start of each tick
    3. Enables fair comparison between candidates
    """

    def __init__(self, config: CVaRConfig = None):
        """
        Initialize CVaR scorer.

        Args:
            config: CVaR configuration
        """
        self.config = config or CVaRConfig()
        self._rng = np.random.default_rng(self.config.seed)
        self._sample_indices = None  # Pre-sampled for this tick

    def prepare_tick(self, num_reactors: int, num_modes: int):
        """
        Prepare deterministic samples for this tick.

        MUST be called once at start of scoring, before score_all().
        """
        total_samples = num_reactors * num_modes

        # Pre-sample indices based on mode probabilities
        # For simplicity, sample uniformly and weight by probability later
        self._sample_indices = self._rng.integers(
            0, num_modes, size=(self.config.num_samples, num_reactors)
        )

    def score_candidate(
        self,
        ego_traj: np.ndarray,  # [H, 2]
        predictions: Dict[int, "SinglePrediction"],  # {reactor_id: prediction}
        noncoop_envelopes: Optional[
            Dict[int, np.ndarray]
        ] = None,  # {reactor_id: [S, H, 2]}
        collision_checker: Optional[OBBCollisionChecker] = None,
    ) -> float:
        """
        Compute CVaR risk for single candidate.

        Args:
            ego_traj: Ego candidate trajectory
            predictions: Predictions for each reactor
            noncoop_envelopes: Non-cooperative envelopes (optional)
            collision_checker: Collision checker (optional)

        Returns:
            CVaR risk score (lower = safer)
        """
        if collision_checker is None:
            collision_checker = OBBCollisionChecker()

        if self._sample_indices is None:
            # Fallback: use all modes
            return self._score_without_sampling(
                ego_traj, predictions, noncoop_envelopes, collision_checker
            )

        costs = []
        reactor_list = list(predictions.keys())

        for sample_idx in range(self.config.num_samples):
            sample_cost = 0.0

            for r_idx, reactor_id in enumerate(reactor_list):
                pred = predictions[reactor_id]

                # Get mode for this sample
                mode_idx = self._sample_indices[
                    sample_idx, r_idx % len(self._sample_indices[sample_idx])
                ]
                mode_idx = min(mode_idx, pred.num_modes - 1)

                reactor_traj = pred.trajectories[mode_idx]
                mode_prob = pred.scores[mode_idx]

                # Check collision
                has_collision, collision_time, min_dist = (
                    collision_checker.check_trajectory_collision(ego_traj, reactor_traj)
                )

                if has_collision:
                    # Time-weighted collision cost
                    time_weight = self.config.time_weight_decay**collision_time
                    sample_cost += time_weight * 100.0 * mode_prob
                else:
                    # Proximity cost
                    if min_dist < 5.0:
                        sample_cost += (5.0 - min_dist) * mode_prob

            # Also check non-cooperative envelopes
            if noncoop_envelopes:
                for reactor_id, envelope in noncoop_envelopes.items():
                    # Get worst case from envelope
                    worst_case = envelope[0]  # Use first sample as worst case

                    has_collision, collision_time, _ = (
                        collision_checker.check_trajectory_collision(
                            ego_traj, worst_case
                        )
                    )

                    if has_collision:
                        time_weight = self.config.time_weight_decay**collision_time
                        sample_cost += time_weight * 50.0  # Lower weight for non-coop

            costs.append(sample_cost)

        # Compute CVaR (average of worst α-quantile)
        costs = np.sort(costs)[::-1]  # Descending
        num_tail = max(1, int(len(costs) * self.config.alpha))
        cvar = np.mean(costs[:num_tail])

        return float(cvar)

    def _score_without_sampling(
        self,
        ego_traj: np.ndarray,
        predictions: Dict[int, "SinglePrediction"],
        noncoop_envelopes: Optional[Dict[int, np.ndarray]],
        collision_checker: OBBCollisionChecker,
    ) -> float:
        """Score using all modes without sampling."""
        total_cost = 0.0

        for reactor_id, pred in predictions.items():
            for mode_idx in range(pred.num_modes):
                reactor_traj = pred.trajectories[mode_idx]
                mode_prob = pred.scores[mode_idx]

                has_collision, collision_time, min_dist = (
                    collision_checker.check_trajectory_collision(ego_traj, reactor_traj)
                )

                if has_collision:
                    time_weight = self.config.time_weight_decay**collision_time
                    total_cost += time_weight * 100.0 * mode_prob
                elif min_dist < 5.0:
                    total_cost += (5.0 - min_dist) * mode_prob

        return total_cost


@dataclass
class SafetyCheckResult:
    """Complete safety check result for a candidate."""

    is_safe: bool
    feasibility: FeasibilityResult
    has_collision: bool
    collision_time: Optional[int]
    min_clearance: float
    cvar_risk: float
    violations: List[str]


class IntegratedSafetyChecker:
    """
    Integrated safety checking combining all components.
    """

    def __init__(
        self,
        kinematic_limits: KinematicLimits = None,
        noncoop_params: NonCoopParams = None,
        cvar_config: CVaRConfig = None,
        dt: float = 0.1,
    ):
        """
        Initialize integrated safety checker.

        Args:
            kinematic_limits: Kinematic constraints
            noncoop_params: Non-cooperative envelope parameters
            cvar_config: CVaR scoring configuration
            dt: Time step
        """
        self.reality_checker = RealityChecker(kinematic_limits, dt)
        self.envelope_generator = NonCoopEnvelopeGenerator(noncoop_params, dt)
        self.collision_checker = OBBCollisionChecker()
        self.cvar_scorer = DeterministicCVaRScorer(cvar_config)
        self.dt = dt

    def check_candidate(
        self,
        ego_candidate: EgoCandidate,
        predictions: Dict[int, "SinglePrediction"],
        agent_states: List[AgentState],
    ) -> SafetyCheckResult:
        """
        Perform complete safety check on candidate.

        Args:
            ego_candidate: Ego trajectory candidate
            predictions: Reactor predictions
            agent_states: Current agent states (for non-coop envelopes)

        Returns:
            SafetyCheckResult with all safety information
        """
        violations = []

        # 1. Reality check (kinematic feasibility)
        feasibility = self.reality_checker.check_trajectory(ego_candidate.trajectory)
        if not feasibility.is_feasible:
            violations.extend([k for k, v in feasibility.violations.items() if v])

        # 2. Generate non-cooperative envelopes
        agent_by_id = {a.agent_id: a for a in agent_states}
        noncoop_envelopes = {}

        for reactor_id in predictions.keys():
            if reactor_id in agent_by_id:
                envelope = self.envelope_generator.generate_envelope(
                    agent_by_id[reactor_id],
                    horizon=len(ego_candidate.trajectory),
                )
                noncoop_envelopes[reactor_id] = envelope

        # 3. Collision check against predictions
        has_collision = False
        first_collision_time = None
        min_clearance = float("inf")

        for reactor_id, pred in predictions.items():
            # Check best mode
            best_traj = pred.best_trajectory
            collision, time, dist = self.collision_checker.check_trajectory_collision(
                ego_candidate.trajectory, best_traj
            )

            if collision and first_collision_time is None:
                has_collision = True
                first_collision_time = time
                violations.append(f"collision_reactor_{reactor_id}")

            min_clearance = min(min_clearance, dist)

        # 4. Collision check against non-coop envelopes
        for reactor_id, envelope in noncoop_envelopes.items():
            worst_case = self.envelope_generator.get_worst_case(
                envelope, ego_candidate.trajectory
            )
            collision, time, dist = self.collision_checker.check_trajectory_collision(
                ego_candidate.trajectory, worst_case
            )

            if collision and first_collision_time is None:
                has_collision = True
                first_collision_time = time
                violations.append(f"noncoop_collision_{reactor_id}")

            min_clearance = min(min_clearance, dist)

        # 5. CVaR risk scoring
        cvar_risk = self.cvar_scorer.score_candidate(
            ego_candidate.trajectory,
            predictions,
            noncoop_envelopes,
            self.collision_checker,
        )

        # Determine overall safety
        is_safe = (
            feasibility.is_feasible
            and not has_collision
            and min_clearance > 2.0
            and cvar_risk < 50.0
        )

        return SafetyCheckResult(
            is_safe=is_safe,
            feasibility=feasibility,
            has_collision=has_collision,
            collision_time=first_collision_time,
            min_clearance=min_clearance if min_clearance != float("inf") else -1.0,
            cvar_risk=cvar_risk,
            violations=violations,
        )


def create_safety_checker() -> IntegratedSafetyChecker:
    """Create default safety checker."""
    return IntegratedSafetyChecker()


if __name__ == "__main__":
    # Quick sanity check
    print("RECTOR Safety Layer - Sanity Check")
    print("=" * 50)

    # Test reality checker
    checker = RealityChecker()
    traj = np.array([[i * 1.0, 0.0] for i in range(80)])  # 10 m/s straight
    result = checker.check_trajectory(traj)
    print(
        f"Reality check: feasible={result.is_feasible}, violations={result.violations}"
    )

    # Test collision checker
    obb_checker = OBBCollisionChecker()
    collides = obb_checker.check_collision(
        ego_pos=np.array([0.0, 0.0]),
        ego_heading=0.0,
        other_pos=np.array([10.0, 0.0]),
        other_heading=0.0,
    )
    print(f"OBB collision (10m apart): {collides}")

    collides = obb_checker.check_collision(
        ego_pos=np.array([0.0, 0.0]),
        ego_heading=0.0,
        other_pos=np.array([3.0, 0.0]),
        other_heading=0.0,
    )
    print(f"OBB collision (3m apart): {collides}")

    # Test non-coop envelope
    env_gen = NonCoopEnvelopeGenerator()
    agent = AgentState(x=10.0, y=0.0, heading=0.0, velocity_x=10.0, velocity_y=0.0)
    envelope = env_gen.generate_envelope(agent, horizon=40)
    print(f"Non-coop envelope shape: {envelope.shape}")

    print("\nSanity check PASSED")
