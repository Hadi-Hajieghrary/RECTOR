"""
RECTOR Planning Loop

This module implements the main planning loop that integrates:
1. Candidate Generation - Generate M ego trajectory candidates
2. Reactor Selection - Select K closest/relevant reactors
3. Batched Prediction - Run M2I conditional model
4. Scoring - Score candidates based on safety and progress
5. Selection - Pick best candidate

Phase 4 Implementation - December 2024
"""

import math
import time
import sys
from dataclasses import dataclass, field
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

# Add RECTOR lib to path
RECTOR_LIB = Path(__file__).parent
sys.path.insert(0, str(RECTOR_LIB))

from data_contracts import (
    PlanningConfig,
    AgentState,
    EgoCandidate,
    EgoCandidateBatch,
)

from safety_layer import OBBCollisionChecker


@dataclass
class SimpleSafetyScore:
    """Simplified safety score for scoring."""

    ttc: float  # Time to collision (-1 if none)
    min_distance: float  # Min distance to any reactor
    collision_probability: float  # Probability of collision
    is_safe: bool  # Overall safety flag


@dataclass
class LaneInfo:
    """Lane information for candidate generation."""

    centerline: np.ndarray  # [N, 2] lane centerline points
    speed_limit: float = 25.0  # m/s
    width: float = 3.7  # meters


class CandidateGenerator:
    """
    Generate M ego trajectory candidates.

    Strategies:
    1. Lane-following at different speeds
    2. Lane changes (left/right)
    3. Yield/stop maneuvers
    4. Previous trajectory with perturbations

    Each candidate is a sequence of (x, y, yaw, v) over H timesteps.
    """

    def __init__(
        self,
        m_candidates: int = 16,
        horizon_steps: int = 80,
        dt: float = 0.1,
        speed_levels: List[float] = None,
    ):
        """
        Initialize candidate generator.

        Args:
            m_candidates: Number of candidates to generate
            horizon_steps: Planning horizon in timesteps
            dt: Time step in seconds
            speed_levels: Speed levels as fractions of speed limit
        """
        self.m_candidates = m_candidates
        self.horizon_steps = horizon_steps
        self.dt = dt
        self.speed_levels = speed_levels or [0.0, 0.5, 0.75, 1.0, 1.1]

    def generate(
        self,
        ego_state: AgentState,
        current_lane: Optional[LaneInfo] = None,
        left_lane: Optional[LaneInfo] = None,
        right_lane: Optional[LaneInfo] = None,
        goal: Optional[np.ndarray] = None,
    ) -> EgoCandidateBatch:
        """
        Generate M candidates starting from current ego state.

        Args:
            ego_state: Current ego vehicle state
            current_lane: Current lane centerline
            left_lane: Left lane (if available)
            right_lane: Right lane (if available)
            goal: Goal position [x, y]

        Returns:
            EgoCandidateBatch containing M candidates
        """
        candidates = []

        # If no lane info, generate simple straight/curved paths
        if current_lane is None:
            current_lane = self._generate_default_lane(ego_state)

        # Strategy 1: Lane following at different speeds
        for i, speed_factor in enumerate(self.speed_levels):
            target_speed = current_lane.speed_limit * speed_factor
            traj = self._generate_lane_following(
                ego_state, current_lane.centerline, target_speed
            )
            candidates.append(
                EgoCandidate(
                    trajectory=traj[:, :2],  # [H, 2] xy only
                    velocities=np.full(self.horizon_steps, target_speed),
                    candidate_id=len(candidates),
                    generation_method="lane_follow",
                )
            )

        # Strategy 2: Lane change left (if available)
        if left_lane is not None:
            for speed_factor in [0.8, 1.0]:
                target_speed = left_lane.speed_limit * speed_factor
                traj = self._generate_lane_change(
                    ego_state,
                    current_lane.centerline,
                    left_lane.centerline,
                    target_speed,
                    direction="left",
                )
                candidates.append(
                    EgoCandidate(
                        trajectory=traj[:, :2],  # [H, 2] xy only
                        velocities=np.full(self.horizon_steps, target_speed),
                        candidate_id=len(candidates),
                        generation_method="lane_change_left",
                    )
                )

        # Strategy 3: Lane change right (if available)
        if right_lane is not None:
            for speed_factor in [0.8, 1.0]:
                target_speed = right_lane.speed_limit * speed_factor
                traj = self._generate_lane_change(
                    ego_state,
                    current_lane.centerline,
                    right_lane.centerline,
                    target_speed,
                    direction="right",
                )
                candidates.append(
                    EgoCandidate(
                        trajectory=traj[:, :2],  # [H, 2] xy only
                        velocities=np.full(self.horizon_steps, target_speed),
                        candidate_id=len(candidates),
                        generation_method="lane_change_right",
                    )
                )

        # Strategy 4: Emergency stop
        traj = self._generate_emergency_stop(ego_state)
        candidates.append(
            EgoCandidate(
                trajectory=traj[:, :2],  # [H, 2] xy only
                velocities=self._compute_decel_profile(ego_state.velocity_x),
                candidate_id=len(candidates),
                generation_method="emergency_stop",
            )
        )

        # Strategy 5: Goal-directed (if goal provided)
        if goal is not None:
            traj = self._generate_goal_directed(ego_state, goal)
            candidates.append(
                EgoCandidate(
                    trajectory=traj[:, :2],  # [H, 2] xy only
                    velocities=np.full(self.horizon_steps, ego_state.velocity_x),
                    candidate_id=len(candidates),
                    generation_method="goal_directed",
                )
            )

        # Pad or truncate to exactly m_candidates
        while len(candidates) < self.m_candidates:
            # Add perturbations of existing candidates
            base_idx = len(candidates) % max(1, len(candidates))
            if candidates:
                base = candidates[base_idx]
                perturbed = self._perturb_trajectory(base)
                perturbed = EgoCandidate(
                    trajectory=perturbed.trajectory,
                    velocities=perturbed.velocities,
                    candidate_id=len(candidates),
                    generation_method=f"perturbed_{base.generation_method}",
                )
                candidates.append(perturbed)
            else:
                break

        candidates = candidates[: self.m_candidates]

        return EgoCandidateBatch.from_candidates(candidates)

    def _generate_default_lane(self, ego_state: AgentState) -> LaneInfo:
        """Generate default lane going straight ahead."""
        heading = ego_state.heading
        cos_h, sin_h = np.cos(heading), np.sin(heading)

        # Generate points along heading direction
        distances = np.linspace(0, 100, 50)
        centerline = np.stack(
            [
                ego_state.x + distances * cos_h,
                ego_state.y + distances * sin_h,
            ],
            axis=1,
        )

        return LaneInfo(centerline=centerline)

    def _generate_lane_following(
        self,
        ego_state: AgentState,
        centerline: np.ndarray,
        target_speed: float,
    ) -> np.ndarray:
        """
        Generate lane-following trajectory.

        Returns:
            [H, 4] trajectory (x, y, yaw, v)
        """
        # Find closest point on centerline
        dists = np.linalg.norm(
            centerline - np.array([ego_state.x, ego_state.y]), axis=1
        )
        closest_idx = np.argmin(dists)

        trajectory = np.zeros((self.horizon_steps, 4))

        current_speed = ego_state.velocity_x
        current_pos = np.array([ego_state.x, ego_state.y])
        current_yaw = ego_state.heading

        lane_idx = closest_idx

        for t in range(self.horizon_steps):
            # Accelerate/decelerate toward target speed
            speed_diff = target_speed - current_speed
            accel = np.clip(speed_diff, -3.0, 2.0)  # m/s^2
            current_speed = np.clip(current_speed + accel * self.dt, 0, 35)

            # Move along lane
            dist_to_travel = current_speed * self.dt

            while dist_to_travel > 0 and lane_idx < len(centerline) - 1:
                segment_vec = centerline[lane_idx + 1] - centerline[lane_idx]
                segment_len = np.linalg.norm(segment_vec)

                if segment_len > 0:
                    dist_along = min(dist_to_travel, segment_len)
                    current_pos = centerline[lane_idx] + segment_vec * (
                        dist_along / segment_len
                    )
                    current_yaw = np.arctan2(segment_vec[1], segment_vec[0])

                    if dist_to_travel >= segment_len:
                        lane_idx += 1
                    dist_to_travel -= dist_along
                else:
                    lane_idx += 1

            trajectory[t, 0] = current_pos[0]
            trajectory[t, 1] = current_pos[1]
            trajectory[t, 2] = current_yaw
            trajectory[t, 3] = current_speed

        return trajectory

    def _generate_lane_change(
        self,
        ego_state: AgentState,
        from_lane: np.ndarray,
        to_lane: np.ndarray,
        target_speed: float,
        direction: str,
    ) -> np.ndarray:
        """Generate lane change trajectory using cubic spline."""
        # Lane change duration
        lc_duration = 3.0  # seconds
        lc_steps = int(lc_duration / self.dt)

        trajectory = np.zeros((self.horizon_steps, 4))

        # Start with lane following in current lane
        start_traj = self._generate_lane_following(ego_state, from_lane, target_speed)

        # End with lane following in target lane
        mid_state = AgentState(
            x=start_traj[lc_steps, 0],
            y=start_traj[lc_steps, 1],
            heading=start_traj[lc_steps, 2],
            velocity_x=target_speed,
            velocity_y=0.0,
            agent_id=-1,
        )
        end_traj = self._generate_lane_following(mid_state, to_lane, target_speed)

        # Blend between lanes during transition
        for t in range(self.horizon_steps):
            if t < lc_steps:
                # During lane change: blend
                alpha = 0.5 * (1 - np.cos(np.pi * t / lc_steps))  # Smooth S-curve
                trajectory[t] = (1 - alpha) * start_traj[t] + alpha * end_traj[0]
            else:
                # After lane change: follow target lane
                trajectory[t] = end_traj[min(t - lc_steps, len(end_traj) - 1)]

        return trajectory

    def _generate_emergency_stop(self, ego_state: AgentState) -> np.ndarray:
        """Generate emergency braking trajectory."""
        trajectory = np.zeros((self.horizon_steps, 4))

        current_speed = float(ego_state.velocity_x)
        current_pos = np.array([ego_state.x, ego_state.y], dtype=np.float64)
        heading = float(ego_state.heading)

        max_decel = 5.0  # m/s^2 (comfortable emergency braking)

        for t in range(self.horizon_steps):
            trajectory[t, 0] = current_pos[0]
            trajectory[t, 1] = current_pos[1]
            trajectory[t, 2] = heading
            trajectory[t, 3] = current_speed

            # Update for next step
            if current_speed > 0:
                decel = min(max_decel, current_speed / self.dt)
                current_speed = max(0, current_speed - decel * self.dt)
                dist = current_speed * self.dt
                current_pos += dist * np.array([np.cos(heading), np.sin(heading)])

        return trajectory

    def _generate_goal_directed(
        self,
        ego_state: AgentState,
        goal: np.ndarray,
    ) -> np.ndarray:
        """Generate trajectory toward goal."""
        trajectory = np.zeros((self.horizon_steps, 4))

        current_pos = np.array([ego_state.x, ego_state.y])
        current_speed = ego_state.velocity_x
        current_yaw = ego_state.heading

        for t in range(self.horizon_steps):
            # Direction to goal
            to_goal = goal - current_pos
            dist_to_goal = np.linalg.norm(to_goal)

            if dist_to_goal > 1.0:
                target_yaw = np.arctan2(to_goal[1], to_goal[0])
            else:
                target_yaw = current_yaw

            # Smooth yaw transition
            yaw_diff = target_yaw - current_yaw
            yaw_diff = np.arctan2(np.sin(yaw_diff), np.cos(yaw_diff))
            max_yaw_rate = 0.5  # rad/s
            yaw_change = np.clip(
                yaw_diff, -max_yaw_rate * self.dt, max_yaw_rate * self.dt
            )
            current_yaw += yaw_change

            # Speed toward goal
            target_speed = min(15.0, dist_to_goal / 5.0)
            speed_diff = target_speed - current_speed
            accel = np.clip(speed_diff, -3.0, 2.0)
            current_speed = max(0, current_speed + accel * self.dt)

            # Move
            dist = current_speed * self.dt
            current_pos += dist * np.array([np.cos(current_yaw), np.sin(current_yaw)])

            trajectory[t, 0] = current_pos[0]
            trajectory[t, 1] = current_pos[1]
            trajectory[t, 2] = current_yaw
            trajectory[t, 3] = current_speed

        return trajectory

    def _compute_decel_profile(self, initial_speed: float) -> np.ndarray:
        """Compute velocity profile for deceleration."""
        velocities = np.zeros(self.horizon_steps)
        speed = initial_speed
        max_decel = 5.0

        for t in range(self.horizon_steps):
            velocities[t] = speed
            speed = max(0, speed - max_decel * self.dt)

        return velocities

    def _perturb_trajectory(self, candidate: EgoCandidate) -> EgoCandidate:
        """Add small perturbation to trajectory."""
        perturbed_traj = candidate.trajectory.copy()  # [H, 2]

        # Add lateral offset (small sine wave)
        amplitude = np.random.uniform(0.3, 1.0)  # meters
        freq = np.random.uniform(0.1, 0.3)
        t = np.arange(len(perturbed_traj)) * self.dt
        lateral_offset = amplitude * np.sin(2 * np.pi * freq * t)

        # Compute approximate heading from trajectory direction
        for i in range(len(perturbed_traj)):
            if i < len(perturbed_traj) - 1:
                direction = perturbed_traj[i + 1] - perturbed_traj[i]
                yaw = np.arctan2(direction[1], direction[0])
            # else: keep previous yaw

            # Apply offset perpendicular to heading
            perturbed_traj[i, 0] += -np.sin(yaw) * lateral_offset[i]
            perturbed_traj[i, 1] += np.cos(yaw) * lateral_offset[i]

        return EgoCandidate(
            trajectory=perturbed_traj,
            velocities=(
                candidate.velocities * np.random.uniform(0.9, 1.1)
                if candidate.velocities is not None
                else None
            ),
            candidate_id=candidate.candidate_id,
            generation_method=candidate.generation_method,
        )


class ReactorSelector:
    """
    Select K most relevant reactors (other vehicles) for prediction.

    Selection criteria:
    1. Distance to ego trajectory
    2. Whether in ego's corridor (potential conflict zone)
    3. Relative velocity (approaching vs receding)
    """

    def __init__(self, k_reactors: int = 3, corridor_width: float = 10.0):
        """
        Initialize reactor selector.

        Args:
            k_reactors: Maximum number of reactors to select
            corridor_width: Width of corridor for relevance check (meters)
        """
        self.k_reactors = k_reactors
        self.corridor_width = corridor_width

    def select(
        self,
        ego_candidates: EgoCandidateBatch,
        agent_states: List[AgentState],
        ego_state: AgentState,
    ) -> List[int]:
        """
        Select top K reactors based on relevance to ego candidates.

        Args:
            ego_candidates: Batch of ego trajectory candidates
            agent_states: List of all agent states
            ego_state: Current ego state

        Returns:
            List of agent_ids for selected reactors
        """
        if not agent_states:
            return []

        # Build ego corridor (envelope of all candidates)
        corridor = self._build_corridor(ego_candidates)

        # Score each agent
        scores = []
        for agent in agent_states:
            if agent.agent_id == ego_state.agent_id:
                continue  # Skip ego

            score = self._compute_relevance(agent, ego_state, corridor)
            scores.append((agent.agent_id, score))

        # Sort by score (higher = more relevant)
        scores.sort(key=lambda x: x[1], reverse=True)

        # Return top K
        selected = [obj_id for obj_id, _ in scores[: self.k_reactors]]
        return selected

    def _build_corridor(self, candidates: EgoCandidateBatch) -> np.ndarray:
        """
        Build corridor envelope from all candidates.

        Returns:
            [H, 4] corridor bounds (x_min, y_min, x_max, y_max)
        """
        H = candidates.candidates[0].trajectory.shape[0]
        corridor = np.zeros((H, 4))

        for t in range(H):
            x_vals = [c.trajectory[t, 0] for c in candidates.candidates]
            y_vals = [c.trajectory[t, 1] for c in candidates.candidates]

            corridor[t, 0] = min(x_vals) - self.corridor_width / 2
            corridor[t, 1] = min(y_vals) - self.corridor_width / 2
            corridor[t, 2] = max(x_vals) + self.corridor_width / 2
            corridor[t, 3] = max(y_vals) + self.corridor_width / 2

        return corridor

    def _compute_relevance(
        self,
        agent: AgentState,
        ego_state: AgentState,
        corridor: np.ndarray,
    ) -> float:
        """
        Compute relevance score for an agent.

        Higher score = more relevant (should be selected as reactor)
        """
        # Distance to ego (closer = more relevant)
        dx = agent.x - ego_state.x
        dy = agent.y - ego_state.y
        dist = np.sqrt(dx * dx + dy * dy)
        distance_score = 100.0 / (1.0 + dist)

        # Check if agent is in corridor at any timestep
        in_corridor = False

        # Simple check: extrapolate agent position
        agent_speed = np.sqrt(agent.velocity_x**2 + agent.velocity_y**2)
        agent_heading = agent.heading

        for t in range(len(corridor)):
            # Predicted agent position
            t_sec = t * 0.1
            pred_x = agent.x + agent.velocity_x * t_sec
            pred_y = agent.y + agent.velocity_y * t_sec

            # Check if in corridor bounds
            if (
                corridor[t, 0] <= pred_x <= corridor[t, 2]
                and corridor[t, 1] <= pred_y <= corridor[t, 3]
            ):
                in_corridor = True
                break

        corridor_score = 50.0 if in_corridor else 0.0

        # Relative velocity (approaching = more relevant)
        rel_vx = agent.velocity_x - ego_state.velocity_x
        rel_vy = agent.velocity_y - ego_state.velocity_y

        # Direction from ego to agent
        if dist > 0.1:
            dir_x, dir_y = dx / dist, dy / dist
            # Closing speed (positive = approaching)
            closing_speed = -(rel_vx * dir_x + rel_vy * dir_y)
            approach_score = 20.0 * max(0, closing_speed / 10.0)
        else:
            approach_score = 20.0  # Very close = very relevant

        return distance_score + corridor_score + approach_score


@dataclass
class ScoringWeights:
    """Weights for candidate scoring."""

    w_collision: float = 100.0  # Collision risk weight
    w_comfort: float = 5.0  # Comfort penalty weight
    w_progress: float = 10.0  # Progress toward goal weight
    w_lane_deviation: float = 2.0  # Lane deviation penalty
    w_hysteresis: float = 3.0  # Temporal consistency weight


class CandidateScorer:
    """
    Score candidates based on safety, comfort, and progress.

    Lower score = better candidate.

    Uses OBB (Oriented Bounding Box) collision detection instead of
    simple circle approximation for accurate collision checking.
    """

    def __init__(
        self,
        weights: ScoringWeights = None,
        dt: float = 0.1,
        vehicle_length: float = 4.5,
        vehicle_width: float = 2.0,
        safety_margin: float = 0.5,
    ):
        """
        Initialize scorer.

        Args:
            weights: Scoring weights
            dt: Time step in seconds
            vehicle_length: Ego vehicle length (meters)
            vehicle_width: Ego vehicle width (meters)
            safety_margin: Additional collision margin (meters)
        """
        self.weights = weights or ScoringWeights()
        self.dt = dt
        self.vehicle_length = vehicle_length
        self.vehicle_width = vehicle_width

        # OBB collision checker for accurate collision detection
        self.collision_checker = OBBCollisionChecker(
            ego_length=vehicle_length,
            ego_width=vehicle_width,
            safety_margin=safety_margin,
        )

    def score_all(
        self,
        candidates: EgoCandidateBatch,
        predictions: Dict[int, List["SinglePrediction"]],
        current_lane: Optional[LaneInfo] = None,
        goal: Optional[np.ndarray] = None,
        previous_selected: Optional[EgoCandidate] = None,
    ) -> Tuple[np.ndarray, List[SimpleSafetyScore]]:
        """
        Score all candidates.

        Args:
            candidates: Batch of ego candidates
            predictions: Reactor predictions {reactor_id: [predictions per candidate]}
            current_lane: Current lane info for deviation penalty
            goal: Goal position for progress reward
            previous_selected: Previously selected candidate for hysteresis

        Returns:
            scores: [M] array of scores (lower = better)
            safety_scores: [M] list of detailed safety scores
        """
        M = candidates.num_candidates
        scores = np.zeros(M)
        safety_scores = []

        for m in range(M):
            candidate = candidates.candidates[m]

            # Collision risk
            collision_score, safety = self._compute_collision_risk(
                candidate, predictions, m
            )
            safety_scores.append(safety)

            # Comfort (acceleration, jerk, lateral accel)
            comfort_score = self._compute_comfort_penalty(candidate)

            # Progress toward goal
            if goal is not None:
                progress_score = self._compute_progress(candidate, goal)
            else:
                progress_score = 0.0

            # Lane deviation
            if current_lane is not None:
                deviation_score = self._compute_lane_deviation(candidate, current_lane)
            else:
                deviation_score = 0.0

            # Hysteresis (similarity to previous)
            if previous_selected is not None:
                hysteresis_score = self._compute_hysteresis(
                    candidate, previous_selected
                )
            else:
                hysteresis_score = 0.0

            # Total score
            scores[m] = (
                self.weights.w_collision * collision_score
                + self.weights.w_comfort * comfort_score
                - self.weights.w_progress * progress_score  # Negative = good
                + self.weights.w_lane_deviation * deviation_score
                + self.weights.w_hysteresis * hysteresis_score
            )

        return scores, safety_scores

    def _compute_collision_risk(
        self,
        candidate: EgoCandidate,
        predictions: Dict[int, List["SinglePrediction"]],
        candidate_idx: int,
    ) -> Tuple[float, SimpleSafetyScore]:
        """
        Compute collision risk for candidate using OBB collision detection.

        Uses CVaR (Conditional Value at Risk) over reactor predictions
        with OBB (Oriented Bounding Box) collision checking for accuracy.
        """
        ego_traj = candidate.trajectory  # [H, 4] or [H, 2]

        min_ttc = float("inf")
        min_distance = float("inf")
        collision_prob = 0.0

        # Compute ego headings from trajectory
        ego_headings = self._compute_headings_from_trajectory(ego_traj[:, :2])

        # Check against each reactor's predictions
        for reactor_id, pred_list in predictions.items():
            if candidate_idx >= len(pred_list):
                continue

            pred = pred_list[candidate_idx]

            # Check each mode
            for mode_idx in range(pred.num_modes):
                mode_traj = pred.trajectories[mode_idx]  # [H, 2]
                mode_prob = pred.scores[mode_idx]

                # Compute reactor headings
                reactor_headings = self._compute_headings_from_trajectory(mode_traj)

                # Use OBB collision checker for accurate detection
                has_collision, collision_time, traj_min_dist = (
                    self.collision_checker.check_trajectory_collision(
                        ego_traj=ego_traj[:, :2],
                        other_traj=mode_traj,
                        ego_heading_traj=ego_headings,
                        other_heading_traj=reactor_headings,
                    )
                )

                min_distance = min(min_distance, traj_min_dist)

                if has_collision:
                    collision_prob += mode_prob

                    # Compute TTC from collision time
                    if collision_time is not None and collision_time > 0:
                        ttc = collision_time * self.dt
                        min_ttc = min(min_ttc, ttc)

        # Convert to risk score
        if collision_prob > 0:
            risk = 1.0 + np.log(1 + collision_prob * 10)
        else:
            risk = 0.0

        # Add TTC penalty
        if min_ttc < 2.0:
            risk += (2.0 - min_ttc) * 0.5

        safety = SimpleSafetyScore(
            ttc=min_ttc if min_ttc != float("inf") else -1.0,
            min_distance=min_distance if min_distance != float("inf") else -1.0,
            collision_probability=min(1.0, collision_prob),
            is_safe=collision_prob < 0.1 and min_ttc > 2.0,
        )

        return risk, safety

    def _compute_headings_from_trajectory(self, trajectory: np.ndarray) -> np.ndarray:
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

        # Convert to risk score
        if collision_prob > 0:
            risk = 1.0 + np.log(1 + collision_prob * 10)
        else:
            risk = 0.0

        # Add TTC penalty
        if min_ttc < 2.0:
            risk += (2.0 - min_ttc) * 0.5

        safety = SimpleSafetyScore(
            ttc=min_ttc if min_ttc != float("inf") else -1.0,
            min_distance=min_distance if min_distance != float("inf") else -1.0,
            collision_probability=min(1.0, collision_prob),
            is_safe=collision_prob < 0.1 and min_ttc > 2.0,
        )

        return risk, safety

    def _compute_comfort_penalty(self, candidate: EgoCandidate) -> float:
        """Compute comfort penalty based on acceleration and jerk."""
        traj = candidate.trajectory
        velocities = candidate.velocities

        # Longitudinal acceleration
        if len(velocities) > 1:
            accels = np.diff(velocities) / self.dt
            max_accel = np.max(np.abs(accels))
            accel_penalty = max(0, max_accel - 2.0)  # Penalty above 2 m/s^2
        else:
            accel_penalty = 0.0

        # Jerk (change in acceleration)
        if len(velocities) > 2:
            jerks = np.diff(accels) / self.dt
            max_jerk = np.max(np.abs(jerks))
            jerk_penalty = max(0, max_jerk - 1.0)  # Penalty above 1 m/s^3
        else:
            jerk_penalty = 0.0

        # Lateral acceleration (curvature * velocity^2)
        lat_accel_penalty = 0.0
        for t in range(1, len(traj) - 1):
            v1 = traj[t, :2] - traj[t - 1, :2]
            v2 = traj[t + 1, :2] - traj[t, :2]
            if np.linalg.norm(v1) > 0.01 and np.linalg.norm(v2) > 0.01:
                # Approximate curvature
                cross = v1[0] * v2[1] - v1[1] * v2[0]
                curvature = (
                    2
                    * cross
                    / (
                        np.linalg.norm(v1)
                        * np.linalg.norm(v2)
                        * np.linalg.norm(v1 + v2)
                        + 1e-6
                    )
                )
                speed = velocities[t] if t < len(velocities) else 0
                lat_accel = abs(curvature) * speed * speed
                if lat_accel > 3.0:  # Penalty above 3 m/s^2
                    lat_accel_penalty += (lat_accel - 3.0) * 0.1

        return accel_penalty + jerk_penalty + lat_accel_penalty

    def _compute_progress(
        self,
        candidate: EgoCandidate,
        goal: np.ndarray,
    ) -> float:
        """Compute progress toward goal (higher = better)."""
        start_pos = candidate.trajectory[0, :2]
        end_pos = candidate.trajectory[-1, :2]

        start_dist = np.linalg.norm(start_pos - goal)
        end_dist = np.linalg.norm(end_pos - goal)

        # Progress = reduction in distance to goal
        progress = start_dist - end_dist
        return max(0, progress)  # Only positive progress counts

    def _compute_lane_deviation(
        self,
        candidate: EgoCandidate,
        lane: LaneInfo,
    ) -> float:
        """Compute average deviation from lane center."""
        traj = candidate.trajectory
        centerline = lane.centerline

        total_deviation = 0.0
        for t in range(len(traj)):
            pos = traj[t, :2]
            dists = np.linalg.norm(centerline - pos, axis=1)
            min_dist = np.min(dists)
            total_deviation += min_dist

        return total_deviation / len(traj)

    def _compute_hysteresis(
        self,
        candidate: EgoCandidate,
        previous: EgoCandidate,
    ) -> float:
        """Compute deviation from previous selected trajectory."""
        # Compare first few timesteps (more important for consistency)
        compare_steps = min(20, len(candidate.trajectory))

        diff = (
            candidate.trajectory[:compare_steps, :2]
            - previous.trajectory[:compare_steps, :2]
        )

        return np.mean(np.linalg.norm(diff, axis=1))


@dataclass
class PlanningResult:
    """Result of one planning iteration."""

    selected_candidate: EgoCandidate
    selected_idx: int
    all_scores: np.ndarray
    safety_scores: List[SimpleSafetyScore]
    timing_ms: Dict[str, float]
    iteration: int


class RECTORPlanner:
    """
    Main RECTOR planning loop.

    Integrates all components:
    1. Candidate Generation
    2. Reactor Selection
    3. M2I Batched Prediction
    4. Scoring
    5. Selection with Hysteresis
    """

    def __init__(
        self,
        config: PlanningConfig = None,
        candidate_generator: CandidateGenerator = None,
        reactor_selector: ReactorSelector = None,
        scorer: CandidateScorer = None,
        adapter: "BatchedM2IAdapter" = None,
    ):
        """
        Initialize planner.

        Args:
            config: Planning configuration
            candidate_generator: Candidate trajectory generator
            reactor_selector: Reactor selection logic
            scorer: Candidate scoring logic
            adapter: Batched M2I adapter (optional - for testing without M2I)
        """
        self.config = config or PlanningConfig()
        self.generator = candidate_generator or CandidateGenerator(
            m_candidates=self.config.num_candidates,
            horizon_steps=self.config.candidate_horizon,
            dt=self.config.candidate_dt,
        )
        self.selector = reactor_selector or ReactorSelector(
            k_reactors=self.config.max_reactors,
        )
        self.scorer = scorer or CandidateScorer()
        self.adapter = adapter

        # State
        self.iteration = 0
        self.previous_result: Optional[PlanningResult] = None

    def plan_tick(
        self,
        ego_state: AgentState,
        agent_states: List[AgentState],
        current_lane: Optional[LaneInfo] = None,
        left_lane: Optional[LaneInfo] = None,
        right_lane: Optional[LaneInfo] = None,
        goal: Optional[np.ndarray] = None,
    ) -> PlanningResult:
        """
        Execute one planning tick.

        Args:
            ego_state: Current ego vehicle state
            agent_states: All other agent states
            current_lane: Current lane info
            left_lane: Left lane (if available)
            right_lane: Right lane (if available)
            goal: Goal position

        Returns:
            PlanningResult containing selected trajectory and debug info
        """
        timing = {}

        # Step 1: Generate candidates
        t0 = time.perf_counter()
        candidates = self.generator.generate(
            ego_state=ego_state,
            current_lane=current_lane,
            left_lane=left_lane,
            right_lane=right_lane,
            goal=goal,
        )
        timing["generate"] = (time.perf_counter() - t0) * 1000

        # Step 2: Select reactors
        t0 = time.perf_counter()
        reactor_ids = self.selector.select(
            ego_candidates=candidates,
            agent_states=agent_states,
            ego_state=ego_state,
        )
        timing["select_reactors"] = (time.perf_counter() - t0) * 1000

        # Step 3: Get predictions (from M2I or dummy)
        t0 = time.perf_counter()
        if self.adapter is not None and self.adapter.is_loaded:
            predictions = self._get_m2i_predictions(
                candidates, reactor_ids, agent_states
            )
        else:
            predictions = self._get_dummy_predictions(
                candidates, reactor_ids, agent_states
            )
        timing["predict"] = (time.perf_counter() - t0) * 1000

        # Step 4: Score candidates
        t0 = time.perf_counter()
        previous = (
            self.previous_result.selected_candidate if self.previous_result else None
        )
        scores, safety_scores = self.scorer.score_all(
            candidates=candidates,
            predictions=predictions,
            current_lane=current_lane,
            goal=goal,
            previous_selected=previous,
        )
        timing["score"] = (time.perf_counter() - t0) * 1000

        # Step 5: Select best candidate
        t0 = time.perf_counter()
        best_idx = self._select_with_hysteresis(scores, candidates, safety_scores)
        timing["select"] = (time.perf_counter() - t0) * 1000

        # Build result
        result = PlanningResult(
            selected_candidate=candidates.candidates[best_idx],
            selected_idx=best_idx,
            all_scores=scores,
            safety_scores=safety_scores,
            timing_ms=timing,
            iteration=self.iteration,
        )

        self.previous_result = result
        self.iteration += 1

        return result

    def _get_m2i_predictions(
        self,
        candidates: EgoCandidateBatch,
        reactor_ids: List[int],
        agent_states: List[AgentState],
    ) -> Dict[int, List["SinglePrediction"]]:
        """
        Get predictions from M2I adapter using true batched inference.

        This method uses the BatchedM2IAdapter to run conditional predictions
        for all (candidate, reactor) pairs in a single batched forward pass.
        """
        from data_contracts import SinglePrediction

        # Check if adapter has necessary data
        if self.adapter._current_scene_cache is None:
            print("Warning: No scene cache available, using dummy predictions")
            return self._get_dummy_predictions(candidates, reactor_ids, agent_states)

        # Build reactor packs if not already cached
        scene_cache = self.adapter._current_scene_cache
        reactor_packs = {}
        for reactor_id in reactor_ids:
            if reactor_id in self.adapter._current_reactor_packs:
                reactor_packs[reactor_id] = self.adapter._current_reactor_packs[
                    reactor_id
                ]
            else:
                # Build pack for this reactor
                new_packs = self.adapter.build_reactor_packs(scene_cache, [reactor_id])
                if reactor_id in new_packs:
                    reactor_packs[reactor_id] = new_packs[reactor_id]

        if len(reactor_packs) == 0:
            return self._get_dummy_predictions(candidates, reactor_ids, agent_states)

        try:
            # Run TRUE batched prediction
            result = self.adapter.predict_batched(
                reactor_packs=reactor_packs,
                ego_candidates=candidates,
                scene_cache=scene_cache,
            )

            # Convert PredictionResult to expected format
            # result has shape [M, K, N_modes, H, 2]
            predictions = {}

            for k, reactor_id in enumerate(result.reactor_ids):
                pred_list = []

                for m in range(len(result.ego_candidate_ids)):
                    # Extract trajectories and scores for this (m, k) pair
                    trajs = result.trajectories[m, k]  # [N_modes, H, 2]
                    scores = result.scores[m, k]  # [N_modes]

                    pred_list.append(
                        SinglePrediction(
                            ego_candidate_id=result.ego_candidate_ids[m],
                            reactor_id=reactor_id,
                            trajectories=trajs,
                            scores=scores,
                        )
                    )

                predictions[reactor_id] = pred_list

            return predictions

        except Exception as e:
            print(f"M2I prediction error: {e}, falling back to dummy predictions")
            return self._get_dummy_predictions(candidates, reactor_ids, agent_states)

    def _get_dummy_predictions(
        self,
        candidates: EgoCandidateBatch,
        reactor_ids: List[int],
        agent_states: List[AgentState],
    ) -> Dict[int, List["SinglePrediction"]]:
        """Generate dummy predictions for testing."""
        from data_contracts import SinglePrediction

        predictions = {}

        # Build agent state lookup
        agent_by_id = {a.agent_id: a for a in agent_states}

        for reactor_id in reactor_ids:
            if reactor_id not in agent_by_id:
                continue

            agent = agent_by_id[reactor_id]
            pred_list = []

            for candidate in candidates.candidates:
                # Generate simple constant-velocity prediction
                H = len(candidate.trajectory)
                trajs = np.zeros((6, H, 2))  # 6 modes
                probs = np.array([0.4, 0.2, 0.15, 0.1, 0.1, 0.05])

                for mode in range(6):
                    # Add some variation per mode
                    speed_factor = 1.0 + (mode - 2) * 0.1
                    for t in range(H):
                        t_sec = t * 0.1
                        trajs[mode, t, 0] = (
                            agent.x + agent.velocity_x * t_sec * speed_factor
                        )
                        trajs[mode, t, 1] = (
                            agent.y + agent.velocity_y * t_sec * speed_factor
                        )

                pred_list.append(
                    SinglePrediction(
                        ego_candidate_id=candidate.candidate_id,
                        reactor_id=reactor_id,
                        trajectories=trajs,
                        scores=probs,
                    )
                )

            predictions[reactor_id] = pred_list

        return predictions

    def _select_with_hysteresis(
        self,
        scores: np.ndarray,
        candidates: EgoCandidateBatch,
        safety_scores: List[SimpleSafetyScore],
    ) -> int:
        """
        Select best candidate with hysteresis preference.

        Prefers previous selection if scores are similar.
        """
        # First, filter out unsafe candidates
        valid_mask = np.array([s.is_safe for s in safety_scores])

        if not np.any(valid_mask):
            # All candidates unsafe - pick safest one
            # Sort by collision probability
            collision_probs = [s.collision_probability for s in safety_scores]
            return int(np.argmin(collision_probs))

        # Among safe candidates, pick lowest score
        valid_scores = np.where(valid_mask, scores, np.inf)
        best_idx = int(np.argmin(valid_scores))

        # Hysteresis check: if previous selection is still valid and close in score
        if self.previous_result is not None:
            prev_idx = self.previous_result.selected_idx
            if prev_idx < len(valid_scores) and valid_mask[prev_idx]:
                # Check if score difference is small
                score_diff = valid_scores[best_idx] - valid_scores[prev_idx]
                if score_diff > -1.0:  # Previous is within 1.0 of best
                    best_idx = prev_idx  # Stick with previous

        return best_idx


def create_planner(config: PlanningConfig = None) -> RECTORPlanner:
    """Create a complete RECTOR planner with all components."""
    config = config or PlanningConfig()

    return RECTORPlanner(
        config=config,
        candidate_generator=CandidateGenerator(
            m_candidates=config.num_candidates,
            horizon_steps=config.candidate_horizon,
            dt=config.candidate_dt,
        ),
        reactor_selector=ReactorSelector(k_reactors=config.max_reactors),
        scorer=CandidateScorer(),
        adapter=None,  # M2I adapter loaded separately
    )


if __name__ == "__main__":
    # Quick sanity check
    print("RECTOR Planning Loop - Sanity Check")
    print("=" * 50)

    # Create planner
    planner = create_planner()

    # Create ego state
    ego = AgentState(
        x=0.0,
        y=0.0,
        heading=0.0,
        velocity_x=10.0,
        velocity_y=0.0,
        agent_id=-1,
    )

    # Create some agents
    agents = [
        AgentState(
            x=30.0, y=0.5, heading=0.0, velocity_x=8.0, velocity_y=0.0, agent_id=1
        ),
        AgentState(
            x=50.0, y=-0.5, heading=0.0, velocity_x=12.0, velocity_y=0.0, agent_id=2
        ),
        AgentState(
            x=20.0, y=3.7, heading=0.0, velocity_x=10.0, velocity_y=0.0, agent_id=3
        ),
    ]

    # Run planning tick
    result = planner.plan_tick(
        ego_state=ego,
        agent_states=agents,
        goal=np.array([200.0, 0.0]),
    )

    print(f"Selected candidate: {result.selected_idx}")
    print(f"Strategy: {result.selected_candidate.generation_method}")
    print(
        f"Scores: min={result.all_scores.min():.2f}, "
        f"max={result.all_scores.max():.2f}"
    )
    print(f"Timing: {result.timing_ms}")
    print(
        f"Safe candidates: {sum(s.is_safe for s in result.safety_scores)}/{len(result.safety_scores)}"
    )
    print("\nSanity check PASSED")
