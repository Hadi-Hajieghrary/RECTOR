"""
ScenarioContext extension for trajectory injection.

CRITICAL: Enables trajectory-conditioned rule evaluation by allowing
ScenarioContext to carry candidate ego trajectories and predicted other-agent futures.

This module provides:
1. TrajectoryInjection dataclass for holding injected trajectory data
2. ScenarioContextExtension class for creating modified contexts
3. EgoStateFactory for creating EgoState from trajectory arrays
4. Helper functions for trajectory manipulation
"""

import numpy as np
from dataclasses import dataclass, field, replace
from typing import Optional, Dict, Any, Tuple, List
import copy

from .context import ScenarioContext, EgoState, Agent
from ..utils.constants import DEFAULT_DT_S


@dataclass
class TrajectoryInjection:
    """
    Container for injected trajectory data.

    Attributes:
        ego_trajectory: [H, 2] or [H, 4] array of (x, y) or (x, y, yaw, speed)
        other_futures: [N_agents, H, 2] predicted trajectories for other agents
        source: Origin of trajectory ("ground_truth", "predicted", "candidate", "perturbed")
        future_start_idx: Timestep where prediction begins (default: 11, matching Waymo)
    """

    ego_trajectory: Optional[np.ndarray] = None  # [H, 2] or [H, 4]
    other_futures: Optional[np.ndarray] = None  # [N_agents, H, 2]
    source: str = "candidate"
    future_start_idx: int = 11  # Timestep where prediction begins


class ScenarioContextExtension:
    """
    Utility class for creating trajectory-injected ScenarioContexts.

    This class does NOT modify ScenarioContext in place. Instead, it creates
    new contexts with injected trajectory data, preserving immutability.

    Usage:
        ext = ScenarioContextExtension(original_ctx)
        new_ctx = ext.with_ego_trajectory(candidate_traj)
        new_ctx = ext.with_other_futures(predicted_futures)
    """

    def __init__(self, ctx: ScenarioContext):
        """
        Initialize with a base ScenarioContext.

        Args:
            ctx: Original ScenarioContext to extend
        """
        self._base_ctx = ctx

    def with_ego_trajectory(
        self,
        trajectory: np.ndarray,
        source: str = "candidate",
        future_start_idx: Optional[int] = None,
    ) -> ScenarioContext:
        """
        Create new context with injected ego trajectory.

        The trajectory replaces the ego's future positions (from future_start_idx onwards).
        History (before future_start_idx) is preserved from the original context.

        Args:
            trajectory: [H, 2] or [H, 4] array.
                       [H, 2]: (x, y) - heading/speed computed from positions
                       [H, 4]: (x, y, yaw, speed) - all states provided
            source: Origin label ("candidate", "predicted", "perturbed", etc.)
            future_start_idx: Timestep where injection begins. If None, uses
                            half of the original trajectory length.

        Returns:
            New ScenarioContext with injected trajectory
        """
        if future_start_idx is None:
            future_start_idx = len(self._base_ctx.ego) // 2

        # Create new EgoState with injected trajectory
        new_ego = EgoStateFactory.inject_trajectory(
            original_ego=self._base_ctx.ego,
            trajectory=trajectory,
            future_start_idx=future_start_idx,
            dt=self._base_ctx.dt,
        )

        # Create new context with the modified ego
        # We use a simple approach: copy and replace
        new_ctx = _shallow_copy_context(self._base_ctx)
        new_ctx.ego = new_ego

        # Store injection metadata on explicit ScenarioContext field
        new_ctx.trajectory_injection = TrajectoryInjection(
            ego_trajectory=trajectory,
            other_futures=None,
            source=source,
            future_start_idx=future_start_idx,
        )

        return new_ctx

    def with_other_futures(
        self,
        futures: np.ndarray,
        future_start_idx: Optional[int] = None,
    ) -> ScenarioContext:
        """
        Create new context with injected other-agent trajectories.

        Replaces the future positions of other agents with predicted trajectories.

        Args:
            futures: [N_agents, H, 2] array of (x, y) positions
            future_start_idx: Timestep where injection begins

        Returns:
            New ScenarioContext with injected other-agent futures
        """
        if future_start_idx is None:
            future_start_idx = len(self._base_ctx.ego) // 2

        # Create new agents list with injected futures
        new_agents = []
        for i, agent in enumerate(self._base_ctx.agents):
            if i < len(futures):
                new_agent = _inject_agent_trajectory(
                    agent=agent,
                    trajectory=futures[i],
                    future_start_idx=future_start_idx,
                )
                new_agents.append(new_agent)
            else:
                # No prediction for this agent, keep original
                new_agents.append(agent)

        # Create new context
        new_ctx = _shallow_copy_context(self._base_ctx)
        new_ctx.agents = new_agents

        # Update injection metadata
        if new_ctx.trajectory_injection is not None:
            new_ctx.trajectory_injection.other_futures = futures
        else:
            new_ctx.trajectory_injection = TrajectoryInjection(
                ego_trajectory=None,
                other_futures=futures,
                future_start_idx=future_start_idx,
            )

        return new_ctx

    def with_both(
        self,
        ego_trajectory: np.ndarray,
        other_futures: np.ndarray,
        source: str = "candidate",
        future_start_idx: Optional[int] = None,
    ) -> ScenarioContext:
        """
        Create new context with both ego and other-agent trajectories injected.

        Args:
            ego_trajectory: [H, 2] or [H, 4] ego trajectory
            other_futures: [N_agents, H, 2] other agent trajectories
            source: Origin label
            future_start_idx: Timestep where injection begins

        Returns:
            New ScenarioContext with all trajectories injected
        """
        # Chain the operations
        ctx = self.with_ego_trajectory(ego_trajectory, source, future_start_idx)
        ext = ScenarioContextExtension(ctx)
        return ext.with_other_futures(other_futures, future_start_idx)


class EgoStateFactory:
    """Factory methods for creating EgoState from trajectories."""

    @staticmethod
    def from_trajectory(
        trajectory: np.ndarray,
        dt: float = DEFAULT_DT_S,
        length: float = 4.5,
        width: float = 2.0,
    ) -> EgoState:
        """
        Create EgoState from a trajectory array.

        Args:
            trajectory: [T, 2] or [T, 4] array
                       [T, 2]: (x, y) - heading/speed computed
                       [T, 4]: (x, y, yaw, speed) - all provided
            dt: Time step in seconds
            length: Vehicle length in meters
            width: Vehicle width in meters

        Returns:
            EgoState object
        """
        trajectory = np.atleast_2d(trajectory)
        T = len(trajectory)

        if trajectory.shape[1] >= 4:
            # Full state provided
            x = trajectory[:, 0]
            y = trajectory[:, 1]
            yaw = trajectory[:, 2]
            speed = trajectory[:, 3]
        elif trajectory.shape[1] >= 2:
            # Only positions provided - compute kinematics
            x = trajectory[:, 0]
            y = trajectory[:, 1]
            yaw, speed = _compute_kinematics_from_positions(x, y, dt)
        else:
            raise ValueError(
                f"Trajectory must have at least 2 columns, got {trajectory.shape[1]}"
            )

        return EgoState(
            x=x,
            y=y,
            yaw=yaw,
            speed=speed,
            length=length,
            width=width,
            valid=np.ones(T, dtype=bool),
        )

    @staticmethod
    def inject_trajectory(
        original_ego: EgoState,
        trajectory: np.ndarray,
        future_start_idx: int,
        dt: float = DEFAULT_DT_S,
    ) -> EgoState:
        """
        Create new EgoState with trajectory injected from future_start_idx.

        Args:
            original_ego: Original EgoState (history preserved)
            trajectory: [H, 2] or [H, 4] future trajectory
            future_start_idx: Index where injection begins
            dt: Time step in seconds

        Returns:
            New EgoState with injected trajectory
        """
        trajectory = np.atleast_2d(trajectory)
        H = len(trajectory)

        # Total length = history + future
        total_len = future_start_idx + H

        # Copy history from original
        x = np.zeros(total_len)
        y = np.zeros(total_len)
        yaw = np.zeros(total_len)
        speed = np.zeros(total_len)
        valid = np.zeros(total_len, dtype=bool)

        # Copy history (up to future_start_idx)
        hist_len = min(future_start_idx, len(original_ego))
        x[:hist_len] = original_ego.x[:hist_len]
        y[:hist_len] = original_ego.y[:hist_len]
        yaw[:hist_len] = original_ego.yaw[:hist_len]
        speed[:hist_len] = original_ego.speed[:hist_len]
        valid[:hist_len] = original_ego.valid[:hist_len]

        # Inject future trajectory
        if trajectory.shape[1] >= 4:
            # Full state provided
            x[future_start_idx:] = trajectory[:, 0]
            y[future_start_idx:] = trajectory[:, 1]
            yaw[future_start_idx:] = trajectory[:, 2]
            speed[future_start_idx:] = trajectory[:, 3]
        else:
            # Only positions - compute kinematics
            x[future_start_idx:] = trajectory[:, 0]
            y[future_start_idx:] = trajectory[:, 1]

            # Use last history state for smooth transition
            if future_start_idx > 0 and hist_len > 0:
                prev_yaw = yaw[future_start_idx - 1]
            else:
                prev_yaw = 0.0

            future_yaw, future_speed = _compute_kinematics_from_positions(
                trajectory[:, 0], trajectory[:, 1], dt, prev_yaw
            )
            yaw[future_start_idx:] = future_yaw
            speed[future_start_idx:] = future_speed

        # Mark future as valid (non-NaN positions)
        valid[future_start_idx:] = ~(
            np.isnan(x[future_start_idx:]) | np.isnan(y[future_start_idx:])
        )

        return EgoState(
            x=x,
            y=y,
            yaw=yaw,
            speed=speed,
            length=original_ego.length,
            width=original_ego.width,
            valid=valid,
        )


def _compute_kinematics_from_positions(
    x: np.ndarray,
    y: np.ndarray,
    dt: float,
    prev_yaw: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute heading and speed from position sequence.

    Args:
        x: [T] x positions
        y: [T] y positions
        dt: Time step in seconds
        prev_yaw: Previous heading for smoothing first point

    Returns:
        Tuple of (yaw, speed) arrays, each [T]
    """
    T = len(x)

    if T < 2:
        return np.array([prev_yaw]), np.array([0.0])

    # Compute displacements
    dx = np.diff(x)
    dy = np.diff(y)

    # Compute speed from displacement
    dist = np.sqrt(dx**2 + dy**2)
    speed = np.zeros(T)
    speed[1:] = dist / dt
    speed[0] = speed[1] if T > 1 else 0.0

    # Compute heading from displacement
    raw_yaw = np.arctan2(dy, dx)

    # Smooth heading to avoid discontinuities
    yaw = _smooth_heading(raw_yaw, prev_yaw)

    return yaw, speed


def _smooth_heading(raw_yaw: np.ndarray, prev_yaw: float) -> np.ndarray:
    """
    Smooth heading to avoid discontinuities.

    Uses angle unwrapping to handle -pi/pi wraparound.

    Args:
        raw_yaw: [T-1] raw heading values from arctan2
        prev_yaw: Previous heading for first point

    Returns:
        [T] smoothed heading array
    """
    T = len(raw_yaw) + 1
    yaw = np.zeros(T)

    if T == 1:
        yaw[0] = prev_yaw
        return yaw

    # Unwrap angles starting from prev_yaw
    yaw[0] = prev_yaw
    for i in range(len(raw_yaw)):
        # Compute angle difference
        diff = raw_yaw[i] - yaw[i]
        # Wrap to [-pi, pi]
        diff = np.arctan2(np.sin(diff), np.cos(diff))
        yaw[i + 1] = yaw[i] + diff

    return yaw


def _shallow_copy_context(ctx: ScenarioContext) -> ScenarioContext:
    """
    Create a shallow copy of ScenarioContext.

    Uses dataclass replace for clean copying.
    """
    return ScenarioContext(
        scenario_id=ctx.scenario_id,
        ego=ctx.ego,  # Will be replaced
        agents=ctx.agents,  # Will be replaced
        map_context=ctx.map_context,
        signals=ctx.signals,
        dt=ctx.dt,
        window_start_ts=ctx.window_start_ts,
        window_size=ctx.window_size,
        dataset_kind=ctx.dataset_kind,
        trajectory_injection=ctx.trajectory_injection,
    )


def _inject_agent_trajectory(
    agent: Agent,
    trajectory: np.ndarray,
    future_start_idx: int,
) -> Agent:
    """
    Create new Agent with injected future trajectory.

    Args:
        agent: Original Agent
        trajectory: [H, 2] future (x, y) positions
        future_start_idx: Index where injection begins

    Returns:
        New Agent with injected trajectory
    """
    trajectory = np.atleast_2d(trajectory)
    H = len(trajectory)

    # Total length
    total_len = future_start_idx + H

    # Initialize arrays
    x = np.zeros(total_len)
    y = np.zeros(total_len)
    yaw = np.zeros(total_len)
    speed = np.zeros(total_len)
    valid = np.zeros(total_len, dtype=bool)

    # Copy history
    hist_len = min(future_start_idx, len(agent))
    x[:hist_len] = agent.x[:hist_len]
    y[:hist_len] = agent.y[:hist_len]
    yaw[:hist_len] = agent.yaw[:hist_len]
    speed[:hist_len] = agent.speed[:hist_len]
    valid[:hist_len] = agent.valid[:hist_len]

    # Inject future
    x[future_start_idx:] = trajectory[:, 0]
    y[future_start_idx:] = trajectory[:, 1]
    valid[future_start_idx:] = ~(
        np.isnan(trajectory[:, 0]) | np.isnan(trajectory[:, 1])
    )

    # Compute kinematics for future (simplified)
    if H > 1:
        dx = np.diff(trajectory[:, 0])
        dy = np.diff(trajectory[:, 1])
        future_yaw = np.arctan2(dy, dx)
        future_speed = np.sqrt(dx**2 + dy**2) / DEFAULT_DT_S

        yaw[future_start_idx:-1] = future_yaw
        yaw[-1] = future_yaw[-1] if len(future_yaw) > 0 else 0.0

        speed[future_start_idx + 1 :] = future_speed
        speed[future_start_idx] = future_speed[0] if len(future_speed) > 0 else 0.0

    return Agent(
        id=agent.id,
        type=agent.type,
        x=x,
        y=y,
        yaw=yaw,
        speed=speed,
        length=agent.length,
        width=agent.width,
        valid=valid,
    )


def inject_ego_trajectory(
    ctx: ScenarioContext,
    trajectory: np.ndarray,
    source: str = "candidate",
    future_start_idx: Optional[int] = None,
) -> ScenarioContext:
    """
    Convenience function to inject ego trajectory into context.

    Args:
        ctx: Original ScenarioContext
        trajectory: [H, 2] or [H, 4] trajectory array
        source: Origin label
        future_start_idx: Injection start index

    Returns:
        New ScenarioContext with injected trajectory
    """
    ext = ScenarioContextExtension(ctx)
    return ext.with_ego_trajectory(trajectory, source, future_start_idx)


def inject_other_futures(
    ctx: ScenarioContext,
    futures: np.ndarray,
    future_start_idx: Optional[int] = None,
) -> ScenarioContext:
    """
    Convenience function to inject other-agent futures into context.

    Args:
        ctx: Original ScenarioContext
        futures: [N_agents, H, 2] trajectories
        future_start_idx: Injection start index

    Returns:
        New ScenarioContext with injected futures
    """
    ext = ScenarioContextExtension(ctx)
    return ext.with_other_futures(futures, future_start_idx)


def get_trajectory_injection(ctx: ScenarioContext) -> Optional[TrajectoryInjection]:
    """
    Get trajectory injection metadata from context.

    Args:
        ctx: ScenarioContext

    Returns:
        TrajectoryInjection if context has injected trajectories, None otherwise
    """
    return ctx.trajectory_injection


def has_injected_trajectory(ctx: ScenarioContext) -> bool:
    """Check if context has an injected ego trajectory."""
    injection = get_trajectory_injection(ctx)
    return injection is not None and injection.ego_trajectory is not None


def has_injected_futures(ctx: ScenarioContext) -> bool:
    """Check if context has injected other-agent futures."""
    injection = get_trajectory_injection(ctx)
    return injection is not None and injection.other_futures is not None
