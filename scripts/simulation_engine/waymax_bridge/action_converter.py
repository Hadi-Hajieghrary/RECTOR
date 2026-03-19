"""
Action Converter: RECTOR trajectory → Waymax Action.

Converts a selected RECTOR trajectory (in ego-centric, normalized coords)
back to a Waymax ``Action`` that the ``PlanningAgentEnvironment`` can
execute for one simulation step.

The conversion path is:

1. Denormalize (× TRAJECTORY_SCALE)
2. Rotate back to global frame (undo ego rotation)
3. Build a ``DeltaGlobal``-style action: (dx, dy, dyaw) per step

The environment MUST use ``DeltaGlobal`` dynamics so that the (dx, dy,
dyaw) action is applied directly as a position/heading update.
Using ``InvertibleBicycleModel`` with these actions causes yaw-spiral
artifacts because the bicycle inverse-kinematics reinterprets the deltas.
"""

from __future__ import annotations

from typing import Optional

import jax.numpy as jnp
import numpy as np

from waymax import datatypes

from .observation_extractor import TRAJECTORY_SCALE


def trajectory_to_action(
    trajectory: np.ndarray,
    ego_pos: np.ndarray,
    ego_yaw: float,
    state,  # PlanningAgentSimulatorState
    steps: int = 1,
    scale: float = TRAJECTORY_SCALE,
) -> datatypes.Action:
    """
    Convert a RECTOR trajectory to a Waymax **single-agent** Action.

    ``PlanningAgentEnvironment`` expects the SDC action with shape
    ``(action_dim,)`` — it internally tiles it to all objects.  For
    ``DeltaGlobal`` dynamics the action_dim is **3**: ``(dx, dy, dyaw)``.

    Parameters
    ----------
    trajectory : np.ndarray [T, 4]
        Selected trajectory from RECTOR in ego-centric normalized coords.
        Columns: (x, y, vx, vy) — first two normalized by ``scale``.
    ego_pos : np.ndarray [2]
        Global (x, y) of the ego reference point used during extraction.
    ego_yaw : float
        Global heading of the ego at the reference time.
    state : PlanningAgentSimulatorState
        Current Waymax simulation state.
    steps : int
        Index (1-based) into the trajectory to extract the delta from.
        Default 1 means we produce an action for the *next* timestep.
    scale : float
        Spatial normalization constant.

    Returns
    -------
    datatypes.Action
        Single-agent action with ``data`` shape ``(action_dim,)`` and
        ``valid`` shape ``(1,)``.
    """
    t = int(state.timestep)

    # ---- Denormalize and rotate to global frame ----------------------------
    traj_ego_x = trajectory[:, 0] * scale  # [T]
    traj_ego_y = trajectory[:, 1] * scale  # [T]

    cos_r = np.cos(ego_yaw)
    sin_r = np.sin(ego_yaw)

    # Ego-frame → Global
    global_x = traj_ego_x * cos_r - traj_ego_y * sin_r + ego_pos[0]
    global_y = traj_ego_x * sin_r + traj_ego_y * cos_r + ego_pos[1]

    # Heading from velocity direction
    if trajectory.shape[1] >= 4:
        vx = trajectory[:, 2]  # Already in ego frame, NOT normalized by scale
        vy = trajectory[:, 3]
        # Rotate velocity to global
        global_vx = vx * cos_r - vy * sin_r
        global_vy = vx * sin_r + vy * cos_r
        global_yaw = np.arctan2(global_vy, global_vx)
    else:
        # Derive yaw from position deltas
        dx_arr = np.diff(global_x, prepend=global_x[0])
        dy_arr = np.diff(global_y, prepend=global_y[0])
        global_yaw = np.arctan2(dy_arr, dx_arr)

    # ---- Build single-agent action (position delta) -------------------------
    sdc_idx = int(jnp.argmax(state.object_metadata.is_sdc))
    cur_x = float(state.sim_trajectory.x[sdc_idx, t])
    cur_y = float(state.sim_trajectory.y[sdc_idx, t])
    cur_yaw = float(state.sim_trajectory.yaw[sdc_idx, t])

    # steps is 1-based (step_in_plan + 1), so subtract 1 for 0-based indexing
    # into the trajectory: step_in_plan=0 → traj[0] = first future position
    step_idx = min(max(steps - 1, 0), len(global_x) - 1)
    dx = global_x[step_idx] - cur_x
    dy = global_y[step_idx] - cur_y
    dyaw = _wrap_angle(global_yaw[step_idx] - cur_yaw)

    # Single-agent action: shape (action_dim,)
    action_data = np.array([dx, dy, dyaw], dtype=np.float32)
    action_valid = np.array([True])

    return datatypes.Action(
        data=jnp.array(action_data),
        valid=jnp.array(action_valid),
    )


def _wrap_angle(angle: float) -> float:
    """Wrap angle to [-π, π]."""
    return float(((angle + np.pi) % (2 * np.pi)) - np.pi)
