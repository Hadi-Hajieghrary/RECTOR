"""
Observation Extractor: Waymax SimulatorState → RECTOR observation tensors.

Converts the JAX-based Waymax simulation state into the NumPy/PyTorch
tensors that the RECTOR ``RuleAwareGeneratorV2`` expects:

* ``ego_history``   – [B, H, 4]            (x, y, heading, speed)
* ``agent_states``  – [B, A, H, 4]         (x, y, heading, speed)
* ``lane_centers``  – [B, L, P, 2]         (x, y)
* ``agent_mask``    – [B, A]               (valid agents)
* ``lane_mask``     – [B, L]               (valid lanes)

All coordinates are ego-centric (rotated so ego heading at t_ref is 0)
and normalized by ``TRAJECTORY_SCALE``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import jax.numpy as jnp


TRAJECTORY_SCALE: float = 50.0  # Spatial normalization divisor
HISTORY_LENGTH: int = 11  # 10 past + 1 current
MAX_AGENTS: int = 32  # Max non-ego agents
MAX_LANES: int = 64  # Max lane polylines
LANE_POINTS: int = 20  # Points per lane polyline


@dataclass
class ObservationExtractorConfig:
    """Config knobs for the observation extractor."""

    trajectory_scale: float = TRAJECTORY_SCALE
    history_length: int = HISTORY_LENGTH
    max_agents: int = MAX_AGENTS
    max_lanes: int = MAX_LANES
    lane_points: int = LANE_POINTS
    roadgraph_topk: int = 2000  # Top-K roadgraph points by distance to ego


def extract_observation(
    state,  # PlanningAgentSimulatorState or SimulatorState
    cfg: Optional[ObservationExtractorConfig] = None,
) -> Dict[str, np.ndarray]:
    """
    Extract RECTOR-compatible observation from a Waymax state.

    The reference timestep is ``state.timestep`` (i.e. the "current"
    step after warm-up).  History spans ``[t-H+1 … t]``.

    Parameters
    ----------
    state : PlanningAgentSimulatorState | SimulatorState
        Current Waymax state.
    cfg : ObservationExtractorConfig, optional

    Returns
    -------
    dict with keys:
        ego_history  : np.ndarray [1, H, 4]
        agent_states : np.ndarray [1, A, H, 4]
        lane_centers : np.ndarray [1, L, P, 2]
        agent_mask   : np.ndarray [1, A]
        lane_mask    : np.ndarray [1, L]
        ego_pos      : np.ndarray [2]       (unnormalized reference x, y)
        ego_yaw      : float                (unnormalized reference heading)
    """
    if cfg is None:
        cfg = ObservationExtractorConfig()

    H = cfg.history_length
    A = cfg.max_agents
    L = cfg.max_lanes
    P = cfg.lane_points
    scale = cfg.trajectory_scale

    # Current timestep index
    t = int(state.timestep)
    t_start = max(0, t - H + 1)

    # ---- Move JAX arrays to numpy -----------------------------------------
    traj = state.sim_trajectory
    meta = state.object_metadata

    all_x = np.asarray(traj.x)  # [N, T]
    all_y = np.asarray(traj.y)  # [N, T]
    all_yaw = np.asarray(traj.yaw)  # [N, T]
    all_vel_x = np.asarray(traj.vel_x)  # [N, T]
    all_vel_y = np.asarray(traj.vel_y)  # [N, T]
    all_valid = np.asarray(traj.valid)  # [N, T]

    is_sdc = np.asarray(meta.is_sdc)  # [N]
    is_valid_obj = np.asarray(meta.is_valid)  # [N]

    sdc_idx = int(np.argmax(is_sdc))

    # ---- Ego reference frame ------------------------------------------------
    ego_ref_x = float(all_x[sdc_idx, t])
    ego_ref_y = float(all_y[sdc_idx, t])
    ego_ref_yaw = float(all_yaw[sdc_idx, t])

    cos_r = np.cos(-ego_ref_yaw)
    sin_r = np.sin(-ego_ref_yaw)

    def _to_ego(x_arr: np.ndarray, y_arr: np.ndarray):
        """Translate + rotate to ego frame."""
        dx = x_arr - ego_ref_x
        dy = y_arr - ego_ref_y
        x_ego = dx * cos_r - dy * sin_r
        y_ego = dx * sin_r + dy * cos_r
        return x_ego, y_ego

    def _heading_to_ego(yaw_arr: np.ndarray):
        """Rotate heading into ego frame."""
        return yaw_arr - ego_ref_yaw

    # Speed = ||(vel_x, vel_y)||
    all_speed = np.sqrt(all_vel_x**2 + all_vel_y**2)

    # ---- Ego history -------------------------------------------------------
    # [H, 4] — padded left if t < H-1
    ego_hist = np.zeros((H, 4), dtype=np.float32)
    for h_i, t_i in enumerate(range(t_start, t + 1)):
        local_h = H - (t - t_start + 1) + h_i
        if t_i < 0 or t_i >= all_x.shape[1]:
            continue
        ex, ey = _to_ego(all_x[sdc_idx, t_i], all_y[sdc_idx, t_i])
        eh = _heading_to_ego(all_yaw[sdc_idx, t_i])
        es = all_speed[sdc_idx, t_i]
        ego_hist[local_h] = [ex / scale, ey / scale, eh, es]

    # ---- Other agents -------------------------------------------------------
    agent_states_arr = np.zeros((A, H, 4), dtype=np.float32)
    agent_mask_arr = np.zeros(A, dtype=np.float32)

    # Gather non-SDC, valid agents; sort by proximity to ego at t
    non_sdc_mask = np.logical_and(is_valid_obj, ~is_sdc)
    non_sdc_indices = np.where(non_sdc_mask)[0]

    if len(non_sdc_indices) > 0:
        # Distance to ego at reference time
        dists = np.sqrt(
            (all_x[non_sdc_indices, t] - ego_ref_x) ** 2
            + (all_y[non_sdc_indices, t] - ego_ref_y) ** 2
        )
        sorted_order = np.argsort(dists)
        top_agents = non_sdc_indices[sorted_order[:A]]

        for a_i, obj_idx in enumerate(top_agents):
            agent_mask_arr[a_i] = 1.0
            for h_i, t_i in enumerate(range(t_start, t + 1)):
                local_h = H - (t - t_start + 1) + h_i
                if t_i < 0 or t_i >= all_x.shape[1]:
                    continue
                if not all_valid[obj_idx, t_i]:
                    continue
                ax, ay = _to_ego(all_x[obj_idx, t_i], all_y[obj_idx, t_i])
                ah = _heading_to_ego(all_yaw[obj_idx, t_i])
                aspd = all_speed[obj_idx, t_i]
                agent_states_arr[a_i, local_h] = [ax / scale, ay / scale, ah, aspd]

    # ---- Lane centers -------------------------------------------------------
    lane_centers_arr = np.zeros((L, P, 2), dtype=np.float32)
    lane_mask_arr = np.zeros(L, dtype=np.float32)

    rg = state.roadgraph_points
    if rg is not None:
        rg_x = np.asarray(rg.x)  # [R]
        rg_y = np.asarray(rg.y)  # [R]
        rg_ids = np.asarray(rg.ids)  # [R]
        rg_valid = np.asarray(rg.valid)  # [R]
        rg_types = np.asarray(rg.types)  # [R]

        valid_mask = rg_valid.astype(bool)
        rg_x = rg_x[valid_mask]
        rg_y = rg_y[valid_mask]
        rg_ids = rg_ids[valid_mask]
        rg_types = rg_types[valid_mask]

        # Group by lane ID (type==1 are lanes)
        # Also include road lines (type==2) and road edges (type==3)
        # for richer context
        unique_ids = np.unique(rg_ids)

        # Sort lanes by distance to ego
        lane_dists = []
        lane_id_list = []
        for lid in unique_ids:
            pts_mask = rg_ids == lid
            lx = rg_x[pts_mask]
            ly = rg_y[pts_mask]
            if len(lx) < 2:
                continue
            # Mean distance to ego
            d = np.mean(np.sqrt((lx - ego_ref_x) ** 2 + (ly - ego_ref_y) ** 2))
            lane_dists.append(d)
            lane_id_list.append(lid)

        if lane_id_list:
            order = np.argsort(lane_dists)
            for l_i, idx in enumerate(order[:L]):
                lid = lane_id_list[idx]
                pts_mask = rg_ids == lid
                lx = rg_x[pts_mask]
                ly = rg_y[pts_mask]

                # Sample/pad to P points
                n_pts = len(lx)
                if n_pts >= P:
                    # Uniform subsample
                    indices = np.linspace(0, n_pts - 1, P, dtype=int)
                    lx = lx[indices]
                    ly = ly[indices]
                else:
                    # Pad by repeating last point
                    lx = np.concatenate([lx, np.full(P - n_pts, lx[-1])])
                    ly = np.concatenate([ly, np.full(P - n_pts, ly[-1])])

                # Transform to ego frame
                ex, ey = _to_ego(lx, ly)
                lane_centers_arr[l_i, :, 0] = ex / scale
                lane_centers_arr[l_i, :, 1] = ey / scale
                lane_mask_arr[l_i] = 1.0

    # ---- Package with batch dimension [1, ...] ------------------------------
    return {
        "ego_history": ego_hist[np.newaxis],  # [1, H, 4]
        "agent_states": agent_states_arr[np.newaxis],  # [1, A, H, 4]
        "lane_centers": lane_centers_arr[np.newaxis],  # [1, L, P, 2]
        "agent_mask": agent_mask_arr[np.newaxis],  # [1, A]
        "lane_mask": lane_mask_arr[np.newaxis],  # [1, L]
        "ego_pos": np.array([ego_ref_x, ego_ref_y], dtype=np.float32),
        "ego_yaw": ego_ref_yaw,
    }
