"""
Metric Collector: gathers per-step and per-scenario metrics.

Wraps the built-in Waymax metrics (overlap, offroad, wrong-way, …)
and adds custom motion-planning metrics (jerk, min clearance, TTC)
computed from the simulated trajectory.

Usage::

    mc = MetricCollector()
    for step in range(num_steps):
        obs = extract_observation(state)
        action = ...
        state = env.step(state, action)
        mc.step(state)
    summary = mc.finalise()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import jax.numpy as jnp

from waymax import config as wconfig, datatypes, metrics as wmetrics


# Built-in metrics that don't require sdc_paths
# (sdc_wrongway, sdc_off_route, sdc_progression need sdc_paths which
#  our custom scenario_loader doesn't populate)
DEFAULT_WAYMAX_METRICS = (
    "log_divergence",
    "overlap",
    "offroad",
    "kinematic_infeasibility",
)

# Metrics requiring sdc_paths (use only if loader supports them)
SDC_PATH_METRICS = (
    "sdc_wrongway",
    "sdc_progression",
    "sdc_off_route",
)


@dataclass
class MetricCollectorConfig:
    """Which metrics to collect and how."""

    waymax_metrics: Sequence[str] = DEFAULT_WAYMAX_METRICS
    compute_jerk: bool = True
    compute_ttc: bool = True
    compute_min_clearance: bool = True
    dt: float = 0.1  # Simulation timestep (seconds)


class MetricCollector:
    """Accumulates metrics over a rollout."""

    def __init__(self, cfg: Optional[MetricCollectorConfig] = None):
        self.cfg = cfg or MetricCollectorConfig()
        self._metrics_cfg = wconfig.MetricsConfig(
            metrics_to_run=tuple(self.cfg.waymax_metrics),
        )
        # Per-step buffers: metric_name → list of float
        self._step_metrics: Dict[str, List[float]] = {}
        # SDC trajectory buffer for derived metrics
        self._sdc_positions: List[np.ndarray] = []  # each [2] (x, y)
        self._sdc_velocities: List[np.ndarray] = []  # each [2] (vx, vy)
        self._sdc_idx: Optional[int] = None
        self._step_count = 0

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def step(self, state) -> Dict[str, float]:
        """
        Record one simulation step.

        Parameters
        ----------
        state : PlanningAgentSimulatorState or SimulatorState
            State *after* ``env.step()`` has been called.

        Returns
        -------
        dict
            Metric values for this single step.
        """
        t = int(state.timestep)

        # SDC index (cache)
        if self._sdc_idx is None:
            self._sdc_idx = int(jnp.argmax(state.object_metadata.is_sdc))
        sdc = self._sdc_idx

        # -- Waymax built-in metrics -----------------------------------------
        step_result: Dict[str, float] = {}

        wm_results = wmetrics.run_metrics(state, self._metrics_cfg)
        for name, mr in wm_results.items():
            # MetricResult.value has shape (..., num_objects)
            val = float(mr.value[sdc])
            valid = bool(mr.valid[sdc])
            step_result[name] = val if valid else float("nan")
            self._append(name, step_result[name])

        # -- SDC state for derived metrics ------------------------------------
        sdc_x = float(state.sim_trajectory.x[sdc, t])
        sdc_y = float(state.sim_trajectory.y[sdc, t])
        sdc_vx = float(state.sim_trajectory.vel_x[sdc, t])
        sdc_vy = float(state.sim_trajectory.vel_y[sdc, t])
        self._sdc_positions.append(np.array([sdc_x, sdc_y]))
        self._sdc_velocities.append(np.array([sdc_vx, sdc_vy]))

        # -- Min clearance (closest non-SDC object) --------------------------
        if self.cfg.compute_min_clearance:
            all_x = np.asarray(state.sim_trajectory.x[:, t])
            all_y = np.asarray(state.sim_trajectory.y[:, t])
            all_valid = np.asarray(state.sim_trajectory.valid[:, t])
            is_sdc = np.asarray(state.object_metadata.is_sdc)
            other_mask = all_valid & ~is_sdc
            if other_mask.any():
                dists = np.sqrt(
                    (all_x[other_mask] - sdc_x) ** 2 + (all_y[other_mask] - sdc_y) ** 2
                )
                min_cl = float(dists.min())
            else:
                min_cl = float("inf")
            step_result["min_clearance"] = min_cl
            self._append("min_clearance", min_cl)

        # -- TTC (time-to-collision estimate) ---------------------------------
        if self.cfg.compute_ttc:
            ttc = self._compute_ttc(state, sdc, t)
            step_result["ttc"] = ttc
            self._append("ttc", ttc)

        self._step_count += 1
        return step_result

    def finalise(self) -> Dict[str, float]:
        """
        Compute aggregated metrics over the full rollout.

        Returns
        -------
        dict
            Aggregated metrics (mean / max / min as appropriate).
        """
        summary: Dict[str, float] = {}
        summary["num_steps"] = float(self._step_count)

        # Waymax metric aggregation
        for name in self.cfg.waymax_metrics:
            vals = np.array(self._step_metrics.get(name, []))
            vals = vals[~np.isnan(vals)]
            if len(vals) == 0:
                summary[f"{name}/mean"] = float("nan")
                continue
            if name == "overlap":
                # Fraction of steps with overlap
                summary[f"{name}/rate"] = float((vals > 0).mean())
                summary[f"{name}/max"] = float(vals.max())
            elif name == "log_divergence":
                summary[f"{name}/mean"] = float(vals.mean())
                summary[f"{name}/final"] = (
                    float(vals[-1]) if len(vals) else float("nan")
                )
                summary[f"{name}/max"] = float(vals.max())
            elif name in ("sdc_progression",):
                summary[f"{name}/total"] = float(vals.sum())
            else:
                summary[f"{name}/rate"] = float((vals > 0).mean())

        # Jerk
        if self.cfg.compute_jerk and len(self._sdc_velocities) >= 3:
            vels = np.stack(self._sdc_velocities)  # [T, 2]
            accel = np.diff(vels, axis=0) / self.cfg.dt  # [T-1, 2]
            jerk = np.diff(accel, axis=0) / self.cfg.dt  # [T-2, 2]
            jerk_mag = np.linalg.norm(jerk, axis=1)
            summary["jerk/mean"] = float(jerk_mag.mean())
            summary["jerk/max"] = float(jerk_mag.max())

        # Min clearance
        if self.cfg.compute_min_clearance:
            mc_vals = np.array(self._step_metrics.get("min_clearance", []))
            mc_vals = mc_vals[np.isfinite(mc_vals)]
            if len(mc_vals):
                summary["min_clearance/min"] = float(mc_vals.min())
                summary["min_clearance/mean"] = float(mc_vals.mean())

        # TTC
        if self.cfg.compute_ttc:
            ttc_vals = np.array(self._step_metrics.get("ttc", []))
            ttc_vals = ttc_vals[np.isfinite(ttc_vals)]
            if len(ttc_vals):
                summary["ttc/min"] = float(ttc_vals.min())
                summary["ttc/mean"] = float(ttc_vals.mean())

        return summary

    def reset(self) -> None:
        """Clear all accumulated data for a new rollout."""
        self._step_metrics.clear()
        self._sdc_positions.clear()
        self._sdc_velocities.clear()
        self._sdc_idx = None
        self._step_count = 0

    # -----------------------------------------------------------------
    # Internal
    # -----------------------------------------------------------------

    def _append(self, name: str, value: float) -> None:
        if name not in self._step_metrics:
            self._step_metrics[name] = []
        self._step_metrics[name].append(value)

    @staticmethod
    def _compute_ttc(state, sdc_idx: int, t: int) -> float:
        """
        Estimate TTC using constant-velocity assumption.

        Returns time in seconds to first collision, or inf if none.
        """
        sdc_x = float(state.sim_trajectory.x[sdc_idx, t])
        sdc_y = float(state.sim_trajectory.y[sdc_idx, t])
        sdc_vx = float(state.sim_trajectory.vel_x[sdc_idx, t])
        sdc_vy = float(state.sim_trajectory.vel_y[sdc_idx, t])
        sdc_len = float(state.sim_trajectory.length[sdc_idx, t])
        sdc_wid = float(state.sim_trajectory.width[sdc_idx, t])
        sdc_radius = np.sqrt(sdc_len**2 + sdc_wid**2) / 2.0

        all_x = np.asarray(state.sim_trajectory.x[:, t])
        all_y = np.asarray(state.sim_trajectory.y[:, t])
        all_vx = np.asarray(state.sim_trajectory.vel_x[:, t])
        all_vy = np.asarray(state.sim_trajectory.vel_y[:, t])
        all_len = np.asarray(state.sim_trajectory.length[:, t])
        all_wid = np.asarray(state.sim_trajectory.width[:, t])
        all_valid = np.asarray(state.sim_trajectory.valid[:, t])
        is_sdc = np.asarray(state.object_metadata.is_sdc)
        other = all_valid & ~is_sdc

        if not other.any():
            return float("inf")

        # Relative position and velocity
        rel_x = all_x[other] - sdc_x
        rel_y = all_y[other] - sdc_y
        rel_vx = all_vx[other] - sdc_vx
        rel_vy = all_vy[other] - sdc_vy
        other_radius = np.sqrt(all_len[other] ** 2 + all_wid[other] ** 2) / 2.0
        combined_r = sdc_radius + other_radius

        # TTC via quadratic: |rel_pos + t * rel_vel|^2 = combined_r^2
        a = rel_vx**2 + rel_vy**2
        b = 2.0 * (rel_x * rel_vx + rel_y * rel_vy)
        c = rel_x**2 + rel_y**2 - combined_r**2

        # Only care about approaching objects (b < 0 roughly)
        disc = b**2 - 4 * a * c
        min_ttc = float("inf")

        valid_disc = disc >= 0
        valid_a = a > 1e-8
        valid = valid_disc & valid_a

        if valid.any():
            sqrt_disc = np.sqrt(disc[valid])
            t1 = (-b[valid] - sqrt_disc) / (2 * a[valid])
            t2 = (-b[valid] + sqrt_disc) / (2 * a[valid])
            # Minimum positive time
            t_pos = np.where(t1 > 0, t1, np.where(t2 > 0, t2, np.inf))
            min_ttc = float(t_pos.min())

        return min_ttc
