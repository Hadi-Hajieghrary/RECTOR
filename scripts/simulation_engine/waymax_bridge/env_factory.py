"""
Environment Factory: Creates Waymax simulation environments.

Bridges scenario data from our custom ``scenario_loader`` to a
``PlanningAgentEnvironment`` with configurable dynamics and reactive
agent policies (IDM or log-playback).

The factory returns an environment + the initial
``PlanningAgentSimulatorState`` ready for the main closed-loop.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import jax
import jax.numpy as jnp

from waymax import agents, config as wconfig, datatypes, dynamics, env


def make_env(
    sim_state: datatypes.SimulatorState,
    dynamics_model: str = "delta",
    agent_model: str = "idm",
    max_num_objects: Optional[int] = None,
    init_steps: int = 11,
    # IDM parameters (§3.5 of plan)
    idm_desired_vel: float = 30.0,
    idm_min_spacing: float = 2.0,
    idm_safe_time_headway: float = 2.0,
    idm_max_accel: float = 2.0,
    idm_max_decel: float = 4.0,
) -> Tuple[env.PlanningAgentEnvironment, "PlanningAgentSimulatorState"]:
    """
    Create a Waymax closed-loop environment from a pre-loaded SimulatorState.

    Parameters
    ----------
    sim_state : datatypes.SimulatorState
        Loaded via ``scenario_loader.load_scenarios``.
    dynamics_model : str
        ``"bicycle"`` (InvertibleBicycleModel) or ``"delta"`` (DeltaGlobal).
    agent_model : str
        ``"idm"`` for IDMRoutePolicy or ``"log_playback"`` for expert replay.
    max_num_objects : int or None
        Must match the padding used by scenario_loader.
        If None, auto-detected from ``sim_state``.
    init_steps : int
        Number of warm-up timesteps (default 11 = WOMD convention).
    idm_* : float
        IDM policy hyper-parameters (only used when agent_model="idm").

    Returns
    -------
    (environment, initial_state)
        Environment and the initial ``PlanningAgentSimulatorState`` after
        warm-up, ready for the simulation loop.
    """

    # ---- Auto-detect max_num_objects from state if not given ----------------
    if max_num_objects is None:
        max_num_objects = int(sim_state.sim_trajectory.x.shape[0])

    # ---- Dynamics -----------------------------------------------------------
    if dynamics_model == "bicycle":
        dyn = dynamics.InvertibleBicycleModel()
    elif dynamics_model == "delta":
        dyn = dynamics.DeltaGlobal()
    else:
        raise ValueError(f"Unknown dynamics_model: {dynamics_model}")

    # ---- Reactive agent policy for non-SDC agents ---------------------------
    # is_controlled_func: returns True for non-SDC agents
    def _non_sdc_controlled(state: datatypes.SimulatorState) -> jax.Array:
        """All valid, non-SDC objects are controlled by the sim agent."""
        return jnp.logical_and(
            ~state.object_metadata.is_sdc,
            state.object_metadata.is_valid,
        )

    # Sim agents must produce TrajectoryUpdate-format actions (dim=5) to
    # match the output of PlanningAgentDynamics.compute_update().as_action().
    # StateDynamics.inverse() produces exactly that format.
    sim_dyn = dynamics.StateDynamics()

    if agent_model == "idm":
        sim_actor = agents.IDMRoutePolicy(
            is_controlled_func=_non_sdc_controlled,
            desired_vel=idm_desired_vel,
            min_spacing=idm_min_spacing,
            safe_time_headway=idm_safe_time_headway,
            max_accel=idm_max_accel,
            max_decel=idm_max_decel,
        )
    elif agent_model == "log_playback":
        sim_actor = agents.create_expert_actor(
            dynamics_model=sim_dyn,
            is_controlled_func=_non_sdc_controlled,
        )
    else:
        raise ValueError(f"Unknown agent_model: {agent_model}")

    # ---- Environment config -------------------------------------------------
    env_config = wconfig.EnvironmentConfig(
        max_num_objects=max_num_objects,
        init_steps=init_steps,
        controlled_object=wconfig.ObjectType.SDC,
    )

    # ---- Initialise params and build environment ----------------------------
    rng = jax.random.PRNGKey(0)

    waymax_env = env.PlanningAgentEnvironment(
        dynamics_model=dyn,
        config=env_config,
        sim_agent_actors=(sim_actor,),
        sim_agent_params=(sim_actor.init(rng, sim_state),),
    )

    # ---- Reset to get initial state -----------------------------------------
    initial_state = waymax_env.reset(sim_state, rng=rng)

    return waymax_env, initial_state
