"""
Experiment configuration dataclasses.

Loaded via Hydra (hydra-core) for reproducible experiment runs.
All paths default to the environment variables set in docker-compose.yml.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional


def _env_or_legacy(primary: str, legacy: str, default: str) -> str:
    """Read a descriptive env var first, then fall back to the legacy name."""
    return os.environ.get(primary, os.environ.get(legacy, default))


@dataclass
class WaymaxBridgeConfig:
    """Configuration for the Waymax ↔ RECTOR bridge."""

    dt_replan: float = 2.0  # Re-planning interval (seconds)
    steps_per_replan: int = 20  # 2.0s × 10Hz
    horizon_seconds: float = 5.0  # Generator prediction horizon
    hz: int = 10  # Simulation frequency
    dynamics_model: str = (
        "delta"  # 'delta' (DeltaGlobal) – matches action_converter output
    )


@dataclass
class AgentConfig:
    """Configuration for non-ego agent behavior."""

    agent_model: str = "idm"  # 'idm' or 'log_playback'
    # IDM parameters (§3.5 of plan)
    idm_desired_vel: float = 30.0  # m/s
    idm_min_spacing: float = 2.0  # m
    idm_safe_time_headway: float = 2.0  # s
    idm_max_accel: float = 2.0  # m/s²
    idm_max_decel: float = 4.0  # m/s²


@dataclass
class SelectorConfig:
    """Configuration for trajectory selection methods."""

    methods: List[str] = field(
        default_factory=lambda: ["confidence", "weighted_sum", "rector_lex"]
    )
    # RECTOR lexicographic epsilon per tier (§6.4)
    rector_epsilon: List[float] = field(
        default_factory=lambda: [1e-3, 1e-3, 1e-3, 1e-3]
    )
    # WS weight tuning (§6.3)
    ws_tuning_scenarios: int = 100
    ws_safety_grid: List[float] = field(
        default_factory=lambda: [10.0, 50.0, 100.0, 500.0]
    )
    ws_legal_grid: List[float] = field(default_factory=lambda: [5.0, 20.0, 50.0])
    ws_road_grid: List[float] = field(default_factory=lambda: [1.0, 5.0])
    ws_comfort_grid: List[float] = field(default_factory=lambda: [0.1, 1.0])


@dataclass
class MiningConfig:
    """Configuration for scenario mining pipeline (§5)."""

    data_dir: str = field(
        default_factory=lambda: os.environ.get(
            "WAYMO_PROCESSED_ROOT",
            "/workspace/data/WOMD/datasets/waymo_open_dataset/motion_v_1_3_0/processed",
        )
    )
    max_scenarios: int = 12_800
    # Per-class targets (§5.3)
    target_stop_sign: int = 300
    target_red_light: int = 300
    target_near_miss: int = 400
    target_speed_limit: int = 200
    target_lane_keep: int = 200
    target_comfort: int = 300
    target_failure: int = 200
    output_dir: str = field(
        default_factory=lambda: _env_or_legacy(
            "SIMULATION_ENGINE_OUTPUT",
            "RECTOR_CLOSEDLOOP_OUTPUT",
            "/workspace/output/closedloop",
        )
        + "/mining"
    )


@dataclass
class MetricConfig:
    """Configuration for metrics and statistical analysis (§7, §9)."""

    bootstrap_resamples: int = 10_000
    alpha: float = 0.05
    # Conservatism guards (§7.3)
    min_route_completion: float = 0.92
    min_speed_fraction: float = 0.85
    max_timeout_rate: float = 0.05
    # Near-miss threshold
    ttc_threshold: float = 1.5  # seconds


@dataclass
class GeneratorConfig:
    """Configuration for trajectory generation (§3.3)."""

    track: str = "A"  # 'A' (RECTOR learned) or 'B' (M2I + Rule Selection)
    num_modes: int = 6
    # Track A
    checkpoint: Optional[str] = None
    # Track B
    m2i_checkpoint: Optional[str] = None


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""

    # Sub-configs
    waymax: WaymaxBridgeConfig = field(default_factory=WaymaxBridgeConfig)
    agents: AgentConfig = field(default_factory=AgentConfig)
    selectors: SelectorConfig = field(default_factory=SelectorConfig)
    mining: MiningConfig = field(default_factory=MiningConfig)
    metrics: MetricConfig = field(default_factory=MetricConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    # Experiment-level
    seed: int = 2026
    num_idm_seeds: int = 5  # §9.2: 5 IDM seed repetitions
    output_dir: str = field(
        default_factory=lambda: _env_or_legacy(
            "SIMULATION_ENGINE_OUTPUT",
            "RECTOR_CLOSEDLOOP_OUTPUT",
            "/workspace/output/closedloop",
        )
    )
    figures_dir: str = field(
        default_factory=lambda: _env_or_legacy(
            "SIMULATION_ENGINE_FIGURES",
            "RECTOR_CLOSEDLOOP_FIGURES",
            "/workspace/output/closedloop/figures",
        )
    )
    videos_dir: str = field(
        default_factory=lambda: _env_or_legacy(
            "SIMULATION_ENGINE_VIDEOS",
            "RECTOR_CLOSEDLOOP_VIDEOS",
            "/workspace/output/closedloop/videos",
        )
    )
    traces_dir: str = field(
        default_factory=lambda: _env_or_legacy(
            "SIMULATION_ENGINE_TRACES",
            "RECTOR_CLOSEDLOOP_TRACES",
            "/workspace/output/closedloop/traces",
        )
    )
