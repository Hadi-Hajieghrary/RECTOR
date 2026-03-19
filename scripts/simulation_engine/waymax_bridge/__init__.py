"""
Waymax ↔ RECTOR Bridge
======================

Bridges JAX-based Waymax simulation with PyTorch-based RECTOR inference
through NumPy at each planning tick (§3.2 of the demonstration plan).

Modules
-------
scenario_loader        – Load WOMD scenarios from augmented TFRecords
env_factory            – Create Waymax env with IDM/log-playback agents
observation_extractor  – Waymax SimulatorState → RECTOR observation arrays
action_converter       – RECTOR trajectory → Waymax action
metric_collector       – Waymax + custom metrics per step
simulation_loop        – Main closed-loop orchestration
"""

from .scenario_loader import ScenarioLoaderConfig, load_scenarios
from .env_factory import make_env
from .observation_extractor import extract_observation, ObservationExtractorConfig
from .action_converter import trajectory_to_action
from .metric_collector import MetricCollector, MetricCollectorConfig
from .simulation_loop import (
    WaymaxRECTORLoop,
    MockLogReplayGenerator,
    ScenarioResult,
    run_batch,
)

__all__ = [
    "ScenarioLoaderConfig",
    "load_scenarios",
    "make_env",
    "extract_observation",
    "ObservationExtractorConfig",
    "trajectory_to_action",
    "MetricCollector",
    "MetricCollectorConfig",
    "WaymaxRECTORLoop",
    "MockLogReplayGenerator",
    "ScenarioResult",
    "run_batch",
]
