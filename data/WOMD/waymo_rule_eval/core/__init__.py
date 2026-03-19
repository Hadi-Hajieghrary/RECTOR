"""Core modules for Waymo Rule Evaluation."""

from .context import Agent, EgoState, MapContext, MapSignals, ScenarioContext
from .geometry import (
    angle_diff,
    compute_bumper_distance,
    compute_relative_position,
    normalize_angle,
    oriented_box_corners,
    sat_collision_check,
)
from .temporal_spatial import FrameSpatialIndex, TemporalSpatialIndex
