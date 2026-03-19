# waymo_rule_eval/utils/version.py
# -*- coding: utf-8 -*-
"""Version tracking for rule evaluation engine."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

# Engine version (pipeline infrastructure)
ENGINE_VERSION = "1.0.0"

# Rule implementation versions (track changes to rule logic)
RULE_VERSIONS: Dict[str, str] = {
    # L0: Critical Safety
    "L0.R2": "1.0.0",  # SafeLongitudinalDistance
    "L0.R3": "1.0.0",  # SafeLateralClearance
    "L0.R4": "1.0.0",  # CrosswalkOccupancy
    # L1: Comfort
    "L1.R1": "1.0.0",  # SmoothAcceleration
    "L1.R2": "1.0.0",  # SmoothBraking
    "L1.R3": "1.0.0",  # SmoothSteering
    "L1.R4": "1.0.0",  # SpeedConsistency
    "L1.R5": "1.0.0",  # LaneChangeSmoothness
    # L3: Surface
    "L3.R3": "1.0.0",  # DrivableSurface
    # L4: Maneuver
    "L4.R3": "1.0.0",  # LeftTurnGap
    # L5: Regulatory
    "L5.R1": "1.0.0",  # TrafficSignalCompliance
    "L5.R2": "1.0.0",  # PriorityViolation
    "L5.R3": "1.0.0",  # ParkingViolation
    "L5.R4": "1.0.0",  # SchoolZoneCompliance
    "L5.R5": "1.0.0",  # ConstructionZoneCompliance
    # L6: Interaction
    "L6.R1": "1.0.0",  # CooperativeLaneChange
    "L6.R2": "1.0.0",  # FollowingDistance
    "L6.R3": "1.0.0",  # IntersectionNegotiation
    "L6.R4": "1.0.0",  # PedestrianInteraction
    "L6.R5": "1.0.0",  # CyclistInteraction
    # L7: Lane/Speed
    "L7.R3": "1.0.0",  # LaneDeparture
    "L7.R4": "1.0.0",  # SpeedLimit
    # L8: Traffic Control
    "L8.R1": "1.0.0",  # RedLight
    "L8.R2": "1.0.0",  # StopSign
    "L8.R3": "1.0.0",  # CrosswalkYield
    "L8.R5": "1.0.0",  # Wrongway
    # L10: Collision
    "L10.R1": "1.0.0",  # Collision
    "L10.R2": "1.0.0",  # VRUClearance
}


def get_engine_version() -> str:
    """Get the current engine version."""
    return ENGINE_VERSION


def get_rule_version(rule_id: str) -> str:
    """Get the implementation version for a specific rule."""
    return RULE_VERSIONS.get(rule_id, "unknown")


def get_all_versions() -> Dict[str, str]:
    """Get all rule versions."""
    return RULE_VERSIONS.copy()


@dataclass
class VersionInfo:
    """Version information for tracking in output records."""

    engine_version: str
    rule_versions: Dict[str, str]

    @classmethod
    def current(cls) -> "VersionInfo":
        """Get current version info."""
        return cls(engine_version=ENGINE_VERSION, rule_versions=RULE_VERSIONS.copy())

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "engine_version": self.engine_version,
            "rule_versions": self.rule_versions,
        }
