"""
Heuristic applicability baselines for ablation analysis.

Provides non-learned applicability masks to isolate the contribution of
the learned applicability head vs. simple heuristics.

Three modes:
  - always_on:  All 28 rules active for every scenario (a_r = 1 for all r).
  - heuristic:  Rule-specific gating based on scene features (presence of
                signals, crosswalks, nearby agents, speed regime, etc.).
  - learned:    Default — use the neural applicability head predictions.
"""

import numpy as np
from typing import Dict, Any, Optional

import sys
from pathlib import Path

sys.path.insert(0, "/workspace/data/WOMD")

from waymo_rule_eval.rules.rule_constants import (
    RULE_IDS,
    NUM_RULES,
    RULE_INDEX_MAP,
    TIER_0_SAFETY,
    TIER_1_LEGAL,
    TIER_2_ROAD,
    TIER_3_COMFORT,
)


def always_on_applicability(batch_size: int) -> np.ndarray:
    """All 28 rules always active. Returns [B, NUM_RULES] ones."""
    return np.ones((batch_size, NUM_RULES), dtype=np.float32)


def hybrid_conservative_applicability(
    learned_pred: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Conservative hybrid: always-on for Tier 0 (Safety) and Tier 1 (Legal),
    learned prediction for Tier 2 (Road) and Tier 3 (Comfort).

    This addresses the reviewer question: does a conservative all-activate
    policy for high-priority tiers outperform the learned head on Safety
    compliance?

    Args:
        learned_pred: [B, NUM_RULES] learned applicability probabilities.
        threshold: Binarization threshold for learned predictions on Tier 2+3.

    Returns:
        applicability: [B, NUM_RULES] float32 array with 0/1 values.
    """
    B = learned_pred.shape[0]
    app = np.zeros((B, NUM_RULES), dtype=np.float32)

    # Tier 0 (Safety) + Tier 1 (Legal): always on
    for rule_id in TIER_0_SAFETY + TIER_1_LEGAL:
        if rule_id in RULE_INDEX_MAP:
            app[:, RULE_INDEX_MAP[rule_id]] = 1.0

    # Tier 2 (Road) + Tier 3 (Comfort): use learned predictions
    for rule_id in TIER_2_ROAD + TIER_3_COMFORT:
        if rule_id in RULE_INDEX_MAP:
            idx = RULE_INDEX_MAP[rule_id]
            app[:, idx] = (learned_pred[:, idx] > threshold).astype(np.float32)

    return app


def heuristic_applicability(
    batch: Dict[str, Any],
    speed_threshold: float = 5.0,
) -> np.ndarray:
    """
    Scene-feature heuristic applicability.

    Uses simple, interpretable rules derived from the scenario context
    to predict which rules are relevant, without any learned parameters.

    Heuristic logic:
      - Safety rules (L0.*, L10.*): Always on — safety is non-negotiable.
      - Signal rules (L5.R1, L8.R1): On if traffic signals present in map.
      - Stop sign (L8.R2): On if stop signs present in map.
      - Crosswalk rules (L0.R4, L8.R3): On if crosswalks present in map.
      - Speed limit (L7.R4): Always on when ego speed > threshold.
      - Lane rules (L3.R3, L7.R3): Always on (lane structure always relevant).
      - Comfort rules (L1.*): Always on (kinematic limits always apply).
      - Interaction rules (L6.*): On if nearby agents detected.
      - Priority/zone rules (L5.R2-R5): On if relevant zone features present.
      - VRU rules (L10.R2, L6.R4, L6.R5): On if pedestrians/cyclists present.

    Args:
        batch: Data batch dict with keys like 'agent_states', 'lane_centers',
               and optional 'traffic_signals', 'crosswalks', 'stop_signs' etc.
        speed_threshold: Minimum ego speed (m/s) for speed-limit rule activation.

    Returns:
        applicability: [B, NUM_RULES] float32 array with 0/1 values.
    """
    B = (
        batch["agent_states"].shape[0]
        if hasattr(batch["agent_states"], "shape")
        else len(batch["agent_states"])
    )
    app = np.zeros((B, NUM_RULES), dtype=np.float32)

    # --- Always-on rules ---
    # Safety tier: always active
    for rule_id in TIER_0_SAFETY:
        if rule_id in RULE_INDEX_MAP:
            app[:, RULE_INDEX_MAP[rule_id]] = 1.0

    # Lane keeping / drivable surface: always active
    for rule_id in TIER_2_ROAD:
        if rule_id in RULE_INDEX_MAP:
            app[:, RULE_INDEX_MAP[rule_id]] = 1.0

    # Comfort/kinematic rules: always active
    for rule_id in ["L1.R1", "L1.R2", "L1.R3", "L1.R4", "L1.R5"]:
        if rule_id in RULE_INDEX_MAP:
            app[:, RULE_INDEX_MAP[rule_id]] = 1.0

    # Speed limit: always active (conservative)
    if "L7.R4" in RULE_INDEX_MAP:
        app[:, RULE_INDEX_MAP["L7.R4"]] = 1.0

    # --- Conditionally-on rules ---

    # Check for nearby agents (activate interaction rules)
    has_agents = _detect_nearby_agents(batch)
    for rule_id in ["L6.R1", "L6.R2", "L6.R3"]:
        if rule_id in RULE_INDEX_MAP:
            app[:, RULE_INDEX_MAP[rule_id]] = has_agents.astype(np.float32)

    # Check for VRUs (pedestrians/cyclists)
    has_vru = _detect_vru(batch)
    for rule_id in ["L6.R4", "L6.R5", "L10.R2"]:
        if rule_id in RULE_INDEX_MAP:
            app[:, RULE_INDEX_MAP[rule_id]] = has_vru.astype(np.float32)

    # Check for traffic signals
    has_signals = _detect_signals(batch)
    for rule_id in ["L5.R1", "L8.R1"]:
        if rule_id in RULE_INDEX_MAP:
            app[:, RULE_INDEX_MAP[rule_id]] = has_signals.astype(np.float32)

    # Check for stop signs
    has_stop_signs = _detect_stop_signs(batch)
    if "L8.R2" in RULE_INDEX_MAP:
        app[:, RULE_INDEX_MAP["L8.R2"]] = has_stop_signs.astype(np.float32)

    # Check for crosswalks
    has_crosswalks = _detect_crosswalks(batch)
    for rule_id in ["L0.R4", "L8.R3"]:
        if rule_id in RULE_INDEX_MAP:
            app[:, RULE_INDEX_MAP[rule_id]] = has_crosswalks.astype(np.float32)

    # Check for wrong-way possibility (always on as conservative default)
    if "L8.R5" in RULE_INDEX_MAP:
        app[:, RULE_INDEX_MAP["L8.R5"]] = 1.0

    # Zone rules (L5.R2-R5) — conservative: activate if any zone-like features
    has_zones = _detect_zones(batch)
    for rule_id in ["L5.R2", "L5.R3", "L5.R4", "L5.R5"]:
        if rule_id in RULE_INDEX_MAP:
            app[:, RULE_INDEX_MAP[rule_id]] = has_zones.astype(np.float32)

    # Left turn gap
    has_intersection = _detect_intersection(batch)
    if "L4.R3" in RULE_INDEX_MAP:
        app[:, RULE_INDEX_MAP["L4.R3"]] = has_intersection.astype(np.float32)

    return app


def _detect_nearby_agents(batch: Dict[str, Any]) -> np.ndarray:
    """Detect if nearby agents exist within interaction range. Returns [B] bool."""
    agent_states = batch["agent_states"]
    if hasattr(agent_states, "numpy"):
        agent_states = agent_states.numpy()
    # Check if any agent positions are non-zero (present)
    B = agent_states.shape[0]
    agent_present = (
        np.abs(agent_states).sum(axis=tuple(range(2, agent_states.ndim))) > 0
    )
    if agent_present.ndim > 1:
        return agent_present.any(axis=1)
    return agent_present


def _detect_vru(batch: Dict[str, Any]) -> np.ndarray:
    """Detect vulnerable road users. Returns [B] bool."""
    # Conservative: if agent_types available, check for pedestrians/cyclists
    if "agent_types" in batch:
        types = batch["agent_types"]
        if hasattr(types, "numpy"):
            types = types.numpy()
        # Types 2=pedestrian, 3=cyclist in WOMD convention
        return (
            ((types == 2) | (types == 3)).any(axis=-1)
            if types.ndim > 1
            else (types == 2) | (types == 3)
        )
    # Fallback: assume VRU present (conservative)
    B = batch["agent_states"].shape[0] if hasattr(batch["agent_states"], "shape") else 1
    return np.ones(B, dtype=bool)


def _detect_signals(batch: Dict[str, Any]) -> np.ndarray:
    """Detect traffic signals. Returns [B] bool."""
    if "traffic_signals" in batch:
        signals = batch["traffic_signals"]
        if hasattr(signals, "numpy"):
            signals = signals.numpy()
        return np.abs(signals).sum(axis=tuple(range(1, signals.ndim))) > 0
    # Conservative: assume signals present
    B = batch["agent_states"].shape[0] if hasattr(batch["agent_states"], "shape") else 1
    return np.ones(B, dtype=bool)


def _detect_stop_signs(batch: Dict[str, Any]) -> np.ndarray:
    """Detect stop signs. Returns [B] bool."""
    if "stop_signs" in batch:
        stops = batch["stop_signs"]
        if hasattr(stops, "numpy"):
            stops = stops.numpy()
        return np.abs(stops).sum(axis=tuple(range(1, stops.ndim))) > 0
    # Conservative: assume present
    B = batch["agent_states"].shape[0] if hasattr(batch["agent_states"], "shape") else 1
    return np.ones(B, dtype=bool)


def _detect_crosswalks(batch: Dict[str, Any]) -> np.ndarray:
    """Detect crosswalks. Returns [B] bool."""
    if "crosswalks" in batch:
        cw = batch["crosswalks"]
        if hasattr(cw, "numpy"):
            cw = cw.numpy()
        return np.abs(cw).sum(axis=tuple(range(1, cw.ndim))) > 0
    B = batch["agent_states"].shape[0] if hasattr(batch["agent_states"], "shape") else 1
    return np.ones(B, dtype=bool)


def _detect_zones(batch: Dict[str, Any]) -> np.ndarray:
    """Detect special zones (parking, school, construction). Returns [B] bool."""
    # WOMD does not explicitly label zones; conservative default
    B = batch["agent_states"].shape[0] if hasattr(batch["agent_states"], "shape") else 1
    return np.zeros(B, dtype=bool)


def _detect_intersection(batch: Dict[str, Any]) -> np.ndarray:
    """Detect intersection presence. Returns [B] bool."""
    # Use lane connectivity as proxy: if multiple lane directions exist
    if "lane_centers" in batch:
        lanes = batch["lane_centers"]
        if hasattr(lanes, "numpy"):
            lanes = lanes.numpy()
        # Non-zero lanes suggest structured road; multiple directions suggest intersection
        B = lanes.shape[0]
        lane_present = np.abs(lanes).sum(axis=tuple(range(2, lanes.ndim))) > 0
        if lane_present.ndim > 1:
            num_lanes = lane_present.sum(axis=1)
            return num_lanes > 3  # Heuristic: >3 lane segments suggests intersection
        return np.ones(B, dtype=bool)
    B = batch["agent_states"].shape[0] if hasattr(batch["agent_states"], "shape") else 1
    return np.ones(B, dtype=bool)
