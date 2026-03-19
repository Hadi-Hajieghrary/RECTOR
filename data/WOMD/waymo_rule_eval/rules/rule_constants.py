"""
Canonical rule ordering and constants.

CRITICAL: This is the SINGLE SOURCE OF TRUTH for all rule-related indexing.
All components (labels, model heads, proxies, tier masks) MUST use these constants.

This module auto-generates the rule ordering from the registry to ensure consistency.
"""

from typing import List, Dict, Set, Optional
from dataclasses import dataclass
import numpy as np


# --- AUTO-GENERATED CANONICAL ORDERING ---


def _build_rule_ids() -> List[str]:
    """
    Build canonical rule ID list from registry.

    Sorted alphabetically for stability across code changes.
    CRITICAL: This is called once at module load. Any rules added later
    will NOT be included in RULE_IDS.
    """
    from .registry import all_rules

    return sorted([r.rule_id for r in all_rules()])


# Canonical rule ordering - alphabetically sorted for stability
RULE_IDS: List[str] = _build_rule_ids()
NUM_RULES: int = len(RULE_IDS)
RULE_INDEX_MAP: Dict[str, int] = {rule_id: idx for idx, rule_id in enumerate(RULE_IDS)}


def get_rule_index(rule_id: str) -> int:
    """
    Get canonical index for a rule ID.

    Args:
        rule_id: Rule identifier (e.g., "L10.R1")

    Returns:
        Integer index in [0, NUM_RULES)

    Raises:
        KeyError: If rule_id is not in the canonical ordering
    """
    return RULE_INDEX_MAP[rule_id]


# --- TIER DEFINITIONS (by rule_id strings, NOT indices) ---

# Tier 0: Safety Critical (MUST NOT violate - lexicographically dominant)
# These rules represent potential harm to self, other vehicles, or VRUs
TIER_0_SAFETY: List[str] = [
    "L0.R2",  # Safe longitudinal distance
    "L0.R3",  # Safe lateral clearance
    "L0.R4",  # Crosswalk occupancy
    "L10.R1",  # Collision with vehicle
    "L10.R2",  # VRU clearance
]

# Tier 1: Legal/Regulatory (traffic law compliance)
TIER_1_LEGAL: List[str] = [
    "L5.R1",  # Traffic signal compliance
    "L5.R2",  # Priority violation
    "L7.R4",  # Speed limit
    "L8.R1",  # Red light
    "L8.R2",  # Stop sign
    "L8.R3",  # Crosswalk yield
    "L8.R5",  # Wrong way
]

# Tier 2: Road Structure (lane keeping, drivable surface)
TIER_2_ROAD: List[str] = [
    "L3.R3",  # Drivable surface
    "L7.R3",  # Lane departure
]

# Tier 3: Comfort/Efficiency (everything else)
TIER_3_COMFORT: List[str] = [
    "L1.R1",  # Smooth acceleration
    "L1.R2",  # Smooth braking
    "L1.R3",  # Smooth steering
    "L1.R4",  # Speed consistency
    "L1.R5",  # Lane change smoothness
    "L4.R3",  # Left turn gap
    "L5.R3",  # Parking violation
    "L5.R4",  # School zone compliance
    "L5.R5",  # Construction zone compliance
    "L6.R1",  # Cooperative lane change
    "L6.R2",  # Following distance
    "L6.R3",  # Intersection negotiation
    "L6.R4",  # Pedestrian interaction
    "L6.R5",  # Cyclist interaction
]


TIER_DEFINITIONS: Dict[int, List[str]] = {
    0: TIER_0_SAFETY,
    1: TIER_1_LEGAL,
    2: TIER_2_ROAD,
    3: TIER_3_COMFORT,
}

# Tier names in priority order (for iteration)
TIERS: List[str] = ["safety", "legal", "road", "comfort"]

# Map tier name to tier index
TIER_NAME_TO_IDX: Dict[str, int] = {
    "safety": 0,
    "legal": 1,
    "road": 2,
    "comfort": 3,
}

# Map tier name to rule list
TIER_BY_NAME: Dict[str, List[str]] = {
    "safety": TIER_0_SAFETY,
    "legal": TIER_1_LEGAL,
    "road": TIER_2_ROAD,
    "comfort": TIER_3_COMFORT,
}

# Safety-critical rules that MUST have proxies
SAFETY_CRITICAL_RULES: Set[str] = set(TIER_0_SAFETY)

# Tier weights for lexicographic scoring (exponential base)
# Tier 0 violations are weighted ~1000x more than Tier 3
TIER_WEIGHTS: Dict[int, float] = {
    0: 1000.0,  # Safety: catastrophic
    1: 100.0,  # Legal: severe
    2: 10.0,  # Road: moderate
    3: 1.0,  # Comfort: minor
}


def get_tier_mask(tier) -> np.ndarray:
    """
    Get [NUM_RULES] boolean mask for a tier.

    Args:
        tier: Tier number (0-3) OR tier name ('safety', 'legal', 'road', 'comfort')

    Returns:
        Boolean array of shape [NUM_RULES] where True indicates rule is in tier
    """
    # Handle string tier names
    if isinstance(tier, str):
        tier_rules = TIER_BY_NAME.get(tier, [])
    else:
        tier_rules = TIER_DEFINITIONS.get(tier, [])

    mask = np.zeros(NUM_RULES, dtype=bool)
    for rule_id in tier_rules:
        if rule_id in RULE_INDEX_MAP:
            mask[RULE_INDEX_MAP[rule_id]] = True
    return mask


def get_tier_indices(tier: int) -> List[int]:
    """
    Get list of canonical indices for rules in a tier.

    Args:
        tier: Tier number (0-3)

    Returns:
        List of rule indices in the canonical ordering
    """
    return [
        RULE_INDEX_MAP[r] for r in TIER_DEFINITIONS.get(tier, []) if r in RULE_INDEX_MAP
    ]


def get_rule_tier(rule_id: str) -> int:
    """
    Get tier for a rule ID.

    Args:
        rule_id: Rule identifier

    Returns:
        Tier number (0-3). Returns 3 (comfort) if not found.
    """
    for tier, rules in TIER_DEFINITIONS.items():
        if rule_id in rules:
            return tier
    return 3  # Default to comfort tier


def get_tier_weight(rule_id: str) -> float:
    """
    Get the tier weight for a rule.

    Args:
        rule_id: Rule identifier

    Returns:
        Weight for the rule's tier
    """
    tier = get_rule_tier(rule_id)
    return TIER_WEIGHTS.get(tier, 1.0)


# --- TIER MASKS (pre-computed for efficiency) ---

TIER_0_MASK: np.ndarray = get_tier_mask(0)
TIER_1_MASK: np.ndarray = get_tier_mask(1)
TIER_2_MASK: np.ndarray = get_tier_mask(2)
TIER_3_MASK: np.ndarray = get_tier_mask(3)


def get_tier_weight_vector() -> np.ndarray:
    """
    Get [NUM_RULES] weight vector based on tier assignments.

    Returns:
        Float array of shape [NUM_RULES] with tier weights
    """
    weights = np.ones(NUM_RULES, dtype=np.float32)
    for rule_id, idx in RULE_INDEX_MAP.items():
        weights[idx] = get_tier_weight(rule_id)
    return weights


# --- VALIDATION (Run at module load) ---


def validate_rule_ordering():
    """
    Validate that rule ordering is consistent with registry.

    Checks:
    1. All rules in registry are in RULE_IDS
    2. All tier-assigned rules exist in RULE_IDS
    3. All tiers have at least one rule
    4. No rule appears in multiple tiers

    Raises:
        AssertionError: If validation fails
    """
    from .registry import all_rules

    registry_ids = set(r.rule_id for r in all_rules())
    constant_ids = set(RULE_IDS)

    # Check registry matches constants
    if registry_ids != constant_ids:
        missing = registry_ids - constant_ids
        extra = constant_ids - registry_ids
        msg = []
        if missing:
            msg.append(f"Rules in registry but not in RULE_IDS: {missing}")
        if extra:
            msg.append(f"Rules in RULE_IDS but not in registry: {extra}")
        raise AssertionError("\n".join(msg))

    # Check all tier rules exist
    all_tier_rules = set()
    for tier, rules in TIER_DEFINITIONS.items():
        for rule_id in rules:
            if rule_id not in RULE_INDEX_MAP:
                raise AssertionError(f"Tier {tier} contains unknown rule: {rule_id}")
            if rule_id in all_tier_rules:
                raise AssertionError(f"Rule {rule_id} appears in multiple tiers")
            all_tier_rules.add(rule_id)

    # Check all rules are assigned to a tier
    unassigned = constant_ids - all_tier_rules
    if unassigned:
        print(
            f"Warning: Rules not assigned to any tier (defaulting to comfort): {unassigned}"
        )

    print(f"✓ Rule ordering validated: {NUM_RULES} rules, 4 tiers")
    print(f"  Tier 0 (Safety): {len(TIER_0_SAFETY)} rules")
    print(f"  Tier 1 (Legal): {len(TIER_1_LEGAL)} rules")
    print(f"  Tier 2 (Road): {len(TIER_2_ROAD)} rules")
    print(f"  Tier 3 (Comfort): {len(TIER_3_COMFORT)} rules")


# --- RULE METADATA ---


@dataclass(frozen=True)
class RuleMetadata:
    """Metadata for a single rule."""

    rule_id: str
    level: int  # L0, L1, L3, etc.
    tier: int  # 0 (safety) to 3 (comfort)
    name: str
    has_proxy: bool = False
    proxy_class: Optional[str] = None  # Class name of proxy


def _parse_level(rule_id: str) -> int:
    """Extract level number from rule ID (e.g., 'L10.R1' -> 10)."""
    try:
        return int(rule_id.split(".")[0][1:])
    except (IndexError, ValueError):
        return -1


def build_rule_metadata() -> Dict[str, RuleMetadata]:
    """
    Build metadata dict for all rules.

    Should be called after proxy registration to include proxy info.

    Returns:
        Dict mapping rule_id to RuleMetadata
    """
    from .registry import all_rules

    metadata = {}
    for entry in all_rules():
        rule_id = entry.rule_id
        level = _parse_level(rule_id)
        tier = get_rule_tier(rule_id)

        # Get name from detector or evaluator
        name = (
            getattr(entry.detector, "name", None)
            or getattr(entry.evaluator, "name", None)
            or rule_id
        )

        metadata[rule_id] = RuleMetadata(
            rule_id=rule_id,
            level=level,
            tier=tier,
            name=name,
            has_proxy=False,  # Will be updated after proxy registration
            proxy_class=None,
        )

    return metadata


# Populated after module load
RULE_METADATA: Dict[str, RuleMetadata] = {}


# --- MODULE INITIALIZATION ---


def _initialize_module():
    """Initialize module state and run validation."""
    global RULE_METADATA

    try:
        validate_rule_ordering()
        RULE_METADATA = build_rule_metadata()
    except Exception as e:
        print(f"Warning: Rule ordering validation failed: {e}")
        # Don't raise - allow module to load for partial functionality


# Run initialization at module load
_initialize_module()
