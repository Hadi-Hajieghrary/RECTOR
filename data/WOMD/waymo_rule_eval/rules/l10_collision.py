"""
L10.R1: Collision Detection Rule

Detects actual collisions between ego and other agents using
Separating Axis Theorem (SAT) on oriented bounding boxes.

Features:
- Uses TemporalSpatialIndex for O(log N) pre-filtering
- Pre-computes box axes for efficiency
- Respects validity masks
- Uses proper dimensions from agents
"""

from dataclasses import dataclass
from typing import List, Tuple

from ..core.context import ScenarioContext
from ..core.geometry import (
    get_box_separating_axes,
    oriented_box_corners,
    sat_collision_check,
)
from ..core.temporal_spatial import TemporalSpatialIndex
from ..utils.constants import (
    COLLISION_PENETRATION_THRESHOLD_M,
    DEFAULT_EGO_LENGTH,
    DEFAULT_EGO_WIDTH,
    MAX_COLLISION_CHECK_RADIUS_M,
)
from .base import (
    ApplicabilityDetector,
    ApplicabilityResult,
    ViolationEvaluator,
    ViolationResult,
)

_RULE_ID = "L10.R1"
_LEVEL = 10
_NAME = "Collision"


@dataclass
class CollisionEvent:
    """Details of a detected collision."""

    frame_idx: int
    agent_id: int
    agent_type: str
    penetration_depth: float
    ego_position: Tuple[float, float]
    agent_position: Tuple[float, float]


class CollisionApplicability(ApplicabilityDetector):
    """Collision detection always applies when there are agents."""

    rule_id = _RULE_ID
    level = _LEVEL
    name = _NAME

    def detect(self, ctx: ScenarioContext) -> ApplicabilityResult:
        """Collision check always applies if there are other agents."""
        n_agents = len(ctx.agents)

        if n_agents == 0:
            return ApplicabilityResult(
                applies=False,
                confidence=1.0,
                reasons=["No other agents in scenario"],
                features={"n_agents": 0},
            )

        return ApplicabilityResult(
            applies=True,
            confidence=1.0,
            reasons=[f"Collision detection active for {n_agents} agents"],
            features={"n_agents": n_agents},
        )


class CollisionViolation(ViolationEvaluator):
    """
    Evaluate collisions using SAT on oriented bounding boxes.

    Uses spatial pre-filtering to avoid O(N*T) complexity.
    """

    rule_id = _RULE_ID
    level = _LEVEL
    name = _NAME

    def __init__(
        self,
        penetration_threshold_m: float = COLLISION_PENETRATION_THRESHOLD_M,
        check_radius_m: float = MAX_COLLISION_CHECK_RADIUS_M,
    ):
        self.penetration_threshold_m = penetration_threshold_m
        self.check_radius_m = check_radius_m

    def evaluate(
        self, ctx: ScenarioContext, applicability: ApplicabilityResult
    ) -> ViolationResult:
        """Detect collisions across all frames."""

        ego = ctx.ego
        agents = ctx.agents
        T = len(ego.x)

        ego_length = getattr(ego, "length", DEFAULT_EGO_LENGTH)
        ego_width = getattr(ego, "width", DEFAULT_EGO_WIDTH)

        # Build spatial index ONCE for all frames
        spatial_idx = TemporalSpatialIndex(agents, T)

        total_penetration = 0.0
        max_penetration = 0.0
        collision_frames: List[int] = []
        collision_events: List[CollisionEvent] = []

        for t in range(T):
            # Skip if ego state is invalid
            if not ego.is_valid_at(t):
                continue

            ex, ey = ego.x[t], ego.y[t]
            eyaw = ego.yaw[t]

            # Get ego bounding box
            ego_box = oriented_box_corners(ex, ey, eyaw, ego_length, ego_width)
            ego_axes = get_box_separating_axes(eyaw)

            # OPTIMIZATION: Only check nearby agents using spatial index
            nearby = spatial_idx.at_frame(t).query_radius(ex, ey, self.check_radius_m)

            for agent, center_dist in nearby:
                # Skip if agent state is invalid at this frame
                if not agent.is_valid_at(t):
                    continue

                ax, ay = agent.x[t], agent.y[t]
                ayaw = agent.yaw[t]
                agent_length = getattr(agent, "length", 4.5)
                agent_width = getattr(agent, "width", 1.8)

                # Get agent bounding box
                agent_box = oriented_box_corners(
                    ax, ay, ayaw, agent_length, agent_width
                )
                agent_axes = get_box_separating_axes(ayaw)

                # SAT collision check
                collides, penetration = sat_collision_check(
                    ego_box, ego_axes, agent_box, agent_axes
                )

                if collides and penetration >= self.penetration_threshold_m:
                    if t not in collision_frames:
                        collision_frames.append(t)

                    total_penetration += penetration
                    max_penetration = max(max_penetration, penetration)

                    collision_events.append(
                        CollisionEvent(
                            frame_idx=t,
                            agent_id=agent.id,
                            agent_type=agent.type,
                            penetration_depth=penetration,
                            ego_position=(ex, ey),
                            agent_position=(ax, ay),
                        )
                    )

        # Compute severity
        n_collision_frames = len(collision_frames)
        severity = total_penetration

        # Normalize severity (cap at 1.0 for extreme cases)
        severity_normalized = min(1.0, max_penetration / 2.0)

        return ViolationResult(
            severity=severity,
            severity_normalized=severity_normalized,
            measurements={
                "n_collision_frames": n_collision_frames,
                "max_penetration_m": max_penetration,
                "total_penetration_m": total_penetration,
                "n_collision_events": len(collision_events),
            },
            explanation=(
                (
                    f"{n_collision_frames} frames with collisions, "
                    f"max penetration {max_penetration:.3f}m"
                )
                if n_collision_frames > 0
                else "No collisions"
            ),
            frame_violations=collision_frames,
        )
