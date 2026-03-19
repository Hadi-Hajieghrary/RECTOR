"""
Base class for differentiable rule proxies.

All proxies must:
1. Define which rule_ids they cover
2. Return costs in [0, 1] where 0 = no violation
3. Be differentiable (support autograd)

The proxies serve as differentiable approximations of the exact rule violations,
enabling gradient-based training while closely matching the behavior of the
discrete rule evaluators.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple, Set


class DifferentiableProxy(nn.Module, ABC):
    """
    Base class for differentiable rule violation proxies.

    Each proxy covers one or more rules from the canonical rule ordering.
    Proxies output violation costs in [0, 1] where:
    - 0 = no violation (fully compliant)
    - 1 = maximum violation

    Subclasses must implement:
    - rule_ids: List of rule IDs this proxy covers
    - forward(): Compute violation costs for trajectories
    """

    def __init__(self):
        super().__init__()

    @property
    @abstractmethod
    def rule_ids(self) -> List[str]:
        """
        List of rule IDs that this proxy covers.

        Must match entries in RULE_IDS from rule_constants.
        """
        pass

    @abstractmethod
    def forward(
        self,
        trajectories: torch.Tensor,
        scene_features: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute violation costs for trajectories.

        Args:
            trajectories: [B, M, H, 2] or [B, M, H, 4] candidate trajectories
                         B = batch size, M = num candidates, H = horizon
                         Last dim: (x, y) or (x, y, yaw, speed)
            scene_features: Dict of scene tensors:
                - agent_positions: [B, N, H, 2] other agent positions
                - agent_sizes: [B, N, 2] (length, width) per agent
                - agent_types: [B, N] agent type indices
                - lane_centers: [B, L, P, 2] lane centerlines
                - lane_headings: [B, L, P] lane headings
                - road_edges: [B, E, 2] road edge points
                - stoplines: [B, S, 2] stopline positions
                - signal_states: [B, H] traffic signal states
                - ego_size: [B, 2] (length, width) of ego

        Returns:
            Dict mapping rule_id to cost tensor of shape [B, M]
            Each cost is in [0, 1] where 0 = no violation
        """
        pass

    def get_rule_indices(self) -> List[int]:
        """Get canonical indices for rules covered by this proxy."""
        from waymo_rule_eval.rules.rule_constants import RULE_INDEX_MAP

        return [RULE_INDEX_MAP[rid] for rid in self.rule_ids if rid in RULE_INDEX_MAP]


def soft_threshold(
    x: torch.Tensor, threshold: float, sharpness: float = 10.0
) -> torch.Tensor:
    """
    Soft threshold function: returns ~0 for x < threshold, ~1 for x > threshold.

    Uses sigmoid for smooth transition:
    σ((x - threshold) * sharpness)

    Args:
        x: Input tensor
        threshold: Threshold value
        sharpness: Controls transition steepness (higher = sharper)

    Returns:
        Values in [0, 1]
    """
    return torch.sigmoid((x - threshold) * sharpness)


def soft_clamp(x: torch.Tensor, min_val: float, max_val: float) -> torch.Tensor:
    """
    Soft clamping that maintains gradients at boundaries.

    Uses smooth approximation of clamp.
    """
    return torch.sigmoid((x - min_val) * 10) * (max_val - min_val) + min_val


def smooth_max(
    x: torch.Tensor, dim: int = -1, temperature: float = 1.0
) -> torch.Tensor:
    """
    Differentiable approximation of max using log-sum-exp.

    smooth_max(x) ≈ max(x) as temperature → 0

    Args:
        x: Input tensor
        dim: Dimension to reduce
        temperature: Smoothing temperature (smaller = sharper)

    Returns:
        Smooth maximum values
    """
    return temperature * torch.logsumexp(x / temperature, dim=dim)


def smooth_min(
    x: torch.Tensor, dim: int = -1, temperature: float = 1.0
) -> torch.Tensor:
    """
    Differentiable approximation of min using negative log-sum-exp.

    Args:
        x: Input tensor
        dim: Dimension to reduce
        temperature: Smoothing temperature

    Returns:
        Smooth minimum values
    """
    return -smooth_max(-x, dim=dim, temperature=temperature)


def exponential_cost(
    violation_amount: torch.Tensor,
    softness: float = 5.0,
) -> torch.Tensor:
    """
    Convert violation amount to [0, 1] cost using exponential.

    cost = 1 - exp(-softness * relu(violation_amount))

    CRITICAL: This gives 0 when violation_amount <= 0 (no violation).
    This is the CORRECT formulation. DO NOT use sigmoid which gives 0.5 at zero.

    Args:
        violation_amount: Amount of violation (positive = violating)
        softness: Controls how quickly cost approaches 1

    Returns:
        Cost in [0, 1]
    """
    return 1.0 - torch.exp(-softness * F.relu(violation_amount))


def safe_divide(
    numerator: torch.Tensor,
    denominator: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Safe division with epsilon for numerical stability."""
    return numerator / (denominator + eps)


def batch_pairwise_distance(
    points_a: torch.Tensor,
    points_b: torch.Tensor,
) -> torch.Tensor:
    """
    Compute pairwise distances between two point sets.

    Args:
        points_a: [..., N, 2] first point set
        points_b: [..., M, 2] second point set

    Returns:
        [..., N, M] pairwise distances
    """
    # Expand for broadcasting
    a = points_a.unsqueeze(-2)  # [..., N, 1, 2]
    b = points_b.unsqueeze(-3)  # [..., 1, M, 2]

    return torch.norm(a - b, dim=-1)  # [..., N, M]


def obb_corners(
    positions: torch.Tensor,
    headings: torch.Tensor,
    sizes: torch.Tensor,
) -> torch.Tensor:
    """
    Compute oriented bounding box corners.

    Args:
        positions: [..., 2] center positions (x, y)
        headings: [...] heading angles in radians
        sizes: [..., 2] (length, width)

    Returns:
        [..., 4, 2] corner positions (front-left, front-right, rear-right, rear-left)
    """
    # Half dimensions
    half_l = sizes[..., 0:1] / 2  # [..., 1]
    half_w = sizes[..., 1:2] / 2  # [..., 1]

    # Rotation matrix components
    cos_h = torch.cos(headings).unsqueeze(-1)  # [..., 1]
    sin_h = torch.sin(headings).unsqueeze(-1)  # [..., 1]

    # Local corner offsets (length along x, width along y)
    local_corners = torch.stack(
        [
            torch.cat([half_l, half_w], dim=-1),  # front-left
            torch.cat([half_l, -half_w], dim=-1),  # front-right
            torch.cat([-half_l, -half_w], dim=-1),  # rear-right
            torch.cat([-half_l, half_w], dim=-1),  # rear-left
        ],
        dim=-2,
    )  # [..., 4, 2]

    # Rotate corners
    rotated_x = local_corners[..., 0] * cos_h - local_corners[..., 1] * sin_h
    rotated_y = local_corners[..., 0] * sin_h + local_corners[..., 1] * cos_h

    # Translate to global position
    corners = torch.stack([rotated_x, rotated_y], dim=-1)
    corners = corners + positions.unsqueeze(-2)

    return corners


def point_in_polygon_2d(
    points: torch.Tensor,
    polygon: torch.Tensor,
) -> torch.Tensor:
    """
    Check if points are inside a convex polygon using cross-product method.

    Args:
        points: [..., 2] query points
        polygon: [P, 2] polygon vertices in order

    Returns:
        [...] boolean tensor
    """
    P = polygon.shape[0]

    # Expand points for edge comparison
    points = points.unsqueeze(-2)  # [..., 1, 2]

    # Check sign of cross product with each edge
    all_inside = torch.ones(points.shape[:-2], dtype=torch.bool, device=points.device)

    for i in range(P):
        p1 = polygon[i]
        p2 = polygon[(i + 1) % P]

        # Edge vector
        edge = p2 - p1

        # Vector to point
        to_point = points.squeeze(-2) - p1

        # Cross product (2D)
        cross = edge[0] * to_point[..., 1] - edge[1] * to_point[..., 0]

        # Update inside check
        all_inside = all_inside & (cross >= 0)

    return all_inside


def compute_time_to_collision(
    ego_pos: torch.Tensor,
    ego_vel: torch.Tensor,
    other_pos: torch.Tensor,
    other_vel: torch.Tensor,
    min_dist: float = 0.5,
) -> torch.Tensor:
    """
    Compute time to collision assuming constant velocity.

    Args:
        ego_pos: [..., 2] ego position
        ego_vel: [..., 2] ego velocity
        other_pos: [..., 2] other agent position
        other_vel: [..., 2] other agent velocity
        min_dist: Minimum distance for collision

    Returns:
        [...] time to collision (inf if no collision)
    """
    # Relative position and velocity
    rel_pos = other_pos - ego_pos
    rel_vel = other_vel - ego_vel

    # Quadratic coefficients: |rel_pos + t * rel_vel|^2 = min_dist^2
    a = torch.sum(rel_vel**2, dim=-1)
    b = 2 * torch.sum(rel_pos * rel_vel, dim=-1)
    c = torch.sum(rel_pos**2, dim=-1) - min_dist**2

    # Discriminant
    disc = b**2 - 4 * a * c

    # Time to collision
    ttc = torch.full_like(a, float("inf"))

    # Only compute where collision is possible
    valid = (disc >= 0) & (a > 1e-6)
    sqrt_disc = torch.sqrt(torch.clamp(disc, min=0))

    t1 = (-b - sqrt_disc) / (2 * a + 1e-6)
    t2 = (-b + sqrt_disc) / (2 * a + 1e-6)

    # Take the earlier positive time
    t_min = torch.where(t1 > 0, t1, t2)
    ttc = torch.where(valid & (t_min > 0), t_min, ttc)

    return ttc
