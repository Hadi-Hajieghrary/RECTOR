"""
Differentiable collision proxy using OBB approximation.

Covers:
- L10.R1: Collision with vehicle
- L10.R2: Collision with VRU (Vulnerable Road User)
- L0.R2: Safe longitudinal distance (clearance)
- L0.R3: Safe lateral clearance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List
from .base import (
    DifferentiableProxy,
    exponential_cost,
    obb_corners,
    batch_pairwise_distance,
    smooth_min,
    safe_divide,
)


class CollisionProxy(DifferentiableProxy):
    """
    Differentiable collision and clearance proxy.

    Uses Oriented Bounding Box (OBB) collision detection with
    soft penetration depth computation.

    Key design choice: Uses exponential cost function that gives
    0 for no overlap and approaches 1 for large overlaps.
    DO NOT use sigmoid which gives 0.5 at zero penetration.
    """

    def __init__(
        self,
        ego_length: float = 4.5,
        ego_width: float = 2.0,
        softness: float = 2.0,
        min_longitudinal_clearance: float = 2.0,
        min_lateral_clearance: float = 0.5,
        vru_safety_margin: float = 1.0,
    ):
        """
        Initialize collision proxy.

        Args:
            ego_length: Default ego vehicle length in meters
            ego_width: Default ego vehicle width in meters
            softness: Controls how sharply cost increases with penetration
            min_longitudinal_clearance: Minimum safe distance ahead (L0.R2)
            min_lateral_clearance: Minimum safe lateral distance (L0.R3)
            vru_safety_margin: Additional margin for VRUs (L10.R2)
        """
        super().__init__()
        self.ego_length = ego_length
        self.ego_width = ego_width
        self.softness = softness
        self.min_longitudinal_clearance = min_longitudinal_clearance
        self.min_lateral_clearance = min_lateral_clearance
        self.vru_safety_margin = vru_safety_margin

    @property
    def rule_ids(self) -> List[str]:
        return ["L0.R2", "L0.R3", "L10.R1", "L10.R2"]

    def forward(
        self,
        trajectories: torch.Tensor,
        scene_features: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute collision costs for trajectories.

        Args:
            trajectories: [B, M, H, 2+] candidate trajectories
            scene_features: Dict containing:
                - agent_positions: [B, N, H, 2]
                - agent_sizes: [B, N, 2]
                - agent_types: [B, N]
                - agent_valid: [B, N, H]
                - ego_size: [B, 2] (optional)

        Returns:
            Dict with rule_id -> [B, M] cost tensors
        """
        B, M, H, _ = trajectories.shape
        device = trajectories.device

        # Get ego trajectory positions and headings
        ego_xy = trajectories[..., :2]  # [B, M, H, 2]

        if trajectories.shape[-1] >= 3:
            ego_yaw = trajectories[..., 2]  # [B, M, H]
        else:
            # Compute heading from velocity
            ego_yaw = self._compute_heading(ego_xy)  # [B, M, H]

        # Get ego size
        if "ego_size" in scene_features:
            ego_size = scene_features["ego_size"]  # [B, 2]
        else:
            ego_size = torch.tensor(
                [[self.ego_length, self.ego_width]], device=device
            ).expand(B, 2)

        # Get other agents
        agent_positions = scene_features.get("agent_positions")  # [B, N, H, 2]
        agent_sizes = scene_features.get("agent_sizes")  # [B, N, 2]
        agent_types = scene_features.get("agent_types")  # [B, N]
        agent_valid = scene_features.get("agent_valid")  # [B, N, H]

        if agent_positions is None:
            # No other agents - zero cost for all rules
            zero_cost = torch.zeros(B, M, device=device)
            return {rid: zero_cost.clone() for rid in self.rule_ids}

        N = agent_positions.shape[1]

        # Create validity mask
        if agent_valid is None:
            agent_valid = torch.ones(B, N, H, dtype=torch.bool, device=device)

        # Create VRU mask (pedestrians and cyclists)
        # Assuming agent_types: 0=vehicle, 1=pedestrian, 2=cyclist
        if agent_types is not None:
            vru_mask = (agent_types == 1) | (agent_types == 2)  # [B, N]
        else:
            vru_mask = torch.zeros(B, N, dtype=torch.bool, device=device)

        vehicle_mask = ~vru_mask

        # Compute collision costs
        # Expand ego trajectory for comparison with all agents
        # [B, M, H, 2] -> [B, M, 1, H, 2]
        ego_xy_exp = ego_xy.unsqueeze(2)
        ego_yaw_exp = ego_yaw.unsqueeze(2)  # [B, M, 1, H]

        # Expand agent positions: [B, N, H, 2] -> [B, 1, N, H, 2]
        agent_xy_exp = agent_positions.unsqueeze(1)

        # Compute distances: [B, M, N, H]
        distances = torch.norm(ego_xy_exp - agent_xy_exp, dim=-1)

        # Compute penetration depths using OBB approximation
        # For efficiency, we use a simplified approach based on distance and sizes
        penetration, long_clearance, lat_clearance = self._compute_clearances(
            ego_xy=ego_xy,
            ego_yaw=ego_yaw,
            ego_size=ego_size,
            agent_xy=agent_positions,
            agent_sizes=agent_sizes,
            agent_valid=agent_valid,
        )
        # penetration: [B, M, N, H]
        # long_clearance: [B, M, N, H]
        # lat_clearance: [B, M, N, H]

        # L10.R1: Vehicle collision
        # Max penetration across time for each vehicle agent
        vehicle_penetration = (
            penetration * vehicle_mask.unsqueeze(1).unsqueeze(-1).float()
        )
        max_vehicle_penetration = vehicle_penetration.max(dim=-1)[0].max(dim=-1)[
            0
        ]  # [B, M]
        l10_r1_cost = exponential_cost(max_vehicle_penetration, self.softness)

        # L10.R2: VRU collision (with additional safety margin)
        # CRITICAL: VRU safety margin means we want extra clearance from VRUs.
        # Violation occurs when:
        #   1. There is actual penetration (collision), OR
        #   2. Clearance to VRU is less than safety_margin (too close)
        #
        # Compute minimum clearance to each VRU
        min_clearance = torch.min(long_clearance, lat_clearance)  # [B, M, N, H]

        # Proximity violation: positive when closer than safety_margin
        vru_proximity_violation = F.relu(self.vru_safety_margin - min_clearance)

        # Combine with actual penetration (for cases where there IS collision)
        # penetration is positive when overlapping, so take max
        vru_total_violation = torch.max(penetration, vru_proximity_violation)

        # Apply VRU mask - only count violations for VRUs
        vru_cost = vru_total_violation * vru_mask.unsqueeze(1).unsqueeze(-1).float()
        max_vru_cost = vru_cost.max(dim=-1)[0].max(dim=-1)[0]  # [B, M]
        l10_r2_cost = exponential_cost(max_vru_cost, self.softness)

        # L0.R2: Longitudinal clearance (following distance)
        # Only relevant for agents in the same or adjacent lanes (laterally close).
        # An agent 50m to the side doesn't require longitudinal clearance.
        # lat_clearance = lat_dist - combined_half_w, so lat_dist = lat_clearance + combined_half_w
        # Use lat_clearance < 3.0m as relevance filter (agent laterally within ~1.5 lane widths)
        lat_relevance = (lat_clearance < 3.0).float()
        long_violation = self.min_longitudinal_clearance - long_clearance
        long_violation = (
            long_violation * agent_valid.unsqueeze(1).float() * lat_relevance
        )
        max_long_violation = long_violation.max(dim=-1)[0].max(dim=-1)[0]  # [B, M]
        l0_r2_cost = exponential_cost(max_long_violation, self.softness)

        # L0.R3: Lateral clearance (passing distance)
        # Only relevant for agents that are longitudinally close.
        # Use long_clearance < 5.0m as relevance filter
        long_relevance = (long_clearance < 5.0).float()
        lat_violation = self.min_lateral_clearance - lat_clearance
        lat_violation = (
            lat_violation * agent_valid.unsqueeze(1).float() * long_relevance
        )
        max_lat_violation = lat_violation.max(dim=-1)[0].max(dim=-1)[0]  # [B, M]
        l0_r3_cost = exponential_cost(max_lat_violation, self.softness)

        return {
            "L0.R2": l0_r2_cost,
            "L0.R3": l0_r3_cost,
            "L10.R1": l10_r1_cost,
            "L10.R2": l10_r2_cost,
        }

    def _compute_heading(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Compute heading angles from position sequence.

        Args:
            positions: [..., H, 2] positions

        Returns:
            [..., H] heading angles
        """
        # Velocity from finite differences
        dx = positions[..., 1:, 0] - positions[..., :-1, 0]
        dy = positions[..., 1:, 1] - positions[..., :-1, 1]

        # Heading from velocity
        heading = torch.atan2(dy, dx)  # [..., H-1]

        # Pad to match length
        heading = F.pad(heading, (0, 1), mode="replicate")

        return heading

    def _compute_clearances(
        self,
        ego_xy: torch.Tensor,
        ego_yaw: torch.Tensor,
        ego_size: torch.Tensor,
        agent_xy: torch.Tensor,
        agent_sizes: torch.Tensor,
        agent_valid: torch.Tensor,
    ) -> tuple:
        """
        Compute penetration depth and clearances between ego and agents.

        Uses OBB approximation based on distance and combined sizes.

        Returns:
            Tuple of (penetration, longitudinal_clearance, lateral_clearance)
            Each is [B, M, N, H]
        """
        B, M, H, _ = ego_xy.shape
        N = agent_xy.shape[1]
        device = ego_xy.device

        # Ego half dimensions
        ego_half_l = ego_size[:, 0:1] / 2  # [B, 1]
        ego_half_w = ego_size[:, 1:2] / 2  # [B, 1]

        # Agent half dimensions
        if agent_sizes is not None:
            agent_half_l = agent_sizes[:, :, 0] / 2  # [B, N]
            agent_half_w = agent_sizes[:, :, 1] / 2  # [B, N]
        else:
            # Default agent size
            agent_half_l = torch.full((B, N), 2.0, device=device)
            agent_half_w = torch.full((B, N), 1.0, device=device)

        # Expand for broadcasting
        # ego_xy: [B, M, H, 2] -> [B, M, 1, H, 2]
        ego_xy_exp = ego_xy.unsqueeze(2)
        ego_yaw_exp = ego_yaw.unsqueeze(2)  # [B, M, 1, H]

        # agent_xy: [B, N, H, 2] -> [B, 1, N, H, 2]
        agent_xy_exp = agent_xy.unsqueeze(1)

        # Relative position in global frame
        rel_pos = agent_xy_exp - ego_xy_exp  # [B, M, N, H, 2]

        # Rotate to ego frame
        cos_yaw = torch.cos(-ego_yaw_exp).unsqueeze(-1)  # [B, M, 1, H, 1]
        sin_yaw = torch.sin(-ego_yaw_exp).unsqueeze(-1)

        # Rotation to ego frame
        rel_x = rel_pos[..., 0:1] * cos_yaw - rel_pos[..., 1:2] * sin_yaw
        rel_y = rel_pos[..., 0:1] * sin_yaw + rel_pos[..., 1:2] * cos_yaw

        rel_local = torch.cat([rel_x, rel_y], dim=-1)  # [B, M, N, H, 2]

        # Distance in longitudinal and lateral directions
        long_dist = torch.abs(rel_local[..., 0])  # [B, M, N, H]
        lat_dist = torch.abs(rel_local[..., 1])  # [B, M, N, H]

        # Combined half dimensions
        # ego_half_l: [B, 1] -> [B, 1, 1, 1]
        # agent_half_l: [B, N] -> [B, 1, N, 1]
        combined_half_l = ego_half_l.unsqueeze(-1).unsqueeze(
            -1
        ) + agent_half_l.unsqueeze(1).unsqueeze(-1)
        combined_half_w = ego_half_w.unsqueeze(-1).unsqueeze(
            -1
        ) + agent_half_w.unsqueeze(1).unsqueeze(-1)

        # Clearances (positive = no overlap, negative = overlap)
        long_clearance = long_dist - combined_half_l  # [B, M, N, H]
        lat_clearance = lat_dist - combined_half_w  # [B, M, N, H]

        # Penetration depth (positive = overlap)
        # Use separating axis theorem: overlap if both long and lat overlap
        long_penetration = F.relu(-long_clearance)
        lat_penetration = F.relu(-lat_clearance)

        # Total penetration (both axes must overlap for collision)
        # Use min of penetrations - if either is 0, no collision
        penetration = torch.min(long_penetration, lat_penetration)

        # Convert clearances to positive values (distance to agent)
        long_clearance = F.relu(long_clearance)
        lat_clearance = F.relu(lat_clearance)

        return penetration, long_clearance, lat_clearance


class VRUClearanceProxy(DifferentiableProxy):
    """
    Specialized proxy for VRU (Vulnerable Road User) clearance.

    Provides additional safety margins for pedestrians and cyclists.
    """

    def __init__(
        self,
        min_clearance: float = 2.0,
        softness: float = 3.0,
    ):
        super().__init__()
        self.min_clearance = min_clearance
        self.softness = softness

    @property
    def rule_ids(self) -> List[str]:
        return ["L0.R4"]  # Crosswalk occupancy

    def forward(
        self,
        trajectories: torch.Tensor,
        scene_features: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute VRU clearance costs.

        Args:
            trajectories: [B, M, H, 2+] candidate trajectories
            scene_features: Dict containing:
                - crosswalk_polygons: [B, C, 4, 2] crosswalk corners
                - crosswalk_valid: [B, C]
                - vru_positions: [B, V, H, 2]
                - vru_valid: [B, V, H]

        Returns:
            Dict with rule_id -> [B, M] cost tensors
        """
        B, M, H, _ = trajectories.shape
        device = trajectories.device

        ego_xy = trajectories[..., :2]  # [B, M, H, 2]

        # Check crosswalk occupancy
        crosswalks = scene_features.get("crosswalk_polygons")  # [B, C, 4, 2]

        if crosswalks is None:
            return {"L0.R4": torch.zeros(B, M, device=device)}

        C = crosswalks.shape[1]

        # Compute minimum distance to any crosswalk at each timestep
        # For simplicity, use center of crosswalk
        crosswalk_centers = crosswalks.mean(dim=2)  # [B, C, 2]

        # Expand for comparison
        ego_exp = ego_xy.unsqueeze(2)  # [B, M, 1, H, 2]
        cw_exp = crosswalk_centers.unsqueeze(1).unsqueeze(3)  # [B, 1, C, 1, 2]

        # Distance to each crosswalk
        dist = torch.norm(ego_exp - cw_exp, dim=-1)  # [B, M, C, H]

        # Check if ego is in crosswalk (simplified: within threshold of center)
        crosswalk_radius = 5.0  # Approximate crosswalk half-width
        in_crosswalk = dist < crosswalk_radius  # [B, M, C, H]

        # Get VRU positions
        vru_pos = scene_features.get("vru_positions")  # [B, V, H, 2]

        if vru_pos is not None:
            V = vru_pos.shape[1]

            # Expand ego for VRU comparison
            ego_vru = ego_xy.unsqueeze(2)  # [B, M, 1, H, 2]
            vru_exp = vru_pos.unsqueeze(1)  # [B, 1, V, H, 2]

            # Distance to VRUs
            vru_dist = torch.norm(ego_vru - vru_exp, dim=-1)  # [B, M, V, H]

            # Clearance violation (positive if too close)
            clearance_violation = self.min_clearance - vru_dist

            # Only count when in crosswalk
            # Combine crosswalk and VRU info
            in_cw_any = in_crosswalk.any(dim=2, keepdim=True)  # [B, M, 1, H]
            clearance_violation = clearance_violation * in_cw_any.float()

            # Max violation across VRUs and time
            max_violation = clearance_violation.max(dim=-1)[0].max(dim=-1)[0]  # [B, M]
            cost = exponential_cost(max_violation, self.softness)
        else:
            # No VRUs - check only crosswalk entry
            in_cw_any = in_crosswalk.any(dim=2).any(dim=-1).float()  # [B, M]
            cost = in_cw_any * 0.1  # Small cost for entering crosswalk

        return {"L0.R4": cost}
