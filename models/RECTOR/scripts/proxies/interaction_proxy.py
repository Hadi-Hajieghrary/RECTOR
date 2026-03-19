"""
Interaction proxies for agent interactions.

Covers:
- L6.R1: Cooperative lane change
- L6.R2: Following distance
- L6.R3: Intersection negotiation
- L6.R4: Pedestrian interaction
- L6.R5: Cyclist interaction
- L4.R3: Left turn gap
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
from .base import (
    DifferentiableProxy,
    exponential_cost,
    compute_time_to_collision,
    smooth_min,
)


class InteractionProxy(DifferentiableProxy):
    """
    Differentiable proxy for agent interaction rules.

    Computes violations related to:
    - Following too closely
    - Unsafe interactions with VRUs
    - Gap acceptance for turns
    """

    def __init__(
        self,
        dt: float = 0.1,
        min_following_time: float = 2.0,  # seconds
        min_vru_distance: float = 3.0,  # meters
        min_ttc: float = 3.0,  # seconds
        softness: float = 2.0,
    ):
        """
        Initialize interaction proxy.

        Args:
            dt: Time step in seconds
            min_following_time: Minimum time headway for following
            min_vru_distance: Minimum distance to VRUs
            min_ttc: Minimum time-to-collision threshold
            softness: Cost function sharpness
        """
        super().__init__()
        self.dt = dt
        self.min_following_time = min_following_time
        self.min_vru_distance = min_vru_distance
        self.min_ttc = min_ttc
        self.softness = softness

    @property
    def rule_ids(self) -> List[str]:
        return ["L4.R3", "L6.R1", "L6.R2", "L6.R3", "L6.R4", "L6.R5"]

    def forward(
        self,
        trajectories: torch.Tensor,
        scene_features: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute interaction costs.

        Args:
            trajectories: [B, M, H, 2+] candidate trajectories
            scene_features: Dict containing:
                - agent_positions: [B, N, H, 2]
                - agent_velocities: [B, N, H, 2]
                - agent_types: [B, N] (0=vehicle, 1=pedestrian, 2=cyclist)
                - agent_valid: [B, N, H]
                - is_turning_left: [B] boolean
                - in_intersection: [B] boolean

        Returns:
            Dict with rule_id -> [B, M] cost tensors
        """
        B, M, H, D = trajectories.shape
        device = trajectories.device

        ego_xy = trajectories[..., :2]  # [B, M, H, 2]

        # Compute ego velocity
        if D >= 4:
            ego_speed = trajectories[..., 3]  # [B, M, H]
            if D >= 3:
                ego_yaw = trajectories[..., 2]
                ego_vel = torch.stack(
                    [
                        ego_speed * torch.cos(ego_yaw),
                        ego_speed * torch.sin(ego_yaw),
                    ],
                    dim=-1,
                )  # [B, M, H, 2]
            else:
                ego_vel = (ego_xy[:, :, 1:] - ego_xy[:, :, :-1]) / self.dt
                ego_vel = F.pad(ego_vel, (0, 0, 0, 1), mode="replicate")
        else:
            ego_vel = (ego_xy[:, :, 1:] - ego_xy[:, :, :-1]) / self.dt
            ego_vel = F.pad(ego_vel, (0, 0, 0, 1), mode="replicate")
            ego_speed = torch.norm(ego_vel, dim=-1)

        # Initialize costs
        zero_cost = torch.zeros(B, M, device=device)

        # Get agent info
        agent_pos = scene_features.get("agent_positions")  # [B, N, H, 2]
        agent_vel = scene_features.get("agent_velocities")  # [B, N, H, 2]
        agent_types = scene_features.get("agent_types")  # [B, N]
        agent_valid = scene_features.get("agent_valid")  # [B, N, H]

        if agent_pos is None:
            return {rid: zero_cost.clone() for rid in self.rule_ids}

        N = agent_pos.shape[1]

        if agent_vel is None:
            agent_vel = (agent_pos[:, :, 1:] - agent_pos[:, :, :-1]) / self.dt
            agent_vel = F.pad(agent_vel, (0, 0, 0, 1), mode="replicate")

        if agent_valid is None:
            agent_valid = torch.ones(B, N, H, dtype=torch.bool, device=device)

        # Create type masks
        if agent_types is not None:
            vehicle_mask = agent_types == 0  # [B, N]
            pedestrian_mask = agent_types == 1
            cyclist_mask = agent_types == 2
        else:
            vehicle_mask = torch.ones(B, N, dtype=torch.bool, device=device)
            pedestrian_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
            cyclist_mask = torch.zeros(B, N, dtype=torch.bool, device=device)

        # L6.R2: Following distance
        l6_r2_cost = self._compute_following_distance_cost(
            ego_xy, ego_speed, agent_pos, agent_vel, vehicle_mask, agent_valid
        )

        # L6.R4: Pedestrian interaction
        l6_r4_cost = self._compute_vru_interaction_cost(
            ego_xy, ego_vel, agent_pos, agent_vel, pedestrian_mask, agent_valid
        )

        # L6.R5: Cyclist interaction
        l6_r5_cost = self._compute_vru_interaction_cost(
            ego_xy, ego_vel, agent_pos, agent_vel, cyclist_mask, agent_valid
        )

        # L4.R3: Left turn gap
        is_turning = scene_features.get(
            "is_turning_left", torch.zeros(B, dtype=torch.bool, device=device)
        )
        l4_r3_cost = self._compute_left_turn_gap_cost(
            ego_xy, ego_vel, agent_pos, agent_vel, vehicle_mask, agent_valid, is_turning
        )

        # L6.R1: Cooperative lane change
        # Simplified: check for cutting off other vehicles
        l6_r1_cost = self._compute_lane_change_cost(
            ego_xy, ego_vel, agent_pos, agent_vel, vehicle_mask, agent_valid
        )

        # L6.R3: Intersection negotiation
        in_intersection = scene_features.get(
            "in_intersection", torch.zeros(B, dtype=torch.bool, device=device)
        )
        l6_r3_cost = self._compute_intersection_cost(
            ego_xy,
            ego_vel,
            agent_pos,
            agent_vel,
            vehicle_mask,
            agent_valid,
            in_intersection,
        )

        return {
            "L4.R3": l4_r3_cost,
            "L6.R1": l6_r1_cost,
            "L6.R2": l6_r2_cost,
            "L6.R3": l6_r3_cost,
            "L6.R4": l6_r4_cost,
            "L6.R5": l6_r5_cost,
        }

    def _compute_following_distance_cost(
        self,
        ego_xy: torch.Tensor,
        ego_speed: torch.Tensor,
        agent_pos: torch.Tensor,
        agent_vel: torch.Tensor,
        vehicle_mask: torch.Tensor,
        agent_valid: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute following distance violation cost.

        Time headway = distance / speed should be > min_following_time
        """
        B, M, H, _ = ego_xy.shape
        N = agent_pos.shape[1]
        device = ego_xy.device

        # Expand for pairwise comparison
        ego_exp = ego_xy.unsqueeze(2)  # [B, M, 1, H, 2]
        agent_exp = agent_pos.unsqueeze(1)  # [B, 1, N, H, 2]

        # Distance
        dist = torch.norm(ego_exp - agent_exp, dim=-1)  # [B, M, N, H]

        # Time headway
        time_headway = dist / (ego_speed.unsqueeze(2) + 0.1)  # [B, M, N, H]

        # Mask for vehicles ahead
        # Simplified: any vehicle within certain distance
        is_ahead = dist < 50.0  # [B, M, N, H]

        # Apply masks
        mask = (
            vehicle_mask.unsqueeze(1).unsqueeze(-1)
            * agent_valid.unsqueeze(1)
            * is_ahead
        )

        # Following violation: time headway < minimum
        violation = F.relu(self.min_following_time - time_headway) * mask.float()

        # Max violation
        max_violation = violation.max(dim=-1)[0].max(dim=-1)[0]  # [B, M]

        return exponential_cost(max_violation, self.softness)

    def _compute_vru_interaction_cost(
        self,
        ego_xy: torch.Tensor,
        ego_vel: torch.Tensor,
        agent_pos: torch.Tensor,
        agent_vel: torch.Tensor,
        vru_mask: torch.Tensor,
        agent_valid: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute VRU interaction cost.

        Penalize getting too close to VRUs or having low TTC.
        """
        B, M, H, _ = ego_xy.shape
        N = agent_pos.shape[1]
        device = ego_xy.device

        if not vru_mask.any():
            return torch.zeros(B, M, device=device)

        # Distance to VRUs
        ego_exp = ego_xy.unsqueeze(2)  # [B, M, 1, H, 2]
        agent_exp = agent_pos.unsqueeze(1)  # [B, 1, N, H, 2]

        dist = torch.norm(ego_exp - agent_exp, dim=-1)  # [B, M, N, H]

        # Apply VRU mask
        mask = vru_mask.unsqueeze(1).unsqueeze(-1) * agent_valid.unsqueeze(1)

        # Distance violation
        dist_violation = F.relu(self.min_vru_distance - dist) * mask.float()

        # TTC violation
        ego_vel_exp = ego_vel.unsqueeze(2)  # [B, M, 1, H, 2]
        agent_vel_exp = agent_vel.unsqueeze(1)  # [B, 1, N, H, 2]

        ttc = compute_time_to_collision(
            ego_exp.squeeze(-2),
            ego_vel_exp.squeeze(-2),
            agent_exp.squeeze(-2),
            agent_vel_exp.squeeze(-2),
            min_dist=1.0,
        )  # [B, M, N, H]

        ttc_violation = F.relu(self.min_ttc - ttc) * mask.float()

        # Combine violations
        total_violation = dist_violation + ttc_violation * 0.5
        max_violation = total_violation.max(dim=-1)[0].max(dim=-1)[0]  # [B, M]

        return exponential_cost(max_violation, self.softness)

    def _compute_left_turn_gap_cost(
        self,
        ego_xy: torch.Tensor,
        ego_vel: torch.Tensor,
        agent_pos: torch.Tensor,
        agent_vel: torch.Tensor,
        vehicle_mask: torch.Tensor,
        agent_valid: torch.Tensor,
        is_turning: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute left turn gap acceptance cost.

        When turning left, need adequate gap in oncoming traffic.
        """
        B, M, H, _ = ego_xy.shape
        device = ego_xy.device

        if not is_turning.any():
            return torch.zeros(B, M, device=device)

        # Compute TTC with oncoming vehicles
        ego_exp = ego_xy.unsqueeze(2)
        agent_exp = agent_pos.unsqueeze(1)
        ego_vel_exp = ego_vel.unsqueeze(2)
        agent_vel_exp = agent_vel.unsqueeze(1)

        ttc = compute_time_to_collision(
            ego_exp.squeeze(-2),
            ego_vel_exp.squeeze(-2),
            agent_exp.squeeze(-2),
            agent_vel_exp.squeeze(-2),
            min_dist=2.0,
        )  # [B, M, N, H]

        # Mask for valid vehicles
        mask = vehicle_mask.unsqueeze(1).unsqueeze(-1) * agent_valid.unsqueeze(1)

        # Minimum TTC when turning
        min_gap = 4.0  # seconds needed for safe left turn
        gap_violation = F.relu(min_gap - ttc) * mask.float()

        max_violation = gap_violation.max(dim=-1)[0].max(dim=-1)[0]  # [B, M]

        # Only apply when turning
        max_violation = max_violation * is_turning.unsqueeze(-1).float()

        return exponential_cost(max_violation, self.softness)

    def _compute_lane_change_cost(
        self,
        ego_xy: torch.Tensor,
        ego_vel: torch.Tensor,
        agent_pos: torch.Tensor,
        agent_vel: torch.Tensor,
        vehicle_mask: torch.Tensor,
        agent_valid: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute cooperative lane change cost.

        Penalize cutting off other vehicles (forcing them to brake).
        """
        B, M, H, _ = ego_xy.shape
        N = agent_pos.shape[1]
        device = ego_xy.device

        # Check if ego is changing lanes (lateral movement)
        lateral_movement = torch.abs(ego_xy[:, :, -1, 1] - ego_xy[:, :, 0, 1])
        lane_change_threshold = 1.5  # meters
        is_lane_changing = lateral_movement > lane_change_threshold

        if not is_lane_changing.any():
            return torch.zeros(B, M, device=device)

        # Check for vehicles we might cut off
        ego_exp = ego_xy.unsqueeze(2)
        agent_exp = agent_pos.unsqueeze(1)

        dist = torch.norm(ego_exp - agent_exp, dim=-1)  # [B, M, N, H]

        # TTC
        ego_vel_exp = ego_vel.unsqueeze(2)
        agent_vel_exp = agent_vel.unsqueeze(1)

        ttc = compute_time_to_collision(
            ego_exp.squeeze(-2),
            ego_vel_exp.squeeze(-2),
            agent_exp.squeeze(-2),
            agent_vel_exp.squeeze(-2),
            min_dist=3.0,
        )

        mask = vehicle_mask.unsqueeze(1).unsqueeze(-1) * agent_valid.unsqueeze(1)

        # Violation: low TTC during lane change
        min_lc_ttc = 3.0  # seconds
        violation = F.relu(min_lc_ttc - ttc) * mask.float()
        max_violation = violation.max(dim=-1)[0].max(dim=-1)[0]

        # Only apply during lane change
        max_violation = max_violation * is_lane_changing.float()

        return exponential_cost(max_violation, self.softness)

    def _compute_intersection_cost(
        self,
        ego_xy: torch.Tensor,
        ego_vel: torch.Tensor,
        agent_pos: torch.Tensor,
        agent_vel: torch.Tensor,
        vehicle_mask: torch.Tensor,
        agent_valid: torch.Tensor,
        in_intersection: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute intersection negotiation cost.

        When in intersection, need adequate clearance from crossing traffic.
        """
        B, M, H, _ = ego_xy.shape
        device = ego_xy.device

        if not in_intersection.any():
            return torch.zeros(B, M, device=device)

        # Similar to left turn gap but for all intersection traffic
        ego_exp = ego_xy.unsqueeze(2)
        agent_exp = agent_pos.unsqueeze(1)
        ego_vel_exp = ego_vel.unsqueeze(2)
        agent_vel_exp = agent_vel.unsqueeze(1)

        ttc = compute_time_to_collision(
            ego_exp.squeeze(-2),
            ego_vel_exp.squeeze(-2),
            agent_exp.squeeze(-2),
            agent_vel_exp.squeeze(-2),
            min_dist=2.0,
        )

        mask = vehicle_mask.unsqueeze(1).unsqueeze(-1) * agent_valid.unsqueeze(1)

        min_int_ttc = 3.0
        violation = F.relu(min_int_ttc - ttc) * mask.float()
        max_violation = violation.max(dim=-1)[0].max(dim=-1)[0]

        max_violation = max_violation * in_intersection.unsqueeze(-1).float()

        return exponential_cost(max_violation, self.softness)
