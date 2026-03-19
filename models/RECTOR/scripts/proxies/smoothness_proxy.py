"""
Smoothness violation proxy.

Covers:
- L1.R1: Acceleration limit
- L1.R2: Braking limit
- L1.R3: Steering rate limit
- L1.R4: Speed consistency
- L1.R5: Lane change smoothness (jerk limit)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List
from .base import DifferentiableProxy, exponential_cost, soft_threshold


class SmoothnessProxy(DifferentiableProxy):
    """
    Differentiable smoothness constraint proxy.

    Computes kinematic violations from trajectory data:
    - Acceleration exceeding comfortable limits
    - Hard braking events
    - Excessive steering rates
    - Speed inconsistency
    - High jerk (acceleration derivative)
    """

    def __init__(
        self,
        dt: float = 0.1,
        max_accel: float = 3.0,  # m/s^2, comfortable acceleration
        max_brake: float = 4.0,  # m/s^2, comfortable braking
        max_steer_rate: float = 0.5,  # rad/s, comfortable steering rate
        max_speed_change: float = 2.0,  # m/s, speed consistency threshold
        max_jerk: float = 2.0,  # m/s^3, jerk limit
        softness: float = 2.0,
    ):
        """
        Initialize smoothness proxy.

        Args:
            dt: Time step in seconds
            max_accel: Maximum comfortable acceleration (m/s^2)
            max_brake: Maximum comfortable braking (m/s^2)
            max_steer_rate: Maximum comfortable steering rate (rad/s)
            max_speed_change: Maximum sudden speed change (m/s)
            max_jerk: Maximum jerk (m/s^3)
            softness: Cost function sharpness
        """
        super().__init__()
        self.dt = dt
        self.max_accel = max_accel
        self.max_brake = max_brake
        self.max_steer_rate = max_steer_rate
        self.max_speed_change = max_speed_change
        self.max_jerk = max_jerk
        self.softness = softness

    @property
    def rule_ids(self) -> List[str]:
        return ["L1.R1", "L1.R2", "L1.R3", "L1.R4", "L1.R5"]

    def forward(
        self,
        trajectories: torch.Tensor,
        scene_features: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute smoothness costs for trajectories.

        Args:
            trajectories: [B, M, H, 2+] candidate trajectories
                         Can be [B, M, H, 2] (x, y only) or
                                [B, M, H, 4] (x, y, yaw, speed)
            scene_features: Dict (unused for smoothness)

        Returns:
            Dict with rule_id -> [B, M] cost tensors
        """
        B, M, H, D = trajectories.shape
        device = trajectories.device

        # Extract positions
        xy = trajectories[..., :2]  # [B, M, H, 2]

        # Compute velocity from positions
        velocity = (xy[:, :, 1:] - xy[:, :, :-1]) / self.dt  # [B, M, H-1, 2]
        speed = torch.norm(velocity, dim=-1)  # [B, M, H-1]

        # Compute acceleration
        if D >= 4:
            # Speed provided directly
            traj_speed = trajectories[..., 3]  # [B, M, H]
            accel = (traj_speed[:, :, 1:] - traj_speed[:, :, :-1]) / self.dt
        else:
            # Compute from positions
            accel = (speed[:, :, 1:] - speed[:, :, :-1]) / self.dt  # [B, M, H-2]

        # Compute heading
        if D >= 3:
            yaw = trajectories[..., 2]  # [B, M, H]
        else:
            # Compute from velocity direction
            dx = xy[:, :, 1:, 0] - xy[:, :, :-1, 0]
            dy = xy[:, :, 1:, 1] - xy[:, :, :-1, 1]
            yaw = torch.atan2(dy, dx)  # [B, M, H-1]
            yaw = F.pad(yaw, (0, 1), mode="replicate")  # [B, M, H]

        # Compute steering rate (yaw rate)
        yaw_diff = yaw[:, :, 1:] - yaw[:, :, :-1]
        # Handle angle wraparound
        yaw_diff = torch.atan2(torch.sin(yaw_diff), torch.cos(yaw_diff))
        steer_rate = yaw_diff / self.dt  # [B, M, H-1]

        # Compute jerk (acceleration derivative)
        if accel.shape[-1] >= 2:
            jerk = (accel[:, :, 1:] - accel[:, :, :-1]) / self.dt  # [B, M, H-3]
        else:
            jerk = torch.zeros(B, M, 1, device=device)

        # L1.R1: Acceleration violation
        # Positive acceleration exceeding max
        accel_violation = F.relu(accel - self.max_accel)
        max_accel_violation = accel_violation.max(dim=-1)[0]  # [B, M]
        l1_r1_cost = exponential_cost(max_accel_violation, self.softness)

        # L1.R2: Braking violation
        # Negative acceleration (braking) exceeding max
        brake_violation = F.relu(-accel - self.max_brake)
        max_brake_violation = brake_violation.max(dim=-1)[0]  # [B, M]
        l1_r2_cost = exponential_cost(max_brake_violation, self.softness)

        # L1.R3: Steering rate violation
        steer_violation = F.relu(torch.abs(steer_rate) - self.max_steer_rate)
        max_steer_violation = steer_violation.max(dim=-1)[0]  # [B, M]
        l1_r3_cost = exponential_cost(max_steer_violation, self.softness)

        # L1.R4: Speed consistency
        # Large sudden speed changes
        if D >= 4:
            speed_for_consistency = trajectories[..., 3]
        else:
            # Pad speed to match trajectory length
            speed_for_consistency = F.pad(speed, (0, 1), mode="replicate")

        speed_diff = torch.abs(
            speed_for_consistency[:, :, 1:] - speed_for_consistency[:, :, :-1]
        )
        speed_change_violation = F.relu(speed_diff - self.max_speed_change)
        max_speed_violation = speed_change_violation.max(dim=-1)[0]  # [B, M]
        l1_r4_cost = exponential_cost(max_speed_violation, self.softness)

        # L1.R5: Jerk limit (lane change smoothness)
        jerk_violation = F.relu(torch.abs(jerk) - self.max_jerk)
        max_jerk_violation = jerk_violation.max(dim=-1)[0]  # [B, M]
        l1_r5_cost = exponential_cost(max_jerk_violation, self.softness)

        return {
            "L1.R1": l1_r1_cost,
            "L1.R2": l1_r2_cost,
            "L1.R3": l1_r3_cost,
            "L1.R4": l1_r4_cost,
            "L1.R5": l1_r5_cost,
        }


class LateralAccelerationProxy(DifferentiableProxy):
    """
    Proxy for lateral acceleration comfort constraints.

    High lateral acceleration indicates uncomfortable turning.
    """

    def __init__(
        self,
        dt: float = 0.1,
        max_lat_accel: float = 2.5,  # m/s^2
        softness: float = 2.0,
    ):
        super().__init__()
        self.dt = dt
        self.max_lat_accel = max_lat_accel
        self.softness = softness

    @property
    def rule_ids(self) -> List[str]:
        # This contributes to overall comfort but maps to smoothness rules
        return []  # No direct rule mapping, used as auxiliary

    def forward(
        self,
        trajectories: torch.Tensor,
        scene_features: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute lateral acceleration cost.

        Lateral acceleration = speed * yaw_rate
        """
        B, M, H, D = trajectories.shape
        device = trajectories.device

        xy = trajectories[..., :2]

        # Compute speed
        velocity = (xy[:, :, 1:] - xy[:, :, :-1]) / self.dt
        speed = torch.norm(velocity, dim=-1)  # [B, M, H-1]

        # Compute yaw rate
        if D >= 3:
            yaw = trajectories[..., 2]
        else:
            dx = xy[:, :, 1:, 0] - xy[:, :, :-1, 0]
            dy = xy[:, :, 1:, 1] - xy[:, :, :-1, 1]
            yaw = torch.atan2(dy, dx)
            yaw = F.pad(yaw, (0, 1), mode="replicate")

        yaw_diff = yaw[:, :, 1:] - yaw[:, :, :-1]
        yaw_diff = torch.atan2(torch.sin(yaw_diff), torch.cos(yaw_diff))
        yaw_rate = yaw_diff / self.dt  # [B, M, H-1]

        # Lateral acceleration
        lat_accel = speed * torch.abs(yaw_rate)  # [B, M, H-1]

        # Violation
        lat_accel_violation = F.relu(lat_accel - self.max_lat_accel)
        max_violation = lat_accel_violation.max(dim=-1)[0]  # [B, M]

        cost = exponential_cost(max_violation, self.softness)

        return {"lateral_acceleration": cost}
