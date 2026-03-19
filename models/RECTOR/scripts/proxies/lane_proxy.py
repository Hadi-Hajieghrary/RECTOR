"""
Lane and road structure proxies.

Covers:
- L3.R3: Drivable surface (staying on road)
- L7.R3: Lane departure
- L7.R4: Speed limit
- L8.R5: Wrong way driving
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
from .base import DifferentiableProxy, exponential_cost, soft_threshold


class LaneProxy(DifferentiableProxy):
    """
    Differentiable lane-following and road structure proxy.

    Computes lane departure and drivable surface violations by:
    1. Projecting trajectory onto nearest lane centerline
    2. Computing lateral offset from lane center
    3. Checking if trajectory stays within road boundaries
    4. Verifying heading alignment with lane direction
    """

    def __init__(
        self,
        max_lateral_offset: float = 1.8,  # Half lane width
        wrong_way_angle_threshold: float = 2.356,  # 135 degrees in radians
        softness: float = 2.0,
    ):
        """
        Initialize lane proxy.

        Args:
            max_lateral_offset: Maximum allowed lateral offset from lane center
            wrong_way_angle_threshold: Angle threshold for wrong-way detection
            softness: Cost function sharpness
        """
        super().__init__()
        self.max_lateral_offset = max_lateral_offset
        self.wrong_way_angle_threshold = wrong_way_angle_threshold
        self.softness = softness

    @property
    def rule_ids(self) -> List[str]:
        return ["L3.R3", "L7.R3", "L8.R5"]

    def forward(
        self,
        trajectories: torch.Tensor,
        scene_features: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute lane/road costs for trajectories.

        Args:
            trajectories: [B, M, H, 2+] candidate trajectories
            scene_features: Dict containing:
                - lane_centers: [B, L, P, 2] lane centerline points
                - lane_headings: [B, L, P] lane headings at each point
                - lane_valid: [B, L] lane validity mask
                - road_edges: [B, E, 2] road edge points (optional)
                - road_edge_valid: [B, E] edge validity (optional)

        Returns:
            Dict with rule_id -> [B, M] cost tensors
        """
        B, M, H, _ = trajectories.shape
        device = trajectories.device

        ego_xy = trajectories[..., :2]  # [B, M, H, 2]

        # Compute heading from trajectory
        if trajectories.shape[-1] >= 3:
            ego_yaw = trajectories[..., 2]  # [B, M, H]
        else:
            dx = ego_xy[:, :, 1:, 0] - ego_xy[:, :, :-1, 0]
            dy = ego_xy[:, :, 1:, 1] - ego_xy[:, :, :-1, 1]
            ego_yaw = torch.atan2(dy, dx)
            ego_yaw = F.pad(ego_yaw, (0, 1), mode="replicate")

        # Get lane features
        lane_centers = scene_features.get("lane_centers")  # [B, L, P, 2]
        lane_headings = scene_features.get("lane_headings")  # [B, L, P]

        if lane_centers is None:
            # No lane info - zero cost
            zero_cost = torch.zeros(B, M, device=device)
            return {rid: zero_cost.clone() for rid in self.rule_ids}

        L, P = lane_centers.shape[1], lane_centers.shape[2]

        # Compute lateral offset and heading alignment
        lateral_offset, heading_error = self._compute_lane_deviation(
            ego_xy=ego_xy,
            ego_yaw=ego_yaw,
            lane_centers=lane_centers,
            lane_headings=lane_headings,
        )
        # lateral_offset: [B, M, H]
        # heading_error: [B, M, H]

        # L7.R3: Lane departure
        lane_departure = F.relu(torch.abs(lateral_offset) - self.max_lateral_offset)
        max_departure = lane_departure.max(dim=-1)[0]  # [B, M]
        l7_r3_cost = exponential_cost(max_departure, self.softness)

        # L8.R5: Wrong way driving
        # Violation if heading error > threshold
        wrong_way_violation = F.relu(
            torch.abs(heading_error) - self.wrong_way_angle_threshold
        )
        max_wrong_way = wrong_way_violation.max(dim=-1)[0]  # [B, M]
        l8_r5_cost = exponential_cost(max_wrong_way, self.softness)

        # L3.R3: Drivable surface
        road_edges = scene_features.get("road_edges")  # [B, E, 2]

        if road_edges is not None:
            off_road = self._compute_off_road_violation(ego_xy, road_edges)
            max_off_road = off_road.max(dim=-1)[0]  # [B, M]
            l3_r3_cost = exponential_cost(max_off_road, self.softness)
        else:
            # Use large lane departure as proxy for off-road
            off_road_threshold = self.max_lateral_offset * 2
            off_road_violation = F.relu(torch.abs(lateral_offset) - off_road_threshold)
            max_off_road = off_road_violation.max(dim=-1)[0]
            l3_r3_cost = exponential_cost(max_off_road, self.softness)

        return {
            "L3.R3": l3_r3_cost,
            "L7.R3": l7_r3_cost,
            "L8.R5": l8_r5_cost,
        }

    def _compute_lane_deviation(
        self,
        ego_xy: torch.Tensor,
        ego_yaw: torch.Tensor,
        lane_centers: torch.Tensor,
        lane_headings: Optional[torch.Tensor],
    ) -> tuple:
        """
        Compute lateral offset and heading error from nearest lane.

        Uses soft minimum over all lane segments.

        Returns:
            Tuple of (lateral_offset, heading_error) each [B, M, H]
        """
        B, M, H, _ = ego_xy.shape
        L, P = lane_centers.shape[1], lane_centers.shape[2]
        device = ego_xy.device

        # Flatten lanes: [B, L, P, 2] -> [B, L*P, 2]
        lane_points = lane_centers.reshape(B, L * P, 2)

        # Compute distance from each trajectory point to each lane point
        # ego_xy: [B, M, H, 2] -> [B, M, H, 1, 2]
        ego_exp = ego_xy.unsqueeze(-2)
        # lane_points: [B, L*P, 2] -> [B, 1, 1, L*P, 2]
        lane_exp = lane_points.unsqueeze(1).unsqueeze(1)

        # Squared distances: [B, M, H, L*P]
        dist_sq = ((ego_exp - lane_exp) ** 2).sum(dim=-1)

        # Find nearest lane point for each trajectory point
        min_dist_sq, min_idx = dist_sq.min(dim=-1)  # [B, M, H]

        # Lateral offset is sqrt of min distance (clamp before sqrt to avoid
        # gradient issues near zero — adding epsilon inside sqrt does not help)
        lateral_offset = torch.sqrt(min_dist_sq.clamp(min=1e-6))  # [B, M, H]

        # Compute heading error
        if lane_headings is not None:
            # lane_headings: [B, L, P] -> [B, L*P]
            lane_head_flat = lane_headings.reshape(B, L * P)

            # Get heading at nearest point
            # min_idx: [B, M, H] -> indices into L*P
            batch_idx = torch.arange(B, device=device).view(B, 1, 1).expand(B, M, H)
            nearest_heading = lane_head_flat[batch_idx, min_idx]  # [B, M, H]

            # Heading error
            heading_diff = ego_yaw - nearest_heading
            heading_error = torch.atan2(
                torch.sin(heading_diff), torch.cos(heading_diff)
            )
        else:
            # No heading info - compute from lane points
            heading_error = torch.zeros_like(lateral_offset)

        return lateral_offset, heading_error

    def _compute_off_road_violation(
        self,
        ego_xy: torch.Tensor,
        road_edges: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute how far trajectory goes off drivable surface.

        Args:
            ego_xy: [B, M, H, 2] trajectory positions
            road_edges: [B, E, 2] road edge points

        Returns:
            [B, M, H] off-road distance (0 if on road)
        """
        B, M, H, _ = ego_xy.shape
        E = road_edges.shape[1]
        device = ego_xy.device

        if E < 2:
            return torch.zeros(B, M, H, device=device)

        # Compute distance to road edge
        ego_exp = ego_xy.unsqueeze(-2)  # [B, M, H, 1, 2]
        edge_exp = road_edges.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, E, 2]

        dist = torch.norm(ego_exp - edge_exp, dim=-1)  # [B, M, H, E]
        min_dist_to_edge = dist.min(dim=-1)[0]  # [B, M, H]

        # If very close to edge, might be off-road
        # This is a simplified heuristic - proper implementation would
        # check if point is inside road polygon
        edge_threshold = 1.0  # meters
        off_road = F.relu(edge_threshold - min_dist_to_edge)

        return off_road


class SpeedLimitProxy(DifferentiableProxy):
    """
    Speed limit violation proxy.

    Computes cost based on exceeding posted speed limits.
    Supports per-lane speed limits by finding the nearest lane
    and using its speed limit.
    """

    def __init__(
        self,
        dt: float = 0.1,
        default_speed_limit: float = 15.0,  # m/s (~35 mph)
        softness: float = 2.0,
    ):
        super().__init__()
        self.dt = dt
        self.default_speed_limit = default_speed_limit
        self.softness = softness

    @property
    def rule_ids(self) -> List[str]:
        return ["L7.R4"]

    def forward(
        self,
        trajectories: torch.Tensor,
        scene_features: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute speed limit violation cost.

        Args:
            trajectories: [B, M, H, 2+]
            scene_features: Dict containing:
                - speed_limits: [B, H] speed limit at each timestep, OR
                - lane_speed_limits: [B, L] speed limit per lane, with lane_centers

        Returns:
            Dict with rule_id -> [B, M] cost
        """
        B, M, H, D = trajectories.shape
        device = trajectories.device

        # Get speed from trajectory
        if D >= 4:
            speed = trajectories[..., 3]  # [B, M, H]
        else:
            xy = trajectories[..., :2]
            velocity = (xy[:, :, 1:] - xy[:, :, :-1]) / self.dt
            speed = torch.norm(velocity, dim=-1)  # [B, M, H-1]
            speed = F.pad(speed, (0, 1), mode="replicate")  # [B, M, H]

        # Get speed limit - prioritize per-lane if available
        speed_limit = self._get_speed_limits(trajectories, scene_features)

        # Violation: how much over limit
        over_limit = F.relu(speed - speed_limit)  # [B, M, H]
        max_over = over_limit.max(dim=-1)[0]  # [B, M]

        # Normalize by speed limit for relative violation
        if isinstance(speed_limit, float):
            norm_over = max_over / speed_limit
        else:
            mean_limit = (
                speed_limit.mean(dim=-1)
                if speed_limit.dim() > 2
                else speed_limit.squeeze()
            )
            if mean_limit.dim() == 0:
                mean_limit = mean_limit.unsqueeze(0)
            norm_over = max_over / (mean_limit.unsqueeze(-1).expand_as(max_over) + 1e-6)

        cost = exponential_cost(norm_over, self.softness)

        return {"L7.R4": cost}

    def _get_speed_limits(
        self,
        trajectories: torch.Tensor,
        scene_features: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Get speed limits for each trajectory point.

        Supports:
        1. Per-lane speed limits (lane_speed_limits + lane_centers)
        2. Per-timestep speed limits (speed_limits)
        3. Single scenario limit
        4. Default limit fallback

        Args:
            trajectories: [B, M, H, 2+]
            scene_features: Scene feature dict

        Returns:
            Speed limits as [B, M, H] or scalar
        """
        B, M, H, D = trajectories.shape
        device = trajectories.device

        # Option 1: Per-lane speed limits
        lane_speed_limits = scene_features.get("lane_speed_limits")  # [B, L]
        lane_centers = scene_features.get("lane_centers")  # [B, L, P, 2]

        if lane_speed_limits is not None and lane_centers is not None:
            return self._lookup_lane_speed_limits(
                trajectories, lane_centers, lane_speed_limits
            )

        # Option 2: Per-timestep speed limits
        speed_limits = scene_features.get("speed_limits")

        if speed_limits is None:
            return self.default_speed_limit

        if speed_limits.dim() == 1:
            # [B,] single limit per scenario
            return speed_limits.unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]
        elif speed_limits.dim() == 2:
            # [B, H] per-timestep limits
            return speed_limits.unsqueeze(1)  # [B, 1, H]
        else:
            return self.default_speed_limit

    def _lookup_lane_speed_limits(
        self,
        trajectories: torch.Tensor,
        lane_centers: torch.Tensor,
        lane_speed_limits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Look up speed limit based on nearest lane.

        Args:
            trajectories: [B, M, H, 2+]
            lane_centers: [B, L, P, 2]
            lane_speed_limits: [B, L]

        Returns:
            Speed limits: [B, M, H]
        """
        B, M, H, D = trajectories.shape
        L, P = lane_centers.shape[1], lane_centers.shape[2]
        device = trajectories.device

        ego_xy = trajectories[..., :2]  # [B, M, H, 2]

        # Compute centroid of each lane for efficiency
        lane_centroids = lane_centers.mean(dim=2)  # [B, L, 2]

        # Compute distance from each trajectory point to each lane centroid
        # ego_xy: [B, M, H, 2] -> [B, M, H, 1, 2]
        ego_exp = ego_xy.unsqueeze(-2)
        # lane_centroids: [B, L, 2] -> [B, 1, 1, L, 2]
        lane_exp = lane_centroids.unsqueeze(1).unsqueeze(1)

        # Squared distances: [B, M, H, L]
        dist_sq = ((ego_exp - lane_exp) ** 2).sum(dim=-1)

        # Find nearest lane for each trajectory point
        nearest_lane = dist_sq.argmin(dim=-1)  # [B, M, H]

        # Look up speed limit for nearest lane
        # lane_speed_limits: [B, L]
        batch_idx = torch.arange(B, device=device).view(B, 1, 1).expand(B, M, H)
        speed_limits = lane_speed_limits[batch_idx, nearest_lane]  # [B, M, H]

        return speed_limits
