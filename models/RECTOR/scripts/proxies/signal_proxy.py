"""
Traffic signal and stop sign proxies.

Covers:
- L5.R1: Traffic signal compliance
- L8.R1: Red light violation
- L8.R2: Stop sign violation
- L8.R3: Crosswalk yield
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
from .base import DifferentiableProxy, exponential_cost, soft_threshold


class SignalProxy(DifferentiableProxy):
    """
    Differentiable traffic signal compliance proxy.

    Computes violations for:
    - Running red lights (solid and arrow)
    - Running flashing red/yellow
    - Not stopping at stop signs
    - Not yielding at crosswalks
    - Protected turn violations
    """

    # Signal state constants (matching Waymo dataset)
    SIGNAL_UNKNOWN = 0
    SIGNAL_RED = 1
    SIGNAL_YELLOW = 2
    SIGNAL_GREEN = 3
    SIGNAL_ARROW_RED = 4
    SIGNAL_ARROW_YELLOW = 5
    SIGNAL_ARROW_GREEN = 6
    SIGNAL_FLASHING_RED = 7
    SIGNAL_FLASHING_YELLOW = 8

    def __init__(
        self,
        dt: float = 0.1,
        stopline_threshold: float = 2.0,  # Distance to consider at stopline
        stop_speed_threshold: float = 0.5,  # Speed to consider stopped
        yellow_penalty_factor: float = 0.3,  # Reduced penalty for yellow
        flashing_penalty_factor: float = 0.5,  # Penalty for flashing violations
        softness: float = 3.0,
    ):
        """
        Initialize signal proxy.

        Args:
            dt: Time step in seconds
            stopline_threshold: Distance threshold for stopline detection
            stop_speed_threshold: Speed below which vehicle is considered stopped
            yellow_penalty_factor: Penalty multiplier for yellow light violations
            flashing_penalty_factor: Penalty for flashing signal violations
            softness: Cost function sharpness
        """
        super().__init__()
        self.dt = dt
        self.stopline_threshold = stopline_threshold
        self.stop_speed_threshold = stop_speed_threshold
        self.yellow_penalty_factor = yellow_penalty_factor
        self.flashing_penalty_factor = flashing_penalty_factor
        self.softness = softness

    @property
    def rule_ids(self) -> List[str]:
        return ["L5.R1", "L8.R1", "L8.R2", "L8.R3"]

    def forward(
        self,
        trajectories: torch.Tensor,
        scene_features: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute signal violation costs.

        Args:
            trajectories: [B, M, H, 2+] candidate trajectories
            scene_features: Dict containing:
                - stoplines: [B, S, 2] stopline center positions
                - stopline_headings: [B, S] stopline orientations
                - signal_states: [B, S, H] signal state per stopline per timestep
                - stop_signs: [B, SS, 2] stop sign positions
                - crosswalk_polygons: [B, C, 4, 2] crosswalk corners

        Returns:
            Dict with rule_id -> [B, M] cost tensors
        """
        B, M, H, D = trajectories.shape
        device = trajectories.device

        ego_xy = trajectories[..., :2]  # [B, M, H, 2]

        # Compute speed
        if D >= 4:
            speed = trajectories[..., 3]  # [B, M, H]
        else:
            velocity = (ego_xy[:, :, 1:] - ego_xy[:, :, :-1]) / self.dt
            speed = torch.norm(velocity, dim=-1)
            speed = F.pad(speed, (0, 1), mode="replicate")

        # Initialize costs
        zero_cost = torch.zeros(B, M, device=device)

        # L8.R1: Red light violation
        stoplines = scene_features.get("stoplines")
        signal_states = scene_features.get("signal_states")

        if stoplines is not None and signal_states is not None:
            l8_r1_cost = self._compute_red_light_cost(
                ego_xy, speed, stoplines, signal_states
            )
        else:
            l8_r1_cost = zero_cost.clone()

        # L8.R2: Stop sign violation
        stop_signs = scene_features.get("stop_signs")

        if stop_signs is not None:
            l8_r2_cost = self._compute_stop_sign_cost(ego_xy, speed, stop_signs)
        else:
            l8_r2_cost = zero_cost.clone()

        # L8.R3: Crosswalk yield
        crosswalks = scene_features.get("crosswalk_polygons")
        vru_positions = scene_features.get("vru_positions")

        if crosswalks is not None:
            l8_r3_cost = self._compute_crosswalk_yield_cost(
                ego_xy, speed, crosswalks, vru_positions
            )
        else:
            l8_r3_cost = zero_cost.clone()

        # L5.R1: General traffic signal compliance
        # Combination of red light and yield violations
        l5_r1_cost = torch.max(l8_r1_cost, l8_r3_cost * 0.5)

        return {
            "L5.R1": l5_r1_cost,
            "L8.R1": l8_r1_cost,
            "L8.R2": l8_r2_cost,
            "L8.R3": l8_r3_cost,
        }

    def _compute_red_light_cost(
        self,
        ego_xy: torch.Tensor,
        speed: torch.Tensor,
        stoplines: torch.Tensor,
        signal_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute red light violation cost.

        Handles all signal types:
        - Solid red: Full violation if crossing without stop
        - Arrow red: Full violation if turning against arrow
        - Flashing red: Violation if not stopping
        - Solid yellow: Reduced penalty
        - Arrow yellow: Reduced penalty for turning
        - Flashing yellow: Caution only, minimal penalty

        Violation if:
        1. Signal is red/flashing red
        2. Ego crosses the stopline
        3. Ego does not stop
        """
        B, M, H, _ = ego_xy.shape
        S = stoplines.shape[1]
        device = ego_xy.device

        # Distance to each stopline: [B, M, H, S]
        ego_exp = ego_xy.unsqueeze(-2)  # [B, M, H, 1, 2]
        stop_exp = stoplines.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, S, 2]

        dist_to_stop = torch.norm(ego_exp - stop_exp, dim=-1)  # [B, M, H, S]

        # Check if crossing stopline (distance < threshold)
        at_stopline = dist_to_stop < self.stopline_threshold  # [B, M, H, S]

        # Check signal state
        # signal_states: [B, S, H] -> [B, 1, H, S]
        states = signal_states.permute(0, 2, 1).unsqueeze(1)

        # Different signal types with different penalties
        is_red = (states == self.SIGNAL_RED).float()
        is_yellow = (states == self.SIGNAL_YELLOW).float()
        is_arrow_red = (states == self.SIGNAL_ARROW_RED).float()
        is_arrow_yellow = (states == self.SIGNAL_ARROW_YELLOW).float()
        is_flashing_red = (states == self.SIGNAL_FLASHING_RED).float()
        is_flashing_yellow = (states == self.SIGNAL_FLASHING_YELLOW).float()

        # Violation: at stopline while red/yellow and not stopped
        not_stopped = (speed > self.stop_speed_threshold).unsqueeze(-1)  # [B, M, H, 1]

        # Full penalty for solid red and arrow red
        red_violation = at_stopline.float() * (is_red + is_arrow_red) * not_stopped

        # Flashing red: must stop but can proceed (like stop sign)
        flashing_red_violation = (
            at_stopline.float()
            * is_flashing_red
            * not_stopped
            * self.flashing_penalty_factor
        )

        # Yellow: reduced penalty
        yellow_violation = (
            at_stopline.float()
            * (is_yellow + is_arrow_yellow)
            * not_stopped
            * self.yellow_penalty_factor
        )

        # Flashing yellow: just caution, minimal penalty
        flashing_yellow_violation = (
            at_stopline.float() * is_flashing_yellow * not_stopped * 0.1
        )

        total_violation = (
            red_violation
            + flashing_red_violation
            + yellow_violation
            + flashing_yellow_violation
        )

        # Max violation across stoplines and time
        max_violation = total_violation.max(dim=-1)[0].max(dim=-1)[0]  # [B, M]

        return exponential_cost(max_violation, self.softness)

    def _compute_stop_sign_cost(
        self,
        ego_xy: torch.Tensor,
        speed: torch.Tensor,
        stop_signs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute stop sign violation cost.

        Violation if ego passes stop sign without stopping.
        """
        B, M, H, _ = ego_xy.shape
        SS = stop_signs.shape[1]
        device = ego_xy.device

        # Distance to each stop sign
        ego_exp = ego_xy.unsqueeze(-2)  # [B, M, H, 1, 2]
        sign_exp = stop_signs.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, SS, 2]

        dist_to_sign = torch.norm(ego_exp - sign_exp, dim=-1)  # [B, M, H, SS]

        # Minimum distance to any stop sign
        min_dist = dist_to_sign.min(dim=-1)[0]  # [B, M, H]

        # Track if we approach and pass stop sign
        # Simple heuristic: violation if min distance decreases then increases
        # while speed never goes below threshold

        # Check if minimum speed is above stop threshold
        min_speed = speed.min(dim=-1)[0]  # [B, M]
        did_not_stop = (min_speed > self.stop_speed_threshold).float()

        # Check if we got close to a stop sign
        got_close = (min_dist.min(dim=-1)[0] < self.stopline_threshold * 2).float()

        violation = did_not_stop * got_close

        return exponential_cost(violation, self.softness)

    def _compute_crosswalk_yield_cost(
        self,
        ego_xy: torch.Tensor,
        speed: torch.Tensor,
        crosswalks: torch.Tensor,
        vru_positions: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute crosswalk yield violation cost.

        Violation if ego enters crosswalk when VRU is present without yielding.
        """
        B, M, H, _ = ego_xy.shape
        C = crosswalks.shape[1]
        device = ego_xy.device

        # Crosswalk centers
        cw_centers = crosswalks.mean(dim=2)  # [B, C, 2]

        # Distance to crosswalks
        ego_exp = ego_xy.unsqueeze(-2)  # [B, M, H, 1, 2]
        cw_exp = cw_centers.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, C, 2]

        dist_to_cw = torch.norm(ego_exp - cw_exp, dim=-1)  # [B, M, H, C]

        # Check if in crosswalk (simplified: within radius)
        crosswalk_radius = 5.0
        in_crosswalk = (dist_to_cw < crosswalk_radius).any(dim=-1)  # [B, M, H]

        if vru_positions is None:
            # No VRUs - small cost for entering crosswalk at speed
            entering_fast = in_crosswalk.float() * (speed > 2.0).float()
            violation = entering_fast.max(dim=-1)[0]  # [B, M]
            return exponential_cost(violation * 0.3, self.softness)

        V = vru_positions.shape[1]

        # Check VRU proximity to crosswalks
        vru_exp = vru_positions.unsqueeze(-2)  # [B, V, H, 1, 2]
        cw_for_vru = cw_centers.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, C, 2]

        vru_to_cw = torch.norm(vru_exp - cw_for_vru, dim=-1)  # [B, V, H, C]
        vru_in_cw = (vru_to_cw < crosswalk_radius).any(dim=-1)  # [B, V, H]

        # Any VRU in any crosswalk at each timestep
        any_vru_in_cw = vru_in_cw.any(dim=1)  # [B, H]

        # Violation: in crosswalk + VRU in crosswalk + not yielding (high speed)
        not_yielding = speed > self.stop_speed_threshold * 2  # [B, M, H]

        violation = (
            in_crosswalk.float()
            * any_vru_in_cw.unsqueeze(1).float()
            * not_yielding.float()
        )

        max_violation = violation.max(dim=-1)[0]  # [B, M]

        return exponential_cost(max_violation, self.softness)
