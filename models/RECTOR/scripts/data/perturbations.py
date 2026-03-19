"""
Trajectory perturbation strategies for data augmentation.

These perturbations are designed to create trajectory variants with
known violations for contrastive learning.
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional, List
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class PerturbationResult:
    """Result of trajectory perturbation."""

    trajectory: np.ndarray  # [H, 2] or [H, 4] perturbed trajectory
    perturbation_type: str  # Type of perturbation applied
    expected_violations: List[str]  # Rule IDs expected to be violated
    params: Dict[str, Any]  # Perturbation parameters used


class TrajectoryPerturbation(ABC):
    """Base class for trajectory perturbations."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of this perturbation type."""
        pass

    @property
    @abstractmethod
    def expected_violations(self) -> List[str]:
        """Rule IDs that this perturbation is expected to violate."""
        pass

    @abstractmethod
    def apply(
        self,
        trajectory: np.ndarray,
        scene_context: Optional[Any] = None,
    ) -> PerturbationResult:
        """
        Apply perturbation to trajectory.

        Args:
            trajectory: [H, 2] or [H, 4] original trajectory
            scene_context: Optional scene information for context-aware perturbations

        Returns:
            PerturbationResult with perturbed trajectory
        """
        pass


class SpeedScaling(TrajectoryPerturbation):
    """
    Scale trajectory speed uniformly.

    Creates speed limit violations (when scaled up) or
    traffic flow violations (when scaled down significantly).
    """

    def __init__(
        self,
        scale_range: Tuple[float, float] = (0.5, 1.5),
        seed: Optional[int] = None,
    ):
        self.scale_range = scale_range
        self.rng = np.random.default_rng(seed)

    @property
    def name(self) -> str:
        return "speed_scaling"

    @property
    def expected_violations(self) -> List[str]:
        return ["L7.R4"]  # Speed limit

    def apply(
        self,
        trajectory: np.ndarray,
        scene_context: Optional[Any] = None,
    ) -> PerturbationResult:
        scale = self.rng.uniform(*self.scale_range)

        # Scale positions relative to start
        start_pos = trajectory[0, :2].copy()
        relative_pos = trajectory[:, :2] - start_pos
        scaled_pos = start_pos + relative_pos * scale

        if trajectory.shape[1] >= 4:
            # Also scale speed if present
            perturbed = trajectory.copy()
            perturbed[:, :2] = scaled_pos
            perturbed[:, 3] *= scale  # Scale speed
        else:
            perturbed = scaled_pos

        return PerturbationResult(
            trajectory=perturbed,
            perturbation_type=self.name,
            expected_violations=self.expected_violations if scale > 1.0 else [],
            params={"scale": scale},
        )


class LateralOffset(TrajectoryPerturbation):
    """
    Add lateral offset to trajectory.

    Creates lane departure violations.
    """

    def __init__(
        self,
        offset_range: Tuple[float, float] = (-2.0, 2.0),  # meters
        seed: Optional[int] = None,
    ):
        self.offset_range = offset_range
        self.rng = np.random.default_rng(seed)

    @property
    def name(self) -> str:
        return "lateral_offset"

    @property
    def expected_violations(self) -> List[str]:
        return ["L7.R3", "L3.R3"]  # Lane departure, drivable surface

    def apply(
        self,
        trajectory: np.ndarray,
        scene_context: Optional[Any] = None,
    ) -> PerturbationResult:
        offset = self.rng.uniform(*self.offset_range)

        # Compute heading from trajectory
        if trajectory.shape[1] >= 3:
            heading = trajectory[:, 2]
        else:
            dx = np.diff(trajectory[:, 0], prepend=trajectory[0, 0])
            dy = np.diff(trajectory[:, 1], prepend=trajectory[0, 1])
            heading = np.arctan2(dy, dx)

        # Perpendicular direction
        perp_x = -np.sin(heading)
        perp_y = np.cos(heading)

        perturbed = trajectory.copy()
        perturbed[:, 0] += offset * perp_x
        perturbed[:, 1] += offset * perp_y

        return PerturbationResult(
            trajectory=perturbed,
            perturbation_type=self.name,
            expected_violations=self.expected_violations if abs(offset) > 1.0 else [],
            params={"offset": offset},
        )


class HardBraking(TrajectoryPerturbation):
    """
    Add hard braking event to trajectory.

    Creates smooth braking violations.
    """

    def __init__(
        self,
        brake_position: float = 0.5,  # Fraction of trajectory
        deceleration: float = 6.0,  # m/s^2
        dt: float = 0.1,
        seed: Optional[int] = None,
    ):
        self.brake_position = brake_position
        self.deceleration = deceleration
        self.dt = dt
        self.rng = np.random.default_rng(seed)

    @property
    def name(self) -> str:
        return "hard_braking"

    @property
    def expected_violations(self) -> List[str]:
        return ["L1.R2", "L1.R5"]  # Smooth braking, jerk

    def apply(
        self,
        trajectory: np.ndarray,
        scene_context: Optional[Any] = None,
    ) -> PerturbationResult:
        H = len(trajectory)
        brake_idx = int(H * self.brake_position)

        # Compute original speed
        if trajectory.shape[1] >= 4:
            speed = trajectory[:, 3].copy()
        else:
            dx = np.diff(trajectory[:, 0])
            dy = np.diff(trajectory[:, 1])
            speed = np.sqrt(dx**2 + dy**2) / self.dt
            speed = np.concatenate([[speed[0]], speed])

        # Apply braking from brake_idx
        new_speed = speed.copy()
        for t in range(brake_idx, H):
            new_speed[t] = max(0, new_speed[t - 1] - self.deceleration * self.dt)

        # Reconstruct trajectory from speed
        if trajectory.shape[1] >= 3:
            heading = trajectory[:, 2]
        else:
            dx = np.diff(trajectory[:, 0], prepend=trajectory[0, 0])
            dy = np.diff(trajectory[:, 1], prepend=trajectory[0, 1])
            heading = np.arctan2(dy, dx)

        # Integrate to get positions
        perturbed = trajectory.copy()
        for t in range(1, H):
            perturbed[t, 0] = (
                perturbed[t - 1, 0] + new_speed[t] * np.cos(heading[t]) * self.dt
            )
            perturbed[t, 1] = (
                perturbed[t - 1, 1] + new_speed[t] * np.sin(heading[t]) * self.dt
            )

        if trajectory.shape[1] >= 4:
            perturbed[:, 3] = new_speed

        return PerturbationResult(
            trajectory=perturbed,
            perturbation_type=self.name,
            expected_violations=self.expected_violations,
            params={
                "brake_position": self.brake_position,
                "deceleration": self.deceleration,
            },
        )


class SharpTurn(TrajectoryPerturbation):
    """
    Add sharp turning maneuver.

    Creates steering rate and lateral acceleration violations.
    """

    def __init__(
        self,
        turn_angle: float = 0.5,  # radians
        turn_position: float = 0.5,
        turn_duration: int = 10,  # timesteps
        dt: float = 0.1,  # seconds per timestep
        seed: Optional[int] = None,
    ):
        self.turn_angle = turn_angle
        self.turn_position = turn_position
        self.turn_duration = turn_duration
        self.dt = dt
        self.rng = np.random.default_rng(seed)

    @property
    def name(self) -> str:
        return "sharp_turn"

    @property
    def expected_violations(self) -> List[str]:
        return ["L1.R3", "L7.R3"]  # Steering rate, lane departure

    def apply(
        self,
        trajectory: np.ndarray,
        scene_context: Optional[Any] = None,
    ) -> PerturbationResult:
        H = len(trajectory)
        turn_start = int(H * self.turn_position)
        turn_end = min(turn_start + self.turn_duration, H)

        # Get or compute heading
        if trajectory.shape[1] >= 3:
            heading = trajectory[:, 2].copy()
        else:
            dx = np.diff(trajectory[:, 0], prepend=trajectory[0, 0])
            dy = np.diff(trajectory[:, 1], prepend=trajectory[0, 1])
            heading = np.arctan2(dy, dx)

        # Add turn
        for t in range(turn_start, turn_end):
            progress = (t - turn_start) / max(1, turn_end - turn_start)
            heading[t:] += self.turn_angle * progress / self.turn_duration

        # Get speed
        if trajectory.shape[1] >= 4:
            speed = trajectory[:, 3]
        else:
            dx = np.diff(trajectory[:, 0])
            dy = np.diff(trajectory[:, 1])
            speed = np.sqrt(dx**2 + dy**2) / self.dt
            speed = np.concatenate([[speed[0]], speed])

        # Reconstruct trajectory
        perturbed = trajectory.copy()
        for t in range(1, H):
            perturbed[t, 0] = (
                perturbed[t - 1, 0] + speed[t] * np.cos(heading[t]) * self.dt
            )
            perturbed[t, 1] = (
                perturbed[t - 1, 1] + speed[t] * np.sin(heading[t]) * self.dt
            )

        if trajectory.shape[1] >= 3:
            perturbed[:, 2] = heading

        return PerturbationResult(
            trajectory=perturbed,
            perturbation_type=self.name,
            expected_violations=self.expected_violations,
            params={
                "turn_angle": self.turn_angle,
                "turn_position": self.turn_position,
            },
        )


class SignalIgnore(TrajectoryPerturbation):
    """
    Extend trajectory through a red light.

    Creates signal compliance violations.
    """

    def __init__(
        self,
        speed_increase: float = 1.5,  # Factor to speed up through light
        seed: Optional[int] = None,
    ):
        self.speed_increase = speed_increase
        self.rng = np.random.default_rng(seed)

    @property
    def name(self) -> str:
        return "signal_ignore"

    @property
    def expected_violations(self) -> List[str]:
        return ["L5.R1", "L8.R1"]  # Signal compliance, red light

    def apply(
        self,
        trajectory: np.ndarray,
        scene_context: Optional[Any] = None,
    ) -> PerturbationResult:
        # Increase speed to run through light
        # This is a simplified version - full implementation would
        # detect signal location and modify trajectory accordingly

        perturbed = trajectory.copy()

        # Scale positions to increase speed
        start_pos = trajectory[0, :2]
        relative_pos = trajectory[:, :2] - start_pos
        perturbed[:, :2] = start_pos + relative_pos * self.speed_increase

        if trajectory.shape[1] >= 4:
            perturbed[:, 3] *= self.speed_increase

        return PerturbationResult(
            trajectory=perturbed,
            perturbation_type=self.name,
            expected_violations=self.expected_violations,
            params={"speed_increase": self.speed_increase},
        )


class GaussianNoise(TrajectoryPerturbation):
    """
    Add Gaussian noise to trajectory.

    General perturbation that may cause various minor violations.
    """

    def __init__(
        self,
        position_std: float = 0.5,  # meters
        growing_std: bool = True,  # Std grows with horizon
        seed: Optional[int] = None,
    ):
        self.position_std = position_std
        self.growing_std = growing_std
        self.rng = np.random.default_rng(seed)

    @property
    def name(self) -> str:
        return "gaussian_noise"

    @property
    def expected_violations(self) -> List[str]:
        return []  # May cause various violations

    def apply(
        self,
        trajectory: np.ndarray,
        scene_context: Optional[Any] = None,
    ) -> PerturbationResult:
        H = len(trajectory)

        if self.growing_std:
            # Std grows linearly with horizon
            std = self.position_std * np.arange(H) / H
            std = std[:, np.newaxis]
        else:
            std = self.position_std

        noise = self.rng.normal(0, std, (H, 2))

        perturbed = trajectory.copy()
        perturbed[:, :2] += noise

        return PerturbationResult(
            trajectory=perturbed,
            perturbation_type=self.name,
            expected_violations=self.expected_violations,
            params={"position_std": self.position_std},
        )


class PerturbationPipeline:
    """
    Applies random perturbations from a set of strategies.
    """

    def __init__(
        self,
        perturbations: Optional[List[TrajectoryPerturbation]] = None,
        seed: Optional[int] = None,
    ):
        if perturbations is None:
            perturbations = [
                SpeedScaling(seed=seed),
                LateralOffset(seed=seed),
                HardBraking(seed=seed),
                SharpTurn(seed=seed),
                GaussianNoise(seed=seed),
            ]

        self.perturbations = perturbations
        self.rng = np.random.default_rng(seed)

    def apply_random(
        self,
        trajectory: np.ndarray,
        scene_context: Optional[Any] = None,
    ) -> PerturbationResult:
        """Apply a random perturbation."""
        idx = self.rng.integers(len(self.perturbations))
        return self.perturbations[idx].apply(trajectory, scene_context)

    def apply_all(
        self,
        trajectory: np.ndarray,
        scene_context: Optional[Any] = None,
    ) -> List[PerturbationResult]:
        """Apply all perturbations and return list of results."""
        results = []
        for pert in self.perturbations:
            results.append(pert.apply(trajectory, scene_context))
        return results
