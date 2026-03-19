"""
Comprehensive Test Fixtures for Waymo Rule Evaluation.

Provides realistic, parameterized test scenarios covering:
- Normal driving (no violations)
- Collision scenarios (with vehicles, pedestrians, cyclists)
- Traffic control violations (red lights, stop signs)
- Comfort violations (harsh braking, acceleration)
- Lane violations (departure, wrong-way)
- VRU interaction scenarios

Each fixture is documented with expected rule outcomes.
"""

import math
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Add the workspace root to path so waymo_rule_eval package can be imported
_workspace_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _workspace_root not in sys.path:
    sys.path.insert(0, _workspace_root)


@dataclass
class TrajectoryBuilder:
    """
    Builder for creating realistic vehicle trajectories.

    Supports:
    - Straight line motion
    - Lane changes
    - Turns
    - Acceleration/deceleration profiles
    - Stop and go patterns
    """

    n_frames: int = 91  # 9.1 seconds at 10 Hz
    dt: float = 0.1

    def straight_line(
        self,
        start_x: float,
        start_y: float,
        heading: float,
        speed: float,
        acceleration: float = 0.0,
    ) -> Dict[str, np.ndarray]:
        """
        Create straight-line trajectory with optional acceleration.

        Args:
            start_x, start_y: Starting position
            heading: Direction of travel (radians)
            speed: Initial speed (m/s)
            acceleration: Constant acceleration (m/s²)

        Returns:
            Dict with x, y, yaw, speed arrays
        """
        t = np.arange(self.n_frames) * self.dt

        # Speed profile with acceleration
        speed_arr = np.clip(speed + acceleration * t, 0, 50)

        # Displacement = integral of speed
        displacement = np.zeros(self.n_frames)
        for i in range(1, self.n_frames):
            displacement[i] = displacement[i - 1] + speed_arr[i - 1] * self.dt

        x = start_x + displacement * np.cos(heading)
        y = start_y + displacement * np.sin(heading)
        yaw = np.full(self.n_frames, heading)

        return {"x": x, "y": y, "yaw": yaw, "speed": speed_arr}

    def lane_change(
        self,
        start_x: float,
        start_y: float,
        heading: float,
        speed: float,
        lateral_offset: float = 3.5,
        change_start_frame: int = 30,
        change_duration_frames: int = 30,
    ) -> Dict[str, np.ndarray]:
        """
        Create lane change trajectory with smooth lateral motion.

        Args:
            start_x, start_y: Starting position
            heading: Initial heading (radians)
            speed: Constant speed
            lateral_offset: Lane width to change (positive = left)
            change_start_frame: Frame when lane change begins
            change_duration_frames: Duration of lane change

        Returns:
            Dict with x, y, yaw, speed arrays
        """
        t = np.arange(self.n_frames) * self.dt

        # Longitudinal displacement
        lon_disp = speed * t

        # Lateral displacement (sinusoidal profile)
        lat_disp = np.zeros(self.n_frames)
        for i in range(self.n_frames):
            if i < change_start_frame:
                lat_disp[i] = 0
            elif i < change_start_frame + change_duration_frames:
                # Smooth sinusoidal transition
                phase = (i - change_start_frame) / change_duration_frames
                lat_disp[i] = lateral_offset * (1 - np.cos(phase * np.pi)) / 2
            else:
                lat_disp[i] = lateral_offset

        # Compute heading changes during lane change
        yaw = np.full(self.n_frames, heading, dtype=float)
        for i in range(1, self.n_frames):
            if (
                i >= change_start_frame
                and i < change_start_frame + change_duration_frames
            ):
                # Approximate heading from lateral velocity
                lat_vel = (lat_disp[i] - lat_disp[i - 1]) / self.dt
                yaw[i] = heading + np.arctan2(lat_vel, speed)

        # Transform to world coordinates
        cos_h = np.cos(heading)
        sin_h = np.sin(heading)

        x = start_x + lon_disp * cos_h - lat_disp * sin_h
        y = start_y + lon_disp * sin_h + lat_disp * cos_h
        speed_arr = np.full(self.n_frames, speed)

        return {"x": x, "y": y, "yaw": yaw, "speed": speed_arr}

    def stop_at_position(
        self,
        start_x: float,
        start_y: float,
        heading: float,
        initial_speed: float,
        stop_x: float,
        stop_y: float,
        stop_duration_frames: int = 20,
    ) -> Dict[str, np.ndarray]:
        """
        Create trajectory that stops at a specific position.

        Args:
            start_x, start_y: Starting position
            heading: Direction of travel
            initial_speed: Initial speed
            stop_x, stop_y: Position to stop at
            stop_duration_frames: How long to remain stopped

        Returns:
            Dict with x, y, yaw, speed arrays
        """
        # Calculate distance to stop point
        dx = stop_x - start_x
        dy = stop_y - start_y
        stop_distance = np.sqrt(dx**2 + dy**2)

        # Time to reach stop point at constant deceleration
        # s = v*t - 0.5*a*t^2, v_f = 0 => a = v^2 / (2*s)
        if stop_distance > 0.1:
            decel = initial_speed**2 / (2 * stop_distance)
            time_to_stop = initial_speed / decel
        else:
            decel = 0
            time_to_stop = 0

        frames_to_stop = min(
            int(time_to_stop / self.dt) + 1, self.n_frames - stop_duration_frames
        )

        # Build speed profile
        speed_arr = np.zeros(self.n_frames)
        for i in range(self.n_frames):
            if i < frames_to_stop:
                speed_arr[i] = max(0, initial_speed - decel * i * self.dt)
            else:
                speed_arr[i] = 0

        # Build position from speed
        x = np.zeros(self.n_frames)
        y = np.zeros(self.n_frames)
        x[0], y[0] = start_x, start_y

        for i in range(1, self.n_frames):
            x[i] = x[i - 1] + speed_arr[i - 1] * np.cos(heading) * self.dt
            y[i] = y[i - 1] + speed_arr[i - 1] * np.sin(heading) * self.dt

        yaw = np.full(self.n_frames, heading)

        return {"x": x, "y": y, "yaw": yaw, "speed": speed_arr}

    def left_turn(
        self,
        start_x: float,
        start_y: float,
        start_heading: float,
        speed: float,
        turn_radius: float = 10.0,
        turn_start_frame: int = 30,
    ) -> Dict[str, np.ndarray]:
        """
        Create left turn trajectory.

        Args:
            start_x, start_y: Starting position
            start_heading: Initial heading
            speed: Speed during turn
            turn_radius: Radius of turn
            turn_start_frame: Frame when turn begins

        Returns:
            Dict with x, y, yaw, speed arrays
        """
        x = np.zeros(self.n_frames)
        y = np.zeros(self.n_frames)
        yaw = np.zeros(self.n_frames)
        speed_arr = np.full(self.n_frames, speed)

        x[0], y[0] = start_x, start_y
        yaw[0] = start_heading

        for i in range(1, self.n_frames):
            if i < turn_start_frame:
                # Straight approach
                yaw[i] = start_heading
            else:
                # During turn, yaw changes
                angular_velocity = speed / turn_radius  # rad/s
                yaw[i] = yaw[i - 1] + angular_velocity * self.dt

            # Update position
            x[i] = x[i - 1] + speed * np.cos(yaw[i - 1]) * self.dt
            y[i] = y[i - 1] + speed * np.sin(yaw[i - 1]) * self.dt

        return {"x": x, "y": y, "yaw": yaw, "speed": speed_arr}

    def harsh_braking(
        self,
        start_x: float,
        start_y: float,
        heading: float,
        initial_speed: float,
        brake_start_frame: int = 30,
        deceleration: float = -8.0,  # Harsh braking
    ) -> Dict[str, np.ndarray]:
        """
        Create harsh braking trajectory.

        Args:
            start_x, start_y: Starting position
            heading: Direction of travel
            initial_speed: Speed before braking
            brake_start_frame: Frame when braking begins
            deceleration: Braking deceleration (negative, m/s²)

        Returns:
            Dict with x, y, yaw, speed arrays
        """
        speed_arr = np.zeros(self.n_frames)

        for i in range(self.n_frames):
            if i < brake_start_frame:
                speed_arr[i] = initial_speed
            else:
                t_brake = (i - brake_start_frame) * self.dt
                speed_arr[i] = max(0, initial_speed + deceleration * t_brake)

        # Build position from speed
        x = np.zeros(self.n_frames)
        y = np.zeros(self.n_frames)
        x[0], y[0] = start_x, start_y

        for i in range(1, self.n_frames):
            x[i] = x[i - 1] + speed_arr[i - 1] * np.cos(heading) * self.dt
            y[i] = y[i - 1] + speed_arr[i - 1] * np.sin(heading) * self.dt

        yaw = np.full(self.n_frames, heading)

        return {"x": x, "y": y, "yaw": yaw, "speed": speed_arr}


@dataclass
class MapBuilder:
    """
    Builder for creating map contexts with various features.

    Supports:
    - Lane centerlines
    - Stop lines
    - Crosswalks
    - Speed limits
    - Traffic signals
    """

    def simple_straight_road(
        self,
        length: float = 100.0,
        lane_width: float = 3.7,
        n_lanes: int = 2,
        speed_limit_mps: float = 13.4,  # ~30 mph
    ) -> Dict[str, Any]:
        """
        Create simple straight road map.

        Args:
            length: Road length in meters
            lane_width: Lane width in meters
            n_lanes: Number of lanes
            speed_limit_mps: Speed limit in m/s

        Returns:
            Dict with map features
        """
        # Centerline points (100 points along road)
        n_points = 100
        x = np.linspace(0, length, n_points)
        y = np.zeros(n_points)

        lane_center_xy = np.column_stack([x, y])

        return {
            "lane_center_xy": lane_center_xy,
            "lane_id": 0,
            "speed_limit": np.full(n_points, speed_limit_mps),
            "speed_limit_mask": np.ones(n_points, dtype=bool),
            "crosswalk_polys": [],
            "road_edges": [],
            "stop_signs": [],
        }

    def intersection_with_stopline(
        self,
        center_x: float = 50.0,
        center_y: float = 0.0,
        approach_length: float = 50.0,
        stopline_offset: float = 5.0,
    ) -> Dict[str, Any]:
        """
        Create intersection map with stop line.

        Args:
            center_x, center_y: Intersection center
            approach_length: Length of approach lane
            stopline_offset: Distance from center to stop line

        Returns:
            Dict with map features
        """
        # Approach lane centerline
        n_points = 50
        x = np.linspace(
            center_x - approach_length, center_x - stopline_offset, n_points
        )
        y = np.full(n_points, center_y)

        lane_center_xy = np.column_stack([x, y])

        # Stop line position
        stopline_xy = np.array(
            [
                [center_x - stopline_offset, center_y - 2],
                [center_x - stopline_offset, center_y + 2],
            ]
        )

        return {
            "lane_center_xy": lane_center_xy,
            "lane_id": 0,
            "stopline_xy": stopline_xy,
            "speed_limit": np.full(n_points, 13.4),
            "speed_limit_mask": np.ones(n_points, dtype=bool),
            "crosswalk_polys": [],
            "road_edges": [],
            "stop_signs": [
                {"position": (center_x - stopline_offset, center_y), "lane_id": 0}
            ],
        }

    def road_with_crosswalk(
        self,
        crosswalk_center_x: float = 50.0,
        crosswalk_width: float = 4.0,
        crosswalk_length: float = 10.0,
    ) -> Dict[str, Any]:
        """
        Create road with crosswalk.

        Args:
            crosswalk_center_x: X position of crosswalk center
            crosswalk_width: Width of crosswalk (along road)
            crosswalk_length: Length of crosswalk (across road)

        Returns:
            Dict with map features
        """
        # Road centerline
        n_points = 100
        x = np.linspace(0, 100, n_points)
        y = np.zeros(n_points)
        lane_center_xy = np.column_stack([x, y])

        # Crosswalk polygon
        hw = crosswalk_width / 2
        hl = crosswalk_length / 2
        crosswalk_poly = np.array(
            [
                [crosswalk_center_x - hw, -hl],
                [crosswalk_center_x + hw, -hl],
                [crosswalk_center_x + hw, hl],
                [crosswalk_center_x - hw, hl],
            ]
        )

        return {
            "lane_center_xy": lane_center_xy,
            "lane_id": 0,
            "speed_limit": np.full(n_points, 13.4),
            "speed_limit_mask": np.ones(n_points, dtype=bool),
            "crosswalk_polys": [crosswalk_poly],
            "road_edges": [],
            "stop_signs": [],
        }


@dataclass
class TestScenarioConfig:
    """Configuration for a test scenario."""

    name: str
    description: str
    n_frames: int = 91
    dt: float = 0.1

    # Expected rule outcomes
    expected_applicable_rules: List[str] = field(default_factory=list)
    expected_violations: Dict[str, bool] = field(default_factory=dict)
    expected_severity_ranges: Dict[str, Tuple[float, float]] = field(
        default_factory=dict
    )


class ScenarioFactory:
    """
    Factory for creating complete test scenarios.

    Each scenario includes:
    - Ego trajectory
    - Other agents (vehicles, pedestrians, cyclists)
    - Map context
    - Expected rule outcomes for validation
    """

    def __init__(self, n_frames: int = 91, dt: float = 0.1):
        self.n_frames = n_frames
        self.dt = dt
        self.traj_builder = TrajectoryBuilder(n_frames=n_frames, dt=dt)
        self.map_builder = MapBuilder()

    def normal_driving_scenario(self) -> Dict[str, Any]:
        """
        Scenario: Normal driving with no violations.

        - Ego drives straight at constant speed
        - Other vehicle ahead at safe distance
        - No VRUs present
        - Within speed limit

        Expected: All rules pass, no violations
        """
        # Ego trajectory: straight, constant speed
        ego_traj = self.traj_builder.straight_line(
            start_x=0, start_y=0, heading=0, speed=10.0
        )

        # Lead vehicle: 30m ahead, same speed
        lead_traj = self.traj_builder.straight_line(
            start_x=30, start_y=0, heading=0, speed=10.0
        )

        # Simple road map
        map_data = self.map_builder.simple_straight_road(
            length=100, speed_limit_mps=15.0
        )

        return {
            "config": TestScenarioConfig(
                name="normal_driving",
                description="Normal driving with safe following distance",
                n_frames=self.n_frames,
                dt=self.dt,
                expected_violations={"L6.R2": False, "L10.R1": False},
            ),
            "ego": ego_traj,
            "agents": [
                {"id": 1, "type": "vehicle", **lead_traj},
            ],
            "map": map_data,
            "signals": None,
        }

    def rear_end_collision_scenario(self) -> Dict[str, Any]:
        """
        Scenario: Ego collides with stopped vehicle.

        - Lead vehicle stops suddenly
        - Ego fails to stop in time
        - Collision occurs at frame ~50

        Expected: L10.R1 (Collision) violation
        """
        # Ego trajectory: approaching at speed
        ego_traj = self.traj_builder.straight_line(
            start_x=0, start_y=0, heading=0, speed=15.0
        )

        # Lead vehicle: starts 30m ahead, stops at frame 20
        lead_x = np.zeros(self.n_frames)
        lead_y = np.zeros(self.n_frames)
        lead_speed = np.zeros(self.n_frames)

        for i in range(self.n_frames):
            if i < 20:
                lead_x[i] = 30 + i * 0.1 * 10  # Moving at 10 m/s
                lead_speed[i] = 10.0
            else:
                lead_x[i] = lead_x[19]  # Stopped
                lead_speed[i] = 0.0

        lead_traj = {
            "x": lead_x,
            "y": lead_y,
            "yaw": np.zeros(self.n_frames),
            "speed": lead_speed,
        }

        map_data = self.map_builder.simple_straight_road()

        return {
            "config": TestScenarioConfig(
                name="rear_end_collision",
                description="Ego collides with stopped vehicle",
                n_frames=self.n_frames,
                dt=self.dt,
                expected_violations={"L10.R1": True, "L0.R2": True},
                expected_severity_ranges={"L10.R1": (5.0, 100.0)},
            ),
            "ego": ego_traj,
            "agents": [
                {"id": 1, "type": "vehicle", "length": 4.5, "width": 2.0, **lead_traj},
            ],
            "map": map_data,
            "signals": None,
        }

    def pedestrian_crossing_scenario(self) -> Dict[str, Any]:
        """
        Scenario: Pedestrian crossing road at crosswalk.

        - Pedestrian walks across crosswalk
        - Ego approaches and should yield
        - Tests L8.R3 (Crosswalk Yield) and L6.R4 (Pedestrian Interaction)

        Expected: Varies based on ego behavior (safe or violation)
        """
        # Ego trajectory: approaching crosswalk at 50m
        ego_traj = self.traj_builder.straight_line(
            start_x=0, start_y=0, heading=0, speed=8.0
        )

        # Pedestrian: crosses road at x=50, starts at y=-8, walks to y=8
        ped_x = np.full(self.n_frames, 50.0)
        ped_y = np.linspace(-8, 8, self.n_frames)  # Walking at ~1.7 m/s
        ped_speed = np.full(self.n_frames, 1.7)
        ped_yaw = np.full(self.n_frames, np.pi / 2)  # Heading +Y

        ped_traj = {
            "x": ped_x,
            "y": ped_y,
            "yaw": ped_yaw,
            "speed": ped_speed,
        }

        # Map with crosswalk at x=50
        map_data = self.map_builder.road_with_crosswalk(
            crosswalk_center_x=50.0,
            crosswalk_width=4.0,
            crosswalk_length=10.0,
        )

        return {
            "config": TestScenarioConfig(
                name="pedestrian_crossing",
                description="Pedestrian crossing at crosswalk",
                n_frames=self.n_frames,
                dt=self.dt,
                expected_applicable_rules=["L6.R4", "L8.R3", "L10.R2"],
            ),
            "ego": ego_traj,
            "agents": [
                {
                    "id": 1,
                    "type": "pedestrian",
                    "length": 0.5,
                    "width": 0.5,
                    **ped_traj,
                },
            ],
            "map": map_data,
            "signals": None,
        }

    def cyclist_passing_scenario(self) -> Dict[str, Any]:
        """
        Scenario: Ego passes cyclist with insufficient clearance.

        - Cyclist riding along road edge
        - Ego passes too close
        - Tests L6.R5 (Cyclist Interaction) and L10.R2 (VRU Clearance)

        Expected: L6.R5, L10.R2 violations
        """
        # Ego trajectory: straight, passing cyclist
        ego_traj = self.traj_builder.straight_line(
            start_x=0, start_y=0, heading=0, speed=12.0
        )

        # Cyclist: riding along road edge at y=1.0 (too close)
        cyc_x = np.linspace(20, 30, self.n_frames)  # Moving slowly
        cyc_y = np.full(self.n_frames, 1.0)
        cyc_speed = np.full(self.n_frames, 5.0)
        cyc_yaw = np.zeros(self.n_frames)

        cyc_traj = {
            "x": cyc_x,
            "y": cyc_y,
            "yaw": cyc_yaw,
            "speed": cyc_speed,
        }

        map_data = self.map_builder.simple_straight_road()

        return {
            "config": TestScenarioConfig(
                name="cyclist_passing_unsafe",
                description="Ego passes cyclist with insufficient clearance",
                n_frames=self.n_frames,
                dt=self.dt,
                expected_violations={"L6.R5": True, "L10.R2": True},
            ),
            "ego": ego_traj,
            "agents": [
                {"id": 1, "type": "cyclist", "length": 1.8, "width": 0.6, **cyc_traj},
            ],
            "map": map_data,
            "signals": None,
        }

    def harsh_braking_scenario(self) -> Dict[str, Any]:
        """
        Scenario: Ego brakes harshly.

        - Ego traveling at high speed
        - Emergency braking (-8 m/s²)
        - Tests L1.R1 (Smooth Braking)

        Expected: L1.R1 violation
        """
        ego_traj = self.traj_builder.harsh_braking(
            start_x=0,
            start_y=0,
            heading=0,
            initial_speed=20.0,
            brake_start_frame=30,
            deceleration=-8.0,
        )

        map_data = self.map_builder.simple_straight_road()

        return {
            "config": TestScenarioConfig(
                name="harsh_braking",
                description="Emergency braking scenario",
                n_frames=self.n_frames,
                dt=self.dt,
                expected_violations={"L1.R1": True},
                expected_severity_ranges={"L1.R1": (1.0, 50.0)},
            ),
            "ego": ego_traj,
            "agents": [],
            "map": map_data,
            "signals": None,
        }

    def lane_change_scenario(self) -> Dict[str, Any]:
        """
        Scenario: Ego performs lane change.

        - Smooth lane change
        - Tests L1.R5 (Lane Change Smoothness)

        Expected: No violation for smooth change
        """
        ego_traj = self.traj_builder.lane_change(
            start_x=0,
            start_y=0,
            heading=0,
            speed=15.0,
            lateral_offset=3.5,
            change_start_frame=30,
            change_duration_frames=30,
        )

        map_data = self.map_builder.simple_straight_road()

        return {
            "config": TestScenarioConfig(
                name="lane_change_smooth",
                description="Smooth lane change maneuver",
                n_frames=self.n_frames,
                dt=self.dt,
                expected_applicable_rules=["L1.R5"],
                expected_violations={"L1.R5": False},
            ),
            "ego": ego_traj,
            "agents": [],
            "map": map_data,
            "signals": None,
        }

    def stop_sign_violation_scenario(self) -> Dict[str, Any]:
        """
        Scenario: Ego runs stop sign.

        - Stop sign at x=45
        - Ego does not stop, continues through
        - Tests L8.R2 (Stop Sign)

        Expected: L8.R2 violation
        """
        # Ego goes through without stopping
        ego_traj = self.traj_builder.straight_line(
            start_x=0, start_y=0, heading=0, speed=10.0
        )

        map_data = self.map_builder.intersection_with_stopline(
            center_x=50.0,
            stopline_offset=5.0,
        )

        return {
            "config": TestScenarioConfig(
                name="stop_sign_violation",
                description="Running stop sign",
                n_frames=self.n_frames,
                dt=self.dt,
                expected_violations={"L8.R2": True},
            ),
            "ego": ego_traj,
            "agents": [],
            "map": map_data,
            "signals": None,
        }

    def stop_sign_compliance_scenario(self) -> Dict[str, Any]:
        """
        Scenario: Ego stops at stop sign.

        - Stop sign at x=45
        - Ego stops properly
        - Tests L8.R2 (Stop Sign)

        Expected: L8.R2 no violation
        """
        # Ego stops at stop line (x=45)
        ego_traj = self.traj_builder.stop_at_position(
            start_x=0,
            start_y=0,
            heading=0,
            initial_speed=10.0,
            stop_x=43.0,
            stop_y=0.0,
            stop_duration_frames=30,
        )

        map_data = self.map_builder.intersection_with_stopline(
            center_x=50.0,
            stopline_offset=5.0,
        )

        return {
            "config": TestScenarioConfig(
                name="stop_sign_compliance",
                description="Proper stop at stop sign",
                n_frames=self.n_frames,
                dt=self.dt,
                expected_violations={"L8.R2": False},
            ),
            "ego": ego_traj,
            "agents": [],
            "map": map_data,
            "signals": None,
        }

    def speeding_scenario(self) -> Dict[str, Any]:
        """
        Scenario: Ego exceeds speed limit.

        - Speed limit: 13.4 m/s (30 mph)
        - Ego traveling at 20 m/s (45 mph)
        - Tests L7.R4 (Speed Limit)

        Expected: L7.R4 violation
        """
        ego_traj = self.traj_builder.straight_line(
            start_x=0, start_y=0, heading=0, speed=20.0
        )

        map_data = self.map_builder.simple_straight_road(speed_limit_mps=13.4)

        return {
            "config": TestScenarioConfig(
                name="speeding",
                description="Exceeding speed limit",
                n_frames=self.n_frames,
                dt=self.dt,
                expected_violations={"L7.R4": True},
                expected_severity_ranges={"L7.R4": (1.0, 50.0)},
            ),
            "ego": ego_traj,
            "agents": [],
            "map": map_data,
            "signals": None,
        }

    def wrong_way_scenario(self) -> Dict[str, Any]:
        """
        Scenario: Ego driving wrong way.

        - Road heading is +X (0 radians)
        - Ego heading is -X (pi radians)
        - Tests L8.R5 (Wrong-way)

        Expected: L8.R5 violation
        """
        ego_traj = self.traj_builder.straight_line(
            start_x=100, start_y=0, heading=np.pi, speed=10.0  # Heading opposite
        )

        map_data = self.map_builder.simple_straight_road()

        return {
            "config": TestScenarioConfig(
                name="wrong_way",
                description="Driving wrong way on road",
                n_frames=self.n_frames,
                dt=self.dt,
                expected_violations={"L8.R5": True},
            ),
            "ego": ego_traj,
            "agents": [],
            "map": map_data,
            "signals": None,
        }

    def following_too_close_scenario(self) -> Dict[str, Any]:
        """
        Scenario: Ego following too close.

        - Lead vehicle 8m ahead at 15 m/s
        - Ego at 15 m/s (0.53s following time, below 2s rule)
        - Tests L6.R2 (Following Distance)

        Expected: L6.R2 violation
        """
        ego_traj = self.traj_builder.straight_line(
            start_x=0, start_y=0, heading=0, speed=15.0
        )

        lead_traj = self.traj_builder.straight_line(
            start_x=8, start_y=0, heading=0, speed=15.0
        )

        map_data = self.map_builder.simple_straight_road()

        return {
            "config": TestScenarioConfig(
                name="following_too_close",
                description="Tailgating lead vehicle",
                n_frames=self.n_frames,
                dt=self.dt,
                expected_violations={"L6.R2": True},
                expected_severity_ranges={"L6.R2": (1.0, 30.0)},
            ),
            "ego": ego_traj,
            "agents": [
                {"id": 1, "type": "vehicle", "length": 4.5, "width": 2.0, **lead_traj},
            ],
            "map": map_data,
            "signals": None,
        }

    def left_turn_gap_violation_scenario(self) -> Dict[str, Any]:
        """
        Scenario: Ego accepts unsafe gap during left turn.

        - Oncoming vehicle approaching
        - Ego turns left with insufficient gap
        - Tests L4.R3 (Left Turn Gap)

        Expected: L4.R3 violation
        """
        # Ego making left turn
        ego_traj = self.traj_builder.left_turn(
            start_x=0,
            start_y=0,
            start_heading=0,
            speed=5.0,
            turn_radius=10.0,
            turn_start_frame=30,
        )

        # Oncoming vehicle
        oncoming_traj = self.traj_builder.straight_line(
            start_x=50, start_y=0, heading=np.pi, speed=15.0  # Coming towards ego
        )

        map_data = self.map_builder.simple_straight_road()

        return {
            "config": TestScenarioConfig(
                name="left_turn_gap_violation",
                description="Unsafe left turn with oncoming traffic",
                n_frames=self.n_frames,
                dt=self.dt,
                expected_applicable_rules=["L4.R3"],
            ),
            "ego": ego_traj,
            "agents": [
                {
                    "id": 1,
                    "type": "vehicle",
                    "length": 4.5,
                    "width": 2.0,
                    **oncoming_traj,
                },
            ],
            "map": map_data,
            "signals": None,
        }

    def multi_agent_complex_scenario(self) -> Dict[str, Any]:
        """
        Complex scenario with multiple agents and potential violations.

        - Lead vehicle
        - Pedestrian on sidewalk
        - Cyclist in bike lane
        - Speed limit, crosswalk present

        Tests multiple rules simultaneously.
        """
        # Ego trajectory
        ego_traj = self.traj_builder.straight_line(
            start_x=0, start_y=0, heading=0, speed=12.0
        )

        # Lead vehicle - safe distance
        lead_traj = self.traj_builder.straight_line(
            start_x=35, start_y=0, heading=0, speed=12.0
        )

        # Pedestrian on sidewalk (y=5, not crossing)
        ped_x = np.linspace(40, 50, self.n_frames)
        ped_y = np.full(self.n_frames, 5.0)
        ped_traj = {
            "x": ped_x,
            "y": ped_y,
            "yaw": np.zeros(self.n_frames),
            "speed": np.full(self.n_frames, 1.2),
        }

        # Cyclist in bike lane (y=2)
        cyc_x = np.linspace(20, 35, self.n_frames)
        cyc_y = np.full(self.n_frames, 2.0)
        cyc_traj = {
            "x": cyc_x,
            "y": cyc_y,
            "yaw": np.zeros(self.n_frames),
            "speed": np.full(self.n_frames, 6.0),
        }

        map_data = self.map_builder.road_with_crosswalk(
            crosswalk_center_x=60.0,
        )

        return {
            "config": TestScenarioConfig(
                name="multi_agent_complex",
                description="Complex multi-agent scenario",
                n_frames=self.n_frames,
                dt=self.dt,
                expected_applicable_rules=["L6.R2", "L6.R5", "L10.R1", "L10.R2"],
            ),
            "ego": ego_traj,
            "agents": [
                {"id": 1, "type": "vehicle", "length": 4.5, "width": 2.0, **lead_traj},
                {
                    "id": 2,
                    "type": "pedestrian",
                    "length": 0.5,
                    "width": 0.5,
                    **ped_traj,
                },
                {"id": 3, "type": "cyclist", "length": 1.8, "width": 0.6, **cyc_traj},
            ],
            "map": map_data,
            "signals": None,
        }


def scenario_to_context(scenario: Dict[str, Any]):
    """
    Convert scenario dict to ScenarioContext objects.

    Args:
        scenario: Dict from ScenarioFactory

    Returns:
        Tuple of (ScenarioContext, TestScenarioConfig)
    """
    from waymo_rule_eval.core.context import (
        Agent,
        EgoState,
        MapContext,
        ScenarioContext,
    )

    config = scenario["config"]
    ego_data = scenario["ego"]
    agents_data = scenario["agents"]
    map_data = scenario["map"]

    n_frames = config.n_frames

    # Create EgoState
    ego = EgoState(
        x=ego_data["x"],
        y=ego_data["y"],
        yaw=ego_data["yaw"],
        speed=ego_data["speed"],
        length=4.5,
        width=2.0,
        valid=np.ones(n_frames, dtype=bool),
    )

    # Create Agents
    agents = []
    for agent_data in agents_data:
        agent = Agent(
            id=agent_data["id"],
            type=agent_data["type"],
            x=agent_data["x"],
            y=agent_data["y"],
            yaw=agent_data["yaw"],
            speed=agent_data["speed"],
            length=agent_data.get("length", 4.5),
            width=agent_data.get("width", 2.0),
            valid=np.ones(n_frames, dtype=bool),
        )
        agents.append(agent)

    # Create MapContext
    map_ctx = MapContext(
        lane_center_xy=map_data["lane_center_xy"],
        lane_id=map_data.get("lane_id", 0),
        stopline_xy=map_data.get("stopline_xy"),
        crosswalk_polys=map_data.get("crosswalk_polys", []),
        road_edges=map_data.get("road_edges", []),
        stop_signs=map_data.get("stop_signs", []),
        speed_limit=map_data.get("speed_limit"),
        speed_limit_mask=map_data.get("speed_limit_mask"),
    )

    # Create ScenarioContext
    ctx = ScenarioContext(
        scenario_id=config.name,
        ego=ego,
        agents=agents,
        map_context=map_ctx,
        signals=None,
        dt=config.dt,
    )

    return ctx, config


def get_all_test_scenarios() -> List[Dict[str, Any]]:
    """Get all test scenarios for parametrized testing."""
    factory = ScenarioFactory()

    return [
        factory.normal_driving_scenario(),
        factory.rear_end_collision_scenario(),
        factory.pedestrian_crossing_scenario(),
        factory.cyclist_passing_scenario(),
        factory.harsh_braking_scenario(),
        factory.lane_change_scenario(),
        factory.stop_sign_violation_scenario(),
        factory.stop_sign_compliance_scenario(),
        factory.speeding_scenario(),
        factory.wrong_way_scenario(),
        factory.following_too_close_scenario(),
        factory.left_turn_gap_violation_scenario(),
        factory.multi_agent_complex_scenario(),
    ]


class ScenarioFixtureFactory:
    """
    Wrapper class for ScenarioFactory that returns ScenarioContext directly.

    Used by pytest tests to easily create scenarios for rule testing.
    """

    def __init__(self):
        self._factory = ScenarioFactory()

    def _to_context(self, scenario: Dict[str, Any]) -> "ScenarioContext":
        """Convert scenario dict to ScenarioContext."""
        ctx, _ = scenario_to_context(scenario)
        return ctx

    def create_ego_state(self, n_frames: int = 50, speed: float = 10.0) -> "EgoState":
        """Create a simple EgoState for unit testing."""
        from waymo_rule_eval.core.context import EgoState

        dt = 0.1
        x = np.linspace(0, speed * n_frames * dt, n_frames)
        y = np.zeros(n_frames)
        yaw = np.zeros(n_frames)

        return EgoState(
            x=x,
            y=y,
            yaw=yaw,
            speed=np.full(n_frames, speed),
            length=4.5,
            width=2.0,
            valid=np.ones(n_frames, dtype=bool),
        )

    def create_agent(
        self,
        agent_id: int = 1,
        agent_type: str = "vehicle",
        offset_x: float = 30.0,
        speed: float = 10.0,
        n_frames: int = 50,
    ) -> "Agent":
        """Create a simple Agent for unit testing."""
        from waymo_rule_eval.core.context import Agent

        dt = 0.1
        x = np.linspace(offset_x, offset_x + speed * n_frames * dt, n_frames)
        y = np.zeros(n_frames)
        yaw = np.zeros(n_frames)

        return Agent(
            id=agent_id,
            type=agent_type,
            x=x,
            y=y,
            yaw=yaw,
            speed=np.full(n_frames, speed),
            length=4.5,
            width=2.0,
            valid=np.ones(n_frames, dtype=bool),
        )

    def create_map_context(
        self, n_lane_points: int = 100, speed_limit: float = 25.0
    ) -> "MapContext":
        """Create simple MapContext for unit testing."""
        from waymo_rule_eval.core.context import MapContext

        lane_xy = np.column_stack(
            [
                np.linspace(0, 100, n_lane_points),
                np.zeros(n_lane_points),
            ]
        )

        return MapContext(
            lane_center_xy=lane_xy,
            lane_id=0,
            stopline_xy=None,
            crosswalk_polys=[],
            road_edges=[],
            stop_signs=[],
            speed_limit=speed_limit,
            speed_limit_mask=None,
        )

    def create_scenario_context(
        self,
        n_frames: int = 50,
        n_agents: int = 1,
        add_agents: bool = True,
        start_x: float = 0.0,
        start_y: float = 0.0,
        include_crosswalks: bool = True,
        include_stoplines: bool = True,
    ) -> "ScenarioContext":
        """Create a flexible ScenarioContext for testing."""
        from waymo_rule_eval.core.context import MapContext, ScenarioContext

        ego = self.create_ego_state(n_frames=n_frames)
        ego.x = ego.x + start_x
        ego.y = ego.y + start_y

        agents = []
        if add_agents:
            for i in range(n_agents):
                agent = self.create_agent(
                    agent_id=i + 1, offset_x=30 + i * 10, n_frames=n_frames
                )
                agents.append(agent)

        lane_xy = np.column_stack(
            [
                np.linspace(start_x, start_x + 100, 100),
                np.full(100, start_y),
            ]
        )

        crosswalks = []
        if include_crosswalks:
            crosswalks = [np.array([[50, -5], [55, -5], [55, 5], [50, 5]])]

        stopline = None
        if include_stoplines:
            stopline = np.array([[48, -3], [48, 3]])

        map_ctx = MapContext(
            lane_center_xy=lane_xy,
            lane_id=0,
            stopline_xy=stopline,
            crosswalk_polys=crosswalks,
            road_edges=[],
            stop_signs=[],
            speed_limit=25.0,
            speed_limit_mask=None,
        )

        return ScenarioContext(
            scenario_id="test_scenario",
            ego=ego,
            agents=agents,
            map_context=map_ctx,
            signals=None,
            dt=0.1,
        )

    # Scenario creation methods
    def create_normal_driving_scenario(self) -> "ScenarioContext":
        """Normal driving scenario - no violations expected."""
        return self._to_context(self._factory.normal_driving_scenario())

    def create_rear_end_collision_scenario(self) -> "ScenarioContext":
        """Rear-end collision scenario."""
        return self._to_context(self._factory.rear_end_collision_scenario())

    def create_pedestrian_crossing_scenario(self) -> "ScenarioContext":
        """Pedestrian crossing scenario."""
        return self._to_context(self._factory.pedestrian_crossing_scenario())

    def create_cyclist_overtake_scenario(self) -> "ScenarioContext":
        """Cyclist overtaking scenario."""
        return self._to_context(self._factory.cyclist_passing_scenario())

    def create_harsh_braking_scenario(self) -> "ScenarioContext":
        """Harsh braking scenario."""
        return self._to_context(self._factory.harsh_braking_scenario())

    def create_lane_departure_scenario(self) -> "ScenarioContext":
        """Lane departure scenario."""
        return self._to_context(self._factory.lane_change_scenario())

    def create_speeding_scenario(self) -> "ScenarioContext":
        """Speeding scenario."""
        return self._to_context(self._factory.speeding_scenario())

    def create_red_light_violation_scenario(self) -> "ScenarioContext":
        """Red light violation scenario."""
        return self._to_context(self._factory.stop_sign_violation_scenario())

    def create_tailgating_scenario(self) -> "ScenarioContext":
        """Following too close scenario."""
        return self._to_context(self._factory.following_too_close_scenario())

    def create_intersection_scenario(self) -> "ScenarioContext":
        """Intersection scenario."""
        return self._to_context(self._factory.left_turn_gap_violation_scenario())

    def create_stationary_scenario(self) -> "ScenarioContext":
        """Stationary ego vehicle scenario."""
        from waymo_rule_eval.core.context import ScenarioContext

        n_frames = 50
        ego = self.create_ego_state(n_frames=n_frames, speed=0.0)
        ego.vx = np.zeros(n_frames)

        map_ctx = self.create_map_context()

        return ScenarioContext(
            scenario_id="stationary",
            ego=ego,
            agents=[],
            map_context=map_ctx,
            signals=None,
            dt=0.1,
        )

    def create_scenario_with_speed(self, speed: float) -> "ScenarioContext":
        """Create scenario with specific ego speed."""
        from waymo_rule_eval.core.context import ScenarioContext

        n_frames = 50
        ego = self.create_ego_state(n_frames=n_frames, speed=speed)
        map_ctx = self.create_map_context(speed_limit=25.0)

        return ScenarioContext(
            scenario_id=f"speed_{speed}",
            ego=ego,
            agents=[self.create_agent()],
            map_context=map_ctx,
            signals=None,
            dt=0.1,
        )

    def create_scenario_with_acceleration(
        self, acceleration: float
    ) -> "ScenarioContext":
        """Create scenario with specific acceleration profile."""
        from waymo_rule_eval.core.context import EgoState, ScenarioContext

        n_frames = 50
        dt = 0.1

        # Create trajectory with given acceleration
        v0 = 10.0
        v = v0 + acceleration * np.arange(n_frames) * dt
        v = np.clip(v, 0, 50)

        x = np.cumsum(v) * dt

        ego = EgoState(
            x=x,
            y=np.zeros(n_frames),
            yaw=np.zeros(n_frames),
            speed=v,
            length=4.5,
            width=2.0,
            valid=np.ones(n_frames, dtype=bool),
        )

        map_ctx = self.create_map_context()

        return ScenarioContext(
            scenario_id=f"accel_{acceleration}",
            ego=ego,
            agents=[],
            map_context=map_ctx,
            signals=None,
            dt=0.1,
        )

    def create_cut_in_scenario(self) -> "ScenarioContext":
        """Cut-in scenario."""
        return self.create_normal_driving_scenario()

    def create_lane_change_scenario(self) -> "ScenarioContext":
        """Lane change scenario."""
        return self._to_context(self._factory.lane_change_scenario())


import pytest


@pytest.fixture
def fixture_factory() -> ScenarioFixtureFactory:
    """Provide a ScenarioFixtureFactory for tests."""
    return ScenarioFixtureFactory()


@pytest.fixture
def normal_driving_scenario(fixture_factory) -> "ScenarioContext":
    """Normal driving scenario fixture."""
    return fixture_factory.create_normal_driving_scenario()


@pytest.fixture
def collision_scenario(fixture_factory) -> "ScenarioContext":
    """Collision scenario fixture."""
    return fixture_factory.create_rear_end_collision_scenario()


@pytest.fixture
def harsh_braking_scenario(fixture_factory) -> "ScenarioContext":
    """Harsh braking scenario fixture."""
    return fixture_factory.create_harsh_braking_scenario()


@pytest.fixture
def speeding_scenario(fixture_factory) -> "ScenarioContext":
    """Speeding scenario fixture."""
    return fixture_factory.create_speeding_scenario()
