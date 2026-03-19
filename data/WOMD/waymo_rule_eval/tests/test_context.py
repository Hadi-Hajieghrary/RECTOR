"""
Comprehensive Unit Tests for Core Context Classes.

Tests EgoState, Agent, MapContext, ScenarioContext with various
edge cases and realistic scenarios.
"""

import os
import sys

import numpy as np
import pytest

# Add the workspace root to path so waymo_rule_eval package can be imported
_workspace_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _workspace_root not in sys.path:
    sys.path.insert(0, _workspace_root)

from waymo_rule_eval.core.context import (
    DEFAULT_DT_S,
    DEFAULT_EGO_LENGTH,
    DEFAULT_EGO_WIDTH,
    Agent,
    EgoState,
    MapContext,
    MapSignals,
    ScenarioContext,
)


class TestEgoState:
    """Comprehensive tests for EgoState class."""

    # ==================== Basic Creation Tests ====================

    def test_basic_creation(self):
        """Test basic EgoState creation with minimal args."""
        ego = EgoState(
            x=np.array([0, 1, 2]),
            y=np.array([0, 0, 0]),
            yaw=np.array([0, 0, 0]),
            speed=np.array([5, 5, 5]),
        )

        assert len(ego) == 3
        assert ego.n_valid == 3
        assert ego.length == DEFAULT_EGO_LENGTH
        assert ego.width == DEFAULT_EGO_WIDTH

    def test_creation_with_all_params(self):
        """Test EgoState creation with all parameters."""
        n = 50
        ego = EgoState(
            x=np.linspace(0, 100, n),
            y=np.zeros(n),
            yaw=np.zeros(n),
            speed=np.ones(n) * 10.0,
            length=5.0,
            width=2.2,
            valid=np.ones(n, dtype=bool),
        )

        assert len(ego) == 50
        assert ego.length == 5.0
        assert ego.width == 2.2
        assert ego.n_valid == 50

    def test_creation_with_nan_values(self):
        """Test EgoState with NaN values creates proper validity mask."""
        x = np.array([0, 1, np.nan, 3, 4])
        y = np.array([0, 0, 0, np.nan, 0])

        ego = EgoState(
            x=x,
            y=y,
            yaw=np.zeros(5),
            speed=np.ones(5) * 5.0,
        )

        # Should have 3 valid (indices 0, 1, 4)
        assert ego.n_valid == 3
        assert ego.is_valid_at(0) is True
        assert ego.is_valid_at(2) is False
        assert ego.is_valid_at(3) is False

    def test_scalar_input_conversion(self):
        """Test that scalar inputs are converted to arrays."""
        ego = EgoState(
            x=5.0,
            y=3.0,
            yaw=0.0,
            speed=10.0,
        )

        assert len(ego) == 1
        assert isinstance(ego.x, np.ndarray)
        assert ego.x[0] == 5.0

    # ==================== Validity Tests ====================

    def test_is_valid_at_boundary(self):
        """Test validity check at boundary indices."""
        ego = EgoState(
            x=np.arange(10),
            y=np.zeros(10),
            yaw=np.zeros(10),
            speed=np.ones(10) * 5.0,
        )

        assert ego.is_valid_at(0) is True
        assert ego.is_valid_at(9) is True
        assert ego.is_valid_at(-1) is False
        assert ego.is_valid_at(10) is False
        assert ego.is_valid_at(100) is False

    def test_custom_validity_mask(self):
        """Test EgoState with custom validity mask."""
        n = 10
        valid = np.array(
            [True, True, False, False, True, True, True, False, True, True]
        )

        ego = EgoState(
            x=np.arange(n),
            y=np.zeros(n),
            yaw=np.zeros(n),
            speed=np.ones(n),
            valid=valid,
        )

        assert ego.n_valid == 7
        assert ego.is_valid_at(2) is False
        assert ego.is_valid_at(7) is False

    # ==================== Position Tests ====================

    def test_position_at(self):
        """Test position retrieval at various indices."""
        ego = EgoState(
            x=np.array([0, 10, 20, 30]),
            y=np.array([0, 5, 10, 15]),
            yaw=np.zeros(4),
            speed=np.ones(4),
        )

        pos = ego.position_at(0)
        assert pos == (0, 0)

        pos = ego.position_at(2)
        assert pos == (20, 10)

    def test_position_at_invalid(self):
        """Test position_at with invalid indices."""
        ego = EgoState(
            x=np.array([0, 1, np.nan, 3]),
            y=np.array([0, 0, 0, 0]),
            yaw=np.zeros(4),
            speed=np.ones(4),
        )

        assert ego.position_at(-1) is None
        assert ego.position_at(10) is None
        assert ego.position_at(2) is None  # NaN position

    # ==================== Kinematics Tests ====================

    def test_get_acceleration_constant_speed(self):
        """Test acceleration computation with constant speed."""
        ego = EgoState(
            x=np.linspace(0, 50, 50),
            y=np.zeros(50),
            yaw=np.zeros(50),
            speed=np.ones(50) * 10.0,
        )

        acc = ego.get_acceleration(dt=0.1)

        # Constant speed should give ~0 acceleration
        assert np.allclose(acc[1:-1], 0.0, atol=0.01)

    def test_get_acceleration_accelerating(self):
        """Test acceleration computation with accelerating vehicle."""
        n = 50
        t = np.arange(n) * 0.1
        speed = 10.0 + 2.0 * t  # 2 m/s² acceleration

        ego = EgoState(
            x=np.zeros(n),
            y=np.zeros(n),
            yaw=np.zeros(n),
            speed=speed,
        )

        acc = ego.get_acceleration(dt=0.1)

        # Should be approximately 2 m/s²
        assert np.mean(np.abs(acc[1:-1] - 2.0)) < 0.1

    def test_get_acceleration_braking(self):
        """Test acceleration during braking (negative acceleration)."""
        n = 50
        t = np.arange(n) * 0.1
        speed = np.maximum(0, 20.0 - 4.0 * t)  # -4 m/s² braking

        ego = EgoState(
            x=np.zeros(n),
            y=np.zeros(n),
            yaw=np.zeros(n),
            speed=speed,
        )

        acc = ego.get_acceleration(dt=0.1)

        # Should be approximately -4 m/s² until stopped
        braking_frames = speed > 1.0
        assert np.mean(acc[braking_frames]) < -3.0

    def test_get_jerk(self):
        """Test jerk (rate of acceleration change) computation."""
        n = 100
        t = np.arange(n) * 0.1
        # Speed profile with varying acceleration
        speed = 10 + 2 * np.sin(t)

        ego = EgoState(
            x=np.zeros(n),
            y=np.zeros(n),
            yaw=np.zeros(n),
            speed=speed,
        )

        jerk = ego.get_jerk(dt=0.1)

        assert len(jerk) == n
        assert not np.any(np.isnan(jerk[2:-2]))

    def test_get_yaw_rate(self):
        """Test yaw rate computation."""
        n = 50
        t = np.arange(n) * 0.1
        yaw = 0.1 * t  # 0.1 rad/s constant yaw rate

        ego = EgoState(
            x=np.zeros(n),
            y=np.zeros(n),
            yaw=yaw,
            speed=np.ones(n) * 10,
        )

        yaw_rate = ego.get_yaw_rate(dt=0.1)

        # Should be approximately 0.1 rad/s
        assert np.mean(np.abs(yaw_rate[1:-1] - 0.1)) < 0.01

    def test_get_lateral_acceleration(self):
        """Test lateral acceleration computation."""
        n = 50
        t = np.arange(n) * 0.1
        yaw = 0.2 * t  # 0.2 rad/s yaw rate
        speed = np.ones(n) * 10  # 10 m/s

        ego = EgoState(
            x=np.zeros(n),
            y=np.zeros(n),
            yaw=yaw,
            speed=speed,
        )

        lat_acc = ego.get_lateral_acceleration(dt=0.1)

        # v * yaw_rate = 10 * 0.2 = 2 m/s²
        assert np.mean(np.abs(lat_acc[1:-1] - 2.0)) < 0.2


class TestAgent:
    """Comprehensive tests for Agent class."""

    def test_vehicle_creation(self):
        """Test vehicle agent creation."""
        agent = Agent(
            id=1,
            type="vehicle",
            x=np.linspace(0, 50, 50),
            y=np.zeros(50),
            yaw=np.zeros(50),
            speed=np.ones(50) * 10,
            length=4.5,
            width=2.0,
        )

        assert agent.id == 1
        assert agent.is_vehicle is True
        assert agent.is_pedestrian is False
        assert agent.is_cyclist is False
        assert agent.is_vru is False

    def test_pedestrian_creation(self):
        """Test pedestrian agent creation."""
        agent = Agent(
            id=2,
            type="pedestrian",
            x=np.linspace(0, 10, 50),
            y=np.zeros(50),
            yaw=np.zeros(50),
            speed=np.ones(50) * 1.4,
            length=0.5,
            width=0.5,
        )

        assert agent.is_pedestrian is True
        assert agent.is_vru is True
        assert agent.is_vehicle is False

    def test_cyclist_creation(self):
        """Test cyclist agent creation."""
        agent = Agent(
            id=3,
            type="cyclist",
            x=np.linspace(0, 30, 50),
            y=np.zeros(50),
            yaw=np.zeros(50),
            speed=np.ones(50) * 5.0,
            length=1.8,
            width=0.6,
        )

        assert agent.is_cyclist is True
        assert agent.is_vru is True
        assert agent.is_vehicle is False

    def test_type_aliases(self):
        """Test various type string aliases."""
        # Car -> vehicle
        car = Agent(
            id=1, type="car", x=[0], y=[0], yaw=[0], speed=[0], length=4.5, width=2.0
        )
        assert car.is_vehicle is True

        # Truck -> vehicle
        truck = Agent(
            id=2, type="truck", x=[0], y=[0], yaw=[0], speed=[0], length=8.0, width=2.5
        )
        assert truck.is_vehicle is True

        # Ped -> pedestrian
        ped = Agent(
            id=3, type="ped", x=[0], y=[0], yaw=[0], speed=[0], length=0.5, width=0.5
        )
        assert ped.is_pedestrian is True

        # Bicycle -> cyclist
        bike = Agent(
            id=4,
            type="bicycle",
            x=[0],
            y=[0],
            yaw=[0],
            speed=[0],
            length=1.8,
            width=0.6,
        )
        assert bike.is_cyclist is True

    def test_validity_and_position(self):
        """Test agent validity and position retrieval."""
        agent = Agent(
            id=1,
            type="vehicle",
            x=np.array([0, 10, 20, 30]),
            y=np.array([0, 5, 10, 15]),
            yaw=np.zeros(4),
            speed=np.ones(4),
            length=4.5,
            width=2.0,
        )

        assert agent.is_valid_at(0) is True
        assert agent.is_valid_at(3) is True
        assert agent.is_valid_at(4) is False

        pos = agent.position_at(2)
        assert pos == (20, 10)


class TestMapContext:
    """Comprehensive tests for MapContext class."""

    def test_basic_creation(self):
        """Test basic MapContext creation."""
        centerline = np.array([[0, 0], [10, 0], [20, 0], [30, 0]])

        map_ctx = MapContext(
            lane_center_xy=centerline,
            lane_id=0,
        )

        assert map_ctx.lane_center_xy.shape == (4, 2)
        assert map_ctx.has_lane_geometry is True
        assert map_ctx.has_crosswalks is False
        assert map_ctx.has_stop_signs is False

    def test_with_crosswalks(self):
        """Test MapContext with crosswalk polygons."""
        centerline = np.array([[0, 0], [50, 0], [100, 0]])

        crosswalk = np.array([[45, -5], [55, -5], [55, 5], [45, 5]])

        map_ctx = MapContext(
            lane_center_xy=centerline,
            crosswalk_polys=[crosswalk],
        )

        assert map_ctx.has_crosswalks is True
        assert len(map_ctx.crosswalk_polys) == 1

    def test_with_stop_signs(self):
        """Test MapContext with stop signs."""
        centerline = np.array([[0, 0], [50, 0], [100, 0]])

        stop_signs = [
            {"position": (45, 0), "lane_id": 0},
        ]

        map_ctx = MapContext(
            lane_center_xy=centerline,
            stop_signs=stop_signs,
        )

        assert map_ctx.has_stop_signs is True
        assert len(map_ctx.stop_signs) == 1

    def test_with_speed_limits(self):
        """Test MapContext with speed limits."""
        n = 100
        centerline = np.column_stack([np.linspace(0, 100, n), np.zeros(n)])
        speed_limit = np.full(n, 13.4)  # 30 mph

        map_ctx = MapContext(
            lane_center_xy=centerline,
            speed_limit=speed_limit,
            speed_limit_mask=np.ones(n, dtype=bool),
        )

        assert map_ctx.has_speed_limits is True
        assert map_ctx.speed_limit[0] == 13.4

    def test_empty_centerline_handling(self):
        """Test MapContext with empty/None centerline."""
        map_ctx = MapContext(
            lane_center_xy=None,
        )

        assert map_ctx.has_lane_geometry is False
        assert map_ctx.lane_center_xy.shape == (0, 2)


class TestScenarioContext:
    """Comprehensive tests for ScenarioContext class."""

    def create_simple_scenario(self, n_frames=50):
        """Helper to create a simple scenario context."""
        ego = EgoState(
            x=np.linspace(0, 50, n_frames),
            y=np.zeros(n_frames),
            yaw=np.zeros(n_frames),
            speed=np.ones(n_frames) * 10.0,
        )

        agent = Agent(
            id=1,
            type="vehicle",
            x=np.linspace(30, 80, n_frames),
            y=np.zeros(n_frames),
            yaw=np.zeros(n_frames),
            speed=np.ones(n_frames) * 10.0,
            length=4.5,
            width=2.0,
        )

        map_ctx = MapContext(
            lane_center_xy=np.column_stack([np.linspace(0, 100, 100), np.zeros(100)]),
        )

        return ScenarioContext(
            scenario_id="test_scenario",
            ego=ego,
            agents=[agent],
            map_context=map_ctx,
            dt=0.1,
        )

    def test_basic_creation(self):
        """Test basic ScenarioContext creation."""
        ctx = self.create_simple_scenario()

        assert ctx.scenario_id == "test_scenario"
        assert ctx.n_frames == 50
        assert len(ctx.agents) == 1
        assert ctx.dt == 0.1

    def test_map_property_alias(self):
        """Test that ctx.map returns ctx.map_context."""
        ctx = self.create_simple_scenario()

        assert ctx.map is ctx.map_context
        assert ctx.map.has_lane_geometry is True

    def test_vehicle_filtering(self):
        """Test filtering agents by type."""
        ego = EgoState(
            x=np.arange(10),
            y=np.zeros(10),
            yaw=np.zeros(10),
            speed=np.ones(10),
        )

        agents = [
            Agent(
                id=1,
                type="vehicle",
                x=np.arange(10),
                y=np.zeros(10),
                yaw=np.zeros(10),
                speed=np.ones(10),
                length=4.5,
                width=2.0,
            ),
            Agent(
                id=2,
                type="pedestrian",
                x=np.arange(10),
                y=np.ones(10) * 5,
                yaw=np.zeros(10),
                speed=np.ones(10) * 1.4,
                length=0.5,
                width=0.5,
            ),
            Agent(
                id=3,
                type="cyclist",
                x=np.arange(10),
                y=np.ones(10) * 3,
                yaw=np.zeros(10),
                speed=np.ones(10) * 5,
                length=1.8,
                width=0.6,
            ),
            Agent(
                id=4,
                type="vehicle",
                x=np.arange(10),
                y=np.ones(10) * -3.5,
                yaw=np.zeros(10),
                speed=np.ones(10) * 12,
                length=4.5,
                width=2.0,
            ),
        ]

        map_ctx = MapContext(lane_center_xy=np.zeros((2, 2)))

        ctx = ScenarioContext(
            scenario_id="multi_agent",
            ego=ego,
            agents=agents,
            map_context=map_ctx,
        )

        assert len(ctx.vehicles) == 2
        assert len(ctx.pedestrians) == 1
        assert len(ctx.cyclists) == 1
        assert len(ctx.vrus) == 2  # pedestrian + cyclist

    def test_valid_agents_at(self):
        """Test getting valid agents at specific timestep."""
        ego = EgoState(
            x=np.arange(10),
            y=np.zeros(10),
            yaw=np.zeros(10),
            speed=np.ones(10),
        )

        # Agent with partial validity
        valid_mask = np.array(
            [True, True, True, False, False, True, True, True, True, True]
        )
        agent = Agent(
            id=1,
            type="vehicle",
            x=np.arange(10),
            y=np.zeros(10),
            yaw=np.zeros(10),
            speed=np.ones(10),
            length=4.5,
            width=2.0,
            valid=valid_mask,
        )

        map_ctx = MapContext(lane_center_xy=np.zeros((2, 2)))

        ctx = ScenarioContext(
            scenario_id="partial_valid",
            ego=ego,
            agents=[agent],
            map_context=map_ctx,
        )

        assert len(ctx.valid_agents_at(0)) == 1
        assert len(ctx.valid_agents_at(3)) == 0
        assert len(ctx.valid_agents_at(5)) == 1

    def test_has_signals(self):
        """Test signal availability check."""
        ctx = self.create_simple_scenario()
        assert ctx.has_signals is False

        # Create with signals
        ego = EgoState(
            x=np.arange(10),
            y=np.zeros(10),
            yaw=np.zeros(10),
            speed=np.ones(10),
        )

        signals = MapSignals(
            ego_lane_id=0,
            signal_state=np.ones(10),  # All green
        )

        map_ctx = MapContext(lane_center_xy=np.zeros((2, 2)))

        ctx_with_signals = ScenarioContext(
            scenario_id="with_signals",
            ego=ego,
            agents=[],
            map_context=map_ctx,
            signals=signals,
        )

        assert ctx_with_signals.has_signals is True


class TestMapSignals:
    """Tests for MapSignals class."""

    def test_basic_creation(self):
        """Test basic signal creation."""
        signals = MapSignals(
            ego_lane_id=0,
            signal_state=np.array([1, 1, 1, 2, 2, 2, 3, 3, 1, 1]),
        )

        assert signals.ego_lane_id == 0
        assert len(signals.signal_state) == 10

    def test_is_red_at(self):
        """Test red light detection."""
        # State 1 = ARROW_STOP (RED), State 4 = STOP (RED), State 6 = GO (GREEN)
        signals = MapSignals(
            ego_lane_id=0,
            signal_state=np.array([6, 6, 4, 4, 4, 6, 6]),  # G, G, R, R, R, G, G
        )

        assert signals.is_red_at(0) is False  # Green
        assert signals.is_red_at(2) is True  # Red
        assert signals.is_red_at(5) is False  # Green

    def test_is_green_at(self):
        """Test green light detection."""
        # State 6 = GO (GREEN), State 4 = STOP (RED)
        signals = MapSignals(
            ego_lane_id=0,
            signal_state=np.array([6, 6, 4, 4, 6]),  # G, G, R, R, G
        )

        assert signals.is_green_at(0) is True
        assert signals.is_green_at(2) is False
        assert signals.is_green_at(4) is True

    def test_boundary_access(self):
        """Test signal state access at boundaries."""
        signals = MapSignals(
            ego_lane_id=0,
            signal_state=np.array([1, 2, 3]),
        )

        assert signals.is_red_at(-1) is False  # Out of bounds
        assert signals.is_red_at(10) is False  # Out of bounds
        assert signals.state_at(-1) == 0  # UNKNOWN
        assert signals.state_at(10) == 0  # UNKNOWN


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
