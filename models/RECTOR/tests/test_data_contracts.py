#!/usr/bin/env python3
"""
Unit tests for RECTOR data contracts.

Tests that:
1. All dataclasses are properly frozen (immutable)
2. Type annotations are correct
3. Factory functions work
4. Coordinate transforms are correct
5. Tensor shapes are validated

Usage:
    pytest models/RECTOR/tests/test_data_contracts.py -v
    python models/RECTOR/tests/test_data_contracts.py
"""

import sys
from pathlib import Path
import numpy as np
import pytest

# Add RECTOR lib to path
RECTOR_LIB = Path(__file__).parent.parent / "scripts" / "lib"
sys.path.insert(0, str(RECTOR_LIB))

from data_contracts import (
    PlanningConfig,
    AgentState,
    AgentHistory,
    EgoCandidate,
    EgoCandidateBatch,
    ReactorTensorPack,
    SceneEmbeddingCache,
    SinglePrediction,
    PredictionResult,
    CollisionCheckResult,
    SafetyScore,
    PlanningResult,
    create_ego_candidate,
    create_candidate_batch,
)


class TestPlanningConfig:
    """Tests for PlanningConfig."""

    def test_default_values(self):
        """Test that default config is valid."""
        config = PlanningConfig()
        assert config.num_candidates == 16
        assert config.max_reactors == 3
        assert config.candidate_horizon == 80
        assert config.device == "cuda"

    def test_custom_values(self):
        """Test creating config with custom values."""
        config = PlanningConfig(
            num_candidates=32,
            max_reactors=5,
            cvar_alpha=0.05,
        )
        assert config.num_candidates == 32
        assert config.max_reactors == 5
        assert config.cvar_alpha == 0.05

    def test_immutability(self):
        """Test that config is immutable (frozen)."""
        config = PlanningConfig()
        # FrozenInstanceError is a subclass of AttributeError (Python ≥ 3.11)
        # or raises TypeError in some dataclass implementations
        with pytest.raises((AttributeError, TypeError)):
            config.num_candidates = 32

    def test_validation(self):
        """Test config validation."""
        config = PlanningConfig()
        assert config.validate() is True


class TestAgentState:
    """Tests for AgentState."""

    def test_creation(self):
        """Test creating an agent state."""
        state = AgentState(
            x=100.0,
            y=200.0,
            heading=0.5,
            velocity_x=10.0,
            velocity_y=2.0,
        )
        assert state.x == 100.0
        assert state.y == 200.0
        assert state.heading == 0.5

    def test_position_property(self):
        """Test position property."""
        state = AgentState(
            x=100.0, y=200.0, heading=0.5, velocity_x=10.0, velocity_y=2.0
        )
        pos = state.position
        assert isinstance(pos, np.ndarray)
        assert pos.shape == (2,)
        np.testing.assert_array_almost_equal(pos, [100.0, 200.0])

    def test_velocity_property(self):
        """Test velocity property."""
        state = AgentState(x=0.0, y=0.0, heading=0.0, velocity_x=10.0, velocity_y=2.0)
        vel = state.velocity
        assert isinstance(vel, np.ndarray)
        np.testing.assert_array_almost_equal(vel, [10.0, 2.0])

    def test_speed_property(self):
        """Test speed calculation."""
        state = AgentState(x=0.0, y=0.0, heading=0.0, velocity_x=3.0, velocity_y=4.0)
        assert state.speed == pytest.approx(5.0)

    def test_immutability(self):
        """Test that state is immutable."""
        state = AgentState(x=0.0, y=0.0, heading=0.0, velocity_x=0.0, velocity_y=0.0)
        with pytest.raises(Exception):
            state.x = 10.0


class TestEgoCandidate:
    """Tests for EgoCandidate."""

    def test_creation(self):
        """Test creating an ego candidate."""
        trajectory = np.random.randn(80, 2).astype(np.float32)
        candidate = EgoCandidate(
            candidate_id=0,
            trajectory=trajectory,
        )
        assert candidate.candidate_id == 0
        assert candidate.horizon == 80

    def test_start_end_positions(self):
        """Test start and end position properties."""
        trajectory = np.array([[0, 0], [1, 1], [2, 2]], dtype=np.float32)
        candidate = EgoCandidate(candidate_id=0, trajectory=trajectory)

        np.testing.assert_array_equal(candidate.start_position, [0, 0])
        np.testing.assert_array_equal(candidate.end_position, [2, 2])

    def test_get_position_at(self):
        """Test position indexing."""
        trajectory = np.array([[i, i * 2] for i in range(10)], dtype=np.float32)
        candidate = EgoCandidate(candidate_id=0, trajectory=trajectory)

        np.testing.assert_array_equal(candidate.get_position_at(5), [5, 10])

    def test_travel_distance(self):
        """Test travel distance computation."""
        # Straight line from (0,0) to (10,0)
        trajectory = np.array([[i, 0] for i in range(11)], dtype=np.float32)
        candidate = EgoCandidate(candidate_id=0, trajectory=trajectory)

        assert candidate.compute_travel_distance() == pytest.approx(10.0)

    def test_factory_function(self):
        """Test create_ego_candidate factory."""
        trajectory = np.random.randn(80, 2)
        candidate = create_ego_candidate(
            trajectory=trajectory,
            candidate_id=5,
            generation_method="optimization",
        )
        assert candidate.candidate_id == 5
        assert candidate.generation_method == "optimization"
        assert candidate.trajectory.dtype == np.float32


class TestEgoCandidateBatch:
    """Tests for EgoCandidateBatch."""

    def test_from_candidates(self):
        """Test creating batch from list of candidates."""
        candidates = [
            create_ego_candidate(np.random.randn(80, 2), candidate_id=i)
            for i in range(16)
        ]
        batch = EgoCandidateBatch.from_candidates(candidates)

        assert batch.num_candidates == 16
        assert batch.horizon == 80
        assert batch.trajectories_tensor.shape == (16, 80, 2)

    def test_factory_function(self):
        """Test create_candidate_batch factory."""
        trajectories = np.random.randn(16, 80, 2)
        batch = create_candidate_batch(trajectories)

        assert batch.num_candidates == 16
        assert len(batch.candidates) == 16


class TestReactorTensorPack:
    """Tests for ReactorTensorPack."""

    def create_sample_pack(self):
        """Create a sample reactor pack for testing."""
        return ReactorTensorPack(
            reactor_id=1,
            position_world=np.array([100.0, 200.0]),
            heading_world=np.pi / 4,  # 45 degrees
            origin=np.array([100.0, 200.0]),
            rotation=np.pi / 4,
            reactor_history_local=np.random.randn(11, 7).astype(np.float32),
            reactor_type=1,
            matrix=np.random.randn(50, 128).astype(np.float32),
            polyline_spans=[slice(0, 10), slice(10, 20)],
            map_start_polyline_idx=5,
            goals_2D=np.random.randn(100, 2).astype(np.float32),
        )

    def test_creation(self):
        """Test creating a reactor pack."""
        pack = self.create_sample_pack()
        assert pack.reactor_id == 1
        assert pack.reactor_type == 1

    def test_transform_to_local_point(self):
        """Test transforming a single point to local frame."""
        pack = ReactorTensorPack(
            reactor_id=1,
            position_world=np.array([0.0, 0.0]),
            heading_world=0.0,  # No rotation
            origin=np.array([0.0, 0.0]),
            rotation=0.0,
            reactor_history_local=np.zeros((11, 7)),
            reactor_type=1,
            matrix=np.zeros((1, 128)),
            polyline_spans=[],
            map_start_polyline_idx=0,
            goals_2D=np.zeros((1, 2)),
        )

        # With no transform, world = local
        world_point = np.array([10.0, 5.0])
        local_point = pack.transform_to_local(world_point)
        np.testing.assert_array_almost_equal(local_point, [10.0, 5.0])

    def test_transform_with_rotation(self):
        """Test transform with 90 degree rotation."""
        pack = ReactorTensorPack(
            reactor_id=1,
            position_world=np.array([0.0, 0.0]),
            heading_world=np.pi / 2,  # 90 degrees
            origin=np.array([0.0, 0.0]),
            rotation=np.pi / 2,
            reactor_history_local=np.zeros((11, 7)),
            reactor_type=1,
            matrix=np.zeros((1, 128)),
            polyline_spans=[],
            map_start_polyline_idx=0,
            goals_2D=np.zeros((1, 2)),
        )

        # Point at (10, 0) in world should be (0, -10) in local
        # (reactor is facing +Y, so +X in world is -Y in local)
        world_point = np.array([10.0, 0.0])
        local_point = pack.transform_to_local(world_point)
        np.testing.assert_array_almost_equal(local_point, [0.0, -10.0], decimal=5)

    def test_transform_roundtrip(self):
        """Test that world→local→world is identity."""
        pack = self.create_sample_pack()

        world_point = np.array([150.0, 250.0])
        local_point = pack.transform_to_local(world_point)
        recovered = pack.transform_to_world(local_point)

        np.testing.assert_array_almost_equal(recovered, world_point, decimal=5)

    def test_transform_batch(self):
        """Test transforming multiple points."""
        pack = ReactorTensorPack(
            reactor_id=1,
            position_world=np.array([100.0, 100.0]),
            heading_world=0.0,
            origin=np.array([100.0, 100.0]),
            rotation=0.0,
            reactor_history_local=np.zeros((11, 7)),
            reactor_type=1,
            matrix=np.zeros((1, 128)),
            polyline_spans=[],
            map_start_polyline_idx=0,
            goals_2D=np.zeros((1, 2)),
        )

        world_points = np.array([[110, 120], [130, 140], [150, 160]], dtype=np.float32)
        local_points = pack.transform_to_local(world_points)

        expected = np.array([[10, 20], [30, 40], [50, 60]], dtype=np.float32)
        np.testing.assert_array_almost_equal(local_points, expected)


class TestPredictionResult:
    """Tests for PredictionResult."""

    def test_creation(self):
        """Test creating a prediction result."""
        M, K, N_modes, H = 16, 3, 6, 80
        result = PredictionResult(
            trajectories=np.random.randn(M, K, N_modes, H, 2).astype(np.float32),
            scores=np.random.rand(M, K, N_modes).astype(np.float32),
            ego_candidate_ids=tuple(range(M)),
            reactor_ids=(10, 20, 30),
        )

        assert result.num_candidates == M
        assert result.num_reactors == K
        assert result.num_modes == N_modes
        assert result.horizon == H

    def test_get_prediction(self):
        """Test getting single prediction."""
        M, K, N_modes, H = 4, 2, 6, 80
        result = PredictionResult(
            trajectories=np.random.randn(M, K, N_modes, H, 2).astype(np.float32),
            scores=np.random.rand(M, K, N_modes).astype(np.float32),
            ego_candidate_ids=tuple(range(M)),
            reactor_ids=(10, 20),
        )

        pred = result.get_prediction(1, 0)
        assert isinstance(pred, SinglePrediction)
        assert pred.ego_candidate_id == 1
        assert pred.reactor_id == 10
        assert pred.trajectories.shape == (N_modes, H, 2)

    def test_get_best_trajectories(self):
        """Test getting best trajectories."""
        M, K, N_modes, H = 4, 2, 6, 80
        scores = np.random.rand(M, K, N_modes).astype(np.float32)

        result = PredictionResult(
            trajectories=np.random.randn(M, K, N_modes, H, 2).astype(np.float32),
            scores=scores,
            ego_candidate_ids=tuple(range(M)),
            reactor_ids=(10, 20),
        )

        best = result.get_best_trajectories()
        assert best.shape == (M, K, H, 2)


class TestSafetyScore:
    """Tests for SafetyScore."""

    def test_safe_candidate(self):
        """Test a safe candidate."""
        collision_results = (
            CollisionCheckResult(has_collision=False, min_distance=10.0),
            CollisionCheckResult(has_collision=False, min_distance=15.0),
        )

        safety = SafetyScore(
            candidate_id=0,
            collision_results=collision_results,
            has_any_collision=False,
            min_clearance=10.0,
            worst_reactor_id=None,
            cvar_risk=0.1,
            safety_score=0.9,
        )

        assert safety.is_safe is True

    def test_unsafe_candidate(self):
        """Test an unsafe candidate."""
        collision_results = (
            CollisionCheckResult(
                has_collision=True, collision_time=50, min_distance=0.0
            ),
        )

        safety = SafetyScore(
            candidate_id=0,
            collision_results=collision_results,
            has_any_collision=True,
            min_clearance=0.0,
            worst_reactor_id=1,
            cvar_risk=0.9,
            safety_score=0.1,
        )

        assert safety.is_safe is False


def run_tests():
    """Run all tests without pytest."""
    print("Running RECTOR Data Contract Tests")
    print("=" * 60)

    test_classes = [
        TestPlanningConfig,
        TestAgentState,
        TestEgoCandidate,
        TestEgoCandidateBatch,
        TestReactorTensorPack,
        TestPredictionResult,
        TestSafetyScore,
    ]

    passed = 0
    failed = 0

    for test_class in test_classes:
        instance = test_class()
        class_name = test_class.__name__

        for method_name in dir(instance):
            if method_name.startswith("test_"):
                try:
                    getattr(instance, method_name)()
                    print(f"  ✓ {class_name}.{method_name}")
                    passed += 1
                except Exception as e:
                    print(f"  ✗ {class_name}.{method_name}: {e}")
                    failed += 1

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")

    return failed == 0


if __name__ == "__main__":
    # Check if pytest is available
    try:
        import pytest

        sys.exit(pytest.main([__file__, "-v"]))
    except ImportError:
        # Fallback to manual test runner
        success = run_tests()
        sys.exit(0 if success else 1)
