"""
Tests for RECTOR Safety Layer (Phase 5)

Tests the safety layer components:
1. RealityChecker - Kinematic feasibility
2. NonCoopEnvelopeGenerator - Conservative envelopes
3. OBBCollisionChecker - Oriented bounding box collision
4. DeterministicCVaRScorer - Risk scoring
5. IntegratedSafetyChecker - Complete safety check
"""

import sys
import pytest
import numpy as np
from pathlib import Path

# Add RECTOR lib to path
RECTOR_LIB = Path(__file__).parent.parent / "scripts" / "lib"
sys.path.insert(0, str(RECTOR_LIB))

from safety_layer import (
    KinematicLimits,
    RealityChecker,
    FeasibilityResult,
    NonCoopParams,
    NonCoopEnvelopeGenerator,
    BoundingBox,
    OBBCollisionChecker,
    CVaRConfig,
    DeterministicCVaRScorer,
    IntegratedSafetyChecker,
    create_safety_checker,
)
from data_contracts import (
    AgentState,
    EgoCandidate,
    SinglePrediction,
)


class TestRealityChecker:
    """Tests for kinematic feasibility checker."""

    @pytest.fixture
    def checker(self):
        return RealityChecker()

    def test_feasible_straight_trajectory(self, checker):
        """Test that a smooth straight trajectory is feasible."""
        # 10 m/s straight trajectory
        traj = np.array([[i * 1.0, 0.0] for i in range(80)])
        result = checker.check_trajectory(traj)

        assert result.is_feasible
        assert result.max_speed < 15.0  # ~10 m/s

    def test_infeasible_teleport(self, checker):
        """Test that a teleporting trajectory is infeasible."""
        traj = np.zeros((80, 2))
        traj[:40] = [[i * 1.0, 0.0] for i in range(40)]
        traj[40:] = [[100.0 + i * 1.0, 0.0] for i in range(40)]  # Jump!

        result = checker.check_trajectory(traj)
        # Should detect extreme acceleration
        assert result.max_a_lon > 50.0 or not result.is_feasible

    def test_infeasible_excessive_speed(self, checker):
        """Test detection of excessive speed."""
        # 50 m/s trajectory (above 40 m/s limit)
        traj = np.array([[i * 5.0, 0.0] for i in range(80)])
        result = checker.check_trajectory(traj)

        assert result.violations["speed_exceeded"]
        assert not result.is_feasible

    def test_infeasible_harsh_braking(self, checker):
        """Test detection of excessive braking."""
        # Start at 30 m/s, brake to 0 in 1 second (-30 m/s² >> -8 m/s² limit)
        traj = np.zeros((80, 2))
        speed = 30.0
        x = 0.0
        for t in range(80):
            if t < 10:
                speed = 30.0 - 3.0 * t  # -30 m/s² for 1 second
            else:
                speed = max(0, speed)
            x += speed * 0.1
            traj[t] = [x, 0.0]

        result = checker.check_trajectory(traj)

        assert result.min_a_lon < -8.0  # Exceeds braking limit
        assert result.violations["a_lon_too_low"]

    def test_short_trajectory(self, checker):
        """Test that short trajectories are handled."""
        traj = np.array([[0.0, 0.0], [1.0, 0.0]])
        result = checker.check_trajectory(traj)

        assert result.is_feasible

    def test_batch_check(self, checker):
        """Test batch checking of multiple predictions."""
        predictions = np.random.randn(3, 4, 40, 2) * 0.5
        # Make trajectories smooth
        for k in range(3):
            for n in range(4):
                predictions[k, n] = np.cumsum(predictions[k, n], axis=0) * 0.1

        valid_mask, penalties = checker.check_batch(predictions)

        assert valid_mask.shape == (3, 4)
        assert isinstance(
            penalties, (float, np.floating)
        )  # Scalar for [K, N, H, 2] input


class TestNonCoopEnvelope:
    """Tests for non-cooperative envelope generator."""

    @pytest.fixture
    def generator(self):
        return NonCoopEnvelopeGenerator()

    def test_envelope_shape(self, generator):
        """Test envelope output shape."""
        agent = AgentState(x=0.0, y=0.0, heading=0.0, velocity_x=10.0, velocity_y=0.0)
        envelope = generator.generate_envelope(agent, horizon=80, num_samples=5)

        assert envelope.shape == (5, 80, 2)

    def test_constant_velocity_sample(self, generator):
        """Test that first sample is constant velocity."""
        agent = AgentState(x=0.0, y=0.0, heading=0.0, velocity_x=10.0, velocity_y=0.0)
        envelope = generator.generate_envelope(agent, horizon=40, num_samples=1)

        # Should travel ~40m in 4 seconds at 10 m/s
        final_x = envelope[0, -1, 0]
        assert 38.0 < final_x < 42.0

    def test_braking_sample(self, generator):
        """Test that braking sample travels less distance."""
        agent = AgentState(x=0.0, y=0.0, heading=0.0, velocity_x=10.0, velocity_y=0.0)
        envelope = generator.generate_envelope(agent, horizon=40, num_samples=5)

        const_vel_distance = np.linalg.norm(envelope[0, -1])
        braking_distance = np.linalg.norm(envelope[1, -1])  # Max braking

        assert braking_distance < const_vel_distance

    def test_worst_case_extraction(self, generator):
        """Test worst-case trajectory extraction."""
        agent = AgentState(x=50.0, y=0.0, heading=0.0, velocity_x=-10.0, velocity_y=0.0)
        envelope = generator.generate_envelope(agent, horizon=40)

        ego_traj = np.array([[i * 1.0, 0.0] for i in range(40)])
        worst_case = generator.get_worst_case(envelope, ego_traj)

        assert worst_case.shape == (40, 2)


class TestOBBCollision:
    """Tests for oriented bounding box collision detection."""

    @pytest.fixture
    def checker(self):
        return OBBCollisionChecker(ego_length=4.5, ego_width=2.0, safety_margin=0.0)

    def test_no_collision_far_apart(self, checker):
        """Test no collision when vehicles are far apart."""
        collides = checker.check_collision(
            ego_pos=np.array([0.0, 0.0]),
            ego_heading=0.0,
            other_pos=np.array([20.0, 0.0]),
            other_heading=0.0,
        )
        assert not collides

    def test_collision_overlapping(self, checker):
        """Test collision when vehicles overlap."""
        collides = checker.check_collision(
            ego_pos=np.array([0.0, 0.0]),
            ego_heading=0.0,
            other_pos=np.array([3.0, 0.0]),  # Within bounding boxes
            other_heading=0.0,
        )
        assert collides

    def test_collision_at_angle(self, checker):
        """Test collision detection with angled vehicles."""
        collides = checker.check_collision(
            ego_pos=np.array([0.0, 0.0]),
            ego_heading=0.0,
            other_pos=np.array([4.0, 2.0]),
            other_heading=np.pi / 4,  # 45 degrees
        )
        # Should collide when boxes overlap
        assert collides

    def test_no_collision_perpendicular(self, checker):
        """Test no collision when vehicles are perpendicular but not overlapping."""
        collides = checker.check_collision(
            ego_pos=np.array([0.0, 0.0]),
            ego_heading=0.0,
            other_pos=np.array([0.0, 5.0]),  # Far enough lateral
            other_heading=np.pi / 2,
        )
        assert not collides

    def test_trajectory_collision(self, checker):
        """Test collision detection over trajectories."""
        # Ego going straight
        ego_traj = np.array([[i * 1.0, 0.0] for i in range(40)])

        # Other crossing path
        other_traj = np.array([[15.0, 10.0 - i * 0.5] for i in range(40)])

        has_collision, collision_time, min_dist = checker.check_trajectory_collision(
            ego_traj, other_traj
        )

        # ego moves along x at 1 m/step, other moves along -y from (15, 10).
        # They cross near t=15 (ego reaches x=15) so collision should be in [10, 25].
        # Bounds are derived from vehicle dimensions (~±5 steps of margin).
        if has_collision:
            num_steps = len(ego_traj)
            assert (
                0 <= collision_time < num_steps
            ), f"collision_time {collision_time} out of bounds"
            assert 10 <= collision_time <= 25

        assert min_dist < 10.0  # Should get close

    def test_trajectory_no_collision(self, checker):
        """Test no collision when trajectories don't intersect."""
        ego_traj = np.array([[i * 1.0, 0.0] for i in range(40)])
        other_traj = np.array(
            [[i * 1.0, 50.0] for i in range(40)]
        )  # Parallel, 50m apart

        has_collision, collision_time, min_dist = checker.check_trajectory_collision(
            ego_traj, other_traj
        )

        assert not has_collision
        assert collision_time is None
        assert min_dist > 40.0


class TestCVaRScorer:
    """Tests for deterministic CVaR risk scoring."""

    @pytest.fixture
    def scorer(self):
        return DeterministicCVaRScorer(CVaRConfig(seed=42))

    def test_low_risk_distant_reactor(self, scorer):
        """Test low risk score for distant reactor."""
        ego_traj = np.array([[i * 1.0, 0.0] for i in range(40)])

        # Reactor far away
        reactor_traj = np.zeros((6, 40, 2))
        for mode in range(6):
            for t in range(40):
                reactor_traj[mode, t] = [100.0, 50.0]

        predictions = {
            1: SinglePrediction(
                ego_candidate_id=0,
                reactor_id=1,
                trajectories=reactor_traj,
                scores=np.array([0.4, 0.2, 0.15, 0.1, 0.1, 0.05]),
            )
        }

        risk = scorer.score_candidate(ego_traj, predictions)
        assert risk < 10.0  # Low risk

    def test_high_risk_collision_course(self, scorer):
        """Test high risk score for collision course."""
        ego_traj = np.array([[i * 1.0, 0.0] for i in range(40)])

        # Reactor on collision course
        reactor_traj = np.zeros((6, 40, 2))
        for mode in range(6):
            for t in range(40):
                reactor_traj[mode, t] = [t * 1.0, 0.0]  # Same trajectory!

        predictions = {
            1: SinglePrediction(
                ego_candidate_id=0,
                reactor_id=1,
                trajectories=reactor_traj,
                scores=np.array([0.4, 0.2, 0.15, 0.1, 0.1, 0.05]),
            )
        }

        risk = scorer.score_candidate(ego_traj, predictions)
        assert risk > 50.0  # High risk

    def test_deterministic_with_seed(self, scorer):
        """Test that same seed produces same results."""
        ego_traj = np.array([[i * 1.0, 0.0] for i in range(40)])

        reactor_traj = np.zeros((6, 40, 2))
        for mode in range(6):
            for t in range(40):
                reactor_traj[mode, t] = [20.0 + mode, 5.0]

        predictions = {
            1: SinglePrediction(
                ego_candidate_id=0,
                reactor_id=1,
                trajectories=reactor_traj,
                scores=np.array([0.4, 0.2, 0.15, 0.1, 0.1, 0.05]),
            )
        }

        scorer1 = DeterministicCVaRScorer(CVaRConfig(seed=42))
        scorer2 = DeterministicCVaRScorer(CVaRConfig(seed=42))

        risk1 = scorer1.score_candidate(ego_traj, predictions)
        risk2 = scorer2.score_candidate(ego_traj, predictions)

        assert risk1 == risk2  # Same seed, same result


class TestIntegratedSafetyChecker:
    """Tests for integrated safety checking."""

    @pytest.fixture
    def checker(self):
        return create_safety_checker()

    def test_safe_candidate(self, checker):
        """Test that a safe candidate passes all checks."""
        candidate = EgoCandidate(
            candidate_id=0,
            trajectory=np.array([[i * 1.0, 0.0] for i in range(40)]),
        )

        # Distant reactor
        reactor_traj = np.zeros((6, 40, 2))
        for mode in range(6):
            for t in range(40):
                reactor_traj[mode, t] = [100.0, 50.0]

        predictions = {
            1: SinglePrediction(
                ego_candidate_id=0,
                reactor_id=1,
                trajectories=reactor_traj,
                scores=np.array([0.4, 0.2, 0.15, 0.1, 0.1, 0.05]),
            )
        }

        agents = [
            AgentState(
                x=100.0, y=50.0, heading=0.0, velocity_x=0.0, velocity_y=0.0, agent_id=1
            )
        ]

        result = checker.check_candidate(candidate, predictions, agents)

        assert result.is_safe
        assert result.feasibility.is_feasible
        assert not result.has_collision

    def test_unsafe_collision(self, checker):
        """Test that collision is detected."""
        candidate = EgoCandidate(
            candidate_id=0,
            trajectory=np.array([[i * 1.0, 0.0] for i in range(40)]),
        )

        # Reactor on collision course
        reactor_traj = np.zeros((6, 40, 2))
        for mode in range(6):
            for t in range(40):
                reactor_traj[mode, t] = [t * 1.0, 0.0]  # Same path

        predictions = {
            1: SinglePrediction(
                ego_candidate_id=0,
                reactor_id=1,
                trajectories=reactor_traj,
                scores=np.array([0.4, 0.2, 0.15, 0.1, 0.1, 0.05]),
            )
        }

        agents = [
            AgentState(
                x=0.0, y=0.0, heading=0.0, velocity_x=10.0, velocity_y=0.0, agent_id=1
            )
        ]

        result = checker.check_candidate(candidate, predictions, agents)

        assert not result.is_safe
        assert result.has_collision
        assert result.collision_time is not None


class TestBoundingBox:
    """Tests for BoundingBox class."""

    def test_get_corners_axis_aligned(self):
        """Test corner computation for axis-aligned box."""
        box = BoundingBox(
            center=np.array([0.0, 0.0]),
            half_size=np.array([2.0, 1.0]),
            heading=0.0,
        )
        corners = box.get_corners()

        assert corners.shape == (4, 2)
        # Corners should be at (-2, -1), (2, -1), (2, 1), (-2, 1)
        assert np.allclose(sorted(corners[:, 0]), [-2, -2, 2, 2])
        assert np.allclose(sorted(corners[:, 1]), [-1, -1, 1, 1])

    def test_get_corners_rotated(self):
        """Test corner computation for rotated box."""
        box = BoundingBox(
            center=np.array([0.0, 0.0]),
            half_size=np.array([2.0, 1.0]),
            heading=np.pi / 2,  # 90 degrees
        )
        corners = box.get_corners()

        # After 90 degree rotation, corners should be at (-1, -2), (-1, 2), (1, 2), (1, -2)
        assert np.allclose(sorted(corners[:, 0]), [-1, -1, 1, 1], atol=1e-6)
        assert np.allclose(sorted(corners[:, 1]), [-2, -2, 2, 2], atol=1e-6)


class TestPerformance:
    """Performance benchmarks for safety layer."""

    def test_reality_check_speed(self):
        """Test reality check performance."""
        import time

        checker = RealityChecker()
        traj = np.random.randn(80, 2).cumsum(axis=0)

        times = []
        for _ in range(100):
            t0 = time.perf_counter()
            checker.check_trajectory(traj)
            times.append((time.perf_counter() - t0) * 1000)

        avg_time = np.mean(times)
        print(f"\nReality check avg time: {avg_time:.3f}ms")

        assert avg_time < 1.0  # Should be < 1ms

    def test_collision_check_speed(self):
        """Test OBB collision check performance."""
        import time

        checker = OBBCollisionChecker()
        ego_traj = np.random.randn(80, 2).cumsum(axis=0)
        other_traj = np.random.randn(80, 2).cumsum(axis=0) + [20, 0]

        times = []
        for _ in range(100):
            t0 = time.perf_counter()
            checker.check_trajectory_collision(ego_traj, other_traj)
            times.append((time.perf_counter() - t0) * 1000)

        avg_time = np.mean(times)
        print(f"\nCollision check avg time: {avg_time:.3f}ms")

        assert avg_time < 10.0  # Should be < 10ms


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
