"""
Tests for RECTOR Planning Loop (Phase 4)

Tests the planning loop components:
1. Candidate Generator
2. Reactor Selector
3. Candidate Scorer
4. RECTORPlanner (integration)
"""

import sys
import pytest
import numpy as np
from pathlib import Path

# Add RECTOR lib to path
RECTOR_LIB = Path(__file__).parent.parent / "scripts" / "lib"
sys.path.insert(0, str(RECTOR_LIB))

from planning_loop import (
    CandidateGenerator,
    ReactorSelector,
    CandidateScorer,
    ScoringWeights,
    RECTORPlanner,
    LaneInfo,
    SimpleSafetyScore,
    create_planner,
)
from data_contracts import (
    PlanningConfig,
    AgentState,
    EgoCandidate,
    EgoCandidateBatch,
    SinglePrediction,
)


class TestCandidateGenerator:
    """Tests for CandidateGenerator."""

    def test_generate_default_candidates(self):
        """Test generating candidates without lane info."""
        generator = CandidateGenerator(m_candidates=16, horizon_steps=80, dt=0.1)
        ego = AgentState(x=0.0, y=0.0, heading=0.0, velocity_x=10.0, velocity_y=0.0)

        candidates = generator.generate(ego_state=ego)

        assert isinstance(candidates, EgoCandidateBatch)
        assert candidates.num_candidates == 16
        assert candidates.horizon == 80

    def test_trajectory_starts_at_ego_position(self):
        """Test that trajectories start near ego position."""
        generator = CandidateGenerator(m_candidates=8, horizon_steps=40, dt=0.1)
        ego = AgentState(x=100.0, y=50.0, heading=0.5, velocity_x=15.0, velocity_y=0.0)

        candidates = generator.generate(ego_state=ego)

        for candidate in candidates.candidates:
            start_pos = candidate.trajectory[0]
            # Should be close to ego position
            dist = np.linalg.norm(start_pos - np.array([ego.x, ego.y]))
            assert dist < 5.0, f"Start position too far from ego: {dist}m"

    def test_lane_following_candidates(self):
        """Test lane-following candidate generation."""
        generator = CandidateGenerator(m_candidates=8, horizon_steps=40, dt=0.1)
        ego = AgentState(x=0.0, y=0.0, heading=0.0, velocity_x=10.0, velocity_y=0.0)

        # Simple straight lane
        lane = LaneInfo(
            centerline=np.array([[i * 10, 0] for i in range(20)]),
            speed_limit=20.0,
        )

        candidates = generator.generate(ego_state=ego, current_lane=lane)

        assert candidates.num_candidates == 8

        # At least one candidate should follow the lane approximately
        best_lane_follow = None
        min_deviation = float("inf")

        for candidate in candidates.candidates:
            # Measure average lateral deviation from y=0
            avg_deviation = np.mean(np.abs(candidate.trajectory[:, 1]))
            if avg_deviation < min_deviation:
                min_deviation = avg_deviation
                best_lane_follow = candidate

        # Best candidate should have low deviation
        assert min_deviation < 2.0

    def test_with_goal(self):
        """Test goal-directed candidate generation."""
        generator = CandidateGenerator(m_candidates=10, horizon_steps=40, dt=0.1)
        ego = AgentState(x=0.0, y=0.0, heading=0.0, velocity_x=10.0, velocity_y=0.0)
        goal = np.array([100.0, 20.0])

        candidates = generator.generate(ego_state=ego, goal=goal)

        # At least one candidate should move toward goal
        progress_made = False
        for candidate in candidates.candidates:
            start_dist = np.linalg.norm(candidate.trajectory[0] - goal)
            end_dist = np.linalg.norm(candidate.trajectory[-1] - goal)
            if end_dist < start_dist:
                progress_made = True
                break

        assert progress_made, "No candidate makes progress toward goal"


class TestReactorSelector:
    """Tests for ReactorSelector."""

    def test_select_closest_agents(self):
        """Test that closest agents are selected."""
        selector = ReactorSelector(k_reactors=3)

        ego = AgentState(x=0.0, y=0.0, heading=0.0, velocity_x=10.0, velocity_y=0.0)

        agents = [
            AgentState(
                x=100.0, y=0.0, heading=0.0, velocity_x=10.0, velocity_y=0.0, agent_id=1
            ),
            AgentState(
                x=20.0, y=0.0, heading=0.0, velocity_x=10.0, velocity_y=0.0, agent_id=2
            ),
            AgentState(
                x=50.0, y=0.0, heading=0.0, velocity_x=10.0, velocity_y=0.0, agent_id=3
            ),
            AgentState(
                x=200.0, y=0.0, heading=0.0, velocity_x=10.0, velocity_y=0.0, agent_id=4
            ),
            AgentState(
                x=30.0, y=0.0, heading=0.0, velocity_x=10.0, velocity_y=0.0, agent_id=5
            ),
        ]

        # Create simple candidates
        generator = CandidateGenerator(m_candidates=4, horizon_steps=40)
        candidates = generator.generate(ego_state=ego)

        selected = selector.select(
            ego_candidates=candidates,
            agent_states=agents,
            ego_state=ego,
        )

        assert len(selected) <= 3
        # Agent 2 (20m), 5 (30m), 3 (50m) should be prioritized
        assert 2 in selected  # Closest

    def test_empty_agents(self):
        """Test with no agents."""
        selector = ReactorSelector(k_reactors=3)
        ego = AgentState(x=0.0, y=0.0, heading=0.0, velocity_x=10.0, velocity_y=0.0)

        generator = CandidateGenerator(m_candidates=4, horizon_steps=40)
        candidates = generator.generate(ego_state=ego)

        selected = selector.select(
            ego_candidates=candidates,
            agent_states=[],
            ego_state=ego,
        )

        assert selected == []


class TestCandidateScorer:
    """Tests for CandidateScorer."""

    @pytest.fixture
    def scorer(self):
        return CandidateScorer()

    @pytest.fixture
    def simple_candidates(self):
        """Create simple test candidates."""
        generator = CandidateGenerator(m_candidates=4, horizon_steps=40)
        ego = AgentState(x=0.0, y=0.0, heading=0.0, velocity_x=10.0, velocity_y=0.0)
        return generator.generate(ego_state=ego)

    @pytest.fixture
    def simple_predictions(self, simple_candidates):
        """Create dummy predictions for testing."""
        predictions = {}
        reactor_id = 1
        pred_list = []

        for candidate in simple_candidates.candidates:
            # Create prediction far from ego
            H = candidate.horizon
            trajs = np.zeros((6, H, 2))
            for mode in range(6):
                for t in range(H):
                    trajs[mode, t, 0] = 50.0 + t * 0.5  # Far away
                    trajs[mode, t, 1] = 10.0

            pred_list.append(
                SinglePrediction(
                    ego_candidate_id=candidate.candidate_id,
                    reactor_id=reactor_id,
                    trajectories=trajs,
                    scores=np.array([0.4, 0.2, 0.15, 0.1, 0.1, 0.05]),
                )
            )

        predictions[reactor_id] = pred_list
        return predictions

    def test_score_all_basic(self, scorer, simple_candidates, simple_predictions):
        """Test basic scoring."""
        scores, safety = scorer.score_all(
            candidates=simple_candidates,
            predictions=simple_predictions,
        )

        assert len(scores) == simple_candidates.num_candidates
        assert len(safety) == simple_candidates.num_candidates

    def test_safe_candidates_flagged(
        self, scorer, simple_candidates, simple_predictions
    ):
        """Test that safe candidates are properly flagged."""
        scores, safety = scorer.score_all(
            candidates=simple_candidates,
            predictions=simple_predictions,
        )

        # All candidates should be safe (predictions are far away)
        num_safe = sum(s.is_safe for s in safety)
        assert num_safe > 0

    def test_goal_progress_affects_score(self, scorer, simple_candidates):
        """Test that goal progress affects scores."""
        # Create predictions
        predictions = {}
        reactor_id = 1
        pred_list = []
        for candidate in simple_candidates.candidates:
            H = candidate.horizon
            trajs = np.zeros((6, H, 2))
            for mode in range(6):
                for t in range(H):
                    trajs[mode, t] = [100.0, 50.0]  # Far away
            pred_list.append(
                SinglePrediction(
                    ego_candidate_id=candidate.candidate_id,
                    reactor_id=reactor_id,
                    trajectories=trajs,
                    scores=np.array([1.0, 0, 0, 0, 0, 0]),
                )
            )
        predictions[reactor_id] = pred_list

        goal = np.array([200.0, 0.0])

        scores_with_goal, _ = scorer.score_all(
            candidates=simple_candidates,
            predictions=predictions,
            goal=goal,
        )

        scores_no_goal, _ = scorer.score_all(
            candidates=simple_candidates,
            predictions=predictions,
        )

        # Scores should be different with goal
        assert not np.allclose(scores_with_goal, scores_no_goal)


class TestRECTORPlanner:
    """Integration tests for RECTORPlanner."""

    @pytest.fixture
    def planner(self):
        config = PlanningConfig()
        return create_planner(config)

    def test_plan_tick_basic(self, planner):
        """Test basic planning tick."""
        ego = AgentState(x=0.0, y=0.0, heading=0.0, velocity_x=10.0, velocity_y=0.0)
        agents = [
            AgentState(
                x=30.0, y=0.5, heading=0.0, velocity_x=8.0, velocity_y=0.0, agent_id=1
            ),
        ]

        result = planner.plan_tick(
            ego_state=ego,
            agent_states=agents,
        )

        assert result.selected_candidate is not None
        assert result.selected_idx >= 0
        assert len(result.all_scores) == 16

    def test_plan_tick_with_goal(self, planner):
        """Test planning with goal."""
        ego = AgentState(x=0.0, y=0.0, heading=0.0, velocity_x=10.0, velocity_y=0.0)
        agents = []
        goal = np.array([200.0, 0.0])

        result = planner.plan_tick(
            ego_state=ego,
            agent_states=agents,
            goal=goal,
        )

        # Should make progress toward goal
        selected = result.selected_candidate
        start_dist = np.linalg.norm(selected.trajectory[0] - goal)
        end_dist = np.linalg.norm(selected.trajectory[-1] - goal)

        assert end_dist < start_dist, "Selected candidate doesn't progress toward goal"

    def test_timing_reported(self, planner):
        """Test that timing is reported."""
        ego = AgentState(x=0.0, y=0.0, heading=0.0, velocity_x=10.0, velocity_y=0.0)

        result = planner.plan_tick(ego_state=ego, agent_states=[])

        assert "generate" in result.timing_ms
        assert "select_reactors" in result.timing_ms
        assert "predict" in result.timing_ms
        assert "score" in result.timing_ms
        assert "select" in result.timing_ms

        # All timings should be positive
        for key, value in result.timing_ms.items():
            assert value >= 0, f"Negative timing for {key}"

    def test_iteration_counter(self, planner):
        """Test iteration counter increments."""
        ego = AgentState(x=0.0, y=0.0, heading=0.0, velocity_x=10.0, velocity_y=0.0)

        result1 = planner.plan_tick(ego_state=ego, agent_states=[])
        result2 = planner.plan_tick(ego_state=ego, agent_states=[])
        result3 = planner.plan_tick(ego_state=ego, agent_states=[])

        assert result1.iteration == 0
        assert result2.iteration == 1
        assert result3.iteration == 2

    def test_hysteresis_prefers_previous(self, planner):
        """Test that hysteresis prefers previous selection when scores similar."""
        ego = AgentState(x=0.0, y=0.0, heading=0.0, velocity_x=10.0, velocity_y=0.0)

        # Run multiple ticks
        results = []
        for _ in range(5):
            result = planner.plan_tick(ego_state=ego, agent_states=[])
            results.append(result)

        # With consistent input, should prefer stable selection
        # (hysteresis should kick in)
        selected_indices = [r.selected_idx for r in results]

        # At least some consecutive selections should be the same
        consecutive_same = sum(
            1
            for i in range(len(selected_indices) - 1)
            if selected_indices[i] == selected_indices[i + 1]
        )

        assert consecutive_same >= 2, "Hysteresis not working - too much switching"


class TestPerformance:
    """Performance benchmarks for planning loop."""

    def test_planning_tick_latency(self):
        """Test that planning tick meets latency target."""
        import time

        planner = create_planner()
        ego = AgentState(x=0.0, y=0.0, heading=0.0, velocity_x=10.0, velocity_y=0.0)
        agents = [
            AgentState(
                x=30.0 + i * 10,
                y=i * 2,
                heading=0.0,
                velocity_x=10.0,
                velocity_y=0.0,
                agent_id=i,
            )
            for i in range(10)
        ]

        # warm-up
        for _ in range(3):
            planner.plan_tick(ego_state=ego, agent_states=agents)

        # Measure
        times = []
        for _ in range(10):
            t0 = time.perf_counter()
            planner.plan_tick(ego_state=ego, agent_states=agents)
            times.append((time.perf_counter() - t0) * 1000)

        avg_time = np.mean(times)
        print(f"\nAverage planning tick time: {avg_time:.2f}ms")

        # Target: <300ms without M2I (M2I adds ~50ms)
        assert avg_time < 500, f"Planning too slow: {avg_time:.2f}ms"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
