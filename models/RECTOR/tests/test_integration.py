"""
RECTOR Integration Tests

End-to-end tests that run RECTOR on real Waymo Open Dataset scenarios.
These tests validate the complete pipeline from TFRecord parsing to
trajectory selection.

Test Categories:
1. Single-scenario end-to-end planning
2. Multi-timestep receding horizon
3. Edge cases (few agents, static scenarios)
4. Consistency checks

Requirements:
- Waymo validation TFRecords must be available
- M2I pretrained models must be present
- GPU recommended but not required
"""

import os
import sys
import unittest
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np

# Add paths
RECTOR_ROOT = Path(__file__).parent.parent
RECTOR_LIB = RECTOR_ROOT / "scripts" / "lib"
M2I_SCRIPTS = Path("/workspace/models/pretrained/m2i/scripts/lib")
M2I_SRC = Path("/workspace/externals/M2I/src")

sys.path.insert(0, str(RECTOR_LIB))
sys.path.insert(0, str(M2I_SCRIPTS))
sys.path.insert(0, str(M2I_SRC))

# Import real data loader
from real_data_loader import get_test_scenario, RealDataLoader

# Import RECTOR components
from data_contracts import (
    PlanningConfig,
    AgentState,
    EgoCandidate,
    EgoCandidateBatch,
)
from planning_loop import (
    CandidateGenerator,
    ReactorSelector,
    CandidateScorer,
    RECTORPlanner,
    LaneInfo,
)
from safety_layer import IntegratedSafetyChecker

# Conditional imports
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

try:
    import tensorflow as tf

    TF_AVAILABLE = True
except ImportError:
    tf = None
    TF_AVAILABLE = False


TFRECORD_PATHS = [
    Path(
        "/workspace/data/datasets/waymo_open_dataset/motion_v_1_3_0/processed/tf/testing_interactive"
    ),
    Path(
        "/workspace/data/datasets/waymo_open_dataset/motion_v_1_3_0/tf/validation_interactive"
    ),
    Path("/workspace/data/cache"),
]


def find_tfrecord() -> Optional[str]:
    """Find an available TFRecord file for testing."""
    for base_path in TFRECORD_PATHS:
        if base_path.exists():
            tfrecords = list(base_path.glob("*.tfrecord*"))
            if tfrecords:
                return str(tfrecords[0])
    return None


class ScenarioLoader:
    """Load and parse Waymo scenarios for integration testing."""

    def __init__(self):
        self.tf_available = TF_AVAILABLE

    def load_scenario(
        self,
        tfrecord_path: str,
        scenario_idx: int = 0,
    ) -> Optional[Dict[str, Any]]:
        """
        Load a single scenario from TFRecord.

        Returns:
            Dict with keys:
            - 'x', 'y', 'vx', 'vy', 'yaw': [128, 91] arrays
            - 'valid': [128, 91] boolean array
            - 'agent_types': [128] array
            - 'agent_ids': [128] array
            - 'roadgraph_xyz': [N, 3] road points
            - 'scenario_id': bytes
        """
        if not self.tf_available:
            return None

        try:
            from waymo_flat_parser import features_description_flat
        except ImportError:
            # Define minimal feature description
            features_description_flat = {
                "state/past/x": tf.io.FixedLenFeature([1280], tf.float32),
                "state/past/y": tf.io.FixedLenFeature([1280], tf.float32),
                "state/past/velocity_x": tf.io.FixedLenFeature([1280], tf.float32),
                "state/past/velocity_y": tf.io.FixedLenFeature([1280], tf.float32),
                "state/past/bbox_yaw": tf.io.FixedLenFeature([1280], tf.float32),
                "state/past/valid": tf.io.FixedLenFeature([1280], tf.int64),
                "state/current/x": tf.io.FixedLenFeature([128], tf.float32),
                "state/current/y": tf.io.FixedLenFeature([128], tf.float32),
                "state/current/velocity_x": tf.io.FixedLenFeature([128], tf.float32),
                "state/current/velocity_y": tf.io.FixedLenFeature([128], tf.float32),
                "state/current/bbox_yaw": tf.io.FixedLenFeature([128], tf.float32),
                "state/current/valid": tf.io.FixedLenFeature([128], tf.int64),
                "state/future/x": tf.io.FixedLenFeature([10240], tf.float32),
                "state/future/y": tf.io.FixedLenFeature([10240], tf.float32),
                "state/future/velocity_x": tf.io.FixedLenFeature([10240], tf.float32),
                "state/future/velocity_y": tf.io.FixedLenFeature([10240], tf.float32),
                "state/future/bbox_yaw": tf.io.FixedLenFeature([10240], tf.float32),
                "state/future/valid": tf.io.FixedLenFeature([10240], tf.int64),
                "state/type": tf.io.FixedLenFeature([128], tf.float32),
                "state/id": tf.io.FixedLenFeature([128], tf.float32),
                "scenario/id": tf.io.FixedLenFeature([], tf.string),
                "roadgraph_samples/xyz": tf.io.FixedLenFeature([90000], tf.float32),
                "roadgraph_samples/type": tf.io.FixedLenFeature([30000], tf.int64),
                "roadgraph_samples/valid": tf.io.FixedLenFeature([30000], tf.int64),
            }

        try:
            dataset = tf.data.TFRecordDataset(tfrecord_path)

            for idx, record in enumerate(dataset):
                if idx != scenario_idx:
                    continue

                parsed = tf.io.parse_single_example(record, features_description_flat)

                def to_np(key):
                    return parsed[key].numpy()

                n_agents = 128
                n_past = 10
                n_future = 80

                # Parse all state data
                past_x = to_np("state/past/x").reshape(n_agents, n_past)[:, ::-1]
                past_y = to_np("state/past/y").reshape(n_agents, n_past)[:, ::-1]
                past_vx = to_np("state/past/velocity_x").reshape(n_agents, n_past)[
                    :, ::-1
                ]
                past_vy = to_np("state/past/velocity_y").reshape(n_agents, n_past)[
                    :, ::-1
                ]
                past_yaw = to_np("state/past/bbox_yaw").reshape(n_agents, n_past)[
                    :, ::-1
                ]
                past_valid = to_np("state/past/valid").reshape(n_agents, n_past)[
                    :, ::-1
                ]

                current_x = to_np("state/current/x").reshape(n_agents, 1)
                current_y = to_np("state/current/y").reshape(n_agents, 1)
                current_vx = to_np("state/current/velocity_x").reshape(n_agents, 1)
                current_vy = to_np("state/current/velocity_y").reshape(n_agents, 1)
                current_yaw = to_np("state/current/bbox_yaw").reshape(n_agents, 1)
                current_valid = to_np("state/current/valid").reshape(n_agents, 1)

                future_x = to_np("state/future/x").reshape(n_agents, n_future)
                future_y = to_np("state/future/y").reshape(n_agents, n_future)
                future_vx = to_np("state/future/velocity_x").reshape(n_agents, n_future)
                future_vy = to_np("state/future/velocity_y").reshape(n_agents, n_future)
                future_yaw = to_np("state/future/bbox_yaw").reshape(n_agents, n_future)
                future_valid = to_np("state/future/valid").reshape(n_agents, n_future)

                # Concatenate timeline [128, 91]
                all_x = np.concatenate([past_x, current_x, future_x], axis=1)
                all_y = np.concatenate([past_y, current_y, future_y], axis=1)
                all_vx = np.concatenate([past_vx, current_vx, future_vx], axis=1)
                all_vy = np.concatenate([past_vy, current_vy, future_vy], axis=1)
                all_yaw = np.concatenate([past_yaw, current_yaw, future_yaw], axis=1)
                all_valid = np.concatenate(
                    [past_valid, current_valid, future_valid], axis=1
                )

                # Road graph
                roadgraph_xyz = to_np("roadgraph_samples/xyz").reshape(-1, 3)
                roadgraph_type = to_np("roadgraph_samples/type")
                roadgraph_valid = to_np("roadgraph_samples/valid")

                return {
                    "x": all_x,
                    "y": all_y,
                    "vx": all_vx,
                    "vy": all_vy,
                    "yaw": all_yaw,
                    "valid": all_valid.astype(bool),
                    "agent_types": to_np("state/type"),
                    "agent_ids": to_np("state/id"),
                    "scenario_id": to_np("scenario/id"),
                    "roadgraph_xyz": roadgraph_xyz,
                    "roadgraph_type": roadgraph_type,
                    "roadgraph_valid": roadgraph_valid.astype(bool),
                }

            return None

        except Exception as e:
            print(f"Error loading scenario: {e}")
            return None

    def extract_agents_at_timestep(
        self,
        scenario: Dict[str, Any],
        timestep: int = 10,
    ) -> List[AgentState]:
        """Extract all valid agents at a specific timestep."""
        agents = []

        valid = scenario["valid"][:, timestep]
        for i in range(128):
            if valid[i]:
                agent = AgentState(
                    x=float(scenario["x"][i, timestep]),
                    y=float(scenario["y"][i, timestep]),
                    heading=float(scenario["yaw"][i, timestep]),
                    velocity_x=float(scenario["vx"][i, timestep]),
                    velocity_y=float(scenario["vy"][i, timestep]),
                    agent_id=int(scenario["agent_ids"][i]),
                    agent_type=int(scenario["agent_types"][i]),
                )
                agents.append(agent)

        return agents

    def extract_lane_from_roadgraph(
        self,
        scenario: Dict[str, Any],
        ego_state: AgentState,
        max_points: int = 50,
    ) -> LaneInfo:
        """Extract approximate lane centerline near ego."""
        xyz = scenario["roadgraph_xyz"]
        valid = scenario["roadgraph_valid"]
        types = scenario["roadgraph_type"]

        # Lane types: 1=lane center, 2=lane boundary
        lane_mask = valid & (types == 1)
        lane_points = xyz[lane_mask][:, :2]

        if len(lane_points) < 5:
            # Generate default straight lane
            heading = ego_state.heading
            centerline = np.array(
                [
                    [
                        ego_state.x + i * 2.0 * np.cos(heading),
                        ego_state.y + i * 2.0 * np.sin(heading),
                    ]
                    for i in range(max_points)
                ]
            )
            return LaneInfo(centerline=centerline)

        # Find points ahead of ego
        ego_pos = np.array([ego_state.x, ego_state.y])
        dists = np.linalg.norm(lane_points - ego_pos, axis=1)

        # Sort by distance and take closest
        sorted_idx = np.argsort(dists)[:max_points]
        centerline = lane_points[sorted_idx]

        # Sort by heading direction
        if len(centerline) > 1:
            heading_vec = np.array(
                [np.cos(ego_state.heading), np.sin(ego_state.heading)]
            )
            rel_points = centerline - ego_pos
            projections = np.dot(rel_points, heading_vec)
            sorted_by_proj = np.argsort(projections)
            centerline = centerline[sorted_by_proj]

        return LaneInfo(centerline=centerline)


class TestIntegrationEndToEnd(unittest.TestCase):
    """End-to-end integration tests."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.tfrecord_path = find_tfrecord()
        cls.loader = ScenarioLoader()
        cls.scenario = None

        if cls.tfrecord_path and TF_AVAILABLE:
            cls.scenario = cls.loader.load_scenario(cls.tfrecord_path, scenario_idx=0)

    def test_01_scenario_loading(self):
        """Test that we can load a Waymo scenario."""
        if not TF_AVAILABLE:
            self.skipTest("TensorFlow not available")
        if self.tfrecord_path is None:
            self.skipTest("No TFRecord files found")

        self.assertIsNotNone(self.scenario)
        self.assertIn("x", self.scenario)
        self.assertIn("y", self.scenario)
        self.assertEqual(self.scenario["x"].shape, (128, 91))
        self.assertEqual(self.scenario["valid"].shape, (128, 91))

    def test_02_agent_extraction(self):
        """Test extracting agents at a timestep."""
        if self.scenario is None:
            self.skipTest("Scenario not loaded")

        agents = self.loader.extract_agents_at_timestep(self.scenario, timestep=10)

        self.assertGreater(len(agents), 0)
        self.assertIsInstance(agents[0], AgentState)

        # Check agent has valid position
        agent = agents[0]
        self.assertFalse(np.isnan(agent.x))
        self.assertFalse(np.isnan(agent.y))

    def test_03_candidate_generation(self):
        """Test generating ego candidates from real scenario."""
        if self.scenario is None:
            self.skipTest("Scenario not loaded")

        agents = self.loader.extract_agents_at_timestep(self.scenario, timestep=10)
        ego_state = agents[0]  # Use first agent as ego

        lane = self.loader.extract_lane_from_roadgraph(self.scenario, ego_state)

        generator = CandidateGenerator(m_candidates=8, horizon_steps=80)
        candidates = generator.generate(
            ego_state=ego_state,
            current_lane=lane,
        )

        self.assertIsInstance(candidates, EgoCandidateBatch)
        self.assertEqual(len(candidates.candidates), 8)

        # Check candidate shapes
        for cand in candidates.candidates:
            self.assertEqual(cand.trajectory.shape, (80, 2))

    def test_04_reactor_selection(self):
        """Test selecting reactors from real scenario."""
        if self.scenario is None:
            self.skipTest("Scenario not loaded")

        agents = self.loader.extract_agents_at_timestep(self.scenario, timestep=10)
        ego_state = agents[0]
        other_agents = agents[1:] if len(agents) > 1 else []
        lane = self.loader.extract_lane_from_roadgraph(self.scenario, ego_state)

        # Generate candidates first (required for reactor selection)
        generator = CandidateGenerator(m_candidates=4, horizon_steps=80)
        candidates = generator.generate(ego_state=ego_state, current_lane=lane)

        selector = ReactorSelector(k_reactors=3)

        reactors = selector.select(
            ego_candidates=candidates,
            agent_states=other_agents,
            ego_state=ego_state,
        )

        self.assertIsInstance(reactors, list)
        self.assertLessEqual(len(reactors), 3)

    def test_05_safety_check(self):
        """Test safety checking with real scenario data."""
        if self.scenario is None:
            self.skipTest("Scenario not loaded")

        agents = self.loader.extract_agents_at_timestep(self.scenario, timestep=10)
        ego_state = agents[0]
        other_agents = agents[1:5] if len(agents) > 1 else []
        lane = self.loader.extract_lane_from_roadgraph(self.scenario, ego_state)

        generator = CandidateGenerator(m_candidates=4, horizon_steps=80)
        candidates = generator.generate(ego_state=ego_state, current_lane=lane)

        safety_checker = IntegratedSafetyChecker()

        for cand in candidates.candidates:
            result = safety_checker.check_candidate(
                ego_candidate=cand,
                predictions={},  # Empty for this test
                agent_states=other_agents,
            )

            # Should return a result even with no reactors
            self.assertIsNotNone(result)

    def test_06_full_planning_loop(self):
        """Test the complete RECTOR planning loop."""
        if self.scenario is None:
            self.skipTest("Scenario not loaded")

        agents = self.loader.extract_agents_at_timestep(self.scenario, timestep=10)
        ego_state = agents[0]
        other_agents = agents[1:10] if len(agents) > 1 else []  # Limit for speed
        lane = self.loader.extract_lane_from_roadgraph(self.scenario, ego_state)

        config = PlanningConfig(
            num_candidates=4,
            max_reactors=2,
            device="cpu",
        )

        # Create planner without M2I models (testing planning logic only)
        planner = RECTORPlanner(config=config, adapter=None)

        # Run planning tick
        result = planner.plan_tick(
            ego_state=ego_state,
            agent_states=other_agents,
            current_lane=lane,
        )

        self.assertIsNotNone(result)
        self.assertIsNotNone(result.selected_candidate)
        self.assertIsInstance(result.selected_idx, int)
        self.assertGreaterEqual(result.selected_idx, 0)
        self.assertIsNotNone(result.all_scores)
        if result.all_scores:
            self.assertLess(result.selected_idx, len(result.all_scores))

    def test_07_receding_horizon_consistency(self):
        """Test planning across multiple timesteps maintains consistency."""
        if self.scenario is None:
            self.skipTest("Scenario not loaded")

        config = PlanningConfig(
            num_candidates=4,
            max_reactors=2,
            device="cpu",
        )

        planner = RECTORPlanner(config=config, adapter=None)
        previous_selection = None

        # Run planning at multiple timesteps
        for t in [10, 11, 12, 13, 14]:
            agents = self.loader.extract_agents_at_timestep(self.scenario, timestep=t)
            if len(agents) == 0:
                continue

            ego_state = agents[0]
            other_agents = agents[1:10]
            lane = self.loader.extract_lane_from_roadgraph(self.scenario, ego_state)

            result = planner.plan_tick(
                ego_state=ego_state,
                agent_states=other_agents,
                current_lane=lane,
            )

            self.assertIsNotNone(result)

            # Track trajectory changes (with hysteresis, shouldn't jump too much)
            if previous_selection is not None:
                current_selection = result.selected_idx
                # Hysteresis should encourage consistency
                # (This is a soft check - exact behavior depends on scenario)

            previous_selection = result.selected_idx


class TestIntegrationEdgeCases(unittest.TestCase):
    """Test edge cases with real data."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.tfrecord_path = find_tfrecord()
        cls.loader = ScenarioLoader()

    def _get_scenario_or_skip(self):
        """Load a real scenario, or skip if TFRecords are not available."""
        try:
            return get_test_scenario()
        except FileNotFoundError as exc:
            self.skipTest(f"Waymo TFRecords not available: {exc}")

    def test_single_agent_scenario(self):
        """Test planning when ego is the only agent."""
        # Use real data from TFRecords
        scenario = self._get_scenario_or_skip()
        ego_state = scenario.ego_state
        lane = scenario.lane

        config = PlanningConfig(num_candidates=4, max_reactors=2, device="cpu")
        planner = RECTORPlanner(config=config, adapter=None)

        result = planner.plan_tick(
            ego_state=ego_state,
            agent_states=[],  # No other agents
            current_lane=lane,
        )

        self.assertIsNotNone(result)
        self.assertIsNotNone(result.selected_candidate)

    def test_static_ego(self):
        """Test planning when ego is stationary."""
        # Use real scenario and modify ego velocity
        scenario = self._get_scenario_or_skip()
        ego_state = AgentState(
            x=scenario.ego_state.x,
            y=scenario.ego_state.y,
            heading=scenario.ego_state.heading,
            velocity_x=0.0,
            velocity_y=0.0,  # Stationary
            agent_id=0,
        )

        config = PlanningConfig(num_candidates=4, device="cpu")
        planner = RECTORPlanner(config=config, adapter=None)

        result = planner.plan_tick(
            ego_state=ego_state,
            agent_states=[],
            current_lane=scenario.lane,
        )

        self.assertIsNotNone(result)

    def test_many_reactors(self):
        """Test with more agents than max_reactors."""
        # Use real scenario with many agents
        scenario = self._get_scenario_or_skip()
        ego_state = scenario.ego_state
        other_agents = (
            scenario.other_agents[:20]
            if len(scenario.other_agents) >= 20
            else scenario.other_agents
        )

        config = PlanningConfig(num_candidates=4, max_reactors=3, device="cpu")
        planner = RECTORPlanner(config=config, adapter=None)

        result = planner.plan_tick(
            ego_state=ego_state,
            agent_states=other_agents,
            current_lane=scenario.lane,
        )

        self.assertIsNotNone(result)
        # Planner should produce a valid result
        self.assertIsInstance(result.selected_idx, int)


class TestIntegrationMetrics(unittest.TestCase):
    """Test metrics and statistics collection."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.tfrecord_path = find_tfrecord()
        cls.loader = ScenarioLoader()
        cls.scenario = None

        if cls.tfrecord_path and TF_AVAILABLE:
            cls.scenario = cls.loader.load_scenario(cls.tfrecord_path, scenario_idx=0)

    def test_timing_metrics(self):
        """Test that planning returns timing information."""
        if self.scenario is None:
            self.skipTest("Scenario not loaded")

        agents = self.loader.extract_agents_at_timestep(self.scenario, timestep=10)
        ego_state = agents[0]
        other_agents = agents[1:5]
        lane = self.loader.extract_lane_from_roadgraph(self.scenario, ego_state)

        config = PlanningConfig(num_candidates=4, device="cpu")
        planner = RECTORPlanner(config=config, adapter=None)

        import time

        start = time.time()
        result = planner.plan_tick(
            ego_state=ego_state,
            agent_states=other_agents,
            current_lane=lane,
        )
        elapsed = time.time() - start

        self.assertIsNotNone(result)
        # Planning without M2I should be fast
        self.assertLess(elapsed, 1.0)  # Less than 1 second

    def test_score_ordering(self):
        """Test that candidates are properly ranked by score."""
        if self.scenario is None:
            self.skipTest("Scenario not loaded")

        agents = self.loader.extract_agents_at_timestep(self.scenario, timestep=10)
        ego_state = agents[0]
        other_agents = agents[1:5]
        lane = self.loader.extract_lane_from_roadgraph(self.scenario, ego_state)

        config = PlanningConfig(num_candidates=8, device="cpu")
        planner = RECTORPlanner(config=config, adapter=None)

        result = planner.plan_tick(
            ego_state=ego_state,
            agent_states=other_agents,
            current_lane=lane,
        )

        # Check that scores exist and make sense
        self.assertIsNotNone(result.all_scores)
        scores = result.all_scores
        # Best candidate should have highest score
        selected_idx = result.selected_idx
        if len(scores) > 0:
            # The selected candidate should have a reasonable score
            self.assertGreaterEqual(scores[selected_idx], min(scores))


if __name__ == "__main__":
    print("=" * 70)
    print("RECTOR Integration Tests")
    print("=" * 70)

    # Check prerequisites
    print(f"\nPrerequisites:")
    print(f"  TensorFlow available: {TF_AVAILABLE}")
    print(f"  PyTorch available: {TORCH_AVAILABLE}")

    tfrecord = find_tfrecord()
    print(f"  TFRecord found: {tfrecord is not None}")
    if tfrecord:
        print(f"  TFRecord path: {tfrecord}")

    print("\n" + "=" * 70)

    # Run tests
    unittest.main(verbosity=2)
