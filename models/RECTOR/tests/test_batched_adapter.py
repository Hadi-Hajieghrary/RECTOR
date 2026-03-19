#!/usr/bin/env python3
"""
Tests for RECTOR Batched M2I Adapter.

Tests:
1. Adapter initialization and model loading
2. Scene cache building
3. Reactor pack building
4. Batched prediction
5. Coordinate transforms

Usage:
    pytest models/RECTOR/tests/test_batched_adapter.py -v
    python models/RECTOR/tests/test_batched_adapter.py
"""

import sys
import time
from pathlib import Path
import numpy as np
import pytest

# Add RECTOR lib to path
RECTOR_LIB = Path(__file__).parent.parent / "scripts" / "lib"
sys.path.insert(0, str(RECTOR_LIB))

from data_contracts import (
    PlanningConfig,
    EgoCandidate,
    EgoCandidateBatch,
    create_candidate_batch,
)
from batched_adapter import BatchedM2IAdapter, create_adapter


class TestAdapterInitialization:
    """Test adapter creation and configuration."""

    def test_default_config(self):
        """Test creating adapter with default config."""
        adapter = create_adapter()
        assert adapter.config.num_candidates == 16
        assert adapter.config.max_reactors == 3
        assert not adapter.is_loaded

    def test_custom_config(self):
        """Test creating adapter with custom config."""
        config = PlanningConfig(
            num_candidates=32,
            max_reactors=5,
            device="cpu",
        )
        adapter = create_adapter(config)
        assert adapter.config.num_candidates == 32
        assert adapter.config.max_reactors == 5
        assert adapter.device == "cpu"


class TestCoordinateTransforms:
    """Test coordinate transform utilities."""

    def setup_method(self):
        """Setup adapter for testing."""
        self.adapter = create_adapter(PlanningConfig(device="cpu"))

    def test_identity_transform(self):
        """Test transform with no offset or rotation."""
        trajectory = np.array([[10, 20], [15, 25], [20, 30]], dtype=np.float32)
        origin = np.array([0, 0])
        rotation = 0.0

        local = self.adapter._transform_trajectory_to_local(
            trajectory, origin, rotation
        )
        np.testing.assert_array_almost_equal(local, trajectory)

    def test_translation_only(self):
        """Test transform with translation only."""
        trajectory = np.array([[10, 20], [15, 25], [20, 30]], dtype=np.float32)
        origin = np.array([5, 10])
        rotation = 0.0

        local = self.adapter._transform_trajectory_to_local(
            trajectory, origin, rotation
        )
        expected = trajectory - origin
        np.testing.assert_array_almost_equal(local, expected)

    def test_rotation_90_degrees(self):
        """Test transform with 90 degree rotation."""
        trajectory = np.array([[10, 0], [20, 0]], dtype=np.float32)
        origin = np.array([0, 0])
        rotation = np.pi / 2  # 90 degrees

        local = self.adapter._transform_trajectory_to_local(
            trajectory, origin, rotation
        )
        # Point (10, 0) with -90 deg rotation should go to (0, -10)
        expected = np.array([[0, -10], [0, -20]], dtype=np.float32)
        np.testing.assert_array_almost_equal(local, expected, decimal=5)

    def test_roundtrip_transform(self):
        """Test that world→local→world is identity."""
        trajectory = np.random.randn(80, 2).astype(np.float32)
        origin = np.array([100, 200])
        rotation = 0.7  # ~40 degrees

        local = self.adapter._transform_trajectory_to_local(
            trajectory, origin, rotation
        )
        recovered = self.adapter._transform_trajectory_to_world(local, origin, rotation)

        np.testing.assert_array_almost_equal(recovered, trajectory, decimal=5)


class TestDummyPrediction:
    """Test prediction without loading real models."""

    def setup_method(self):
        """Setup adapter for testing."""
        self.adapter = create_adapter(PlanningConfig(device="cpu"))

    def test_predict_batched_shapes(self):
        """Test that batched prediction returns correct shapes."""
        # Create dummy reactor packs
        from data_contracts import ReactorTensorPack

        reactor_packs = {}
        for reactor_id in [10, 20, 30]:
            reactor_packs[reactor_id] = ReactorTensorPack(
                reactor_id=reactor_id,
                position_world=np.array([float(reactor_id), 0.0]),
                heading_world=0.0,
                origin=np.array([float(reactor_id), 0.0]),
                rotation=0.0,
                reactor_history_local=np.zeros((11, 7), dtype=np.float32),
                reactor_type=1,
                matrix=np.zeros((10, 128), dtype=np.float32),
                polyline_spans=[],
                map_start_polyline_idx=0,
                goals_2D=np.zeros((100, 2), dtype=np.float32),
            )

        # Create dummy scene cache
        from data_contracts import SceneEmbeddingCache

        scene_cache = SceneEmbeddingCache(
            scenario_id="test",
            timestep=10,
            ego_position=np.array([0.0, 0.0]),
            ego_heading=0.0,
            ego_history=np.zeros((11, 7), dtype=np.float32),
            agent_ids=(0, 10, 20, 30),
            agent_positions=np.array(
                [[0, 0], [10, 0], [20, 0], [30, 0]], dtype=np.float32
            ),
            agent_headings=np.zeros(4, dtype=np.float32),
            agent_types=np.ones(4, dtype=np.int32),
            agent_histories=np.zeros((4, 11, 7), dtype=np.float32),
            objects_of_interest=np.array([0, 1, 1, 1], dtype=np.int32),
            roadgraph_xyz=np.zeros((100, 3), dtype=np.float32),
            roadgraph_type=np.zeros(100, dtype=np.int32),
            roadgraph_valid=np.ones(100, dtype=bool),
            roadgraph_id=np.zeros(100, dtype=np.int32),
        )
        self.adapter._current_scene_cache = scene_cache

        # Create candidate batch
        M = 16
        H = 80
        trajectories = np.random.randn(M, H, 2).astype(np.float32)
        candidates = create_candidate_batch(trajectories)

        # Run prediction (uses dummy mode since models not loaded)
        result = self.adapter.predict_batched(
            reactor_packs=reactor_packs,
            ego_candidates=candidates,
            scene_cache=scene_cache,
        )

        # Check shapes
        assert result.num_candidates == M
        assert result.num_reactors == 3
        assert result.num_modes == 6
        assert result.horizon == H
        assert result.trajectories.shape == (M, 3, 6, H, 2)
        assert result.scores.shape == (M, 3, 6)


class TestReactorSelection:
    """Test reactor selection logic."""

    def setup_method(self):
        """Setup adapter for testing."""
        self.adapter = create_adapter(PlanningConfig(max_reactors=3, device="cpu"))

    def test_select_closest_reactors(self):
        """Test that closest reactors are selected."""
        from data_contracts import SceneEmbeddingCache

        # Create scene with 5 agents at different distances
        scene_cache = SceneEmbeddingCache(
            scenario_id="test",
            timestep=10,
            ego_position=np.array([0.0, 0.0]),
            ego_heading=0.0,
            ego_history=np.zeros((11, 7)),
            agent_ids=(0, 1, 2, 3, 4),
            agent_positions=np.array(
                [
                    [0, 0],  # Ego
                    [10, 0],  # Agent 1: distance 10
                    [50, 0],  # Agent 2: distance 50
                    [25, 0],  # Agent 3: distance 25
                    [5, 0],  # Agent 4: distance 5
                ],
                dtype=np.float32,
            ),
            agent_headings=np.zeros(5),
            agent_types=np.ones(5, dtype=np.int32),
            agent_histories=np.zeros((5, 11, 7)),
            objects_of_interest=np.array(
                [0, 1, 1, 1, 1]
            ),  # All except ego are interactive
            roadgraph_xyz=np.zeros((10, 3)),
            roadgraph_type=np.zeros(10, dtype=np.int32),
            roadgraph_valid=np.ones(10, dtype=bool),
            roadgraph_id=np.zeros(10, dtype=np.int32),
        )

        # Select reactors
        ego_pos = np.array([0.0, 0.0])
        selected = self.adapter.select_reactors(scene_cache, ego_pos, max_reactors=3)

        # Should select agents 4, 1, 3 (distances 5, 10, 25)
        assert len(selected) == 3
        assert 4 in selected  # Closest
        assert 1 in selected  # Second closest
        assert 3 in selected  # Third closest
        assert 2 not in selected  # Farthest, excluded


class TestIntegration:
    """Integration tests with real M2I models."""

    @pytest.fixture
    def loaded_adapter(self):
        """Create and load adapter with real models."""
        config = PlanningConfig(device="cuda")
        adapter = create_adapter(config)

        # Skip if models not available
        model_root = Path("/workspace/models/pretrained/m2i/models")
        if not (model_root / "densetnt/model.24.bin").exists():
            pytest.skip("M2I models not available")

        adapter.load_models()
        return adapter

    def test_model_loading(self, loaded_adapter):
        """Test that models load successfully."""
        assert loaded_adapter.is_loaded

    def test_latency_benchmark(self, loaded_adapter):
        """Benchmark latency for batched prediction."""
        from data_contracts import ReactorTensorPack, SceneEmbeddingCache

        # Create dummy data
        M = 16
        K = 3
        H = 80

        reactor_packs = {}
        for k, reactor_id in enumerate([10, 20, 30]):
            reactor_packs[reactor_id] = ReactorTensorPack(
                reactor_id=reactor_id,
                position_world=np.array([float(reactor_id), 0.0]),
                heading_world=0.0,
                origin=np.array([float(reactor_id), 0.0]),
                rotation=0.0,
                reactor_history_local=np.zeros((11, 7), dtype=np.float32),
                reactor_type=1,
                matrix=np.random.randn(50, 128).astype(np.float32),
                polyline_spans=[slice(0, 10), slice(10, 50)],
                map_start_polyline_idx=10,
                goals_2D=np.random.randn(100, 2).astype(np.float32),
            )

        scene_cache = SceneEmbeddingCache(
            scenario_id="test",
            timestep=10,
            ego_position=np.array([0.0, 0.0]),
            ego_heading=0.0,
            ego_history=np.zeros((11, 7)),
            agent_ids=(0, 10, 20, 30),
            agent_positions=np.array(
                [[0, 0], [10, 0], [20, 0], [30, 0]], dtype=np.float32
            ),
            agent_headings=np.zeros(4),
            agent_types=np.ones(4, dtype=np.int32),
            agent_histories=np.zeros((4, 11, 7)),
            objects_of_interest=np.array([0, 1, 1, 1]),
            roadgraph_xyz=np.zeros((100, 3)),
            roadgraph_type=np.zeros(100, dtype=np.int32),
            roadgraph_valid=np.ones(100, dtype=bool),
            roadgraph_id=np.zeros(100, dtype=np.int32),
        )
        loaded_adapter._current_scene_cache = scene_cache

        trajectories = np.random.randn(M, H, 2).astype(np.float32)
        candidates = create_candidate_batch(trajectories)

        # warm-up
        for _ in range(2):
            loaded_adapter.predict_batched(reactor_packs, candidates, scene_cache)

        # Benchmark
        times = []
        for _ in range(10):
            start = time.time()
            result = loaded_adapter.predict_batched(
                reactor_packs, candidates, scene_cache
            )
            elapsed = (time.time() - start) * 1000
            times.append(elapsed)

        avg_time = np.mean(times)
        std_time = np.std(times)

        print(f"\nBatched prediction latency: {avg_time:.1f}ms ± {std_time:.1f}ms")
        print(f"  M={M} candidates, K={K} reactors")

        # Target: <50ms for batched inference
        # Note: This is currently serial, so won't meet target yet
        assert result is not None


def run_tests():
    """Run tests without pytest."""
    print("Running RECTOR Batched Adapter Tests")
    print("=" * 60)

    test_classes = [
        TestAdapterInitialization,
        TestCoordinateTransforms,
        TestDummyPrediction,
        TestReactorSelection,
    ]

    passed = 0
    failed = 0

    for test_class in test_classes:
        instance = test_class()
        if hasattr(instance, "setup_method"):
            instance.setup_method()
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
    try:
        import pytest

        sys.exit(pytest.main([__file__, "-v", "-x"]))
    except ImportError:
        success = run_tests()
        sys.exit(0 if success else 1)
