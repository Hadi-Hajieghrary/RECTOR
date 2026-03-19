"""
Tests for the pipeline module including window scheduling and execution.

Tests the new windowed execution pipeline:
- WindowSpec generation
- Window slicing/extraction
- WindowedExecutor
- Output sinks
"""

import json
import os
import tempfile

import numpy as np
import pytest

from waymo_rule_eval.core.context import (
    Agent,
    EgoState,
    MapContext,
    ScenarioContext,
)
from waymo_rule_eval.io import CsvSink, JsonlSink, make_sink
from waymo_rule_eval.pipeline import (
    WindowedExecutor,
    WindowResult,
    WindowSpec,
    extract_window,
    make_windows,
    make_windows_timed,
)


@pytest.fixture
def sample_timestamps():
    """10 seconds at 10Hz = 100 frames."""
    return np.arange(0, 10, 0.1)


@pytest.fixture
def long_timestamps():
    """20 seconds at 10Hz = 200 frames."""
    return np.arange(0, 20, 0.1)


@pytest.fixture
def sample_ego():
    """100-frame ego trajectory."""
    n = 100
    return EgoState(
        x=np.linspace(0, 100, n),  # Moving in +x at 10 m/s
        y=np.zeros(n),
        yaw=np.zeros(n),
        speed=np.full(n, 10.0),
        length=4.5,
        width=2.0,
    )


@pytest.fixture
def sample_scenario(sample_ego):
    """Complete 100-frame scenario."""
    n = 100

    # Add a vehicle agent
    vehicle = Agent(
        id=1,
        type="vehicle",
        x=np.linspace(30, 130, n),  # Moving ahead
        y=np.full(n, 3.5),  # In adjacent lane
        yaw=np.zeros(n),
        speed=np.full(n, 10.0),
        length=4.5,
        width=2.0,
    )

    # Add a pedestrian
    pedestrian = Agent(
        id=2,
        type="pedestrian",
        x=np.full(n, 50.0),  # Stationary
        y=np.linspace(10, -10, n),  # Crossing road
        yaw=np.full(n, -np.pi / 2),
        speed=np.full(n, 2.0),
        length=0.5,
        width=0.5,
    )

    return ScenarioContext(
        scenario_id="test_scenario",
        dt=0.1,
        ego=sample_ego,
        agents=[vehicle, pedestrian],
        map_context=MapContext(
            lane_center_xy=np.array([[0, 0], [100, 0]]),
            road_edges=[],
        ),
    )


class TestMakeWindows:
    """Tests for make_windows function."""

    def test_basic_window_generation(self, sample_timestamps):
        """Test basic window generation with default params."""
        windows = make_windows(sample_timestamps, window_size_s=5.0, stride_s=1.0)

        assert len(windows) > 0
        assert all(isinstance(w, WindowSpec) for w in windows)

    def test_window_size_correct(self, sample_timestamps):
        """Windows should have approximately correct duration."""
        windows = make_windows(sample_timestamps, window_size_s=5.0, stride_s=1.0)

        for w in windows[:-1]:  # Last might be shorter
            assert abs(w.duration_s - 5.0) < 0.2  # Allow some tolerance

    def test_window_stride_correct(self, sample_timestamps):
        """Windows should be spaced by stride."""
        windows = make_windows(sample_timestamps, window_size_s=3.0, stride_s=1.0)

        for i in range(1, len(windows)):
            stride = windows[i].start_ts - windows[i - 1].start_ts
            assert abs(stride - 1.0) < 0.2

    def test_window_indices_valid(self, sample_timestamps):
        """All window indices should be valid."""
        windows = make_windows(sample_timestamps, window_size_s=5.0, stride_s=1.0)
        n = len(sample_timestamps)

        for w in windows:
            assert 0 <= w.start_idx < n
            assert 0 < w.end_idx <= n
            assert w.start_idx < w.end_idx

    def test_minimum_frames_enforced(self, sample_timestamps):
        """Windows should have at least min_frames."""
        windows = make_windows(
            sample_timestamps, window_size_s=5.0, stride_s=1.0, min_frames=10
        )

        for w in windows:
            assert w.n_frames >= 10

    def test_empty_timestamps(self):
        """Empty timestamps should return empty list."""
        windows = make_windows(np.array([]), window_size_s=5.0, stride_s=1.0)
        assert windows == []

    def test_short_timestamps(self):
        """Timestamps shorter than window should produce limited windows."""
        timestamps = np.arange(0, 2, 0.1)  # Only 2 seconds
        windows = make_windows(timestamps, window_size_s=5.0, stride_s=1.0)

        # Should still produce at least one window if min_frames met
        assert len(windows) >= 1
        # Last window end should not exceed array length
        assert windows[-1].end_idx <= len(timestamps)

    def test_stride_larger_than_window(self, sample_timestamps):
        """Non-overlapping windows when stride > window."""
        windows = make_windows(sample_timestamps, window_size_s=2.0, stride_s=3.0)

        # Windows should not overlap
        for i in range(1, len(windows)):
            assert windows[i].start_idx >= windows[i - 1].end_idx


class TestMakeWindowsTimed:
    """Tests for make_windows_timed function."""

    def test_basic_usage(self):
        """Basic window generation from frame count."""
        windows = make_windows_timed(T=100, dt=0.1, window_size_s=5.0, stride_s=1.0)

        assert len(windows) > 0

    def test_matches_make_windows(self, sample_timestamps):
        """Should produce same result as make_windows with explicit timestamps."""
        windows_explicit = make_windows(
            sample_timestamps, window_size_s=5.0, stride_s=1.0
        )
        windows_timed = make_windows_timed(
            T=len(sample_timestamps), dt=0.1, window_size_s=5.0, stride_s=1.0
        )

        assert len(windows_explicit) == len(windows_timed)
        for we, wt in zip(windows_explicit, windows_timed):
            assert we.start_idx == wt.start_idx
            assert we.end_idx == wt.end_idx


class TestWindowSpec:
    """Tests for WindowSpec dataclass."""

    def test_n_frames_property(self):
        """n_frames should return correct count."""
        w = WindowSpec(start_idx=10, end_idx=60, start_ts=1.0, end_ts=6.0)
        assert w.n_frames == 50

    def test_duration_s_property(self):
        """duration_s should return correct duration."""
        w = WindowSpec(start_idx=0, end_idx=50, start_ts=0.0, end_ts=5.0)
        assert w.duration_s == 5.0


class TestExtractWindow:
    """Tests for extract_window function."""

    def test_basic_extraction(self, sample_scenario):
        """Basic window extraction."""
        windowed = extract_window(sample_scenario, 10, 30)

        assert len(windowed.ego.x) == 20
        assert len(windowed.agents) == 2
        assert all(len(a.x) == 20 for a in windowed.agents)

    def test_scenario_id_preserved(self, sample_scenario):
        """Scenario ID should be preserved."""
        windowed = extract_window(sample_scenario, 0, 50)
        assert windowed.scenario_id == sample_scenario.scenario_id

    def test_dt_preserved(self, sample_scenario):
        """dt should be preserved."""
        windowed = extract_window(sample_scenario, 0, 50)
        assert windowed.dt == sample_scenario.dt

    def test_map_shared(self, sample_scenario):
        """Map elements should be shared (not copied)."""
        windowed = extract_window(sample_scenario, 0, 50)
        assert windowed.map_context is sample_scenario.map_context

    def test_window_frame_count(self, sample_scenario):
        """Window should have correct frame count."""
        windowed = extract_window(sample_scenario, 10, 30)

        # Window should have 20 frames
        assert len(windowed.ego.x) == 20
        assert windowed.n_frames == 20

    def test_window_metadata(self, sample_scenario):
        """Window should have correct metadata."""
        windowed = extract_window(sample_scenario, 10, 30)

        assert windowed.window_start_ts is not None
        # window_start_ts should be 10 * 0.1 = 1.0 seconds
        assert abs(windowed.window_start_ts - 1.0) < 0.01
        assert windowed.window_size == 20

    def test_ego_data_correct(self, sample_scenario):
        """Ego trajectory should be correctly sliced."""
        windowed = extract_window(sample_scenario, 10, 30)

        # Check that x values are from the correct range
        # Original: x goes from 0 to 100 over 100 frames
        # Frames 10-30: x should be ~10 to ~30
        assert 9 < windowed.ego.x[0] < 12
        assert 28 < windowed.ego.x[-1] < 32

    def test_agent_data_correct(self, sample_scenario):
        """Agent trajectories should be correctly sliced."""
        windowed = extract_window(sample_scenario, 10, 30)

        # Vehicle agent: x from 30 to 130 over 100 frames
        # Frames 10-30: should be from ~40 to ~60
        vehicle = windowed.agents[0]
        assert 38 < vehicle.x[0] < 42
        assert 58 < vehicle.x[-1] < 62

    def test_extract_full_range(self, sample_scenario):
        """Extracting full range should work."""
        windowed = extract_window(sample_scenario, 0, 100)

        assert len(windowed.ego.x) == 100
        np.testing.assert_array_equal(windowed.ego.x, sample_scenario.ego.x)

    def test_extract_single_frame_fails(self, sample_scenario):
        """Single frame should still work but be minimal."""
        windowed = extract_window(sample_scenario, 50, 51)

        assert len(windowed.ego.x) == 1


class TestWindowedExecutor:
    """Tests for WindowedExecutor class."""

    def test_initialization(self):
        """Test executor initialization."""
        executor = WindowedExecutor(window_size_s=8.0, stride_s=2.0, dt=0.1)

        assert executor.window_size_s == 8.0
        assert executor.stride_s == 2.0
        assert executor.dt == 0.1

    def test_register_rules(self):
        """Test rule registration."""
        executor = WindowedExecutor()
        executor.register_all_rules()

        assert len(executor.rules) == 28

    def test_make_windows(self, sample_scenario):
        """Test window generation for scenario."""
        executor = WindowedExecutor(window_size_s=3.0, stride_s=1.0)

        windows = executor.make_windows(sample_scenario)

        assert len(windows) > 0
        # Each window is (idx, start, end)
        assert all(len(w) == 3 for w in windows)

    def test_evaluate_single_window(self, sample_scenario):
        """Test evaluating a single window."""
        executor = WindowedExecutor(window_size_s=3.0, stride_s=1.0)
        executor.register_all_rules()

        result = executor.evaluate_window(sample_scenario, 0, 0, 30)

        assert isinstance(result, WindowResult)
        assert result.scenario_id == "test_scenario"
        assert result.window_idx == 0
        assert result.window_size == 30
        assert len(result.rule_results) == 28

    def test_run_windowed_evaluation(self, sample_scenario):
        """Test full windowed evaluation run."""
        executor = WindowedExecutor(window_size_s=3.0, stride_s=2.0)
        executor.register_all_rules()

        results = executor.run(sample_scenario)

        assert len(results) > 0
        assert all(isinstance(r, WindowResult) for r in results)

    def test_run_with_sink(self, sample_scenario, tmp_path):
        """Test windowed evaluation with output sink."""
        executor = WindowedExecutor(window_size_s=3.0, stride_s=2.0)
        executor.register_all_rules()

        output_file = str(tmp_path / "output.jsonl")

        with JsonlSink(output_file) as sink:
            results = executor.run(sample_scenario, sink=sink)

        # Check output file was created
        assert os.path.exists(output_file)

        # Check content
        with open(output_file) as f:
            lines = f.readlines()

        # Should have one line per window * rules
        assert len(lines) > 0

        # Parse first record
        record = json.loads(lines[0])
        assert "scenario_id" in record
        assert "rule_id" in record
        assert "applies" in record

    def test_window_result_to_dict(self, sample_scenario):
        """Test WindowResult serialization."""
        executor = WindowedExecutor(window_size_s=3.0, stride_s=1.0)
        executor.register_all_rules()

        result = executor.evaluate_window(sample_scenario, 0, 0, 30)
        result_dict = result.to_dict()

        assert "scenario_id" in result_dict
        assert "window_idx" in result_dict
        assert "n_violations" in result_dict
        assert "rules" in result_dict

    def test_window_result_to_flat_records(self, sample_scenario):
        """Test WindowResult flat serialization."""
        executor = WindowedExecutor(window_size_s=3.0, stride_s=1.0)
        executor.register_all_rules()

        result = executor.evaluate_window(sample_scenario, 0, 0, 30)
        records = result.to_flat_records()

        assert len(records) == 28  # One per rule

        for rec in records:
            assert "scenario_id" in rec
            assert "rule_id" in rec
            assert "applies" in rec
            assert "severity" in rec

    def test_run_batch(self, sample_scenario):
        """Test batch processing of multiple scenarios."""
        executor = WindowedExecutor(window_size_s=3.0, stride_s=2.0)
        executor.register_all_rules()

        # Create two scenarios
        scenario2 = ScenarioContext(
            scenario_id="test_scenario_2",
            dt=sample_scenario.dt,
            ego=sample_scenario.ego,
            agents=sample_scenario.agents,
            map_context=sample_scenario.map_context,
        )

        results = executor.run_batch([sample_scenario, scenario2])

        assert len(results) == 2
        assert "test_scenario" in results
        assert "test_scenario_2" in results


class TestJsonlSink:
    """Tests for JSONL output sink."""

    def test_write_single_record(self, tmp_path):
        """Test writing a single record."""
        path = str(tmp_path / "output.jsonl")

        with JsonlSink(path) as sink:
            sink.write({"key": "value", "num": 42})

        with open(path) as f:
            content = f.read()

        record = json.loads(content.strip())
        assert record["key"] == "value"
        assert record["num"] == 42

    def test_write_multiple_records(self, tmp_path):
        """Test writing multiple records."""
        path = str(tmp_path / "output.jsonl")

        with JsonlSink(path) as sink:
            sink.write({"id": 1})
            sink.write({"id": 2})
            sink.write({"id": 3})

        with open(path) as f:
            lines = f.readlines()

        assert len(lines) == 3
        assert json.loads(lines[0])["id"] == 1
        assert json.loads(lines[2])["id"] == 3

    def test_write_many(self, tmp_path):
        """Test write_many method."""
        path = str(tmp_path / "output.jsonl")

        records = [{"id": i} for i in range(10)]

        with JsonlSink(path) as sink:
            sink.write_many(records)

        with open(path) as f:
            lines = f.readlines()

        assert len(lines) == 10

    def test_creates_directory(self, tmp_path):
        """Test that missing directories are created."""
        path = str(tmp_path / "subdir" / "nested" / "output.jsonl")

        with JsonlSink(path) as sink:
            sink.write({"test": True})

        assert os.path.exists(path)


class TestCsvSink:
    """Tests for CSV output sink."""

    def test_write_records(self, tmp_path):
        """Test writing CSV records."""
        path = str(tmp_path / "output.csv")

        with CsvSink(path) as sink:
            sink.write({"name": "Alice", "score": 95})
            sink.write({"name": "Bob", "score": 87})

        with open(path) as f:
            content = f.read()

        lines = content.strip().split("\n")
        assert len(lines) == 3  # Header + 2 data rows
        assert "name,score" in lines[0]
        assert "Alice,95" in lines[1]

    def test_column_order_preserved(self, tmp_path):
        """Test that column order is consistent."""
        path = str(tmp_path / "output.csv")

        with CsvSink(path, columns=["id", "name", "value"]) as sink:
            sink.write({"name": "X", "id": 1, "value": 100})
            sink.write({"id": 2, "value": 200, "name": "Y"})

        with open(path) as f:
            lines = f.readlines()

        assert "id,name,value" in lines[0]


class TestMakeSink:
    """Tests for make_sink factory function."""

    def test_jsonl_extension(self, tmp_path):
        """Test JSONL sink created for .jsonl extension."""
        path = str(tmp_path / "output.jsonl")
        sink = make_sink(path)

        assert isinstance(sink, JsonlSink)
        sink.close()

    def test_csv_extension(self, tmp_path):
        """Test CSV sink created for .csv extension."""
        path = str(tmp_path / "output.csv")
        sink = make_sink(path)

        assert isinstance(sink, CsvSink)
        sink.close()

    def test_default_is_jsonl(self, tmp_path):
        """Test default sink is JSONL."""
        path = str(tmp_path / "output.txt")
        sink = make_sink(path)

        assert isinstance(sink, JsonlSink)
        sink.close()


class TestWindowedPipelineIntegration:
    """End-to-end integration tests for windowed pipeline."""

    def test_full_windowed_pipeline(self, sample_scenario, tmp_path):
        """Test complete windowed evaluation pipeline."""
        executor = WindowedExecutor(window_size_s=3.0, stride_s=1.0)
        executor.register_all_rules()

        output_file = str(tmp_path / "results.jsonl")

        with make_sink(output_file) as sink:
            results = executor.run(sample_scenario, sink=sink)

        # Verify results
        assert len(results) > 0

        # Verify output file
        with open(output_file) as f:
            records = [json.loads(line) for line in f]

        # Should have windows * rules records
        expected_records = len(results) * 28
        assert len(records) == expected_records

        # Check record structure
        for rec in records:
            assert "scenario_id" in rec
            assert "window_idx" in rec
            assert "rule_id" in rec
            assert "applies" in rec
            assert rec["scenario_id"] == "test_scenario"

    def test_windows_cover_scenario(self, sample_scenario):
        """Verify windows properly cover the scenario timeline."""
        executor = WindowedExecutor(window_size_s=3.0, stride_s=1.0)

        windows = executor.make_windows(sample_scenario)

        # First window should start at beginning
        assert windows[0][1] == 0  # start_idx

        # Last window should reach near end
        last_end = windows[-1][2]
        assert last_end >= 90  # Should reach close to 100 frames

    def test_violations_detected_in_windows(self, sample_scenario, tmp_path):
        """Test that violations are properly detected per window."""
        executor = WindowedExecutor(window_size_s=5.0, stride_s=2.0)
        executor.register_all_rules()

        results = executor.run(sample_scenario)

        # Collect all violation counts
        total_violations = sum(r.n_violations for r in results)
        total_applicable = sum(r.n_applicable_rules for r in results)

        # Should have some applicable rules across windows
        assert total_applicable > 0

    def test_batch_processing_with_sink(self, sample_scenario, tmp_path):
        """Test batch processing with output sink."""
        executor = WindowedExecutor(window_size_s=5.0, stride_s=2.0)
        executor.register_all_rules()

        # Create multiple scenarios
        scenarios = []
        for i in range(3):
            s = ScenarioContext(
                scenario_id=f"scenario_{i}",
                dt=sample_scenario.dt,
                ego=sample_scenario.ego,
                agents=sample_scenario.agents,
                map_context=sample_scenario.map_context,
            )
            scenarios.append(s)

        output_file = str(tmp_path / "batch_results.jsonl")

        with make_sink(output_file) as sink:
            results = executor.run_batch(scenarios, sink=sink)

        # Verify all scenarios processed
        assert len(results) == 3

        # Verify output contains records from all scenarios
        with open(output_file) as f:
            records = [json.loads(line) for line in f]

        scenario_ids = set(r["scenario_id"] for r in records)
        assert scenario_ids == {"scenario_0", "scenario_1", "scenario_2"}

    def test_window_severity_aggregation(self, sample_scenario):
        """Test that severity is properly aggregated per window."""
        executor = WindowedExecutor(window_size_s=5.0, stride_s=2.0)
        executor.register_all_rules()

        results = executor.run(sample_scenario)

        for result in results:
            # Total severity should be sum of individual severities
            computed = sum(r.severity for r in result.rule_results)
            assert abs(result.total_severity - computed) < 0.001


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_very_short_scenario(self):
        """Test with scenario shorter than window size."""
        short_ego = EgoState(
            x=[0, 1, 2, 3, 4],
            y=[0, 0, 0, 0, 0],
            yaw=[0, 0, 0, 0, 0],
            speed=[1, 1, 1, 1, 1],
            length=4.5,
            width=2.0,
        )

        scenario = ScenarioContext(
            scenario_id="short",
            dt=0.1,
            ego=short_ego,
            agents=[],
            map_context=MapContext(lane_center_xy=np.array([[0, 0], [10, 0]])),
        )

        executor = WindowedExecutor(
            window_size_s=8.0, stride_s=2.0  # Larger than scenario
        )
        executor.register_all_rules()

        results = executor.run(scenario)

        # Should still produce at least one result
        assert len(results) >= 1

    def test_single_frame_scenario(self):
        """Test with single-frame scenario."""
        single_ego = EgoState(
            x=[0],
            y=[0],
            yaw=[0],
            speed=[10],
            length=4.5,
            width=2.0,
        )

        # Single frame scenario needs a minimal map_context
        scenario = ScenarioContext(
            scenario_id="single",
            dt=0.1,
            ego=single_ego,
            agents=[],
            map_context=MapContext(lane_center_xy=np.array([[0, 0], [10, 0]])),
        )

        executor = WindowedExecutor()
        executor.register_all_rules()

        # Should not crash
        results = executor.run(scenario)

        # May have no windows if min_frames not met
        assert isinstance(results, list)

    def test_no_agents(self, sample_ego):
        """Test scenario with no other agents."""
        scenario = ScenarioContext(
            scenario_id="no_agents",
            dt=0.1,
            ego=sample_ego,
            agents=[],  # No agents
            map_context=MapContext(lane_center_xy=np.array([[0, 0], [100, 0]])),
        )

        executor = WindowedExecutor(window_size_s=3.0, stride_s=1.0)
        executor.register_all_rules()

        results = executor.run(scenario)

        assert len(results) > 0
        # Agent-dependent rules should not be applicable

    def test_many_windows(self, long_timestamps):
        """Test with many overlapping windows."""
        windows = make_windows(
            long_timestamps, window_size_s=2.0, stride_s=0.2  # High overlap
        )

        # Should create many windows
        assert len(windows) > 50

    def test_non_uniform_timestamps(self):
        """Test with non-uniform timestamp spacing."""
        timestamps = np.array([0, 0.1, 0.25, 0.35, 0.5, 0.65, 0.8, 1.0])

        windows = make_windows(timestamps, window_size_s=0.5, stride_s=0.2)

        # Should still work despite non-uniform spacing
        assert len(windows) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
