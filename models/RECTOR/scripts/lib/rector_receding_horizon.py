#!/usr/bin/env python3
"""
RECTOR Receding Horizon Planner

Runs RECTOR planning in a receding horizon fashion across a Waymo scenario.
At each timestep, the planner:
1. Observes current state
2. Generates ego trajectory candidates
3. Predicts reactor responses
4. Selects the best trajectory
5. Executes one step and advances time

This simulates how RECTOR would operate on an autonomous vehicle.

Usage:
    python rector_receding_horizon.py --tfrecord <path> --num_scenarios 5

Output:
    - Trajectory predictions and selections per timestep
    - Metrics (ADE, FDE, collision rates)
    - Visualization movies
"""

import argparse
import os
import pickle
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

# Setup paths
RECTOR_ROOT = Path(__file__).parent.parent.parent
RECTOR_LIB = Path(__file__).parent  # Already in lib
M2I_SCRIPTS = Path("/workspace/models/pretrained/m2i/scripts/lib")
M2I_SRC = Path("/workspace/externals/M2I/src")

sys.path.insert(0, str(RECTOR_LIB))
sys.path.insert(0, str(M2I_SCRIPTS))
sys.path.insert(0, str(M2I_SRC))

# Suppress TF warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings

warnings.filterwarnings("ignore")

# RECTOR imports
from data_contracts import PlanningConfig, AgentState
from planning_loop import (
    RECTORPlanner,
    CandidateGenerator,
    ReactorSelector,
    LaneInfo,
    PlanningResult,
)

# Optional imports
try:
    import tensorflow as tf

    TF_AVAILABLE = True
except ImportError:
    tf = None
    TF_AVAILABLE = False

try:
    from visualize import BEVRenderer, RECTORMovieGenerator, BEVConfig

    VIS_AVAILABLE = True
except ImportError:
    VIS_AVAILABLE = False


@dataclass
class RecedingHorizonConfig:
    """Configuration for receding horizon planning."""

    # Scenario parameters
    start_timestep: int = 10  # First prediction timestep (needs history)
    end_timestep: int = 80  # Last prediction timestep
    step_size: int = 1  # Advance by N timesteps each iteration

    # Planning parameters
    planning_config: PlanningConfig = field(default_factory=PlanningConfig)

    # Output
    generate_movies: bool = False
    output_dir: str = "output/rector_receding"


@dataclass
class TimestepResult:
    """Result from one timestep of receding horizon planning."""

    timestep: int
    planning_result: PlanningResult
    ego_state: AgentState
    selected_trajectory: np.ndarray
    timing_ms: float


@dataclass
class ScenarioResult:
    """Complete result for one scenario."""

    scenario_id: str
    timestep_results: List[TimestepResult]
    total_time_ms: float
    metrics: Dict[str, float]


class WaymoScenarioParser:
    """Parse Waymo TFRecord scenarios for RECTOR."""

    def __init__(self):
        if not TF_AVAILABLE:
            raise RuntimeError("TensorFlow is required for Waymo parsing")

        self._load_feature_description()

    def _load_feature_description(self):
        """Load TFRecord feature description."""
        try:
            from waymo_flat_parser import features_description_flat

            self.features = features_description_flat
        except ImportError:
            # Minimal feature description
            self.features = {
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

    def parse_scenario(
        self, tfrecord_path: str, scenario_idx: int = 0
    ) -> Optional[Dict]:
        """Parse a single scenario from TFRecord."""
        dataset = tf.data.TFRecordDataset(tfrecord_path)

        for idx, record in enumerate(dataset):
            if idx != scenario_idx:
                continue

            parsed = tf.io.parse_single_example(record, self.features)

            def to_np(key):
                return parsed[key].numpy()

            n_agents = 128
            n_past = 10
            n_future = 80

            # Parse states
            past_x = to_np("state/past/x").reshape(n_agents, n_past)[:, ::-1]
            past_y = to_np("state/past/y").reshape(n_agents, n_past)[:, ::-1]
            past_vx = to_np("state/past/velocity_x").reshape(n_agents, n_past)[:, ::-1]
            past_vy = to_np("state/past/velocity_y").reshape(n_agents, n_past)[:, ::-1]
            past_yaw = to_np("state/past/bbox_yaw").reshape(n_agents, n_past)[:, ::-1]
            past_valid = to_np("state/past/valid").reshape(n_agents, n_past)[:, ::-1]

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

            # Concatenate timeline
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
                "scenario_id": to_np("scenario/id").decode("utf-8"),
                "roadgraph_xyz": roadgraph_xyz,
                "roadgraph_type": roadgraph_type,
                "roadgraph_valid": roadgraph_valid.astype(bool),
            }

        return None

    def get_agents_at_timestep(
        self,
        scenario: Dict,
        timestep: int,
    ) -> Tuple[AgentState, List[AgentState]]:
        """Extract ego and other agents at a specific timestep."""
        valid = scenario["valid"][:, timestep]

        agents = []
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

        # First valid agent is ego
        if not agents:
            return None, []

        ego = agents[0]
        others = agents[1:]

        return ego, others

    def extract_lane(
        self,
        scenario: Dict,
        ego_state: AgentState,
        max_points: int = 100,
    ) -> LaneInfo:
        """Extract approximate lane near ego."""
        xyz = scenario["roadgraph_xyz"]
        valid = scenario["roadgraph_valid"]
        types = scenario["roadgraph_type"]

        # Lane center points
        lane_mask = valid & (types == 1)
        lane_points = xyz[lane_mask][:, :2]

        if len(lane_points) < 5:
            # Default straight lane
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

        # Find closest points to ego
        ego_pos = np.array([ego_state.x, ego_state.y])
        dists = np.linalg.norm(lane_points - ego_pos, axis=1)

        # Sort by distance
        sorted_idx = np.argsort(dists)[:max_points]
        centerline = lane_points[sorted_idx]

        # Sort by heading direction
        heading_vec = np.array([np.cos(ego_state.heading), np.sin(ego_state.heading)])
        rel_points = centerline - ego_pos
        projections = np.dot(rel_points, heading_vec)
        sorted_by_proj = np.argsort(projections)
        centerline = centerline[sorted_by_proj]

        return LaneInfo(centerline=centerline)


class RECTORRecedingHorizon:
    """
    Run RECTOR planning in a receding horizon loop.

    At each timestep:
    1. Observe current state
    2. Generate candidates
    3. Predict reactions (M2I or dummy)
    4. Score and select
    5. Advance time
    """

    def __init__(
        self,
        config: RecedingHorizonConfig = None,
        m2i_adapter: Optional[Any] = None,
    ):
        self.config = config or RecedingHorizonConfig()
        self.m2i_adapter = m2i_adapter

        # Initialize planner
        self.planner = RECTORPlanner(
            config=self.config.planning_config,
            adapter=m2i_adapter,
        )

        # Initialize parser
        self.parser = WaymoScenarioParser() if TF_AVAILABLE else None

        # Initialize visualizer
        if VIS_AVAILABLE and self.config.generate_movies:
            output_dir = Path(self.config.output_dir) / "movies"
            self.movie_gen = RECTORMovieGenerator(str(output_dir))
        else:
            self.movie_gen = None

    def run_scenario(
        self,
        tfrecord_path: str,
        scenario_idx: int = 0,
    ) -> Optional[ScenarioResult]:
        """Run receding horizon planning on a single scenario."""

        if self.parser is None:
            print("Error: TensorFlow not available")
            return None

        # Parse scenario
        scenario = self.parser.parse_scenario(tfrecord_path, scenario_idx)
        if scenario is None:
            print(f"Error: Could not parse scenario {scenario_idx}")
            return None

        scenario_id = scenario["scenario_id"]
        print(f"Processing scenario: {scenario_id}")

        timestep_results = []
        total_start = time.time()

        # Run receding horizon loop
        for t in range(
            self.config.start_timestep,
            self.config.end_timestep + 1,
            self.config.step_size,
        ):

            # Get current state
            ego_state, other_agents = self.parser.get_agents_at_timestep(scenario, t)
            if ego_state is None:
                print(f"  t={t}: No valid ego state")
                continue

            # Get lane info
            lane = self.parser.extract_lane(scenario, ego_state)

            # Run planning
            t_start = time.time()
            result = self.planner.plan_tick(
                ego_state=ego_state,
                agent_states=other_agents,
                current_lane=lane,
            )
            timing_ms = (time.time() - t_start) * 1000

            # Store result
            timestep_result = TimestepResult(
                timestep=t,
                planning_result=result,
                ego_state=ego_state,
                selected_trajectory=result.selected_candidate.trajectory,
                timing_ms=timing_ms,
            )
            timestep_results.append(timestep_result)

            # Add frame to movie
            if self.movie_gen is not None:
                self.movie_gen.add_frame(
                    result,
                    ego_state,
                    other_agents,
                    lane,
                    title=f"Scenario {scenario_id} | t={t}",
                )

            # Progress
            if t % 10 == 0:
                print(
                    f"  t={t}: selected #{result.selected_idx}, "
                    f"score={result.all_scores[result.selected_idx]:.2f}, "
                    f"time={timing_ms:.1f}ms"
                )

        total_time = (time.time() - total_start) * 1000

        # Save movie
        if self.movie_gen is not None:
            self.movie_gen.save_movie(f"{scenario_id}.mp4")

        # Compute metrics
        metrics = self._compute_metrics(scenario, timestep_results)

        return ScenarioResult(
            scenario_id=scenario_id,
            timestep_results=timestep_results,
            total_time_ms=total_time,
            metrics=metrics,
        )

    def _compute_metrics(
        self,
        scenario: Dict,
        results: List[TimestepResult],
    ) -> Dict[str, float]:
        """Compute planning metrics."""
        metrics = {}

        if not results:
            return metrics

        # Timing metrics
        timings = [r.timing_ms for r in results]
        metrics["mean_planning_time_ms"] = float(np.mean(timings))
        metrics["max_planning_time_ms"] = float(np.max(timings))
        metrics["p95_planning_time_ms"] = float(np.percentile(timings, 95))

        # Selection consistency (how often we switch candidates)
        selections = [r.planning_result.selected_idx for r in results]
        switches = sum(
            1 for i in range(1, len(selections)) if selections[i] != selections[i - 1]
        )
        metrics["selection_switches"] = switches
        metrics["selection_switch_rate"] = switches / max(1, len(selections) - 1)

        # Score statistics
        scores = [
            r.planning_result.all_scores[r.planning_result.selected_idx]
            for r in results
        ]
        metrics["mean_selected_score"] = float(np.mean(scores))
        metrics["min_selected_score"] = float(np.min(scores))

        return metrics

    def run_batch(
        self,
        tfrecord_path: str,
        num_scenarios: int = 5,
    ) -> List[ScenarioResult]:
        """Run receding horizon on multiple scenarios."""
        results = []

        for i in range(num_scenarios):
            print(f"\n{'='*60}")
            print(f"Scenario {i+1}/{num_scenarios}")
            print(f"{'='*60}")

            result = self.run_scenario(tfrecord_path, scenario_idx=i)
            if result:
                results.append(result)
                print(f"\nMetrics:")
                for k, v in result.metrics.items():
                    print(f"  {k}: {v:.3f}")

        # Aggregate metrics
        if results:
            print(f"\n{'='*60}")
            print("Aggregate Metrics")
            print(f"{'='*60}")

            all_timings = []
            for r in results:
                all_timings.extend([tr.timing_ms for tr in r.timestep_results])

            print(f"  Mean planning time: {np.mean(all_timings):.1f}ms")
            print(f"  P95 planning time: {np.percentile(all_timings, 95):.1f}ms")
            print(f"  P99 planning time: {np.percentile(all_timings, 99):.1f}ms")

        return results


def find_tfrecord() -> Optional[str]:
    """Find an available TFRecord file."""
    paths = [
        Path(
            "/workspace/data/WOMD/datasets/waymo_open_dataset/motion_v_1_3_0/processed/tf/testing_interactive"
        ),
        Path(
            "/workspace/data/WOMD/datasets/waymo_open_dataset/motion_v_1_3_0/tf/validation_interactive"
        ),
    ]

    for base_path in paths:
        if base_path.exists():
            tfrecords = list(base_path.glob("*.tfrecord*"))
            if tfrecords:
                return str(tfrecords[0])

    return None


def main():
    parser = argparse.ArgumentParser(description="RECTOR Receding Horizon Planner")

    parser.add_argument(
        "--tfrecord", type=str, default=None, help="Path to TFRecord file"
    )
    parser.add_argument(
        "--num_scenarios",
        "-n",
        type=int,
        default=3,
        help="Number of scenarios to process",
    )
    parser.add_argument("--start_t", type=int, default=10, help="Start timestep (>=10)")
    parser.add_argument("--end_t", type=int, default=80, help="End timestep (<=90)")
    parser.add_argument("--step", type=int, default=1, help="Timestep increment")
    parser.add_argument(
        "--candidates", "-m", type=int, default=8, help="Number of candidates"
    )
    parser.add_argument(
        "--reactors", "-k", type=int, default=3, help="Number of reactors"
    )
    parser.add_argument(
        "--generate-movies", action="store_true", help="Generate visualization movies"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="output/rector_receding",
        help="Output directory",
    )

    args = parser.parse_args()

    # Find TFRecord
    tfrecord_path = args.tfrecord or find_tfrecord()
    if tfrecord_path is None:
        print("Error: No TFRecord file found. Use --tfrecord to specify.")
        return 1

    print(f"TFRecord: {tfrecord_path}")

    # Create config
    planning_config = PlanningConfig(
        num_candidates=args.candidates,
        max_reactors=args.reactors,
        device="cpu",  # Use CPU for now
    )

    rh_config = RecedingHorizonConfig(
        start_timestep=args.start_t,
        end_timestep=args.end_t,
        step_size=args.step,
        planning_config=planning_config,
        generate_movies=args.generate_movies,
        output_dir=args.output,
    )

    # Run
    rh_planner = RECTORRecedingHorizon(config=rh_config)
    results = rh_planner.run_batch(tfrecord_path, args.num_scenarios)

    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "results.pkl"
    with open(results_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\nResults saved to: {results_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
