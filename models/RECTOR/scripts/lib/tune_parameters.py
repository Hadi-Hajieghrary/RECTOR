#!/usr/bin/env python3
"""
RECTOR Parameter Tuning Utilities

Tools for tuning RECTOR planning parameters:
1. Grid search over parameter combinations
2. Sensitivity analysis
3. Pareto frontier visualization
4. Configuration export

Key Parameters:
- num_candidates (M): Number of ego trajectory candidates
- max_reactors (K): Number of reactive agents to consider
- cvar_alpha: Risk level (0.1 = 10% worst-case focus)
- hysteresis_bonus: Bonus for staying on current trajectory
- collision_radius: Safety buffer for collision checking

Usage:
    python tune_parameters.py --grid-search
    python tune_parameters.py --sensitivity cvar_alpha
"""

import argparse
import itertools
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

# Setup paths
RECTOR_ROOT = Path(__file__).parent.parent.parent
RECTOR_LIB = Path(__file__).parent  # Already in lib
sys.path.insert(0, str(RECTOR_LIB))

# Suppress warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings

warnings.filterwarnings("ignore")

# RECTOR imports
from data_contracts import PlanningConfig, AgentState
from planning_loop import RECTORPlanner, LaneInfo, PlanningResult


@dataclass
class TuningConfig:
    """Configuration for parameter tuning."""

    # Grid search ranges
    m_candidates_range: List[int] = None
    k_reactors_range: List[int] = None
    cvar_alpha_range: List[float] = None
    hysteresis_range: List[float] = None

    # Evaluation settings
    num_samples: int = 50
    num_scenarios: int = 5

    def __post_init__(self):
        if self.m_candidates_range is None:
            self.m_candidates_range = [4, 8, 16, 32]
        if self.k_reactors_range is None:
            self.k_reactors_range = [1, 2, 3, 5]
        if self.cvar_alpha_range is None:
            self.cvar_alpha_range = [0.05, 0.1, 0.2, 0.5]
        if self.hysteresis_range is None:
            self.hysteresis_range = [0.0, 0.1, 0.2, 0.5]


@dataclass
class EvaluationResult:
    """Result of evaluating one parameter configuration."""

    config: Dict[str, Any]

    # Performance metrics
    mean_planning_time_ms: float
    p95_planning_time_ms: float

    # Quality metrics
    mean_score: float
    score_variance: float

    # Safety metrics
    collision_rate: float
    mean_min_distance: float

    # Consistency metrics
    selection_switch_rate: float
    hysteresis_effectiveness: float


from real_data_loader import create_test_scenarios_from_real, get_default_loader


def create_test_scenarios(num_scenarios: int = 10) -> List[Dict]:
    """Load diverse test scenarios from real Waymo TFRecords."""
    return create_test_scenarios_from_real(num_scenarios=num_scenarios)


def evaluate_config(
    planning_config: PlanningConfig,
    scenarios: List[Dict],
    num_samples: int = 20,
) -> EvaluationResult:
    """
    Evaluate a single parameter configuration.

    Runs the planner on multiple scenarios and collects metrics.
    """
    planner = RECTORPlanner(config=planning_config, adapter=None)

    timings = []
    scores = []
    min_distances = []
    switches = []

    for scenario in scenarios:
        ego_state = scenario["ego_state"]
        other_agents = scenario["other_agents"]
        lane = scenario["lane"]

        prev_selection = None

        for _ in range(num_samples // len(scenarios)):
            # Run planning
            t_start = time.time()
            result = planner.plan_tick(
                ego_state=ego_state,
                agent_states=other_agents,
                current_lane=lane,
            )
            timing_ms = (time.time() - t_start) * 1000

            timings.append(timing_ms)
            scores.append(result.all_scores[result.selected_idx])

            # Track selection switches
            if prev_selection is not None:
                switches.append(1 if result.selected_idx != prev_selection else 0)
            prev_selection = result.selected_idx

    return EvaluationResult(
        config=asdict(planning_config),
        mean_planning_time_ms=float(np.mean(timings)),
        p95_planning_time_ms=float(np.percentile(timings, 95)),
        mean_score=float(np.mean(scores)),
        score_variance=float(np.var(scores)),
        collision_rate=0.0,  # Would need safety check
        mean_min_distance=0.0,  # Would need trajectory analysis
        selection_switch_rate=float(np.mean(switches)) if switches else 0.0,
        hysteresis_effectiveness=1.0 - float(np.mean(switches)) if switches else 1.0,
    )


def grid_search(
    tuning_config: TuningConfig,
    scenarios: List[Dict],
) -> List[EvaluationResult]:
    """
    Perform grid search over parameter combinations.

    Returns list of evaluation results sorted by score.
    """
    results = []

    # Generate all combinations
    combinations = list(
        itertools.product(
            tuning_config.m_candidates_range,
            tuning_config.k_reactors_range,
            tuning_config.cvar_alpha_range,
        )
    )

    print(f"Grid search: {len(combinations)} combinations")
    print("-" * 60)

    for i, (m, k, alpha) in enumerate(combinations):
        print(f"[{i+1}/{len(combinations)}] M={m}, K={k}, alpha={alpha:.2f}", end=" ")

        config = PlanningConfig(
            num_candidates=m,
            max_reactors=k,
            cvar_alpha=alpha,
            device="cpu",
        )

        result = evaluate_config(config, scenarios, tuning_config.num_samples)
        results.append(result)

        print(
            f"-> time={result.mean_planning_time_ms:.1f}ms, score={result.mean_score:.1f}"
        )

    # Sort by score (descending)
    results.sort(key=lambda r: r.mean_score, reverse=True)

    return results


def sensitivity_analysis(
    param_name: str,
    param_values: List[Any],
    base_config: PlanningConfig,
    scenarios: List[Dict],
    num_samples: int = 20,
) -> Dict[str, List[float]]:
    """
    Analyze sensitivity to a single parameter.

    Varies one parameter while keeping others fixed.
    """
    results = {
        "param_values": [],
        "mean_time_ms": [],
        "p95_time_ms": [],
        "mean_score": [],
        "score_variance": [],
        "switch_rate": [],
    }

    print(f"Sensitivity analysis: {param_name}")
    print("-" * 60)

    for value in param_values:
        # Create modified config
        config_dict = asdict(base_config)
        config_dict[param_name] = value
        config = PlanningConfig(**config_dict)

        print(f"  {param_name}={value}", end=" ")

        result = evaluate_config(config, scenarios, num_samples)

        results["param_values"].append(value)
        results["mean_time_ms"].append(result.mean_planning_time_ms)
        results["p95_time_ms"].append(result.p95_planning_time_ms)
        results["mean_score"].append(result.mean_score)
        results["score_variance"].append(result.score_variance)
        results["switch_rate"].append(result.selection_switch_rate)

        print(
            f"-> time={result.mean_planning_time_ms:.1f}ms, score={result.mean_score:.1f}"
        )

    return results


def find_pareto_frontier(
    results: List[EvaluationResult],
    x_metric: str = "mean_planning_time_ms",
    y_metric: str = "mean_score",
) -> List[EvaluationResult]:
    """
    Find Pareto-optimal configurations.

    A configuration is Pareto-optimal if no other configuration
    is better on both metrics.
    """
    pareto = []

    for result in results:
        x = getattr(result, x_metric)
        y = getattr(result, y_metric)

        # Check if dominated by any other result
        is_dominated = False
        for other in results:
            if result is other:
                continue

            other_x = getattr(other, x_metric)
            other_y = getattr(other, y_metric)

            # For time, lower is better; for score, higher is better
            if other_x <= x and other_y >= y and (other_x < x or other_y > y):
                is_dominated = True
                break

        if not is_dominated:
            pareto.append(result)

    # Sort by x metric
    pareto.sort(key=lambda r: getattr(r, x_metric))

    return pareto


def export_best_config(
    results: List[EvaluationResult],
    output_path: str,
    criteria: str = "score",
) -> PlanningConfig:
    """
    Export the best configuration based on criteria.

    Criteria options:
    - "score": Highest mean score
    - "speed": Lowest p95 time
    - "balanced": Good tradeoff (from Pareto frontier)
    """
    if criteria == "score":
        best = max(results, key=lambda r: r.mean_score)
    elif criteria == "speed":
        best = min(results, key=lambda r: r.p95_planning_time_ms)
    elif criteria == "balanced":
        pareto = find_pareto_frontier(results)
        # Pick middle of Pareto frontier
        best = pareto[len(pareto) // 2]
    else:
        raise ValueError(f"Unknown criteria: {criteria}")

    # Create config
    config = PlanningConfig(
        num_candidates=best.config["num_candidates"],
        max_reactors=best.config["max_reactors"],
        cvar_alpha=best.config["cvar_alpha"],
    )

    # Export
    output_dict = {
        "config": asdict(config),
        "metrics": {
            "mean_planning_time_ms": best.mean_planning_time_ms,
            "p95_planning_time_ms": best.p95_planning_time_ms,
            "mean_score": best.mean_score,
        },
        "criteria": criteria,
    }

    with open(output_path, "w") as f:
        json.dump(output_dict, f, indent=2)

    print(f"Exported best config to: {output_path}")

    return config


RECOMMENDED_CONFIGS = {
    "fast": PlanningConfig(
        num_candidates=4,
        max_reactors=2,
        cvar_alpha=0.2,
        candidate_horizon=40,  # Shorter horizon
    ),
    "balanced": PlanningConfig(
        num_candidates=8,
        max_reactors=3,
        cvar_alpha=0.1,
        candidate_horizon=80,
    ),
    "thorough": PlanningConfig(
        num_candidates=16,
        max_reactors=5,
        cvar_alpha=0.05,
        candidate_horizon=80,
    ),
    "conservative": PlanningConfig(
        num_candidates=8,
        max_reactors=3,
        cvar_alpha=0.05,  # More risk-averse
        min_safety_margin=3.0,
        collision_radius=4.0,
    ),
}


def print_recommended_configs():
    """Print recommended configuration presets."""
    print("\nRecommended Configuration Presets")
    print("=" * 60)

    for name, config in RECOMMENDED_CONFIGS.items():
        print(f"\n{name.upper()}:")
        print(f"  M (candidates):   {config.num_candidates}")
        print(f"  K (reactors):     {config.max_reactors}")
        print(f"  CVaR alpha:       {config.cvar_alpha}")
        print(f"  Horizon:          {config.candidate_horizon}")


def main():
    parser = argparse.ArgumentParser(description="RECTOR Parameter Tuning")

    parser.add_argument(
        "--grid-search", action="store_true", help="Run grid search over parameters"
    )
    parser.add_argument(
        "--sensitivity",
        type=str,
        default=None,
        help="Run sensitivity analysis for a parameter",
    )
    parser.add_argument(
        "--presets", action="store_true", help="Show recommended configuration presets"
    )
    parser.add_argument(
        "--export", type=str, default=None, help="Export best config to JSON file"
    )
    parser.add_argument(
        "--criteria",
        type=str,
        default="balanced",
        choices=["score", "speed", "balanced"],
        help="Optimization criteria",
    )
    parser.add_argument(
        "--num_scenarios", type=int, default=5, help="Number of test scenarios"
    )
    parser.add_argument(
        "--num_samples", type=int, default=20, help="Samples per scenario"
    )

    args = parser.parse_args()

    if args.presets:
        print_recommended_configs()
        return 0

    # Create test scenarios
    print("Creating test scenarios...")
    scenarios = create_test_scenarios(args.num_scenarios)
    print(f"Created {len(scenarios)} test scenarios\n")

    if args.grid_search:
        # Run grid search
        tuning_config = TuningConfig(
            num_samples=args.num_samples,
            num_scenarios=args.num_scenarios,
        )

        results = grid_search(tuning_config, scenarios)

        # Print top results
        print("\n" + "=" * 60)
        print("Top 5 Configurations (by score)")
        print("=" * 60)

        for i, result in enumerate(results[:5]):
            print(
                f"\n{i+1}. M={result.config['num_candidates']}, "
                f"K={result.config['max_reactors']}, "
                f"alpha={result.config['cvar_alpha']:.2f}"
            )
            print(
                f"   Time: {result.mean_planning_time_ms:.1f}ms (p95: {result.p95_planning_time_ms:.1f}ms)"
            )
            print(
                f"   Score: {result.mean_score:.2f} (var: {result.score_variance:.2f})"
            )

        # Pareto frontier
        pareto = find_pareto_frontier(results)
        print(f"\nPareto-optimal configurations: {len(pareto)}")

        # Export if requested
        if args.export:
            export_best_config(results, args.export, args.criteria)

    elif args.sensitivity:
        # Run sensitivity analysis
        base_config = RECOMMENDED_CONFIGS["balanced"]

        param_ranges = {
            "num_candidates": [2, 4, 8, 16, 32],
            "max_reactors": [1, 2, 3, 4, 5],
            "cvar_alpha": [0.01, 0.05, 0.1, 0.2, 0.5, 1.0],
        }

        if args.sensitivity not in param_ranges:
            print(f"Unknown parameter: {args.sensitivity}")
            print(f"Available: {list(param_ranges.keys())}")
            return 1

        results = sensitivity_analysis(
            args.sensitivity,
            param_ranges[args.sensitivity],
            base_config,
            scenarios,
            args.num_samples,
        )

        # Print results
        print("\n" + "=" * 60)
        print(f"Sensitivity: {args.sensitivity}")
        print("=" * 60)

        for i in range(len(results["param_values"])):
            print(
                f"{args.sensitivity}={results['param_values'][i]:>6} | "
                f"time={results['mean_time_ms'][i]:>7.1f}ms | "
                f"score={results['mean_score'][i]:>10.1f}"
            )

    else:
        # Default: show presets
        print_recommended_configs()

    return 0


if __name__ == "__main__":
    sys.exit(main())
