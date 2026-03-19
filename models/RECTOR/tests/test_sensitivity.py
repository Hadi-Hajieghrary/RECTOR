#!/usr/bin/env python3
"""
RECTOR Sensitivity Test: GO/NO-GO Gate #1

This is the CRITICAL first test that validates whether M2I's conditional model
actually responds to changes in the influencer trajectory. If this test fails,
interactive planning is fundamentally impossible without retraining the model.

The test:
    1. Uses the existing M2I pipeline to create proper mappings
    2. Creates two contrasting ego trajectories:
       - YIELD: Stay in place (zero displacement)
       - ASSERT: Move forward 50m over 8 seconds
    3. Injects each as the "influencer" trajectory into the reactor's mapping
    4. Runs conditional inference and compares the reactor predictions

Pass Criteria (ANY of the following):
    - Mean displacement between predictions > 1.0m
    - Mode probability shift > 0.1
    - Top mode index changes between conditions

If FAIL: Stop all RECTOR development. Debug the conditioning interface.
If PASS: Proceed with infrastructure development.

Usage:
    python models/RECTOR/tests/test_sensitivity.py
    python models/RECTOR/tests/test_sensitivity.py -v
    bash models/RECTOR/scripts/bash/run_sensitivity_test.sh

Exit Codes:
    0: PASS - Sensitivity validated, proceed with development
    1: FAIL - Sensitivity test failed, interactive planning blocked
    2: ERROR - Test could not run (missing data, model, etc.)

Authors: RECTOR Team
Date: December 2024
Version: 1.0.0
"""

import argparse
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple
import copy

# Suppress TensorFlow logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import numpy as np

# Add M2I models scripts path FIRST (for local modules)
M2I_SCRIPTS = Path("/workspace/models/pretrained/m2i/scripts/lib")
sys.path.insert(0, str(M2I_SCRIPTS))

# Add M2I source to path
M2I_SRC = Path("/workspace/externals/M2I/src")
sys.path.insert(0, str(M2I_SRC))


@dataclass
class SensitivityMetrics:
    """Metrics computed from sensitivity test."""

    mean_displacement: float  # Mean L2 distance between predictions (meters)
    max_displacement: float  # Max displacement at any timestep
    final_displacement: float  # Displacement at t=80 (final timestep)
    prob_shift: float  # L1 distance between mode probabilities
    top_mode_yield: int  # Top mode index for yield condition
    top_mode_assert: int  # Top mode index for assert condition
    yield_travel: float  # Reactor travel under yield condition
    assert_travel: float  # Reactor travel under assert condition

    def passed(self) -> bool:
        """Check if sensitivity test passed."""
        displacement_ok = self.mean_displacement > 1.0
        prob_ok = self.prob_shift > 0.1
        mode_change = self.top_mode_yield != self.top_mode_assert
        travel_delta = abs(self.yield_travel - self.assert_travel) > 2.0
        return displacement_ok or prob_ok or mode_change or travel_delta

    def summary(self) -> str:
        """Generate human-readable summary."""
        status = "✓ PASS" if self.passed() else "✗ FAIL"

        lines = [
            f"\n{'='*60}",
            f"SENSITIVITY TEST RESULT: {status}",
            f"{'='*60}",
            f"",
            f"Displacement Metrics:",
            f"  Mean displacement:  {self.mean_displacement:.3f}m (threshold: >1.0m)",
            f"  Max displacement:   {self.max_displacement:.3f}m",
            f"  Final displacement: {self.final_displacement:.3f}m",
            f"",
            f"Probability Metrics:",
            f"  Probability shift:  {self.prob_shift:.3f} (threshold: >0.1)",
            f"  Top mode (yield):   {self.top_mode_yield}",
            f"  Top mode (assert):  {self.top_mode_assert}",
            f"  Mode changed:       {'Yes ✓' if self.top_mode_yield != self.top_mode_assert else 'No'}",
            f"",
            f"Trajectory Shape:",
            f"  Reactor travel (yield):  {self.yield_travel:.2f}m",
            f"  Reactor travel (assert): {self.assert_travel:.2f}m",
            f"  Delta:                   {abs(self.yield_travel - self.assert_travel):.2f}m",
            f"",
        ]

        if self.passed():
            lines.extend(
                [
                    f"✓ The conditional model IS sensitive to influencer changes.",
                    f"✓ Interactive planning is VIABLE. Proceed with RECTOR development.",
                ]
            )
        else:
            lines.extend(
                [
                    f"✗ The conditional model shows NO sensitivity to influencer changes.",
                    f"✗ Interactive planning is BLOCKED.",
                    f"",
                    f"ROOT CAUSE ANALYSIS:",
                    f"  The M2I conditional model uses RASTER ENCODING for influencer trajectories,",
                    f"  not vector-based 'influencer_traj' fields. The conditional model expects:",
                    f"  - 'raster' and 'raster_inf' in args.other_params",
                    f"  - A 150-channel image (60 scene + 90 influencer trajectory channels)",
                    f"  - Influencer trajectory rendered as raster channels",
                    f"",
                    f"  Currently loading 124/302 weights because CNN encoder is disabled.",
                    f"  The model runs in 'marginal' mode (ignores influencer entirely).",
                    f"",
                    f"RESOLUTION OPTIONS:",
                    f"  A. Enable raster pathway and implement influencer trajectory rasterization",
                    f"     - Add 'raster' and 'raster_inf' to conditional args.other_params",
                    f"     - Rasterize influencer trajectory into image channels 60-140",
                    f"     - Model will then use 150-channel CNN encoder",
                    f"  B. Fine-tune model to accept vector-based influencer conditioning",
                    f"  C. Use alternative conditioning approach (e.g., trajectory as polyline)",
                    f"",
                    f"RECOMMENDED NEXT STEP:",
                    f"  Implement Option A in _build_conditional_args() and run_conditional_inference()",
                ]
            )

        lines.append(f"{'='*60}\n")
        return "\n".join(lines)


@dataclass
class TestConfig:
    """Configuration for sensitivity test."""

    data_dir: Path
    device: str
    scenario_idx: int
    timestep: int
    verbose: bool
    yield_distance: float = 0.0
    assert_distance: float = 50.0


def create_influencer_trajectory(
    start_pos: Tuple[float, float], heading: float, distance: float, horizon: int = 80
) -> np.ndarray:
    """Create straight-line influencer trajectory."""
    trajectory = np.zeros((horizon, 2), dtype=np.float32)

    if distance == 0:
        trajectory[:, 0] = start_pos[0]
        trajectory[:, 1] = start_pos[1]
    else:
        t = np.linspace(0, 1, horizon)
        trajectory[:, 0] = start_pos[0] + distance * np.cos(heading) * t
        trajectory[:, 1] = start_pos[1] + distance * np.sin(heading) * t

    return trajectory


def inject_influencer_into_mapping(
    reactor_mapping: Dict,
    influencer_traj: np.ndarray,
) -> Dict:
    """Inject influencer trajectory into reactor's mapping."""
    cond_mapping = copy.deepcopy(reactor_mapping)

    # Ensure shape is [K, H, 2]
    if influencer_traj.ndim == 2:
        influencer_traj = influencer_traj[np.newaxis, ...]

    best_traj = influencer_traj[0]  # [H, 2]

    cond_mapping["influencer_traj"] = best_traj.astype(np.float32)
    cond_mapping["influencer_all_trajs"] = influencer_traj.astype(np.float32)
    cond_mapping["influencer_scores"] = np.ones(len(influencer_traj), dtype=np.float32)

    map_start = cond_mapping.get("map_start_polyline_idx", 0)
    cond_mapping["gt_influencer_traj_idx"] = map_start + 1

    return cond_mapping


def compute_trajectory_travel(traj: np.ndarray) -> float:
    """Compute total distance traveled along trajectory."""
    if traj.ndim == 3:
        traj = traj[0]
    diffs = np.diff(traj, axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))


def compute_sensitivity_metrics(
    pred_yield: np.ndarray,
    scores_yield: np.ndarray,
    pred_assert: np.ndarray,
    scores_assert: np.ndarray,
) -> SensitivityMetrics:
    """Compute sensitivity metrics comparing two prediction sets."""

    top_yield = np.argmax(scores_yield) if scores_yield is not None else 0
    top_assert = np.argmax(scores_assert) if scores_assert is not None else 0

    traj_yield = pred_yield[top_yield]
    traj_assert = pred_assert[top_assert]

    displacements = np.linalg.norm(traj_yield - traj_assert, axis=1)

    if scores_yield is not None and scores_assert is not None:
        prob_yield = scores_yield / (np.sum(scores_yield) + 1e-8)
        prob_assert = scores_assert / (np.sum(scores_assert) + 1e-8)
        prob_shift = float(np.sum(np.abs(prob_yield - prob_assert)))
    else:
        prob_shift = 0.0

    return SensitivityMetrics(
        mean_displacement=float(np.mean(displacements)),
        max_displacement=float(np.max(displacements)),
        final_displacement=float(displacements[-1]),
        prob_shift=prob_shift,
        top_mode_yield=int(top_yield),
        top_mode_assert=int(top_assert),
        yield_travel=compute_trajectory_travel(traj_yield),
        assert_travel=compute_trajectory_travel(traj_assert),
    )


def run_sensitivity_test(config: TestConfig) -> SensitivityMetrics:
    """Run the complete sensitivity test."""

    print("\n" + "=" * 60)
    print("RECTOR SENSITIVITY TEST (GO/NO-GO GATE #1)")
    print("=" * 60)

    # Import the existing M2I predictor
    print("\n[1/5] Loading M2I predictor...")
    try:
        from m2i_receding_horizon_full import RecedingHorizonM2I

        predictor = RecedingHorizonM2I(device=config.device)
        predictor.load_model()  # Load DenseTNT model and initialize args
        print("      ✓ Predictor initialized and model loaded")
    except Exception as e:
        print(f"      ✗ Failed to initialize predictor: {e}")
        raise

    # Find TFRecord files
    print(f"\n[2/5] Loading scenario from {config.data_dir}...")
    tfrecord_files = sorted(config.data_dir.glob("*.tfrecord*"))
    if not tfrecord_files:
        raise FileNotFoundError(f"No TFRecord files in {config.data_dir}")

    tfrecord_path = str(tfrecord_files[0])
    if config.verbose:
        print(f"      Using {tfrecord_path}")

    # Parse scenario
    scenario_data = predictor.parse_full_scenario(tfrecord_path, config.scenario_idx)
    if scenario_data is None:
        raise ValueError(f"Could not parse scenario {config.scenario_idx}")

    sid = scenario_data.get("scenario_id", b"unknown")
    if isinstance(sid, bytes):
        sid = sid.decode("utf-8")
    print(f"      ✓ Loaded scenario: {sid}")

    # Check for interactive agents
    interactive_mask = scenario_data["objects_of_interest"] > 0
    interactive_indices = np.where(interactive_mask)[0]
    if len(interactive_indices) < 2:
        raise ValueError(f"Need 2 interactive agents, found {len(interactive_indices)}")

    print(f"      Interactive agents: {len(interactive_indices)}")

    # Create mappings for both agents at the specified timestep
    print(f"\n[3/5] Creating agent mappings at t={config.timestep}...")

    # Agent 0 = influencer (ego), Agent 1 = reactor
    mapping_influencer = predictor.create_mapping_at_timestep(
        scenario_data, config.timestep, agent_idx=0
    )
    mapping_reactor = predictor.create_mapping_at_timestep(
        scenario_data, config.timestep, agent_idx=1
    )

    if mapping_influencer is None or mapping_reactor is None:
        raise ValueError("Could not create mappings for agents")

    inf_x, inf_y = mapping_influencer["cent_x"], mapping_influencer["cent_y"]
    inf_heading = mapping_influencer["angle"]

    if config.verbose:
        print(
            f"      Influencer: ({inf_x:.1f}, {inf_y:.1f}), heading={np.degrees(inf_heading):.1f}°"
        )
        print(
            f"      Reactor: ({mapping_reactor['cent_x']:.1f}, {mapping_reactor['cent_y']:.1f})"
        )

    print("      ✓ Mappings created")

    # Create contrasting influencer trajectories
    print("\n[4/5] Creating contrasting influencer trajectories...")

    yield_traj = create_influencer_trajectory(
        start_pos=(inf_x, inf_y),
        heading=inf_heading,
        distance=config.yield_distance,
    )

    assert_traj = create_influencer_trajectory(
        start_pos=(inf_x, inf_y),
        heading=inf_heading,
        distance=config.assert_distance,
    )

    print(f"      YIELD: stay at ({inf_x:.1f}, {inf_y:.1f})")
    print(f"      ASSERT: move {config.assert_distance}m forward")

    mapping_yield = inject_influencer_into_mapping(mapping_reactor, yield_traj)
    mapping_assert = inject_influencer_into_mapping(mapping_reactor, assert_traj)

    print("      ✓ Influencer trajectories injected")

    # Load conditional model and run inference
    print("\n[5/5] Running conditional inference...")

    # Load conditional model
    predictor._load_conditional_model()
    if predictor.conditional_model is None:
        raise RuntimeError("Failed to load conditional model")

    print("      ✓ Conditional model loaded")

    # Run inference with YIELD condition
    print("      Running with YIELD condition...")
    if config.verbose:
        print(
            f"        Influencer traj start: ({yield_traj[0, 0]:.2f}, {yield_traj[0, 1]:.2f})"
        )
        print(
            f"        Influencer traj end: ({yield_traj[-1, 0]:.2f}, {yield_traj[-1, 1]:.2f})"
        )

    pred_yield, scores_yield = predictor.run_conditional_inference(
        mapping_yield, yield_traj[np.newaxis, ...], None  # [1, 80, 2]
    )

    if pred_yield is None:
        raise RuntimeError("YIELD inference failed")
    if config.verbose:
        print(f"        Prediction shape: {pred_yield.shape}")
        print(
            f"        Prediction mode 0 end: ({pred_yield[0, -1, 0]:.2f}, {pred_yield[0, -1, 1]:.2f})"
        )

    # Run inference with ASSERT condition
    print("      Running with ASSERT condition...")
    if config.verbose:
        print(
            f"        Influencer traj start: ({assert_traj[0, 0]:.2f}, {assert_traj[0, 1]:.2f})"
        )
        print(
            f"        Influencer traj end: ({assert_traj[-1, 0]:.2f}, {assert_traj[-1, 1]:.2f})"
        )

    pred_assert, scores_assert = predictor.run_conditional_inference(
        mapping_assert, assert_traj[np.newaxis, ...], None  # [1, 80, 2]
    )

    if pred_assert is None:
        raise RuntimeError("ASSERT inference failed")
    if config.verbose:
        print(f"        Prediction shape: {pred_assert.shape}")
        print(
            f"        Prediction mode 0 end: ({pred_assert[0, -1, 0]:.2f}, {pred_assert[0, -1, 1]:.2f})"
        )

    # Compute metrics
    metrics = compute_sensitivity_metrics(
        pred_yield, scores_yield, pred_assert, scores_assert
    )

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="RECTOR Sensitivity Test (GO/NO-GO Gate #1)"
    )

    parser.add_argument(
        "--data_dir",
        "-d",
        type=Path,
        default=Path(
            "/workspace/data/datasets/waymo_open_dataset/motion_v_1_3_0/processed/tf/validation_interactive"
        ),
        help="Directory containing TFRecord files",
    )

    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--scenario_idx", type=int, default=0)
    parser.add_argument("--timestep", "-t", type=int, default=10)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--yield_distance", type=float, default=0.0)
    parser.add_argument("--assert_distance", type=float, default=50.0)

    args = parser.parse_args()

    if not args.data_dir.exists():
        print(f"ERROR: Data directory not found: {args.data_dir}")
        sys.exit(2)

    if args.device == "cuda":
        import torch

        if not torch.cuda.is_available():
            print("WARNING: CUDA not available, using CPU")
            args.device = "cpu"

    config = TestConfig(
        data_dir=args.data_dir,
        device=args.device,
        scenario_idx=args.scenario_idx,
        timestep=args.timestep,
        verbose=args.verbose,
        yield_distance=args.yield_distance,
        assert_distance=args.assert_distance,
    )

    try:
        metrics = run_sensitivity_test(config)
        print(metrics.summary())
        sys.exit(0 if metrics.passed() else 1)

    except Exception as e:
        print(f"\nERROR: Test could not complete: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    main()
