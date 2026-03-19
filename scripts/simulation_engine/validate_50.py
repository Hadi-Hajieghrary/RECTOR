#!/usr/bin/env python3
"""
Validation script: Run 50 scenarios × 3 selectors and print summary.

Usage:
    source /opt/venv/bin/activate
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    export XLA_PYTHON_CLIENT_MEM_FRACTION=0.25
    cd /workspace
    python -m simulation_engine.validate_50

Notes:
    - Do NOT add parent to sys.path — 'selectors' clashes with stdlib.
    - max_objects=32 (reduced from 128) speeds up overlap metric ~16x.
    - max_rg_points=5000 (reduced from 30000) is sufficient for validation.
    - Offroad metric is excluded by default (very slow on non-JIT path).
    - env_factory auto-detects max_num_objects from the SimulatorState.
"""

import os
import time
import json

import numpy as np

from simulation_engine.waymax_bridge.scenario_loader import (
    load_scenarios,
    ScenarioLoaderConfig,
)
from simulation_engine.waymax_bridge.simulation_loop import (
    WaymaxRECTORLoop,
    MockLogReplayGenerator,
)
from simulation_engine.waymax_bridge.metric_collector import MetricCollectorConfig
from simulation_engine.selectors.confidence import ConfidenceSelector
from simulation_engine.selectors.weighted_sum import WeightedSumSelector
from simulation_engine.selectors.rector_lex import RECTORLexSelector
from simulation_engine.config import WaymaxBridgeConfig, AgentConfig


NUM_SCENARIOS = 50
MAX_SIM_STEPS = 80  # 8 seconds at 10Hz


def main():
    t_global = time.time()

    loader_cfg = ScenarioLoaderConfig(
        max_scenarios=NUM_SCENARIOS,
        max_objects=32,  # reduced from 128 for faster overlap computation
        max_rg_points=5000,  # reduced from 30000 for speed
    )
    bridge_cfg = WaymaxBridgeConfig(dynamics_model="delta", steps_per_replan=20)
    agent_cfg = AgentConfig(agent_model="log_playback")
    mc_cfg = MetricCollectorConfig(
        waymax_metrics=("log_divergence", "overlap", "kinematic_infeasibility"),
    )
    generator = MockLogReplayGenerator(num_modes=6, horizon=50)

    selectors = {
        "Confidence": ConfidenceSelector(),
        "WeightedSum": WeightedSumSelector(weights=np.ones(28)),
        "RECTORLex": RECTORLexSelector(epsilon=[1e-3, 1e-3, 1e-3, 1e-3]),
    }

    # Collect results: {selector_name -> list_of_metric_dicts}
    all_results = {name: [] for name in selectors}
    failed = []

    # Pre-load all scenarios
    print(f"Loading up to {NUM_SCENARIOS} scenarios...")
    scenarios = []
    for sid, sim_state in load_scenarios(loader_cfg):
        scenarios.append((sid, sim_state))
    print(f"Loaded {len(scenarios)} scenarios.\n")

    for s_idx, (sid, sim_state) in enumerate(scenarios):
        for sel_name, sel in selectors.items():
            try:
                loop = WaymaxRECTORLoop(
                    generator=generator,
                    selector=sel,
                    bridge_cfg=bridge_cfg,
                    agent_cfg=agent_cfg,
                    metric_cfg=mc_cfg,
                )
                result = loop.run_scenario(
                    sim_state, scenario_id=sid, max_sim_steps=MAX_SIM_STEPS
                )
                all_results[sel_name].append(result.metrics)
            except Exception as e:
                failed.append((sid, sel_name, str(e)))
                print(f"  FAIL [{s_idx+1}] {sid} | {sel_name}: {e}")

        # Progress
        if (s_idx + 1) % 5 == 0 or s_idx == 0:
            elapsed = time.time() - t_global
            eta = elapsed / (s_idx + 1) * (len(scenarios) - s_idx - 1)
            print(
                f"  [{s_idx+1}/{len(scenarios)}] "
                f"elapsed={elapsed:.0f}s eta={eta:.0f}s"
            )

    # --- Summary -------------------------------------------------------
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print(f"Scenarios: {len(scenarios)}, Selectors: {len(selectors)}")
    print(f"Failures: {len(failed)}")

    metric_keys = [
        "overlap/rate",
        "offroad/rate",
        "log_divergence/mean",
        "log_divergence/final",
        "min_clearance/min",
        "ttc/min",
        "ttc/mean",
        "jerk/mean",
        "jerk/max",
        "kinematic_infeasibility/rate",
    ]

    for sel_name in selectors:
        results = all_results[sel_name]
        if not results:
            print(f"\n{sel_name}: no results")
            continue
        print(f"\n{sel_name} ({len(results)} scenarios):")
        for mk in metric_keys:
            vals = [r.get(mk, float("nan")) for r in results]
            vals = np.array(vals)
            vals_clean = vals[np.isfinite(vals)]
            if len(vals_clean) == 0:
                print(f"  {mk:40s}: n/a")
            else:
                print(
                    f"  {mk:40s}: "
                    f"mean={vals_clean.mean():.4f}  "
                    f"std={vals_clean.std():.4f}  "
                    f"min={vals_clean.min():.4f}  "
                    f"max={vals_clean.max():.4f}"
                )

    total_time = time.time() - t_global
    print(f"\nTotal wall time: {total_time:.1f}s ({total_time/60:.1f}min)")

    # Save raw results
    out_dir = "/workspace/output/closedloop"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "validate_50_results.json")
    with open(out_path, "w") as f:
        json.dump(
            {
                name: [
                    {k: float(v) if np.isfinite(v) else None for k, v in r.items()}
                    for r in results_list
                ]
                for name, results_list in all_results.items()
            },
            f,
            indent=2,
        )
    print(f"Results saved to {out_path}")

    if failed:
        print(f"\nFailed scenarios:")
        for sid, sel_name, err in failed:
            print(f"  {sid} | {sel_name}: {err}")

    print("\nVALIDATION COMPLETE")


if __name__ == "__main__":
    main()
