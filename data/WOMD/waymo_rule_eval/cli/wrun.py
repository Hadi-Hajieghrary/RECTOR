#!/usr/bin/env python
# waymo_rule_eval/cli/wrun.py
# -*- coding: utf-8 -*-
"""
Command-line interface for Waymo Rule Evaluation.

Usage:
    python -m waymo_rule_eval.cli.wrun --scenario /path/to/*.tfrecord --out results.jsonl
    python -m waymo_rule_eval.cli.wrun --synthetic scenario.json --out results.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

from ..core.context import ScenarioContext
from ..io import CsvSink, JsonlSink, make_sink
from ..pipeline.rule_executor import WindowedExecutor
from ..utils.wre_logging import get_logger, reset_ctx, set_ctx

log = get_logger(__name__)


def parse_args(args=None):
    """Parse command-line arguments."""
    ap = argparse.ArgumentParser(
        prog="waymo-rule-eval",
        description="Evaluate Waymo motion scenarios against safety rules",
    )

    # Source input (mutually exclusive)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--synthetic",
        type=str,
        metavar="PATH",
        help="Path to synthetic JSON scenario file",
    )
    src.add_argument(
        "--scenario",
        type=str,
        metavar="GLOB",
        help="Glob pattern for Waymo Motion Scenario TFRecord files",
    )

    # Window parameters
    ap.add_argument(
        "--window-size",
        type=float,
        default=8.0,
        metavar="SECONDS",
        help="Window size in seconds (default: 8.0)",
    )
    ap.add_argument(
        "--stride",
        type=float,
        default=2.0,
        metavar="SECONDS",
        help="Stride between windows in seconds (default: 2.0)",
    )

    # Output
    ap.add_argument(
        "--out",
        type=str,
        required=True,
        metavar="PATH",
        help="Output path (.jsonl, .csv, or .parquet)",
    )

    # Run configuration
    ap.add_argument(
        "--run-id",
        type=str,
        default="run-0001",
        metavar="ID",
        help="Run identifier for tracking (default: run-0001)",
    )
    ap.add_argument(
        "--rules",
        type=str,
        default=None,
        metavar="IDS",
        help="Comma-separated list of rule IDs to run (default: all)",
    )
    ap.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )

    return ap.parse_args(args)


def iter_synthetic_scenarios(
    path: str,
) -> Iterator[ScenarioContext]:
    """
    Load synthetic scenarios from JSON.

    Args:
        path: Path to synthetic scenario JSON file

    Yields:
        ScenarioContext for each scenario
    """
    from ..data_access.adapter_motion_scenario import create_scenario_from_arrays

    def _as_1d_float(values, field_name: str) -> np.ndarray:
        arr = np.asarray(values, dtype=float).reshape(-1)
        if arr.size == 0:
            raise ValueError(f"Field '{field_name}' must be a non-empty array")
        return arr

    def _derive_yaw(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        if len(x) < 2:
            return np.zeros_like(x, dtype=float)
        dx = np.diff(x)
        dy = np.diff(y)
        yaw = np.zeros_like(x, dtype=float)
        yaw[1:] = np.arctan2(dy, dx)
        yaw[0] = yaw[1]
        return yaw

    def _derive_speed(x: np.ndarray, y: np.ndarray, dt_s: float) -> np.ndarray:
        if len(x) < 2:
            return np.zeros_like(x, dtype=float)
        dx = np.diff(x)
        dy = np.diff(y)
        v = np.sqrt(dx * dx + dy * dy) / max(dt_s, 1e-6)
        speed = np.zeros_like(x, dtype=float)
        speed[1:] = v
        speed[0] = speed[1]
        return speed

    with open(path, "r", encoding="utf-8") as fp:
        payload = json.load(fp)

    if isinstance(payload, list):
        scenarios = payload
    elif isinstance(payload, dict) and isinstance(payload.get("scenarios"), list):
        scenarios = payload["scenarios"]
    elif isinstance(payload, dict):
        scenarios = [payload]
    else:
        raise ValueError(
            "Synthetic input must be a scenario object, a list of scenarios, "
            "or an object containing a 'scenarios' list"
        )

    for idx, scenario in enumerate(scenarios):
        if not isinstance(scenario, dict):
            log.warning(
                f"Skipping synthetic scenario {idx}: expected object, got {type(scenario)}"
            )
            continue

        dt_s = float(scenario.get("dt", 0.1))
        scenario_id = str(scenario.get("scenario_id") or f"synthetic-{idx:04d}")

        ego_data = scenario.get("ego", scenario)
        if not isinstance(ego_data, dict):
            log.warning(f"Skipping {scenario_id}: 'ego' must be an object")
            continue

        x = _as_1d_float(ego_data.get("x", []), "ego.x")
        y = _as_1d_float(ego_data.get("y", []), "ego.y")
        if len(x) != len(y):
            raise ValueError(
                f"Scenario {scenario_id}: ego.x and ego.y length mismatch "
                f"({len(x)} != {len(y)})"
            )

        yaw_raw = ego_data.get("yaw")
        speed_raw = ego_data.get("speed")

        yaw = (
            _as_1d_float(yaw_raw, "ego.yaw")
            if yaw_raw is not None
            else _derive_yaw(x, y)
        )
        speed = (
            _as_1d_float(speed_raw, "ego.speed")
            if speed_raw is not None
            else _derive_speed(x, y, dt_s)
        )

        if len(yaw) != len(x):
            raise ValueError(
                f"Scenario {scenario_id}: ego.yaw length mismatch ({len(yaw)} != {len(x)})"
            )
        if len(speed) != len(x):
            raise ValueError(
                f"Scenario {scenario_id}: ego.speed length mismatch ({len(speed)} != {len(x)})"
            )

        agent_data = scenario.get("agents", [])
        if not isinstance(agent_data, list):
            raise ValueError(f"Scenario {scenario_id}: 'agents' must be a list")

        normalized_agents = []
        for agent_idx, agent in enumerate(agent_data):
            if not isinstance(agent, dict):
                log.warning(
                    f"Scenario {scenario_id}: skipping agent {agent_idx}, expected object"
                )
                continue

            ax = _as_1d_float(agent.get("x", []), f"agents[{agent_idx}].x")
            ay = _as_1d_float(agent.get("y", []), f"agents[{agent_idx}].y")
            if len(ax) != len(ay):
                raise ValueError(
                    f"Scenario {scenario_id}: agents[{agent_idx}] x/y length mismatch"
                )

            ayaw_raw = agent.get("yaw")
            aspeed_raw = agent.get("speed")
            ayaw = (
                _as_1d_float(ayaw_raw, f"agents[{agent_idx}].yaw")
                if ayaw_raw is not None
                else _derive_yaw(ax, ay)
            )
            aspeed = (
                _as_1d_float(aspeed_raw, f"agents[{agent_idx}].speed")
                if aspeed_raw is not None
                else _derive_speed(ax, ay, dt_s)
            )

            if len(ayaw) != len(ax) or len(aspeed) != len(ax):
                raise ValueError(
                    f"Scenario {scenario_id}: agents[{agent_idx}] yaw/speed length mismatch"
                )

            normalized_agents.append(
                {
                    "id": int(agent.get("id", agent_idx + 1)),
                    "type": agent.get("type", "vehicle"),
                    "x": ax,
                    "y": ay,
                    "yaw": ayaw,
                    "speed": aspeed,
                    "length": float(agent.get("length", 4.5)),
                    "width": float(agent.get("width", 1.8)),
                }
            )

        try:
            ctx = create_scenario_from_arrays(
                scenario_id=scenario_id,
                ego_x=x,
                ego_y=y,
                ego_yaw=yaw,
                ego_speed=speed,
                agent_data=normalized_agents,
                dt=dt_s,
            )
            yield ctx
        except Exception as exc:
            log.error(f"Failed to build synthetic scenario {scenario_id}: {exc}")
            continue


def iter_tfrecord_scenarios(
    glob_pattern: str,
) -> Iterator[ScenarioContext]:
    """
    Load scenarios from TFRecord files.

    Args:
        glob_pattern: Glob pattern for TFRecord files

    Yields:
        ScenarioContext for each scenario found
    """
    import glob

    from ..data_access.adapter_motion_scenario import MotionScenarioReader

    files = sorted(glob.glob(glob_pattern))
    if not files:
        log.warning(f"No files found matching: {glob_pattern}")
        return

    log.info(f"Found {len(files)} TFRecord files")

    reader = MotionScenarioReader()

    for fpath in files:
        log.info(f"Processing {fpath}")
        try:
            for ctx in reader.read_tfrecord(fpath):
                yield ctx
        except Exception as e:
            log.error(f"Failed to load {fpath}: {e}")
            continue


def main(args=None):
    """Main CLI entry point."""
    parsed = parse_args(args)

    # Configure logging
    if parsed.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # Set context for logging
    tokens = set_ctx(run_id=parsed.run_id)
    sink = None

    try:
        # Create output sink
        sink = make_sink(parsed.out)

        # Parse rule filter if provided
        rule_filter = None
        if parsed.rules:
            rule_filter = [r.strip() for r in parsed.rules.split(",")]
            log.info(f"Filtering to rules: {rule_filter}")

        # Create executor
        executor = WindowedExecutor(
            window_size_s=parsed.window_size, stride_s=parsed.stride
        )
        executor.register_all_rules()

        # Apply rule filter if provided
        if rule_filter:
            filtered_rules = [r for r in executor.rules if r.rule_id in rule_filter]
            if not filtered_rules:
                log.error(f"No matching rules found for filter: {rule_filter}")
                sink.close()
                return 1
            executor._executor._rules = filtered_rules
            log.info(
                f"Filtered to {len(filtered_rules)} rules: "
                f"{[r.rule_id for r in filtered_rules]}"
            )

        log.info(
            "Starting rule evaluation",
            extra={
                "extra_fields": {
                    "out": parsed.out,
                    "window_size": parsed.window_size,
                    "stride": parsed.stride,
                    "source": "synthetic" if parsed.synthetic else "tfrecord",
                    "run_id": parsed.run_id,
                }
            },
        )

        # Collect scenarios
        if parsed.synthetic:
            scenarios = list(
                iter_synthetic_scenarios(
                    parsed.synthetic,
                )
            )
        else:
            scenarios = list(
                iter_tfrecord_scenarios(
                    parsed.scenario,
                )
            )

        if not scenarios:
            log.error("No scenarios found to process")
            sink.close()
            return 1

        log.info(f"Processing {len(scenarios)} scenarios")

        # Run evaluation
        results = executor.run_batch(scenarios)

        # Write results
        n_written = 0
        for scenario_id, window_results in results.items():
            for window_result in window_results:
                for rule_result in window_result.rule_results:
                    sev_norm = 0.0
                    explanation = ""
                    if rule_result.violation:
                        sev_norm = rule_result.violation.severity_normalized
                        explanation = rule_result.violation.explanation
                        if isinstance(explanation, list):
                            explanation = "; ".join(explanation)
                    record = {
                        "run_id": parsed.run_id,
                        "scenario_id": window_result.scenario_id,
                        "window_start_ts": window_result.window_start_ts,
                        "window_idx": window_result.window_idx,
                        "rule_id": rule_result.rule_id,
                        "applies": rule_result.applies,
                        "severity": rule_result.severity,
                        "severity_normalized": sev_norm,
                        "explanation": explanation,
                    }
                    sink.write(record)
                    n_written += 1

        sink.close()

        log.info(f"Completed: wrote {n_written} records to {parsed.out}")
        print(f"[OK] Wrote {n_written} results to {parsed.out}")
        return 0

    except KeyboardInterrupt:
        log.warning("Interrupted by user (SIGINT)")
        try:
            if sink is not None:
                sink.close()
        except Exception:
            pass
        return 130

    except Exception as e:
        log.error(f"Fatal error: {e}", exc_info=True)
        try:
            if sink is not None:
                sink.close()
        except Exception:
            pass
        raise

    finally:
        reset_ctx(tokens)


if __name__ == "__main__":
    sys.exit(main() or 0)
