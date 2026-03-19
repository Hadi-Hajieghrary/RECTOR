# Test Suite Overview

This directory verifies both core library behavior and script-level execution paths for the RECTOR subproject.

## Coverage map

Current tests cover:

- **Library contracts and adapters**: data contracts, batched interfaces, safety-layer behavior, planning-loop invariants.
- **Integration flows**: end-to-end planner behavior on real scenarios when local TFRecords are available.
- **Script modules (common/corner/complex usage)**:
	- CLI entrypoint `--help` smoke checks across evaluation, visualization, inference, training, and tuning scripts.
	- Statistical utility edge cases (`bootstrap_ci`, paired Wilcoxon behavior with sparse nonzero differences).
	- Inference pipeline unit behavior (shortlisting shapes, selection flow, return-all outputs) using lightweight dummy model/scorer fixtures.
	- Repository-wide syntax health for all Python files under `models/RECTOR/scripts`.

## Running with the project interpreter

Use the environment-specific Python executable directly:

- Full suite:
	- `/opt/venv/bin/python -m pytest -q models/RECTOR/tests`
- Script-level suite only:
	- `/opt/venv/bin/python -m pytest -q models/RECTOR/tests/test_scripts_entrypoints.py models/RECTOR/tests/test_scripts_stat_utils.py models/RECTOR/tests/test_scripts_inference_pipeline.py models/RECTOR/tests/test_scripts_syntax_health.py`

## Dataset-dependent tests

Some integration tests require local Waymo TFRecord files. If these files are unavailable, those tests now **skip** (rather than fail) with a clear message. This keeps CI and container validation stable while preserving real-data coverage when datasets are mounted.

## Why this matters for paper workflows

The suite is designed to catch behavioral drift before experiments are re-run or manuscript tables are regenerated. In practice, this protects method reproducibility and reduces the risk of silently inconsistent results after codebase refactors.
