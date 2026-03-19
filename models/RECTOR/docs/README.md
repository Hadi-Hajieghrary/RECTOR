# RECTOR Design Documentation

> **Directory:** `models/RECTOR/docs/`
> **Purpose:** Design notes, implementation plans, and debugging records for core RECTOR subsystems
> **Audience:** Developers working on RECTOR internals and adjacent evaluation or planning code

---

## Overview

This directory collects the internal design references for the RECTOR stack. The documents here are implementation-oriented: they explain why specific subsystems were built the way they were, what constraints shaped the design, and which findings materially changed the code path.

Use these files as engineering references, not as end-user documentation. For runnable entry points, CLI usage, and repository-level setup, start from the higher-level READMEs linked below.

## Documents

### M2I+RECTOR Pipeline (March 2026)

The M2I+RECTOR pipeline documentation is split across the following files:

| Document | Description |
|----------|-------------|
| [../scripts/lib/README.md](../scripts/lib/README.md) | `m2i_trajectory_generator.py` and `m2i_rule_selector.py` API reference |
| [../scripts/README.md](../scripts/README.md) | `generate_m2i_movies.py` usage and arguments |
| [../scripts/experiments/README.md](../scripts/experiments/README.md) | `evaluate_m2i_rector.py` evaluation pipeline |
| [../README.md](../README.md) | Overall M2I+RECTOR pipeline architecture and quick start |

Key implementation details:
- **M2I DenseTNT** (pretrained, 270 weights, about 120 MB) generates 6-mode, 80-step trajectories
- **KinematicRuleEvaluator** scores 28 rules across 4 tiers (Safety > Legal > Road > Comfort)
- **Receding-horizon planning** re-predicts every 2 seconds from the current scene state
- **40 simulation videos** are stored in `movies/m2i_rector/` (9.1 seconds each, 1800x1000, 10 fps)

### Classical Planning Infrastructure (Phases 3–5)

### `PHASE3_BATCHED_ADAPTER.md` — Batched M2I Adapter Design

Describes the `BatchedM2IAdapter`, which wraps M2I for efficient batch inference:
- Scene caching (`SceneEmbeddingCache`): encode once, reuse across candidates
- Reactor tensor packs: VectorNet-compatible data preparation
- True batched prediction: all M×K candidate-reactor pairs in one forward pass
- `M2IArgsContext`: thread-safe isolation of M2I's global `args` state
- Data contracts: `PlanningConfig`, `EgoCandidateBatch`, `ReactorTensorPack`, `PredictionResult`
- Test status at the time of writing: 10/10 passing

### `PHASE4_PLANNING_LOOP.md` — Planning Pipeline Design

Describes the five-stage `RECTORPlanner` pipeline:
1. **CandidateGenerator**: M=16 candidates via lane following, lane changes, emergency stop, goal-directed, perturbations
2. **ReactorSelector**: K=3 most relevant agents via distance + corridor + approach scoring
3. **Prediction**: M2I conditional inference via `BatchedM2IAdapter`
4. **CandidateScorer**: Multi-objective scoring (collision=100, comfort=5, progress=10, lane_deviation=2, hysteresis=3)
5. **Selection with hysteresis**: Threshold=1.0 score improvement required to switch
- Test status at the time of writing: 15/15 passing, about 250 ms without M2I

### `PHASE5_SAFETY_LAYER.md` — Safety Layer Design

Describes the `IntegratedSafetyChecker`, which combines:
- **RealityChecker**: Kinematic limits (a_lon [-8, 4], a_lat 6, jerk 12, yaw_rate 1.0, v_max 40)
- **NonCoopEnvelopeGenerator**: 5 worst-case trajectory samples per agent
- **OBBCollisionChecker**: SAT algorithm with safety margin
- **DeterministicCVaRScorer**: α=0.1, CRN for fair comparison, time-weighted costs
- Test status at the time of writing: 25/25 passing, under 10 ms per check

### `SENSITIVITY_TEST_FINDING.md` — Critical Debugging Finding

Documents a critical bug discovery and its fix:
- **Problem**: The conditional model produced zero sensitivity (mean displacement 0.000 m)
- **Root cause**: Missing `'raster'` and `'raster_inf'` flags in `other_params`, preventing CNN encoder initialization (178 weights not loaded)
- **Fix**: Added both flags, a 150-channel BEV image, and influencer trajectory rasterization into channels 60-139
- **Result**: 294/302 weights loaded, mean displacement 43.2 m, probability shift 0.165
- **Gate #1 PASSED** after fix

---

## Relationship to Code

| Document | Implements | Code Location |
|----------|-----------|---------------|
| Phase 3 | `BatchedM2IAdapter` | `scripts/lib/batched_adapter.py` |
| Phase 4 | `RECTORPlanner` | `scripts/lib/planning_loop.py` |
| Phase 5 | `IntegratedSafetyChecker` | `scripts/lib/safety_layer.py` |
| Sensitivity | `test_sensitivity.py` fix | `tests/test_sensitivity.py`, `pretrained/m2i/scripts/lib/m2i_receding_horizon_full.py` |

Where possible, treat the code as the source of truth and these documents as rationale. If implementation details diverge, update the document to match the code path that is actually exercised.
