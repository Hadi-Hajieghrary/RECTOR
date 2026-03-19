# Phase 3: Batched M2I Adapter

## Summary

**Status**: ✅ COMPLETE (10/10 tests passing)
**Date**: December 2024
**Location**: `/workspace/models/RECTOR/scripts/lib/batched_adapter.py`

## Overview

The Batched M2I Adapter wraps the M2I conditional prediction pipeline to enable efficient batch inference. Instead of running M×K serial forward passes for M ego candidates and K reactors, the adapter:

1. **Encodes scene ONCE** per planning tick
2. **Builds reactor tensor packs ONCE** per reactor
3. **Batches conditional inference** across all (candidate, reactor) pairs

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        BatchedM2IAdapter                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌───────────────────┐    ┌───────────────────┐   ┌─────────────────┐  │
│  │   PlanningConfig  │    │  RecedingHorizon  │   │ SceneEmbedding  │  │
│  │                   │    │       M2I         │   │     Cache       │  │
│  │ - max_reactors    │───>│ - DenseTNT        │   │ - map_polylines │  │
│  │ - num_candidates  │    │ - Relation V2V    │   │ - agent_states  │  │
│  │ - candidate_horizon│   │ - Conditional V2V │   │ - lane_features │  │
│  └───────────────────┘    └───────────────────┘   └─────────────────┘  │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │                    Core Methods                                    │ │
│  ├───────────────────────────────────────────────────────────────────┤ │
│  │  load_models()        → Load DenseTNT + Relation + Conditional    │ │
│  │  build_scene_cache()  → Extract & cache scene embedding           │ │
│  │  build_reactor_packs()→ Create ReactorTensorPack per agent        │ │
│  │  predict_batched()    → Batched conditional prediction            │ │
│  └───────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

## Key Classes

### BatchedM2IAdapter

Main adapter class that wraps M2I for planning use:

```python
from batched_adapter import BatchedM2IAdapter
from data_contracts import PlanningConfig

config = PlanningConfig(max_reactors=3, num_candidates=16)
adapter = BatchedM2IAdapter(config)
adapter.load_models()

# Per-tick workflow
scene_cache = adapter.build_scene_cache(scenario_data, timestep=10)
reactor_packs = adapter.build_reactor_packs(scene_cache, reactor_ids=[1, 2, 3])
predictions = adapter.predict_batched(reactor_packs, ego_candidates, scene_cache)
```

### M2IArgsContext

Context manager to isolate M2I's global state. M2I uses module-level globals (`utils.args`, `vectornet.args`) that get mutated when switching models. This prevents state contamination:

```python
with M2IArgsContext() as ctx:
    ctx.set_args(conditional_args)
    # All M2I operations use conditional_args
    # After exiting, previous args are restored
```

## Coordinate Transforms

The adapter handles world ↔ reactor-local coordinate transforms via private methods:

```python
# World → Reactor-local (before conditional prediction)
local_traj = adapter._transform_trajectory_to_local(world_traj, reactor_pose)

# Reactor-local → World (after prediction)
world_traj = adapter._transform_trajectory_to_world(local_traj, reactor_pose)
```

Key methods (on `BatchedM2IAdapter`):
- `_transform_trajectory_to_local()` - Transform to agent-centric frame
- `_transform_trajectory_to_world()` - Transform back to world frame
- Handles translation + rotation correctly

## Integration with Data Contracts

The adapter uses frozen dataclasses from `data_contracts.py`:

| Input | Type | Description |
|-------|------|-------------|
| `config` | `PlanningConfig` | Planning parameters |
| `ego_candidates` | `EgoCandidateBatch` | M candidate trajectories |

| Output | Type | Description |
|--------|------|-------------|
| `scene_cache` | `SceneEmbeddingCache` | Cached scene features |
| `reactor_packs` | `Dict[int, ReactorTensorPack]` | Per-reactor tensor packs |
| `predictions` | `List[PredictionResult]` | Conditional predictions |

## Test Results

```
$ python models/RECTOR/tests/test_batched_adapter.py

test_default_config         PASSED
test_custom_config          PASSED
test_identity_transform     PASSED
test_translation_only       PASSED
test_rotation_90_degrees    PASSED
test_roundtrip_transform    PASSED
test_predict_batched_shapes PASSED
test_select_closest_reactors PASSED
test_model_loading          PASSED
test_latency_benchmark      PASSED

10 passed in 85.88s
```

## Performance Characteristics

| Metric | Value | Notes |
|--------|-------|-------|
| Model Loading | ~60s | One-time startup cost |
| Scene Cache | ~10ms | Per-tick, cached |
| Reactor Packs | ~5ms/reactor | Computed once per reactor |
| Batched Inference | TBD | Target <50ms for M=16×K=3 |

## Dependencies

- **M2I Framework**: `externals/M2I/src/`
- **Pretrained Models**: `models/pretrained/m2i/models/`
- **Data Contracts**: `models/RECTOR/scripts/lib/data_contracts.py`
- **M2I Receding Horizon**: `models/pretrained/m2i/scripts/lib/m2i_receding_horizon_full.py`

## Files Created

| File | Purpose |
|------|---------|
| `scripts/lib/batched_adapter.py` | Main adapter implementation (931 lines) |
| `tests/test_batched_adapter.py` | Unit tests (10 tests) |
| `docs/PHASE3_BATCHED_ADAPTER.md` | This documentation |

## Next Steps (Phase 4)

With the batched adapter in place, Phase 4 will implement:

1. **Candidate Generation** - Generate M ego trajectory candidates
2. **Scoring Function** - Score candidates based on reactor predictions
3. **Selection Logic** - Select best candidate considering safety
4. **Planning Loop** - Main loop integrating all components
