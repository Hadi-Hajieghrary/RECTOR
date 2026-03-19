# RECTOR Sensitivity Test Finding

## Executive Summary

**DATE**: December 2024
**STATUS**: 🟢 RESOLVED
**GO/NO-GO GATE #1**: PASSED ✓

The sensitivity test initially failed, revealing a critical architectural issue. After root cause analysis and implementing the fix, the test now PASSES with strong sensitivity metrics.

---

## Final Test Results (After Fix)

```
SENSITIVITY TEST RESULT: ✓ PASS

Displacement Metrics:
  Mean displacement:  43.213m (threshold: >1.0m)  ← Was 0.000m
  Max displacement:   103.102m
  Final displacement: 103.102m

Probability Metrics:
  Probability shift:  0.165 (threshold: >0.1)     ← Was 0.000
  Mode changed:       No

Trajectory Shape:
  Reactor travel (yield):  106.16m
  Reactor travel (assert): 59.16m
  Delta:                   47.00m                  ← Was 0.00m

✓ The conditional model IS sensitive to influencer changes.
✓ Interactive planning is VIABLE. Proceed with RECTOR development.
```

**Weights Loaded: 294/302** (vs 124/302 before fix)

---

## Initial Failure Results

```
SENSITIVITY TEST RESULT: ✗ FAIL

Displacement Metrics:
  Mean displacement:  0.000m (threshold: >1.0m)
  Max displacement:   0.000m
  Final displacement: 0.000m

Probability Metrics:
  Probability shift:  0.000 (threshold: >0.1)
  Mode changed:       No
```

---

## Root Cause Analysis

### The Issue

The M2I conditional model was trained with **raster-based influencer encoding**, not vector-based `influencer_traj` fields. The model architecture expects:

1. **`'raster'` and `'raster_inf'` in `args.other_params`**
2. **A 150-channel raster image**:
   - Channels 0-59: Scene context (roads, ego history, other agents)
   - Channels 60-139: Influencer future trajectory (80 timesteps)
   - The extra 90 channels in CNNEncoder: `in_channels = 60 + 90`

### Evidence

From `/workspace/externals/M2I/src/modeling/vectornet.py`:
```python
class CNNEncoder(nn.Module):
    def __init__(self):
        if 'train_reactor' in args.other_params and 'raster_inf' in args.other_params:
            in_channels = 60 + 90  # ← Conditional mode
        else:
            in_channels = 60       # ← Marginal mode
```

From `/workspace/externals/M2I/src/utils_cython.pyx`:
```python
if 'train_reactor' in args.other_params and 'raster_inf' in args.other_params:
    if gt_influencer_traj_ is not None:
        # Rasterize influencer trajectory into channels 60+
        for j in range(time_frames):
            x = _raster_float_to_int(gt_influencer_traj_[current_time_frame, 0], 0, raster_scale)
            y = _raster_float_to_int(gt_influencer_traj_[current_time_frame, 1], 1, raster_scale)
            if _in_image(x, y):
                image[x, y, j + 60] = 1
```

### Current Implementation Issue

In `/workspace/models/pretrained/m2i/scripts/lib/m2i_receding_horizon_full.py`, the `_build_conditional_args()` method has:

```python
args.image = None  # ← WRONG: Conditional DOES use raster
args.other_params = OtherParams([
    'l1_loss',
    'densetnt',
    'goals_2D',
    # ...
    'train_reactor',
    # ← MISSING: 'raster' and 'raster_inf'!
], {'eval_time': 80})
```

This causes:
- CNN encoder not initialized (missing 178 weights out of 302)
- Influencer trajectory not rasterized
- Model runs in "marginal mode" - ignores influencer entirely

---

## Resolution Path

### Option A: Enable Raster Pathway (RECOMMENDED)

**Steps:**

1. **Update `_build_conditional_args()`**:
   ```python
   args.image = np.zeros([224, 224, 150], dtype=np.int8)  # Full conditional image
   args.other_params = OtherParams([
       'l1_loss',
       'densetnt',
       'goals_2D',
       'enhance_global_graph',
       'laneGCN',
       'point_sub_graph',
       'laneGCN-4',
       'stride_10_2',
       'train_pair_interest',
       'train_reactor',
       'raster',           # ← ADD THIS
       'raster_inf',       # ← ADD THIS
   ], {'eval_time': 80})
   ```

2. **Implement `_rasterize_influencer_trajectory()`**:
   ```python
   def _rasterize_influencer_trajectory(
       self,
       base_image: np.ndarray,      # [224, 224, 60] scene image
       influencer_traj: np.ndarray, # [80, 2] or [6, 80, 2] future trajectory
       normalizer,                  # Coordinate normalizer
   ) -> np.ndarray:
       """Render influencer trajectory into raster image channels 60+."""
       # Expand to 150 channels
       full_image = np.zeros([224, 224, 150], dtype=np.int8)
       full_image[:, :, :60] = base_image

       # Transform and rasterize each timestep
       for t in range(min(80, influencer_traj.shape[0])):
           x_world = influencer_traj[t, 0]
           y_world = influencer_traj[t, 1]
           x_local, y_local = self._world_to_local(x_world, y_world, normalizer)
           x_pix, y_pix = self._local_to_pixel(x_local, y_local)
           if 0 <= x_pix < 224 and 0 <= y_pix < 224:
               full_image[x_pix, y_pix, 60 + t] = 1

       return full_image
   ```

3. **Update `run_conditional_inference()`**:
   ```python
   def run_conditional_inference(
       self,
       reactor_mapping: Dict,
       influencer_trajectory: np.ndarray,
       influencer_scores: np.ndarray,
   ) -> Tuple[np.ndarray, np.ndarray]:
       # Get base scene image from reactor mapping
       base_image = reactor_mapping.get('image', self.args.image)[:, :, :60]

       # Rasterize influencer trajectory
       full_image = self._rasterize_influencer_trajectory(
           base_image, influencer_trajectory, reactor_mapping['normalizer']
       )

       # Create conditional mapping with rasterized image
       cond_mapping = reactor_mapping.copy()
       cond_mapping['image'] = full_image

       # Run inference
       return self._run_inference(cond_mapping)
   ```

### Option B: Vector-Based Conditioning (Requires Retraining)

Modify the VectorNet architecture to accept influencer trajectory as an additional polyline, similar to how map lanes are encoded. This requires:
- Model architecture changes
- Retraining on Waymo dataset
- ~100+ GPU hours

### Option C: Alternative Architecture

Use a separate neural network that takes both:
- Base M2I predictions
- Influencer trajectory

And outputs adjusted reactor predictions. This is essentially what M2I-Plan does at a higher level.

---

## The Fix Applied

The following changes were made to `/workspace/models/pretrained/m2i/scripts/lib/m2i_receding_horizon_full.py`:

### 1. Updated `_build_conditional_args()` (line ~430)

```python
# Before (broken):
args.image = None  # Conditional doesn't use raster
args.other_params = OtherParams([
    # ... missing 'raster' and 'raster_inf'
    'train_reactor',
], {'eval_time': 80})

# After (fixed):
args.image = np.zeros([224, 224, 150], dtype=np.int8)  # Full conditional image
args.other_params = OtherParams([
    'l1_loss',
    'densetnt',
    'goals_2D',
    'enhance_global_graph',
    'laneGCN',
    'point_sub_graph',
    'laneGCN-4',
    'stride_10_2',
    'train_pair_interest',
    'train_reactor',
    'raster',      # ← ADDED: Enable raster CNN encoder
    'raster_inf',  # ← ADDED: Enable influencer trajectory rasterization
], {'eval_time': 80})
```

### 2. Updated `_create_conditional_mapping()` (line ~1280)

Added code to rasterize the influencer trajectory into image channels 60-139:

```python
# CRITICAL: Rasterize influencer trajectory into image channels 60+
if 'raster' in self.conditional_args.other_params and 'raster_inf' in self.conditional_args.other_params:
    # Expand base image to 150 channels
    full_image = np.zeros([224, 224, 150], dtype=np.int8)
    full_image[:, :, :60] = base_image

    # Rasterize each timestep of influencer trajectory
    for t in range(80):
        x_world, y_world = best_traj[t, 0], best_traj[t, 1]
        # Transform to local coordinates
        x_local, y_local = transform_to_local(x_world, y_world, normalizer)
        # Convert to pixel coordinates
        x_pix, y_pix = local_to_pixel(x_local, y_local)
        if in_bounds(x_pix, y_pix):
            full_image[x_pix, y_pix, 60 + t] = 1

    cond_mapping['image'] = full_image
```

---

## Recommended Next Steps

1. **Implement Option A** in `m2i_receding_horizon_full.py`
2. **Re-run sensitivity test** to verify model now responds
3. **Proceed with RECTOR development** once test passes

---

## Appendix: Weight Count Analysis

| Model | Expected | Loaded | Status |
|-------|----------|--------|--------|
| DenseTNT (marginal) | 270 | 270 | ✓ |
| Relation V2V | 220 | 220 | ✓ |
| Conditional V2V (no raster) | 302 | 124 | ✗ Missing CNN |
| Conditional V2V (with raster) | 302 | ~302 | Expected |

The 178 missing weights (302 - 124) correspond to the CNNEncoder layers that aren't initialized when `'raster'` is not in `other_params`.
