# M2I Model Architecture: Three-Stage Prediction Pipeline

This directory contains the three pretrained M2I model weights that implement the complete factorized prediction pipeline. This document explains the architecture, role, and technical details of each model.

## Overview: The M2I Factorization Strategy

M2I decomposes the intractable joint prediction problem into a mathematically principled sequence of three specialized models:

```
P(Y_influencer, Y_reactor | X) ≈ P(Y_influencer | X) × P(Y_reactor | X, Y_influencer)
                                   └─────┬─────┘      └──────────┬──────────┘
                                    Marginal Model      Conditional Model
                                         ↑                       ↑
                                         └───────┬───────────────┘
                                           Relation Model
                                        (assigns roles)
```

Each model serves a specific purpose in the pipeline and is trained independently.

---

## Model 1: DenseTNT (Marginal Predictor)

### File Location
```
models/pretrained/m2i/models/densetnt/model.24.bin
```

### Purpose
Generate **independent** multi-modal trajectory predictions for each agent in the scene, without considering interactions.

### Architecture Details

**Context Encoder**: Hybrid approach combining two complementary representations

1. **VectorNet Branch**
   - Input: Vectorized polylines (agent trajectories + map features)
   - Architecture: Graph neural network over polyline graphs
   - Output: Agent-centric and map-centric embeddings

2. **Raster Branch**
   - Input: Bird's-eye-view rasterized image (224×224 pixels)
   - Architecture: VGG16 backbone pretrained on ImageNet
   - Channels: Historical agent positions, map lanes, crosswalks
   - Output: Spatial context features

**Prediction Head**: DenseTNT (Dense Goal-based TNT)

- **Strategy**: Anchor-free, goal-based prediction
- **Process**:
  1. Sample dense goal points from lane centerlines
  2. For each goal, decode a complete trajectory
  3. Score trajectories using goal likelihood and path plausibility
  4. Select top-K diverse modes (K=6 in M2I)

**Output Format**:
```python
{
  'rst': np.ndarray,    # Shape: [N_agents, 6, 80, 2] - 6 modes, 80 timesteps, XY coords
  'score': np.ndarray,  # Shape: [N_agents, 6] - confidence score per mode
  'ids': np.ndarray     # Shape: [N_agents] - Waymo track IDs
}
```

### Training Characteristics

- **Dataset**: Full Waymo training set (all scenarios, not just interactive)
- **Loss**: Goal-reaching loss + trajectory regression loss + diversity loss
- **Metric optimized**: minFDE (Final Displacement Error)
- **Checkpoint**: Epoch 24 (selected based on validation performance)

### Why "Marginal"?

This model treats each agent **independently**:
- No information about other agents' future intentions
- Assumes agent acts in isolation (no yielding, no reactive behavior)
- Produces **scene-inconsistent** predictions (may predict collisions)

This is intentional - consistency is added by the conditional model.

---

## Model 2: Relation V2V (Relation Predictor)

### File Location
```
models/pretrained/m2i/models/relation_v2v/model.25.bin
```

### Purpose
Classify the **relationship** between two interacting vehicles and assign **influencer** / **reactor** roles.

### Architecture Details

**Input**: Same hybrid context encoding as DenseTNT
- VectorNet: Agent history + map polylines
- Raster: BEV image with both agents highlighted

**Prediction Head**: 3-way classifier

**Output Classes**:
1. **PASS**: Agent 1 passes Agent 2 → Agent 1 = influencer, Agent 2 = reactor
2. **YIELD**: Agent 1 yields to Agent 2 → Agent 2 = influencer, Agent 1 = reactor
3. **NONE**: No meaningful interaction → both treated as independent

**Classification Criteria** (learned from data):
- Spatial proximity of future paths
- Relative arrival times at conflict point
- Speed differentials
- Map topology (merging lanes, intersections)

### Training Details

**Dataset**: Interactive scenarios only (exactly 2 `objects_of_interest`)

**Auto-Labeling Heuristic**:
To generate training labels, M2I uses a proxy for "influence":
1. Find point of **closest spatial approach** between agents' ground truth futures
2. Compare **arrival times** (t₁ vs t₂)
3. Agent arriving **later** = reactor (yielded)
4. Agent arriving **earlier** = influencer (passed)

⚠️ **Important limitation**: This heuristic is an approximation. True causality may differ.

**Training objective**: Cross-entropy loss on 3-way classification

**Checkpoint**: Epoch 25

### Why Relation Prediction Matters

The relation classifier enables the key factorization:
- Determines which agent's future should be predicted marginally (influencer)
- Determines which agent's future should be conditioned (reactor)
- Encodes social semantics (yielding vs. passing) critical for safe prediction

---

## Model 3: Conditional V2V (Conditional Predictor)

### File Location
```
models/pretrained/m2i/models/conditional_v2v/model.29.bin
```

### Purpose
Predict the **reactor's** trajectory **conditioned on** a specific influencer trajectory hypothesis.

### Architecture Details

**Key Innovation**: Augmented context encoding

The conditional model uses the **same base architecture** as DenseTNT (VectorNet + VGG16 raster) but with a critical modification:

**Augmented Input**:
1. **VectorNet Branch**:
   - Standard inputs: reactor's past, map polylines
   - **+ Extra polyline**: Influencer's future trajectory (from marginal prediction)
   - This extra polyline is encoded just like a map feature

2. **Raster Branch**:
   - Standard channels: map, historical positions
   - **+ 80 extra channels**: Influencer's future trajectory rasterized
   - One channel per future timestep (0.1s intervals)

This augmentation allows the model to "see" what the influencer is predicted to do and generate appropriate reactive behaviors.

**Prediction Head**: Same DenseTNT goal-based decoder
- Generates K=6 trajectory modes for the reactor
- Each mode is implicitly conditioned on the input influencer trajectory

### Conditional Prediction Process

For a single scenario with N influencer modes:

1. **For each** influencer mode i ∈ {1, ..., N}:
   ```python
   # Render influencer mode i
   augmented_context = add_influencer_trajectory(scene_context, influencer_modes[i])

   # Generate K reactor modes conditioned on influencer mode i
   reactor_modes[i] = conditional_model.predict(augmented_context)  # Shape: [K, 80, 2]
   ```

2. **Result**: N × K joint trajectory pairs
   ```
   (influencer_mode_1, reactor_mode_1.1), ..., (influencer_mode_1, reactor_mode_1.K)
   (influencer_mode_2, reactor_mode_2.1), ..., (influencer_mode_2, reactor_mode_2.K)
   ...
   (influencer_mode_N, reactor_mode_N.1), ..., (influencer_mode_N, reactor_mode_N.K)
   ```

3. **Sample Selection**: Compute joint likelihood and select top-K pairs
   ```python
   joint_score[i,j] = marginal_score[i] × conditional_score[i,j]
   selected = argsort(joint_score)[:K]  # Top-K most likely joint predictions
   ```

### Training Details

**Dataset**: Interactive scenarios with **relation labels**
- Only scenarios where relation model predicted PASS or YIELD
- Training samples: (reactor trajectory | influencer ground truth future)

**Training objective**:
- Goal-reaching loss (conditioned on influencer)
- Trajectory regression loss
- Negative log-likelihood

**Checkpoint**: Epoch 29

### Why Conditioning Works

By seeing the influencer's future trajectory, the conditional model learns to:
- **Yield**: Generate slowing/stopping trajectories when influencer crosses path
- **Wait**: Hold position if influencer occupies target lane
- **Accelerate**: Speed up to avoid conflict
- **Lane change**: Adjust lateral position in response to influencer

This reactive behavior is what makes the final joint prediction **scene-compliant**.

---

## Model Interactions: The Complete Pipeline

### Pipeline Flow

```
Input: Scene context X, two agents A and B

1. Relation Model:
   relation = predict_relation(X, A, B)
   if relation == YIELD(A, B):
       influencer = B, reactor = A
   elif relation == PASS(A, B):
       influencer = A, reactor = B
   else:
       # Independent prediction, no conditioning
       return marginal_predictions(A), marginal_predictions(B)

2. Marginal Model:
   influencer_modes = predict_marginal(X, influencer)  # Shape: [6, 80, 2]
   influencer_scores = get_scores(influencer_modes)     # Shape: [6]

3. Conditional Model:
   reactor_modes = []
   reactor_scores = []
   for i in range(6):
       X_aug = augment_context(X, influencer_modes[i])
       modes_i = predict_conditional(X_aug, reactor)    # Shape: [6, 80, 2]
       scores_i = get_scores(modes_i)                   # Shape: [6]
       reactor_modes.append(modes_i)
       reactor_scores.append(scores_i)

   reactor_modes = stack(reactor_modes)                 # Shape: [6, 6, 80, 2]
   reactor_scores = stack(reactor_scores)               # Shape: [6, 6]

4. Sample Selection:
   joint_scores = influencer_scores[:, None] * reactor_scores  # Broadcasting
   flat_indices = argsort(joint_scores.flatten())[::-1][:6]    # Top-6

   final_predictions = []
   for idx in flat_indices:
       i, j = idx // 6, idx % 6
       final_predictions.append({
           'influencer': influencer_modes[i],
           'reactor': reactor_modes[i, j],
           'score': joint_scores[i, j]
       })

Output: 6 joint trajectory pairs, ranked by likelihood
```

### Computational Complexity

**Relation**: O(1) forward pass per scenario
**Marginal**: O(1) forward pass per agent
**Conditional**: O(N) forward passes (N = number of influencer modes)

**Total cost**: Dominated by conditional model (6 forward passes for N=6)

---

## Model Variants and Specializations

### Why "V2V" (Vehicle-to-Vehicle)?

The provided pretrained models are specialized for **vehicle-to-vehicle** interactions:
- Trained only on scenarios where both agents are VEHICLE type
- WOMD has abundant v2v interaction data (thousands of examples)
- Performance is highest for this pairing

**Other interaction types** (v2p, v2c, p2p, etc.) have limited training data in WOMD, leading to lower performance. The M2I framework is general and could be retrained on more diverse interaction datasets.

---

## Model File Format

All three models use the same serialization format:

**File type**: PyTorch model checkpoint (`.bin`)

**Contents**:
```python
checkpoint = {
    'model_state_dict': OrderedDict(...),  # Model weights
    'epoch': int,                           # Training epoch
    # Optimizer state, scheduler state may also be present
}
```

**Loading**:
```python
import torch
from modeling.vectornet import VectorNet  # From M2I repo

model = VectorNet(config)
checkpoint = torch.load('model.24.bin', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

---

## Pretrained Weight Sources

The `.bin` files are **not included** in this repository due to size.

**To obtain weights**:
1. Download from M2I official release (if available)
2. Train from scratch using M2I training code
3. Contact M2I authors

**Expected file sizes**:
- `densetnt/model.24.bin`: ~100-200 MB
- `relation_v2v/model.25.bin`: ~100-200 MB
- `conditional_v2v/model.29.bin`: ~100-200 MB

---

## Model Configuration Files

Each model directory may contain an `eval.args` file (not included in all setups):

**Purpose**: Stores hyper-parameters used during model evaluation

**Typical contents**:
```bash
--gpu
--future_frame_num 80
--mode_num 6
--data validation_interactive
--other_params ...
```

These are reference files from the upstream M2I repository.

---

## Performance Characteristics

### DenseTNT (Marginal)
- **Strength**: Diverse, plausible individual trajectories
- **Weakness**: Ignores interactions → may predict collisions
- **Metrics**: Strong minFDE, good diversity

### Relation V2V
- **Strength**: Learns social semantics from data
- **Weakness**: Dependent on auto-labeling heuristic quality
- **Metrics**: ~85-90% classification accuracy on WOMD interactive

### Conditional V2V
- **Strength**: Reactive, scene-compliant predictions
- **Weakness**: Computational cost scales with influencer modes
- **Metrics**: Combined with marginal, achieves best mAP

### Combined Pipeline
- **Official WOMD ranking metric (mAP)**: 0.16 (vehicle), 0.08 (all types)
- **Significantly outperforms** marginal-only and joint baselines

---

## Troubleshooting

### "Model file not found"
**Cause**: Weight files not downloaded
**Solution**: Place `.bin` files in exact paths shown above

### "State dict mismatch"
**Cause**: Model architecture version incompatibility
**Solution**: Ensure upstream M2I repo version matches pretrained weights

### "CUDA out of memory"
**Cause**: Batch size too large or too many influencer modes
**Solution**: Reduce batch size or run on CPU (slower)

### "Predictions look unrealistic"
**Cause**: Parser compatibility issue or incorrect model order
**Solution**: Verify `waymo_flat_parser.py` is applied before dataset creation

---

## Related Documentation

- [M2I Inference Scripts](../README.md) - How to run these models
- [Parser Compatibility](../scripts/lib/waymo_flat_parser.py) - Data format fixes
- [Main README](../../../../README.md) - Complete pipeline overview

---

## Research Context

These models implement the approach described in:

**M2I: From Factored Marginal Trajectory Prediction to Interactive Prediction**
Sun et al., CVPR 2022

Key contributions:
1. Influencer-reactor factorization of joint prediction
2. Augmented context encoding for conditional prediction
3. State-of-the-art results on WOMD interactive benchmark

The factorization strategy is mathematically principled and empirically effective.
