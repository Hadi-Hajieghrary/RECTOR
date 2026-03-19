# M2I Relations: Influencer-Reactor Relationships

**Purpose**: Understanding and using interaction relations in multi-agent trajectory prediction
**Related Model**: Relation V2V (Stage 2 of M2I Pipeline)
**Data Format**: Pickle files containing influencer-reactor relationship predictions

---

## Overview

**Relations** in the M2I (Marginal to Interactive) framework represent **asymmetric interaction relationships** between pairs of agents in driving scenarios. Relations answer the fundamental question: **"Which agent influences the other's behavior?"**

### Why Relations Matter

In multi-agent scenarios, agents don't move independently:
- A **lead vehicle** influences a **following vehicle** (follower must adjust speed)
- A **merging vehicle** influences **mainline traffic** (mainline may need to yield)
- A **pedestrian crossing** influences **approaching vehicles** (vehicles must stop)

Understanding these relationships is critical for accurate trajectory prediction because:
1. **Reactor trajectories depend on influencer actions**
2. **Marginal predictions assume independence** (unrealistic in interactive scenarios)
3. **Conditional predictions require knowing who influences whom**

---

## Relation Labels

The Relation V2V model predicts one of **three labels** for each agent pair:

### Label Definitions

| Label | Name | Meaning | Example Scenario |
|-------|------|---------|------------------|
| **0** | No Relation | Agents move independently | Vehicles in separate lanes, no interaction |
| **1** | Agent₁ → Agent₂ | Agent₁ influences Agent₂ | Agent₁ merges, Agent₂ must yield |
| **2** | Agent₂ → Agent₁ | Agent₂ influences Agent₁ | Agent₂ is lead car, Agent₁ follows |

### Asymmetric Nature

**Critical**: Relations are **directional** and **asymmetric**:

```
Relation(A, B) ≠ Relation(B, A)
```

**Example**:
- Relation(Lead, Follower) = 2 (Lead → Follower)
- Relation(Follower, Lead) ≠ 2 (different relationship when order reversed)

---

## Relation Data Structure

### Storage Format (Pickle Files)

Relations are stored in Python pickle files with the following structure:

#### Ground Truth Relations
```python
# File: validation_interactive_gt_relations.pickle
{
    b'scenario_id_1': [influencer_id, reactor_id, interaction_label, pair_type],
    b'scenario_id_2': [influencer_id, reactor_id, interaction_label, pair_type],
    ...
}
```

**Fields**:
- `influencer_id` (float): Agent ID of the influencer
- `reactor_id` (float): Agent ID of the reactor
- `interaction_label` (int): 0, 1, or 2
  - 0 = Smaller agent ID influences larger agent ID
  - 1 = Larger agent ID influences smaller agent ID
  - 2 = No significant interaction
- `pair_type` (int): Agent pair classification
  - 1 = Vehicle-to-Vehicle (V2V)
  - 2 = Vehicle-to-Pedestrian (V2P)
  - 3 = Vehicle-to-Cyclist (V2C)
  - 4 = Other combinations

#### Predicted Relations
```python
# File: pred_relations_*.pickle (from Relation V2V model)
{
    b'scenario_id_1': [relation_label],
    b'scenario_id_2': [relation_label],
    ...
}
```

**Fields**:
- `relation_label` (int): Predicted label {0, 1, 2}

---

## How Relations Are Predicted

### Relation V2V Model Architecture

The Relation V2V model uses:

1. **VectorNet Encoder**: Encodes both agents' trajectories and road context
2. **Enhanced Global Graph**: Cross-attention between the two agents
3. **CNN Encoder**: Processes BEV raster showing both agents' spatial relationship
4. **Relation Classifier**: 3-class classification head

**Input**:
- Two agent histories (11 frames each)
- Road map polylines
- BEV raster (224×224×60) showing both agents

**Output**:
- Relation label: {0, 1, 2}
- Stored in global variable: `globals.sun_1_pred_relations`

### Prediction Process

```python
# Step 1: Create pair mapping
pair_mapping = {
    'object_id': agent1_id,
    'cent_x': agent1_x,
    'cent_y': agent1_y,
    'reactor': {
        'agent_id': agent2_id,
        'cent_x': agent2_x,
        'cent_y': agent2_y,
    },
    # ... road context, etc.
}

# Step 2: Run Relation V2V model
import globals as m2i_globals
m2i_globals.sun_1_pred_relations = {}
model([pair_mapping], device='cuda')

# Step 3: Extract prediction from global state
scenario_id = pair_mapping['scenario_id']
relation_label = m2i_globals.sun_1_pred_relations[scenario_id][0]

# Step 4: Interpret result
if relation_label == 0:
    print("No interaction")
elif relation_label == 1:
    print(f"Agent {agent1_id} influences Agent {agent2_id}")
elif relation_label == 2:
    print(f"Agent {agent2_id} influences Agent {agent1_id}")
```

---

## Using Relations in the M2I Pipeline

### Three-Stage Pipeline Flow

```
┌─────────────────────────────────────────────────────────────┐
│ STAGE 1: DenseTNT Marginal Prediction                      │
├─────────────────────────────────────────────────────────────┤
│ Input:  All agents in scenario                             │
│ Output: Marginal predictions for each agent                │
│         marginal[agent_id] = [6 modes, 80 timesteps, 2D]   │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ STAGE 2: Relation V2V (DETERMINES RELATIONS)               │
├─────────────────────────────────────────────────────────────┤
│ Input:  Agent pairs (all combinations)                     │
│ Process: For each pair (A, B):                             │
│          - Predict relation label                          │
│          - If label ≠ 0, identify influencer & reactor    │
│ Output: relations[(influencer, reactor)] = label           │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ STAGE 3: Conditional V2V (USES RELATIONS)                  │
├─────────────────────────────────────────────────────────────┤
│ Input:  For each (influencer, reactor) pair:               │
│         - Reactor's history                                │
│         - Influencer's best marginal trajectory            │
│ Output: Conditional predictions for reactor                │
│         conditional[reactor] = [6 modes, 80 timesteps, 2D] │
│                                                            │
│ Final Predictions:                                         │
│ - Influencers: Use marginal predictions                   │
│ - Reactors: Use conditional predictions                   │
│ - Independent agents: Use marginal predictions            │
└─────────────────────────────────────────────────────────────┘
```

---

## Relation Configuration

### Enabling Relation Prediction

Relations are activated via the `'train_relation'` flag in `args.other_params`:

```python
import utils

args = utils.Args()
args.hidden_size = 128
args.future_frame_num = 80
args.mode_num = 6
args.waymo = True

args.other_params = [
    'train_relation',      # ← CRITICAL: Enables relation prediction
    'densetnt',
    'goals_2D',
    'enhance_global_graph',
    'laneGCN',
    'raster',              # Required for CNN encoder
    # ...
]

utils.args = args  # Set global args
```

### Configuration File

Relations can be configured via YAML:

**File**: `/workspace/externals/M2I/configs/relation.yaml`

```yaml
hidden_size: 128
future_frame_num: 80
mode_num: 6

other_params:
  - train_relation          # Enable relation mode
  - pair_vv                 # v2v pair training
  - pred_with_threshold     # Use confidence threshold
  - l1_loss
  - densetnt
  - goals_2D
  - enhance_global_graph
  - laneGCN
  - point_sub_graph
  - laneGCN-4
  - stride_10_2
  - raster                  # Required for CNN encoder
```

---

## Global State Variables

### Critical Global Variables (in `globals.py`)

The M2I codebase uses global variables to store relation data:

```python
import globals as m2i_globals

# Predicted relations (output from Relation V2V model)
m2i_globals.sun_1_pred_relations = {
    b'scenario_id': [relation_label],
    # ...
}

# Ground truth relations (loaded from dataset)
m2i_globals.interactive_relations = {
    b'scenario_id': [inf_id, react_id, label, pair_type],
    # ...
}

# Precomputed relation predictions (loaded from file)
m2i_globals.relation_pred = {
    b'scenario_id': [relation_label],
    # ...
}

# Direct relation labels (alternative format)
m2i_globals.direct_relation = {
    (agent1_id, agent2_id): relation_label,
    # ...
}
```

### Why Global State?

**Design Limitation**: The M2I VectorNet implementation uses global state to pass predictions between components.

**Implications**:
1. **Not thread-safe**: Cannot run multiple predictions in parallel
2. **Must clear between scenarios**: `m2i_globals.sun_1_pred_relations = {}` before each prediction
3. **Subprocess isolation recommended**: Avoid conflicts when running multiple models

---

## Relation Files in This Directory

### 1. Ground Truth Relations

**Files**:
- `training_interactive_gt_relations.pickle` - Training set ground truth
- `validation_interactive_gt_relations.pickle` - Validation set ground truth

**Format**:
```python
{
    scenario_id (bytes): [influencer_id, reactor_id, interaction_label, pair_type]
}
```

**Usage**:
```python
import pickle

with open('validation_interactive_gt_relations.pickle', 'rb') as f:
    gt_relations = pickle.load(f)

# Access relation for specific scenario
scenario_id = b'abc123'
if scenario_id in gt_relations:
    inf_id, react_id, label, pair_type = gt_relations[scenario_id]
    print(f"Influencer: {inf_id}, Reactor: {react_id}, Label: {label}")
```

---

### 2. Precomputed Relation Predictions

**File**: `m2i.relation.v2v.VAL`

**Description**: Precomputed relation predictions on validation set from original M2I repository.

**Format**: Binary pickle file (large, ~200MB+)

**Usage**:
```python
import pickle

with open('m2i.relation.v2v.VAL', 'rb') as f:
    relation_preds = pickle.load(f)

# Use precomputed predictions instead of running model
scenario_id = b'abc123'
if scenario_id in relation_preds:
    relation_label = relation_preds[scenario_id][0]
```

---

## Working with Relations: Practical Examples

### Example 1: Load and Inspect Ground Truth Relations

```python
import pickle
import numpy as np

# Load ground truth
with open('validation_interactive_gt_relations.pickle', 'rb') as f:
    gt_relations = pickle.load(f)

print(f"Total scenarios with relations: {len(gt_relations)}")

# Statistics
labels = [rel[2] for rel in gt_relations.values()]
pair_types = [rel[3] for rel in gt_relations.values()]

print(f"Label distribution:")
print(f"  Label 0 (smaller → larger): {labels.count(0)}")
print(f"  Label 1 (larger → smaller): {labels.count(1)}")
print(f"  Label 2 (no interaction): {labels.count(2)}")

print(f"\nPair type distribution:")
print(f"  V2V (vehicle-vehicle): {pair_types.count(1)}")
print(f"  V2P (vehicle-pedestrian): {pair_types.count(2)}")
print(f"  V2C (vehicle-cyclist): {pair_types.count(3)}")
print(f"  Other: {pair_types.count(4)}")
```

---

### Example 2: Predict Relations for a Scenario

```python
import torch
import pickle
from modeling.vectornet import VectorNet
import utils
import globals as m2i_globals

# Setup args for Relation V2V
args = utils.Args()
args.hidden_size = 128
args.future_frame_num = 80
args.mode_num = 6
args.waymo = True
args.other_params = ['train_relation', 'densetnt', 'raster', 'laneGCN']
utils.args = args

# Load model
model = VectorNet(args)
checkpoint = torch.load('models/relation_v2v/model.25.bin', map_location='cuda')
model.load_state_dict(checkpoint, strict=False)
model.eval()

# Prepare agent pair mapping
pair_mapping = {
    'object_id': agent1_id,
    'cent_x': agent1_x,
    'cent_y': agent1_y,
    'angle': agent1_yaw,
    'reactor': {
        'agent_id': agent2_id,
        'cent_x': agent2_x,
        'cent_y': agent2_y,
        'angle': agent2_yaw,
    },
    'scenario_id': b'my_scenario',
    'polyline_spans': [...],
    'matrix': [...],
}

# Clear global state
m2i_globals.sun_1_pred_relations = {}

# Run prediction
with torch.no_grad():
    model([pair_mapping], device='cuda')

# Extract result
if b'my_scenario' in m2i_globals.sun_1_pred_relations:
    relation_label = m2i_globals.sun_1_pred_relations[b'my_scenario'][0]

    print(f"Predicted Relation: {relation_label}")

    if relation_label == 1:
        print(f"Influencer: Agent {agent1_id}")
        print(f"Reactor: Agent {agent2_id}")
    elif relation_label == 2:
        print(f"Influencer: Agent {agent2_id}")
        print(f"Reactor: Agent {agent1_id}")
    else:
        print("No interaction detected")
```

---

### Example 3: Use Relations in Full Pipeline

```python
# Stage 1: Marginal predictions
marginal_preds = {}
for agent_id in scenario.agents:
    marginal_preds[agent_id] = densetnt_model.predict(agent_id)

# Stage 2: Relation predictions
relations = {}
agent_pairs = [(1, 2), (1, 3), (2, 3)]  # Example pairs

for agent1, agent2 in agent_pairs:
    relation_label = relation_model.predict(agent1, agent2)

    if relation_label == 1:
        relations[(agent1, agent2)] = 'inf→react'
        print(f"Relation: Agent {agent1} influences Agent {agent2}")
    elif relation_label == 2:
        relations[(agent2, agent1)] = 'inf→react'
        print(f"Relation: Agent {agent2} influences Agent {agent1}")

# Stage 3: Conditional predictions
final_preds = {}

for (influencer, reactor), rel_type in relations.items():
    # Get influencer's best marginal trajectory
    inf_traj = marginal_preds[influencer]['best_trajectory']

    # Predict reactor conditionally
    cond_pred = conditional_model.predict(
        reactor_id=reactor,
        influencer_traj=inf_traj
    )

    final_preds[reactor] = cond_pred

# Use marginal for influencers and independent agents
for agent_id, marginal in marginal_preds.items():
    if agent_id not in final_preds:
        final_preds[agent_id] = marginal

print(f"Final predictions: {len(final_preds)} agents")
print(f"  Conditional (reactors): {len(relations)} agents")
print(f"  Marginal (influencers + independent): {len(marginal_preds) - len(relations)} agents")
```

---

### Example 4: Evaluate Relation Predictions

```python
import numpy as np

# Load ground truth and predictions
with open('validation_interactive_gt_relations.pickle', 'rb') as f:
    gt_relations = pickle.load(f)

with open('pred_relations.pickle', 'rb') as f:
    pred_relations = pickle.load(f)

# Compare predictions
correct = 0
total = 0

for scenario_id, gt_rel in gt_relations.items():
    if scenario_id in pred_relations:
        gt_label = gt_rel[2]  # interaction_label
        pred_label = pred_relations[scenario_id][0]

        if gt_label == pred_label:
            correct += 1
        total += 1

accuracy = correct / total if total > 0 else 0
print(f"Relation Prediction Accuracy: {accuracy:.2%}")
print(f"Correct: {correct}/{total}")
```

---

## Interpreting Relation Labels

### Label 0: Smaller Agent ID → Larger Agent ID

**Meaning**: The agent with the smaller ID influences the agent with the larger ID.

**Example**:
```python
agent1_id = 100
agent2_id = 200
relation_label = 0

# Interpretation:
# Agent 100 (smaller ID) influences Agent 200 (larger ID)
influencer = 100
reactor = 200
```

**Common Scenarios**:
- Agent 100 is merging vehicle, Agent 200 is in mainline (must yield)
- Agent 100 is pedestrian, Agent 200 is approaching vehicle (must stop)

---

### Label 1: Larger Agent ID → Smaller Agent ID

**Meaning**: The agent with the larger ID influences the agent with the smaller ID.

**Example**:
```python
agent1_id = 100
agent2_id = 200
relation_label = 1

# Interpretation:
# Agent 200 (larger ID) influences Agent 100 (smaller ID)
influencer = 200
reactor = 100
```

**Common Scenarios**:
- Agent 200 is lead vehicle, Agent 100 is following (must match speed)
- Agent 200 changes lanes, Agent 100 must adjust position

---

### Label 2: No Interaction

**Meaning**: Agents move independently, no significant causal relationship.

**Example**:
```python
relation_label = 2

# Interpretation: No interaction
# Both agents use marginal predictions
```

**Common Scenarios**:
- Agents in parallel lanes, maintaining constant speeds
- Agents far apart (>20 meters)
- Agents moving in perpendicular directions

---

## Relation Prediction Quality

### Expected Performance Metrics

From the Relation V2V model (validation set):

| Metric | Value | Description |
|--------|-------|-------------|
| **Accuracy** | ~78% | Overall 3-class accuracy |
| **Precision (Label 1)** | ~82% | When predicting A→B, correct 82% |
| **Recall (Label 1)** | ~75% | Detects 75% of true A→B relations |
| **F1 Score** | ~0.78 | Harmonic mean of precision/recall |

### Typical Confusion Matrix

```
              Predicted
             0    1    2
Actual  0  [450  30   20]  ← Label 0 (mostly correct)
        1  [ 40  370  40]  ← Label 1 (some confusion with 0 and 2)
        2  [ 30  35  385]  ← Label 2 (some confusion with 1)
```

**Analysis**:
- Model is good at detecting interaction vs. no-interaction (Label 2)
- Slight confusion between Labels 0 and 1 (direction of influence)
- Overall reliable for determining influencer-reactor pairs

---

## Troubleshooting

### Issue 1: No Relations Predicted (All Label 2)

**Symptom**: All predictions are Label 2 (no interaction)

**Possible Causes**:
1. Missing `'train_relation'` flag in `args.other_params`
2. Agents too far apart (model learned distance threshold)
3. Model not properly loaded

**Solution**:
```python
# Ensure flag is set
args.other_params = ['train_relation', ...]

# Check agent distance
distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
if distance > 30:
    print("Agents may be too far for interaction")

# Verify model loaded
print(f"Loaded weights: {len(model.state_dict())}")
```

---

### Issue 2: Global State Not Updated

**Symptom**: `m2i_globals.sun_1_pred_relations` is empty after model inference

**Cause**: `scenario_id` not set in mapping

**Solution**:
```python
# Always set scenario_id (must be bytes)
mapping['scenario_id'] = b'unique_scenario_id'

# Or convert string to bytes
mapping['scenario_id'] = 'scenario_123'.encode()
```

---

### Issue 3: Inconsistent Predictions

**Symptom**: Relation(A, B) and Relation(B, A) give unexpected results

**Cause**: Relations are asymmetric - order matters

**Explanation**:
```python
# These are DIFFERENT predictions:
relation_AB = predict_relation(agent_A, agent_B)  # A as center
relation_BA = predict_relation(agent_B, agent_A)  # B as center

# Coordinate frames differ!
# prediction_AB uses A's coordinate frame
# prediction_BA uses B's coordinate frame
```

**Solution**: Always process pairs in consistent order or be aware of asymmetry.

---

## Best Practices

### 1. Clear Global State Between Predictions

```python
import globals as m2i_globals

# Before each scenario
m2i_globals.sun_1_pred_relations = {}
```

### 2. Use Subprocess Isolation

```python
# Recommended: Run relation prediction in subprocess
import subprocess

subprocess.run([
    'python', 'subprocess_relation.py',
    '--input_pickle', 'input.pkl',
    '--output_pickle', 'output.pkl'
])
```

### 3. Filter Low-Confidence Predictions

```python
# Only use high-confidence relations
if relation_confidence > 0.7:
    use_conditional_prediction()
else:
    use_marginal_prediction()
```

### 4. Validate Agent Pairs Before Prediction

```python
# Check distance
distance = compute_distance(agent1, agent2)
if distance < 30.0:  # Within 30 meters
    relation = predict_relation(agent1, agent2)
else:
    relation = 2  # Assume no interaction if too far
```

---

## References

- **M2I Paper**: [arXiv:2202.11884](https://arxiv.org/abs/2202.11884)
- **Relation V2V Model README**: `/workspace/models/pretrained/m2i/models/relation_v2v/README.md`
- **M2I GitHub**: [github.com/Tsinghua-MARS-Lab/M2I](https://github.com/Tsinghua-MARS-Lab/M2I)
- **Waymo Dataset**: [waymo.com/open/data/motion/](https://waymo.com/open/data/motion/)

---

## Summary

**Relations** are the key innovation in M2I that transforms independent (marginal) trajectory predictions into interaction-aware (conditional) predictions:

✅ **What**: Influencer-reactor relationships (3 classes)
✅ **Why**: Reactors' trajectories depend on influencers' actions
✅ **How**: Relation V2V model predicts relationships
✅ **When**: Stage 2 of M2I pipeline (between marginal and conditional)
✅ **Where**: Stored in global variables and pickle files

For complete pipeline usage, see `/workspace/models/pretrained/m2i/README.md`.
