# data/scripts/lib

Python helper modules for Waymo dataset loading, preprocessing, and visualization.

**Last Updated:** November 14, 2025

## Key Files

| File | Role | Status |
|------|------|--------|
| `waymo_dataset.py` | **PRIMARY:** PyTorch dataset classes for loading Waymo data | ✅ Updated |
| `waymo_preprocess.py` | Converts scenario protobuf to .npz (legacy - scenario format unavailable) | ⚠️ Legacy |
| `filter_interactive_training.py` | Filters interactive scenarios from tf_example format | ✅ Working |
| `viz_waymo_tfexample.py` | Visualizes raw TFExample TFRecords | ✅ Working |
| `viz_waymo_scenario.py` | Visualizes preprocessed .npz scenarios | ✅ Working |
| `viz_trajectory.py` | Compares predicted vs ground truth trajectories | ✅ Working |

## waymo_dataset.py - Main Dataset Classes

**Updated November 14, 2025** with two dataset loaders:

### 1. WaymoTFExampleDataset (NEW - Recommended)

**Direct TFRecord loading without preprocessing:**

```python
from waymo_dataset import WaymoTFExampleDataset, build_tfexample_dataloader

# Option A: Use the builder function
dataloader = build_tfexample_dataloader(
    data_dir="data/datasets/waymo_open_dataset/motion_v_1_3_0/raw/tf_example/training_interactive",
    split="training_interactive",
    batch_size=4,
    num_workers=4,
    shuffle=True,
    history_frames=11,
    short_horizon_frames=80,
    long_horizon_frames=200,
)

# Option B: Create dataset directly
dataset = WaymoTFExampleDataset(
    data_dir="path/to/tfrecords",
    split="training_interactive",
    history_frames=11,
    short_horizon_frames=80,
    long_horizon_frames=200,
    max_scenarios=100,  # Optional: limit for testing
)
```

**Features:**
- ✅ Loads directly from TFRecord files
- ✅ No preprocessing required
- ✅ Works with all tf_example splits
- ✅ Automatic interactive pair detection
- ✅ Complete trajectory data (10 past + 1 current + 80 future = 91 frames)
- ✅ Road graph extraction (30,000 points)
- ✅ Batch collation included

**Output format:**
```python
sample = dataset[0]
# Keys: scenario_id, agent_trajectories, agent_valid, agent_types, 
#       agent_velocities, agent_headings, roadgraph_xyz, roadgraph_type, 
#       roadgraph_valid, interactive_pairs

# Shapes:
# agent_trajectories: [128, 211, 2] - all agents, all frames, (x,y)
# agent_valid: [128, 211] - validity mask
# interactive_pairs: [K, 2] - detected interactive pairs
```

### 2. WaymoITPDataset (Traditional)

**Loads preprocessed .npz files:**

```python
from waymo_dataset import WaymoITPDataset, build_dataloader

dataset = WaymoITPDataset(
    data_dir="data/datasets/waymo_open_dataset/motion_v_1_3_0/processed/training_interactive",
    split="train",
    history_frames=11,
    short_horizon_frames=80,
    long_horizon_frames=160,
    augment=True,
    use_gt_influencer=True,
)
```

**Features:**
- ✅ Faster loading (preprocessed .npz format)
- ✅ Smaller disk usage (compressed)
- ✅ Data augmentation support
- ✅ Map features included
- ❌ Requires preprocessing step

**Output format:**
```python
sample = dataset[0]
# Keys: scenario_id, history, future_short, future_long, valid_mask,
#       relation, agent_types, map_polylines, other_agents, ego_last_state

# Shapes:
# history: [2, 11, 2] - interactive pair history
# future_short: [2, 80, 2] - short horizon
# future_long: [2, 80, 2] - long horizon
# map_polylines: [256, 20, 7] - map features
```

## Common Usage Patterns

### Quick Testing
```python
# Test with 10 scenarios
dataset = WaymoTFExampleDataset(
    data_dir="path/to/tfrecords",
    split="training_interactive",
    max_scenarios=10,
)

# Check first sample
sample = dataset[0]
print(f"Scenario: {sample['scenario_id']}")
print(f"Trajectories: {sample['agent_trajectories'].shape}")
print(f"Interactive pairs: {len(sample['interactive_pairs'])}")
```

### Production Training
```python
dataloader = build_tfexample_dataloader(
    data_dir="path/to/tfrecords",
    split="training_interactive",
    batch_size=32,
    num_workers=8,
    shuffle=True,
    max_scenarios=None,  # Load all
)

for batch in dataloader:
    # batch['agent_trajectories']: [B, 128, 211, 2]
    # batch['interactive_pairs']: list of [K_i, 2] arrays
    ...
```

## Key Findings (November 14, 2025)

### TF_Example Format Contains Complete Data

**Previous misconception:** tf_example only has current frame, lacks history
**Truth:** tf_example has complete 91-frame trajectories:
- `state/past/*`: 10 frames (1.0s history)
- `state/current/*`: 1 frame  
- `state/future/*`: 80 frames (8.0s future)
- **Total**: 91 frames = 9.1 seconds at 10Hz

### Data Format Status

**✅ Available: tf_example format**
- All splits available (training, validation, testing, *_interactive)
- Complete trajectory data
- Road graph included
- Can be loaded directly or preprocessed

**❌ Unavailable: scenario protobuf format**
- May not be available in public dataset (as of Nov 2025)
- waymo_preprocess.py expects this format
- Use tf_example instead

## Testing

All dataset classes are tested:
```bash
# Test TFExample loading
python data/tests/test_tfexample_loading.py

# Test preprocessed .npz loading
python data/tests/test_preprocessing.py --data-dir path/to/processed/data
```
