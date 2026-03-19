# Augmentation Module

The `augmentation/` module provides tools to augment Waymo Open Motion Dataset TFRecords with rule evaluation data, creating enriched datasets suitable for machine learning pipelines.

---

## Table of Contents

1. [Overview](#overview)
2. [Philosophy](#philosophy)
3. [Output Formats](#output-formats)
4. [TFRecord Augmentation](#tfrecord-augmentation)
5. [Command-Line Interface](#command-line-interface)
6. [Python API](#python-api)
7. [Data Schema](#data-schema)
8. [Processing Large Datasets](#processing-large-datasets)
9. [Integration with ML Pipelines](#integration-with-ml-pipelines)
10. [Troubleshooting](#troubleshooting)

---

## Overview

This module enables two main workflows:

> **Execution context:** Run `python -m waymo_rule_eval.augmentation...` commands from `/workspace/data/WOMD` (or set `PYTHONPATH=/workspace/data/WOMD`). The package is not installed as a site package in this workspace.

| Workflow | Tool | Output |
|----------|------|--------|
| TFRecord Augmentation | `augment_cli.py` | Augmented TFRecords with rule features |
| JSONL Generation | `process_scenarios.py` | JSONL with evaluation details |

The TFRecord augmentation is designed for seamless integration with existing Waymo data pipelines, preserving all original data while adding rule evaluation features.

---

## Philosophy

### Preservation of Original Data

The augmentation process **never modifies or removes original data**. Augmented TFRecords contain:
- All original Waymo scenario features (unchanged)
- Additional rule evaluation features (appended)

This ensures backward compatibility with existing data loaders.

### ML-Ready Features

Rule evaluations are encoded as fixed-size arrays for easy batching:

```
rule/applicability: [1, 1, 0, 1, ...]  # 28-element array (all 28 rules registered)
rule/violations:    [0, 0, 0, 1, ...]  # 28-element array
rule/severity:      [0.0, 0.0, 0.0, 0.8, ...]  # 28-element float array
rule/ids:           ["L0.R2", "L0.R3", ...]  # 28 string labels (fixed ordering)
```

### Fault Tolerance

For large-scale processing:
- Corrupted TFRecord files are skipped (not crash)
- Errors are logged with file names
- Processing continues with remaining files
- Summary reports skipped files at completion

---

## Output Formats

### TFRecord Augmentation

Produces augmented TFRecords with original + rule features.

**Best for:**
- Training ML models on Waymo data
- Maintaining compatibility with existing pipelines
- Large-scale processing

**Produced by:** `augment_cli.py`

### JSONL Generation

Produces JSON Lines with detailed evaluation results.

**Best for:**
- Analysis and debugging
- Human-readable evaluation reports
- Integration with pandas/data analysis tools

**Produced by:** `process_scenarios.py`

---

## TFRecord Augmentation

### Quick Start

```bash
cd /workspace/data/WOMD

# Augment a directory of TFRecords
python -m waymo_rule_eval.augmentation.augment_cli \
    --input /path/to/raw/scenario/validation_interactive \
    --output /path/to/augmented/scenario/
```

### How It Works

```
┌─────────────────────┐
│  Original TFRecord  │
│  (Waymo Scenario)   │
└─────────────────────┘
          │
          │  1. Read scenario
          ▼
┌─────────────────────┐
│  Parse scenario     │
│  → ScenarioContext  │
└─────────────────────┘
          │
          │  2. Evaluate rules
          ▼
┌─────────────────────┐
│  RuleExecutor       │
│  → ScenarioResult   │
└─────────────────────┘
          │
          │  3. Encode features
          ▼
┌─────────────────────┐
│  Original + Rules   │
│  Combined TFRecord  │
└─────────────────────┘
```

### CLI Reference

```bash
python -m waymo_rule_eval.augmentation.augment_cli [OPTIONS]
```

| Option | Required | Default | Description |
|--------|----------|---------|-------------|
| `--input` | Yes | - | Input directory containing TFRecord files |
| `--output` | Yes | - | Output directory for augmented files |

### Example: Full Dataset Augmentation

```bash
cd /workspace/data/WOMD

# Augment validation split
python -m waymo_rule_eval.augmentation.augment_cli \
    --input /data/waymo/motion_v_1_3_0/raw/scenario/validation_interactive \
    --output /data/waymo/motion_v_1_3_0/processed/augmented/scenario/validation_interactive

# Augment testing split
python -m waymo_rule_eval.augmentation.augment_cli \
    --input /data/waymo/motion_v_1_3_0/raw/scenario/testing_interactive \
    --output /data/waymo/motion_v_1_3_0/processed/augmented/scenario/testing_interactive

# Augment training split (largest, ~745 files)
python -m waymo_rule_eval.augmentation.augment_cli \
    --input /data/waymo/motion_v_1_3_0/raw/scenario/training_interactive \
    --output /data/waymo/motion_v_1_3_0/processed/augmented/scenario/training_interactive
```

---

## Command-Line Interface

### augment_cli.py

Main tool for TFRecord augmentation.

```bash
cd /workspace/data/WOMD

python -m waymo_rule_eval.augmentation.augment_cli \
    --input /path/to/input \
    --output /path/to/output
```

**Output:**
- One augmented TFRecord per input TFRecord
- Same filename as input
- `augmentation.log` file in output directory

### process_scenarios.py

Tool for JSONL generation with detailed results.

```bash
cd /workspace/data/WOMD

python -m waymo_rule_eval.augmentation.process_scenarios \
    --input "/path/to/*.tfrecord*" \
    --output results.jsonl \
    --window-size 1.0 \
    --stride 1.0
```

| Option | Default | Description |
|--------|---------|-------------|
| `--input` | (required) | Input glob pattern |
| `--output` | (required) | Output JSONL file |
| `--window-size` | 1.0 | Window size in seconds |
| `--stride` | 1.0 | Stride between windows |
| `--max-scenarios` | None | Limit scenarios processed |
| `--workers` | 1 | Parallel workers |

---

## Python API

### TFRecordAugmenter Class

Low-level class for augmenting individual TFRecord files.

```python
from waymo_rule_eval.augmentation.tfrecord_augmenter import TFRecordAugmenter

# Initialize augmenter (registers all rules automatically)
augmenter = TFRecordAugmenter()

# Process a single file
augmenter.augment_file(
    input_path="/path/to/input.tfrecord",
    output_path="/path/to/output.tfrecord"
)

# Or process with custom handling
from waymo_rule_eval.data_access.adapter_motion_scenario import MotionScenarioReader

reader = MotionScenarioReader()
for scenario in reader.read_tfrecord("input.tfrecord"):
    # Get rule evaluation as arrays
    features = augmenter._evaluate_scenario(scenario)

    # features contains:
    # - applicability: np.array of int64, shape (28,)
    # - violations: np.array of int64, shape (28,)
    # - severity: np.array of float32, shape (28,)
    # - rule_ids: list of str, length 28
```

### Processing Multiple Files

```python
import glob
from waymo_rule_eval.augmentation.tfrecord_augmenter import TFRecordAugmenter
from pathlib import Path

augmenter = TFRecordAugmenter()

input_dir = Path("/path/to/raw/scenario")
output_dir = Path("/path/to/augmented/scenario")
output_dir.mkdir(parents=True, exist_ok=True)

for input_path in glob.glob(str(input_dir / "*.tfrecord*")):
    output_path = output_dir / Path(input_path).name

    try:
        augmenter.augment_file(str(input_path), str(output_path))
        print(f"✓ Processed: {input_path}")
    except Exception as e:
        print(f"✗ Failed: {input_path}: {e}")
```

---

## Data Schema

### Augmented TFRecord Features

The augmenter adds these features to each TFRecord example:

| Feature | Type | Shape | Description |
|---------|------|-------|-------------|
| `scenario/proto` | bytes | () | Original serialized Scenario proto (preserved) |
| `scenario/id` | bytes | () | Scenario identifier |
| `rule/applicability` | int64 | (28,) | 1 if rule applies, 0 otherwise |
| `rule/violations` | int64 | (28,) | 1 if rule violated, 0 otherwise |
| `rule/severity` | float32 | (28,) | Severity score (0.0 = no violation; higher = worse; may exceed 1.0 for some rules) |
| `rule/ids` | bytes | (28,) | Rule identifier strings |

> **Note**: The schema defines 28 slots. All 28 rules are registered in `RuleRegistry`; some may have zero applicability depending on the scenario data.

### Rule Ordering

Rules are stored in a fixed order for consistent indexing:

| Index | Rule ID | Name |
|-------|---------|------|
| 0 | L0.R2 | Safe Longitudinal Distance |
| 1 | L0.R3 | Safe Lateral Clearance |
| 2 | L0.R4 | Crosswalk Occupancy |
| 3 | L1.R1 | Smooth Acceleration |
| 4 | L1.R2 | Smooth Braking |
| 5 | L1.R3 | Smooth Steering |
| 6 | L1.R4 | Speed Consistency |
| 7 | L1.R5 | Lane Change Smoothness |
| 8 | L3.R3 | Drivable Surface |
| 9 | L4.R3 | Left Turn Gap |
| 10 | L5.R1 | Traffic Signal Compliance |
| 11 | L5.R2 | Priority Violation |
| 12 | L5.R3 | Parking Violation |
| 13 | L5.R4 | School Zone Compliance |
| 14 | L5.R5 | Construction Zone Compliance |
| 15 | L6.R1 | Cooperative Lane Change |
| 16 | L6.R2 | Following Distance |
| 17 | L6.R3 | Intersection Negotiation |
| 18 | L6.R4 | Pedestrian Interaction |
| 19 | L6.R5 | Cyclist Interaction |
| 20 | L7.R3 | Lane Departure |
| 21 | L7.R4 | Speed Limit |
| 22 | L8.R1 | Red Light |
| 23 | L8.R2 | Stop Sign |
| 24 | L8.R3 | Crosswalk Yield |
| 25 | L8.R5 | Wrong-Way |
| 26 | L10.R1 | Collision |
| 27 | L10.R2 | VRU Clearance |

### Reading Augmented TFRecords

```python
import tensorflow as tf

# Define feature description
feature_description = {
    'scenario/proto': tf.io.FixedLenFeature([], tf.string),
    'scenario/id': tf.io.FixedLenFeature([], tf.string),
    'rule/applicability': tf.io.FixedLenFeature([28], tf.int64),
    'rule/violations': tf.io.FixedLenFeature([28], tf.int64),
    'rule/severity': tf.io.FixedLenFeature([28], tf.float32),
    'rule/ids': tf.io.FixedLenFeature([28], tf.string),
}

def parse_augmented_scenario(example_proto):
    return tf.io.parse_single_example(example_proto, feature_description)

# Load dataset
dataset = tf.data.TFRecordDataset("augmented.tfrecord")
dataset = dataset.map(parse_augmented_scenario)

for example in dataset.take(1):
    print(f"Scenario: {example['scenario/id'].numpy().decode()}")
    print(f"Violations: {example['rule/violations'].numpy()}")
    print(f"Severity: {example['rule/severity'].numpy()}")
```

---

## Processing Large Datasets

### Expected Processing Times

| Dataset | Files | Estimated Time |
|---------|-------|----------------|
| validation_interactive | 150 | ~15 minutes |
| testing_interactive | 150 | ~15 minutes |
| training_interactive | 745 | ~60-90 minutes |

### Running in Background

```bash
# Use nohup for long-running jobs
nohup python -m waymo_rule_eval.augmentation.augment_cli \
    --input /path/to/input \
    --output /path/to/output \
    > augmentation.log 2>&1 &

# Monitor progress
tail -f augmentation.log
```

### Handling Corrupted Files

The CLI automatically handles corrupted TFRecords:

```
Found 745 input files
Processing files: 100%|██████████| 745/745 [40:23<00:00, 3.25s/it]

Completed!
Total scenarios: 92679
Skipped 9 corrupted files: [
    'training.tfrecord-00175-of-01000',
    'training.tfrecord-00185-of-01000',
    ...
]
```

Skipped files are logged but don't stop processing.

---

## Integration with ML Pipelines

### TensorFlow Data Pipeline

```python
import tensorflow as tf

# Feature description for augmented data
FEATURE_DESC = {
    'scenario/proto': tf.io.FixedLenFeature([], tf.string),
    'rule/applicability': tf.io.FixedLenFeature([28], tf.int64),
    'rule/violations': tf.io.FixedLenFeature([28], tf.int64),
    'rule/severity': tf.io.FixedLenFeature([28], tf.float32),
}

def parse_fn(example_proto):
    features = tf.io.parse_single_example(example_proto, FEATURE_DESC)

    # Extract features for training
    rule_features = tf.concat([
        tf.cast(features['rule/applicability'], tf.float32),
        tf.cast(features['rule/violations'], tf.float32),
        features['rule/severity']
    ], axis=0)  # Shape: (84,) = 28 * 3

    return features['scenario/proto'], rule_features

# Create dataset
files = tf.io.gfile.glob("/path/to/augmented/*.tfrecord*")
dataset = tf.data.TFRecordDataset(files)
dataset = dataset.map(parse_fn)
dataset = dataset.batch(32).prefetch(tf.data.AUTOTUNE)
```

### PyTorch DataLoader

```python
import torch
from torch.utils.data import Dataset, DataLoader
import tensorflow as tf
import numpy as np

class AugmentedWaymoDataset(Dataset):
    def __init__(self, tfrecord_path):
        # Parse TFRecord into memory (for small datasets)
        self.examples = []

        dataset = tf.data.TFRecordDataset(tfrecord_path)
        for raw_record in dataset:
            example = tf.train.Example()
            example.ParseFromString(raw_record.numpy())

            self.examples.append({
                'violations': np.array(
                    example.features.feature['rule/violations'].int64_list.value
                ),
                'severity': np.array(
                    example.features.feature['rule/severity'].float_list.value
                )
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        return {
            'violations': torch.tensor(ex['violations'], dtype=torch.float32),
            'severity': torch.tensor(ex['severity'], dtype=torch.float32)
        }

# Use with DataLoader
dataset = AugmentedWaymoDataset("augmented.tfrecord")
loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

### Violation Statistics

```python
import tensorflow as tf
import numpy as np

def compute_violation_stats(tfrecord_pattern):
    """Compute rule violation statistics across dataset."""

    feature_desc = {
        'rule/applicability': tf.io.FixedLenFeature([28], tf.int64),
        'rule/violations': tf.io.FixedLenFeature([28], tf.int64),
    }

    applicability_counts = np.zeros(28)
    violation_counts = np.zeros(28)
    total_scenarios = 0

    files = tf.io.gfile.glob(tfrecord_pattern)
    for filepath in files:
        dataset = tf.data.TFRecordDataset(filepath)
        for record in dataset:
            example = tf.io.parse_single_example(record, feature_desc)

            applicability_counts += example['rule/applicability'].numpy()
            violation_counts += example['rule/violations'].numpy()
            total_scenarios += 1

    # Compute rates
    applicability_rate = applicability_counts / total_scenarios
    violation_rate = np.divide(
        violation_counts,
        applicability_counts,
        where=applicability_counts > 0
    )

    return {
        'total_scenarios': total_scenarios,
        'applicability_rate': applicability_rate,
        'violation_rate': violation_rate
    }

stats = compute_violation_stats("/path/to/augmented/*.tfrecord*")
print(f"Total scenarios: {stats['total_scenarios']}")
print(f"Mean applicability: {np.mean(stats['applicability_rate']):.2%}")
print(f"Mean violation rate: {np.mean(stats['violation_rate']):.2%}")
```

---

## Troubleshooting

### Common Issues

**1. "No module named 'waymo_rule_eval'"**

Run from the correct directory:
```bash
cd /workspace/data
python -m waymo_rule_eval.augmentation.augment_cli ...
```

**2. "DataLossError: truncated record"**

Some source TFRecords may be corrupted. The CLI handles this automatically by skipping corrupted files.

**3. All scenarios show 0 violations (all rule arrays are zeros)**

This was caused by a bug in `augment_cli.py` where `register_all_rules()` was not called after creating the `RuleExecutor`. The fix (applied 2025-01) adds the missing call:

```python
# BEFORE (buggy):
executor = RuleExecutor()
# No rules registered → 0 results for all scenarios

# AFTER (fixed):
executor = RuleExecutor()
executor.register_all_rules()  # Now properly registers all 28 rules
```

**Impact**: Any datasets augmented before the fix will have all-zero rule evaluation arrays. To fix affected data:

```bash
# Re-run augmentation on affected splits
# First, delete the corrupted augmented files
rm -rf /path/to/augmented/scenario/testing_interactive/
rm -rf /path/to/augmented/scenario/training_interactive/

# Re-run augmentation with fixed code
python -m waymo_rule_eval.augmentation.augment_cli \
    --input /path/to/raw/scenario/testing_interactive/ \
    --output /path/to/augmented/scenario/testing_interactive/

python -m waymo_rule_eval.augmentation.augment_cli \
    --input /path/to/raw/scenario/training_interactive/ \
    --output /path/to/augmented/scenario/training_interactive/
```

**4. Memory issues with large files**

Process files individually or in smaller batches:
```bash
# Process one file at a time
for f in /path/to/input/*.tfrecord*; do
    python -m waymo_rule_eval.augmentation.augment_cli \
        --input "$f" \
        --output /path/to/output/
done
```

**4. Slow processing**

- Check disk I/O (TFRecord reading is I/O bound)
- Use SSD if available
- Consider running multiple instances for different file ranges

### Logging

The CLI produces an `augmentation.log` file with:
- Start/end timestamps
- Files processed
- Scenarios per file
- Errors encountered
- Skipped files summary

---

## Files in This Module

| File | Purpose |
|------|---------|
| `augment_cli.py` | Main CLI for TFRecord augmentation |
| `tfrecord_augmenter.py` | Core augmentation logic |
| `process_scenarios.py` | JSONL generation tool |
| `run_augmentation.sh` | Batch processing shell script |
| `comprehensive_test.py` | Testing and validation script |

---

## See Also

- [`../pipeline/README.md`](../pipeline/README.md) - Rule execution details
- [`../rules/README.md`](../rules/README.md) - Complete rule catalog
- [`../data_access/README.md`](../data_access/README.md) - Data loading
