# Data Processing Tests

This directory contains test scripts for validating data preprocessing and loading pipelines.

## Test Files

### `test_preprocessing.py`
Comprehensive test suite for Waymo data preprocessing and loading.

**Usage:**
```bash
# Test with existing preprocessed data
python data/tests/test_preprocessing.py \
    --data-dir data/datasets/waymo_open_dataset/motion_v_1_3_0/processed/training_interactive \
    --num-samples 5

# Run end-to-end test (preprocess + load)
python data/tests/test_preprocessing.py \
    --test-preprocessing \
    --input-dir data/datasets/waymo_open_dataset/motion_v_1_3_0/raw/scenario/training \
    --output-dir /tmp/test_preprocessed \
    --max-files 2

# Test batching
python data/tests/test_preprocessing.py \
    --data-dir data/datasets/waymo_open_dataset/motion_v_1_3_0/processed/training_interactive \
    --test-batching \
    --batch-size 8
```

**What it tests:**
- ✅ Preprocessing script correctness
- ✅ Data loader can read preprocessed files
- ✅ Batch shapes and dimensions are correct
- ✅ Data value ranges are reasonable
- ✅ Map feature encoding works properly
- ✅ Augmentation functionality
- ✅ End-to-end pipeline integrity

**Test output:**
```
TEST 1: Preprocessing
  ✓ Preprocessed 15 scenarios
  
TEST 2: Data Loading
  ✓ Loaded dataset with 15 samples
  ✓ Sample shapes are correct
  ✓ Data ranges are valid
  
TEST 3: Batching
  ✓ Batch shapes are correct
  ✓ Collate function works
  
TEST 4: Augmentation
  ✓ Augmentation applied correctly
  
All tests passed!
```

### `test_tfexample_loading.py`
Test suite for direct TFExample (TFRecord) loading without preprocessing.

**✅ TESTED AND PASSING (November 14, 2025)**

**Usage:**
```bash
# Test all interactive splits (default)
python data/tests/test_tfexample_loading.py

# Or run directly
cd /workspace
python data/tests/test_tfexample_loading.py
```

**What it tests:**
- ✅ TFRecord file discovery and loading
- ✅ Feature parsing from tf_example format  
- ✅ Agent trajectory extraction (10 past + 1 current + 80 future = 91 frames)
- ✅ Road graph extraction (30,000 points)
- ✅ Interactive pair detection (proximity-based)
- ✅ Dataloader batch collation
- ✅ All interactive splits (testing/training/validation)

**Test Results (November 14, 2025):**
```
Testing split: testing_interactive
  Found 15 TFRecord files
  Loaded 10 scenarios
  Valid agents: 8/128
  Interactive pairs: 4
  ✅ PASSED

Testing split: training_interactive
  Found 15 TFRecord files
  Loaded 10 scenarios
  Valid agents: 23/128
  Interactive pairs: 208
  ✅ PASSED

Testing split: validation_interactive
  Found 15 TFRecord files
  Loaded 10 scenarios
  Valid agents: 28/128
  Interactive pairs: 127
  ✅ PASSED

🎉 ALL TESTS PASSED
```

**Data Format Verified:**
- Agent trajectories: [128, 211, 2] - 211 frames total
- Agent validity: [128, 211] - validity masks
- Road graph: [20000, 3] - xyz coordinates
- Interactive pairs: [K, 2] - automatically detected
- Velocities, headings, types: All present

**Key Finding:** TF_example format contains COMPLETE trajectory data including:
- 10 past frames (state/past/*)
- 1 current frame (state/current/*)
- 80 future frames (state/future/*)
- Total: 91 frames at 10Hz = 9.1 seconds

This contradicts earlier assumptions that tf_example lacked history data.

### `test_data_loading.py`
Legacy test for older data loading format.

## Test Files (To be added)

When adding new data processing functionality:

1. Create test file in this directory
2. Follow naming convention: `test_*.py`
3. Use pytest or unittest framework
4. Include docstrings explaining what is being tested
5. Update this README

## Running All Tests

```bash
# Run all tests in this directory
python -m pytest data/tests/ -v

# Or using unittest
python -m unittest discover data/tests/
```

## Test Data

- Test data should use a small subset of the full dataset
- Consider using `--max-files 5` option for quick tests
- Temporary outputs should go to `/tmp/` or be cleaned up

## Requirements

```bash
pip install pytest numpy torch waymo-open-dataset-tf-2-11-0
```
