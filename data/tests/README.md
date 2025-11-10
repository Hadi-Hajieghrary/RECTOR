# Data Processing Tests

This directory contains test scripts for validating data preprocessing and loading pipelines.

## Test Files

### `test_preprocessing.py`
Comprehensive test suite for Waymo data preprocessing and loading.

**Usage:**
```bash
# Test with existing preprocessed data
python data/tests/test_preprocessing.py \
    --data-dir data/processed/itp_training \
    --num-samples 5

# Run end-to-end test (preprocess + load)
python data/tests/test_preprocessing.py \
    --test-preprocessing \
    --input-dir data/waymo_open_dataset_motion_v_1_3_0/scenario/training \
    --output-dir /tmp/test_preprocessed \
    --max-files 2

# Test batching
python data/tests/test_preprocessing.py \
    --data-dir data/processed/itp_training \
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
