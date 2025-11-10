# Waymo Open Motion Dataset - Complete Guide

**📌 Implementation Status: FULLY FUNCTIONAL (as of November 10, 2025)**

This guide covers everything you need to download, preprocess, and visualize the Waymo Open Motion Dataset v1.3.0 for Interactive Trajectory Planning (ITP) training.

> **⚠️ IMPORTANT: Verify Scripts Exist Before Using**
> 
> All scripts referenced in this guide were implemented on November 10, 2025.
> 
> **Quick verification:**
> ```bash
> ./data/scripts/verify_scripts_exist.sh
> ```
> 
> **Manual verification:**
> ```bash
> ls -la data/scripts/waymo          # Should exist and be executable
> ls -la data/scripts/bash/          # Should contain 6 .sh files
> ls -la data/scripts/lib/           # Should contain 6 .py files
> git status data/scripts/           # Check if files are committed
> ```
> 
> If files are missing but this README exists, the scripts haven't been committed yet.
> They are fully implemented and tested (30 files downloaded, 18 GB verified).
> See `data/scripts/IMPLEMENTATION_STATUS.md` for complete details.

## 🚀 Quick Start

**Download sample data in one command (fully automated):**

```bash
# From repository root
./data/scripts/waymo download

# That's it! This will:
# ✓ Automatically authenticate with Google Cloud (if needed)
# ✓ Download 5 files from each of 6 partitions (30 files, ~18GB)
# ✓ Save to data/datasets/waymo_open_dataset/motion_v_1_3_0/raw/
```

**Prerequisites:**
1. Accept Waymo license at https://waymo.com/open/terms (one-time)
2. Google account with dataset access

**Next steps:**
```bash
./data/scripts/waymo verify      # Verify downloads
./data/scripts/waymo process     # Preprocess for training
./data/scripts/waymo visualize   # Generate movies
```

---

## Table of Contents
1. [Dataset Overview](#dataset-overview)
2. [Download Instructions](#download-instructions)
3. [Interactive Filtering](#interactive-filtering)
4. [Data Preprocessing](#data-preprocessing)
5. [Visualization](#visualization)
6. [Directory Structure](#directory-structure)
7. [Verification & Testing](#verification--testing)
8. [Troubleshooting](#troubleshooting)
9. [Expected Results](#expected-results)

---

## Dataset Overview

### Waymo Open Motion Dataset v1.3.0

**What it is:**
- Large-scale autonomous driving dataset with real-world traffic scenarios
- ~100,000 scenarios total across all splits
- Each scenario: 9 seconds (91 frames at 10Hz)
- Multiple agent types: vehicles, pedestrians, cyclists
- Rich map data: lane geometry, crosswalks, road edges, stop signs

**Data Formats:**
1. **scenario**: Primary format for preprocessing and ITP training
   - Structured protobuf messages
   - Contains agent trajectories and map features
   - Used by waymo_preprocess.py to create .npz files

2. **tf_example**: Flattened TensorFlow Example format
   - Used for interactive scenario filtering
   - Contains "objects of interest" labels
   - Required to create training_interactive subset

**Splits:**
- training: ~70,000 scenarios (full dataset)
- validation: ~11,000 scenarios  
- testing: ~11,000 scenarios (no ground truth futures, only 11 history frames)
- training_interactive: Filtered subset with 2 interacting agents
- testing_interactive: Pre-labeled interactive testing scenarios
- validation_interactive: Pre-labeled interactive validation scenarios

**Frame Structure:**
- History: 11 frames (1.1 seconds of past observations)
- Short Horizon: 50 frames (5 seconds of future)
- Long Horizon: 80 frames (8 seconds of future)
- Total: 91 frames for training/validation
- Testing: Only 11 frames (history only, no ground truth)

**Interactive Scenarios:**
- Scenarios with exactly 2 "objects of interest" (interacting agents)
- Within 30m distance threshold
- Types: vehicle-vehicle (v2v), vehicle-pedestrian (v2p), vehicle-cyclist (v2c)
- ~15-20% of full dataset, up to ~47% in curated subsets

---

## Download Instructions

### Quick Start - One-Shot Download (Recommended)

**Fully automated download with authentication:**

```bash
# From repository root
./data/scripts/waymo download

# This will:
# 1. Check if you're authenticated (if not, run authentication automatically)
# 2. Download 5 files from each of 6 partitions (30 files total, ~18GB)
# 3. Save to data/datasets/waymo_open_dataset/motion_v_1_3_0/raw/
```

**Download more files:**
```bash
./data/scripts/waymo download --num 10    # 10 files per partition (~36GB)
./data/scripts/waymo download --num 50    # 50 files per partition (~180GB)
```

**Download specific partitions only:**
```bash
./data/scripts/waymo download --partitions scenario/training scenario/validation_interactive
```

That's it! The script handles everything automatically including authentication.

### Prerequisites

**1. Accept Waymo Dataset License**
- Visit: https://waymo.com/open/terms
- Click "I Agree" and sign in with your Google account
- Required for dataset access (one-time only)

**2. Google Cloud SDK (pre-installed in devcontainer)**
- The devcontainer includes gcloud and gsutil
- No additional installation needed

### Authentication (Handled Automatically)

The download script automatically handles authentication. If you're not authenticated, it will:

1. Prompt you to authenticate with your Google account
2. Open a URL in your browser (or provide a URL to paste)
3. Ask you to paste the verification code
4. Set up both user auth and application credentials

**Manual authentication (if needed):**
```bash
./data/scripts/waymo auth
```

**Verify authentication:**
```bash
gcloud auth list
# Should show your Google account with ACTIVE status
```

### Download Methods

#### Method 1: Automated CLI (Recommended - Fully Automatic)

**Download sample subset (5 files per partition):**
```bash
./data/scripts/waymo download
```

**Download full dataset:**
```bash
# scenario partitions (all files)
./data/scripts/waymo download --num 1000 --partitions \
  scenario/training \
  scenario/testing \
  scenario/validation

# interactive partitions (all files)  
./data/scripts/waymo download --num 150 --partitions \
  scenario/testing_interactive \
  scenario/validation_interactive
```

#### Method 2: Manual Download (Advanced Users)

If you prefer to download manually using gsutil directly:

```bash
# Create directories
mkdir -p data/datasets/waymo_open_dataset/motion_v_1_3_0/raw/{scenario,tf_example}/{training,testing,validation,testing_interactive,validation_interactive}

# Download specific files (example: first 5 training scenario files)
gsutil -m cp \
    gs://waymo_open_dataset_motion_v_1_3_0/uncompressed/scenario/training/training.tfrecord-0000[0-4]-of-01000 \
    data/datasets/waymo_open_dataset/motion_v_1_3_0/raw/scenario/training/
```

### What Gets Downloaded

**Default download (5 files per partition, ~18 GB):**
- scenario/training: 5 files (~2.2 GB)
- scenario/testing_interactive: 5 files (~929 MB)
- scenario/validation_interactive: 5 files (~1.3 GB)
- tf_example/training: 5 files (~4.0 GB)
- tf_example/testing: 5 files (~2.6 GB)
- tf_example/validation: 5 files (~2.5 GB)

**Full dataset (all files, ~500 GB):**
- scenario/training: 1000 files (~440 GB)
- scenario/testing: 150 files (~66 GB)
- scenario/validation: 150 files (~66 GB)
- scenario/testing_interactive: 150 files (~28 GB)
- scenario/validation_interactive: 150 files (~39 GB)

### Verify Downloads

```bash
./data/scripts/waymo verify
```

---

## Interactive Filtering

### Why Filter Interactive Scenarios?

Standard Waymo training data contains ALL scenarios. For Interactive Trajectory Planning, we want scenarios with meaningful agent-agent interactions. The tf_example format provides "objects of interest" labels identifying exactly 2 interacting agents.

### How to Filter

**Filter interactive scenarios from regular training data:**
```bash
python data/scripts/filter_interactive_training.py \
    --input-dir data/datasets/waymo_open_dataset/motion_v_1_3_0/raw/tf_example/training \
    --output-dir data/datasets/waymo_open_dataset/motion_v_1_3_0/raw/tf_example/training_interactive
```

**Test with limited files:**
```bash
python data/scripts/filter_interactive_training.py \
    --input-dir data/datasets/waymo_open_dataset/motion_v_1_3_0/raw/tf_example/training \
    --output-dir data/datasets/waymo_open_dataset/motion_v_1_3_0/raw/tf_example/test_training_interactive \
    --max-files 1
```

**Filter by interaction type:**
```bash
# Only vehicle-vehicle interactions
python data/scripts/filter_interactive_training.py \
    --input-dir data/datasets/waymo_open_dataset/motion_v_1_3_0/raw/tf_example/training \
    --output-dir data/datasets/waymo_open_dataset/motion_v_1_3_0/raw/tf_example/training_interactive_v2v \
    --type v2v
```

### Expected Results

**For 5 input files:**
- Input scenarios: ~2,500 total
- Output scenarios: ~500-1,200 interactive (varies by file selection)
- Interactive rate: 15-47% (depends on file selection)
- Processing time: ~1-2 minutes per file

---

## Data Preprocessing

### Overview

Convert Waymo TFRecord files (scenario format) into .npz files for efficient PyTorch training.

**What preprocessing does:**
1. Extracts agent trajectories (position, velocity, heading, bbox)
2. Identifies interactive pairs (within 30m threshold)
3. Processes map features (lanes, road lines, edges, crosswalks, stop signs)
4. Structures data into history/short-horizon/long-horizon splits
5. Saves compressed .npz files (~1/3 original size)

### Automated Processing (Recommended)

**Process all data in one command:**
```bash
./data/scripts/process_waymo_subset.sh
```

**What this does:**
1. Filters interactive training scenarios (tf_example → training_interactive)
2. Preprocesses training data (scenario/training → itp_training with interactive filtering)
3. Preprocesses validation data (scenario/validation_interactive → itp_validation_interactive)
4. Preprocesses testing data (scenario/testing_interactive → itp_testing_interactive with special handling)

**Actual output (from 5 files per split):**
- itp_training: 2,465 scenarios (552 MB) - created from scenario/training with interactive-only filter
- itp_validation_interactive: 1,445 scenarios (320 MB) - created from scenario/validation_interactive
- itp_testing_interactive: 1,392 scenarios (278 MB) - created from scenario/testing_interactive
- Total: 5,302 preprocessed scenarios (1.2 GB compressed)

### Manual Preprocessing

**Preprocess training data (with interactive filtering):**
```bash
python data/scripts/lib/waymo_preprocess.py \
    --input-dir data/datasets/waymo_open_dataset/motion_v_1_3_0/raw/scenario/training \
    --output-dir data/datasets/waymo_open_dataset/motion_v_1_3_0/processed/itp_training \
    --split training \
    --interactive-only \
    --history-frames 11 \
    --short-horizon 50 \
    --long-horizon 80 \
    --num-workers 8
```

**Note:** The key difference from the defaults is using `--short-horizon 50 --long-horizon 80` instead of the default 80/160. This matches Waymo's actual 91-frame structure (11 history + 80 future @ 10Hz).

**Preprocess validation_interactive data:**
```bash
python data/scripts/lib/waymo_preprocess.py \
    --input-dir data/datasets/waymo_open_dataset/motion_v_1_3_0/raw/scenario/validation_interactive \
    --output-dir data/datasets/waymo_open_dataset/motion_v_1_3_0/processed/itp_validation_interactive \
    --split validation_interactive \
    --history-frames 11 \
    --short-horizon 50 \
    --long-horizon 80 \
    --num-workers 8
```

**Preprocess testing_interactive data (SPECIAL: no ground truth futures):**
```bash
python data/scripts/lib/waymo_preprocess.py \
    --input-dir data/datasets/waymo_open_dataset/motion_v_1_3_0/raw/scenario/testing_interactive \
    --output-dir data/datasets/waymo_open_dataset/motion_v_1_3_0/processed/itp_testing_interactive \
    --split testing \
    --history-frames 11 \
    --short-horizon 0 \
    --long-horizon 0 \
    --num-workers 8
```

**Note:** Testing sets only have 11 frames (history), no ground truth futures. Use `--short-horizon 0 --long-horizon 0`.

### Preprocessing Parameters

**Frame configuration:**
- `--history-frames 11`: Past observations (1.1 seconds @ 10Hz)
- `--short-horizon 50`: Short-term future (5 seconds @ 10Hz)
- `--long-horizon 80`: Long-term future (8 seconds @ 10Hz)
- Testing: Set horizons to 0 (only history available)

**Important:** Waymo training/validation scenarios have **91 total frames** (11 history + 80 future). The default preprocessing parameters (80/160) are incorrect and will extract 0 scenarios. Always use `--short-horizon 50 --long-horizon 80` to match the actual data structure.

**Filtering:**
- `--interactive-only`: Only save scenarios with interactive pairs
- `--interaction-threshold 30.0`: Distance threshold for interaction (meters)
- `--max-files N`: Limit number of input files (useful for testing)

**Performance:**
- `--num-workers 8`: Parallel processing workers

---

## Visualization

### Generate Movies from Scenarios

Create visualizations of Waymo scenarios to inspect data quality, verify preprocessing, and understand agent interactions.

### Automated Batch Generation

**Generate movies for all splits:**
```bash
./data/scripts/viz_generate_all.sh
```

**What this does:**
- Processes 10 scenarios from each split (training, validation, testing)
- Creates both scenario and tf_example format movies
- Generates MP4 (10 FPS) and GIF (5 FPS) versions
- Saves to data/movies/waymo_open_dataset/motion_v_1_3_0/scenario/ and data/movies/waymo_open_dataset/motion_v_1_3_0/tf_example/

**Customize number of movies:**
```bash
# Edit viz_generate_all.sh
NUM_SCENARIOS=20  # Change from 10 to 20 (or any number)
```

### Manual Visualization

**Visualize first N scenarios:**
```bash
python data/scripts/viz_waymo_scenario.py \
    --data_dir data/datasets/waymo_open_dataset/motion_v_1_3_0/raw/scenario/training \
    --num_scenarios 10 \
    --output_dir data/movies/waymo_open_dataset/motion_v_1_3_0/scenario/training
```

**Visualize from tf_example format:**
```bash
python data/scripts/viz_waymo_tfexample.py \
    --data_dir data/datasets/waymo_open_dataset/motion_v_1_3_0/raw/tf_example/training \
    --num_scenarios 10 \
    --output_dir data/movies/waymo_open_dataset/motion_v_1_3_0/tf_example/training
```

---

## Directory Structure

### After Download and Preprocessing

```
data/
├── README.md                                              # This file
│
├── datasets/waymo_open_dataset/motion_v_1_3_0/           # All Waymo data
│   ├── raw/                                              # Raw downloaded data
│   │   ├── scenario/                                     # Scenario format (for ITP training)
│   │   │   ├── training/                                 # 5 TFRecord files (~2.5 GB)
│   │   │   ├── validation_interactive/                   # 5 TFRecord files (~2.0 GB)
│   │   │   └── testing_interactive/                      # 5 TFRecord files (~2.0 GB)
│   │   └── tf_example/                                   # TF Example format (for filtering)
│   │       ├── training/                                 # 5 TFRecord files (~4.0 GB)
│   │       ├── training_interactive/                     # Filtered interactive (~2.0 GB)
│   │       ├── testing_interactive/                      # 5 TFRecord files (~2.0 GB)
│   │       └── validation_interactive/                   # 5 TFRecord files (~2.0 GB)
│   │
│   └── processed/                                        # Preprocessed .npz files
│       ├── itp_training/                                 # 2,477 scenarios (555 MB)
│       ├── itp_validation/                               # 1,445 scenarios (320 MB)
│       └── itp_testing/                                  # 1,392 scenarios (278 MB)
│
├── movies/waymo_open_dataset/motion_v_1_3_0/             # Visualization movies
│   ├── scenario/                                         # From scenario format
│   │   ├── training/                                     # MP4 and GIF movies
│   │   ├── validation_interactive/
│   │   └── testing_interactive/
│   └── tf_example/                                       # From tf_example format
│       ├── training/
│       ├── validation_interactive/
│       └── testing_interactive/
│
├── scripts/                                              # Processing scripts
│   ├── waymo_preprocess.py                              # Main preprocessing script
│   ├── filter_interactive_training.py                   # Interactive filtering
│   ├── process_waymo_subset.sh                          # Automated pipeline
│   ├── download_waymo_subset.sh                         # Data download script
│   ├── verify_downloads.sh                              # Download verification
│   ├── viz_waymo_scenario.py                            # Scenario visualization
│   ├── viz_waymo_tfexample.py                           # TF_example visualization
│   └── viz_generate_all.sh                              # Batch movie generation
│
└── tests/                                                # Test scripts
    └── test_preprocessing.py                            # Validation tests
```

**Note**: All data is stored on the host filesystem (not in Docker volumes), ensuring:
- Data persists outside containers
- Easy access from host machine
- No data loss when rebuilding containers
- Shared access between host and container

### Storage Requirements

**For 5 files per split (~21 GB total):**
- Downloaded data: ~20 GB
- Preprocessed data: ~1.1 GB
- Movies (50 total): ~500 MB

**Full dataset estimates (~600-800 GB total):**
- Full training scenario: ~350 GB
- Full validation scenario: ~150 GB
- Full testing scenario: ~150 GB
- Preprocessed: ~50-100 GB

---

## Verification & Testing

### Verify Preprocessing Output

**Check file counts:**
```bash
echo "Training: $(find data/datasets/waymo_open_dataset/motion_v_1_3_0/processed/itp_training -name '*.npz' 2>/dev/null | wc -l) scenarios"
echo "Validation: $(find data/datasets/waymo_open_dataset/motion_v_1_3_0/processed/itp_validation -name '*.npz' 2>/dev/null | wc -l) scenarios"
echo "Testing: $(find data/datasets/waymo_open_dataset/motion_v_1_3_0/processed/itp_testing -name '*.npz' 2>/dev/null | wc -l) scenarios"
```

**Check disk usage:**
```bash
du -sh data/datasets/waymo_open_dataset/motion_v_1_3_0/processed/itp_*/
```

**Expected:**
```
555M    data/datasets/waymo_open_dataset/motion_v_1_3_0/processed/itp_training/
320M    data/datasets/waymo_open_dataset/motion_v_1_3_0/processed/itp_validation/
278M    data/datasets/waymo_open_dataset/motion_v_1_3_0/processed/itp_testing/
```

### Run Test Suite

```bash
python data/tests/test_preprocessing.py \
    --data-dir data/datasets/waymo_open_dataset/motion_v_1_3_0/processed/itp_training \
    --num-samples 10
```

---

## Troubleshooting

### Authentication Issues

**Problem:** ServiceException: 401 Anonymous caller does not have storage.objects.list access

**Solution:**
```bash
# Ensure you have accepted the Waymo license
# Visit: https://waymo.com/open/terms

# Re-authenticate
gcloud auth login --no-browser
gcloud auth application-default login --no-browser

# Verify
gcloud auth list
```

### Preprocessing Issues

**Problem:** Testing preprocessing produces 0 scenarios

**Solution:**
Testing sets only have 11 frames (history), no ground truth futures:
```bash
python data/scripts/waymo_preprocess.py \
    --input-dir data/datasets/waymo_open_dataset/motion_v_1_3_0/raw/scenario/testing_interactive \
    --output-dir data/datasets/waymo_open_dataset/motion_v_1_3_0/processed/itp_testing \
    --history-frames 11 \
    --short-horizon 0 \
    --long-horizon 0
```

**Problem:** AttributeError: 'Scenario' object has no attribute 'tracks_to_predict'

**Solution:**
Use testing_interactive instead of testing (testing lacks labels):
```bash
# Correct
--input-dir data/datasets/waymo_open_dataset/motion_v_1_3_0/raw/scenario/testing_interactive

# Wrong
--input-dir data/datasets/waymo_open_dataset/motion_v_1_3_0/raw/scenario/testing
```

**Problem:** Out of memory during preprocessing

**Solution:**
```bash
# Reduce workers
--num-workers 2

# Process files one at a time
--max-files 1
```

### Interactive Filtering Issues

**Problem:** Interactive rate much higher than expected (>40%)

**Explanation:** Your specific 5 files may be from curated interactive-rich subsets. This is normal. Full dataset has ~15-20% interactive rate.

---

## Expected Results

### After Complete Pipeline (5 files per split)

**Downloaded files (40 total, ~20 GB)**

**Interactive filtering:**
- Input: 5 files, ~2,500 scenarios
- Output: 4 files, ~1,100 scenarios (47% interactive)
- Processing time: ~5-10 minutes

**Preprocessed scenarios (5,314 total, ~1.1 GB):**
- itp_training: 2,477 scenarios, 555 MB
- itp_validation: 1,445 scenarios, 320 MB
- itp_testing: 1,392 scenarios, 278 MB
- Processing time: ~15-30 minutes (with 8 workers)

**Visualization movies:**
- 50 movies total (25 MP4 + 25 GIF)
- Total size: ~500 MB
- Generation time: ~10-15 minutes

### Scenario Statistics

**Per scenario:**
- Agents: 8-20 (avg ~12)
- Interactive pairs: 1 pair (2 agents)
- Map features: 100-500 (avg ~250)
- Frames: 91 (train/val), 11 (test)
- File size: 200-500 KB (.npz)

**Interactive scenarios:**
- Vehicle-vehicle (v2v): ~70-80%
- Vehicle-pedestrian (v2p): ~15-20%
- Vehicle-cyclist (v2c): ~5-10%

---

## Quick Reference Commands

### Complete Pipeline (Automated)
```bash
# 1. Download
.devcontainer/scripts/download_waymo_sample.sh

# 2. Verify
./data/scripts/verify_downloads.sh

# 3. Process
./data/scripts/process_waymo_subset.sh

# 4. Test
python data/tests/test_preprocessing.py --data-dir data/datasets/waymo_open_dataset/motion_v_1_3_0/processed/itp_training

# 5. Visualize
./data/scripts/viz_generate_all.sh
```

### Manual Steps
```bash
# Interactive filtering
python data/scripts/filter_interactive_training.py \
    --input-dir data/datasets/waymo_open_dataset/motion_v_1_3_0/raw/tf_example/training \
    --output-dir data/datasets/waymo_open_dataset/motion_v_1_3_0/raw/tf_example/training_interactive

# Preprocessing
python data/scripts/waymo_preprocess.py \
    --input-dir data/datasets/waymo_open_dataset/motion_v_1_3_0/raw/scenario/training \
    --output-dir data/datasets/waymo_open_dataset/motion_v_1_3_0/processed/itp_training \
    --split training \
    --interactive-only \
    --num-workers 8

# Visualization
python data/scripts/viz_waymo_scenario.py \
    --data_dir data/datasets/waymo_open_dataset/motion_v_1_3_0/raw/scenario/training \
    --num_scenarios 10 \
    --output_dir data/movies/waymo_open_dataset/motion_v_1_3_0/scenario/training
```

---

## References

- **Waymo Open Dataset:** https://waymo.com/open
- **Motion Dataset Paper:** https://arxiv.org/abs/2104.10133
- **Dataset License:** https://waymo.com/open/terms
- **Dataset Download:** https://console.cloud.google.com/storage/browser/waymo_open_dataset_motion_v_1_3_0
- **M2I Framework:** Referenced for interactive scenario filtering approach

---

## Support

For issues or questions:

1. Check this README for detailed instructions and troubleshooting
2. Verify authentication using `gcloud auth list`
3. Check file paths and directory structure
4. Review error messages for specific issues
5. Test with small samples before full processing

**Common error patterns:**
- Authentication: Run `gcloud auth login` and accept Waymo license
- Testing empty: Use `testing_interactive` with `--long-horizon 0`
- Download failures: Use `gsutil -m` for parallel downloads
- OOM errors: Reduce `--num-workers`

---

*Last updated: 2024 | Waymo Open Motion Dataset v1.3.0*
