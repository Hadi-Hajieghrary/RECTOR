# Waymo Data Processing Scripts

Simple, intuitive tools for managing the Waymo Open Motion Dataset v1.3.0: download, verify, filter, preprocess, and visualize.

This folder contains a small CLI plus modular bash and Python scripts.

## Quick start

```bash
# From repository root
./data/scripts/waymo pipeline        # Download → verify → process → visualize

# Or run steps individually
./data/scripts/waymo download        # Download sample data (default 5 files/partition)
./data/scripts/waymo verify          # Check downloads
./data/scripts/waymo filter          # Create training_interactive from tf_example/training
./data/scripts/waymo process         # Preprocess TFRecords into .npz for training
./data/scripts/waymo visualize       # Generate movies
./data/scripts/waymo status          # Show dataset status and counts
```

Tip: If you see a “permission denied” when running the CLI, make it executable once:

```bash
chmod +x ./data/scripts/waymo
```

## Layout

```
data/scripts/
├── waymo              # Main CLI (entry point)
├── bash/              # Modular task scripts
│   ├── download.sh    # Download from GCS (uses gsutil)
│   ├── verify.sh      # Verify downloads
│   ├── filter.sh      # Filter interactive scenarios
│   ├── process.sh     # Preprocess data
│   └── visualize.sh   # Generate movies
└── lib/               # Python modules
    ├── waymo_preprocess.py
    ├── filter_interactive_training.py
    ├── viz_waymo_scenario.py
    ├── viz_waymo_tfexample.py
    ├── viz_trajectory.py
    └── waymo_dataset.py
```

## Commands

### Download data

```bash
./data/scripts/waymo download [--num N] [--partitions p1 p2 ...]
# Examples
./data/scripts/waymo download                # default 5 files per partition
./data/scripts/waymo download --num 10
./data/scripts/waymo download --partitions scenario/training tf_example/validation_interactive
```

Requires gsutil + license acceptance. See `References/WAYMO_DOWNLOAD_INSTRUCTIONS.md`.

### Verify downloads

```bash
./data/scripts/waymo verify [--quiet]
```

### Filter interactive scenarios

```bash
./data/scripts/waymo filter [--type v2v|v2p|v2c]
```

### Preprocess data

Converts Waymo TFRecords (scenario format) into compressed `.npz` for ITP training.

```bash
./data/scripts/waymo process [--split SPLIT] [--format scenario|tf_example] [--workers N] [--interactive-only]

# Examples
./data/scripts/waymo process                               # training split, 8 workers
./data/scripts/waymo process --split validation_interactive
./data/scripts/waymo process --workers 16 --interactive-only
```

Under the hood this runs `lib/waymo_preprocess.py` via `bash/process.sh`.

### Visualize data

```bash
./data/scripts/waymo visualize [--format scenario|tf_example] [--split SPLIT] [--num N] [--fps FPS]
```

### Status summary

```bash
./data/scripts/waymo status
```

Prints counts for raw TFRecords, processed `.npz`, and generated movies.

### Clean outputs

```bash
./data/scripts/waymo clean [processed|movies|all]
```

### End-to-end pipeline

```bash
./data/scripts/waymo pipeline [--num N] [--workers N] [--viz N]
```

Runs: download → verify → preprocess training → visualize sample movies.

## Python entry points

You can also call the Python scripts directly if preferred:

```bash
python data/scripts/lib/waymo_preprocess.py --help
python data/scripts/lib/viz_waymo_scenario.py --help
python data/scripts/lib/viz_waymo_tfexample.py --help
```

Typical training dataloader usage:

```python
from data.scripts.lib.waymo_dataset import WaymoITPDataset, build_dataloader
```

Note: If importing from outside the repo root, ensure `PYTHONPATH` includes the repository root (so `data.scripts.lib` resolves).

## Requirements

- Google Cloud SDK with gsutil (for downloads)
- Python 3.10+ with numpy, tensorflow (CPU ok), torch, matplotlib, tqdm
- Waymo Open Dataset protos (installed via waymo-open-dataset pip package)

See `References/requirements.base.txt` and `externals/waymo-open-dataset/README.md`.

## Troubleshooting

- Authentication errors: re-run `gcloud auth login` and accept Waymo license
- Empty processed set: ensure you pointed to `scenario/*` (not lidar/camera) and used interactive splits for testing
- OOM or slow: reduce `--workers`, use `--max-files` inside `waymo_preprocess.py`

## Notes

- This CLI is a thin wrapper over the bash scripts to make commands easy to remember and discover via `--help`.
- Raw data is expected under `data/datasets/waymo_open_dataset/motion_v_1_3_0/raw`.
- Processed data is written under `data/datasets/waymo_open_dataset/motion_v_1_3_0/processed`.

## Quick Reference (copy/paste)

The following is a curated list of common commands you can copy and run. They assume your working directory is `data/scripts/` (use `./waymo ...`). If you're at repo root, prefix with `./data/scripts/`.

```bash
# ============================================
# BASIC WORKFLOW
# ============================================

# Complete pipeline (download → verify → process → visualize)
./waymo pipeline

# Check status
./waymo status

# ============================================
# DOWNLOAD
# ============================================

# Download 5 files per partition (default)
./waymo download

# Download more files
./waymo download --num 10
./waymo download --num 20

# Download specific partitions
./waymo download --partitions scenario/training
./waymo download --partitions scenario/training tf_example/training

# ============================================
# VERIFY
# ============================================

# Show detailed verification
./waymo verify

# Quiet mode (exit code only)
./waymo verify --quiet

# ============================================
# FILTER
# ============================================

# Filter vehicle-to-vehicle interactions
./waymo filter --type v2v

# ============================================
# PROCESS
# ============================================

# Process training data (default: 8 workers)
./waymo process

# Process with more workers
./waymo process --workers 16

# Process specific split
./waymo process --split training
./waymo process --split validation_interactive
./waymo process --split testing_interactive

# Only keep interactive scenarios
./waymo process --interactive-only

# Process tf_example format
./waymo process --format tf_example --split training

# Combine options
./waymo process --split training --workers 16 --interactive-only

# ============================================
# VISUALIZE
# ============================================

# Generate 10 movies (default)
./waymo visualize

# Generate more movies
./waymo visualize --num 20
./waymo visualize --num 50

# Visualize specific format and split
./waymo visualize --format scenario --split training
./waymo visualize --format tf_example --split validation_interactive

# Custom frame rate
./waymo visualize --fps 15
./waymo visualize --fps 20

# Combine options
./waymo visualize --format scenario --split training --num 30 --fps 15

# ============================================
# PIPELINE
# ============================================

# Run complete pipeline with defaults
./waymo pipeline

# Customize pipeline
./waymo pipeline --num 10          # Download 10 files
./waymo pipeline --workers 16      # Use 16 workers for processing
./waymo pipeline --viz 20          # Generate 20 visualizations

# Combine all
./waymo pipeline --num 10 --workers 16 --viz 20

# ============================================
# STATUS & CLEAN
# ============================================

# Show current dataset status
./waymo status

# Clean processed data only
./waymo clean processed

# Clean movies only
./waymo clean movies

# Clean everything (keeps raw data)
./waymo clean all

# ============================================
# HELP
# ============================================

# General help
./waymo --help

# Command-specific help
./waymo download --help
./waymo process --help
./waymo visualize --help

# ============================================
# ADVANCED: Direct Script Usage
# ============================================

# Use bash scripts directly
bash bash/download.sh 5 scenario/training
bash bash/verify.sh --quiet
bash bash/process.sh --split training --workers 8
bash bash/visualize.sh --format scenario --num 10
bash bash/filter.sh --type v2v

# Use Python modules directly
python lib/waymo_preprocess.py --input-dir ... --output-dir ...
python lib/viz_waymo_scenario.py --data_dir ... --output_dir ...
python lib/filter_interactive_training.py --input-dir ... --output-dir ...

# ============================================
# ENVIRONMENT VARIABLES
# ============================================

# Override default paths
export WAYMO_RAW_ROOT=/custom/path/to/raw
export WAYMO_PROCESSED_ROOT=/custom/path/to/processed
export WAYMO_MOVIES_ROOT=/custom/path/to/movies

# Check paths
./waymo status

# ============================================
# EXAMPLES BY USE CASE
# ============================================

# 1. Quick test with sample data
./waymo pipeline

# 2. Download and process large dataset
./waymo download --num 50
./waymo process --workers 16
./waymo visualize --num 100

# 3. Process multiple splits
for split in training validation_interactive testing_interactive; do
    ./waymo download --partitions scenario/$split
    ./waymo process --split $split --workers 16
    ./waymo visualize --split $split --num 10
done

# 4. Filter and process interactive training
./waymo filter --type v2v
./waymo process --format tf_example --split training_interactive

# 5. Generate lots of visualizations
./waymo visualize --num 100 --fps 20

# ============================================
# TROUBLESHOOTING
# ============================================

# Check if gsutil is installed
which gsutil

# Verify Python dependencies
python -c "import tensorflow; import waymo_open_dataset; print('OK')"

# Check current status
./waymo status

# Verify downloads
./waymo verify

# Clean and start fresh
./waymo clean all
./waymo pipeline
```

## Scripts Reorganization (migration guide)

The Waymo dataset scripts were reorganized for better usability. Highlights:

- New structure with a single entry point `./waymo`, modular bash scripts in `bash/`, and Python modules in `lib/`.
- Old multi-script flows are replaced by `./waymo` subcommands (`download`, `verify`, `filter`, `process`, `visualize`, `pipeline`, `status`, `clean`).
- Python modules can still be called directly (e.g., `python lib/waymo_preprocess.py`).

Old → New examples:

```bash
# Before
bash download_waymo_subset.sh
bash verify_downloads.sh
bash process_waymo_subset.sh

# After
./waymo download
./waymo verify
./waymo process
# Or just
./waymo pipeline
```

More examples:

```bash
# Custom download
./waymo download --num 10 --partitions scenario/training

# Processing with options
./waymo process --workers 16 --interactive-only

# Visualizations
./waymo visualize --num 20 --fps 15
```

Key benefits: single entry point, flexible flags (no script edits), better help, status checking, and a complete pipeline command.

## Reorganization Summary

You get:

- Simple: one command for everything
- Flexible: CLI flags for customization
- Complete: built-in pipeline
- Documented: `--help` at each level
- Organized & modular: bash and Python separated; individual components usable

Quick start examples:

```bash
./waymo pipeline
./waymo download --num 20 --partitions scenario/training
./waymo process --split training --workers 16 --interactive-only
./waymo visualize --split training --num 30 --fps 15
./waymo status
```

bash bash/filter.sh --type v2v

```All visualizations are saved to:

- `data/movies/scenario/` - Scenario format visualizations

### Using Python Modules Directly- `data/movies/tf_example/` - TF Example format visualizations



```bash## Requirements

# Preprocessing

python lib/waymo_preprocess.py \```bash

    --input-dir data/datasets/waymo_open_dataset/motion_v_1_3_0/raw/scenario/training \pip install tensorflow waymo-open-dataset-tf-2-11-0 matplotlib numpy

    --output-dir data/datasets/waymo_open_dataset/motion_v_1_3_0/processed/itp_training \```

    --split training \

    --num-workers 8## Notes



# Visualization- All scripts should be run from the repository root directory

python lib/viz_waymo_scenario.py \- Paths are relative to the repository root

    --data_dir data/datasets/waymo_open_dataset/motion_v_1_3_0/raw/scenario/training \- Movies are created at 10 fps by default

    --num_scenarios 10 \- Both MP4 (for video players) and GIF (for GitHub display) formats are generated

    --output_dir data/movies/waymo_open_dataset/motion_v_1_3_0/scenario/training \
    --fps 10

# Filtering
python lib/filter_interactive_training.py \
    --input-dir data/datasets/waymo_open_dataset/motion_v_1_3_0/raw/tf_example/training \
    --output-dir data/datasets/waymo_open_dataset/motion_v_1_3_0/raw/tf_example/training_interactive \
    --type v2v
```

## 🌍 Environment Variables

Override default paths:

```bash
export WAYMO_RAW_ROOT=/path/to/raw/data
export WAYMO_PROCESSED_ROOT=/path/to/processed/data
export WAYMO_MOVIES_ROOT=/path/to/movies

./waymo status  # Will show custom paths
```

## 📊 Data Formats

### Raw Data (TFRecord)

- **scenario**: Protocol buffer format with rich scene information
- **tf_example**: TensorFlow example format optimized for ML

### Processed Data (.npz)

Numpy compressed format containing:
- Agent trajectories (history + future)
- Map features (lanes, road edges, crosswalks)
- Interactive pairs
- Validity masks
- Velocities and headings

### Movies

- **MP4**: High-quality video (H.264)
- **GIF**: Animated previews

## 🛠️ Requirements

- **Google Cloud SDK**: For downloading (`gsutil`)
- **Python 3.8+**: For processing and visualization
- **TensorFlow 2.11+**: For reading TFRecord files
- **waymo-open-dataset**: Waymo SDK
- **FFmpeg**: For MP4 generation
- **ImageMagick**: For GIF generation

Install Python dependencies:

```bash
pip install tensorflow waymo-open-dataset-tf-2-11-0 numpy matplotlib tqdm
```

## 💡 Tips

1. **Start Small**: Use default settings (5 files) to test before downloading everything
2. **Parallel Processing**: Increase `--workers` based on your CPU cores
3. **Storage**: Each TFRecord is ~3GB; plan accordingly
4. **Interactive Only**: Use `--interactive-only` to reduce dataset size by ~50%
5. **Incremental**: Download and process in batches instead of all at once

## 🐛 Troubleshooting

### `gsutil` not found

```bash
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
gcloud init
```

### Permission denied

```bash
chmod +x waymo
chmod +x bash/*.sh
```

### No module named 'waymo_open_dataset'

```bash
pip install waymo-open-dataset-tf-2-11-0
```

### Out of memory during processing

```bash
# Reduce workers
./waymo process --workers 4

# Or process smaller batches
./waymo download --num 5
./waymo process
./waymo download --num 5  # Next batch
./waymo process
```

## 📚 References

- [Waymo Open Dataset](https://waymo.com/open/)
- [Waymo Open Motion Dataset](https://waymo.com/open/data/motion/)
- [Dataset Documentation](https://waymo.com/open/data/motion/tfexample)

## 📝 License

This code follows the same license as the Waymo Open Dataset.
