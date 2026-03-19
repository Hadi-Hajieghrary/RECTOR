# WOMD — Waymo Open Motion Dataset Scripts

Scripts for downloading, inspecting, and managing Waymo Open Motion Dataset (WOMD) files within the RECTOR project.

## Directory Structure

```
scripts/WOMD/
├── download_waymo_sample.sh    # Downloads dataset samples from Google Cloud Storage
├── check_waymo_status.sh       # Reports current download status by format and split
├── clear_waymo_data.sh         # Removes all downloaded tfrecord files
└── README.md                   # This file
```

## Prerequisites

**For downloading** (`download_waymo_sample.sh`):

- **Google Cloud SDK** (`gsutil`, `gcloud`) — installed in the devcontainer by the Dockerfile (Layer 9).
- **Google Cloud authentication** — must be completed before downloading:
  ```bash
  gcloud auth login --no-browser
  gcloud auth application-default login --no-browser
  ```
- **Waymo Open Dataset terms** accepted at https://waymo.com/open/

**For status checking and cleanup** (`check_waymo_status.sh`, `clear_waymo_data.sh`):

- No external dependencies — these scripts use only `find`, `du`, and standard coreutils. They work without Google Cloud authentication.

## Shell Aliases

The devcontainer's `post-create.sh` defines these aliases in `~/.bashrc` for convenience:

| Alias | Command | Description |
|-------|---------|-------------|
| `waymo-download` | `bash scripts/WOMD/download_waymo_sample.sh` | Download 5 files per split |
| `waymo-status` | `bash scripts/WOMD/check_waymo_status.sh` | Show dataset status |
| `waymo-sample` | `NUM_SAMPLE_FILES=10 bash scripts/WOMD/download_waymo_sample.sh` | Quick download of 10 files per split |

## Dataset Location

All data is stored under `$WAYMO_DATA_ROOT` (default: `/workspace/data/WOMD/datasets/waymo_open_dataset/motion_v_1_3_0`):

```
motion_v_1_3_0/
├── raw/                           # Downloaded files ($WAYMO_RAW_ROOT)
│   ├── scenario/                  # Primary protobuf format
│   │   ├── training/
│   │   ├── validation/
│   │   ├── testing/
│   │   ├── training_20s/
│   │   ├── validation_interactive/
│   │   └── testing_interactive/
│   ├── tf/                        # TensorFlow Example format
│   │   ├── training/
│   │   ├── validation/
│   │   └── testing/
│   └── lidar_and_camera/          # Raw sensor data (very large)
│       ├── training/
│       ├── validation/
│       └── testing/
└── processed/                     # Processed data ($WAYMO_PROCESSED_ROOT)
```

### Dataset Formats

| Format | Description | Size per file | Use case |
|--------|-------------|--------------|----------|
| `scenario` | Primary protobuf format, compact | ~400MB | Motion planning (default) |
| `tf` / `tf_example` | TensorFlow Example format | ~500MB | Legacy TF pipelines (can be converted from scenario) |
| `lidar_and_camera` | Raw sensor data (point clouds, images) | ~2GB+ | Perception tasks (very large, optional) |

### Dataset Splits

| Split | Description | Duration | Included by default |
|-------|-------------|----------|-------------------|
| `training` | Standard training scenarios | 9 seconds | Yes |
| `validation` | Standard validation scenarios | 9 seconds | Yes |
| `testing` | Standard testing scenarios | 9 seconds | Yes |
| `training_20s` | Extended training scenarios | 20 seconds | No (`DOWNLOAD_20S=1`) |
| `validation_interactive` | Interactive scenarios with human-driven vehicles | 9 seconds | No (`DOWNLOAD_INTERACTIVE=1`) |
| `testing_interactive` | Interactive scenarios with human-driven vehicles | 9 seconds | No (`DOWNLOAD_INTERACTIVE=1`) |

---

## download_waymo_sample.sh

**Purpose:** Downloads a configurable sample of Waymo Open Motion Dataset files from Google Cloud Storage via `gsutil`.

**GCS Bucket:** `gs://waymo_open_dataset_motion_v_1_3_0/uncompressed/`

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `WOMD_VERSION` | `v_1_3_0` | Dataset version string, used to construct the GCS bucket URL |
| `NUM_SAMPLE_FILES` | `5` | Number of files to download per split. Each file is a `.tfrecord` containing multiple scenarios |
| `WAYMO_RAW_ROOT` | `/workspace/data/WOMD/datasets/waymo_open_dataset/motion_v_1_3_0/raw` | Local destination directory for downloads |
| `DOWNLOAD_INTERACTIVE` | `0` | Set to `1` to include `validation_interactive` and `testing_interactive` splits |
| `DOWNLOAD_20S` | `0` | Set to `1` to include `training_20s` (20-second extended scenarios) |
| `DOWNLOAD_TF` | `0` | Set to `1` to download TF Example format (`tf_example/` in GCS). Can alternatively be generated from scenario format |
| `DOWNLOAD_LIDAR` | `0` | Set to `1` to download `lidar_and_camera` format. Warning: files are very large (~2GB+ each) |

### Pre-Flight Checks

The script performs three checks before downloading and exits with an error if any fail:

1. **`gsutil` availability** — verifies the Google Cloud SDK command is installed.
2. **Authentication** — runs `gcloud auth application-default print-access-token` to confirm valid credentials.
3. **Bucket access** — attempts `gsutil ls` on the training split to verify the account has accepted the Waymo terms and can access the data.

### Download Behavior

- **Directory structure:** Creates the full directory hierarchy under `$WAYMO_RAW_ROOT` with subdirectories for each format and split. Adds `.gitkeep` files so the structure is preserved in Git even when empty.
- **Download method:** Uses `gsutil -m cp -n -I` where:
  - `-m` enables parallel transfers (multi-threaded)
  - `-n` enables no-clobber (skips files that already exist locally)
  - `-I` reads file list from stdin
- **File selection:** Lists files in the GCS bucket with `gsutil ls`, pipes through `head -n $NUM_SAMPLE_FILES` to limit the count, then downloads the selected files.
- **Default downloads:** Always downloads `scenario` format for `training`, `validation`, and `testing` splits.
- **Optional downloads:** Additional formats and splits are controlled by the environment flags listed above.
- **Summary:** After downloading, prints total file count and size with instructions for downloading more.

### Usage Examples

```bash
# Default: 5 files per split, scenario format only
waymo-download

# Download 20 files per split
NUM_SAMPLE_FILES=20 waymo-download

# Quick start with 10 files
waymo-sample

# Include interactive scenarios
DOWNLOAD_INTERACTIVE=1 waymo-download

# Include 20-second extended training scenarios
DOWNLOAD_20S=1 waymo-download

# Include lidar and camera data (very large)
DOWNLOAD_LIDAR=1 waymo-download

# Include TF Example format
DOWNLOAD_TF=1 waymo-download

# Combine multiple options
DOWNLOAD_INTERACTIVE=1 DOWNLOAD_20S=1 NUM_SAMPLE_FILES=10 waymo-download

# Run directly without alias
bash scripts/WOMD/download_waymo_sample.sh
```

---

## check_waymo_status.sh

**Purpose:** Reports the current state of downloaded Waymo dataset files, showing a breakdown by format and split with file counts and sizes.

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `WAYMO_RAW_ROOT` | `/workspace/data/WOMD/datasets/waymo_open_dataset/motion_v_1_3_0/raw` | Directory to inspect |

### How It Works

1. **Checks data directory exists** — exits with an error if `$WAYMO_RAW_ROOT` is missing, suggesting to run the download script.

2. **Iterates over all three formats:** `scenario`, `tf`, `lidar_and_camera`.

3. **For each format,** checks all six splits: `training`, `validation`, `testing`, `training_20s`, `validation_interactive`, `testing_interactive`.

4. **For each existing split directory:**
   - Counts files matching `*tfrecord*` (via `find -type f -name '*tfrecord*' | wc -l`)
   - Measures total size (via `du -sh`)
   - Reports as either `✓ <split>: <count> files (<size>)` or `✗ <split>: empty`

5. **Prints overall totals** — total file count and combined size across all formats and splits.

6. **If zero files found** — prints a warning and suggests running the download script.

### Example Output

```
════════════════════════════════════════════════════════════
  RECTOR - Waymo Dataset Status
════════════════════════════════════════════════════════════

[INFO] Data directory: /workspace/data/WOMD/datasets/waymo_open_dataset/motion_v_1_3_0/raw

Format: scenario
  ✓ training: 5 files (2.1G)
  ✓ validation: 5 files (450M)
  ✓ testing: 5 files (380M)
  ✗ training_20s: empty
  ✗ validation_interactive: empty
  ✗ testing_interactive: empty

Format: tf
  ✗ training: empty
  ✗ validation: empty
  ✗ testing: empty

════════════════════════════════════════════════════════════
[INFO] Total files: 15
[INFO] Total size: 2.9G
════════════════════════════════════════════════════════════
```

### Usage

```bash
# Via alias
waymo-status

# Direct
bash scripts/WOMD/check_waymo_status.sh
```

---

## clear_waymo_data.sh

**Purpose:** Removes all downloaded Waymo dataset `.tfrecord` files, freeing disk space while preserving the directory structure.

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `WAYMO_RAW_ROOT` | `/workspace/data/WOMD/datasets/waymo_open_dataset/motion_v_1_3_0/raw` | Raw dataset directory to clean |

Note: This script cleans only the raw dataset tree under `WAYMO_RAW_ROOT`.

### How It Works

1. **Checks directory exists** — exits cleanly if `$WAYMO_RAW_ROOT` doesn't exist.

2. **Shows current status** — counts all `*tfrecord*` files and reports total size. Exits if no files found.

3. **Prompts for confirmation** — displays:
   ```
   ⚠️  Delete all downloaded Waymo data? [y/N]
   ```
   Only proceeds on `y` or `Y`. Any other input (including Enter) cancels.

4. **Deletes data files** — runs `find "$DATA_DIR" -type f -name '*tfrecord*' -delete` to remove all tfrecord files.

5. **Cleans empty directories** — runs `find "$DATA_DIR" -type d -empty -delete` to remove directories that became empty after file deletion.

6. **Verifies deletion** — re-counts remaining files and reports success or warns if some files remain.

7. **Suggests re-download** — prints `To download again: waymo-download` on success.

### What Gets Deleted

- All files matching `*tfrecord*` anywhere under `$WAYMO_RAW_ROOT`
- Empty directories left after file deletion

### What Is Preserved

- The `$WAYMO_RAW_ROOT` directory itself (if non-empty)
- Any non-tfrecord files (e.g., `.gitkeep`, processed outputs in other formats)
- Parent directories (`data/`, `data/WOMD/datasets/`, etc.)

### Usage

```bash
# Direct (no alias defined — must run explicitly)
bash scripts/WOMD/clear_waymo_data.sh
```

This script is intentionally not aliased to prevent accidental data deletion.
