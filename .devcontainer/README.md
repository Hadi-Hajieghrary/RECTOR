# RECTOR Development Container

Comprehensive reference for the `.devcontainer/` folder in the **RECTOR (Rule-Enforced Constrained Trajectory Optimization and Reasoning)** project.

This devcontainer provides a fully configured, GPU-accelerated development environment for trajectory planning research using the Waymo Open Motion Dataset. It is built on NVIDIA CUDA 12.1 with cuDNN 8 on Ubuntu 22.04, optimized for the RTX A3000 (Ampere architecture, SM 8.6). The first build takes 10-15 minutes; subsequent starts take under 30 seconds.

---

## Table of Contents

- [File Structure](#file-structure)
- [Image Build: Dockerfile](#image-build-dockerfile)
- [Container Orchestration: docker-compose.yml](#container-orchestration-docker-composeyml)
- [VS Code Integration: devcontainer.json](#vs-code-integration-devcontainerjson)
- [Python Dependencies](#python-dependencies)
- [Lifecycle Scripts](#lifecycle-scripts)
- [Scripts Reference](#scripts-reference)
- [Shell Aliases and Commands](#shell-aliases-and-commands)
- [Working with the Waymo Dataset](#working-with-the-waymo-dataset)
- [GPU Configuration](#gpu-configuration)
- [Data Persistence and Directory Structure](#data-persistence-and-directory-structure)
- [Getting Started](#getting-started)
- [Known Issues](#known-issues)
- [Troubleshooting](#troubleshooting)
- [System Requirements](#system-requirements)

---

## File Structure

```
.devcontainer/
├── devcontainer.json                 # VS Code devcontainer configuration
├── docker-compose.yml                # Docker Compose orchestration (GPU, volumes, env)
├── Dockerfile                        # Multi-layer image build (CUDA, Python, ML frameworks)
├── requirements.base.txt             # Stable Python packages (rarely changes)
├── requirements.project.txt          # Project-specific Python packages (add new ones here)
├── post-create.sh                    # Runs once after container creation
├── fix-permissions.sh                # Normalizes file permissions on workspace directories
├── README.md                         # This file
└── scripts/
    ├── setup-externals.sh            # Clones and installs M2I and Waymo repos
    └── setup_git_hooks.sh            # Installs Git pre-commit hook

scripts/                              # Project-level scripts (outside .devcontainer/)
├── git-pre-commit-hook.sh            # Pre-commit hook: generates workspace documentation
├── git-pre-commit-generate-movies.sh # Pre-commit hook: generates BEV movies from scenarios
└── WOMD/                             # Waymo Open Motion Dataset scripts
    ├── download_waymo_sample.sh      # Downloads Waymo Open Dataset samples via gsutil
    ├── check_waymo_status.sh         # Reports dataset download status
    └── clear_waymo_data.sh           # Removes downloaded Waymo tfrecord files
```

| File | Purpose | Changes |
|------|---------|---------|
| `devcontainer.json` | VS Code settings, extensions, ports, environment variables, mounts, lifecycle commands | Occasionally |
| `docker-compose.yml` | GPU access, volume mounts, environment variables, shared memory | Occasionally |
| `Dockerfile` | Image build: CUDA base, Python, PyTorch, TensorFlow, system tools, TeX Live | Rarely (layer-cached) |
| `requirements.base.txt` | Stable scientific/dev Python packages | Rarely |
| `requirements.project.txt` | Project-specific Python packages — add new packages here | Frequently |
| `post-create.sh` | Post-creation setup: directories, aliases, external repos, env checks | Occasionally |
| `fix-permissions.sh` | Sets `chmod 755` on key workspace directories | Rarely |

---

## Image Build: Dockerfile

The Dockerfile builds on `nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04`. The code is organized into 11 commented sections (labeled "LAYER 1" through "LAYER 11") ordered from least-frequently-changed to most-frequently-changed, maximizing Docker layer cache reuse. Note that some sections contain multiple `RUN` instructions (e.g., Layers 4, 6, and 9 each have 2-3 separate `RUN` commands), so the actual Docker layer count is higher than 11.

### Section Summary

| Layer | Contents | Change Frequency |
|-------|----------|-----------------|
| 1 | **Core system packages**: locales, tzdata, sudo, curl, ca-certificates, gnupg, lsb-release, build-essential, cmake, git, pkg-config, wget, unzip, python3.10, python3.10-venv, python3-pip, python3-dev, protobuf-compiler, libprotobuf-dev, OpenCV dependencies (libgl1, libglib2.0-0, libsm6, libxext6, libxrender-dev), ffmpeg, vim, nano, htop, tmux, screen, apt-transport-https | Rarely |
| 2 | **Locale setup**: `en_US.UTF-8`, timezone UTC | Rarely |
| 3 | **Non-root user**: creates `developer` (UID 1000, GID 1000) with passwordless sudo | Rarely |
| 4 | **Python virtual environment**: `/opt/venv` with upgraded pip, setuptools, wheel | Rarely |
| 5 | **PyTorch**: torch 2.2.0+cu121, torchvision 0.17.0+cu121, torchaudio 2.2.0+cu121 (from PyTorch cu121 index) | Rarely |
| 6 | **TensorFlow + Waymo**: tensorflow-cpu 2.11.0, waymo-open-dataset-tf-2-11-0 | Rarely |
| 7 | **Base Python packages**: from `requirements.base.txt` (numpy, scipy, matplotlib, pytest, etc.) | Occasionally |
| 8 | **Project Python packages**: from `requirements.project.txt` (empty by default) | Frequently |
| 9 | **External tools** (3 separate `RUN` commands): Google Cloud SDK (gsutil), Bazel 5.4.0, Git LFS | Occasionally |
| 10 | **Additional system packages**: GNU parallel, TeX Live (texlive-latex-base, texlive-latex-recommended, texlive-latex-extra, texlive-fonts-recommended, texlive-fonts-extra, texlive-science, texlive-bibtex-extra, biber, latexmk) | Occasionally |
| 11 | **Helper scripts**: copies `post-create.sh` and `fix-permissions.sh` into `/usr/local/bin/` and makes them executable | Frequently |

### Final Steps

After the layers, the Dockerfile:
- Creates `/workspace` owned by `developer`
- Switches to user `developer`
- Sets working directory to `/workspace`
- Default command: `bash`

### Build Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `DEBIAN_FRONTEND` | `noninteractive` | Suppresses interactive prompts during apt-get |
| `USERNAME` | `developer` | Non-root user name |
| `USER_UID` | `1000` | User ID |
| `USER_GID` | `1000` | Group ID |

---

## Container Orchestration: docker-compose.yml

Defines the `rector-dev` service with GPU access, volume mounts, and environment configuration.

| Setting | Value |
|---------|-------|
| `image` | `rector-dev:latest` |
| `working_dir` | `/workspace` |
| `command` | `sleep infinity` |

### GPU Configuration

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
runtime: nvidia  # Legacy support for older Docker versions
```

One NVIDIA GPU is reserved. Both the modern `deploy.resources` syntax and the legacy `runtime: nvidia` are specified for compatibility across Docker versions.

### Shared Memory

```yaml
shm_size: "4gb"
```

Allocated for PyTorch DataLoaders that use multiprocessing. Increase to `8gb` or higher if you encounter shared memory errors with many workers.

### Volumes

| Mount | Source | Target | Type | Purpose |
|-------|--------|--------|------|---------|
| Workspace | `..` (project root) | `/workspace` | Bind (cached) | Full project access from host |
| Python venv | `rector-venv` | `/opt/venv` | Named volume | Python virtual environment (rebuilt with container) |

The workspace is a bind mount so all files are directly accessible and persistent on the host. The Python venv is a named volume because it is built from the Dockerfile layers and does not need host persistence.

### Environment Variables

| Variable | Value | Purpose |
|----------|-------|---------|
| `NVIDIA_VISIBLE_DEVICES` | `all` | Expose all GPUs to the NVIDIA runtime |
| `NVIDIA_DRIVER_CAPABILITIES` | `compute,utility` | Enable compute and utility driver capabilities |
| `CUDA_VISIBLE_DEVICES` | `0` | Restrict PyTorch/CUDA to GPU 0 |
| `WAYMO_DATA_ROOT` | `/workspace/data/WOMD/datasets/waymo_open_dataset/motion_v_1_3_0` | Waymo dataset base directory |
| `WAYMO_RAW_ROOT` | `/workspace/data/WOMD/datasets/waymo_open_dataset/motion_v_1_3_0/raw` | Raw (downloaded) Waymo data |
| `WAYMO_PROCESSED_ROOT` | `/workspace/data/WOMD/datasets/waymo_open_dataset/motion_v_1_3_0/processed` | Processed Waymo data |
| `WAYMO_MOVIES_ROOT` | `/workspace/data/WOMD/movies` | Generated visualization movies |
| `TORCH_HOME` | `/workspace/.cache/torch` | PyTorch model/cache directory |
| `HF_HOME` | `/workspace/.cache/huggingface` | HuggingFace model/cache directory |
| `TORCH_CUDA_ARCH_LIST` | `${TORCH_CUDA_ARCH_LIST:-}` | CUDA compute capability (auto-detected; override via host env var if needed) |

---

## VS Code Integration: devcontainer.json

### Basic Configuration

| Setting | Value |
|---------|-------|
| Container name | `RECTOR - Rule-Enforced Constrained Trajectory Optimization and Reasoning` |
| Docker Compose file | `docker-compose.yml` |
| Service | `rector-dev` |
| Workspace folder | `/workspace` |
| Remote user | `developer` |
| Container user | `developer` |

### Dev Container Features

Two features are installed into the running container by VS Code:

| Feature | Configuration |
|---------|--------------|
| `ghcr.io/devcontainers/features/common-utils:2` | Installs zsh, oh-my-zsh, upgrades packages. User: `developer` (1000:1000) |
| `ghcr.io/devcontainers/features/git:1` | Installs latest Git version via PPA |

### VS Code Extensions (20 total)

**Python Development:**

| Extension ID | Name |
|-------------|------|
| `ms-python.python` | Python |
| `ms-python.vscode-pylance` | Pylance |
| `ms-python.black-formatter` | Black Formatter |
| `ms-python.flake8` | Flake8 |
| `charliermarsh.ruff` | Ruff (linter, VS Code extension only — not pip-installed) |
| `ms-python.mypy-type-checker` | Mypy Type Checker |

**Jupyter:**

| Extension ID | Name |
|-------------|------|
| `ms-toolsai.jupyter` | Jupyter |
| `ms-toolsai.jupyter-keymap` | Jupyter Keymap |
| `ms-toolsai.jupyter-renderers` | Jupyter Renderers |
| `ms-toolsai.vscode-jupyter-cell-tags` | Jupyter Cell Tags |
| `ms-toolsai.vscode-jupyter-slideshow` | Jupyter Slideshow |

**AI Assistance:**

| Extension ID | Name |
|-------------|------|
| `github.copilot` | GitHub Copilot |
| `github.copilot-chat` | GitHub Copilot Chat |
| `visualstudioexptteam.vscodeintellicode` | IntelliCode |

**Development Workflow:**

| Extension ID | Name |
|-------------|------|
| `eamodio.gitlens` | GitLens |
| `ms-azuretools.vscode-docker` | Docker |
| `mechatroner.rainbow-csv` | Rainbow CSV |
| `streetsidesoftware.code-spell-checker` | Code Spell Checker |
| `gruntfuggly.todo-tree` | TODO Tree |
| `James-Yu.latex-workshop` | LaTeX Workshop |

### Editor Settings

| Setting | Value | Effect |
|---------|-------|--------|
| `python.defaultInterpreterPath` | `/opt/venv/bin/python` | Uses the container's virtual environment |
| `python.analysis.typeCheckingMode` | `basic` | Pylance type checking level |
| `python.analysis.autoImportCompletions` | `true` | Auto-suggest imports |
| `python.formatting.provider` | `black` | Uses Black as the Python formatter |
| `[python].editor.formatOnSave` | `true` | Auto-formats Python files on save |
| `[python].editor.codeActionsOnSave` | `source.organizeImports: explicit` | Organizes imports on save |
| `editor.formatOnSave` | `true` | Auto-formats all files on save |
| `editor.codeActionsOnSave` | `source.organizeImports: explicit` | Organizes imports on save (global, in addition to Python-scoped) |
| `editor.rulers` | `[100]` | Visual ruler at column 100 (guide only — does not configure Black's line length) |
| `editor.renderWhitespace` | `trailing` | Highlights trailing whitespace |
| `terminal.integrated.defaultProfile.linux` | `bash` | Default terminal shell |

**Hidden files** (excluded from VS Code file explorer):

`__pycache__`, `*.pyc`, `.pytest_cache`, `.ipynb_checkpoints`, `*.o`, `.mypy_cache`, `.ruff_cache`

Note: `*.so` files are **not** excluded (`false`).

**File watcher exclusions** (for performance):

`.git/objects`, `.git/subtree-cache`, `node_modules`, `__pycache__`, `.cache`, `data`

**Jupyter settings:**

| Setting | Value |
|---------|-------|
| `jupyter.askForKernelRestart` | `false` |
| Notebook cell toolbar (default) | Right |
| Notebook cell toolbar (jupyter-notebook) | Left |

### Port Forwarding

| Port | Label | Auto-forward |
|------|-------|-------------|
| 6006 | TensorBoard | Notify |
| 8888 | Jupyter | Notify |

### Container Environment Variables

These are set in `docker-compose.yml` (available to both `devcontainer` and `docker exec` sessions), with `WAYMO_DATA_ROOT` also set via `containerEnv` in `devcontainer.json`:

| Variable | Value | Purpose |
|----------|-------|---------|
| `PYTHONUNBUFFERED` | `1` | Disable Python output buffering |
| `PYTHONIOENCODING` | `utf-8` | Force UTF-8 encoding |
| `PYTHONDONTWRITEBYTECODE` | `1` | Prevent `.pyc` file generation |
| `PYTHONPATH` | `/workspace:/workspace/externals/M2I/src` | Module search paths |
| `CUDA_VISIBLE_DEVICES` | `0` | Restrict to GPU 0 |
| `TORCH_HOME` | `/workspace/.cache/torch` | PyTorch cache |
| `HF_HOME` | `/workspace/.cache/huggingface` | HuggingFace cache |
| `TORCH_CUDA_ARCH_LIST` | `${TORCH_CUDA_ARCH_LIST:-}` | CUDA compute capability (auto-detected; override via host env var) |
| `WAYMO_DATA_ROOT` | `/workspace/data/WOMD/datasets/waymo_open_dataset/motion_v_1_3_0` | Waymo dataset root |
| `WAYMO_RAW_ROOT` | `/workspace/data/WOMD/datasets/waymo_open_dataset/motion_v_1_3_0/raw` | Raw dataset root |
| `WAYMO_PROCESSED_ROOT` | `/workspace/data/WOMD/datasets/waymo_open_dataset/motion_v_1_3_0/processed` | Processed dataset root |
| `WAYMO_MOVIES_ROOT` | `/workspace/data/WOMD/movies` | BEV movies output root |

### Mounts

```json
"mounts": [
  "type=bind,source=${localEnv:HOME}${localEnv:USERPROFILE}/.ssh,target=/home/developer/.ssh,readonly,consistency=cached"
]
```

Mounts the host machine's SSH keys (read-only) so Git operations over SSH work inside the container without additional setup. Uses `${localEnv:HOME}` (Linux/macOS) and `${localEnv:USERPROFILE}` (Windows) for cross-platform compatibility.

### Lifecycle Commands

| Hook | Command | When |
|------|---------|------|
| `postCreateCommand` | `bash .devcontainer/post-create.sh` | Once, after the container is first created |
| `postStartCommand` | `echo 'RECTOR devcontainer started ✅'` | Every time the container starts |

---

## Python Dependencies

### Two-Tier Strategy

Dependencies are split into two files to optimize Docker layer caching:

1. **`requirements.base.txt`** — Stable scientific and development packages. Changes to this file invalidate Docker layer 7 and rebuild everything from layer 7 onward, but layers 1-6 (including PyTorch and TensorFlow, ~10GB) remain cached.

2. **`requirements.project.txt`** — Project-specific packages. Changes here only invalidate Docker layer 8, keeping all base packages cached. This file is empty by default and is where you should add new dependencies.

### requirements.base.txt — Full Package List

| Category | Packages |
|----------|----------|
| **Scientific computing** | `numpy>=1.23.0`, `scipy>=1.10.0` |
| **Configuration and utilities** | `pyyaml>=6.0`, `tqdm>=4.66.0`, `pickle5>=0.0.11` |
| **Visualization** | `matplotlib>=3.7.0`, `seaborn>=0.12.0` |
| **Geometry** | `shapely>=2.0.0` |
| **Development tools** | `cython>=3.0.0`, `jupyterlab>=4.0.0`, `tensorboard>=2.15.0` |
| **Testing** | `pytest>=7.4.0`, `pytest-cov>=4.1.0` |
| **Code quality** | `black>=23.0.0`, `flake8>=6.0.0`, `mypy>=1.5.0` |
| **Image processing** | `opencv-python>=4.8.0`, `pillow>=10.0.0` |
| **Data formats** | `h5py>=3.9.0` |
| **Notebook utilities** | `ipywidgets>=8.0.0`, `ipython>=8.12.0` |

Note: **Ruff** is available only as a VS Code extension (`charliermarsh.ruff`), not as a pip package in these requirements.

### requirements.project.txt

Empty by default. Add project-specific packages here:

```
# Example:
transformers>=4.30.0
accelerate>=0.20.0
```

### Adding New Packages

**Python packages** — add to `requirements.project.txt`, then rebuild the container (`F1` → "Dev Containers: Rebuild Container"). Only layer 8 rebuilds; PyTorch, TensorFlow, and base packages stay cached.

**System (apt) packages** — add to Layer 10 in the `Dockerfile`:

```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    your-new-package \
    && rm -rf /var/lib/apt/lists/*
```

Then rebuild the container.

---

## Lifecycle Scripts

### post-create.sh

Runs once after the container is first created (`postCreateCommand`). Executes the following steps in order:

1. **Makes scripts executable** — `chmod +x` on all `.sh` files in `.devcontainer/`, `.devcontainer/scripts/`, `scripts/`, and `scripts/WOMD/`.

2. **Sets up external repositories** — calls `.devcontainer/scripts/setup-externals.sh` if present. Clones M2I and Waymo Open Dataset repos into `/workspace/externals/` and installs them.

3. **Installs Git hooks** — calls `.devcontainer/scripts/setup_git_hooks.sh` if present. Installs the pre-commit hook that auto-generates workspace documentation.

4. **Creates directory structure** — ensures these directories exist:
   - `data/WOMD/datasets/waymo_open_dataset/motion_v_1_3_0/raw`
   - `data/WOMD/datasets/waymo_open_dataset/motion_v_1_3_0/processed`
   - `data/WOMD/movies`
   - `models/checkpoints`
   - `models/pretrained`
   - `output`
   - `logs`
   - `notebooks`
   - `externals`
   - `.cache/torch`
   - `.cache/huggingface`

5. **Installs project requirements** — runs `pip install -r requirements.txt` if a root-level `requirements.txt` exists. Falls back to replacing `tensorflow==2.11.0` with `tensorflow-cpu==2.11.0` if the initial install fails.

6. **Installs project in editable mode** — runs `pip install -e .` if `setup.py` or `pyproject.toml` exists at the project root.

7. **Adds shell aliases** — appends aliases to `~/.bashrc` (see [Shell Aliases and Commands](#shell-aliases-and-commands)).

8. **Runs environment checks** — executes a Python script that verifies:
   - Python version
   - PyTorch version and CUDA availability
   - CUDA version, GPU name, and GPU memory (if CUDA is available)
   - TensorFlow version
   - Waymo Open Dataset API availability

9. **Checks LaTeX tools** — verifies that `pdflatex`, `biber`, and `latexmk` are installed.

10. **Checks for Waymo data** — if the raw data directory exists, runs `scripts/WOMD/check_waymo_status.sh`. Otherwise, prints a message suggesting `waymo-download`.

11. **Prints Quick Start guide** — displays available commands (note: references a `tb` alias that is not actually defined — see [Known Issues](#known-issues)).

### fix-permissions.sh

Recursively sets permissions to `755` (`chmod -R 755`) on the following workspace directories, creating them if they don't exist:

| Directory |
|-----------|
| `/workspace/data` |
| `/workspace/models` |
| `/workspace/output` |
| `/workspace/logs` |
| `/workspace/.cache` |
| `/workspace/notebooks` |

Run manually if you encounter permission issues:

```bash
bash /usr/local/bin/fix-permissions.sh
```

---

## Scripts Reference

Scripts are organized across three locations:

| Location | Purpose | Documentation |
|----------|---------|---------------|
| `.devcontainer/scripts/` | Devcontainer setup (run during container creation) | Below |
| `scripts/` | Git hook scripts (project-level) | See `scripts/README.md` |
| `scripts/WOMD/` | Waymo Open Motion Dataset management | See `scripts/WOMD/README.md` |

All scripts use `bash`. Most use `set -euo pipefail` (exit on error, undefined variable, pipe failure); see individual READMEs for exceptions.

### .devcontainer/scripts/setup-externals.sh

**Purpose:** Clones and installs two external repositories into `/workspace/externals/`.

**Repositories:**

| Repository | Source | Install Directory |
|-----------|--------|-------------------|
| M2I | `https://github.com/Tsinghua-MARS-Lab/M2I.git` | `/workspace/externals/M2I` |
| Waymo Open Dataset | `https://github.com/waymo-research/waymo-open-dataset.git` | `/workspace/externals/waymo-open-dataset` |

**Behavior:**
- Skips cloning if the directory already exists.
- For M2I:
  - Installs requirements from `requirements.txt`, filtering out `tensorflow-gpu` and `waymo-open-dataset-tf-2-4-0` (uses `grep -vE` to `/tmp/m2i_requirements_filtered.txt`).
  - Installs the package in editable mode (`pip install -e .`) if `setup.py` or `pyproject.toml` exists.
  - Attempts to build Cython extensions in `src/` via `python setup.py build_ext --inplace`. Falls back silently if the build fails.
- For Waymo Open Dataset:
  - Installs requirements from `requirements.txt`, filtering out `tensorflow-gpu`, `tensorflow==*`, and all `waymo-open-dataset-tf*` packages (uses `grep -vE` to `/tmp/waymo_requirements_filtered.txt`).
  - Installs the package in editable mode (`pip install -e .`) if `setup.py` or `pyproject.toml` exists.
- Both repos use the container's pre-installed TensorFlow 2.11 and waymo-open-dataset-tf-2-11-0 rather than installing their own versions.

**Called by:** `post-create.sh` (step 2).

### .devcontainer/scripts/setup_git_hooks.sh

**Purpose:** Installs the Git pre-commit hook.

**Behavior:**
1. Checks that `.git` directory exists (skips if not in a git repo).
2. Copies `scripts/git-pre-commit-hook.sh` to `.git/hooks/pre-commit`.
3. Makes the hook executable.
4. Runs the hook once as a test to verify it works.

**Called by:** `post-create.sh` (step 3).

---

## Shell Aliases and Commands

The following aliases are added to `~/.bashrc` by `post-create.sh`:

### Training and Evaluation

> **Note**: The `rector-train`, `rector-eval`, and `rector-viz` aliases are commented out in `post-create.sh` pending creation of the corresponding scripts. They are not active in the current container.

Use the full script paths directly, for example:
```bash
cd /workspace/models/RECTOR
python scripts/experiments/evaluate_m2i_rector.py \
  --data_dir /workspace/data/WOMD/datasets/waymo_open_dataset/motion_v_1_3_0/processed/augmented/scenario/validation_interactive \
  --max_scenarios 100
```

### Development Tools

| Alias | Command | Description |
|-------|---------|-------------|
| `jlab` | `jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root` | Launch Jupyter Lab on port 8888 |
| `pyinfo` | *(inline Python)* | Print PyTorch version, CUDA availability, and GPU name |

### Dataset Management

| Alias | Command | Description |
|-------|---------|-------------|
| `waymo-download` | `bash scripts/WOMD/download_waymo_sample.sh` | Download 5 samples per split |
| `waymo-status` | `bash scripts/WOMD/check_waymo_status.sh` | Show dataset download status |
| `waymo-sample` | `NUM_SAMPLE_FILES=10 bash scripts/WOMD/download_waymo_sample.sh` | Quick download of 10 samples per split |

### GPU Monitoring

| Alias | Command | Description |
|-------|---------|-------------|
| `gpu` | `nvidia-smi` | Show GPU status |
| `gpu-watch` | `watch -n 1 nvidia-smi` | Continuously monitor GPU (updates every second) |

---

## Working with the Waymo Dataset

The Waymo dataset is **not** downloaded automatically. The container only creates the directory structure, installs the tools (Google Cloud SDK, `gsutil`), and provides helper scripts and aliases.

### Authentication

Before downloading, you must authenticate with Google Cloud and accept the dataset terms:

```bash
# Authenticate
gcloud auth login --no-browser
gcloud auth application-default login --no-browser
```

You must also accept the Waymo Open Dataset terms at https://waymo.com/open/

### Downloading

```bash
# Download 5 files per split (training, validation, testing)
waymo-download

# Download more samples
NUM_SAMPLE_FILES=20 waymo-download

# Quick start with 10 samples
waymo-sample

# Include interactive scenarios
DOWNLOAD_INTERACTIVE=1 waymo-download

# Include 20-second extended training scenarios
DOWNLOAD_20S=1 waymo-download

# Include lidar and camera data (very large files)
DOWNLOAD_LIDAR=1 waymo-download

# Combine options
DOWNLOAD_INTERACTIVE=1 DOWNLOAD_20S=1 NUM_SAMPLE_FILES=10 waymo-download
```

### Checking Status

```bash
waymo-status
```

Example output:

```
Format: scenario
  ✓ training: 5 files (2.1G)
  ✓ validation: 5 files (450M)
  ✓ testing: 5 files (380M)

Total files: 30
Total size: 5.5G
```

### Clearing Data

```bash
bash scripts/WOMD/clear_waymo_data.sh
```

Prompts for confirmation before deleting. Removes all `.tfrecord` files while preserving the directory structure.

### Dataset Formats

| Format | Description | Size |
|--------|-------------|------|
| `scenario` | Primary protobuf format, compact | Small-Medium |
| `tf` | TensorFlow Example format (can be converted from scenario) | Medium |
| `lidar_and_camera` | Raw sensor data | Very large |

### Dataset Splits

| Split | Description | Duration |
|-------|-------------|----------|
| `training` | Standard training scenarios | 9 seconds |
| `validation` | Standard validation scenarios | 9 seconds |
| `testing` | Standard testing scenarios | 9 seconds |
| `training_20s` | Extended training scenarios | 20 seconds |
| `validation_interactive` | Interactive scenarios with human-driven vehicles | 9 seconds |
| `testing_interactive` | Interactive scenarios with human-driven vehicles | 9 seconds |

Data is stored at `/workspace/data/WOMD/datasets/waymo_open_dataset/motion_v_1_3_0/raw/<format>/<split>/`.

### Using an Existing Dataset

If you already have the Waymo dataset on your host machine, mount it by adding to `docker-compose.yml`:

```yaml
volumes:
  - /absolute/path/to/your/waymo_dataset:/workspace/data/WOMD/datasets/waymo_open_dataset/motion_v_1_3_0/raw:cached
```

Then rebuild the container.

---

## GPU Configuration

| Setting | Value | Source |
|---------|-------|--------|
| CUDA version | 12.1.1 with cuDNN 8 | Dockerfile base image |
| GPU count | 1 | `docker-compose.yml` |
| Visible device | GPU 0 | `CUDA_VISIBLE_DEVICES=0` in `docker-compose.yml` |
| Shared memory | 4GB | `shm_size: "4gb"` in `docker-compose.yml` |
| Compute capability target | Auto-detected (default) | `TORCH_CUDA_ARCH_LIST` in `docker-compose.yml` (empty = PyTorch auto-detects at runtime; override with `export TORCH_CUDA_ARCH_LIST="8.6"` on host) |

### CUDA_LAUNCH_BLOCKING (Optional Debugging)

`CUDA_LAUNCH_BLOCKING=1` makes CUDA operations synchronous, which provides exact error locations in stack traces but costs approximately 5-10% performance. It is **not set by default**. To enable during debugging:

```bash
export CUDA_LAUNCH_BLOCKING=1
```

### Shared Memory

If you encounter DataLoader errors related to shared memory, either reduce the number of workers:

```python
DataLoader(..., num_workers=2)
```

Or increase shared memory in `docker-compose.yml`:

```yaml
shm_size: "8gb"
```

---

## Data Persistence and Directory Structure

All project data resides on the host filesystem via the bind mount and persists across container rebuilds.

```
RECTOR/
├── .cache/                    # Model and library caches
│   ├── torch/                 # PyTorch model cache (TORCH_HOME)
│   └── huggingface/           # HuggingFace model cache (HF_HOME)
├── data/                      # All datasets and data artifacts
│   └── WOMD/                  # Waymo Open Motion Dataset
│       ├── datasets/          # Dataset mount point
│       │   └── waymo_open_dataset/motion_v_1_3_0/
│       │       ├── raw/       # Downloaded tfrecord files (WAYMO_RAW_ROOT)
│       │       └── processed/ # Converted and augmented data (WAYMO_PROCESSED_ROOT)
│       ├── scripts/           # Data processing scripts
│       ├── src/               # C++ converter (Bazel)
│       ├── movies/            # Generated BEV movies (WAYMO_MOVIES_ROOT)
│       ├── visualizations/    # Generated static visualizations
│       └── waymo_rule_eval/   # Rule evaluation framework
├── models/                    # Model files
│   ├── checkpoints/           # Training checkpoints
│   └── pretrained/            # Downloaded pretrained models
├── externals/                 # External repositories (M2I, waymo-open-dataset)
├── notebooks/                 # Jupyter notebooks
├── output/                    # Training outputs
└── logs/                      # Log files
```

### Named Volume

| Volume | Container Path | Purpose |
|--------|---------------|---------|
| `rector-venv` | `/opt/venv` | Python virtual environment |

The venv is a named Docker volume because it is built from the Dockerfile layers. To force a full Python package reinstall:

```bash
# From the host
docker volume rm rector-venv
```

Then rebuild the container.

---

## Getting Started

### First-Time Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Hadi-Hajieghrary/RECTOR.git
   cd RECTOR
   ```

2. **Open in VS Code:**
   ```bash
   code .
   ```

3. **Start the container:**
   Press `F1` and select "Dev Containers: Reopen in Container", or click the `><` icon in the bottom-left corner and choose "Reopen in Container".

4. **Wait for the initial build.** When you see "RECTOR devcontainer started ✅", the environment is ready.

### Verify the Environment

```bash
python --version              # Python 3.10.x
pyinfo                        # PyTorch version, CUDA availability, GPU name
gpu                           # nvidia-smi output
waymo-status                  # Dataset download status
git lfs version               # Git LFS version
```

### Alternative: Docker Compose Without VS Code

```bash
cd /path/to/RECTOR

# Build
docker compose -f .devcontainer/docker-compose.yml build

# Start
docker compose -f .devcontainer/docker-compose.yml up -d

# Enter container
docker exec -it rector-dev-1 bash

# Stop
docker compose -f .devcontainer/docker-compose.yml down
```

---

## Known Issues

No blocking known issues at this time.

---

## Troubleshooting

### Container Won't Start

```bash
# Check Docker is running
sudo systemctl status docker

# Verify GPU access from Docker
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi

# Clean rebuild
docker compose -f .devcontainer/docker-compose.yml down -v
docker compose -f .devcontainer/docker-compose.yml build --no-cache
```

### GPU Not Working Inside Container

```bash
echo $CUDA_VISIBLE_DEVICES       # Should show "0"
nvidia-smi                        # Should show your GPU
python -c "import torch; print(torch.cuda.is_available())"  # Should print True
```

### Permission Issues

```bash
# Inside container
bash /usr/local/bin/fix-permissions.sh

# From host
sudo chown -R 1000:1000 /path/to/RECTOR
```

### Shared Memory Errors

Increase shared memory in `docker-compose.yml`:

```yaml
shm_size: "8gb"  # Increase from 4gb
```

### Waymo Download Problems

```bash
# Re-authenticate
gcloud auth revoke --all
gcloud auth login --no-browser
gcloud auth application-default login --no-browser

# Test access
gsutil ls gs://waymo_open_dataset_motion_v_1_3_0/uncompressed/scenario/training
```

Ensure you have accepted the Waymo Open Dataset terms at https://waymo.com/open/

---

## System Requirements

### Required Software

- **Docker** >= 20.10 with Compose V2
- **NVIDIA Docker Runtime** (`nvidia-docker2`)
- **VS Code** with the Dev Containers extension (for devcontainer workflow)
- **NVIDIA GPU Driver** compatible with CUDA 12.1

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| OS | Linux (Ubuntu 20.04+) | Ubuntu 22.04 |
| GPU | NVIDIA (Compute >= 5.0) | RTX A3000 or better (Ampere, SM 8.6) |
| RAM | 16GB | 32GB |
| Disk | 50GB free | 100GB+ (for full Waymo dataset) |

### Quick Verification

```bash
docker --version && docker compose version
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```
