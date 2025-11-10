# RECTOR Development Container

Welcome to the **RECTOR (Rule-Enforced Constrained Trajectory Optimization and Reasoning)** development environment. This containerized setup gives you everything you need to start working on trajectory planning research immediately—no installation headaches, no dependency conflicts, just open and code.

---

## What You're Getting

This development container is built on **NVIDIA CUDA 12.2 with cuDNN 8 on Ubuntu 22.04**, specifically optimized for the RTX A3000 GPU. It includes:

- **Deep Learning Frameworks**: PyTorch 2.2.0 with CUDA support, plus TensorFlow for Waymo dataset tools
- **AI Coding Assistants**: GitHub Copilot and IntelliCode to accelerate your development
- **Scientific Stack**: NumPy, SciPy, Matplotlib, and specialized geometry tools
- **Development Tools**: Jupyter Lab, TensorBoard, comprehensive testing and linting tools
- **Waymo Dataset Integration**: Pre-configured tools and scripts for dataset management
- **GPU Optimization**: Properly configured CUDA environment with debugging support

The first build takes 10-15 minutes, but after that, you're up and running in under 30 seconds every time.

---

## Getting Started

### First-Time Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Hadi-Hajieghrary/RECTOR.git
   cd RECTOR
   ```

2. **Open in VS Code**:
   ```bash
   code .
   ```

3. **Start the container**:
   - Press `F1` and select "Dev Containers: Reopen in Container"
   - Or click the `><` icon in the bottom-left corner and choose "Reopen in Container"

4. **Wait for the initial build**. You'll see the progress in the terminal. When you see "RECTOR devcontainer started ✅", you're ready to go.

### Verify Everything Works

Once inside the container, run these commands to confirm your environment:

```bash
# Check Python and packages
python --version          # Should show Python 3.10.x
pyinfo                    # Shows Python, PyTorch, and CUDA versions

# Verify GPU access
gpu                       # Runs nvidia-smi to show GPU status

# Check Waymo dataset tools
waymo-status              # Shows current dataset status

# Confirm Git LFS is ready
git lfs version           # Should show Git LFS version
```

### Start Working

You can now launch your development tools:

```bash
jlab                      # Start Jupyter Lab on http://localhost:8888
tb                        # Start TensorBoard on http://localhost:6006
```

Or jump straight into development with the RECTOR shortcuts:

```bash
rector-train              # Run training scripts
rector-eval               # Run evaluation
rector-viz                # Run visualization
```

---

## Understanding Your Environment

### What's in the Python Environment

Your Python 3.10 environment lives at `/opt/venv` and is automatically activated. Here's what's pre-installed:

**Scientific Computing**: 
- Core libraries: numpy ≥1.23.0, scipy ≥1.10.0
- Visualization: matplotlib ≥3.7.0, seaborn ≥0.12.0
- Geometry: shapely ≥2.0.0

**Deep Learning**:
- PyTorch 2.2.0 with CUDA 12.1 support (including torchvision and torchaudio)
- TensorFlow 2.11.0 (CPU-only to avoid CUDA conflicts)
- Waymo Open Dataset API

**Development Tools**:
- Interactive: jupyterlab ≥4.0.0, ipython ≥8.12.0
- Monitoring: tensorboard ≥2.15.0
- Building: cython ≥3.0.0

**Code Quality**:
- Testing: pytest ≥7.4.0, pytest-cov ≥4.1.0
- Formatting: black ≥23.0.0
- Linting: flake8 ≥6.0.0, ruff (modern fast linter)
- Type checking: mypy ≥1.5.0

**Utilities**:
- Image processing: opencv-python ≥4.8.0, pillow ≥10.0.0
- Data formats: h5py ≥3.9.0, pyyaml ≥6.0
- Progress bars: tqdm ≥4.66.0

To add more packages:
```bash
pip install package-name

# Or for project-wide packages
echo "package-name>=version" >> requirements.txt
pip install -r requirements.txt
```

### GPU Configuration

Your GPU setup is already optimized:

- **CUDA Version**: 12.2.2 with cuDNN 8
- **Visible Device**: GPU 0 by default (change in docker-compose.yml if needed)
- **Debug Mode**: `CUDA_LAUNCH_BLOCKING=1` is enabled for better error messages
- **Shared Memory**: 4GB allocated for PyTorch DataLoaders

The debug mode makes CUDA operations synchronous, which gives you exact error locations but costs about 5-10% performance. For production training, disable it:

```bash
export CUDA_LAUNCH_BLOCKING=0
```

If you encounter DataLoader errors about shared memory:
```python
# Reduce workers
DataLoader(..., num_workers=2)

# Or increase shared memory in docker-compose.yml
shm_size: "8gb"
```

### VS Code Features

The container comes with 19 pre-installed extensions that make development smoother:

**Python Development**:
- Full Python support with Pylance for intelligent code completion
- Black formatter (auto-formats on save)
- Flake8 and Ruff for linting
- MyPy for type checking

**Jupyter Integration**:
- Complete Jupyter support with enhanced rendering
- Keyboard shortcuts and cell organization
- Slideshow capabilities for presentations

**AI Assistance**:
- GitHub Copilot for AI-powered code suggestions
- Copilot Chat for interactive help
- IntelliCode for smart autocomplete based on patterns

**Development Workflow**:
- GitLens for advanced Git integration
- Docker extension for container management
- Rainbow CSV for data file visualization
- TODO Tree to track tasks
- Code spell checker

Your editor is configured to:
- Auto-format with Black on save (100 character line length)
- Organize imports automatically
- Highlight trailing whitespace
- Hide cache directories from the file explorer
- Exclude large data folders from file watching (for performance)

### AI-Powered Development

GitHub Copilot and IntelliCode can significantly speed up your development:

- **Code Suggestions**: Copilot suggests entire functions and code blocks as you type
- **Interactive Help**: Use Copilot Chat to ask questions about your code
- **Pattern Learning**: IntelliCode learns from your codebase to suggest better completions
- **Documentation**: Get explanations and examples instantly

Studies show developers using Copilot complete tasks 40-50% faster. It's particularly helpful for:
- Boilerplate code
- Understanding unfamiliar APIs
- Writing tests
- Debugging errors

### Git and Version Control

Git is pre-configured with useful features:

**Git LFS** for large files is already installed. Use it for models and datasets:
```bash
git lfs track "*.safetensors"
git lfs track "*.ckpt"
git lfs track "*.h5"
```

**Git Pre-Commit Hook** automatically generates workspace documentation:
- `WORKSPACE_STRUCTURE.md` - Complete workspace tree (excludes data files)
- `DATA_INVENTORY.md` - Data folder structure with file counts and sizes

The hook runs automatically before each commit, keeping your workspace documentation up-to-date. If you need to reinstall it:
```bash
bash .devcontainer/scripts/setup_git_hooks.sh
```

**SSH Keys** from your host machine are automatically mounted (read-only) at `/home/developer/.ssh`, so Git operations just work:
```bash
git clone git@github.com:your-org/private-repo.git
git push origin main
```

No need to set up SSH keys inside the container—they're seamlessly available.

---

## Working with the Waymo Dataset

### Downloading Dataset Samples

The container includes scripts to download Waymo Open Dataset samples:

```bash
# Download 5 files per split (training, validation, testing)
waymo-download

# Download more samples
NUM_SAMPLE_FILES=20 waymo-download

# Quick start with 10 samples
waymo-sample

# Include interactive scenarios (validation_interactive, testing_interactive)
DOWNLOAD_INTERACTIVE=1 waymo-download

# Include 20-second extended training scenarios (training_20s)
DOWNLOAD_20S=1 waymo-download

# Include lidar and camera data (warning: very large files)
DOWNLOAD_LIDAR=1 waymo-download

# Combine options for comprehensive download
DOWNLOAD_INTERACTIVE=1 DOWNLOAD_20S=1 NUM_SAMPLE_FILES=10 waymo-download
```

### Checking Dataset Status

See what you've downloaded:
```bash
waymo-status
```

This shows a breakdown by format and split, with file counts and sizes:
```
Format: scenario
  ✓ training: 5 files (2.1G)
  ✓ validation: 5 files (450M)
  ✓ testing: 5 files (380M)

Total files: 30
Total size: 5.5G
```

### Clearing Downloaded Data

To remove all downloaded Waymo dataset files:
```bash
waymo-clear
```

This will prompt for confirmation before deleting. It removes all `.tfrecord` files while keeping the directory structure intact, so you can re-download later.

### Authentication

Before downloading, you need to authenticate with Google Cloud:

```bash
gcloud auth login --no-browser
gcloud auth application-default login --no-browser
```

You'll also need to accept the Waymo Open Dataset terms at https://waymo.com/open/

### Dataset Formats

The Waymo dataset comes in three formats:

- **scenario**: Motion planning format (this is what you'll primarily use)
- **tf_example**: Legacy TensorFlow format
- **lidar_and_camera**: Raw sensor data (very large)

### Dataset Splits

The scenario format includes several splits:

- **training**: Standard training scenarios (8-10 seconds)
- **validation**: Standard validation scenarios
- **testing**: Standard testing scenarios
- **training_20s**: Extended 20-second training scenarios for longer horizon planning
- **validation_interactive**: Interactive validation scenarios with human-driven vehicles
- **testing_interactive**: Interactive testing scenarios with human-driven vehicles

Data is stored at `/workspace/data/datasets/waymo_open_dataset/motion_v_1_3_0/raw/` with subdirectories for each format and split.

The interactive scenarios are particularly useful for testing models that need to predict or interact with human-driven vehicles, while the 20-second scenarios enable longer planning horizons.

### Using an Existing Dataset

If you already have the Waymo dataset on your host machine, you can mount it instead of downloading. Edit `.devcontainer/docker-compose.yml`:

```yaml
volumes:
  - /absolute/path/to/your/waymo_dataset:/workspace/data/datasets/waymo_open_dataset/motion_v_1_3_0/raw:cached
```

Then rebuild the container.

---

## Handy Commands and Shortcuts

The container sets up several aliases to make your workflow smoother:

### RECTOR-Specific Commands

```bash
rector-train    # python scripts/train.py
rector-eval     # python scripts/validate.py
rector-viz      # python scripts/visualize.py
```

### Service Launchers

```bash
jlab           # Launch Jupyter Lab (http://localhost:8888)
tb             # Launch TensorBoard (http://localhost:6006)
```

### Dataset Management

```bash
waymo-download  # Download 5 samples per split
waymo-status    # Check what's downloaded
waymo-sample    # Quick download of 10 samples
waymo-clear     # Remove all downloaded data
```

### GPU Monitoring

```bash
gpu            # Show GPU status (nvidia-smi)
gpu-watch      # Continuously monitor GPU (updates every second)
```

### Environment Info

```bash
pyinfo         # Show Python, PyTorch, and CUDA versions
```

These aliases are defined in `post-create.sh` and automatically available in your shell.

---

## Understanding the File Structure

Here's how the devcontainer is organized:

```
.devcontainer/
├── devcontainer.json          # VS Code container configuration
├── docker-compose.yml         # Docker orchestration (GPU, volumes, environment)
├── Dockerfile                 # Image build instructions
├── requirements.base.txt      # Python packages installed during build
├── post-create.sh             # Runs after container creation (setup, aliases)
├── fix-permissions.sh         # Fixes file ownership if needed
├── scripts/
│   ├── download_waymo_sample.sh  # Waymo download utility
│   └── check_waymo_status.sh     # Dataset status checker
└── README.md                  # This file
```

**devcontainer.json** tells VS Code how to connect to the container, which extensions to install, what ports to forward, and environment variables to set.

**docker-compose.yml** defines the container service, including GPU access, volume mounts, and environment configuration.

**Dockerfile** builds the actual image with CUDA, Python, PyTorch, TensorFlow, and all the tools.

**requirements.base.txt** lists the Python packages installed during the image build (you can add project-specific packages to `requirements.txt` in the workspace root).

**post-create.sh** runs automatically after the container is created. It sets up directories, installs workspace packages, and adds convenient shell aliases.

---

## Data Persistence and Volumes

The container uses both bind mounts and named volumes:

### Bind Mounts (Live Syncing)

- **Your workspace** (`..:/workspace`) syncs live between your host and container
- Changes in either location are immediately reflected

### Named Volumes (Persistent Storage)

- **rector-cache** (`/workspace/.cache`) - Stores PyTorch models, HuggingFace caches, etc.
- **rector-venv** (`/opt/venv`) - The Python virtual environment
- **waymo-data** - Waymo dataset storage

These volumes persist even when you rebuild or delete the container.

### Managing Volumes

```bash
# See your volumes
docker volume ls | grep rector

# Check volume details
docker volume inspect rector-cache

# Backup a volume
docker run --rm -v rector-cache:/data -v $(pwd):/backup \
  ubuntu tar czf /backup/rector-cache.tar.gz -C /data .

# Restore a volume
docker run --rm -v rector-cache:/data -v $(pwd):/backup \
  ubuntu tar xzf /backup/rector-cache.tar.gz -C /data

# Remove volumes (⚠️ deletes all data)
docker volume rm rector-cache rector-venv waymo-data
```

---

## When Things Go Wrong

### Container Won't Start

First, check the basics:
```bash
# Is Docker running?
sudo systemctl status docker

# Can Docker access your GPU?
docker run --rm --gpus all nvidia/cuda:12.2.2-base-ubuntu22.04 nvidia-smi

# Check your GPU driver
nvidia-smi
```

If everything looks good but the container still won't start, try a clean rebuild:
```bash
docker compose -f .devcontainer/docker-compose.yml down -v
docker compose -f .devcontainer/docker-compose.yml build --no-cache
```

### GPU Not Working Inside Container

```bash
# Check from inside the container
echo $CUDA_VISIBLE_DEVICES    # Should show "0" or your GPU number
nvidia-smi                      # Should show your GPU
python -c "import torch; print(torch.cuda.is_available())"  # Should print True
```

If PyTorch can't see the GPU, check your NVIDIA Docker runtime installation.

### Permission Issues

If you get permission errors when accessing files:

```bash
# Inside container
bash /usr/local/bin/fix-permissions.sh

# Outside container (from host)
sudo chown -R 1000:1000 /path/to/RECTOR
```

The container runs as user ID 1000, so files should be owned by that user.

### Shared Memory Errors

If DataLoaders crash with "Bus error" or out-of-memory errors, you need more shared memory. Edit `docker-compose.yml`:

```yaml
shm_size: "8gb"  # Increase from 4gb
```

Or use fewer workers:
```python
DataLoader(..., num_workers=2)  # Reduce from default
```

### Port Conflicts

If Jupyter or TensorBoard can't start because ports are in use, change them in `devcontainer.json`:

```json
"forwardPorts": [6007, 8889]  # Instead of 6006, 8888
```

Then rebuild the container.

### Package Installation Failures

If pip fails to install packages:

```bash
# Upgrade pip itself
pip install --upgrade pip setuptools wheel

# Clear the cache
pip cache purge

# Try installing without cache
pip install --no-cache-dir package-name
```

### Waymo Download Problems

If downloads fail:

```bash
# Re-authenticate
gcloud auth revoke --all
gcloud auth login --no-browser
gcloud auth application-default login --no-browser

# Test access
gsutil ls gs://waymo_open_dataset_motion_v_1_3_0/uncompressed/scenario/training
```

Make sure you've accepted the Waymo terms at https://waymo.com/open/

---

## Best Practices

### Development Workflow

- **Commit often**: The container is disposable—your Git history is your safety net
- **Use the aliases**: `rector-train` is clearer and easier than `python scripts/train.py`
- **Monitor your GPU**: Run `gpu-watch` in a split terminal during training
- **Check the dataset**: Use `waymo-status` before starting experiments to confirm data availability

### Performance Optimization

- **Disable CUDA debug mode** for benchmarking or production training:
  ```bash
  export CUDA_LAUNCH_BLOCKING=0
  ```

- **Adjust DataLoader workers** based on your workload. For the RTX A3000, 4-8 workers is usually optimal:
  ```python
  DataLoader(..., num_workers=4, persistent_workers=True)
  ```

- **Caches are persistent**: PyTorch and HuggingFace caches are stored in volumes, so models only download once

- **Large directories are excluded**: The file watcher ignores `.cache/` and `data/` for better performance

### Code Quality

- **Black auto-formats** on save—don't fight it, embrace consistent style
- **Use type hints** so mypy can catch bugs before runtime
- **Write tests** as you go—pytest is ready to use
- **Check the linters** (flake8, ruff) regularly to catch issues early

### Git Workflow

- **SSH keys just work**—they're mounted from your host, so push and pull freely
- **Use Git LFS** for large files like model checkpoints and processed datasets
- **Check .gitignore**—cache directories are already excluded

### Dataset Management

- **Start with samples**: Use `waymo-sample` to get started quickly with a small subset
- **Check status regularly**: `waymo-status` shows what you have and how much space it's using
- **Mount from host** if you already have the full dataset (edit docker-compose.yml)
- **Use volumes** for persistence—even if you delete the container, the dataset stays

### Working with Others

- **Pin package versions** in requirements.txt (use `==` not `>=`) for reproducibility
- **Test in the container** before pushing—ensures everyone has the same environment
- **Document custom commands** if you add new aliases to post-create.sh
- **Share the container config** changes through Git so the whole team stays in sync

---

## System Requirements

Before you start, make sure you have:

### Required Software

- **Docker** ≥20.10 with Compose V2
- **NVIDIA Docker Runtime** (nvidia-docker2)
- **VS Code** with the Dev Containers extension
- **NVIDIA GPU Driver** compatible with CUDA 12.2

### Hardware Requirements

- **OS**: Linux (Ubuntu 20.04 or newer recommended)
- **GPU**: NVIDIA GPU with Compute Capability ≥5.0
- **RAM**: 16GB minimum, 32GB recommended for comfortable development
- **Disk**: 50GB free space (100GB+ if downloading full Waymo dataset)
- **Network**: Reliable internet connection for dataset downloads

### Quick Verification

```bash
# Docker and Compose
docker --version
docker compose version

# NVIDIA Docker
docker run --rm --gpus all nvidia/cuda:12.2.2-base-ubuntu22.04 nvidia-smi

# GPU Driver
nvidia-smi
```

### SSH Setup for Git

If you plan to use Git with SSH (recommended):

```bash
chmod 700 ~/.ssh
chmod 600 ~/.ssh/id_*
chmod 644 ~/.ssh/*.pub

# Test GitHub connection
ssh -T git@github.com
```

---

## Alternative: Docker Compose

If you prefer not to use VS Code, you can run the container directly with Docker Compose:

```bash
cd /path/to/RECTOR

# Build the image
docker compose -f .devcontainer/docker-compose.yml build

# Start the container
docker compose -f .devcontainer/docker-compose.yml up -d

# Enter the container
docker exec -it rector-dev-1 bash

# Stop when done
docker compose -f .devcontainer/docker-compose.yml down
```

This gives you the same environment without the VS Code integration.

---

## Summary

You now have a professional, GPU-accelerated development environment with:

✅ **Zero configuration needed**—just open and start coding  
✅ **GPU-optimized setup**—CUDA 12.2 tuned for RTX A3000  
✅ **AI-powered development**—Copilot and IntelliCode at your fingertips  
✅ **Complete reproducibility**—everyone on the team has identical environments  
✅ **Production-grade tools**—everything you need for serious research

**Build time**: 10-15 minutes (first time only)  
**Startup time**: Under 30 seconds (after initial build)  
**Team alignment**: 100% identical environments across all machines

Ready to build the future of trajectory optimization!

---

**Environment Details**  
CUDA: 12.2.2 | PyTorch: 2.2.0+cu121 | Python: 3.10 | Image: rector-dev:latest

*Last updated: November 9, 2025*
