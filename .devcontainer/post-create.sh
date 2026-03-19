#!/usr/bin/env bash
set -euo pipefail

echo "🚀 RECTOR Post-Create Setup Starting..."

# Make scripts executable
chmod +x .devcontainer/*.sh 2>/dev/null || true
chmod +x .devcontainer/scripts/*.sh 2>/dev/null || true
chmod +x scripts/*.sh 2>/dev/null || true
chmod +x scripts/WOMD/*.sh 2>/dev/null || true

# Setup external repositories (M2I and Waymo Open Dataset)
echo ""
if [ -f ".devcontainer/scripts/setup-externals.sh" ]; then
    bash .devcontainer/scripts/setup-externals.sh
fi
echo ""

# Setup Git hooks
echo "🔧 Setting up Git hooks..."
if [ -f ".devcontainer/scripts/setup_git_hooks.sh" ]; then
    bash .devcontainer/scripts/setup_git_hooks.sh
else
    echo "⚠️  Git hooks setup script not found"
fi
echo ""

# Create directory structure
echo "📁 Creating directory structure..."
mkdir -p \
    data/WOMD/datasets/waymo_open_dataset/motion_v_1_3_0/raw \
    data/WOMD/datasets/waymo_open_dataset/motion_v_1_3_0/processed \
    data/WOMD/movies \
    models/checkpoints \
    models/pretrained \
    output \
    output/closedloop/figures \
    output/closedloop/videos \
    output/closedloop/traces \
    output/closedloop/mining \
    output/closedloop/ws_tuning \
    logs \
    notebooks \
    externals \
    .cache/torch \
    .cache/huggingface \
    .cache/jax

# Install project requirements if available
if [ -f "requirements.txt" ]; then
    echo "📦 Installing project requirements..."
    pip install --no-cache-dir -r requirements.txt || {
        echo "⚠️  Some requirements failed to install"
        echo "Attempting fallback with tensorflow-cpu..."
        sed 's/^tensorflow==2\.11\.0/tensorflow-cpu==2.11.0/g' requirements.txt > /tmp/requirements_fallback.txt
        pip install --no-cache-dir -r /tmp/requirements_fallback.txt || echo "⚠️  Please check requirements manually"
    }
fi

# Install project in editable mode
if [ -f "setup.py" ] || [ -f "pyproject.toml" ]; then
    echo "📦 Installing project in editable mode..."
    pip install -e . || echo "⚠️  Project installation failed (continue anyway)"
fi

# Add useful bash aliases (only if not already added)
if ! grep -q "RECTOR Development Shortcuts" ~/.bashrc 2>/dev/null; then
echo "⚙️  Adding helpful aliases..."
cat >> ~/.bashrc <<'EOF'

# RECTOR Development Shortcuts
# Note: These scripts are not yet created; uncomment when available
# alias rector-train='python scripts/train.py'
# alias rector-eval='python scripts/validate.py'
# alias rector-viz='python scripts/visualize.py'
alias jlab='jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root'

# Waymo dataset helpers
alias waymo-download='bash scripts/WOMD/download_waymo_sample.sh'
alias waymo-status='bash scripts/WOMD/check_waymo_status.sh'
alias waymo-sample='NUM_SAMPLE_FILES=10 bash scripts/WOMD/download_waymo_sample.sh'

# Quick GPU check
alias gpu='nvidia-smi'
alias gpu-watch='watch -n 1 nvidia-smi'

# Python environment info
alias pyinfo='python -c "import torch; print(f\"PyTorch: {torch.__version__}\"); print(f\"CUDA: {torch.cuda.is_available()}\"); print(f\"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}\")"'

# TensorBoard shortcut
alias tb='tensorboard --logdir=output --host=0.0.0.0 --port=6006'

# Closed-loop simulation shortcuts
alias rector-validate-waymax='python .devcontainer/scripts/validate-waymax.py'
alias rector-mine='python -m scripts.simulation_engine.run_mining'
alias rector-closedloop='python -m scripts.simulation_engine.run_experiment'
alias rector-analyze='python -m scripts.simulation_engine.run_analysis'

EOF
else
echo "⚙️  Aliases already configured, skipping..."
fi

# Run GPU and environment checks
echo ""
echo "🔍 Checking environment..."
python - <<'PYCHECK'
import sys
print(f"Python: {sys.version.split()[0]}")

try:
    import torch
    print(f"✓ PyTorch: {torch.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✓ CUDA version: {torch.version.cuda}")
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
except ImportError:
    print("✗ PyTorch not installed")

try:
    import tensorflow as tf
    print(f"✓ TensorFlow: {tf.__version__}")
except ImportError:
    print("✗ TensorFlow not installed")

try:
    import waymo_open_dataset
    print("✓ Waymo Open Dataset API installed")
except ImportError:
    print("✗ Waymo Open Dataset API not installed")

try:
    import jax
    print(f"✓ JAX: {jax.__version__}")
    devices = jax.devices()
    for d in devices:
        print(f"  Device: {d.device_kind} ({d.platform})")
except ImportError:
    print("✗ JAX not installed (required for Waymax closed-loop simulation)")
except Exception as e:
    print(f"⚠ JAX installed but device init failed: {e}")

try:
    import waymax
    print("✓ Waymax (closed-loop simulator) installed")
except ImportError:
    print("✗ Waymax not installed — run: pip install git+https://github.com/waymo-research/waymax.git@main")
except Exception as e:
    print(f"⚠ Waymax import error: {e}")

PYCHECK

# Check LaTeX tools
echo ""
echo "🔍 Checking LaTeX tools..."
pdflatex --version 2>/dev/null | head -1 && echo "✓ pdflatex installed" || echo "✗ pdflatex not found"
biber --version 2>/dev/null | head -1 && echo "✓ biber installed" || echo "✗ biber not found"
latexmk --version 2>/dev/null | head -1 && echo "✓ latexmk installed" || echo "✗ latexmk not found"

# Check for Waymo dataset
echo ""
if [ -d "data/WOMD/datasets/waymo_open_dataset/motion_v_1_3_0/raw" ]; then
    echo "📊 Checking Waymo dataset..."
    bash scripts/WOMD/check_waymo_status.sh 2>/dev/null || echo "ℹ️  Run 'waymo-status' to check dataset"
else
    echo "ℹ️  Waymo dataset not found. Use 'waymo-download' to download data."
fi

echo ""
echo "✅ Post-create setup complete!"
echo ""
echo "🎯 Quick Start:"
echo "  • Download Waymo data: waymo-download"
echo "  • Check status: waymo-status"
echo "  • Validate Waymax: rector-validate-waymax"
echo "  • Start JupyterLab: jlab"
echo "  • Start TensorBoard: tb"
echo "  • GPU info: gpu or pyinfo"
echo "  • Run closed-loop: rector-closedloop"
echo ""

# Fix permissions
bash /usr/local/bin/fix-permissions.sh || true
