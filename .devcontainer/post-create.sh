#!/usr/bin/env bash
set -euo pipefail

echo "🚀 RECTOR Post-Create Setup Starting..."

# Make scripts executable
chmod +x .devcontainer/*.sh 2>/dev/null || true
chmod +x .devcontainer/scripts/*.sh 2>/dev/null || true
chmod +x scripts/*.sh 2>/dev/null || true
chmod +x scripts/waymo/*.sh 2>/dev/null || true

# Setup external repositories (M2I and Waymo Open Dataset)
echo ""
if [ -f ".devcontainer/scripts/setup-externals.sh" ]; then
    bash .devcontainer/scripts/setup-externals.sh
fi
echo ""

# Create directory structure
echo "📁 Creating directory structure..."
mkdir -p \
    data/datasets/waymo_open_dataset/motion_v_1_3_0/raw \
    data/datasets/waymo_open_dataset/motion_v_1_3_0/processed \
    data/cache \
    models/checkpoints \
    models/pretrained \
    output \
    logs \
    notebooks \
    .cache/torch \
    .cache/huggingface

# Install project requirements if available
if [ -f "requirements.txt" ]; then
    echo "📦 Installing project requirements..."
    pip install --no-cache-dir -r requirements.txt || {
        echo "⚠️  Some requirements failed to install"
        echo "Attempting fallback with tensorflow-cpu..."
        sed -i 's/^tensorflow==2\.11\.0/tensorflow-cpu==2.11.0/g' requirements.txt 2>/dev/null || true
        pip install --no-cache-dir -r requirements.txt || echo "⚠️  Please check requirements manually"
    }
fi

# Install project in editable mode
if [ -f "setup.py" ] || [ -f "pyproject.toml" ]; then
    echo "📦 Installing project in editable mode..."
    pip install -e . || echo "⚠️  Project installation failed (continue anyway)"
fi

# Add useful bash aliases
echo "⚙️  Adding helpful aliases..."
cat >> ~/.bashrc <<'EOF'

# RECTOR Development Shortcuts
alias rector-train='python scripts/train.py'
alias rector-eval='python scripts/validate.py'
alias rector-viz='python scripts/visualize.py'
alias jlab='jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root'

# Waymo dataset helpers
alias waymo-download='bash .devcontainer/scripts/download_waymo_sample.sh'
alias waymo-status='bash .devcontainer/scripts/check_waymo_status.sh'
alias waymo-sample='NUM_SAMPLE_FILES=10 bash .devcontainer/scripts/download_waymo_sample.sh'

# Quick GPU check
alias gpu='nvidia-smi'
alias gpu-watch='watch -n 1 nvidia-smi'

# Python environment info
alias pyinfo='python -c "import torch; print(f\"PyTorch: {torch.__version__}\"); print(f\"CUDA: {torch.cuda.is_available()}\"); print(f\"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}\")"'

EOF

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

PYCHECK

# Check for Waymo dataset
echo ""
if [ -d "data/datasets/waymo_open_dataset/motion_v_1_3_0/raw" ]; then
    echo "📊 Checking Waymo dataset..."
    ./data/scripts/waymo status 2>/dev/null || echo "ℹ️  Run './data/scripts/waymo status' to check dataset"
else
    echo "ℹ️  Waymo dataset not found. Use './data/scripts/waymo download' to download data."
fi

echo ""
echo "✅ Post-create setup complete!"
echo ""
echo "🎯 Quick Start:"
echo "  • Download Waymo data: waymo-download"
echo "  • Check status: waymo-status"
echo "  • Start JupyterLab: jlab"
echo "  • Start TensorBoard: tb"
echo "  • GPU info: gpu or pyinfo"
echo ""
