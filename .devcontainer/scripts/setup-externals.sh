#!/usr/bin/env bash
set -euo pipefail

echo "🔧 Setting up external repositories..."

# Create externals directory
EXTERNALS_DIR="/workspace/externals"
mkdir -p "$EXTERNALS_DIR"

# M2I Repository
M2I_REPO="https://github.com/Tsinghua-MARS-Lab/M2I.git"
M2I_DIR="$EXTERNALS_DIR/M2I"

if [ ! -d "$M2I_DIR" ]; then
    echo "📦 Cloning M2I repository..."
    git clone "$M2I_REPO" "$M2I_DIR" || {
        echo "⚠️  Failed to clone M2I repository"
        exit 1
    }
else
    echo "✓ M2I repository already exists"
fi

# Waymo Open Dataset Repository
WAYMO_REPO="https://github.com/waymo-research/waymo-open-dataset.git"
WAYMO_DIR="$EXTERNALS_DIR/waymo-open-dataset"

if [ ! -d "$WAYMO_DIR" ]; then
    echo "📦 Cloning Waymo Open Dataset repository..."
    git clone "$WAYMO_REPO" "$WAYMO_DIR" || {
        echo "⚠️  Failed to clone Waymo Open Dataset repository"
        exit 1
    }
else
    echo "✓ Waymo Open Dataset repository already exists"
fi

# Install M2I
echo ""
echo "📦 Installing M2I..."
cd "$M2I_DIR"

# Check if M2I has requirements
if [ -f "requirements.txt" ]; then
    echo "  Installing M2I requirements (skipping incompatible packages)..."
    # Filter out packages incompatible with our environment
    # We already have: tensorflow 2.11, waymo-open-dataset-tf-2-11-0
    grep -vE "tensorflow-gpu|waymo-open-dataset-tf-2-4-0" requirements.txt > /tmp/m2i_requirements_filtered.txt || true
    if [ -s /tmp/m2i_requirements_filtered.txt ]; then
        pip install --no-cache-dir -r /tmp/m2i_requirements_filtered.txt || {
            echo "⚠️  Some M2I requirements failed to install (continuing anyway)"
        }
    fi
    rm -f /tmp/m2i_requirements_filtered.txt
    echo "  Note: Using TensorFlow 2.11 and waymo-open-dataset-tf-2-11-0 from base environment"
fi

# Install M2I in editable mode
if [ -f "setup.py" ]; then
    echo "  Installing M2I package..."
    pip install -e . || {
        echo "⚠️  M2I installation failed"
    }
elif [ -f "pyproject.toml" ]; then
    echo "  Installing M2I package..."
    pip install -e . || {
        echo "⚠️  M2I installation failed"
    }
fi

# Build M2I Cython extensions if available
if [ -d "src" ]; then
    echo "  Building M2I Cython extensions..."
    cd src
    if [ -f "setup.py" ]; then
        python setup.py build_ext --inplace 2>/dev/null || {
            echo "⚠️  M2I Cython build failed (optional - will use Python fallback)"
        }
    fi
    cd "$M2I_DIR"
fi

# Install Waymo Open Dataset tools
echo ""
echo "📦 Installing Waymo Open Dataset tools..."
cd "$WAYMO_DIR"

# Check if waymo has requirements
if [ -f "requirements.txt" ]; then
    echo "  Installing Waymo requirements (skipping incompatible packages)..."
    # Filter out specific tensorflow/waymo versions that conflict with our environment
    grep -vE "tensorflow-gpu|tensorflow==|waymo-open-dataset-tf" requirements.txt > /tmp/waymo_requirements_filtered.txt || true
    if [ -s /tmp/waymo_requirements_filtered.txt ]; then
        pip install --no-cache-dir -r /tmp/waymo_requirements_filtered.txt || {
            echo "⚠️  Some Waymo requirements failed to install (continuing anyway)"
        }
    fi
    rm -f /tmp/waymo_requirements_filtered.txt
    echo "  Note: Using TensorFlow 2.11 and waymo-open-dataset-tf-2-11-0 from base environment"
fi

# Install waymo package
if [ -f "setup.py" ]; then
    echo "  Installing Waymo package..."
    pip install -e . || {
        echo "⚠️  Waymo installation failed (may need manual setup)"
    }
elif [ -f "pyproject.toml" ]; then
    echo "  Installing Waymo package..."
    pip install -e . || {
        echo "⚠️  Waymo installation failed (may need manual setup)"
    }
fi

# Return to workspace
cd /workspace

echo ""
echo "✅ External repositories setup complete!"
echo "  • M2I: $M2I_DIR"
echo "  • Waymo Open Dataset: $WAYMO_DIR"
echo ""
