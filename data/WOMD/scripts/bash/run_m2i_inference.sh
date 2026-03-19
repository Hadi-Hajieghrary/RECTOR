#!/bin/bash
#
# Run M2I Trajectory Prediction Inference
#
# This script runs the M2I pipeline to predict trajectories for
# interacting agents in Waymo Motion Dataset scenarios.
#
# Usage:
#   ./run_m2i_inference.sh [split] [num_scenarios] [mode]
#
# Arguments:
#   split          - Dataset split: validation_interactive (default), training_interactive, testing_interactive
#   num_scenarios  - Number of scenarios to process (default: 10)
#   mode           - Prediction mode: precomputed (default), baseline, full
#
# Examples:
#   ./run_m2i_inference.sh                                    # Default: validation, 10 scenarios
#   ./run_m2i_inference.sh validation_interactive 100         # 100 validation scenarios
#   ./run_m2i_inference.sh training_interactive 50 baseline   # 50 training with baseline predictor

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="/workspace"
PYTHON="/opt/venv/bin/python"

# Resolve script path from candidate locations
resolve_script() {
    for candidate in "$@"; do
        if [ -f "$candidate" ]; then
            echo "$candidate"
            return 0
        fi
    done
    return 1
}

# M2I inference script locations (prefer local model checkout, fallback to externals)
M2I_SCRIPT="$(resolve_script \
    "${WORKSPACE_ROOT}/models/pretrained/m2i/run_inference.py" \
    "${WORKSPACE_ROOT}/externals/M2I/scripts/run_inference.py" \
    "${WORKSPACE_ROOT}/externals/M2I/src/run_inference.py" \
    || true)"

VIZ_SCRIPT="$(resolve_script \
    "${WORKSPACE_ROOT}/models/pretrained/m2i/visualize_predictions.py" \
    "${WORKSPACE_ROOT}/externals/M2I/scripts/visualize_predictions.py" \
    "${WORKSPACE_ROOT}/externals/M2I/src/visualize_predictions.py" \
    || true)"

# Pre-flight checks
if [ ! -f "$PYTHON" ]; then
    echo "WARNING: Python interpreter not found at $PYTHON"
    echo "Falling back to system python3"
    PYTHON="python3"
fi

if [ -z "$M2I_SCRIPT" ]; then
    echo "WARNING: M2I inference script not found in expected locations"
    echo "Checked:"
    echo "  - ${WORKSPACE_ROOT}/models/pretrained/m2i/run_inference.py"
    echo "  - ${WORKSPACE_ROOT}/externals/M2I/scripts/run_inference.py"
    echo "  - ${WORKSPACE_ROOT}/externals/M2I/src/run_inference.py"
    echo "Skipping M2I inference. Ensure models/pretrained/m2i/ is set up."
    exit 0
fi

# Data paths
DATA_BASE="${WORKSPACE_ROOT}/data/WOMD/datasets/waymo_open_dataset/motion_v_1_3_0/processed/tf"

# Parse arguments
SPLIT="${1:-validation_interactive}"
NUM_SCENARIOS="${2:-10}"
MODE="${3:-precomputed}"

# Validate split
case "$SPLIT" in
    training_interactive|validation_interactive|testing_interactive)
        ;;
    *)
        echo "ERROR: Invalid split '$SPLIT'"
        echo "Valid splits: training_interactive, validation_interactive, testing_interactive"
        exit 1
        ;;
esac

# Validate mode
case "$MODE" in
    precomputed|baseline|full)
        ;;
    *)
        echo "ERROR: Invalid mode '$MODE'"
        echo "Valid modes: precomputed, baseline, full"
        exit 1
        ;;
esac

# Set data directory
DATA_DIR="${DATA_BASE}/${SPLIT}"

# Set output directory
OUTPUT_DIR="${WORKSPACE_ROOT}/output/m2i_predictions/${SPLIT}"

echo "========================================================================"
echo "M2I Trajectory Prediction Inference"
echo "========================================================================"
echo ""
echo "Configuration:"
echo "  Split:          ${SPLIT}"
echo "  Data directory: ${DATA_DIR}"
echo "  Output:         ${OUTPUT_DIR}"
echo "  Num scenarios:  ${NUM_SCENARIOS}"
echo "  Mode:           ${MODE}"
echo ""

# Check data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: Data directory not found: $DATA_DIR"
    echo ""
    echo "Available splits:"
    ls -la "$DATA_BASE" 2>/dev/null || echo "  Data base not found: $DATA_BASE"
    exit 1
fi

# Check for TFRecord files
NUM_FILES=$(ls -1 "$DATA_DIR"/*.tfrecord* 2>/dev/null | wc -l)
if [ "$NUM_FILES" -eq 0 ]; then
    echo "ERROR: No TFRecord files found in $DATA_DIR"
    exit 1
fi
echo "Found ${NUM_FILES} TFRecord files"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run inference
echo "Running M2I inference..."
echo ""

"$PYTHON" "$M2I_SCRIPT" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --num_scenarios "$NUM_SCENARIOS" \
    --mode "$MODE"

INFERENCE_STATUS=$?

if [ $INFERENCE_STATUS -eq 0 ]; then
    echo ""
    echo "========================================================================"
    echo "Inference Complete!"
    echo "========================================================================"
    echo ""
    echo "Predictions saved to: ${OUTPUT_DIR}/predictions.pkl"
    echo ""

    # Prompt for visualization
    read -p "Generate visualizations? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        if [ -z "$VIZ_SCRIPT" ]; then
            echo "WARNING: Visualization script not found in expected locations"
            echo "Skipping visualization."
        else
        VIZ_OUTPUT="${OUTPUT_DIR}/visualizations"
        echo ""
        echo "Generating visualizations..."

        "$PYTHON" "$VIZ_SCRIPT" \
            --predictions "${OUTPUT_DIR}/predictions.pkl" \
            --output_dir "$VIZ_OUTPUT" \
            --num_scenarios 10 \
            --mode both

        echo ""
        echo "Visualizations saved to: $VIZ_OUTPUT"
        fi
    fi
else
    echo ""
    echo "ERROR: Inference failed with status $INFERENCE_STATUS"
    exit $INFERENCE_STATUS
fi

echo ""
echo "Done!"
