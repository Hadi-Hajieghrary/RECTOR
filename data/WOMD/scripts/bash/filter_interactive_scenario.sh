#!/bin/bash
# Wrapper script to run the interactive scenario filter
# Usage: ./filter_interactive_scenario.sh [INPUT_DIR] [OUTPUT_DIR]

# Default paths
INPUT_DIR=${1:-"/workspace/data/WOMD/datasets/waymo_open_dataset/motion_v_1_3_0/raw/scenario/training_20s"}
OUTPUT_DIR=${2:-"/workspace/data/WOMD/datasets/waymo_open_dataset/motion_v_1_3_0/raw/scenario/training_20s_interactive"}

# Script path
SCRIPT_PATH="/workspace/data/WOMD/scripts/lib/filter_interactive_scenario.py"
PYTHON="/opt/venv/bin/python"

if [ ! -f "$PYTHON" ]; then
    PYTHON="python3"
fi

echo "Running Interactive Scenario Filter..."
echo "Input:  $INPUT_DIR"
echo "Output: $OUTPUT_DIR"
echo "Script: $SCRIPT_PATH"
echo ""

"$PYTHON" "$SCRIPT_PATH" \
    --input-dir "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --num-workers 8

# Check exit code
if [ $? -eq 0 ]; then
    echo "Success!"
else
    echo "Failed!"
    exit 1
fi
