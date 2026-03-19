#!/bin/bash
# Generate static visualizations from Waymo scenario data
# Usage: ./generate_visualizations.sh [NUM_SCENARIOS]

NUM=${1:-5}
SCRIPT="/workspace/data/WOMD/scripts/lib/visualize_scenario.py"
PYTHON="/opt/venv/bin/python"

echo "========================================"
echo "Generating Static Visualizations"
echo "========================================"
echo "Scenarios per split: $NUM"
echo ""

# Generate from scenario format with all visualization types
echo "--- Scenario Format ---"
$PYTHON $SCRIPT --format scenario --split validation_interactive --num $NUM --multi-frame --combined
$PYTHON $SCRIPT --format scenario --split testing_interactive --num $NUM --multi-frame --combined

# Generate from tf format
echo ""
echo "--- TF Format ---"
$PYTHON $SCRIPT --format tf --split training_interactive --num $NUM

echo ""
echo "========================================"
echo "Done! Visualizations saved to /workspace/data/WOMD/visualizations/"
echo "========================================"
