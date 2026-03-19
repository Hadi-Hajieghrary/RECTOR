#!/bin/bash
# Generate BEV movies from Waymo scenario data
# Usage: ./generate_movies.sh [NUM_SCENARIOS]

NUM=${1:-5}
SCRIPT="/workspace/data/WOMD/scripts/lib/generate_bev_movie.py"
PYTHON="/opt/venv/bin/python"

echo "========================================"
echo "Generating BEV Movies"
echo "========================================"
echo "Scenarios per split: $NUM"
echo ""

# Generate from scenario format
echo "--- Scenario Format ---"
$PYTHON $SCRIPT --format scenario --split training_interactive --num $NUM
$PYTHON $SCRIPT --format scenario --split validation_interactive --num $NUM
$PYTHON $SCRIPT --format scenario --split testing_interactive --num $NUM

# Generate from tf format
echo ""
echo "--- TF Format ---"
$PYTHON $SCRIPT --format tf --split training_interactive --num $NUM
$PYTHON $SCRIPT --format tf --split validation_interactive --num $NUM
$PYTHON $SCRIPT --format tf --split testing_interactive --num $NUM

echo ""
echo "========================================"
echo "Done! Movies saved to /workspace/data/WOMD/movies/bev/"
echo "========================================"
