#!/bin/bash
# Generate visualization movies from Waymo data
# Usage: bash/visualize.sh [--format FORMAT] [--split SPLIT] [--num N] [--fps FPS]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASETS_ROOT="$(dirname "$SCRIPT_DIR")/datasets/waymo_open_dataset/motion_v_1_3_0"
DATA_ROOT="${WAYMO_RAW_ROOT:-$DATASETS_ROOT/raw}"
MOVIES_ROOT="${WAYMO_MOVIES_ROOT:-$(dirname "$SCRIPT_DIR")/movies/waymo_open_dataset/motion_v_1_3_0}"

# Default parameters
FORMAT="scenario"
SPLIT="training"
NUM_SCENARIOS=10
FPS=10

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --format)
            FORMAT="$2"
            shift 2
            ;;
        --split)
            SPLIT="$2"
            shift 2
            ;;
        --num)
            NUM_SCENARIOS="$2"
            shift 2
            ;;
        --fps)
            FPS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "========================================"
echo "Waymo Visualization"
echo "========================================"
echo "Format: $FORMAT"
echo "Split: $SPLIT"
echo "Scenarios: $NUM_SCENARIOS"
echo "FPS: $FPS"
echo ""

# Check input exists
INPUT_DIR="$DATA_ROOT/$FORMAT/$SPLIT"
if [ ! -d "$INPUT_DIR" ]; then
    echo "ERROR: Input directory not found: $INPUT_DIR"
    exit 1
fi

# Create output directory
OUTPUT_DIR="$MOVIES_ROOT/$FORMAT/$SPLIT"
mkdir -p "$OUTPUT_DIR"

# Choose visualization script
if [ "$FORMAT" == "scenario" ]; then
    VIZ_SCRIPT="$SCRIPT_DIR/lib/viz_waymo_scenario.py"
else
    VIZ_SCRIPT="$SCRIPT_DIR/lib/viz_waymo_tfexample.py"
fi

# Run visualization
python "$VIZ_SCRIPT" \
    --data_dir "$INPUT_DIR" \
    --num_scenarios "$NUM_SCENARIOS" \
    --output_dir "$OUTPUT_DIR" \
    --fps "$FPS"

echo ""
echo "Visualization complete!"
echo "Movies: $OUTPUT_DIR"
find "$OUTPUT_DIR" -name "*.mp4" 2>/dev/null | wc -l | xargs echo "  MP4 files:"
find "$OUTPUT_DIR" -name "*.gif" 2>/dev/null | wc -l | xargs echo "  GIF files:"
