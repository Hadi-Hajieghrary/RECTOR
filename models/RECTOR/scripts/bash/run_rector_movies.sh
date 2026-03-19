#!/bin/bash
#
# RECTOR Movie Generator Pipeline
#
# Generate visualization movies showing RECTOR planned trajectories.
#
# Usage:
#   ./run_rector_movies.sh [OPTIONS]
#
# Options:
#   -n, --num_scenarios    Number of scenarios to process (default: 3)
#   --start_t              First prediction timestep (default: 10)
#   --end_t                Last prediction timestep (default: 80)
#   --step                 Timestep increment (default: 1, use 5 for fast preview)
#   -o, --output_dir       Output directory (default: /workspace/models/RECTOR/movies)
#   -h, --help             Show this help message
#
# Examples:
#   # Quick preview: 3 scenarios, every 5th timestep
#   ./run_rector_movies.sh -n 3 --step 5
#
#   # Full quality: 5 scenarios, every timestep
#   ./run_rector_movies.sh -n 5 --start_t 10 --end_t 80 --step 1
#
#   # Single scenario for debugging
#   ./run_rector_movies.sh -n 1 --step 2
#

set -e

# Default values
NUM_SCENARIOS=3
DATA_DIR="/workspace/data/datasets/waymo_open_dataset/motion_v_1_3_0/processed/tf/validation_interactive"
OUTPUT_DIR="/workspace/models/RECTOR/movies"
START_T=0
END_T=90
STEP=1
FPS=10

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--num_scenarios)
            NUM_SCENARIOS="$2"
            shift 2
            ;;
        -d|--data_dir)
            DATA_DIR="$2"
            shift 2
            ;;
        -o|--output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --start_t)
            START_T="$2"
            shift 2
            ;;
        --end_t)
            END_T="$2"
            shift 2
            ;;
        --step)
            STEP="$2"
            shift 2
            ;;
        --fps)
            FPS="$2"
            shift 2
            ;;
        -h|--help)
            head -28 "$0" | tail -25
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Print banner
echo "============================================================"
echo "RECTOR MOVIE GENERATOR"
echo "============================================================"
echo ""
echo "Configuration:"
echo "  Number of scenarios:     $NUM_SCENARIOS"
echo "  Data directory:          $DATA_DIR"
echo "  Output directory:        $OUTPUT_DIR"
echo "  Timestep range:          $START_T to $END_T (step=$STEP)"
echo "  FPS:                     $FPS"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run movie generator
echo "============================================================"
echo "GENERATING RECTOR PLANNING MOVIES"
echo "============================================================"
echo ""

START_TIME=$(date +%s)

cd "$SCRIPT_DIR/../lib"

python generate_rector_movies.py \
    --tfrecord "$DATA_DIR" \
    --num_scenarios "$NUM_SCENARIOS" \
    --start_t "$START_T" \
    --end_t "$END_T" \
    --step "$STEP" \
    --output_dir "$OUTPUT_DIR" \
    --fps "$FPS"

TOTAL_TIME=$(($(date +%s) - START_TIME))

# Summary
echo ""
echo "============================================================"
echo "COMPLETE"
echo "============================================================"
echo ""
echo "Timing:"
echo "  Total time:          ${TOTAL_TIME}s"
echo ""
echo "Output location:"
echo "  Movies:      $OUTPUT_DIR"
echo ""

# Count outputs
NUM_MOVIES=$(find "$OUTPUT_DIR" -name "*.mp4" 2>/dev/null | wc -l)
NUM_GIFS=$(find "$OUTPUT_DIR" -name "*.gif" 2>/dev/null | wc -l)

echo "Generated files:"
echo "  MP4 movies: $NUM_MOVIES"
echo "  GIF movies: $NUM_GIFS"
echo ""

# List generated movies
if [ "$NUM_MOVIES" -gt 0 ]; then
    echo "Generated movies:"
    ls -1 "$OUTPUT_DIR"/*.mp4 2>/dev/null | head -10
fi

echo ""
echo "Done!"
