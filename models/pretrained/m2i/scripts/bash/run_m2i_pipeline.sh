#!/bin/bash
#
# M2I Receding Horizon Pipeline: Inference + Movie Generation
#
# This script runs the receding horizon M2I pipeline:
# 1. DenseTNT marginal prediction at each timestep (receding horizon)
# 2. Generate visualization movies showing predictions evolving over time
#
# The receding horizon approach simulates real-time autonomous vehicle operation:
# - At each timestep t, uses observations [t-10 : t] to predict [t+1 : t+80]
# - Shows how predictions improve as more observations become available
#
# Usage:
#   ./run_m2i_pipeline.sh [OPTIONS]
#
# Options:
#   -n, --num_scenarios    Number of scenarios to process (default: 10)
#   -d, --data_dir         Input data directory (default: validation_interactive)
#   -o, --output_dir       Output directory for predictions (default: /workspace/output/m2i_live/receding_horizon)
#   -m, --movies_dir       Output directory for movies (default: /workspace/models/pretrained/m2i/movies/receding_horizon)
#   --start_t              First prediction timestep (default: 10)
#   --end_t                Last prediction timestep (default: 90)
#   --step                 Timestep increment (default: 1 for full resolution)
#   --device               Device to use: cuda or cpu (default: cuda)
#   --subprocess-pipeline  Use subprocess 3-stage pipeline (DenseTNT + Relation + Conditional)
#   --no-movies            Skip movie generation
#   -h, --help             Show this help message
#
# Examples:
#   # Quick test: 2 scenarios with movies
#   ./run_m2i_pipeline.sh -n 2
#
#   # Full run: 50 scenarios, full timestep range
#   ./run_m2i_pipeline.sh -n 50 --start_t 10 --end_t 90 --step 1
#
#   # Fast preview: larger steps (fewer predictions per scenario)
#   ./run_m2i_pipeline.sh -n 10 --step 5
#
#   # Full 3-stage pipeline with subprocess isolation (recommended)
#   ./run_m2i_pipeline.sh -n 10 --subprocess-pipeline
#
# Or from workspace root:
#   bash models/pretrained/m2i/scripts/bash/run_m2i_pipeline.sh -n 10
#

set -e  # Exit on error

# Default values
NUM_SCENARIOS=10
DATA_DIR="/workspace/data/datasets/waymo_open_dataset/motion_v_1_3_0/processed/tf/validation_interactive"
OUTPUT_DIR="/workspace/output/m2i_live/receding_horizon"
MOVIES_DIR="/workspace/models/pretrained/m2i/movies/receding_horizon"
START_T=10
END_T=90
STEP=1
DEVICE="cuda"
SUBPROCESS_PIPELINE=""
GENERATE_MOVIES="--generate_movies"

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
        -m|--movies_dir)
            MOVIES_DIR="$2"
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
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --subprocess-pipeline)
            SUBPROCESS_PIPELINE="--subprocess-pipeline"
            shift
            ;;
        --no-movies)
            GENERATE_MOVIES=""
            shift
            ;;
        -h|--help)
            head -47 "$0" | tail -44
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Find the first TFRecord file in the data directory
TFRECORD=$(find "$DATA_DIR" -name "*.tfrecord*" | head -1)
if [ -z "$TFRECORD" ]; then
    echo "ERROR: No TFRecord files found in: $DATA_DIR"
    exit 1
fi

# Print banner
echo "============================================================"
echo "M2I RECEDING HORIZON PIPELINE"
echo "============================================================"
echo ""
echo "Configuration:"
echo "  Number of scenarios:     $NUM_SCENARIOS"
echo "  TFRecord file:           $(basename "$TFRECORD")"
echo "  Data directory:          $DATA_DIR"
echo "  Output directory:        $OUTPUT_DIR"
echo "  Movies directory:        $MOVIES_DIR"
echo "  Timestep range:          $START_T to $END_T (step=$STEP)"
echo "  Device:                  $DEVICE"
if [ -n "$SUBPROCESS_PIPELINE" ]; then
    echo "  Pipeline:                3-stage subprocess (DenseTNT + Relation + Conditional)"
else
    echo "  Pipeline:                DenseTNT only (marginal prediction)"
fi
if [ -n "$GENERATE_MOVIES" ]; then
    echo "  Movie generation:        Enabled"
else
    echo "  Movie generation:        Disabled"
fi
echo ""

# Create output directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$MOVIES_DIR"

# Change to lib directory (where Python scripts are)
cd "$SCRIPT_DIR/../lib"

# Run Receding Horizon Pipeline
echo "============================================================"
echo "RUNNING RECEDING HORIZON INFERENCE"
echo "============================================================"
echo ""

START_TIME=$(date +%s)

python m2i_receding_horizon_full.py \
    --tfrecord "$TFRECORD" \
    --num_scenarios "$NUM_SCENARIOS" \
    --start_t "$START_T" \
    --end_t "$END_T" \
    --step "$STEP" \
    --output "$OUTPUT_DIR/predictions.pickle" \
    --movies_dir "$MOVIES_DIR" \
    --device "$DEVICE" \
    $SUBPROCESS_PIPELINE \
    $GENERATE_MOVIES

TOTAL_TIME=$(($(date +%s) - START_TIME))

# Summary
echo ""
echo "============================================================"
echo "PIPELINE COMPLETE"
echo "============================================================"
echo ""
echo "Timing:"
echo "  Total time:          ${TOTAL_TIME}s"
echo ""
echo "Output locations:"
echo "  Predictions: $OUTPUT_DIR/predictions.pickle"
if [ -n "$GENERATE_MOVIES" ]; then
    echo "  Movies:      $MOVIES_DIR"
fi
echo ""

# Count outputs
if [ -n "$GENERATE_MOVIES" ]; then
    NUM_MOVIES=$(find "$MOVIES_DIR" -name "*.mp4" 2>/dev/null | wc -l)
    NUM_GIFS=$(find "$MOVIES_DIR" -name "*.gif" 2>/dev/null | wc -l)

    echo "Generated files:"
    echo "  MP4 movies: $NUM_MOVIES"
    echo "  GIF movies: $NUM_GIFS"
    echo ""

    # List generated movies
    if [ "$NUM_MOVIES" -gt 0 ]; then
        echo "Generated movies:"
        ls -1 "$MOVIES_DIR"/*.mp4 2>/dev/null | head -10
        if [ "$NUM_MOVIES" -gt 10 ]; then
            echo "  ... and $((NUM_MOVIES - 10)) more"
        fi
    fi
fi

echo ""
echo "Done!"
