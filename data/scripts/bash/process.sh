#!/bin/bash
# Process Waymo data into preprocessed format
# Usage: bash/process.sh [--split SPLIT] [--format FORMAT] [--workers N]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASETS_ROOT="$(dirname "$SCRIPT_DIR")/datasets/waymo_open_dataset/motion_v_1_3_0"
DATA_ROOT="${WAYMO_RAW_ROOT:-$DATASETS_ROOT/raw}"
PROCESSED_ROOT="${WAYMO_PROCESSED_ROOT:-$DATASETS_ROOT/processed}"

# Default parameters
SPLIT="training"
FORMAT="scenario"
WORKERS=8
INTERACTIVE_ONLY=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --split)
            SPLIT="$2"
            shift 2
            ;;
        --format)
            FORMAT="$2"
            shift 2
            ;;
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        --interactive-only)
            INTERACTIVE_ONLY="--interactive-only"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "========================================"
echo "Waymo Data Processing"
echo "========================================"
echo "Input:  $DATA_ROOT/$FORMAT/$SPLIT"
echo "Output: $PROCESSED_ROOT/itp_$SPLIT"
echo "Workers: $WORKERS"
echo ""

# Check input exists
if [ ! -d "$DATA_ROOT/$FORMAT/$SPLIT" ]; then
    echo "ERROR: Input directory not found!"
    exit 1
fi

# Create output directory
mkdir -p "$PROCESSED_ROOT/itp_$SPLIT"

# Run preprocessing
python "$SCRIPT_DIR/lib/waymo_preprocess.py" \
    --input-dir "$DATA_ROOT/$FORMAT/$SPLIT" \
    --output-dir "$PROCESSED_ROOT/itp_$SPLIT" \
    --split "$SPLIT" \
    --num-workers "$WORKERS" \
    $INTERACTIVE_ONLY

echo ""
echo "Processing complete!"
echo "Processed files: $(find "$PROCESSED_ROOT/itp_$SPLIT" -name "*.npz" | wc -l)"
