#!/bin/bash
# Filter interactive scenarios from training data
# Usage: bash/filter.sh [--type TYPE]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASETS_ROOT="$(dirname "$SCRIPT_DIR")/datasets/waymo_open_dataset/motion_v_1_3_0"
DATA_ROOT="${WAYMO_RAW_ROOT:-$DATASETS_ROOT/raw}"

# Default parameters
INTERACTION_TYPE="v2v"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --type)
            INTERACTION_TYPE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "========================================"
echo "Filter Interactive Training Data"
echo "========================================"
echo "Type: $INTERACTION_TYPE"
echo ""

INPUT_DIR="$DATA_ROOT/tf_example/training"
OUTPUT_DIR="$DATA_ROOT/tf_example/training_interactive"

if [ ! -d "$INPUT_DIR" ]; then
    echo "ERROR: Training data not found: $INPUT_DIR"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

python "$SCRIPT_DIR/lib/filter_interactive_training.py" \
    --input-dir "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --type "$INTERACTION_TYPE"

echo ""
echo "Filtering complete!"
echo "Interactive scenarios: $(find "$OUTPUT_DIR" -name "*.tfrecord*" | wc -l)"
