#!/bin/bash
# Run augmentation on all Waymo Motion Dataset splits
#
# Usage: ./run_augmentation.sh [output_dir] [workers]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"
DATA_DIR="${WORKSPACE_DIR}/datasets/waymo_open_dataset/motion_v_1_3_0/raw/scenario"
OUTPUT_DIR="${1:-${WORKSPACE_DIR}/augmented}"
WORKERS="${2:-4}"

echo "=============================================="
echo "Waymo Rule Augmentation Pipeline"
echo "=============================================="
echo "Data directory: ${DATA_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Workers: ${WORKERS}"
echo ""

# Check if data exists
if [ ! -d "${DATA_DIR}" ]; then
    echo "ERROR: Data directory not found: ${DATA_DIR}"
    exit 1
fi

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Process each split
for split in testing_interactive training_interactive validation_interactive; do
    SPLIT_DIR="${DATA_DIR}/${split}"

    if [ ! -d "${SPLIT_DIR}" ]; then
        echo "Skipping ${split} (directory not found)"
        continue
    fi

    # Count files
    NUM_FILES=$(find "${SPLIT_DIR}" -name "*.tfrecord*" 2>/dev/null | wc -l)

    if [ "${NUM_FILES}" -eq 0 ]; then
        echo "Skipping ${split} (no tfrecord files found)"
        continue
    fi

    echo ""
    echo "Processing ${split} (${NUM_FILES} files)..."
    echo "----------------------------------------------"

    SPLIT_OUTPUT="${OUTPUT_DIR}/${split}"
    mkdir -p "${SPLIT_OUTPUT}"

    # Run processing (set PYTHONPATH so python -m finds waymo_rule_eval package)
    cd "${WORKSPACE_DIR}"
    PYTHONPATH="${WORKSPACE_DIR}:${PYTHONPATH:-}" \
    python -m waymo_rule_eval.augmentation.process_scenarios \
        --input "${SPLIT_DIR}/*.tfrecord*" \
        --output "${SPLIT_OUTPUT}/" \
        --window-size 8.0 \
        --stride 1.0 \
        --workers "${WORKERS}" \
        2>&1 | tee "${SPLIT_OUTPUT}/processing.log"
done

echo ""
echo "=============================================="
echo "Augmentation complete!"
echo "Output saved to: ${OUTPUT_DIR}"
echo "=============================================="

# Print summary of output files
echo ""
echo "Output files:"
find "${OUTPUT_DIR}" -name "*.jsonl" -exec wc -l {} \; | head -20
