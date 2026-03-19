#!/bin/bash
# =============================================================================
# RECTOR Sensitivity Test Runner
# =============================================================================
#
# This script runs the critical GO/NO-GO sensitivity test that validates
# whether M2I's conditional model responds to influencer trajectory changes.
#
# Usage:
#   bash models/RECTOR/scripts/bash/run_sensitivity_test.sh [OPTIONS]
#
# Options:
#   -v, --verbose     Enable verbose output
#   -d, --data_dir    Path to TFRecord directory
#   --device          cuda or cpu (default: cuda)
#   --scenario_idx    Which scenario to test (default: 0)
#   -h, --help        Show this help message
#
# Exit Codes:
#   0: PASS - Sensitivity validated, proceed with development
#   1: FAIL - Sensitivity test failed, interactive planning blocked
#   2: ERROR - Test could not run
#
# =============================================================================

set -e

# Defaults
VERBOSE=""
DATA_DIR="/workspace/data/datasets/waymo_open_dataset/motion_v_1_3_0/processed/tf/validation_interactive"
DEVICE="cuda"
SCENARIO_IDX="0"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)
            VERBOSE="-v"
            shift
            ;;
        -d|--data_dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --scenario_idx)
            SCENARIO_IDX="$2"
            shift 2
            ;;
        -h|--help)
            head -30 "$0" | tail -25
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 2
            ;;
    esac
done

# Change to workspace root
cd /workspace

echo "=============================================="
echo "RECTOR Sensitivity Test (GO/NO-GO Gate #1)"
echo "=============================================="
echo ""
echo "Data directory: $DATA_DIR"
echo "Device: $DEVICE"
echo "Scenario index: $SCENARIO_IDX"
echo ""

# Activate virtual environment if it exists
if [ -f "/workspace/.venv/bin/activate" ]; then
    source /workspace/.venv/bin/activate
fi

# Run the test
python models/RECTOR/tests/test_sensitivity.py \
    --data_dir "$DATA_DIR" \
    --device "$DEVICE" \
    --scenario_idx "$SCENARIO_IDX" \
    $VERBOSE

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=============================================="
    echo "✓ SENSITIVITY TEST PASSED"
    echo "  Proceed with RECTOR development."
    echo "=============================================="
elif [ $EXIT_CODE -eq 1 ]; then
    echo ""
    echo "=============================================="
    echo "✗ SENSITIVITY TEST FAILED"
    echo "  Interactive planning is BLOCKED."
    echo "  Debug the conditioning interface before proceeding."
    echo "=============================================="
else
    echo ""
    echo "=============================================="
    echo "⚠ TEST ERROR (exit code: $EXIT_CODE)"
    echo "  Check data paths and model weights."
    echo "=============================================="
fi

exit $EXIT_CODE
