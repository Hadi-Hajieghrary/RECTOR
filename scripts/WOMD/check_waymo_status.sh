#!/usr/bin/env bash
# RECTOR - Waymo Dataset Status Checker
# Shows the current state of the downloaded Waymo dataset
set -euo pipefail

# Colors
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
print_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
print_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
print_header() { echo -e "${BLUE}$1${NC}"; }

RAW_DIR="${WAYMO_RAW_ROOT:-/workspace/data/WOMD/datasets/waymo_open_dataset/motion_v_1_3_0/raw}"
PROCESSED_DIR="${WAYMO_PROCESSED_ROOT:-/workspace/data/WOMD/datasets/waymo_open_dataset/motion_v_1_3_0/processed}"

echo "════════════════════════════════════════════════════════════"
echo "  RECTOR - Waymo Dataset Status"
echo "════════════════════════════════════════════════════════════"
echo ""
print_info "Raw directory: ${RAW_DIR}"
print_info "Processed directory: ${PROCESSED_DIR}"
echo ""

if [ ! -d "$RAW_DIR" ] && [ ! -d "$PROCESSED_DIR" ]; then
    print_warn "Dataset directories do not exist!"
    echo "Run: scripts/WOMD/download_waymo_sample.sh"
    exit 1
fi

# Function to check format status
check_format() {
    local base_dir=$1
    local format=$2
    local split_list=$3
    local format_dir="${base_dir}/${format}"

    if [ ! -d "$format_dir" ]; then
        echo "  ${format}: Not present"
        return
    fi

    print_header "Format: ${format} (${base_dir})"

    for split in $split_list; do
        local split_dir="${format_dir}/${split}"
        if [ -d "$split_dir" ]; then
            local file_count=$(find "$split_dir" -type f -name '*tfrecord*' 2>/dev/null | wc -l)
            local size=$(du -sh "$split_dir" 2>/dev/null | cut -f1)

            if [ "$file_count" -gt 0 ]; then
                echo "  ✓ ${split}: ${file_count} files (${size})"
            else
                echo "  ✗ ${split}: empty"
            fi
        fi
    done
    echo ""
}

# Check raw formats
check_format "$RAW_DIR" "scenario" "training validation testing training_20s validation_interactive testing_interactive"
check_format "$RAW_DIR" "lidar_and_camera" "training validation testing"

# Check processed formats
check_format "$PROCESSED_DIR" "tf" "training validation testing training_interactive validation_interactive testing_interactive"

# Overall summary
count_tfrecords() {
    local dir=$1
    if [ -d "$dir" ]; then
        find "$dir" -type f -name '*tfrecord*' 2>/dev/null | wc -l
    else
        echo 0
    fi
}

total_files=$((
    $(count_tfrecords "$RAW_DIR") +
    $(count_tfrecords "$PROCESSED_DIR")
))
raw_size=$(du -sh "$RAW_DIR" 2>/dev/null | cut -f1 || echo "0")
processed_size=$(du -sh "$PROCESSED_DIR" 2>/dev/null | cut -f1 || echo "0")

echo "════════════════════════════════════════════════════════════"
print_info "Total files: ${total_files}"
print_info "Raw size: ${raw_size}"
print_info "Processed size: ${processed_size}"
echo "════════════════════════════════════════════════════════════"

if [ "$total_files" -eq 0 ]; then
    echo ""
    print_warn "No dataset files found!"
    echo ""
    echo "To download sample data:"
    echo "  bash scripts/WOMD/download_waymo_sample.sh"
    echo ""
fi
