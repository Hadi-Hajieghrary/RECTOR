#!/usr/bin/env bash
# RECTOR - Waymo Dataset Status Checker
# Shows the current state of the downloaded Waymo dataset
set -euo pipefail

# Colors
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
print_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
print_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
print_header() { echo -e "${BLUE}$1${NC}"; }

DATA_DIR="${WAYMO_DATA_ROOT:-/workspace/data/waymo_open_dataset_motion_v_1_3_0}"

echo "════════════════════════════════════════════════════════════"
echo "  RECTOR - Waymo Dataset Status"
echo "════════════════════════════════════════════════════════════"
echo ""
print_info "Data directory: ${DATA_DIR}"
echo ""

if [ ! -d "$DATA_DIR" ]; then
    print_warn "Dataset directory does not exist!"
    echo "Run: .devcontainer/scripts/download_waymo_sample.sh"
    exit 1
fi

# Function to check format status
check_format() {
    local format=$1
    local format_dir="${DATA_DIR}/${format}"
    
    if [ ! -d "$format_dir" ]; then
        echo "  ${format}: Not present"
        return
    fi
    
    print_header "Format: ${format}"
    
    for split in training validation testing training_20s validation_interactive testing_interactive; do
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

# Check each format
check_format "scenario"
check_format "tf_example"
check_format "lidar_and_camera"

# Overall summary
total_files=$(find "$DATA_DIR" -type f -name '*tfrecord*' 2>/dev/null | wc -l)
total_size=$(du -sh "$DATA_DIR" 2>/dev/null | cut -f1)

echo "════════════════════════════════════════════════════════════"
print_info "Total files: ${total_files}"
print_info "Total size: ${total_size}"
echo "════════════════════════════════════════════════════════════"

if [ "$total_files" -eq 0 ]; then
    echo ""
    print_warn "No dataset files found!"
    echo ""
    echo "To download sample data:"
    echo "  bash .devcontainer/scripts/download_waymo_sample.sh"
    echo ""
fi
