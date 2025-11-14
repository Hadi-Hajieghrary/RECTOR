#!/bin/bash
# Download Waymo dataset files
# Usage: bash/download.sh [num_files] [partitions...]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${WAYMO_RAW_ROOT:-$(dirname "$SCRIPT_DIR")/datasets/waymo_open_dataset/motion_v_1_3_0/raw}"
GCS_BASE="gs://waymo_open_dataset_motion_v_1_3_0/uncompressed"

# Check and setup authentication
check_auth() {
    echo "Checking Google Cloud authentication..."
    
    # Check if gcloud is installed
    if ! command -v gcloud &> /dev/null; then
        echo "ERROR: gcloud CLI not installed"
        echo "Install: curl https://sdk.cloud.google.com | bash"
        exit 1
    fi
    
    # Check if authenticated (store result to avoid calling twice)
    local active_account=$(gcloud auth list --filter=status:ACTIVE --format="value(account)" 2>/dev/null | head -1)
    
    if [ -z "$active_account" ]; then
        echo ""
        echo "╔════════════════════════════════════════════════════════════════════╗"
        echo "║  Starting Automatic Authentication                                 ║"
        echo "╚════════════════════════════════════════════════════════════════════╝"
        echo ""
        echo "Running authentication automatically..."
        echo ""
        
        # Run authentication script
        BASH_DIR="$(dirname "${BASH_SOURCE[0]}")"
        if [ -f "$BASH_DIR/authenticate.sh" ]; then
            bash "$BASH_DIR/authenticate.sh" || {
                echo ""
                echo "ERROR: Authentication failed"
                echo "Please manually authenticate and try again."
                exit 1
            }
        else
            echo "ERROR: authenticate.sh not found"
            echo ""
            echo "Please run these commands manually:"
            echo "  1. gcloud auth login --no-browser"
            echo "  2. gcloud auth application-default login --no-browser"
            echo ""
            exit 1
        fi
    else
        echo "✓ Authenticated as: $active_account"
    fi
    
    # Remind about license acceptance
    echo ""
    echo "NOTE: Make sure you've accepted the Waymo Open Dataset license at:"
    echo "      https://waymo.com/open/terms"
    echo ""
}

# Parse arguments
NUM_FILES="${1:-5}"
shift || true
PARTITIONS=("$@")

# Default partitions if none specified
if [ ${#PARTITIONS[@]} -eq 0 ]; then
    PARTITIONS=(
        "scenario/training"
        "scenario/testing_interactive"
        "scenario/validation_interactive"
        "tf_example/training"
        "tf_example/testing_interactive"
        "tf_example/validation_interactive"
    )
fi

# Check gsutil
if ! command -v gsutil &> /dev/null; then
    echo "ERROR: gsutil not installed"
    echo "Install: curl https://sdk.cloud.google.com | bash"
    exit 1
fi

# Check authentication
check_auth

echo "========================================"
echo "Waymo Dataset Download"
echo "========================================"
echo "Files per partition: $NUM_FILES"
echo "Partitions: ${PARTITIONS[*]}"
echo "Destination: $DATA_ROOT"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
[[ ! $REPLY =~ ^[Yy]$ ]] && echo "Aborted." && exit 0

# Download each partition
for partition in "${PARTITIONS[@]}"; do
    echo ""
    echo "Downloading $partition..."
    mkdir -p "$DATA_ROOT/$partition"
    
    # List files and download first N
    files=$(gsutil ls "$GCS_BASE/$partition/" | head -n "$NUM_FILES")
    
    if [ -z "$files" ]; then
        echo "WARNING: No files found in $partition"
        continue
    fi
    
    echo "$files" | gsutil -m cp -I "$DATA_ROOT/$partition/" || {
        echo "WARNING: Failed to download some files from $partition"
    }
done

echo ""
echo "Download complete!"
