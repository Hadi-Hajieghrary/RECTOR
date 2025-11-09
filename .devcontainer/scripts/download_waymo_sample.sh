#!/usr/bin/env bash
# RECTOR - Waymo Dataset Sample Downloader
# Downloads a small sample of Waymo data for development and testing
set -euo pipefail

# Colors for output
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; RED='\033[0;31m'; NC='\033[0m'
print_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
print_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
print_step() { echo -e "${BLUE}[STEP]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Configuration
WOMD_VERSION="${WOMD_VERSION:-v_1_3_0}"
BUCKET="gs://waymo_open_dataset_motion_${WOMD_VERSION}/uncompressed"
DATA_DIR="${WAYMO_DATA_ROOT:-/workspace/data/waymo_open_dataset_motion_v_1_3_0}"
NUM_SAMPLE_FILES="${NUM_SAMPLE_FILES:-5}"

echo "════════════════════════════════════════════════════════════"
echo "  RECTOR - Waymo Dataset Sample Downloader"
echo "════════════════════════════════════════════════════════════"
echo ""
print_info "Version: ${WOMD_VERSION}"
print_info "Target: ${DATA_DIR}"
print_info "Samples per split: ${NUM_SAMPLE_FILES} files"
echo ""

# Check if gsutil is available
if ! command -v gsutil &> /dev/null; then
    print_error "gsutil not found. Please install Google Cloud SDK."
    echo "Run: apt-get update && apt-get install -y google-cloud-cli"
    exit 1
fi

# Check authentication
print_step "Checking Google Cloud authentication..."
if ! gcloud auth application-default print-access-token >/dev/null 2>&1; then
    print_warn "Not authenticated with Google Cloud."
    echo ""
    echo "Please run these commands to authenticate:"
    echo "  1. gcloud auth login --no-browser"
    echo "  2. gcloud auth application-default login --no-browser"
    echo ""
    echo "Make sure you've accepted the Waymo Open Dataset terms at:"
    echo "  https://waymo.com/open/"
    exit 1
fi

print_info "✓ Authentication OK"

# Check bucket access
print_step "Verifying access to Waymo bucket..."
if ! gsutil ls "${BUCKET}/scenario/training" >/dev/null 2>&1; then
    print_error "Cannot access Waymo bucket!"
    echo ""
    echo "Possible reasons:"
    echo "  1. Haven't accepted Waymo Open Dataset terms"
    echo "  2. Using wrong Google account"
    echo "  3. Network/firewall issues"
    echo ""
    echo "Visit: https://waymo.com/open/ and accept the terms"
    exit 1
fi

print_info "✓ Bucket access OK"

# Create directory structure
print_step "Creating directory structure..."
mkdir -p "${DATA_DIR}"/{scenario,tf_example,lidar_and_camera}/{training,validation,testing}

# Download function
download_sample() {
    local format=$1
    local split=$2
    local num_files=$3
    local source="${BUCKET}/${format}/${split}"
    local dest="${DATA_DIR}/${format}/${split}"
    
    print_step "Downloading ${format}/${split} (${num_files} files)..."
    
    # List and download files
    files=$(gsutil ls "${source}/*" 2>/dev/null | head -n "$num_files" || true)
    
    if [ -z "$files" ]; then
        print_warn "No files found for ${format}/${split}"
        return 0
    fi
    
    mkdir -p "$dest"
    echo "$files" | gsutil -m cp -n -I "$dest/" 2>&1 | grep -v "Copying\|Operation completed"  || true
    
    local count=$(find "$dest" -type f | wc -l)
    print_info "✓ Downloaded ${count} files to ${format}/${split}"
}

# Download scenario format (primary format for motion planning)
echo ""
print_step "Downloading SCENARIO format..."
download_sample "scenario" "training" $NUM_SAMPLE_FILES
download_sample "scenario" "validation" $NUM_SAMPLE_FILES
download_sample "scenario" "testing" $NUM_SAMPLE_FILES

# Optional: Download interactive scenarios (human-driven vehicles)
if [ "${DOWNLOAD_INTERACTIVE:-0}" = "1" ]; then
    echo ""
    print_step "Downloading INTERACTIVE scenarios..."
    download_sample "scenario" "validation_interactive" $NUM_SAMPLE_FILES
    download_sample "scenario" "testing_interactive" $NUM_SAMPLE_FILES
fi

# Optional: Download 20-second extended scenarios
if [ "${DOWNLOAD_20S:-0}" = "1" ]; then
    echo ""
    print_step "Downloading 20-second extended scenarios..."
    download_sample "scenario" "training_20s" $NUM_SAMPLE_FILES
fi

# Download TF Example format (legacy format, still useful)
echo ""
print_step "Downloading TF_EXAMPLE format..."
download_sample "tf_example" "training" $NUM_SAMPLE_FILES
download_sample "tf_example" "validation" $NUM_SAMPLE_FILES
download_sample "tf_example" "testing" $NUM_SAMPLE_FILES

# Optional: Download lidar_and_camera (very large, skip by default)
if [ "${DOWNLOAD_LIDAR:-0}" = "1" ]; then
    echo ""
    print_step "Downloading LIDAR_AND_CAMERA format..."
    download_sample "lidar_and_camera" "training" $NUM_SAMPLE_FILES
    download_sample "lidar_and_camera" "validation" $NUM_SAMPLE_FILES
    download_sample "lidar_and_camera" "testing" $NUM_SAMPLE_FILES
fi

# Summary
echo ""
echo "════════════════════════════════════════════════════════════"
print_info "✓ Sample download complete!"
echo ""
print_info "Dataset location: ${DATA_DIR}"
print_info "Total files: $(find ${DATA_DIR} -type f -name '*tfrecord*' | wc -l)"
print_info "Total size: $(du -sh ${DATA_DIR} 2>/dev/null | cut -f1)"
echo ""
echo "To download more files, set NUM_SAMPLE_FILES:"
echo "  NUM_SAMPLE_FILES=20 bash .devcontainer/scripts/download_waymo_sample.sh"
echo ""
echo "To include interactive scenarios (validation_interactive, testing_interactive):"
echo "  DOWNLOAD_INTERACTIVE=1 bash .devcontainer/scripts/download_waymo_sample.sh"
echo ""
echo "To include 20-second extended training scenarios:"
echo "  DOWNLOAD_20S=1 bash .devcontainer/scripts/download_waymo_sample.sh"
echo ""
echo "To include lidar/camera data:"
echo "  DOWNLOAD_LIDAR=1 bash .devcontainer/scripts/download_waymo_sample.sh"
echo ""
echo "Combine options:"
echo "  DOWNLOAD_INTERACTIVE=1 DOWNLOAD_20S=1 NUM_SAMPLE_FILES=10 waymo-download"
echo "════════════════════════════════════════════════════════════"
