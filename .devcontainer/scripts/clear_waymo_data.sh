#!/usr/bin/env bash
# RECTOR - Waymo Dataset Cleaner
# Removes downloaded Waymo dataset files
set -euo pipefail

# Colors
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; BLUE='\033[0;34m'; NC='\033[0m'
print_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
print_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }
print_step() { echo -e "${BLUE}[STEP]${NC} $1"; }

DATA_DIR="${WAYMO_DATA_ROOT:-/workspace/data/datasets/waymo_open_dataset/motion_v_1_3_0}"

echo "════════════════════════════════════════════════════════════"
echo "  RECTOR - Waymo Dataset Cleaner"
echo "════════════════════════════════════════════════════════════"
echo ""

if [ ! -d "$DATA_DIR" ]; then
    print_info "No dataset directory found at: ${DATA_DIR}"
    exit 0
fi

# Show current status
total_files=$(find "$DATA_DIR" -type f -name '*tfrecord*' 2>/dev/null | wc -l)
total_size=$(du -sh "$DATA_DIR" 2>/dev/null | cut -f1)

if [ "$total_files" -eq 0 ]; then
    print_info "No dataset files found. Nothing to clean."
    exit 0
fi

print_warn "Current dataset status:"
echo "  Location: ${DATA_DIR}"
echo "  Files: ${total_files}"
echo "  Size: ${total_size}"
echo ""

# Confirmation prompt
read -p "⚠️  Delete all downloaded Waymo data? [y/N] " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    print_info "Cancelled. No files were deleted."
    exit 0
fi

# Delete files
print_step "Removing dataset files..."

# Remove all tfrecord files but keep directory structure
find "$DATA_DIR" -type f -name '*tfrecord*' -delete 2>/dev/null || true

# Optionally remove empty directories
find "$DATA_DIR" -type d -empty -delete 2>/dev/null || true

# Verify deletion
remaining_files=$(find "$DATA_DIR" -type f -name '*tfrecord*' 2>/dev/null | wc -l)

if [ "$remaining_files" -eq 0 ]; then
    print_info "✓ All dataset files removed successfully!"
    echo ""
    print_info "To download again: waymo-download"
else
    print_warn "Some files may remain: ${remaining_files} files"
fi

echo "════════════════════════════════════════════════════════════"
