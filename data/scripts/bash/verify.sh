#!/bin/bash
# Verify downloaded Waymo dataset files
# Usage: bash/verify.sh [--quiet]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${WAYMO_RAW_ROOT:-$(dirname "$SCRIPT_DIR")/datasets/waymo_open_dataset/motion_v_1_3_0/raw}"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

QUIET=false
[[ "$1" == "--quiet" ]] && QUIET=true

# Check directory
check_dir() {
    local dir=$1
    local name=$2
    
    if [ ! -d "$dir" ]; then
        $QUIET || echo -e "${RED}✗${NC} Missing: $name"
        return 0
    fi
    
    local count=$(find "$dir" -type f -name "*.tfrecord*" 2>/dev/null | wc -l)
    
    if [ $count -gt 0 ]; then
        $QUIET || echo -e "${GREEN}✓${NC} $name: $count files"
    else
        $QUIET || echo -e "${RED}✗${NC} $name: No files"
    fi
    
    return $count
}

$QUIET || echo "Verifying Waymo Dataset..."
$QUIET || echo "========================================"

TOTAL=0

# Check all standard partitions
for format in scenario tf_example; do
    for split in training testing_interactive validation_interactive; do
        check_dir "$DATA_ROOT/$format/$split" "$format/$split"
        TOTAL=$((TOTAL + $?))
    done
done

# Check training_interactive if it exists
if [ -d "$DATA_ROOT/tf_example/training_interactive" ]; then
    check_dir "$DATA_ROOT/tf_example/training_interactive" "tf_example/training_interactive"
    TOTAL=$((TOTAL + $?))
fi

$QUIET || echo "========================================"
$QUIET || echo "Total files: $TOTAL"

# Exit code
if [ $TOTAL -eq 0 ]; then
    $QUIET || echo -e "${RED}No data found!${NC}"
    exit 1
fi

exit 0
