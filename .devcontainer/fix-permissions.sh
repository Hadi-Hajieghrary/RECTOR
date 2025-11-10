#!/usr/bin/env bash
set -euo pipefail

echo "🔧 Fixing permissions..."

# Directories to ensure proper permissions
DIRS=(
    "/workspace/data"
    "/workspace/models"
    "/workspace/output"
    "/workspace/logs"
    "/workspace/.cache"
    "/workspace/notebooks"
)

for dir in "${DIRS[@]}"; do
    if [ -d "$dir" ]; then
        chmod -R 755 "$dir" 2>/dev/null || true
        echo "  ✓ Fixed $dir"
    else
        mkdir -p "$dir"
        chmod -R 755 "$dir" 2>/dev/null || true
        echo "  ✓ Created $dir"
    fi
done

echo "✅ Permissions normalized"
