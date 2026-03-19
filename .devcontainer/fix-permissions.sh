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
        find "$dir" -type d -exec chmod 755 {} + 2>/dev/null || true
        find "$dir" -type f -exec chmod 644 {} + 2>/dev/null || true
        echo "  ✓ Fixed $dir"
    else
        mkdir -p "$dir"
        chmod 755 "$dir"
        echo "  ✓ Created $dir"
    fi
done

echo "✅ Permissions normalized"
