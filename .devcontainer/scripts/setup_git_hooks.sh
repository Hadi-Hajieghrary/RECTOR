#!/usr/bin/env bash
# RECTOR - Git Hooks Setup
# Installs Git pre-commit hook for workspace documentation
set -euo pipefail

echo "🔧 Setting up RECTOR Git hooks..."

HOOK_SOURCE=".devcontainer/scripts/git-pre-commit-hook.sh"
HOOK_TARGET=".git/hooks/pre-commit"

# Check if we're in a git repository
if [ ! -d ".git" ]; then
    echo "❌ Error: Not in a git repository root"
    exit 1
fi

# Copy the pre-commit hook
if [ -f "$HOOK_SOURCE" ]; then
    cp "$HOOK_SOURCE" "$HOOK_TARGET"
    chmod +x "$HOOK_TARGET"
    echo "✓ Installed pre-commit hook"
else
    echo "❌ Error: Hook source not found at $HOOK_SOURCE"
    exit 1
fi

# Test the hook
echo ""
echo "🧪 Testing pre-commit hook..."
if bash "$HOOK_TARGET"; then
    echo ""
    echo "✅ Git hooks installed successfully!"
    echo ""
    echo "The following files will be auto-generated on each commit:"
    echo "  • WORKSPACE_STRUCTURE.md - Workspace directory tree"
    echo "  • DATA_INVENTORY.md - Data folder inventory with statistics"
    echo ""
    echo "These files are tracked in Git and will help document your workspace."
else
    echo "❌ Hook test failed"
    exit 1
fi
