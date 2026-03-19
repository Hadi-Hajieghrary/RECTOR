#!/usr/bin/env bash
# RECTOR - Git Hooks Setup
# Installs Git pre-commit hook for workspace documentation
set -euo pipefail

echo "🔧 Setting up RECTOR Git hooks..."

HOOK_SOURCE="scripts/git-pre-commit-hook.sh"
HOOK_TARGET=".git/hooks/pre-commit"

# Check if we're in a git repository
if [ ! -d ".git" ]; then
    echo "⚠️  Skipping: Not in a git repository root (no .git directory)"
    echo "   Git hooks will be configured when a repository is initialized."
    exit 0
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

# Verify the hook is installed (without executing it)
echo ""
if [ -x "$HOOK_TARGET" ]; then
    echo "✅ Git hooks installed successfully!"
    echo ""
    echo "The following documentation snapshots will be refreshed on each commit:"
    echo "  • WORKSPACE_STRUCTURE.md - Workspace directory tree"
    echo "  • data/DATA_INVENTORY.md - Data folder inventory with statistics"
    echo ""
    echo "These tracked files provide quick navigation and provenance for the workspace."
else
    echo "⚠️  Hook installed but may not be executable"
fi
