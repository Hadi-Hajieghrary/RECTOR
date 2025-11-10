#!/usr/bin/env bash
# RECTOR - Git Pre-Commit Hook
# Generates workspace structure documentation before each commit
set -euo pipefail

WORKSPACE_TREE_FILE="WORKSPACE_STRUCTURE.md"
DATA_INVENTORY_FILE="data/DATA_INVENTORY.md"

echo "🔍 Generating workspace structure documentation..."

# Generate main workspace tree (excluding data folder contents)
cat > "$WORKSPACE_TREE_FILE" << 'HEADER'
# RECTOR Workspace Structure

Auto-generated workspace structure (updated on commit).

## Workspace Tree

```
HEADER

# Generate tree excluding data folder files, git, cache, and other large directories
tree -L 3 -a \
    -I '.git|__pycache__|*.pyc|.pytest_cache|.ipynb_checkpoints|*.egg-info|.cache|.mypy_cache|.ruff_cache|node_modules|References|.specstory' \
    --dirsfirst \
    -F >> "$WORKSPACE_TREE_FILE"

echo '```' >> "$WORKSPACE_TREE_FILE"

# Add timestamp
echo "" >> "$WORKSPACE_TREE_FILE"
echo "*Last updated: $(date -u '+%Y-%m-%d %H:%M:%S UTC')*" >> "$WORKSPACE_TREE_FILE"

# Generate data folder inventory
cat > "$DATA_INVENTORY_FILE" << 'DATAHEADER'
# RECTOR Data Folder Inventory

Auto-generated data folder structure (updated on commit).

## Data Directory Structure

```
DATAHEADER

# Generate data folder tree (directories only)
if [ -d "data" ]; then
    tree data/ -d -L 4 --dirsfirst -F >> "$DATA_INVENTORY_FILE"
    
    echo '```' >> "$DATA_INVENTORY_FILE"
    echo "" >> "$DATA_INVENTORY_FILE"
    
    # Add file counts and sizes for each main data subdirectory
    echo "## Data Statistics" >> "$DATA_INVENTORY_FILE"
    echo "" >> "$DATA_INVENTORY_FILE"
    
    for subdir in data/*/; do
        if [ -d "$subdir" ]; then
            dirname=$(basename "$subdir")
            file_count=$(find "$subdir" -type f 2>/dev/null | wc -l)
            total_size=$(du -sh "$subdir" 2>/dev/null | cut -f1)
            
            echo "### $dirname/" >> "$DATA_INVENTORY_FILE"
            echo "- **Files**: $file_count" >> "$DATA_INVENTORY_FILE"
            echo "- **Total Size**: $total_size" >> "$DATA_INVENTORY_FILE"
            echo "" >> "$DATA_INVENTORY_FILE"
            
            # For waymo dataset, show detailed split information with file listings
            if [[ "$dirname" == "waymo"* ]]; then
                echo "**Waymo Dataset Splits**:" >> "$DATA_INVENTORY_FILE"
                echo "" >> "$DATA_INVENTORY_FILE"
                
                # Check scenario format
                if [ -d "$subdir/scenario" ]; then
                    echo "#### Scenario Format" >> "$DATA_INVENTORY_FILE"
                    for split_dir in "$subdir/scenario"/*/; do
                        if [ -d "$split_dir" ]; then
                            split_name=$(basename "$split_dir")
                            split_files=$(find "$split_dir" -type f -name '*tfrecord*' 2>/dev/null | wc -l)
                            split_size=$(du -sh "$split_dir" 2>/dev/null | cut -f1)
                            if [ "$split_files" -gt 0 ]; then
                                echo "- **$split_name**: $split_files files ($split_size)" >> "$DATA_INVENTORY_FILE"
                                echo '  ```' >> "$DATA_INVENTORY_FILE"
                                find "$split_dir" -type f -name '*tfrecord*' -printf '  %f (%s bytes)\n' 2>/dev/null | sort | head -20 >> "$DATA_INVENTORY_FILE"
                                if [ "$split_files" -gt 20 ]; then
                                    echo "  ... and $((split_files - 20)) more files" >> "$DATA_INVENTORY_FILE"
                                fi
                                echo '  ```' >> "$DATA_INVENTORY_FILE"
                            else
                                echo "- **$split_name**: empty" >> "$DATA_INVENTORY_FILE"
                            fi
                        fi
                    done
                    echo "" >> "$DATA_INVENTORY_FILE"
                fi
                
                # Check tf_example format
                if [ -d "$subdir/tf_example" ]; then
                    echo "#### TF Example Format" >> "$DATA_INVENTORY_FILE"
                    for split_dir in "$subdir/tf_example"/*/; do
                        if [ -d "$split_dir" ]; then
                            split_name=$(basename "$split_dir")
                            split_files=$(find "$split_dir" -type f -name '*tfrecord*' 2>/dev/null | wc -l)
                            split_size=$(du -sh "$split_dir" 2>/dev/null | cut -f1)
                            if [ "$split_files" -gt 0 ]; then
                                echo "- **$split_name**: $split_files files ($split_size)" >> "$DATA_INVENTORY_FILE"
                                echo '  ```' >> "$DATA_INVENTORY_FILE"
                                find "$split_dir" -type f -name '*tfrecord*' -printf '  %f (%s bytes)\n' 2>/dev/null | sort | head -20 >> "$DATA_INVENTORY_FILE"
                                if [ "$split_files" -gt 20 ]; then
                                    echo "  ... and $((split_files - 20)) more files" >> "$DATA_INVENTORY_FILE"
                                fi
                                echo '  ```' >> "$DATA_INVENTORY_FILE"
                            else
                                echo "- **$split_name**: empty" >> "$DATA_INVENTORY_FILE"
                            fi
                        fi
                    done
                    echo "" >> "$DATA_INVENTORY_FILE"
                fi
                
                # Check lidar_and_camera format
                if [ -d "$subdir/lidar_and_camera" ]; then
                    echo "#### Lidar & Camera Format" >> "$DATA_INVENTORY_FILE"
                    for split_dir in "$subdir/lidar_and_camera"/*/; do
                        if [ -d "$split_dir" ]; then
                            split_name=$(basename "$split_dir")
                            split_files=$(find "$split_dir" -type f -name '*tfrecord*' 2>/dev/null | wc -l)
                            split_size=$(du -sh "$split_dir" 2>/dev/null | cut -f1)
                            if [ "$split_files" -gt 0 ]; then
                                echo "- **$split_name**: $split_files files ($split_size)" >> "$DATA_INVENTORY_FILE"
                                echo '  ```' >> "$DATA_INVENTORY_FILE"
                                find "$split_dir" -type f -name '*tfrecord*' -printf '  %f (%s bytes)\n' 2>/dev/null | sort | head -20 >> "$DATA_INVENTORY_FILE"
                                if [ "$split_files" -gt 20 ]; then
                                    echo "  ... and $((split_files - 20)) more files" >> "$DATA_INVENTORY_FILE"
                                fi
                                echo '  ```' >> "$DATA_INVENTORY_FILE"
                            else
                                echo "- **$split_name**: empty" >> "$DATA_INVENTORY_FILE"
                            fi
                        fi
                    done
                    echo "" >> "$DATA_INVENTORY_FILE"
                fi
            else
                # For non-Waymo directories, list all files
                files=$(find "$subdir" -type f ! -name '.gitkeep' 2>/dev/null)
                if [ -n "$files" ]; then
                    echo "**Files:**" >> "$DATA_INVENTORY_FILE"
                    echo '```' >> "$DATA_INVENTORY_FILE"
                    find "$subdir" -type f ! -name '.gitkeep' -printf '%P\n' 2>/dev/null | sort >> "$DATA_INVENTORY_FILE"
                    echo '```' >> "$DATA_INVENTORY_FILE"
                fi
                echo "" >> "$DATA_INVENTORY_FILE"
            fi
        fi
    done
else
    echo "No data directory found." >> "$DATA_INVENTORY_FILE"
    echo '```' >> "$DATA_INVENTORY_FILE"
fi

# Add timestamp
echo "" >> "$DATA_INVENTORY_FILE"
echo "*Last updated: $(date -u '+%Y-%m-%d %H:%M:%S UTC')*" >> "$DATA_INVENTORY_FILE"

# Stage the generated files if they've changed
if [ -f "$WORKSPACE_TREE_FILE" ]; then
    git add "$WORKSPACE_TREE_FILE"
    echo "✓ Updated $WORKSPACE_TREE_FILE"
fi

if [ -f "$DATA_INVENTORY_FILE" ]; then
    git add "$DATA_INVENTORY_FILE"
    echo "✓ Updated $DATA_INVENTORY_FILE"
fi

echo "✅ Workspace documentation updated"

# Generate movies from Waymo scenarios
MOVIE_GENERATION_SCRIPT="/workspace/.devcontainer/scripts/git-pre-commit-generate-movies.sh"
if [ -f "$MOVIE_GENERATION_SCRIPT" ]; then
    echo ""
    bash "$MOVIE_GENERATION_SCRIPT"
fi

exit 0
