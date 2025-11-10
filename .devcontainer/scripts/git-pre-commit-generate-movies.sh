#!/bin/bash
set -e

echo "=== Pre-commit Hook: Generating Waymo Scenario Movies ==="

# Base directories
DATA_DIR="/workspace/data/datasets/waymo_open_dataset/motion_v_1_3_0/raw"
MOVIES_DIR="/workspace/data/movies"
MAX_SCENARIOS=10

# Check if viz_waymo_scenario.py exists
VISUALIZE_SCRIPT="/workspace/data/scripts/lib/viz_waymo_scenario.py"
if [ ! -f "$VISUALIZE_SCRIPT" ]; then
    echo "Warning: Visualization script not found at $VISUALIZE_SCRIPT"
    echo "Skipping movie generation."
    exit 0
fi

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "Warning: Data directory not found at $DATA_DIR"
    echo "Skipping movie generation."
    exit 0
fi

# Function to generate movies for a specific split
generate_movies() {
    local format=$1
    local split=$2
    local input_dir="${DATA_DIR}/${format}/${split}"
    local output_dir="${MOVIES_DIR}/${format}/${split}"
    
    # Check if input directory exists and has tfrecord files
    if [ ! -d "$input_dir" ]; then
        return
    fi
    
    local file_count=$(find "$input_dir" -name "*.tfrecord*" 2>/dev/null | wc -l)
    if [ "$file_count" -eq 0 ]; then
        return
    fi
    
    echo "Processing $format/$split (found $file_count files)..."
    
    # Create output directory
    mkdir -p "$output_dir"
    
    # Check if movies already exist (check MP4s since they're the primary format)
    local existing_movies=$(find "$output_dir" -name "*.mp4" 2>/dev/null | wc -l)
    if [ "$existing_movies" -ge "$MAX_SCENARIOS" ]; then
        echo "  Already have $existing_movies movies (max: $MAX_SCENARIOS), skipping generation."
        # Still add any existing GIFs to staging
        find "$output_dir" -name "*.gif" -exec git add {} \; 2>/dev/null || true
        return
    fi
    
    # Generate movies (suppress TensorFlow warnings)
    python "$VISUALIZE_SCRIPT" \
        --data_dir "$input_dir" \
        --num_scenarios "$MAX_SCENARIOS" \
        --output_dir "$output_dir" \
        --fps 10 \
        --dpi 100 \
        2>&1 | grep -v "tensorflow\|oneDNN\|GPU\|CUDA\|TensorRT\|NUMA" || true
    
    # Add generated movies to git staging (both MP4 and GIF)
    if [ -d "$output_dir" ]; then
        find "$output_dir" -name "*.mp4" -exec git add {} \; 2>/dev/null || true
        find "$output_dir" -name "*.gif" -exec git add {} \; 2>/dev/null || true
    fi
}

# List of splits to process for scenario format
SCENARIO_SPLITS=(
    "training"
    "validation"
    "testing"
    "validation_interactive"
    "testing_interactive"
)

# List of splits to process for tf_example format
TFEXAMPLE_SPLITS=(
    "training"
    "validation"
    "testing"
)

# Generate movies for scenario format
echo ""
echo "Generating movies for scenario format..."
for split in "${SCENARIO_SPLITS[@]}"; do
    generate_movies "scenario" "$split"
done

# Generate movies for tf_example format
echo ""
echo "Generating movies for tf_example format..."
TFEXAMPLE_SCRIPT="/workspace/data/scripts/viz_waymo_tfexample.py"
if [ -f "$TFEXAMPLE_SCRIPT" ]; then
    for split in "${TFEXAMPLE_SPLITS[@]}"; do
        local input_dir="${DATA_DIR}/tf_example/${split}"
        local output_dir="${MOVIES_DIR}/tf_example/${split}"
        
        # Check if input directory exists and has tfrecord files
        if [ ! -d "$input_dir" ]; then
            continue
        fi
        
        local file_count=$(find "$input_dir" -name "*.tfrecord*" 2>/dev/null | wc -l)
        if [ "$file_count" -eq 0 ]; then
            continue
        fi
        
        echo "Processing tf_example/$split (found $file_count files)..."
        
        # Create output directory
        mkdir -p "$output_dir"
        
        # Check if movies already exist (check MP4s since they're the primary format)
        local existing_movies=$(find "$output_dir" -name "*.mp4" 2>/dev/null | wc -l)
        if [ "$existing_movies" -ge "$MAX_SCENARIOS" ]; then
            echo "  Already have $existing_movies movies (max: $MAX_SCENARIOS), skipping generation."
            # Still add any existing GIFs to staging
            find "$output_dir" -name "*.gif" -exec git add {} \; 2>/dev/null || true
            continue
        fi
        
        # Generate movies (suppress TensorFlow warnings)
        python "$TFEXAMPLE_SCRIPT" \
            --data_dir "$input_dir" \
            --num_scenarios "$MAX_SCENARIOS" \
            --output_dir "$output_dir" \
            --fps 10 \
            --dpi 100 \
            2>&1 | grep -v "tensorflow\|oneDNN\|GPU\|CUDA\|TensorRT\|NUMA" || true
        
        # Add generated movies to git staging (both MP4 and GIF)
        if [ -d "$output_dir" ]; then
            find "$output_dir" -name "*.mp4" -exec git add {} \; 2>/dev/null || true
            find "$output_dir" -name "*.gif" -exec git add {} \; 2>/dev/null || true
        fi
    done
else
    echo "  TF Example visualization script not found, skipping."
fi

# Count total movies generated
total_movies=$(find "$MOVIES_DIR" -name "*.mp4" 2>/dev/null | wc -l)
total_gifs=$(find "$MOVIES_DIR" -name "*.gif" 2>/dev/null | wc -l)
echo ""
echo "=== Movie Generation Complete ==="
echo "Total MP4 files: $total_movies"
echo "Total GIF files: $total_gifs"
echo ""

exit 0
