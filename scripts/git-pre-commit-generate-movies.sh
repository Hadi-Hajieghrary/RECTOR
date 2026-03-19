#!/usr/bin/env bash
set -e

echo "=== Pre-commit Hook: Generating Waymo Scenario Movies ==="

# Base directories
DATA_DIR="/workspace/data/WOMD/datasets/waymo_open_dataset/motion_v_1_3_0"
MOVIES_DIR="/workspace/data/WOMD/movies/bev"
MAX_SCENARIOS=5
PYTHON="/opt/venv/bin/python"

# Check if generate_bev_movie.py exists
VISUALIZE_SCRIPT="/workspace/data/WOMD/scripts/lib/generate_bev_movie.py"
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

# Function to generate movies for a specific format and split
generate_movies() {
    local format=$1
    local split=$2

    if [ "$format" == "scenario" ]; then
        local input_dir="${DATA_DIR}/raw/scenario/${split}"
    else
        local input_dir="${DATA_DIR}/processed/tf/${split}"
    fi

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

    # Check if movies already exist
    local existing_movies=$(find "$output_dir" -name "*.mp4" 2>/dev/null | wc -l)
    if [ "$existing_movies" -ge "$MAX_SCENARIOS" ]; then
        echo "  Already have $existing_movies movies (max: $MAX_SCENARIOS), skipping generation."
        return
    fi

    # Generate movies (suppress TensorFlow warnings)
    $PYTHON "$VISUALIZE_SCRIPT" \
        --format "$format" \
        --split "$split" \
        --num "$MAX_SCENARIOS" \
        --output-dir "$output_dir" \
        2>&1 | grep -v "tensorflow\|oneDNN\|GPU\|CUDA\|TensorRT\|NUMA" || true
}

# List of splits to process
SPLITS=(
    "training_interactive"
    "validation_interactive"
    "testing_interactive"
)

# Generate movies for scenario format
echo ""
echo "Generating movies for scenario format..."
for split in "${SPLITS[@]}"; do
    generate_movies "scenario" "$split"
done

# Generate movies for tf format
echo ""
echo "Generating movies for tf format..."
for split in "${SPLITS[@]}"; do
    generate_movies "tf" "$split"
done

echo ""
echo "=== Movie Generation Complete ==="

# Show summary
echo ""
echo "Generated movies:"
find "$MOVIES_DIR" -name "*.mp4" 2>/dev/null | wc -l | xargs echo "  MP4 files:"
find "$MOVIES_DIR" -name "*.gif" 2>/dev/null | wc -l | xargs echo "  GIF files:"

exit 0
