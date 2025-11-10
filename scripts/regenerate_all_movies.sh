#!/bin/bash
# Regenerate all movies with both MP4 and GIF formats

set -e

echo "=== Regenerating all movies with MP4 and GIF formats ==="

# Scenario movies
echo ""
echo "=== Scenario Movies ==="
for split in training validation testing; do
    echo "Processing scenario/$split..."
    python scripts/visualize_waymo_scenario.py \
        --data_dir data/waymo_open_dataset_motion_v_1_3_0/scenario/$split \
        --num_scenarios 10 \
        --output_dir data/movies/scenario/$split \
        --fps 10
done

# TF Example movies
echo ""
echo "=== TF Example Movies ==="
for split in training validation testing; do
    echo "Processing tf_example/$split..."
    python scripts/visualize_waymo_tfexample.py \
        --data_dir data/waymo_open_dataset_motion_v_1_3_0/tf_example/$split \
        --num_scenarios 10 \
        --output_dir data/movies/tf_example/$split \
        --fps 10
done

echo ""
echo "=== Summary ==="
echo "Scenario movies:"
find data/movies/scenario -name "*.mp4" | wc -l | xargs echo "  MP4 files:"
find data/movies/scenario -name "*.gif" | wc -l | xargs echo "  GIF files:"

echo "TF Example movies:"
find data/movies/tf_example -name "*.mp4" | wc -l | xargs echo "  MP4 files:"
find data/movies/tf_example -name "*.gif" | wc -l | xargs echo "  GIF files:"

echo ""
echo "Total size:"
du -sh data/movies/
