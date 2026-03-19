#!/bin/bash
# Script to run the Converter from Scenario to tf.Example format in parallel
# Usage: ./convert_scenario_tf.sh [INPUT_DIR] [OUTPUT_DIR]

INPUT_DIR=${1:-"/workspace/data/WOMD/datasets/waymo_open_dataset/motion_v_1_3_0/raw/scenario/training_interactive"}
OUTPUT_DIR=${2:-"/workspace/data/WOMD/datasets/waymo_open_dataset/motion_v_1_3_0/processed/tf/training_interactive"}
CONVERTER="/workspace/externals/waymo-open-dataset/src/bazel-bin/waymo_open_dataset/data_conversion/convert_scenario_to_tf_example"

# Pre-flight check for converter binary
if [ ! -f "$CONVERTER" ]; then
    echo "ERROR: Converter binary not found at $CONVERTER"
    echo ""
    echo "Build it first with:"
    echo "  bash /workspace/data/WOMD/src/build.sh"
    exit 1
fi

if [ ! -d "$INPUT_DIR" ]; then
  echo "ERROR: Input directory not found at $INPUT_DIR"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

# Count total files
TOTAL=$(ls "$INPUT_DIR"/*.tfrecord-* 2>/dev/null | wc -l)
if [ "$TOTAL" -eq 0 ]; then
  echo "ERROR: No TFRecord files found in $INPUT_DIR"
  exit 1
fi

echo "Converting $TOTAL files from Scenario to tf.Example format..."
echo "Input:  $INPUT_DIR"
echo "Output: $OUTPUT_DIR"
echo ""

# Convert all files in parallel (6 jobs at a time to avoid memory issues)
export TF_CPP_MIN_LOG_LEVEL=2
ls "$INPUT_DIR"/*.tfrecord-* | parallel --will-cite -j6 --progress \
  "$CONVERTER --input={} --output=$OUTPUT_DIR/{/} 2>/dev/null"