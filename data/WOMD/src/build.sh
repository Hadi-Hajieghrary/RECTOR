#!/bin/bash
# Build script for convert_scenario_to_tf_example tool
#
# This script:
# 1. Copies our source code to Waymo's directory (temporary)
# 2. Builds the tool using Waymo's Bazel workspace
# 3. Verifies the binary was created

set -e

echo "================================"
echo "Building convert_scenario_to_tf_example"
echo "================================"
echo ""

# Source and destination paths
SRC_FILE="/workspace/data/WOMD/src/convert_scenario_to_tf_example.cc"
DEST_DIR="/workspace/externals/waymo-open-dataset/src/waymo_open_dataset/data_conversion"
DEST_FILE="${DEST_DIR}/convert_scenario_to_tf_example.cc"
WAYMO_SRC="/workspace/externals/waymo-open-dataset/src"
BINARY="${WAYMO_SRC}/bazel-bin/waymo_open_dataset/data_conversion/convert_scenario_to_tf_example"

# Step 1: Copy source
echo "Step 1: Copying source file and configuring build..."
echo "  From: ${SRC_FILE}"
echo "  To:   ${DEST_FILE}"
cp "${SRC_FILE}" "${DEST_FILE}"

# Append BUILD target if not present
if ! grep -q "name = \"convert_scenario_to_tf_example\"" "${DEST_DIR}/BUILD"; then
    echo "  Appending build target to ${DEST_DIR}/BUILD..."
    cat "/workspace/data/WOMD/src/BUILD" >> "${DEST_DIR}/BUILD"
fi

echo "  ✓ Source copied and build configured"
echo ""

# Step 2: Build with Bazel
echo "Step 2: Building with Bazel..."
echo "  Workspace: ${WAYMO_SRC}"
echo "  Target: //waymo_open_dataset/data_conversion:convert_scenario_to_tf_example"
cd "${WAYMO_SRC}"
bazel build //waymo_open_dataset/data_conversion:convert_scenario_to_tf_example
echo "  ✓ Build complete"
echo ""

# Step 3: Verify binary
echo "Step 3: Verifying binary..."
if [ -f "${BINARY}" ]; then
    echo "  ✓ Binary created successfully"
    echo "  Location: ${BINARY}"
    ls -lh "${BINARY}"
else
    echo "  ✗ ERROR: Binary not found at expected location"
    echo "  Expected: ${BINARY}"
    exit 1
fi

echo ""
echo "================================"
echo "Build successful!"
echo "================================"
echo ""
echo "To use the converter:"
echo "  ${BINARY} \\"
echo "    --input=input.tfrecord \\"
echo "    --output=output.tfrecord"
echo ""
