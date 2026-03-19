# Building the Scenario→TF Conversion Tool

## Overview

The conversion tool source code is in `/workspace/data/WOMD/src/` but must be built from the Waymo repository workspace to link against Waymo's libraries.

## Why This Approach?

- ✅ Keeps our custom code separate from the external Waymo repository
- ✅ Doesn't modify files in `externals/waymo-open-dataset/`
- ✅ Still gets full access to Waymo's conversion libraries
- ✅ Makes it clear which code is ours vs. external

## Build Instructions

### Prerequisites

1. **Bazel 5.4.0** (already installed in devcontainer)
2. **Waymo Open Dataset repository** at `/workspace/externals/waymo-open-dataset/`

### Build Steps

```bash
# First, copy our source to Waymo's directory (temporary, for build only)
cp /workspace/data/WOMD/src/convert_scenario_to_tf_example.cc \
   /workspace/externals/waymo-open-dataset/src/waymo_open_dataset/data_conversion/

# Navigate to Waymo's source directory
cd /workspace/externals/waymo-open-dataset/src

# Build the conversion tool
bazel build //waymo_open_dataset/data_conversion:convert_scenario_to_tf_example
```

**Note:** We copy the source temporarily for the build. Our canonical source remains in `/workspace/data/WOMD/src/`. The Waymo repository's BUILD file already has the configuration we need.

### Build Output

Binary location:
```
/workspace/externals/waymo-open-dataset/src/bazel-bin/waymo_open_dataset/data_conversion/convert_scenario_to_tf_example
```

### Verify Build

```bash
# Check binary exists
ls -lh /workspace/externals/waymo-open-dataset/src/bazel-bin/waymo_open_dataset/data_conversion/convert_scenario_to_tf_example

# Test conversion
/workspace/externals/waymo-open-dataset/src/bazel-bin/waymo_open_dataset/data_conversion/convert_scenario_to_tf_example \
  --input=/workspace/data/WOMD/datasets/waymo_open_dataset/motion_v_1_3_0/raw/scenario/testing_interactive/testing_interactive.tfrecord-00000-of-00150 \
  --output=/tmp/test_output.tfrecord
```

## Source Code Organization

```
/workspace/
├── data/
│   └── WOMD/
│       └── src/                          # Our custom code
│           ├── convert_scenario_to_tf_example.cc
│           ├── BUILD
│           └── README.md
│
└── externals/
    └── waymo-open-dataset/
        └── src/
            ├── WORKSPACE             # Bazel workspace (for building)
            ├── waymo_open_dataset/
            │   └── data_conversion/
            │       ├── scenario_conversion.h    # Waymo's library
            │       └── scenario_conversion.cc   # (we link against this)
            └── bazel-bin/
                └── waymo_open_dataset/
                    └── data_conversion/
                        └── convert_scenario_to_tf_example  # Built binary
```

## Automated Build Script

We provide a build script that automates the entire process:

```bash
/workspace/data/WOMD/src/build.sh
```

This script:
1. Copies `convert_scenario_to_tf_example.cc` to Waymo's directory
2. Builds using Bazel
3. Verifies the binary was created

Make it executable: `chmod +x /workspace/data/WOMD/src/build.sh`

## Troubleshooting

### Build Fails with "Target not found"

Make sure you're building from the Waymo workspace:
```bash
cd /workspace/externals/waymo-open-dataset/src
bazel build //waymo_open_dataset/data_conversion:convert_scenario_to_tf_example
```

### Build Fails with Missing Dependencies

Clean and rebuild:
```bash
cd /workspace/externals/waymo-open-dataset/src
bazel clean
bazel build //waymo_open_dataset/data_conversion:convert_scenario_to_tf_example
```

### Binary Not Found After Build


Check the bazel-bin directory:
```bash
find /workspace/externals/waymo-open-dataset/src/bazel-bin -name "convert_scenario_to_tf_example" -type f
```

## Integration with Pipeline

The batch conversion wrapper automatically uses the built binary:

```bash
# The convert_scenario_tf.sh script uses the binary at:
CONVERTER="/workspace/externals/waymo-open-dataset/src/bazel-bin/waymo_open_dataset/data_conversion/convert_scenario_to_tf_example"
```

Just make sure the binary is built before running:
```bash
cd /workspace
bash data/WOMD/scripts/bash/convert_scenario_tf.sh
```

