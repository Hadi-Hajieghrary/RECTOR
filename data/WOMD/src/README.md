# `data/WOMD/src/` — Bazel C++ Converter (Scenario TFRecord to TF Example TFRecord)

This folder contains a small C++ tool for converting Waymo Motion **Scenario** TFRecords
(each record is a `scenario_pb2.Scenario` proto) into the **Motion TF Example** format
(each record is a `tf.train.Example`), which is the format used by M2I and the visualization scripts.

Why use a C++ tool?
- Waymo's official conversion utilities are implemented in C++ and are significantly faster than pure Python
- The Motion TF Example feature layout has many fields; using Waymo's existing conversion library reduces risk

---

## Files

### `convert_scenario_to_tf_example.cc`
A minimal `main()` wrapper around Waymo's scenario conversion library.

What it does (in detail):
1. Reads a single input TFRecord file (`--input`)
2. Iterates record-by-record:
   - Parses each record into `scenario_pb2::Scenario`
   - Converts it into a `tf::Example` using Waymo's conversion API
3. Writes each converted example to the output TFRecord file (`--output`)

Flags:
- `--input`: Input TFRecord file path (Scenario protos)
- `--output`: Output TFRecord file path (TF Examples)

The converter is intentionally single-file; directory-wide conversion is typically done with
a bash wrapper (`data/WOMD/scripts/bash/convert_scenario_tf.sh`) that runs the binary on many files.

---

### `BUILD`
Bazel target definition.

Key point:
- The target depends on external Bazel repositories like `@waymo_open_dataset` and `@org_tensorflow`

That means:
- This folder is expected to be built **inside a larger Bazel workspace** that already defines those
  external repos (via a `WORKSPACE` file somewhere above this folder)

If you try to run `bazel build` here without a workspace that defines those repos, it will fail.

---

### `build.sh`
A convenience build wrapper.

What it does:
1. Copies the source file to the Waymo workspace at `/workspace/externals/waymo-open-dataset/src/waymo_open_dataset/data_conversion/`
2. Builds the target there via Bazel (`bazel build`)
3. The resulting binary ends up at `/workspace/externals/waymo-open-dataset/src/bazel-bin/waymo_open_dataset/data_conversion/convert_scenario_to_tf_example`
4. Prints a short “how to run” message

This script assumes:
- You have Bazel installed
- You're in an environment where `/workspace` is writable
- The parent Bazel workspace correctly defines the external repos

---

## Relationship to the Bash Wrapper `convert_scenario_tf.sh`

There are **two potential converter binaries** referenced in this repository:

1. The binary built from this folder (`build.sh` output):
   - `/workspace/externals/waymo-open-dataset/src/bazel-bin/waymo_open_dataset/data_conversion/convert_scenario_to_tf_example`

2. A binary from an external waymo-open-dataset checkout:
   - `/workspace/externals/waymo-open-dataset/src/bazel-bin/waymo_open_dataset/data_conversion/convert_scenario_to_tf_example`

The wrapper `data/WOMD/scripts/bash/convert_scenario_tf.sh` currently points to (2).

If you want to use the binary built by `data/WOMD/src/build.sh`, update the wrapper to point to the new location.

---

## Typical usage (single file)

```bash
/workspace/externals/waymo-open-dataset/src/bazel-bin/waymo_open_dataset/data_conversion/convert_scenario_to_tf_example \
    --input /path/to/input.tfrecord-00000-of-01000 \
    --output /path/to/output.tfrecord-00000-of-01000
```

For directory-wide conversion, run the bash wrapper from `/workspace`:

```bash
cd /workspace
bash data/WOMD/scripts/bash/convert_scenario_tf.sh
```

(Ensure GNU `parallel` is installed for parallel processing.)
