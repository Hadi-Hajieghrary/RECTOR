# RECTOR: Waymo Motion Data Processing and ITP Prep

This repository organizes data utilities and references for working with the Waymo Open Motion Dataset v1.3.0, with an emphasis on preprocessing for Interactive Trajectory Planning (ITP) and light visualization/testing. It includes:

- A small, friendly CLI for data flows (download → verify → filter → preprocess → visualize)
- Modular bash scripts and Python modules for each step
- Reference materials for environment/devcontainer and external baselines

## Quick start

Prerequisites:
- Linux with bash
- Google Cloud SDK (gsutil) and acceptance of Waymo license
- Python 3.10+ with numpy, tensorflow (CPU ok), torch, matplotlib, tqdm

From the repository root:

```bash
# 1) Download a small sample (default: 5 files/partition)
./data/scripts/waymo download

# 2) Verify layout
./data/scripts/waymo verify

# 3) Preprocess training split into compressed .npz files
./data/scripts/waymo process --split training --interactive-only --workers 8

# 4) Generate a few movies to sanity-check
./data/scripts/waymo visualize --format scenario --split training --num 10

# 5) Summary
./data/scripts/waymo status
```

One-shot pipeline:
```bash
./data/scripts/waymo pipeline --num 5 --workers 8 --viz 10
```

## Where things live

- Data roots: `data/datasets/waymo_open_dataset/motion_v_1_3_0/{raw,processed}`
- Movies: `data/movies/waymo_open_dataset/motion_v_1_3_0/`
- CLI + scripts: `data/scripts/{waymo,bash/,lib/}`
- Tests: `data/tests/`
- References: `References/` (devcontainer, docs, migration notes)
- Externals: `externals/` (Waymo API and M2I baseline code)

See `data/README.md` for a detailed end-to-end guide, parameters, and troubleshooting.

## External components

- `externals/waymo-open-dataset/`: Vendor code for Waymo protos and docs
- `externals/M2I/`: Baseline modeling code referenced for ITP workflows

These folders are not modified by the CLI but may be useful as references.

## Development notes

- The CLI is a thin wrapper over bash/Python modules; you can call those directly if preferred.
- Make sure gsutil can access the Waymo buckets after license acceptance.
- For data-heavy operations, start small (few files, workers=2) to validate paths and environment.

## License

See `LICENSE`.
