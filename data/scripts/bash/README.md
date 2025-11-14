# data/scripts/bash

Shell scripts orchestrating dataset lifecycle steps.

| Script | Purpose |
|--------|---------|
| `authenticate.sh` | Establish credentials / tokens for accessing licensed Waymo data. |
| `download.sh` | Batch download raw dataset shards with resume logic. |
| `filter.sh` | Apply scenario selection criteria (agent count, types, horizon). |
| `process.sh` | Convert raw shards into processed NPZ format (invokes Python libs). |
| `verify.sh` | Run integrity checks and shape validations post-processing. |
| `visualize.sh` | Bulk generation of raster and trajectory visual artifacts.

## Execution Order
1. `authenticate.sh`
2. `download.sh`
3. `filter.sh` (optional)
4. `process.sh`
5. `verify.sh`
6. `visualize.sh` (optional)

## Environment Variables
- `DATA_ROOT`: Destination directory for raw and processed data.
- `NUM_WORKERS`: Parallelism for processing.
