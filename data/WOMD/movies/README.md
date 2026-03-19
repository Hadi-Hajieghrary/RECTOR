# WOMD Bird's-Eye-View Scenario Videos

This directory contains 120 bird's-eye-view (BEV) animation videos generated from raw Waymo Open Motion Dataset scenarios. These videos provide visual context for the driving situations that RECTOR is trained and evaluated on — they show the raw data before any prediction or rule evaluation is applied.

---

## Directory Structure

```text
movies/bev/
├── scenario/                          From raw Scenario proto format
│   ├── training_interactive/          20 videos from training split
│   ├── validation_interactive/        20 videos from validation split
│   └── testing_interactive/           20 videos from testing split
└── tf/                                From processed TFRecord format
    ├── training_interactive/          20 videos from training split
    ├── validation_interactive/        20 videos from validation split
    └── testing_interactive/           20 videos from testing split
```

**Total:** 120 videos (60 scenario format + 60 TFRecord format)

---

## What Each Video Shows

Each animation renders the full 9.1-second WOMD scenario (91 timesteps at 10 Hz) in a top-down view:

- **All agents**: Vehicles, pedestrians, and cyclists rendered as oriented rectangles with color-coded trajectories
  - Blue: Vehicle trajectories
  - Green: Cyclist trajectories
  - Orange: Pedestrian trajectories
- **Road geometry**: Lane centerlines, road edges, and boundaries from the HD map
- **Traffic infrastructure**: Crosswalks, stop signs, and speed bumps where available
- **Temporal evolution**: Frame-by-frame animation showing how the scene develops over the 9.1-second window

### Scenario Format vs TFRecord Format

Both formats represent the same underlying WOMD data, but are parsed through different code paths:

- **Scenario format** (`scenario/`): Parsed from Waymo's `Scenario` protobuf, preserving the full scenario metadata including traffic signal states and scenario IDs. Uses `data/WOMD/scripts/lib/generate_bev_movie.py` with `--format scenario`.

- **TFRecord format** (`tf/`): Parsed from the processed TFRecord representation. This format mirrors what the training pipeline sees. Uses `generate_bev_movie.py` with `--format tf`.

Having both formats visualized helps verify that data conversion preserves scenario content.

---

## Interactive Scenarios

All videos are from the **interactive** subsets of WOMD, which contain scenarios with meaningful multi-agent interactions (vehicles negotiating right-of-way, pedestrians crossing in traffic, lane changes requiring gap acceptance, etc.). Non-interactive scenarios (isolated vehicles on empty roads) are filtered out during dataset preparation.

This focus on interactive scenarios is intentional: RECTOR's rule-aware selection is most valuable precisely when multiple agents interact and traffic rules must be considered.

---

## Generation

**Script:** `data/WOMD/scripts/lib/generate_bev_movie.py`

**Triggered by:** The git pre-commit hook (`scripts/git-pre-commit-generate-movies.sh`) automatically generates up to 5 videos per format/split combination on each commit, ensuring the repository always contains representative visualizations.

```bash
# Manual generation
python data/WOMD/scripts/lib/generate_bev_movie.py \
    --format scenario \
    --split validation_interactive \
    --num 20 \
    --output-dir data/WOMD/movies/bev/scenario/validation_interactive/
```

---

## Related Documentation

- [../../scripts/lib/README.md](../../scripts/lib/README.md) — BEV generation script documentation
- [../../../models/RECTOR/movies/README.md](../../../models/RECTOR/movies/README.md) — RECTOR planning demonstration videos (with rule evaluation overlay)
- [../../../output/closedloop/videos/README.md](../../../output/closedloop/videos/README.md) — Enhanced BEV movies with violation panels
