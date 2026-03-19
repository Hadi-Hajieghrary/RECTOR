# WOMD Static Scenario Visualizations

This directory contains 60 static PNG renderings of individual Waymo Open Motion Dataset scenarios. Each image provides a single-frame bird's-eye-view snapshot showing the full scenario context: all agent trajectories, road geometry, and traffic infrastructure.

These visualizations serve as a quick inspection tool — allowing researchers to browse scenario content without rendering full videos.

---

## Directory Structure

```text
visualizations/
├── scenario/                          From raw Scenario proto format
│   ├── training_interactive/          15 PNG images
│   ├── validation_interactive/        15 PNG images
│   └── testing_interactive/           15 PNG images
└── tf/                                From processed TFRecord format
    ├── training_interactive/          5 PNG images
    ├── validation_interactive/        5 PNG images
    └── testing_interactive/           5 PNG images
```

---

## What Each Image Shows

Each PNG renders a complete WOMD scenario as a static BEV (bird's-eye-view) plot:

- **Agent trajectories**: Full 9.1-second trajectories for all agents (vehicles, pedestrians, cyclists), color-coded by agent type, with markers at key timesteps
- **Road geometry**: Lane centerlines and road boundaries from the HD map
- **Ego vehicle**: The self-driving car highlighted with a distinct marker
- **Temporal context**: Past trajectories (observed history) and future trajectories (ground truth) distinguished by line style

Unlike the animated videos in `../movies/`, these static images compress the entire temporal evolution into a single frame. This makes them useful for rapid browsing and for inclusion in slide decks or print documents where video is not supported.

---

## Generation

**Script:** `data/WOMD/scripts/lib/visualize_scenario.py`

```bash
python data/WOMD/scripts/lib/visualize_scenario.py \
    --format scenario \
    --split validation_interactive \
    --num 15 \
    --output-dir data/WOMD/visualizations/scenario/validation_interactive/
```

---

## Related Documentation

- [../movies/README.md](../movies/README.md) — Animated BEV videos of the same scenarios
- [../../scripts/lib/README.md](../../scripts/lib/README.md) — Visualization script documentation
