# Waymo Open Dataset Visualizations

This directory contains visualizations of sample scenarios from the Waymo Open Motion Dataset v1.3.0.

## File Formats

Each scenario is available in two formats:
- **MP4**: High-quality video format (smaller file size, better compression)
- **GIF**: Animated GIF format (larger file size, viewable directly on GitHub)

## Directory Structure

```
movies/
├── scenario/          # Scenario format visualizations
│   ├── training/     # Training split (up to 10 samples)
│   ├── validation/   # Validation split (up to 10 samples)
│   └── testing/      # Testing split (up to 10 samples)
└── tf_example/       # TF Example format visualizations
    ├── training/     # Training split (up to 10 samples)
    ├── validation/   # Validation split (up to 10 samples)
    └── testing/      # Testing split (up to 10 samples)
```

## Visualization Content

### Scenario Format Movies
- Agent trajectories (past and future)
- Map features (lanes, crosswalks, road edges, speed bumps, stop signs)
- Agent bounding boxes with headings
- Color-coded by trajectory type (history in blue, future in green)

### TF Example Format Movies
- Agent positions over time
- Roadgraph overlay
- Agent states (valid/invalid)
- Bounding boxes with proper dimensions

## Generation

Movies are automatically generated via git pre-commit hook when new data is added.
To manually regenerate all movies:

```bash
./scripts/regenerate_all_movies.sh
```

## Technical Details

- **Frame Rate**: 
  - MP4: 10 FPS
  - GIF: 5 FPS (reduced for smaller file size)
- **Resolution**: 1200x1200 pixels @ 100 DPI
- **Encoding**: 
  - MP4: FFmpeg with H.264 codec
  - GIF: Pillow writer

## File Size Estimates

- Scenario MP4: ~300-750 KB each
- Scenario GIF: ~1-5 MB each
- TF Example MP4: ~50-100 KB each
- TF Example GIF: ~500KB-2MB each

Total estimated size: ~50-100 MB for all movies (61 scenarios × 2 formats)
