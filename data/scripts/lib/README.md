# data/scripts/lib

Python helper modules invoked by bash scripts for dataset manipulation and visualization.

| File | Role |
|------|------|
| `waymo_dataset.py` | Raw dataset parsing utilities (legacy vendor-style loader). |
| `waymo_preprocess.py` | Feature extraction and transformation pipeline (normalize, center, filter lanes). |
| `filter_interactive_training.py` | Selects scenarios with interactive agent dynamics for focused training. |
| `viz_waymo_tfexample.py` | Renders raw TFExample contents for inspection. |
| `viz_waymo_scenario.py` | Draws map + agent trajectories over time for processed NPZ scenario. |
| `viz_trajectory.py` | Compares predicted multi-modal trajectories vs ground truth.

## Common Patterns
- Functions accept a scenario dict and output numpy arrays or matplotlib figures.
- Normalization relies on `normalizer` objects embedded in mapping.
