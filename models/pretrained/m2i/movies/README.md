# M2I Receding Horizon Movies

> **Directory:** `models/pretrained/m2i/movies/receding_horizon/`
> **Purpose:** Animated visualizations of M2I standalone receding horizon predictions
> **Producer:** `models/pretrained/m2i/scripts/lib/m2i_receding_horizon_full.py`

---

## Content

Each file pair (`{scenario_id}_receding_horizon.mp4` + `.gif`) shows M2I's receding-horizon prediction for one Waymo interactive scenario:

- **Per-frame**: At each timestep t, uses history [t-10:t] to predict 6 modes of future [t+1:t+80]
- **BEV rendering**: Road polylines (lanes, edges, crosswalks), oriented agent boxes, prediction trajectories with confidence colormap
- **Agents**: Influencer (red), reactor (orange), other vehicles (blue), pedestrians (green), cyclists (purple)
- **Predictions**: 6 modes colored by plasma colormap, linewidth proportional to score, endpoint markers

## How They're Created

```bash
bash models/pretrained/m2i/scripts/bash/run_m2i_pipeline.sh \
    -n 10 -m /workspace/models/pretrained/m2i/movies/receding_horizon
```

## Relationship to RECTOR Movies

These are **M2I standalone baseline** movies — showing what M2I predicts without RECTOR's rule-aware selection layer. Compare with:
- `models/RECTOR/movies/` — RECTOR predictions (with rule-based trajectory selection)
- `models/RECTOR/movies/receding_v2/` — RECTOR receding horizon predictions
