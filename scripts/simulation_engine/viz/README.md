# Closed-Loop Visualization and Analysis

This module generates all visual artifacts from closed-loop simulation results: publication-quality statistical plots, LaTeX tables, per-scenario safety profiles, and animated bird's-eye-view (BEV) rollout videos. All outputs target IEEE publication standards (300 DPI, serif fonts, appropriate column widths).

---

## Modules

### generate_all.py — Master Orchestrator

**Purpose:** Single entry point that runs all visualization and analysis scripts in sequence.

**Pipeline:**
1. Metric distributions (violin + grid plots)
2. Safety profiles (per-scenario horizontal bars)
3. Summary tables (LaTeX + Markdown)
4. Scenario heatmaps (scenario × metric matrix)
5. BEV rollout videos (optional, slow)

```bash
python scripts/simulation_engine/viz/generate_all.py \
    --results /workspace/output/closedloop/validate_50_results.json \
    --outdir /workspace/output/closedloop/artifacts/ \
    --selector Confidence
```

### bev_rollout.py — Simple BEV Animation

**Purpose:** Generates frame-by-frame 2D top-down animations of closed-loop scenario execution.

**Visualization elements:**
- Road polylines color-coded by type (lane center=#888888 gray, road line=#aaaaaa light gray dashed, road edge=#555555 dark gray, crosswalk=#ddbb44 yellow-gold, stop sign=#cc0000 red)
- Ego vehicle as a green rectangle (turns red on collision)
- Agent bounding boxes as blue oriented rectangles
- Logged human trajectory as a dashed cyan line
- Executed RECTOR trajectory as a solid green line
- Metrics overlay (overlap rate, step index)

**Key functions:**
- `render_frame()` — Renders a single BEV frame with all elements
- `_extract_road_polylines()` — Groups roadgraph points by segment ID into drawable polylines
- `render_scenario_rollout()` — Complete pipeline: load → simulate → render → stitch to MP4

**Configuration** via `BEVConfig` dataclass:
- View range: 60m half-side
- Figure size: 6×6 inches at 150 DPI
- Vehicle dimensions: ego 4.8×2.0m, agent 4.5×1.9m

### enhanced_bev_rollout.py — Dual-Panel BEV with Rule Violations

**Purpose:** The primary deliverable for supplementary material — 50 diverse BEV videos with a violations panel showing why RECTOR selects each trajectory.

**Dual-panel format:**
- **Left panel**: BEV scene with 6 candidate trajectories ahead of ego, selected trajectory highlighted (thick + arrow), executed trail, logged trail, agents, road geometry
- **Right panel**: Grouped bar chart showing per-tier violation costs for each candidate, with the selected candidate indicated by a bold border

**Scenario mining** (`mine_interesting_scenarios()`):
- Scans up to 1,600 WOMD scenarios by default (shards_to_scan=40 × scenarios_per_shard=40) across multiple TFRecord shards
- Classifies each ego trajectory into maneuver categories (TURN, LANE_CHANGE, EXIT_RAMP, COMPLEX, OTHER) based on heading change, lateral displacement, speed variation, and lateral reversals
- Samples proportionally: 30% turns, 30% lane changes, 20% exit ramps, 15% complex, 5% other

**Trajectory visualization** (`candidates_to_world()`):
- Converts model output from ego-local normalized [K,T,4] to world coordinates [K,T,2]
- Pipeline: denormalize (×50) → rotate by ego_yaw → translate by ego_pos

**Output:** 50 MP4 files at `/workspace/output/closedloop/videos/` + `enhanced_summary.json`

See [../../../output/closedloop/videos/README.md](../../../output/closedloop/videos/README.md) for detailed documentation of the generated videos.

### generate_bev_batch.py — Batch BEV Movie Generator

**Purpose:** Generates closed-loop BEV rollout movies for multiple scenarios in batch. Loads all scenarios once, then renders each one with frame-by-frame simulation and stitches into MP4 videos.

**Key features:**
- Configurable number of scenarios (default: 50)
- Resumable via `--start-idx`
- Uses simple single-panel BEV format (from `bev_rollout.py`)
- Produces per-scenario MP4 files plus a `summary.json` manifest

**Usage:**
```bash
python -m simulation_engine.viz.generate_bev_batch \
    --num-scenarios 50 \
    --outdir /workspace/output/closedloop/videos \
    --max-steps 80
```

### generate_bev_challenging.py — Challenging Infrastructure BEV Movies

**Purpose:** Mines WOMD validation shards for scenarios with complex road infrastructure (intersections, crosswalks, stop signs, speed bumps, dense lane networks), then renders closed-loop BEV movies with full road geometry.

**Scenario mining** (`mine_complex_scenarios()`):
- Scores scenarios by infrastructure complexity: crosswalks, stop signs, speed bumps, traffic lights, lane count, agent count
- Selects top-N most complex scenarios by composite score

**Usage:**
```bash
python -m simulation_engine.viz.generate_bev_challenging \
    --target-movies 50 \
    --shards-to-scan 30 --scenarios-per-shard 30 \
    --outdir /workspace/output/closedloop/videos
```

### metric_distributions.py — Statistical Distribution Plots

**Purpose:** Publication-quality violin and box plots for 6 hero metrics across all simulated scenarios.

**Hero metrics:**
1. Overlap rate — collision frequency
2. Log divergence (mean) — deviation from human driver
3. Kinematic infeasibility rate — physics violation frequency
4. Minimum clearance — closest approach to any object
5. Mean jerk — smoothness indicator
6. Maximum jerk — worst-case smoothness

**Outputs:**
- `metric_hero_violin.{pdf,png}` — 1×6 violin plot with median lines and scatter overlay
- `metric_full_grid.{pdf,png}` — 3×4 grid of box plots with all 11 metrics and mean/std annotations

### scenario_heatmap.py — Scenario × Metric Matrix

**Purpose:** Compact overview showing how each scenario performs across 7 key metrics simultaneously.

**Format:** Normalized heatmap with scenarios (rows) sorted by aggregate score (worst at top). For "lower is better" metrics, the normalization is inverted so that red always means bad. Cell values show actual metric values while color indicates relative performance.

**Output:** `scenario_metric_heatmap.{pdf,png}`

### scenario_safety_profile.py — Per-Scenario Safety Assessment

**Purpose:** Identifies which specific scenarios have safety concerns and quantifies severity.

**Outputs:**
- `scenario_safety_profile.{pdf,png}` — 3-panel horizontal bar chart (overlap rate, log divergence, min clearance) with color-coded severity (green=safe, amber=marginal, red=unsafe)
- `scenario_safety_pie.{pdf,png}` — Pie chart: collision-free vs. minor overlap vs. significant overlap

### summary_table.py — LaTeX/Markdown Tables

**Purpose:** Generates publication-ready summary statistics tables.

**Statistics computed:** n, mean, std, median, Q1, Q3, min, max for 10 key metrics.

**Outputs:**
- `closedloop_stats_table.tex` — LaTeX tabular with booktabs style
- `closedloop_stats_table.md` — GitHub-flavored Markdown
- `closedloop_summary.txt` — Human-readable text summary with collision-free count

---

## IEEE Publication Formatting

All figures follow IEEE conference/journal standards:
- **DPI:** 300
- **Fonts:** Serif (Times), 8pt base size
- **Column width:** 3.5 inches (single column) or 7.16 inches (double column)
- **Color palette:** Colorblind-safe where possible
- **Layout:** Tight with minimal whitespace

---

## Related Documentation

- [../README.md](../README.md) — Simulation engine overview
- [../../../output/closedloop/videos/README.md](../../../output/closedloop/videos/README.md) — Generated video documentation
- [../../../models/RECTOR/output/artifacts/README.md](../../../models/RECTOR/output/artifacts/README.md) — Publication artifact inventory
