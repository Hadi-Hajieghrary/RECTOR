# IEEE T-IV 2026 Figure Generation Scripts

> Reference note updated: March 16, 2026

## Scripts

### `regenerate_all_figures.py` (PRIMARY -- post-fix)
- **Generates**: 10 figures (selection_comparison, pareto_front_ci, epsilon_sensitivity, ade_fde_distribution, percentile_analysis, violation_reduction, tier_violation_comparison, ablation_radar, component_impact, test_set_generalization)
- **Reads from**: `/workspace/output/experiments/cache/per_mode_data.npz`, `/workspace/Source_Reference/output/evaluation/*.json`
- **Output**: `IEEE_T-IV_2026/Figures/` (PDF + PNG)
- **Status**: Current (March 16, 2026)

### `regenerate_protocol_comparison.py` (post-fix)
- **Generates**: protocol_comparison_heatmap
- **Reads from**: `/workspace/output/experiments/exp2/applicability_ablation.json`, `/workspace/Source_Reference/output/evaluation/canonical_results.json`
- **Output**: `IEEE_T-IV_2026/Figures/protocol_comparison_heatmap.{pdf,png}`
- **Status**: Current (March 16, 2026)

### `create_real_figures.py`
- **Generates**: scenario_gallery, qualitative_example_{highway,intersection,lane_change}
- **Reads from**: `/workspace/Source_Reference/output/evaluation/canonical_results.json`, `/workspace/output/closedloop/bev_frames/`
- **Output**: `IEEE_T-IV_2026/Figures/` (PDF + PNG)
- **Status**: Current (March 15, 2026) -- BEV frames not proxy-dependent

### `generate_all_new_figures.py` (pre-fix, partially superseded)
- **Generates**: 8 figures including per_rule_violation_heatmap, protocol_comparison_heatmap (old), tail_risk, training_loss, waterfall, test_generalization
- **Reads from**: `/workspace/Source_Reference/output/evaluation/`, `/workspace/Source_Reference/output/rule_evaluation/`
- **Status**: per_rule_violation_heatmap still valid (full evaluator, not proxy-dependent); protocol_comparison superseded by `regenerate_protocol_comparison.py`

### `generate_architecture_diagram.py`
- **Generates**: rector_architecture_diagram
- **Status**: Static diagram, no data dependency

### `create_presentation_video.py`
- **Generates**: Presentation video from BEV frames
- **Reads from**: `/workspace/output/closedloop/bev_frames/`

### `regenerate_source_data_and_figures.py`
- **Purpose**: Regenerates source JSON data and then calls `regenerate_all_figures.py`
- **Status**: Utility wrapper

## Regeneration Order

To regenerate all paper figures from scratch:

```bash
cd /workspace/IEEE_T-IV_2026/scripts

# Step 1: Main figures from experiment cache
python3 regenerate_all_figures.py

# Step 2: Protocol comparison heatmap
python3 regenerate_protocol_comparison.py

# Step 3: BEV-based qualitative figures (requires closed-loop frames)
python3 create_real_figures.py
```
