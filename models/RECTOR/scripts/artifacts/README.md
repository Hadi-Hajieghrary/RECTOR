# Artifact Generation Scripts

This directory contains the scripts that transform evaluation results and training logs into publication-ready figures and LaTeX tables for the IEEE T-IV 2026 paper.

## Master Orchestrator

`generate_all_artifacts.py` — runs the full pipeline in dependency order, producing 48 figures (PDF + PNG) and 10 LaTeX tables. All outputs are written to `models/RECTOR/output/artifacts/`.

```bash
cd /workspace/models/RECTOR
python scripts/artifacts/generate_all_artifacts.py
```

## Individual Scripts

| Script | Output | Description |
|--------|--------|-------------|
| `generate_main_results_table.py` | `tables/main_results.tex` | minADE, minFDE, miss rate, violation rates |
| `generate_training_curves.py` | `figures/stage1_training.png`, `stage2_training.png`, `training_combined.png`, `learning_rate.png` | Loss and metric curves for both training stages |
| `generate_rule_violation_analysis.py` | `figures/tier_violation_comparison.png`, `violation_reduction.png`, `rule_breakdown.png` | Rule compliance analysis |
| `generate_applicability_analysis.py` | `figures/applicability_heatmap.png`, `applicability_tier_f1.png` | Applicability head performance |
| `generate_ablation_study.py` | `figures/ablation_comparison.png`, `ablation_radar.png`, `component_impact.png` | Component ablation |
| `generate_distribution_plots.py` | `figures/ade_fde_distribution.png`, `percentile_analysis.png`, `miss_rate_analysis.png`, `ade_fde_heatmap.png` | Error distribution analysis |
| `generate_efficiency_stats.py` | `figures/latency_breakdown.png`, `parameter_breakdown.png`, `throughput_comparison.png` | Efficiency metrics |
| `generate_qualitative_visualizations.py` | `figures/qualitative_example_*.png`, `scenario_gallery.png`, `selection_comparison.png` | Qualitative scenario examples |

## Prerequisites

Evaluation results must be present in `output/evaluation/` before running artifact generation. See [`models/RECTOR/scripts/evaluation/README.md`](../evaluation/README.md) for instructions on producing those results.
