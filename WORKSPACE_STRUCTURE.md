# RECTOR Workspace Structure

Auto-generated workspace structure (updated on commit).

## Workspace Tree

```
./
├── assets/
│   ├── frames/
│   │   ├── closedloop_010_frame.png
│   │   ├── closedloop_complex_frame.png
│   │   ├── closedloop_exitramp_frame.png
│   │   ├── closedloop_lanechange_frame.png
│   │   ├── closedloop_turn_frame.png
│   │   ├── m2i_rector_000_frame.png
│   │   ├── m2i_rector_020_frame.png
│   │   ├── m2i_rector_frame.png
│   │   └── m2i_rector_intersection_frame.png
│   └── gifs/
│       ├── closedloop_complex.gif
│       ├── closedloop_lane_change.gif
│       ├── closedloop_turn.gif
│       └── m2i_rector_planning.gif
├── .claude/
│   └── settings.local.json
├── data/
│   ├── WOMD/
│   │   ├── datasets/
│   │   ├── movies/
│   │   ├── scripts/
│   │   ├── src/
│   │   ├── visualizations/
│   │   └── waymo_rule_eval/
│   ├── DATA_INVENTORY.md
│   └── README.md
├── .devcontainer/
│   ├── scripts/
│   │   ├── setup-externals.sh*
│   │   ├── setup_git_hooks.sh*
│   │   └── validate-waymax.py
│   ├── devcontainer.json
│   ├── docker-compose.yml
│   ├── Dockerfile
│   ├── fix-permissions.sh*
│   ├── post-create.sh*
│   ├── README.md
│   ├── requirements.base.txt
│   └── requirements.project.txt
├── experiments/
├── externals/
│   ├── M2I/
│   │   ├── configs/
│   │   ├── scripts/
│   │   ├── src/
│   │   ├── conda.cuda111.yaml
│   │   ├── .gitignore
│   │   ├── LICENSE
│   │   ├── README.md
│   │   └── requirements.txt
│   ├── waymo-open-dataset/
│   │   ├── docs/
│   │   ├── src/
│   │   ├── tutorial/
│   │   ├── CONTRIBUTING.md
│   │   ├── .gitignore
│   │   ├── LICENSE
│   │   ├── LOG.md
│   │   └── README.md
│   └── README.md
├── logs/
│   └── rector_training.log
├── models/
│   ├── checkpoints/
│   ├── pretrained/
│   │   ├── m2i/
│   │   ├── .gitkeep
│   │   └── README.md
│   ├── RECTOR/
│   │   ├── checkpoints/
│   │   ├── docs/
│   │   ├── models/
│   │   ├── movies/
│   │   ├── output/
│   │   ├── scripts/
│   │   ├── tests/
│   │   └── README.md
│   └── README.md
├── notebooks/
├── output/
│   ├── closedloop/
│   │   ├── bev_frames/
│   │   └── videos/
│   ├── evaluation/
│   │   ├── always_on_cross_results.json
│   │   ├── always_on_results.json
│   │   ├── canonical_results.json
│   │   ├── evaluation_results.json
│   │   ├── hybrid_conservative_results.json
│   │   ├── hybrid_cross_results.json
│   │   ├── learned_cross_results.json
│   │   ├── learned_protB_results.json
│   │   ├── per_scenario_confidence.csv
│   │   ├── per_scenario_lexicographic.csv
│   │   ├── per_scenario_metrics.csv
│   │   ├── per_scenario_protB_confidence.csv
│   │   ├── per_scenario_protB_lexicographic.csv
│   │   ├── per_scenario_protB_weighted_sum.csv
│   │   ├── per_scenario_weighted_sum.csv
│   │   └── val_test_distribution.json
│   ├── app_head_fresh_log.txt
│   ├── artifact_generation_log.txt
│   ├── canonical_eval_log.txt
│   ├── README.md
│   ├── rector_continue_5ep_log.txt
│   ├── rector_full_log.txt
│   └── simple_eval_log.txt
├── reference/
│   └── IEEE_T-IV_2026/
│       ├── docs/
│       ├── Figures/
│       ├── presentation/
│       ├── Reviews/
│       ├── scripts/
│       ├── Sections/
│       ├── experiment_artifacts.tex
│       ├── ieeeconf.cls
│       ├── IEEEtran.bst
│       ├── IEEEtran.cls
│       ├── Main.aux
│       ├── Main.bbl
│       ├── Main.blg
│       ├── Main.fdb_latexmk
│       ├── Main.fls
│       ├── Main.log
│       ├── Main.out
│       ├── Main.pdf
│       ├── Main.synctex.gz
│       └── Main.tex
├── scripts/
│   ├── analysis/
│   │   └── val_test_distribution_compare.py
│   ├── simulation_engine/
│   │   ├── selectors/
│   │   ├── viz/
│   │   ├── waymax_bridge/
│   │   ├── config.py
│   │   ├── __init__.py
│   │   ├── README.md
│   │   └── validate_50.py
│   ├── WOMD/
│   │   ├── check_waymo_status.sh*
│   │   ├── clear_waymo_data.sh*
│   │   ├── download_waymo_sample.sh*
│   │   └── README.md
│   ├── git-pre-commit-generate-movies.sh*
│   ├── git-pre-commit-hook.sh*
│   └── README.md
├── src/
│   └── scenario_converter.cc
├── .gitignore
├── LICENSE
├── README.md
└── WORKSPACE_STRUCTURE.md

59 directories, 97 files
```

*Last updated: 2026-03-18 07:40:53 UTC*
