# Publication Artifacts

This directory is populated by running the artifact generation pipeline from `models/RECTOR/scripts/artifacts/`. It contains all publication-ready figures and LaTeX tables for the IEEE T-IV 2026 submission.

## Contents

```
artifacts/
├── figures/     24 PDF + 24 PNG figure files (generated at 300 DPI, IEEE-compliant)
├── tables/      10 LaTeX table files (booktabs style, ready for \input{})
├── summary.json Artifact generation log (timestamps, file counts, any warnings)
└── README.md    This file
```

## Generating Artifacts

```bash
cd /workspace/models/RECTOR
python scripts/artifacts/generate_all_artifacts.py
```

This requires evaluation results to be present in `output/evaluation/`. See [`models/RECTOR/scripts/evaluation/README.md`](../../scripts/evaluation/README.md) for the evaluation pipeline.

## Subdirectory Inventories

- [`figures/`](figures/) — 24 figures covering training curves, rule compliance, ablation, error distributions, efficiency, and qualitative examples
- [`tables/`](tables/) — 10 LaTeX tables for main results, ablations, rule catalog, efficiency, and applicability head performance
