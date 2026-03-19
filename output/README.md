# RECTOR Evaluation and Simulation Outputs

This directory contains all generated outputs from RECTOR's evaluation pipeline, closed-loop simulation experiments, and training logs. These outputs constitute the empirical evidence base for the paper's quantitative claims.

Every result here is reproducible from source scripts in the repository. No output is manually curated — each file traces back to a specific script invocation documented below.

This directory is best read as a derived-artifacts area rather than hand-authored documentation. Source code and analysis logic live elsewhere; this folder preserves the outputs they emit.

---

## Directory Structure

```text
output/
├── evaluation/                        Quantitative evaluation results
│   ├── canonical_results.json         Single source of truth for all paper metrics
│   ├── evaluation_results.json        Standard open-loop evaluation metrics
│   ├── per_scenario_metrics.csv       Per-scenario ADE/FDE/miss (lexicographic)
│   ├── per_scenario_confidence.csv    Per-scenario metrics (confidence selection)
│   ├── per_scenario_weighted_sum.csv  Per-scenario metrics (weighted-sum selection)
│   └── per_scenario_lexicographic.csv Per-scenario metrics (lexicographic selection)
├── closedloop/                        Closed-loop simulation outputs
│   └── videos/                        55 enhanced BEV rollout videos
│       ├── enhanced_summary.json      Generation summary with per-video metadata
│       └── scenario_*.mp4             Individual scenario rollout videos
├── artifact_generation_log.txt        Log from generate_all_artifacts.py
├── canonical_eval_log.txt             Log from canonical evaluation run
├── app_head_fresh_log.txt             Stage 1: applicability head training log
├── rector_full_log.txt                Stage 2: full RECTOR end-to-end training log
├── rector_continue_5ep_log.txt        Continuation training log
├── simple_eval_log.txt                Simple evaluation log
└── README.md                          This file
```

---

## Evaluation Results

### canonical_results.json — Canonical Evaluation Output

**Produced by:** `models/RECTOR/scripts/evaluation/evaluate_canonical.py`

This is the main evaluation record used for the paper tables. Reported quantitative results should trace back to this file. It evaluates 12,800 validation scenarios under multiple protocol combinations.

**Contents:**

| Section | Description |
|---------|-------------|
| `metadata` | Checkpoint hash, evaluation config, rule counts, timestamp |
| `geometric_metrics` | Oracle (best-of-6) minADE, minFDE, miss rate |
| `selection_strategies` | Per-strategy results for confidence, weighted-sum, and lexicographic selection |
| `protocol_results` | Cross-product: {proxy-24, full-28} × {predicted, oracle} applicability |
| `per_rule_applicability_metrics` | Per-rule precision, recall, F1, accuracy, support for all 28 rules |

**Key Results (12,800 scenarios):**

| Metric | Oracle (best-of-6) | Confidence | Weighted Sum | Lexicographic |
|--------|-------------------|------------|--------------|---------------|
| minADE / selADE | 0.684 m | 2.208 m | 2.043 m | 2.043 m |
| minFDE / selFDE | 1.270 m | 4.330 m | 3.813 m | 3.813 m |
| Miss Rate | 18.56% | — | — | — |
| Total Violations | — | 23.86% | 15.03% | 15.03% |
| Safety+Legal Violations | — | 21.87% | 13.15% | 13.15% |

**Why lexicographic matches weighted-sum here:** With the current proxy-24 rule set and learned applicability, the lexicographic and weighted-sum selectors converge to the same selections in most scenarios. The distinction becomes critical under adversarial injection (see `adversarial_injection.py`) and with the full 28-rule set where tier ordering prevents safety-comfort tradeoffs.

### evaluation_results.json — Standard Open-Loop Metrics

**Produced by:** `models/RECTOR/scripts/evaluation/evaluate_rector.py`

Simple open-loop evaluation matching the training pipeline exactly. Used for quick sanity checks.

| Metric | Value |
|--------|-------|
| minADE | 0.687 m |
| minFDE | 1.273 m |
| Miss Rate | 18.36% |
| Samples | 43,219 |
| Throughput | ~98 samples/sec |

### Per-Scenario CSVs

**Produced by:** `evaluate_canonical.py` with `--per_scenario_metrics` flag

Each CSV contains one row per scenario with columns: `scenario_id`, `ADE`, `FDE`, `miss`. Three strategy variants are provided for paired statistical tests (bootstrap CIs, Wilcoxon signed-rank).

---

## Closed-Loop Simulation Videos

### Enhanced BEV Rollout Movies (55 videos, 23 MB)

**Produced by:** `scripts/simulation_engine/viz/enhanced_bev_rollout.py`

See [closedloop/videos/README.md](closedloop/videos/README.md) for detailed documentation of each video, the dual-panel visualization format, the scenario mining methodology, and the maneuver category distribution.

**Maneuver Distribution:**

| Category | Count | Description |
|----------|-------|-------------|
| TURN | 16 | Intersections, U-turns, sharp heading changes |
| LANE_CHANGE | 18 | Lateral shifts, merges, overtaking |
| EXIT_RAMP | 11 | Highway exits, moderate turns with lateral displacement |
| COMPLEX | 8 | Stop-and-go, multi-phase maneuvers |
| OTHER | 2 | Straight driving, cruising |
| **Total** | **55** | |

---

## Training Logs

These text files record the complete training history for both stages of RECTOR's two-stage training strategy.

### app_head_fresh_log.txt — Stage 1: Applicability Head Pretraining

**Produced by:** `models/RECTOR/scripts/training/train_applicability.py`

Records per-epoch: training loss, validation loss, per-rule F1/precision/recall, per-tier accuracy, learning rate schedule (OneCycleLR). The applicability head learns which of 28 traffic rules are relevant to each scenario, training on top of a frozen M2I scene encoder.

### rector_full_log.txt — Stage 2: Full End-to-End Training

**Produced by:** `models/RECTOR/scripts/training/train_rector.py`

Records per-epoch: training loss (7-component composite), validation loss, minADE, minFDE, miss rate, learning rate schedule. All 8.82M parameters train jointly — the M2I encoder is unfrozen and fine-tuned at 0.1x the base learning rate.

### Training curves from these logs

The artifact generator `models/RECTOR/scripts/artifacts/generate_training_curves.py` parses these log files to produce publication-quality learning curve figures at `models/RECTOR/output/artifacts/figures/`.

---

## Reproducing These Outputs

```bash
# Standard evaluation
cd /workspace/models/RECTOR
python scripts/evaluation/evaluate_rector.py --checkpoint models/best.pt

# Canonical evaluation (all paper tables)
python scripts/evaluation/evaluate_canonical.py \
    --checkpoint models/best.pt --per_rule_metrics

# Enhanced BEV movies
cd /workspace
python scripts/simulation_engine/viz/enhanced_bev_rollout.py \
    --target_count 55 --shards_to_scan 60 --scenarios_per_shard 60
```

---

## Related Documentation

- [../models/RECTOR/output/artifacts/README.md](../models/RECTOR/output/artifacts/README.md) — Publication figures and LaTeX tables
- [../models/RECTOR/README.md](../models/RECTOR/README.md) — Scientific overview of RECTOR methods
- [../scripts/simulation_engine/README.md](../scripts/simulation_engine/README.md) — Closed-loop simulation engine documentation
