# RECTOR Evaluation Scripts

This module contains all scripts for quantitative evaluation of RECTOR's trajectory prediction and rule-aware selection. It is the primary source of numbers reported in the paper — every metric in every table should be reproducible by running the scripts documented here.

The evaluation suite addresses five distinct research questions, each answered by a dedicated script (or library module).

---

## Scripts

### evaluate_canonical.py — Canonical Evaluation for Paper Tables

**Research question:** How does RECTOR perform across protocol combinations (rule sets x applicability modes x selection strategies)?

This is the most important evaluation script. It runs 12,800 WOMD validation scenarios and evaluates results under multiple protocol axes. The code evaluates:

- **4 protocol combinations** (all using lexicographic selection): every pairing of rule set and applicability source.
- **3 selection strategies** (all using proxy-24 rules with predicted applicability): confidence, weighted-sum, and lexicographic.

The full 2x2x3 = 12-cell Cartesian product is **not** computed. The protocol results use lexicographic selection exclusively, and the strategy comparison uses the proxy-24/predicted configuration exclusively.

| Protocol axis | Options | Purpose |
|---------------|---------|---------|
| Rule set | proxy-24, full-28 | Do the 4 rules without proxies matter? |
| Applicability | predicted (learned), oracle (ground-truth) | How good is the learned applicability head? |
| Selection strategy | confidence, weighted-sum, lexicographic | Which strategy produces the most compliant trajectories? |

**CLI arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint` | `/workspace/models/RECTOR/models/best.pt` | Path to trained checkpoint |
| `--val_dir` | `(WOMD validation_interactive)` | Path to validation TFRecords |
| `--output` | `/workspace/output/evaluation/canonical_results.json` | Output JSON path |
| `--embed_dim` | 256 | Embedding dimension (must match training) |
| `--decoder_hidden_dim` | 256 | Decoder hidden dimension |
| `--decoder_num_layers` | 4 | Decoder layer count |
| `--latent_dim` | 64 | Latent space dimension |
| `--num_modes` | 6 | Number of trajectory modes |
| `--dropout` | 0.1 | Dropout rate |
| `--use_m2i_encoder` | True | Use M2I encoder |
| `--m2i_checkpoint` | `(pretrained M2I path)` | M2I checkpoint path |
| `--batch_size` | 64 | Batch size |
| `--num_workers` | 4 | Data loader workers |
| `--max_batches` | None | Limit batches for quick testing |
| `--eval_gt` | False | Evaluate human GT trajectories instead of RECTOR predictions |
| `--seed` | 42 | Random seed |
| `--strategies` | `[confidence, weighted_sum, lexicographic]` | Selection strategies to evaluate |
| `--epsilon` | 1e-3 | Tolerance for lexicographic tier comparison |
| `--applicability_mode` | `learned` | Applicability source: `learned`, `always_on`, `heuristic`, or `hybrid_conservative` |
| `--per_rule_metrics` | False | Compute per-rule precision/recall/F1 for applicability head |

**Applicability modes:**

| Mode | Description |
|------|-------------|
| `learned` | Use the trained applicability head's predictions (default, Protocol A) |
| `always_on` | All 28 rules treated as applicable in every scenario |
| `heuristic` | Rules activated by geometric scene-feature checks |
| `hybrid_conservative` | Always-on for Tier 0 (Safety) and Tier 1 (Legal), learned predictions for Tier 2 (Road) and Tier 3 (Comfort) |

**Produces:**
- `canonical_results.json` — the canonical evaluation record for the paper tables, containing:
  - Oracle geometric metrics (minADE, minFDE, miss rate)
  - Per-strategy selected metrics (selADE, selFDE, violation rates by tier) under Protocol A
  - Per-strategy selected metrics under Protocol B (full-28 rules, oracle applicability)
  - **Cross-evaluation results** (`selection_strategies_cross_eval`): mode-specific applicability for selection, oracle applicability for violation evaluation — used to compare trajectory selection quality across applicability modes on a common evaluation basis
  - Per-protocol-combination results (lexicographic selection under each rule-set/applicability pairing)
  - Per-rule applicability metrics (precision, recall, F1, support for all 28 rules) when `--per_rule_metrics` is set
- `per_scenario_metrics.csv` — per-scenario geometric metrics (scenario_idx, minADE, minFDE, miss)
- `per_scenario_{strategy}.csv` — per-scenario selection metrics for each strategy (selADE, selFDE, violation indicators)
- `per_scenario_protB_{strategy}.csv` — per-scenario Protocol B metrics for each strategy

```bash
python evaluation/evaluate_canonical.py \
    --checkpoint /workspace/models/RECTOR/models/best.pt \
    --per_rule_metrics
```

### evaluate_rector.py — Standard Open-Loop Evaluation

**Research question:** What are the basic trajectory accuracy metrics?

A streamlined evaluation that matches the training pipeline exactly (same data loading, same normalization, same collation). Used for quick sanity checks and training-time validation.

**Produces:** `evaluation_results.json` with minADE, minFDE, miss rate, and elapsed_time. Note: throughput (samples/sec) is **not** computed or saved; only wall-clock elapsed time is recorded.

```bash
python evaluation/evaluate_rector.py --checkpoint models/best.pt
```

### compute_bootstrap_cis.py — Bootstrap Confidence Intervals

**Research question:** Are the reported improvements statistically significant?

Computes 95% bootstrap confidence intervals (10,000 resamples, seed=42) for all headline violation-rate metrics across selection strategies.

**Methodology limitation:** This script reads aggregated proportions from `canonical_results.json` and reconstructs synthetic Bernoulli binary arrays (matching the observed proportion and sample count) rather than reading actual per-scenario CSVs. This means the bootstrap CIs reflect sampling variability of the observed proportion but do not capture per-scenario correlations or the exact permutation structure of the original data. For more precise CIs, the script would need to read the `per_scenario_{strategy}.csv` files directly.

**Note on Wilcoxon tests:** The source code defines a `paired_wilcoxon()` function, but `main()` never calls it. The output contains **only** bootstrap confidence intervals — no p-values or paired Wilcoxon signed-rank test results are produced.

**Produces:** `bootstrap_cis.json` with per-metric CIs for each strategy and protocol combination.

```bash
python evaluation/compute_bootstrap_cis.py \
    --input /workspace/output/evaluation/canonical_results.json
```

### heuristic_applicability.py — Non-Learned Baselines (Library Module)

**Research question:** Does the learned applicability head outperform simple heuristics?

This is a **library module** (no `main()`, no CLI entry point). It exports three functions that are imported by `evaluate_canonical.py`:

| Function | Method | Description |
|----------|--------|-------------|
| `always_on_applicability(batch_size)` | a_r = 1 for all rules | All 28 rules treated as applicable in every scenario |
| `heuristic_applicability(batch, ...)` | Scene-feature gating | Rules activated by geometric checks (nearby agents -> collision rules, signals -> traffic rules, crosswalks -> VRU rules) |
| `hybrid_conservative_applicability(learned_pred, batch_size)` | Mixed policy | Always-on for Tier 0 (Safety) and Tier 1 (Legal) rules; retains learned predictions for Tier 2 (Road) and Tier 3 (Comfort). Imports tier constants from `rule_constants.py`. |

To produce ablation rows comparing the learned head against these baselines, run `evaluate_canonical.py` with `--applicability_mode {always_on, heuristic, hybrid_conservative}`.

### adversarial_injection.py — Safety Tier Stress Test

**Research question:** Does RECTOR's Safety tier actually prevent selection of dangerous trajectories?

Runs its own independent inference on WOMD validation data (loads the model and dataset directly). It injects known-violating trajectories into the candidate set and measures how often each strategy selects the adversarial option. It imports utility functions (`compute_tier_scores`, `select_by_*`, etc.) from `evaluate_canonical.py` but does **not** read `canonical_results.json`.

| Injection Type | What It Does | Violated Rules |
|---------------|-------------|----------------|
| `collision_course` | Trajectory aimed at nearest agent | L10.R1, L10.R2, L0.R2, L0.R3 |
| `clearance_violation` | Lateral offset clipping safe distance | L0.R2, L0.R3 |
| `vru_incursion` | Trajectory through crosswalk with pedestrian | L0.R4, L8.R3 |

**Produces:** `adversarial_injection_results.json` with adversarial selection rates per strategy. The key metric is "Adv.Sel%" — the fraction of scenarios where the strategy selected the injected dangerous trajectory. For lexicographic selection, this should be 0% when the adversarial trajectory has higher-tier violations than all alternatives.

```bash
python evaluation/adversarial_injection.py \
    --checkpoint models/best.pt
```

### weight_grid_search.py — Weighted-Sum Fairness Test

**Research question:** Is RECTOR only winning because the weighted-sum baseline isn't properly tuned?

Runs its own independent inference on WOMD validation data (loads the model and dataset directly, caches model outputs, then sweeps weight configurations without re-inference). It imports utility functions from `evaluate_canonical.py` but does **not** read `canonical_results.json`.

Default weight grids:
- Safety: [100, 500, 1000, 2000, 5000]
- Legal: [10, 50, 100, 200, 500]
- Road: [1, 5, 10, 20, 50]
- Comfort: 1.0 (fixed)
- Total: 5 x 5 x 5 = 125 configurations

**Produces:** `weight_grid_search_results.json` with all 125 configurations, the Pareto frontier, and comparison to lexicographic selection. Demonstrates that even oracle-tuned weighted-sum cannot match lexicographic safety guarantees because finite weights always permit implicit tier tradeoffs.

```bash
python evaluation/weight_grid_search.py \
    --checkpoint models/best.pt
```

---

## Evaluation Data Flow

```
WOMD validation TFRecords (150 shards, 12,800 scenarios)
           |
           v
   RECTOR model inference
   (6 modes x 50 timesteps x 28 rule costs)
           |
           +------------------------------------------+---------------------------+
           |                                          |                           |
evaluate_canonical.py                   weight_grid_search.py         adversarial_injection.py
  (runs own inference)                   (runs own inference,          (runs own inference,
           |                              imports utility fns)          imports utility fns)
           v                                          |                           |
  canonical_results.json                              v                           v
           |                              weight_grid_*.json          adversarial_*.json
  per_scenario_metrics.csv
  per_scenario_{strategy}.csv
           |
  compute_bootstrap_cis.py
   (reads canonical JSON)
           |
           v
  bootstrap_cis.json


evaluate_rector.py
  (runs own inference,
   standalone sanity check)
           |
           v
  evaluation_results.json
```

Each of the four inference-running scripts (`evaluate_canonical.py`, `evaluate_rector.py`, `weight_grid_search.py`, `adversarial_injection.py`) loads the RECTOR model and WOMD data independently. `weight_grid_search.py` and `adversarial_injection.py` import shared utility functions (selection strategies, tier-score computation, proxy evaluation) from `evaluate_canonical.py`, but they do **not** read its output JSON. Only `compute_bootstrap_cis.py` reads the output of another script (`canonical_results.json`).

---

## Key Design Decisions

1. **Canonical evaluation record**: All paper tables read from `canonical_results.json`. This keeps reported numbers aligned across downstream table-generation scripts.

2. **Per-scenario CSVs for downstream analysis**: Every scenario gets a row with all metrics in `per_scenario_{strategy}.csv`. These CSVs enable distribution plots and could support paired statistical tests if `compute_bootstrap_cis.py` were extended to read them directly.

3. **Protocol structure**: The evaluation computes 4 protocol combinations (rule-set x applicability, all under lexicographic selection), 3 selection strategies under Protocol A (proxy-24/predicted), 3 selection strategies under Protocol B (full-28/oracle), and 3 cross-evaluation strategies (mode-specific selection, oracle evaluation). The model is run once and results are cached in memory, then sliced by protocol combination. This makes evaluation fast (~9 minutes for 12,800 scenarios, ~1 minute with `--max_batches 20`).

4. **Adversarial injection**: Rather than only reporting average-case improvements, adversarial injection tests the worst-case guarantee — the core theoretical claim of lexicographic selection.

---

## Related Documentation

- [../../README.md](../../README.md) — RECTOR scientific overview
- [../artifacts/README.md](../artifacts/README.md) — How evaluation results become paper figures and tables
- [../../../../output/README.md](../../../../output/README.md) — Where evaluation outputs are stored
