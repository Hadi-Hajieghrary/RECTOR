# RECTOR Focused Experiments

This directory contains scripts that answer specific research questions beyond the standard train/evaluate workflow. Each script targets a particular claim or concern that a reviewer might raise, producing evidence that strengthens the paper's argument.

Use `scripts/evaluation/` for primary metrics. Use this directory when the main tables are stable and you need deeper supporting evidence: ablation variants, cross-system comparisons, failure analysis, or supplementary diagnostics.

---

## Scripts

### evaluate_m2i_rector.py — M2I + RECTOR Cross-System Evaluation

**Research question:** Does RECTOR's rule scoring work on trajectories from an external predictor (M2I DenseTNT)?

Runs the pretrained M2I trajectory generator and applies RECTOR's rule evaluation to its outputs, comparing oracle (best-of-K by GT ADE), M2I confidence-pick, and rule-reranked selection. Demonstrates that rule-aware selection is not predictor-specific.

**Produces:** M2I-specific evaluation_results.json with oracle/M2I/rule-picked ADE/FDE, per-tier violations, timing. Also produces per_scenario_details.json with per-scenario oracle/M2I/rule ADE/FDE, tier violations, and timing.

### full_rule_evaluation.py — Complete 28-Rule Validation

**Research question:** Do differentiable proxy scores align with exact discrete rule evaluation?

Reconstructs complete `ScenarioContext` objects from WOMD protos and runs predicted trajectories through the full `waymo_rule_eval.RuleExecutor` (SAT collision, point-in-polygon, exact threshold checks). Validates proxy fidelity.

**Produces:** full_rule_evaluation_results.json with per-tier and per-rule violation rates from exact evaluation.

### analyze_kinematics.py — Trajectory Physics Validation

**Research question:** Are RECTOR's predicted trajectories kinematically plausible?

Computes speed, acceleration, and jerk distributions across validation predictions. Reports P95 for max speed and max jerk, P95/P99 for max acceleration, and P90 for the comfort acceleration threshold suggestion. Validates that outputs fall within a realistic kinematic range before comfort-tier rules are applied.

**Produces:** Console output with kinematic statistics and threshold suggestions.

### count_parameters.py — Model Size Analysis

**Research question:** How large is RECTOR relative to the M2I backbone?

Loads a checkpoint, categorizes parameters by component (scene_encoder, applicability_head, trajectory_head, tiered_scorer, rule_proxies), and reports exact counts.

**Produces:** real_param_counts.json (total: 8.82M, applicability head: ~1.5M).

### run_rule_evaluation.py — Standalone Rule Evaluation Pipeline

Full standalone script that loads the trained RECTOR model, runs inference on validation scenarios via DataLoader, evaluates rule violations on predicted trajectories (using DifferentiableRuleProxies if available, otherwise falling back to kinematic checks for Comfort and Safety tiers only), aggregates violation counts by tier, and saves results to JSON. Does not evaluate all 28 rules — proxy-based evaluation covers the rules exposed by DifferentiableRuleProxies, and the kinematic fallback only checks Comfort (acceleration > 3 m/s^2) and Safety (acceleration > 6 m/s^2 or speed > 30 m/s).

**Produces:** rule_violation_results.json with per-tier violation rates and counts.

### compute_rule_metrics.py — Aggregate Proxy Metric Computation

Computes three aggregate proxy metrics from RECTOR trajectory predictions: miss rate (FDE > 2m, Safety-tier proxy), close-agent rate (minimum distance to other agents < 2m, Safety-tier proxy), and off-lane rate (maximum distance from nearest lane center > 3m, Road-tier proxy). These three rates are then mapped to tier-level violation rates. Does not compute per-rule breakdowns, severity distributions, or applicability frequencies.

**Produces:** real_rule_metrics.json with raw proxy rates and mapped tier violation rates.

### debug_trajectory.py — Single-Batch Kinematic Debugger

Runs RECTOR inference on a single batch (4 samples from one file) and prints per-timestep trajectory positions, displacements, velocities, and accelerations for predicted and ground-truth trajectories. Used for inspecting raw model output format and verifying scale conversions. Does not perform any rule evaluation.

**Produces:** Console output with trajectory shapes, normalized/meter-scale positions, displacements, and acceleration statistics.

---

## When to Use Each Script

| Script | Purpose | When to Use |
|--------|---------|-------------|
| `evaluate_m2i_rector.py` | M2I cross-system comparison | Claiming method generality |
| `full_rule_evaluation.py` | Proxy fidelity validation (all 28 rules) | Claiming proxy accuracy |
| `analyze_kinematics.py` | Physical plausibility check | Reporting comfort metrics |
| `count_parameters.py` | Model efficiency reporting | Reporting model size |
| `run_rule_evaluation.py` | Proxy/kinematic rule evaluation | Comfort and Safety tier violation rates |
| `compute_rule_metrics.py` | Aggregate proxy metrics | Tier-level miss/proximity/lane rates |
| `debug_trajectory.py` | Trajectory format inspection | Debugging kinematics and scale issues |

---

## Related Documentation

- [../evaluation/README.md](../evaluation/README.md) — Primary evaluation suite (use for paper metrics)
- [../diagnostics/README.md](../diagnostics/README.md) — Development-time debugging tools
- [../../README.md](../../README.md) — RECTOR scientific overview
