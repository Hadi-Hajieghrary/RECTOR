# Trajectory Selection Strategies

This module implements three trajectory selection strategies for RECTOR's closed-loop simulation, enabling controlled A/B comparison of how selection method affects driving safety and compliance.

The central research question: **Given the same set of trajectory candidates with the same rule violation scores, does the selection method matter?** RECTOR's answer is yes — lexicographic selection provides strict safety guarantees that weighted-sum approaches cannot.

---

## The Selection Problem

At each replanning instant, the trajectory generator produces K=6 candidate ego trajectories. Each candidate has:
- A **confidence score** (model's belief in trajectory quality)
- **Per-rule violation costs** [K, 28] (how much each candidate violates each of 28 rules)
- **Per-tier aggregated scores** [K, 4] (violations grouped by Safety, Legal, Road, Comfort)
- A **rule applicability mask** [28] (which rules are relevant in this scenario)

The selector must choose one candidate. The three strategies differ in how they use this information.

---

## Strategies

### ConfidenceSelector — Baseline

**File:** `confidence.py`

**Method:** `argmax(confidence)`

Selects the trajectory with the highest model confidence, completely ignoring rule violation information. This represents what a standard trajectory predictor does — pick the most likely mode.

**Complexity:** O(K)

**Limitation:** Cannot avoid safety violations. If the most confident trajectory runs a red light, the selector will choose it.

**Purpose in paper:** Establishes the baseline violation rate that rule-unaware prediction produces.

### WeightedSumSelector — Tuned Baseline

**File:** `weighted_sum.py`

**Method:** `argmin(Σ w_i · rule_cost_i · applicability_i)`

Computes a weighted sum over all applicable rule violations and selects the trajectory with the lowest total cost. Weights [28] are tuned via cross-validation grid search (see `evaluation/weight_grid_search.py`, 125 configurations -- this is an external dependency outside the simulation engine).

**Complexity:** O(K·R) where R=28 rules

**Limitation:** Weighted sums allow **implicit tradeoffs between tiers**. With any finite weight assignment, there exists some combination of comfort improvements that can outweigh a safety violation. This is not a theoretical concern — the weight grid search (`weight_grid_search.py`) demonstrates that even oracle-tuned weights cannot guarantee safety dominance.

**Purpose in paper:** Shows that even a well-tuned weighted baseline cannot match lexicographic selection's safety guarantees.

### RECTORLexSelector — Lexicographic Selection (Proposed)

**File:** `rector_lex.py`

**Method:** Tier-by-tier elimination with epsilon tolerance

The lexicographic selector processes tiers in strict priority order:

```
surviving_candidates = {all K candidates}

For tier in [Safety, Legal, Road, Comfort]:
    best_score = min(tier_scores[surviving_candidates])
    surviving_candidates = {c : tier_score[c] <= best_score + epsilon}

selected = argmax(confidence[surviving_candidates])
```

**Complexity:** O(K·T) where T=4 tiers

**Key property — The Safety Guarantee:** By construction, **no number of comfort improvements can compensate for a safety violation.** If candidate A has a safety violation and candidate B does not, the selector will always prefer B regardless of how A scores on legal, road, or comfort tiers. This is the core theoretical contribution.

**Epsilon tolerance** (default 1e-3 per tier): Prevents overfitting to floating-point noise. Two candidates within epsilon of each other on a tier are considered equivalent, and the tie is broken by the next tier. Among final survivors, the highest-confidence candidate is selected.

---

## DecisionTrace — Audit Trail

Every selection produces a `DecisionTrace` (frozen dataclass in `base.py`) that records the full decision:

| Field | Type | Description |
|-------|------|-------------|
| `method` | str | "confidence", "weighted_sum", or "rector_lex" |
| `selected_idx` | int | Chosen candidate (0..K-1) |
| `reason` | str | Human-readable explanation |
| `active_rules` | [28] | Applicability mask |
| `first_diff_tier` | int | Tier (0-3) that first eliminated candidates |
| `per_tier_survivors` | List | Survivor masks at each tier |
| `per_candidate_scores` | [K, 4] | Tier scores for all candidates |
| `timestamp_ms` | float | Wall-clock selection time |

This audit trail enables post-hoc analysis: in how many scenarios did Safety tier matter? How often does lexicographic selection differ from confidence? Which tier is the most discriminating?

---

## Adding a New Selector

1. Subclass `BaseSelector` from `base.py`
2. Implement `select(candidates, probs, tier_scores, rule_scores, rule_applicability)` → `(int, DecisionTrace)`
3. Register in the simulation loop configuration

---

## Related Documentation

- [../README.md](../README.md) — Simulation engine overview
- [../waymax_bridge/README.md](../waymax_bridge/README.md) — How simulation data flows
- [../viz/README.md](../viz/README.md) — How selector comparisons are visualized
- [../../../models/RECTOR/scripts/evaluation/weight_grid_search.py](../../../models/RECTOR/scripts/evaluation/weight_grid_search.py) — Weighted-sum tuning
