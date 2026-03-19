# Differentiable Rule Proxies

This directory contains 24 differentiable approximations of discrete traffic rule evaluations. These proxies enable gradient-based learning and fast candidate scoring within the RECTOR pipeline, where exact rule evaluation (from `data/WOMD/waymo_rule_eval/`) would be too rigid, too slow, or non-differentiable.

---

## Why Proxies Are Needed

The exact rule evaluation framework (`data/WOMD/waymo_rule_eval/rules/`) uses hard geometric checks (SAT collision detection, point-in-polygon tests, threshold comparisons) that produce binary or discontinuous outputs. These are ideal for ground-truth labeling and validation but cannot be used for:

1. **Gradient-based training**: RECTOR's soft lexicographic loss requires continuous, differentiable violation scores to propagate gradients through trajectory generation.
2. **Fast candidate scoring**: At inference time, the proxy aggregator evaluates all K=6 candidates x 24 rules in a single batched forward pass on GPU — far faster than running the full RuleExecutor per candidate.
3. **Differentiable selection**: The `DifferentiableTieredSelection` module uses proxy outputs as soft attention weights over candidate trajectories during training.

The proxies achieve **Spearman rho = 0.91** correlation with exact rule evaluators, meaning they faithfully preserve the relative ordering of candidates — which is what matters for selection.

---

## Proxy Inventory

### base.py — Framework and Utilities

Defines the `DifferentiableProxy` abstract base class and shared utility functions:

| Utility | Purpose | Mathematical Form |
|---------|---------|-------------------|
| `soft_threshold(x, threshold, sharpness)` | Smooth step function | sigmoid(sharpness * (x - threshold)) |
| `soft_clamp(x, min_val, max_val)` | Smooth clamping that maintains gradients at boundaries | sigmoid(10 * (x - min_val)) * (max_val - min_val) + min_val |
| `exponential_cost(violation, softness)` | Main cost function | 1 - exp(-softness * ReLU(violation)) |
| `smooth_max(x, dim, temperature)` | Differentiable maximum | temperature * log(sum(exp(x_i / temperature))) |
| `smooth_min(x, dim, temperature)` | Differentiable minimum | -smooth_max(-x, dim, temperature) |
| `safe_divide(numerator, denominator, eps)` | Division with epsilon for numerical stability | numerator / (denominator + eps) |
| `obb_corners(positions, headings, sizes)` | OBB corner points | Rotation matrix applied to local corners; takes 3 tensor args: positions [..., 2], headings [...], sizes [..., 2] |
| `batch_pairwise_distance(points_a, points_b)` | Distance matrix | Efficient L2 distance computation |
| `point_in_polygon_2d(points, polygon)` | Convex polygon containment test | Cross-product sign check against each edge |
| `compute_time_to_collision(ego_pos, ego_vel, other_pos, other_vel, min_dist)` | TTC estimate | Quadratic solver for constant-velocity assumption; takes 4 tensor args + min_dist (default 0.5) |

### collision_proxy.py — Collision and Clearance (Safety Tier)

**`CollisionProxy`** — Covers 4 safety-critical rules:

| Rule | Proxy ID | What It Measures |
|------|----------|-----------------|
| L0.R2 | Safe longitudinal distance | Bumper-to-bumper following distance via ego-frame projection |
| L0.R3 | Safe lateral clearance | Perpendicular distance to adjacent agents |
| L10.R1 | Vehicle collision | OBB overlap depth between ego and agents |
| L10.R2 | VRU collision | Distance to vulnerable road users (pedestrians, cyclists) |

**`VRUClearanceProxy`** — Covers L0.R4 (crosswalk occupancy): ego proximity to crosswalk polygons when VRUs are present.

**Key algorithm:** Uses ego-frame relative position decomposition (longitudinal and lateral components) rather than full SAT for efficiency. The exponential cost function provides smooth gradients near the violation boundary.

### lane_proxy.py — Lane Following and Road Structure (Road Tier)

**`LaneProxy`** — Covers 3 road-structure rules:

| Rule | Proxy ID | What It Measures |
|------|----------|-----------------|
| L3.R3 | Drivable surface | Minimum distance from trajectory to road edges when available (1.0 m threshold); falls back to lane centerline distance when road edges are absent |
| L7.R3 | Lane departure | Lateral offset from nearest lane center (max_lateral_offset=1.8 m) |
| L8.R5 | Wrong-way driving | Heading error between ego and lane direction (threshold: 2.356 rad / 135 degrees) |

**`SpeedLimitProxy`** — Covers L7.R4: looks up per-lane speed limits via lane centroid proximity, computes violation for exceeding the limit (default_speed_limit=15.0 m/s, approximately 35 mph).

**Key algorithm:** Soft minimum over lane segments ensures the proxy is robust to lane topology — the ego is only penalized based on its nearest lane, not all lanes.

### signal_proxy.py — Traffic Control Compliance (Legal Tier)

**`SignalProxy`** — Covers 4 traffic control rules:

| Rule | Proxy ID | What It Measures |
|------|----------|-----------------|
| L5.R1 | Traffic signal compliance | Speed at stopline when signal is red/yellow |
| L8.R1 | Red light running | Crossing stopline + red signal + not stopped |
| L8.R2 | Stop sign violation | Approaching stop sign without reaching full stop (< 0.5 m/s) |
| L8.R3 | Crosswalk yield | Speed through crosswalk when VRUs are present |

**Signal state handling:** Differentiates between red (full cost), flashing red (treated as stop sign), yellow (reduced penalty, factor=0.3), and green (zero cost).

**Default thresholds:** stopline_threshold=2.0 m, stop_speed_threshold=0.5 m/s, yellow_penalty_factor=0.3.

### interaction_proxy.py — Agent Interactions (Legal/Comfort Tier)

**`InteractionProxy`** — Covers 6 interaction rules:

| Rule | Proxy ID | What It Measures |
|------|----------|-----------------|
| L4.R3 | Left turn gap | Time gap to oncoming traffic during left turns (TTC-based) |
| L6.R1 | Cooperative lane change | Lateral movement detection + TTC-based gap acceptance |
| L6.R2 | Following distance | Time headway = distance / speed for lead vehicle |
| L6.R3 | Intersection negotiation | TTC-based clearance from crossing traffic at uncontrolled intersections |
| L6.R4 | Pedestrian interaction | Distance + TTC to nearest pedestrian |
| L6.R5 | Cyclist interaction | Distance + TTC to nearest cyclist |

**Default thresholds:** min_following_time=2.0 s, min_vru_distance=3.0 m, min_ttc=3.0 s.

### smoothness_proxy.py — Kinematic Comfort (Comfort Tier)

**`SmoothnessProxy`** — Covers 5 comfort rules:

| Rule | Proxy ID | What It Measures |
|------|----------|-----------------|
| L1.R1 | Smooth acceleration | Forward acceleration magnitude (threshold: 3.0 m/s^2) |
| L1.R2 | Smooth braking | Deceleration magnitude (threshold: 4.0 m/s^2) |
| L1.R3 | Smooth steering | Yaw rate magnitude (threshold: 0.5 rad/s) |
| L1.R4 | Speed consistency | Per-step speed difference vs threshold (2.0 m/s) |
| L1.R5 | Jerk limit | Third derivative of position (acceleration rate of change, threshold: 2.0 m/s^3) |

**`LateralAccelerationProxy`** — Auxiliary comfort metric (rule_ids=[], returns non-canonical key "lateral_acceleration"): lateral_accel = speed x yaw_rate.

**Key algorithm:** Derives kinematic quantities (velocity, acceleration, jerk) from position trajectories using finite differences with dt=0.1s. Handles angle wraparound for heading differences to avoid discontinuities at +/-pi.

### aggregator.py — Unified Rule Cost Output

**`DifferentiableRuleProxies`** — Combines all 7 proxy instances into a single module that produces a unified [B, M, 28] cost tensor:

1. Initializes 7 proxies (collision, VRU clearance, smoothness, lane, speed limit, signal, interaction)
2. Runs each proxy on trajectories + scene features
3. Maps individual proxy outputs to the canonical 28-rule ordering (matching `rule_constants.RULE_IDS`)
4. Returns the complete cost tensor

Note: `LateralAccelerationProxy` is defined in smoothness_proxy.py but is **not** included in the aggregator's proxy list. It is available as a standalone auxiliary module.

**Additional methods:**
- `forward_with_names()` — Returns a named dictionary for debugging
- `get_tier_weighted_costs()` — Applies tier weights (safety > legal > road > comfort)

**ProxyRegistry** — Static utility for mapping rule IDs to proxy classes. The `register()` method uses a plain dict, so if a rule ID is registered more than once the later registration silently overwrites the earlier one. There is no uniqueness check.

---

## Coverage: 24 of 28 Canonical Rules

The 7 proxies in the aggregator collectively cover 24 canonical rules:

| Proxy | Rules | Count |
|-------|-------|-------|
| CollisionProxy | L0.R2, L0.R3, L10.R1, L10.R2 | 4 |
| VRUClearanceProxy | L0.R4 | 1 |
| SmoothnessProxy | L1.R1, L1.R2, L1.R3, L1.R4, L1.R5 | 5 |
| LaneProxy | L3.R3, L7.R3, L8.R5 | 3 |
| SpeedLimitProxy | L7.R4 | 1 |
| SignalProxy | L5.R1, L8.R1, L8.R2, L8.R3 | 4 |
| InteractionProxy | L4.R3, L6.R1, L6.R2, L6.R3, L6.R4, L6.R5 | 6 |
| **Total** | | **24** |

Four rules lack proxy implementations:

| Rule | Reason No Proxy |
|------|----------------|
| L5.R2 (Priority violation) | Requires complex right-of-way reasoning not reducible to geometry |
| L5.R3 (Parking violation) | Rare in WOMD interactive scenarios (0 positive samples) |
| L5.R4 (School zone) | Requires zone boundary detection not available in WOMD |
| L5.R5 (Construction zone) | Same — zone features not in standard WOMD |

When evaluating with the full 28-rule set (via `evaluate_canonical.py --rule_set full-28`), these 4 rules use the ground-truth labels from the augmented TFRecords rather than proxy approximations.

---

## File Inventory

| File | Contents |
|------|----------|
| base.py | `DifferentiableProxy` ABC, utility functions |
| collision_proxy.py | `CollisionProxy`, `VRUClearanceProxy` |
| lane_proxy.py | `LaneProxy`, `SpeedLimitProxy` |
| signal_proxy.py | `SignalProxy` |
| smoothness_proxy.py | `SmoothnessProxy`, `LateralAccelerationProxy` |
| interaction_proxy.py | `InteractionProxy` |
| aggregator.py | `DifferentiableRuleProxies`, `ProxyRegistry` |
| \_\_init\_\_.py | Public re-exports |

---

## Related Documentation

- [../models/README.md](../models/README.md) — How proxies integrate into the RECTOR architecture
- [../../../../data/WOMD/waymo_rule_eval/rules/README.md](../../../../data/WOMD/waymo_rule_eval/rules/README.md) — Exact (non-differentiable) rule implementations
- [../evaluation/README.md](../evaluation/README.md) — How proxy-24 vs full-28 evaluation is handled
