# RECTOR Demonstration Videos

This directory contains bird's-eye-view (BEV) videos demonstrating the M2I+RECTOR receding-horizon planning pipeline in action on real Waymo Open Motion Dataset scenarios. These videos serve as supplementary material for the paper, providing visual evidence of rule-aware trajectory selection in diverse interactive driving situations.

---

## M2I+RECTOR Planning Demonstrations (40 videos)

**Directory:** [`m2i_rector/`](m2i_rector/)

Each video shows the M2I+RECTOR pipeline operating in a receding-horizon loop on a real WOMD interactive scenario. The pipeline replans every 20 steps (2 seconds), generating fresh trajectory candidates and selecting the most rule-compliant option.

### What Each Video Shows

- **Ego vehicle** (blue): The autonomous vehicle executing the RECTOR-selected trajectory
- **Candidate trajectories**: Multiple DenseTNT-generated trajectory modes, color-coded by rule compliance:
  - Green → low violation cost (safe, legal, comfortable)
  - Red → high violation cost (safety or legal violations detected)
- **Selected trajectory** (highlighted green): The candidate chosen by the lexicographic rule selector — always the one that minimizes safety violations first, then legal, then road, then comfort
- **Surrounding agents**: Other vehicles, pedestrians, and cyclists with their observed and predicted trajectories
- **Road geometry**: Lane centerlines, boundaries, crosswalks, and traffic infrastructure

### Pipeline Details

At each replanning instant, the pipeline:

1. **Generates candidates**: DenseTNT (pretrained M2I backbone) produces 6 diverse ego trajectory modes over a 5-second horizon
2. **Evaluates rules**: Each candidate is scored against 28 traffic rules using a kinematic rule evaluator
3. **Selects lexicographically**: The `M2IRuleSelector` compares candidates tier-by-tier:
   - Safety tier (collision, VRU clearance, safe distance) — highest priority
   - Legal tier (red lights, stop signs, speed limits)
   - Road tier (drivable surface, lane keeping)
   - Comfort tier (smooth acceleration, braking, steering) — lowest priority
4. **Executes**: The selected trajectory is followed until the next replanning instant

### Video Index

| Video | Scenario ID | Description |
|-------|------------|-------------|
| `m2i_rector_000_*.mp4` – `m2i_rector_039_*.mp4` | WOMD validation interactive | 40 diverse scenarios spanning highway merges, urban intersections, lane changes, and VRU interactions |

### Generation Details

| Parameter | Value |
|-----------|-------|
| Prediction interval | 20 steps (2 seconds) |
| Minimum ego speed | 3.0 m/s (filters out stationary scenarios) |
| Trajectory modes | 6 per replanning instant |
| Data source | WOMD validation_interactive TFRecords |
| Frame rate | 10 fps |

**Produced by:** `models/RECTOR/scripts/visualization/generate_m2i_movies.py`

```bash
cd /workspace/models/RECTOR
python scripts/visualization/generate_m2i_movies.py \
    --num_scenarios 40 \
    --predict_interval 20 \
    --min_ego_speed 3.0
```

---

## Why These Videos Matter

For an autonomous driving paper, static metrics (ADE, FDE, miss rate) tell only part of the story. Reviewers need to see:

1. **Behavioral correctness**: Does the selected trajectory *look* like reasonable driving? These videos show that RECTOR selects trajectories that follow lanes, respect traffic signals, and maintain safe distances — not just trajectories that happen to be geometrically close to ground truth.

2. **Selection in action**: The color-coded candidates make it visible when the confidence-best trajectory (what a standard predictor would select) differs from the rule-best trajectory (what RECTOR selects). In scenarios with safety violations, the viewer can see RECTOR choosing the green (safe) mode over the red (unsafe but high-confidence) mode.

3. **Temporal consistency**: The receding-horizon format shows how selections evolve over time. RECTOR maintains consistent, smooth behavior across replanning instants rather than oscillating between modes.

4. **Diverse driving situations**: The 40 scenarios cover highway driving, urban intersections, lane changes, merges, and interactions with vulnerable road users — demonstrating generalization across the WOMD distribution.

---

## Related Documentation

- [../scripts/visualization/README.md](../scripts/visualization/) — Visualization script documentation
- [../scripts/lib/m2i_rule_selector.py](../scripts/lib/m2i_rule_selector.py) — Lexicographic rule selector implementation
- [../scripts/lib/m2i_trajectory_generator.py](../scripts/lib/m2i_trajectory_generator.py) — DenseTNT trajectory generation wrapper
- [../../output/closedloop/videos/README.md](../../output/closedloop/videos/README.md) — Enhanced closed-loop BEV movies with violation panels
