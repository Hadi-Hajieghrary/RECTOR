# `relation_v2v/` — Relation model bundle (V2V influencer/reactor)

This folder holds configuration artifacts for the **Relation V2V** component of M2I.

In this workspace, the released checkpoint is present as `model.25.bin` alongside `eval.args`.

**Role in M2I:**
- Given an *interactive* scenario (two “objects of interest”),
  the relation model predicts **who influences whom** (directionality).
- Upstream M2I stores relation outputs into a global dict (`globals.sun_1_pred_relations`).

---

## Files

### `eval.args`
A JSON configuration used by upstream M2I.

This repo does **not** directly consume this file, but it documents the expected settings
used when evaluating the released pretrained checkpoint.

Key fields (interpreted):

- `do_eval: true`
  This config was used in evaluation mode.

- `config: "relation.yaml"`
  Upstream M2I YAML config name for relation.

- `hidden_size: 128`
  VectorNet hidden dimension.

- `mode_num: 6`, `future_frame_num: 80`
  6 trajectory hypotheses, each 80 future timesteps.

- `data_dir: "/data/.../waymo/training"`
  The original training/eval data location used by the authors (not relevant to this repo’s paths).

- `model_recover_path: ".../model.25.bin"`
  The expected checkpoint file for the relation model.

- `other_params` (important behavioral switches):
  - `train_relation: true`   : this config corresponds to relation training/eval
  - `pair_vv: true`          : vehicle-to-vehicle interactions
  - `pred_with_threshold: true` : apply a relation-confidence threshold
  - `vehicle_r_pred_threshold: 0.7` : threshold value for accepting relation predictions
  - `raster: true`           : uses rasterized BEV context as an input modality
  - plus multiple graph/model toggles (`laneGCN`, `point_sub_graph`, etc.)

---

## Missing expected file

Expected weight (not included in this repo):
- `model.25.bin`

The scripts in `models/pretrained/m2i/` expect it at:
- `models/pretrained/m2i/models/relation_v2v/model.25.bin`
