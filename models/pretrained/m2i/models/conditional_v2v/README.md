# `conditional_v2v/` — Conditional model bundle (reactor prediction conditioned on influencer)

This folder holds configuration artifacts for the **Conditional V2V** component of M2I.

In this workspace, the released checkpoint is present as `model.29.bin` alongside `eval.args`.

**Role in M2I:**
- After the relation model decides who is influencer vs reactor,
  the conditional model predicts the **reactor’s future trajectory**
  conditioned on a hypothesis about the influencer’s future.

In other words:
- DenseTNT: “everyone independently”
- Relation: “who reacts to whom?”
- Conditional: “given the influencer’s intent, how does the reactor respond?”

---

## Files

### `eval.args`
A JSON configuration used by upstream M2I.

This repo does **not** directly consume this file, but it documents the expected settings
used when evaluating the released pretrained checkpoint.

Key fields (interpreted):

- `do_eval: true`
- `config: "conditional_pred.yaml"`
- `hidden_size: 128`
- `mode_num: 6`, `future_frame_num: 80`

- `model_recover_path: ".../model.29.bin"`
  The expected checkpoint file.

- `eval_batch_size: 1`
  (Conditional evaluation is often run with batch size 1 due to heavier conditioning logic.)

- `influencer_pred_file_path: "cvpr_10019_files/validation_interactive_m2i_v.pickle"`
  In the original workflow, the conditional stage consumes marginal influencer predictions.

- `relation_pred_file_path: "cvpr_10019_files/m2i.relation.v2v.VAL"`
  In the original workflow, the conditional stage consumes relation predictions.

- `other_params` switches (very important to semantics):
  - `train_reactor: true`       : this config is for the conditional/reactor stage
  - `gt_influencer_traj: true`  : suggests conditioning on ground-truth influencer trajectories in some modes
  - `raster_inf: true` and `raster: true` : influencer/reactor raster context is used
  - `pair_vv: true`             : vehicle-to-vehicle interactive setting
  - plus shared toggles (`laneGCN`, `densetnt`, etc.)

---

## Missing expected file

Expected weight (not included in this repo):
- `model.29.bin`

The scripts in `models/pretrained/m2i/` expect it at:
- `models/pretrained/m2i/models/conditional_v2v/model.29.bin`
