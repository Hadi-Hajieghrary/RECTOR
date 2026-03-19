# `models/pretrained/m2i/models/densetnt/` — DenseTNT marginal model

This directory contains the **DenseTNT** pretrained weight file used for
*marginal* trajectory prediction (predict each agent independently).

Expected file:
- `model.24.bin`

Used by:
- `models/pretrained/m2i/scripts/lib/m2i_live_inference.py`
- `models/pretrained/m2i/scripts/lib/m2i_gpu_pipeline.py`
- `models/pretrained/m2i/scripts/lib/m2i_receding_horizon_full.py` (depending on mode)

In this workspace, the pretrained checkpoint is present as `model.24.bin`.
