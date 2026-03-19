#!/bin/bash
set -e
echo "=========================================="
echo "FULL 12,800-SCENARIO EVALUATION RE-RUNS"
echo "Started: $(date)"
echo "=========================================="

EVAL_SCRIPT="/workspace/models/RECTOR/scripts/evaluation/evaluate_canonical.py"
WEIGHT_SCRIPT="/workspace/models/RECTOR/scripts/evaluation/weight_grid_search.py"
OUTDIR="/workspace/output/evaluation"
VAL_DIR="/workspace/data/WOMD/datasets/waymo_open_dataset/motion_v_1_3_0/processed/augmented/scenario/validation_interactive"
MAX_BATCHES=200   # 200 * 64 = 12,800 scenarios
PYTHON_BIN="/opt/venv/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="python3"
fi

# Run 1: Learned mode (default) — gets Protocol B strategies (Table XXIX) + cross-eval learned (Table XXVII)
echo ""
echo "[1/4] Running learned mode (full 12,800 scenarios)..."
"$PYTHON_BIN" "$EVAL_SCRIPT" \
    --val_dir "$VAL_DIR" \
    --max_batches "$MAX_BATCHES" \
    --applicability_mode learned \
    --per_rule_metrics \
    --output "$OUTDIR/full_12800_learned.json" \
    2>&1 | tail -20
echo "Done: $(date)"

# Run 2: Hybrid conservative mode — cross-eval for Table XXVII
echo ""
echo "[2/4] Running hybrid_conservative mode (full 12,800 scenarios)..."
"$PYTHON_BIN" "$EVAL_SCRIPT" \
    --val_dir "$VAL_DIR" \
    --max_batches "$MAX_BATCHES" \
    --applicability_mode hybrid_conservative \
    --per_rule_metrics \
    --output "$OUTDIR/full_12800_hybrid_conservative.json" \
    2>&1 | tail -20
echo "Done: $(date)"

# Run 3: Always-on mode — cross-eval for Table XXVII
echo ""
echo "[3/4] Running always_on mode (full 12,800 scenarios)..."
"$PYTHON_BIN" "$EVAL_SCRIPT" \
    --val_dir "$VAL_DIR" \
    --max_batches "$MAX_BATCHES" \
    --applicability_mode always_on \
    --per_rule_metrics \
    --output "$OUTDIR/full_12800_always_on.json" \
    2>&1 | tail -20
echo "Done: $(date)"

# Run 4: Weight grid search on canonical 12,800 scenarios
echo ""
echo "[4/4] Running weight grid search (full 12,800 scenarios)..."
"$PYTHON_BIN" "$WEIGHT_SCRIPT" \
    --val_dir "$VAL_DIR" \
    --max_batches "$MAX_BATCHES" \
    --output "$OUTDIR/full_12800_weight_grid.json" \
    2>&1 | tail -30
echo "Done: $(date)"

echo ""
echo "=========================================="
echo "ALL EVALUATIONS COMPLETE: $(date)"
echo "=========================================="
