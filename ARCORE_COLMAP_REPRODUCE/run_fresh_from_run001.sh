#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPRO="$SCRIPT_DIR"
ROOT="$(cd "$REPRO/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${RUN_DIR:-$REPRO/fresh_from_run001_$STAMP}"
INIT_CKPT="${INIT_CKPT:-$REPRO/posneg_run_001/best_model.pt}"

if [[ ! -f "$INIT_CKPT" ]]; then
  echo "ERROR: init checkpoint not found: $INIT_CKPT"
  exit 1
fi

echo "========================================"
echo "Fresh training from run_001 checkpoint"
echo "RUN_DIR   : $RUN_DIR"
echo "INIT_CKPT : $INIT_CKPT"
echo "========================================"

# Conservative settings to avoid overfitting/drift regressions.
INIT_CKPT="$INIT_CKPT" \
RUN_DIR="$RUN_DIR" \
PYTHON_BIN="$PYTHON_BIN" \
DEVICE="${DEVICE:-cuda}" \
EPOCHS="${EPOCHS:-35}" \
BATCH_SIZE="${BATCH_SIZE:-48}" \
NUM_WORKERS="${NUM_WORKERS:-12}" \
LR="${LR:-2e-5}" \
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}" \
KP_WEIGHT="${KP_WEIGHT:-5.0}" \
TRAIN_RATIO="${TRAIN_RATIO:-0.82}" \
VAL_RATIO="${VAL_RATIO:-0.10}" \
SEED="${SEED:-42}" \
AUGMENT="${AUGMENT:-1}" \
ROI_TRAIN="${ROI_TRAIN:-1}" \
ROI_CONTEXT="${ROI_CONTEXT:-1.45}" \
ROI_JITTER="${ROI_JITTER:-0.06}" \
ROI_MIN_SIZE="${ROI_MIN_SIZE:-96}" \
ROI_PROB="${ROI_PROB:-0.65}" \
ROT_MAX_DEG="${ROT_MAX_DEG:-90}" \
SYMMETRY_AWARE="${SYMMETRY_AWARE:-1}" \
bash "$REPRO/run_train_posneg_curated.sh"

echo "========================================"
echo "Evaluating baseline run_001 checkpoint on same split"
echo "========================================"

"$PYTHON_BIN" "$REPRO/eval_objectron_tracking.py" \
  --checkpoint "$INIT_CKPT" \
  --video_dirs_txt "$RUN_DIR/included_blue_dirs.txt" "$RUN_DIR/included_negative_dirs.txt" \
  --out_dir "$RUN_DIR/eval_baseline_run001" \
  --device "${DEVICE:-cuda}" --amp --batch_size 64 \
  --threshold 0.55 --threshold_on 0.72 --threshold_off 0.42 \
  --smooth_alpha 0.55 --pose_smooth_alpha 0.60 \
  --flow_blend 0.15 --flow_win 21 --flow_max_err 20 --flow_min_valid 6 \
  --roi_refine --roi_refine_weight 0.35 --roi_context 1.45 --roi_min_size 96 --roi_score_gate 0.30 \
  --max_center_jump_frac 0.04 --min_area_ratio 0.70 --max_area_ratio 1.45 --area_ema 0.95

echo "========================================"
echo "Evaluating trained checkpoint"
echo "========================================"

"$PYTHON_BIN" "$REPRO/eval_objectron_tracking.py" \
  --checkpoint "$RUN_DIR/best_model.pt" \
  --video_dirs_txt "$RUN_DIR/included_blue_dirs.txt" "$RUN_DIR/included_negative_dirs.txt" \
  --out_dir "$RUN_DIR/eval_all" \
  --device "${DEVICE:-cuda}" --amp --batch_size 64 \
  --threshold 0.55 --threshold_on 0.72 --threshold_off 0.42 \
  --smooth_alpha 0.55 --pose_smooth_alpha 0.60 \
  --flow_blend 0.15 --flow_win 21 --flow_max_err 20 --flow_min_valid 6 \
  --roi_refine --roi_refine_weight 0.35 --roi_context 1.45 --roi_min_size 96 --roi_score_gate 0.30 \
  --max_center_jump_frac 0.04 --min_area_ratio 0.70 --max_area_ratio 1.45 --area_ema 0.95

echo "Done. Summary:"
echo "-- baseline run001 --"
cat "$RUN_DIR/eval_baseline_run001/summary.txt"
echo "-- trained model --"
cat "$RUN_DIR/eval_all/summary.txt"
