#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPRO="$SCRIPT_DIR"
ROOT="$(cd "$REPRO/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

BLUE_EXCLUDE=(
  video_05_blue
  video_17_blue
  video_35_blue
  video_37_blue
)

RED_EXCLUDE=(
  video_03_red
  video_04_red
  video_05_red
  video_06_red
  video_07_red
  video_08_red
  video_09_red
  video_10_red
)

SILVER_EXCLUDE=(
  video_01_silver
  video_08_silver
)

contains() {
  local needle="$1"; shift
  local x
  for x in "$@"; do
    if [[ "$x" == "$needle" ]]; then
      return 0
    fi
  done
  return 1
}

collect_dirs() {
  local cls_dir="$1"; shift
  local -n _out_ref="$1"; shift
  local excludes=("$@")
  local p name
  _out_ref=()
  while IFS= read -r -d '' p; do
    name="$(basename "$p")"
    if contains "$name" "${excludes[@]}"; then
      continue
    fi
    if [[ -f "$p/objectron_prep/annotations.json" ]]; then
      _out_ref+=("$p")
    fi
  done < <(find "$cls_dir" -mindepth 1 -maxdepth 1 -type d -name 'video_*' -print0 | sort -z)
}

BLUE_DIRS=()
RED_DIRS=()
SILVER_DIRS=()
NEG_DIRS=()

collect_dirs "$REPRO/Blue" BLUE_DIRS "${BLUE_EXCLUDE[@]}"
collect_dirs "$REPRO/Red" RED_DIRS "${RED_EXCLUDE[@]}"
collect_dirs "$REPRO/Silver" SILVER_DIRS "${SILVER_EXCLUDE[@]}"

NEG_DIRS=("${RED_DIRS[@]}" "${SILVER_DIRS[@]}")

echo "Using curated dataset:"
echo "  Positive (Blue): ${#BLUE_DIRS[@]} videos"
echo "  Negative (Red): ${#RED_DIRS[@]} videos"
echo "  Negative (Silver): ${#SILVER_DIRS[@]} videos"

if [[ ${#BLUE_DIRS[@]} -eq 0 || ${#NEG_DIRS[@]} -eq 0 ]]; then
  echo "ERROR: empty positive or negative directory list."
  exit 1
fi

INIT_CKPT="${INIT_CKPT:-$REPRO/posneg_run_001/best_model.pt}"
if [[ ! -f "$INIT_CKPT" ]]; then
  echo "ERROR: init checkpoint not found: $INIT_CKPT"
  exit 1
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${RUN_DIR:-$REPRO/posneg_run_002_finetune_$STAMP}"
mkdir -p "$RUN_DIR"

echo "Run dir: $RUN_DIR"
echo "Init checkpoint: $INIT_CKPT"

printf "%s\n" "${BLUE_DIRS[@]}" > "$RUN_DIR/included_blue_dirs.txt"
printf "%s\n" "${RED_DIRS[@]}" > "$RUN_DIR/included_red_dirs.txt"
printf "%s\n" "${SILVER_DIRS[@]}" > "$RUN_DIR/included_silver_dirs.txt"
printf "%s\n" "${NEG_DIRS[@]}" > "$RUN_DIR/included_negative_dirs.txt"

EPOCHS="${EPOCHS:-40}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-8}"
LR="${LR:-5e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
KP_WEIGHT="${KP_WEIGHT:-5.0}"
TRAIN_RATIO="${TRAIN_RATIO:-0.82}"
VAL_RATIO="${VAL_RATIO:-0.10}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-cuda}"
AUGMENT="${AUGMENT:-1}"
ROI_TRAIN="${ROI_TRAIN:-1}"
ROI_CONTEXT="${ROI_CONTEXT:-1.45}"
ROI_JITTER="${ROI_JITTER:-0.08}"
ROI_MIN_SIZE="${ROI_MIN_SIZE:-96}"
ROI_PROB="${ROI_PROB:-0.75}"
ROT_MAX_DEG="${ROT_MAX_DEG:-180}"
SYMMETRY_AWARE="${SYMMETRY_AWARE:-1}"

AUG_FLAGS=()
if [[ "$AUGMENT" == "1" ]]; then
  AUG_FLAGS+=(--augment)
fi

ROI_FLAGS=()
if [[ "$ROI_TRAIN" == "1" ]]; then
  ROI_FLAGS+=(--roi_train --roi_context "$ROI_CONTEXT" --roi_jitter "$ROI_JITTER" --roi_min_size "$ROI_MIN_SIZE" --roi_prob "$ROI_PROB")
fi

AUG_EXTRAS=(--rot_max_deg "$ROT_MAX_DEG")
if [[ "$SYMMETRY_AWARE" == "1" ]]; then
  AUG_EXTRAS+=(--symmetry_aware)
fi

set -x
"$PYTHON_BIN" "$REPRO/train_objectron_can_posneg.py" \
  --pos_dirs "${BLUE_DIRS[@]}" \
  --neg_dirs "${NEG_DIRS[@]}" \
  --output_dir "$RUN_DIR" \
  --image_size 320 \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --lr "$LR" \
  --weight_decay "$WEIGHT_DECAY" \
  --kp_weight "$KP_WEIGHT" \
  --train_ratio "$TRAIN_RATIO" \
  --val_ratio "$VAL_RATIO" \
  --split_mode video \
  --seed "$SEED" \
  --device "$DEVICE" \
  --amp \
  --balance \
  --auto_pos_weight \
  "${AUG_FLAGS[@]}" \
  "${AUG_EXTRAS[@]}" \
  "${ROI_FLAGS[@]}" \
  --init_checkpoint "$INIT_CKPT" | tee "$RUN_DIR/train.log"
set +x

echo "Training complete. Artifacts in: $RUN_DIR"
