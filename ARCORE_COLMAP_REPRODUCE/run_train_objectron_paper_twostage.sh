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

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${RUN_DIR:-$REPRO/paper_twostage_$STAMP}"
mkdir -p "$RUN_DIR"

printf "%s\n" "${BLUE_DIRS[@]}" > "$RUN_DIR/included_blue_dirs.txt"
printf "%s\n" "${RED_DIRS[@]}" > "$RUN_DIR/included_red_dirs.txt"
printf "%s\n" "${SILVER_DIRS[@]}" > "$RUN_DIR/included_silver_dirs.txt"
printf "%s\n" "${NEG_DIRS[@]}" > "$RUN_DIR/included_negative_dirs.txt"

DETECTOR_EPOCHS="${DETECTOR_EPOCHS:-60}"
DETECTOR_BATCH_SIZE="${DETECTOR_BATCH_SIZE:-8}"
DETECTOR_LR="${DETECTOR_LR:-3e-4}"
DETECTOR_WEIGHT_DECAY="${DETECTOR_WEIGHT_DECAY:-1e-4}"
DETECTOR_CONTEXT="${DETECTOR_CONTEXT:-1.35}"
DETECTOR_MIN_SIZE="${DETECTOR_MIN_SIZE:-48}"
DETECTOR_SCORE_THRESH="${DETECTOR_SCORE_THRESH:-0.35}"

REGRESSOR_EPOCHS="${REGRESSOR_EPOCHS:-250}"
REGRESSOR_BATCH_SIZE="${REGRESSOR_BATCH_SIZE:-64}"
REGRESSOR_LR="${REGRESSOR_LR:-1e-2}"
REGRESSOR_LR_FINAL="${REGRESSOR_LR_FINAL:-1e-6}"
REGRESSOR_WEIGHT_DECAY="${REGRESSOR_WEIGHT_DECAY:-1e-4}"
REGRESSOR_BACKBONE="${REGRESSOR_BACKBONE:-efficientnet_lite0}"
REGRESSOR_DROPOUT="${REGRESSOR_DROPOUT:-0.0}"

NUM_WORKERS="${NUM_WORKERS:-8}"
TRAIN_RATIO="${TRAIN_RATIO:-0.82}"
VAL_RATIO="${VAL_RATIO:-0.10}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-cuda}"
CROP_CONTEXT="${CROP_CONTEXT:-1.35}"
CROP_MIN_SIZE="${CROP_MIN_SIZE:-96}"
CROP_JITTER="${CROP_JITTER:-0.08}"
ROT_MAX_DEG="${ROT_MAX_DEG:-180}"
AUGMENT="${AUGMENT:-1}"
DETECTOR_AUGMENT="${DETECTOR_AUGMENT:-1}"
SYMMETRY_AWARE="${SYMMETRY_AWARE:-1}"

AUG_FLAGS=()
if [[ "$AUGMENT" == "1" ]]; then
  AUG_FLAGS+=(--augment)
fi

DET_AUG_FLAGS=()
if [[ "$DETECTOR_AUGMENT" == "1" ]]; then
  DET_AUG_FLAGS+=(--detector_augment)
fi

SYM_FLAGS=()
if [[ "$SYMMETRY_AWARE" == "1" ]]; then
  SYM_FLAGS+=(--symmetry_aware)
fi

set -x
"$PYTHON_BIN" "$REPRO/train_objectron_paper_twostage.py" \
  --pos_dirs "${BLUE_DIRS[@]}" \
  --neg_dirs "${NEG_DIRS[@]}" \
  --output_dir "$RUN_DIR" \
  --train_ratio "$TRAIN_RATIO" \
  --val_ratio "$VAL_RATIO" \
  --seed "$SEED" \
  --device "$DEVICE" \
  --amp \
  --num_workers "$NUM_WORKERS" \
  --detector_epochs "$DETECTOR_EPOCHS" \
  --detector_batch_size "$DETECTOR_BATCH_SIZE" \
  --detector_lr "$DETECTOR_LR" \
  --detector_weight_decay "$DETECTOR_WEIGHT_DECAY" \
  --detector_context "$DETECTOR_CONTEXT" \
  --detector_min_size "$DETECTOR_MIN_SIZE" \
  --detector_score_thresh "$DETECTOR_SCORE_THRESH" \
  --regressor_backbone "$REGRESSOR_BACKBONE" \
  --regressor_epochs "$REGRESSOR_EPOCHS" \
  --regressor_batch_size "$REGRESSOR_BATCH_SIZE" \
  --regressor_lr "$REGRESSOR_LR" \
  --regressor_lr_final "$REGRESSOR_LR_FINAL" \
  --regressor_weight_decay "$REGRESSOR_WEIGHT_DECAY" \
  --regressor_dropout "$REGRESSOR_DROPOUT" \
  --crop_context "$CROP_CONTEXT" \
  --crop_min_size "$CROP_MIN_SIZE" \
  --crop_jitter "$CROP_JITTER" \
  --rot_max_deg "$ROT_MAX_DEG" \
  "${AUG_FLAGS[@]}" \
  "${DET_AUG_FLAGS[@]}" \
  "${SYM_FLAGS[@]}" | tee "$RUN_DIR/train.log"
set +x

echo "Training complete. Artifacts in: $RUN_DIR"
