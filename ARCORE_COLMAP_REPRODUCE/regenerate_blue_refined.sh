#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPRO="$SCRIPT_DIR"
CLASS=Blue
SAM_CKPT="$ROOT/sam2_checkpoints/sam2.1_hiera_large.pt"
SAM_CFG="configs/sam2.1/sam2.1_hiera_l.yaml"

source "$ROOT/venv_sam2/bin/activate"

SAM2_PYTHON="${SAM2_PYTHON:-python}"
GEN_PYTHON="${GEN_PYTHON:-python}"
SAM2_DEVICE="${SAM2_DEVICE:-auto}"
SKIP_REMASK="${SKIP_REMASK:-0}"
START_FROM="${START_FROM:-}"

has_mod() {
  local py="$1"
  local mod="$2"
  "$py" - <<PY >/dev/null 2>&1
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("$mod") else 1)
PY
}

if ! has_mod "$GEN_PYTHON" cv2; then
  if has_mod "$SAM2_PYTHON" cv2; then
    echo "GEN_PYTHON='$GEN_PYTHON' has no cv2; switching to SAM2_PYTHON='$SAM2_PYTHON'"
    GEN_PYTHON="$SAM2_PYTHON"
  else
    echo "ERROR: OpenCV (cv2) not found in GEN_PYTHON='$GEN_PYTHON' or SAM2_PYTHON='$SAM2_PYTHON'"
    echo "Activate venv_sam2 or set GEN_PYTHON to a python with cv2 installed."
    exit 1
  fi
fi

if [ "$SAM2_DEVICE" = "auto" ]; then
  if "$SAM2_PYTHON" - <<'PY' >/dev/null 2>&1
import torch
raise SystemExit(0 if torch.cuda.is_available() else 1)
PY
  then
    SAM2_DEVICE="cuda"
  else
    SAM2_DEVICE="cpu"
  fi
fi

GPU_ID="${GPU_ID:-0}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"

VIDEOS=(
  video_04_blue video_05_blue video_06_blue video_08_blue video_10_blue
  video_11_blue video_12_blue video_19_blue video_25_blue video_30_blue
  video_32_blue video_33_blue video_34_blue video_35_blue video_36_blue
  video_37_blue video_38_blue
)

REMASK_VIDEOS=(video_12_blue video_30_blue video_32_blue)

is_in_array() {
  local needle="$1"; shift
  local x
  for x in "$@"; do
    if [ "$x" = "$needle" ]; then
      return 0
    fi
  done
  return 1
}

count_masks() {
  local d="$1"
  [ -d "$d" ] || { echo 0; return; }
  find "$d" -maxdepth 1 -type f -name 'frame_*.png' | wc -l
}

count_frames() {
  local d="$1"
  [ -d "$d" ] || { echo 0; return; }
  find "$d" -maxdepth 1 -type f -name '*.png' | wc -l
}

pick_pointcloud() {
  local out="$1"
  local d="$out/scaled_masked"
  if [ -f "$d/points3D_clean_eps0p0038.ply" ]; then
    echo "$d/points3D_clean_eps0p0038.ply"
    return
  fi
  if [ -f "$d/points3D_clean_eps0p0020.ply" ]; then
    echo "$d/points3D_clean_eps0p0020.ply"
    return
  fi
  if [ -f "$d/points3D.ply" ]; then
    echo "$d/points3D.ply"
    return
  fi
  echo ""
}

height_trim_for_video() {
  local v="$1"
  case "$v" in
    video_25_blue) echo "0.12" ;;
    video_10_blue|video_11_blue) echo "0.10" ;;
    video_30_blue|video_32_blue) echo "0.06" ;;
    video_05_blue|video_34_blue|video_37_blue|video_38_blue) echo "0.06" ;;
    *) echo "0.08" ;;
  esac
}

horizontal_trim_for_video() {
  local v="$1"
  case "$v" in
    video_28_blue) echo "0.03" ;;   # user requested wider bbox for this one
    video_10_blue|video_11_blue) echo "0.10" ;;
    video_30_blue|video_32_blue) echo "0.05" ;;
    video_05_blue|video_34_blue|video_37_blue) echo "0.04" ;;
    *) echo "0.08" ;;
  esac
}

target_color_for_video() {
  local _v="$1"
  echo "blue"
}

center_zoom_for_video() {
  local v="$1"
  case "$v" in
    video_30_blue|video_32_blue) echo "1.60" ;;
    *) echo "1.00" ;;
  esac
}

use_snap_for_video() {
  local v="$1"
  case "$v" in
    video_34_blue|video_35_blue|video_36_blue|video_37_blue|video_38_blue) return 0 ;;
    *) return 1 ;;
  esac
}

run_sam2_until_done() {
  local rotated="$1"
  local masks="$2"
  local target_color="$3"
  local center_zoom="$4"
  local nframes
  nframes="$(count_frames "$rotated")"
  if [ "$nframes" -eq 0 ]; then
    echo "No frames found: $rotated"
    return 1
  fi

  mkdir -p "$masks"

  local pps=16
  local crop=0
  local track=80
  local attempt=0

  while :; do
    local before after
    before="$(count_masks "$masks")"
    if [ "$before" -ge "$nframes" ]; then
      break
    fi

    attempt=$((attempt + 1))
    if [ "$attempt" -gt 30 ]; then
      echo "SAM2 did not finish after $attempt attempts ($before/$nframes)"
      return 1
    fi

    echo "SAM2 attempt $attempt ($before/$nframes) pps=$pps crop=$crop track=$track"
    if [ "$SAM2_DEVICE" = "cuda" ]; then
      env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:64 \
        "$SAM2_PYTHON" "$ROOT/DATASET GENERATION/run_sam2_video_mask.py" \
          --frames_dir "$rotated" \
          --out_dir "$masks" \
          --ckpt "$SAM_CKPT" \
          --cfg "$SAM_CFG" \
          --device cuda \
          --safe_cuda \
          --target_color "$target_color" \
          --center_zoom "$center_zoom" \
          --points_per_side "$pps" \
          --crop_n_layers "$crop" \
          --max_track_frames "$track" || true
    else
      env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
        "$SAM2_PYTHON" "$ROOT/DATASET GENERATION/run_sam2_video_mask.py" \
          --frames_dir "$rotated" \
          --out_dir "$masks" \
          --ckpt "$SAM_CKPT" \
          --cfg "$SAM_CFG" \
          --device cpu \
          --target_color "$target_color" \
          --center_zoom "$center_zoom" \
          --points_per_side "$pps" \
          --crop_n_layers "$crop" \
          --max_track_frames "$track" || true
    fi

    after="$(count_masks "$masks")"
    if [ "$after" -gt "$before" ]; then
      continue
    fi

    if [ "$track" -gt 20 ]; then
      track=$((track / 2))
      [ "$track" -lt 20 ] && track=20
      continue
    fi
    if [ "$pps" -gt 8 ]; then
      pps=$((pps - 4))
      [ "$pps" -lt 8 ] && pps=8
      continue
    fi
    echo "SAM2 made no progress with minimum settings."
    return 1
  done

  local done_count
  done_count="$(count_masks "$masks")"
  echo "SAM2 done: $done_count/$nframes masks"
  [ "$done_count" -ge "$nframes" ]
}

rebuild_scaled_from_masks() {
  local out="$1"
  local txt="$out/colmap/txt"
  local colmap_db="$out/colmap/database.db"
  local masks="$out/masks_sam2"
  local scaled="$out/scaled_masked"

  rm -rf "$scaled"
  mkdir -p "$scaled"

  "$GEN_PYTHON" "$ROOT/DATASET GENERATION/run_masked_filter_generic.py" \
    --colmap_txt "$txt" \
    --masks_dir "$masks" \
    --database_db "$colmap_db" \
    --out_dir "$scaled" \
    --real_height 0.13 \
    --min_total_obs 4 \
    --min_inmask_ratio 0.10 \
    --skip_desc_db || return 1

  "$GEN_PYTHON" "$ROOT/DATASET GENERATION/npz_viewer.py" \
    --input_ply "$scaled/points3D.ply" \
    --output_ply "$scaled/points3D_clean_eps0p0038.ply" \
    --eps 0.0038 \
    --min_points 5 \
    --keep_top_k 1 \
    --no_view || return 1

  return 0
}

FAILED=()

for V in "${VIDEOS[@]}"; do
  if [ -n "$START_FROM" ] && [[ "$V" < "$START_FROM" ]]; then
    continue
  fi

  OUT="$REPRO/$CLASS/$V"
  ROTATED="$OUT/frames_rotated"
  MASKS="$OUT/masks_sam2"
  OBJ="$OUT/objectron_prep"
  SCALED="$OUT/scaled_masked"
  TARGET_COLOR="$(target_color_for_video "$V")"
  CENTER_ZOOM="$(center_zoom_for_video "$V")"

  echo "=============================="
  echo "Refining: $CLASS/$V"
  echo "=============================="

  if [ ! -d "$OUT" ]; then
    echo "Missing video output dir: $OUT"
    FAILED+=("$CLASS/$V")
    continue
  fi

  if is_in_array "$V" "${REMASK_VIDEOS[@]}" && [ "$SKIP_REMASK" != "1" ]; then
    echo "Full remask + scaled rebuild for $V"
    rm -rf "$MASKS" "$OUT/_sam2_numeric_frames" "$SCALED" "$OBJ"
    mkdir -p "$OBJ"

    if ! run_sam2_until_done "$ROTATED" "$MASKS" "$TARGET_COLOR" "$CENTER_ZOOM"; then
      echo "SAM2 failed on $V"
      FAILED+=("$CLASS/$V")
      continue
    fi

    if ! rebuild_scaled_from_masks "$OUT"; then
      echo "Scaled masked rebuild failed on $V"
      FAILED+=("$CLASS/$V")
      continue
    fi
  fi

  PC="$(pick_pointcloud "$OUT")"
  if [ -z "$PC" ]; then
    echo "No usable point cloud for $V"
    FAILED+=("$CLASS/$V")
    continue
  fi

  HTRIM="$(height_trim_for_video "$V")"
  XTRIM="$(horizontal_trim_for_video "$V")"
  SNAP_ARGS=()
  if use_snap_for_video "$V"; then
    SNAP_ARGS=(--snap_bbox_to_mask --snap_iou_threshold 0.70)
  fi

  rm -rf "$OBJ/overlays" "$OBJ/annotations.json"
  mkdir -p "$OBJ/overlays"

  "$GEN_PYTHON" "$ROOT/DATASET GENERATION/generate+objectron_dataset_colmap.py" \
    --point_cloud "$PC" \
    --cameras_txt "$SCALED/cameras.txt" \
    --images_txt "$SCALED/images.txt" \
    --image_folder "$ROTATED" \
    --overlay_dir "$OBJ/overlays" \
    --output_json "$OBJ/annotations.json" \
    --stride 1 \
    --start 0 \
    --sort_by name \
    --height_trim "$HTRIM" \
    --height_trim_mode auto \
    --horizontal_trim "$XTRIM" \
    --masks_dir "$MASKS" \
    --fit_to_masks \
    --fit_sample_stride 1 \
    --fit_max_frames 2000 \
    "${SNAP_ARGS[@]}" || {
      echo "Annotation regeneration failed on $V"
      FAILED+=("$CLASS/$V")
      continue
    }

  echo "DONE: $CLASS/$V"
done

echo "=============================="
if [ "${#FAILED[@]}" -gt 0 ]; then
  echo "Refinement finished with failures:"
  printf '  - %s\n' "${FAILED[@]}"
  exit 1
else
  echo "Refinement finished successfully."
fi
echo "=============================="
