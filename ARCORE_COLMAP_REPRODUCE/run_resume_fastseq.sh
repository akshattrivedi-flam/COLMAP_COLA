#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_ROOT="$ROOT/ARCORE_COLMAP_DATA"
REPRO_ROOT="$SCRIPT_DIR"
SAM_CKPT="$ROOT/sam2_checkpoints/sam2.1_hiera_large.pt"
SAM_CFG="configs/sam2.1/sam2.1_hiera_l.yaml"

source "$ROOT/venv_sam2/bin/activate"

GPU_ID="${GPU_ID:-0}"
CPU_THREADS="${CPU_THREADS:-1}"
MAPPER_THREADS="${MAPPER_THREADS:-8}"
MAPPER_SAFE_THREADS="${MAPPER_SAFE_THREADS:-1}"
SAFE_CUDA="${SAFE_CUDA:-1}"
SAM2_DEVICE="${SAM2_DEVICE:-cuda}"
SAM2_POINTS_PER_SIDE="${SAM2_POINTS_PER_SIDE:-16}"
SAM2_CROP_N_LAYERS="${SAM2_CROP_N_LAYERS:-0}"
SAM2_MAX_TRACK_FRAMES="${SAM2_MAX_TRACK_FRAMES:-120}"
SAM2_MIN_POINTS_PER_SIDE="${SAM2_MIN_POINTS_PER_SIDE:-8}"
SAM2_MIN_TRACK_FRAMES="${SAM2_MIN_TRACK_FRAMES:-20}"
SAM2_DISABLE_CPU_FALLBACK="${SAM2_DISABLE_CPU_FALLBACK:-1}"
FORCE_REGEN_ANN="${FORCE_REGEN_ANN:-0}"

export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export OMP_NUM_THREADS="$CPU_THREADS"
export OPENBLAS_NUM_THREADS="$CPU_THREADS"
export MKL_NUM_THREADS="$CPU_THREADS"
export NUMEXPR_NUM_THREADS="$CPU_THREADS"
ulimit -c 0

FAILED_FILE="$REPRO_ROOT/_failed_resume_$(date +%F_%H%M%S).txt"
: > "$FAILED_FILE"

CLASS_ONLY="${CLASS_ONLY:-}"
START_FROM="${START_FROM:-}"
if [ -n "$START_FROM" ] && [ -z "$CLASS_ONLY" ]; then
  echo "ERROR: START_FROM requires CLASS_ONLY (e.g., CLASS_ONLY=Blue START_FROM=video_19_blue)"
  exit 1
fi

count_png() {
  local d="$1"
  [ -d "$d" ] || { echo 0; return; }
  find "$d" -maxdepth 1 -type f -name "*.png" | wc -l
}

is_done() {
  local out="$1"
  python - "$out" <<'PY'
from pathlib import Path
import json, sys
out = Path(sys.argv[1])
ann = out/'objectron_prep/annotations.json'
ov  = out/'objectron_prep/overlays'
if not ann.exists() or not ov.exists():
    print('0'); raise SystemExit
try:
    data = json.loads(ann.read_text())
    ann_n = len(data) if isinstance(data, list) else 0
except Exception:
    print('0'); raise SystemExit
ov_n = len(list(ov.glob('*.png')))
print('1' if ann_n > 0 and ov_n >= ann_n else '0')
PY
}

target_color_for_class() {
  local class_name="$1"
  case "$class_name" in
    Blue) echo "blue" ;;
    Red) echo "red" ;;
    Silver) echo "silver" ;;
    *) echo "none" ;;
  esac
}

center_zoom_for_video() {
  local class_name="$1"
  local vname="$2"
  if [ "$class_name" = "Blue" ] && { [ "$vname" = "video_30_blue" ] || [ "$vname" = "video_32_blue" ]; }; then
    echo "1.60"
  else
    echo "1.00"
  fi
}

process_video() {
  local CLASS="$1"
  local SRC_VID="$2"
  local VNAME
  VNAME="$(basename "$SRC_VID")"

  local SRC_RGB="$SRC_VID/rgb"
  [ -d "$SRC_RGB" ] || SRC_RGB="$SRC_VID"

  local OUT="$REPRO_ROOT/$CLASS/$VNAME"
  local ROTATED="$OUT/frames_rotated"
  local COLMAP_DIR="$OUT/colmap"
  local TXT_DIR="$COLMAP_DIR/txt"
  local MASKS="$OUT/masks_sam2"
  local SCALED="$OUT/scaled_masked"
  local OBJ="$OUT/objectron_prep"
  local TARGET_COLOR
  TARGET_COLOR="$(target_color_for_class "$CLASS")"
  local CENTER_ZOOM
  CENTER_ZOOM="$(center_zoom_for_video "$CLASS" "$VNAME")"

  mkdir -p "$OUT"

  echo "=============================="
  echo "Processing: $CLASS / $VNAME"
  echo "=============================="

  # 1) Rotate frames if missing
  if [ "$(count_png "$ROTATED")" -eq 0 ]; then
    rm -rf "$ROTATED"
    mkdir -p "$ROTATED"
    python - "$SRC_RGB" "$ROTATED" <<'PY'
import cv2, sys
from pathlib import Path
src = Path(sys.argv[1]); dst = Path(sys.argv[2])
imgs = sorted([p for p in src.rglob('*') if p.suffix.lower() in {'.jpg','.jpeg','.png'}])
n = 0
for i, p in enumerate(imgs):
    im = cv2.imread(str(p))
    if im is None:
        continue
    im = cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
    cv2.imwrite(str(dst / f'frame_{i:06d}.png'), im)
    n += 1
print('rotated_frames', n)
if n == 0:
    raise SystemExit('No images found.')
PY
  fi

  local NFRAMES
  NFRAMES="$(count_png "$ROTATED")"

  # 2) COLMAP + TXT if missing
  if [ ! -f "$TXT_DIR/points3D.txt" ]; then
    rm -rf "$COLMAP_DIR"
    mkdir -p "$COLMAP_DIR/sparse" "$TXT_DIR"

    QT_QPA_PLATFORM=offscreen colmap feature_extractor \
      --database_path "$COLMAP_DIR/database.db" \
      --image_path "$ROTATED" \
      --FeatureExtraction.num_threads "$CPU_THREADS" \
      --SiftExtraction.max_num_features 4096 || return 1

    QT_QPA_PLATFORM=offscreen colmap sequential_matcher \
      --database_path "$COLMAP_DIR/database.db" \
      --FeatureMatching.num_threads "$CPU_THREADS" \
      --SequentialMatching.overlap 30 || return 1

    QT_QPA_PLATFORM=offscreen colmap mapper \
      --database_path "$COLMAP_DIR/database.db" \
      --image_path "$ROTATED" \
      --output_path "$COLMAP_DIR/sparse" \
      --Mapper.ba_global_max_num_iterations 1 \
      --Mapper.ba_global_max_refinements 0 \
      --Mapper.ba_local_max_num_iterations 5 \
      --Mapper.ba_local_max_refinements 1 \
      --Mapper.max_num_models 1 \
      --Mapper.num_threads "$MAPPER_THREADS" || {
        echo "Mapper failed; retrying safe mode for $CLASS/$VNAME"
        rm -rf "$COLMAP_DIR/sparse"
        mkdir -p "$COLMAP_DIR/sparse"
        QT_QPA_PLATFORM=offscreen colmap mapper \
          --database_path "$COLMAP_DIR/database.db" \
          --image_path "$ROTATED" \
          --output_path "$COLMAP_DIR/sparse" \
          --Mapper.ba_global_max_num_iterations 1 \
          --Mapper.ba_global_max_refinements 0 \
          --Mapper.ba_local_max_num_iterations 3 \
          --Mapper.ba_local_max_refinements 1 \
          --Mapper.max_num_models 1 \
          --Mapper.num_threads "$MAPPER_SAFE_THREADS" || return 1
      }

    [ -d "$COLMAP_DIR/sparse/0" ] || return 1

    if ! QT_QPA_PLATFORM=offscreen colmap model_converter \
      --input_path "$COLMAP_DIR/sparse/0" \
      --output_path "$TXT_DIR" \
      --output_type TXT; then
      echo "model_converter failed; trying pycolmap fallback for $CLASS/$VNAME"
      python - <<PY
import pycolmap
from pathlib import Path
inp = Path("$COLMAP_DIR/sparse/0")
out = Path("$TXT_DIR")
out.mkdir(parents=True, exist_ok=True)
recon = pycolmap.Reconstruction(str(inp))
recon.write_text(str(out))
print('pycolmap wrote TXT to', out)
PY
    fi

    [ -f "$TXT_DIR/points3D.txt" ] || return 1
  fi

  # 3) SAM2 masks with resume/retry
  local NMASKS
  NMASKS="$(find "$MASKS" -maxdepth 1 -type f -name 'frame_*.png' 2>/dev/null | wc -l || true)"
  if [ "${NMASKS:-0}" -lt "$NFRAMES" ]; then
    mkdir -p "$MASKS"
    SAM2_SAFE_FLAG=()
    if [ "$SAFE_CUDA" = "1" ]; then
      SAM2_SAFE_FLAG+=(--safe_cuda)
    fi

    local sam_attempt=0
    local before=0
    local sam_points="$SAM2_POINTS_PER_SIDE"
    local sam_crop="$SAM2_CROP_N_LAYERS"
    local sam_track="$SAM2_MAX_TRACK_FRAMES"
    while :; do
      NMASKS="$(find "$MASKS" -maxdepth 1 -type f -name 'frame_*.png' 2>/dev/null | wc -l || true)"
      [ "${NMASKS:-0}" -ge "$NFRAMES" ] && break

      sam_attempt=$((sam_attempt + 1))
      if [ "$sam_attempt" -gt 20 ]; then
        echo "SAM2 did not finish after $sam_attempt attempts for $CLASS/$VNAME"
        return 1
      fi

      before="${NMASKS:-0}"
      echo "SAM2 attempt $sam_attempt for $CLASS/$VNAME ($before/$NFRAMES masks) [dev=$SAM2_DEVICE pps=$sam_points crop=$sam_crop track=$sam_track]"

      local sam_rc=0

      if [ "$SAM2_DEVICE" = "cpu" ]; then
        env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
          python "$ROOT/DATASET GENERATION/run_sam2_video_mask.py" \
          --frames_dir "$ROTATED" \
          --out_dir "$MASKS" \
          --ckpt "$SAM_CKPT" \
	          --cfg "$SAM_CFG" \
	          --target_color "$TARGET_COLOR" \
	          --center_zoom "$CENTER_ZOOM" \
	          --points_per_side "$sam_points" \
	          --crop_n_layers "$sam_crop" \
	          --max_track_frames "$sam_track" \
          --device cpu || sam_rc=$?
      else
        env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
          PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:64 \
          python "$ROOT/DATASET GENERATION/run_sam2_video_mask.py" \
          --frames_dir "$ROTATED" \
          --out_dir "$MASKS" \
          --ckpt "$SAM_CKPT" \
	          --cfg "$SAM_CFG" \
	          --target_color "$TARGET_COLOR" \
	          --center_zoom "$CENTER_ZOOM" \
	          --points_per_side "$sam_points" \
	          --crop_n_layers "$sam_crop" \
	          --max_track_frames "$sam_track" \
          --device cuda \
          "${SAM2_SAFE_FLAG[@]}" || sam_rc=$?

        if [ "$sam_rc" -ne 0 ] && [ "$SAM2_DISABLE_CPU_FALLBACK" != "1" ]; then
          echo "SAM2 CUDA failed; retrying CPU for $CLASS/$VNAME"
          env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
            python "$ROOT/DATASET GENERATION/run_sam2_video_mask.py" \
            --frames_dir "$ROTATED" \
            --out_dir "$MASKS" \
            --ckpt "$SAM_CKPT" \
	            --cfg "$SAM_CFG" \
	            --target_color "$TARGET_COLOR" \
	            --center_zoom "$CENTER_ZOOM" \
	            --points_per_side "$sam_points" \
	            --crop_n_layers "$sam_crop" \
	            --max_track_frames "$sam_track" \
            --device cpu || sam_rc=$?
        fi
      fi

      NMASKS="$(find "$MASKS" -maxdepth 1 -type f -name 'frame_*.png' 2>/dev/null | wc -l || true)"
      if [ "${NMASKS:-0}" -le "$before" ]; then
        if [ "$SAM2_DEVICE" = "cuda" ]; then
          if [ "$sam_track" -gt "$SAM2_MIN_TRACK_FRAMES" ]; then
            sam_track=$((sam_track / 2))
            if [ "$sam_track" -lt "$SAM2_MIN_TRACK_FRAMES" ]; then
              sam_track="$SAM2_MIN_TRACK_FRAMES"
            fi
            echo "SAM2 made no progress; reducing track chunk to $sam_track for $CLASS/$VNAME"
            continue
          fi
          if [ "$sam_points" -gt "$SAM2_MIN_POINTS_PER_SIDE" ]; then
            sam_points=$((sam_points - 4))
            if [ "$sam_points" -lt "$SAM2_MIN_POINTS_PER_SIDE" ]; then
              sam_points="$SAM2_MIN_POINTS_PER_SIDE"
            fi
            echo "SAM2 made no progress; reducing points_per_side to $sam_points for $CLASS/$VNAME"
            continue
          fi
          if [ "$sam_crop" -gt 0 ]; then
            sam_crop=0
            echo "SAM2 made no progress; forcing crop_n_layers=0 for $CLASS/$VNAME"
            continue
          fi
        fi

        echo "SAM2 made no progress on attempt $sam_attempt for $CLASS/$VNAME"
        if [ "$sam_rc" -ne 0 ]; then
          echo "Last SAM2 exit code: $sam_rc"
        fi
        return 1
      fi

      if [ "$sam_rc" -ne 0 ]; then
        echo "SAM2 progressed despite non-zero exit ($sam_rc); continuing for $CLASS/$VNAME"
      fi
    done
  fi

  # 4) Masked filtering + point-cloud clean
  if [ ! -f "$SCALED/points3D_clean_eps0p0038.ply" ]; then
    rm -rf "$SCALED"
    mkdir -p "$SCALED"

    python "$ROOT/DATASET GENERATION/run_masked_filter_generic.py" \
      --colmap_txt "$TXT_DIR" \
      --masks_dir "$MASKS" \
      --database_db "$COLMAP_DIR/database.db" \
      --out_dir "$SCALED" \
      --real_height 0.13 \
      --min_total_obs 4 \
      --min_inmask_ratio 0.10 \
      --skip_desc_db || return 1

    python "$ROOT/DATASET GENERATION/npz_viewer.py" \
      --input_ply "$SCALED/points3D.ply" \
      --output_ply "$SCALED/points3D_clean_eps0p0038.ply" \
      --eps 0.0038 \
      --min_points 5 \
      --keep_top_k 1 \
      --no_view || return 1
  fi

  # 5) Overlays + annotations
  if [ "$FORCE_REGEN_ANN" = "1" ] || [ "$(is_done "$OUT")" != "1" ]; then
    rm -rf "$OBJ/overlays" "$OBJ/annotations.json"
    mkdir -p "$OBJ/overlays"

	    python "$ROOT/DATASET GENERATION/generate+objectron_dataset_colmap.py" \
	      --point_cloud "$SCALED/points3D_clean_eps0p0038.ply" \
	      --cameras_txt "$SCALED/cameras.txt" \
	      --images_txt "$SCALED/images.txt" \
	      --image_folder "$ROTATED" \
	      --overlay_dir "$OBJ/overlays" \
	      --output_json "$OBJ/annotations.json" \
	      --stride 1 \
	      --start 0 \
	      --sort_by name \
	      --height_trim 0.08 \
	      --height_trim_mode auto \
	      --horizontal_trim 0.08 \
	      --masks_dir "$MASKS" \
	      --fit_to_masks \
	      --fit_sample_stride 3 \
	      --fit_max_frames 360 || return 1
	  fi

  echo "DONE: $CLASS/$VNAME"
  return 0
}

for CLASS in Blue Red Silver; do
  if [ -n "$CLASS_ONLY" ] && [ "$CLASS" != "$CLASS_ONLY" ]; then
    continue
  fi

  mkdir -p "$REPRO_ROOT/$CLASS"
  for SRC_VID in "$DATA_ROOT/$CLASS"/*; do
    [ -d "$SRC_VID" ] || continue
    VNAME="$(basename "$SRC_VID")"

    if [ -n "$START_FROM" ] && [[ "$VNAME" < "$START_FROM" ]]; then
      continue
    fi

    OUT="$REPRO_ROOT/$CLASS/$VNAME"

    if [ "$FORCE_REGEN_ANN" != "1" ] && [ "$(is_done "$OUT")" = "1" ]; then
      echo "SKIP DONE: $CLASS/$VNAME"
      continue
    fi

    if ! process_video "$CLASS" "$SRC_VID"; then
      echo "FAILED: $CLASS/$VNAME"
      echo "$CLASS/$VNAME" >> "$FAILED_FILE"
    fi
  done
done

echo "=============================="
echo "Batch finished."
echo "Failed list: $FAILED_FILE"
echo "=============================="
