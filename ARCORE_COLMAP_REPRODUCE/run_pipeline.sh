#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SRC="$ROOT/ARCORE_COLMAP_DATA"
DST="$SCRIPT_DIR"
SCRIPTS="$ROOT/DATASET GENERATION"
SAM_CKPT="$ROOT/sam2_checkpoints/sam2.1_hiera_large.pt"
SAM_CFG="configs/sam2.1/sam2.1_hiera_l.yaml"
REAL_HEIGHT=0.13

VIDEOS=(video_01_silver video_02_silver video_03_silver video_04_silver video_05_silver)
if [ -n "${VIDEOS_OVERRIDE:-}" ]; then
  IFS=',' read -r -a VIDEOS <<< "$VIDEOS_OVERRIDE"
fi

for V in "${VIDEOS[@]}"; do
  echo "=== Processing $V ==="
  src_rgb="$SRC/$V/rgb"
  out="$DST/$V"
  rotated="$out/frames_rotated"
  masks="$out/masks_sam2"
  colmap_dir="$out/colmap"
  txt_dir="$colmap_dir/txt"
  scaled="$out/scaled_masked"
  checks="$out/checks"

  mkdir -p "$rotated" "$masks" "$colmap_dir" "$txt_dir" "$scaled" "$checks"

  echo "[1/8] Rotate frames"
  if [ -n "${SKIP_ROTATION_VIDEOS:-}" ] && echo ",$SKIP_ROTATION_VIDEOS," | grep -q ",$V,"; then
    python3 "$SCRIPTS/rotate_frames_90cw.py" --in_dir "$src_rgb" --out_dir "$rotated" --ext png --no_rotate
  else
    python3 "$SCRIPTS/rotate_frames_90cw.py" --in_dir "$src_rgb" --out_dir "$rotated" --ext png
  fi
  mkdir -p "$checks/rotation"
  count=0
  for f in "$rotated"/frame_*.jpg; do
    cp "$f" "$checks/rotation/"
    count=$((count+1))
    if [ "$count" -ge 5 ]; then break; fi
  done

  echo "[2/8] COLMAP feature extraction (GPU)"
  QT_QPA_PLATFORM=offscreen colmap feature_extractor \
    --database_path "$colmap_dir/database.db" \
    --image_path "$rotated" \
    --ImageReader.camera_model SIMPLE_RADIAL \
    --ImageReader.single_camera 1 \
    --FeatureExtraction.use_gpu 1

  echo "[3/8] COLMAP sequential matching (GPU)"
  QT_QPA_PLATFORM=offscreen colmap sequential_matcher \
    --database_path "$colmap_dir/database.db" \
    --FeatureMatching.use_gpu 1 \
    --SequentialMatching.overlap 10

  echo "[4/8] COLMAP mapping"
  if [ -f "$colmap_dir/sparse" ]; then
    rm -f "$colmap_dir/sparse"
  fi
  mkdir -p "$colmap_dir/sparse"
  QT_QPA_PLATFORM=offscreen colmap mapper \
    --database_path "$colmap_dir/database.db" \
    --image_path "$rotated" \
    --output_path "$colmap_dir/sparse" \
    --Mapper.ba_global_max_num_iterations 1 \
    --Mapper.ba_global_max_refinements 0
  if [ ! -d "$colmap_dir/sparse/0" ]; then
    echo "ERROR: COLMAP mapping did not produce sparse/0 for $V"
    exit 1
  fi

  echo "[5/8] COLMAP model conversion"
  if ! QT_QPA_PLATFORM=offscreen colmap model_converter \
    --input_path "$colmap_dir/sparse/0" \
    --output_path "$txt_dir" \
    --output_type TXT; then
    echo "WARN: colmap model_converter crashed, using pycolmap fallback"
    python3 - <<PY
import pycolmap
from pathlib import Path

inp = Path("$colmap_dir") / "sparse" / "0"
out = Path("$txt_dir")
out.mkdir(parents=True, exist_ok=True)
recon = pycolmap.Reconstruction(str(inp))
recon.write_text(str(out))
print("pycolmap wrote TXT to", out)
PY
  fi
  if [ ! -f "$txt_dir/points3D.txt" ]; then
    echo "ERROR: COLMAP model conversion failed for $V (points3D.txt missing)"
    exit 1
  fi

  echo "[6/8] SAM2 masks (GPU)"
  python3 "$SCRIPTS/run_sam2_video_mask.py" \
    --frames_dir "$rotated" \
    --out_dir "$masks" \
    --ckpt "$SAM_CKPT" \
    --cfg "$SAM_CFG" \
    --device cuda \
    --safe_cuda \
    --min_mask_region_area 0

  mkdir -p "$checks/masks"
  python3 "$SCRIPTS/overlay_masks.py" \
    --frames_dir "$rotated" \
    --masks_dir "$masks" \
    --out_dir "$checks/masks" \
    --max_frames 50

  echo "[7/8] Masked filtering + scaling"
  python3 "$SCRIPTS/run_masked_filter_generic.py" \
    --colmap_txt "$txt_dir" \
    --masks_dir "$masks" \
    --database_db "$colmap_dir/database.db" \
    --out_dir "$scaled" \
    --real_height "$REAL_HEIGHT" \
    --skip_desc_db

  echo "[8/8] Clean PLY (DBSCAN) + Visual checks + Objectron annotations"
  python3 "$SCRIPTS/npz_viewer.py" \
    --input_ply "$scaled/points3D.ply" \
    --output_ply "$scaled/points3D_clean_eps0p0038.ply" \
    --eps 0.0038 \
    --min_points 5 \
    --keep_top_k 1 \
    --no_view
  mkdir -p "$checks/colmap_stride"
  python3 "$SCRIPTS/strides_viewer_colmap.py" \
    --point_cloud "$scaled/points3D.ply" \
    --cameras_txt "$scaled/cameras.txt" \
    --images_txt "$scaled/images.txt" \
    --image_folder "$rotated" \
    --output_dir "$checks/colmap_stride" \
    --stride 50 \
    --sort_by name \
    --point_size 1 \
    --overlay_alpha 0.5

  mkdir -p "$out/objectron_prep/overlays"
  python3 "$SCRIPTS/generate+objectron_dataset_colmap.py" \
    --point_cloud "$scaled/points3D_clean_eps0p0038.ply" \
    --cameras_txt "$scaled/cameras.txt" \
    --images_txt "$scaled/images.txt" \
    --image_folder "$rotated" \
    --overlay_dir "$out/objectron_prep/overlays" \
    --output_json "$out/objectron_prep/annotations.json" \
    --stride 1 \
    --start 0 \
    --sort_by name

  mkdir -p "$checks/final_overlays"
  count=0
  for f in "$out/objectron_prep/overlays"/frame_*.jpg; do
    cp "$f" "$checks/final_overlays/"
    count=$((count+1))
    if [ "$count" -ge 10 ]; then break; fi
  done

  echo "=== Done $V ==="
  echo

done
