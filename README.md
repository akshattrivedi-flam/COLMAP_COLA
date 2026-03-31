# COLMAP_COLA Objectron Tracking Package

This repository contains the curated Objectron-related code, compact dataset metadata, and model artifacts used for ARCore/COLMAP can tracking work.

## Included

- `ARCORE_COLMAP_REPRODUCE/`
  - legacy single-stage training, inference, and evaluation scripts
  - paper-aligned two-stage Objectron training and inference scripts
  - portable launcher scripts with repo-relative paths
  - compact Objectron annotation JSONs for the `Blue`, `Red`, and `Silver` videos
  - key checkpoints and reports from:
    - `posneg_run_001`
    - `fresh_from_run001_20260330_155107`
- `OBJECTRON_OG/`
  - official Objectron code used by the evaluator for Objectron-style 3D IoU and geometry handling
- `DATASET GENERATION/`
  - compact dataset-generation scripts used to produce Objectron-style annotations from COLMAP/SAM2 pipelines
- `Objectron.pdf`
  - Objectron paper reference used to align the tracking work

## Intentionally Excluded

The original local workspace contains raw image frames, masks, COLMAP reconstructions, virtual environments, logs, videos, and other large artifacts that are too large for a careful GitHub push.

Excluded categories include:

- raw video frames such as `frames_rotated/` and `rgb/`
- COLMAP databases and sparse reconstructions
- mask folders and rendered overlays
- local virtual environments
- large logs and comparison videos

The compact annotation JSONs are included because they are the portable Objectron training metadata. The full raw image dataset remains outside this repo.

## Important Model Artifacts

- baseline checkpoint: `ARCORE_COLMAP_REPRODUCE/posneg_run_001/best_model.pt`
- improved legacy checkpoint: `ARCORE_COLMAP_REPRODUCE/fresh_from_run001_20260330_155107/best_model.pt`

## Notes

- `eval_objectron_tracking.py` now supports both legacy single-stage checkpoints and the newer paper two-stage checkpoint format.
- Relative paths inside run manifests are resolved from the manifest file location, so pushed run metadata is portable.
- The paper-style two-stage branch is implemented in:
  - `ARCORE_COLMAP_REPRODUCE/objectron_paper_twostage.py`
  - `ARCORE_COLMAP_REPRODUCE/train_objectron_paper_twostage.py`
  - `ARCORE_COLMAP_REPRODUCE/infer_objectron_paper_twostage.py`

## Quick Commands

Legacy curated training:

```bash
bash ARCORE_COLMAP_REPRODUCE/run_train_posneg_curated.sh
```

Paper two-stage training:

```bash
bash ARCORE_COLMAP_REPRODUCE/run_train_objectron_paper_twostage.sh
```

Evaluate a checkpoint:

```bash
python3 ARCORE_COLMAP_REPRODUCE/eval_objectron_tracking.py \
  --checkpoint ARCORE_COLMAP_REPRODUCE/fresh_from_run001_20260330_155107/best_model.pt \
  --video_dirs ARCORE_COLMAP_REPRODUCE/Blue/video_08_blue \
  --out_dir /tmp/objectron_eval
```
