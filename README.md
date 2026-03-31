# COLMAP_COLA Objectron Tracking Package

This repository contains the curated Objectron-related code, compact dataset metadata, selected checkpoints, and migration docs for the ARCore/COLMAP can-tracking pipeline.

## Start Here

If you are setting this up on a new computer, read these first:

- `SETUP.md`
- `checklist.md`
- `requirements-main.txt`
- `requirements-sam2.txt`
- `requirements-objectron-og.txt`
- `requirements-system.txt`

## Repository Contents

- `ARCORE_COLMAP_REPRODUCE/`
  - legacy single-stage Objectron-style training, inference, and evaluation
  - paper-aligned two-stage Objectron branch
  - repo-relative launcher scripts
  - compact annotation JSONs for `Blue`, `Red`, and `Silver`
  - selected checkpoints and evaluation outputs
- `DATASET GENERATION/`
  - COLMAP and SAM2 based utilities for generating Objectron-style annotations
- `OBJECTRON_OG/`
  - original Objectron code used for paper-aligned geometry and evaluation reference
- `Objectron.pdf`
  - Objectron paper reference

## Intentionally Excluded

The original workspace is much larger than what is practical to push to GitHub. The repository intentionally excludes most heavy local artifacts such as:

- raw frames in `frames_rotated/` and `rgb/`
- large mask folders and overlays
- COLMAP databases and sparse reconstructions
- virtual environments
- logs and temporary videos

For full migration, clone the repo and then copy the large data from the old machine as described in `SETUP.md` and `checklist.md`.

## Important Model Artifacts

- baseline legacy checkpoint: `ARCORE_COLMAP_REPRODUCE/posneg_run_001/best_model.pt`
- improved legacy checkpoint: `ARCORE_COLMAP_REPRODUCE/fresh_from_run001_20260330_155107/best_model.pt`

## Environment Split

Use separate environments:

- main env: PyTorch training, inference, evaluation, and optional SAM2 dataset generation
- `OBJECTRON_OG` env: TensorFlow-only original Objectron code with `numpy<2`

This separation is important for stability.

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
