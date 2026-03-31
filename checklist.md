# Migration Checklist

Use this checklist on the new machine after cloning the repo.

## Machine

- [ ] Ubuntu `22.04` or `24.04`
- [ ] NVIDIA driver installed and working
- [ ] CUDA-capable GPU available
- [ ] At least `32 GB` RAM
- [ ] At least `250 GB` free disk

## Base Software

- [ ] `git` installed
- [ ] `ffmpeg` installed
- [ ] `python3.10` installed
- [ ] `python3.10-venv` installed
- [ ] `nvidia-smi` works
- [ ] `colmap` installed and available on `PATH`
- [ ] `colmap -h` runs successfully

## Repository

- [ ] `git clone https://github.com/akshattrivedi-flam/COLMAP_COLA.git`
- [ ] repo opens correctly
- [ ] `SETUP.md` reviewed

## Main Environment

- [ ] `.venv_objectron` created
- [ ] `torch==2.7.1` and `torchvision==0.22.1` installed from the CUDA wheel index
- [ ] `pip install -r requirements-main.txt` completed
- [ ] `pip install -r requirements-sam2.txt` completed if full regeneration is needed
- [ ] `python -c "import torch; print(torch.cuda.is_available())"` returns `True`
- [ ] `python -c "import sam2"` works if regeneration is needed

## OBJECTRON_OG Environment

- [ ] `.venv_objectron_og` created
- [ ] `pip install -r requirements-objectron-og.txt` completed
- [ ] `PYTHONPATH` includes `OBJECTRON_OG/Objectron` when using that env
- [ ] `python -c "import tensorflow as tf"` works in that env

## Data Transfer From Old Machine

- [ ] `ARCORE_COLMAP_REPRODUCE/Blue/` copied
- [ ] `ARCORE_COLMAP_REPRODUCE/Red/` copied
- [ ] `ARCORE_COLMAP_REPRODUCE/Silver/` copied
- [ ] `ARCORE_COLMAP_REPRODUCE/posneg_run_001/` copied
- [ ] `ARCORE_COLMAP_REPRODUCE/fresh_from_run001_20260330_155107/` copied
- [ ] `sam2_checkpoints/sam2.1_hiera_large.pt` copied
- [ ] `ARCORE_COLMAP_DATA/` copied if full raw regeneration is needed

## Folder Checks

- [ ] `ARCORE_COLMAP_REPRODUCE/Blue/video_01_blue/objectron_prep/annotations.json` exists
- [ ] one or more video folders contain `frames_rotated/` or `rgb/`
- [ ] `ARCORE_COLMAP_REPRODUCE/posneg_run_001/best_model.pt` exists
- [ ] `ARCORE_COLMAP_REPRODUCE/fresh_from_run001_20260330_155107/best_model.pt` exists
- [ ] `sam2_checkpoints/sam2.1_hiera_large.pt` exists if regeneration is needed

## Validation

- [ ] `python -c "import torch, torchvision, cv2, timm, pycolmap, open3d"` works in `.venv_objectron`
- [ ] `ffmpeg -version | head -n 1` works
- [ ] `colmap -h | head -n 1` works
- [ ] small evaluation run succeeds

Suggested smoke test:

```bash
source .venv_objectron/bin/activate

python ARCORE_COLMAP_REPRODUCE/eval_objectron_tracking.py \
  --checkpoint ARCORE_COLMAP_REPRODUCE/fresh_from_run001_20260330_155107/best_model.pt \
  --video_dirs ARCORE_COLMAP_REPRODUCE/Blue/video_08_blue \
  --out_dir /tmp/objectron_eval_smoke
```

## Training Readiness

- [ ] `bash ARCORE_COLMAP_REPRODUCE/run_train_posneg_curated.sh` can see Blue, Red, and Silver videos
- [ ] `bash ARCORE_COLMAP_REPRODUCE/run_train_objectron_paper_twostage.sh` can see Blue, Red, and Silver videos
- [ ] output directories can be created

## Full Regeneration Readiness

- [ ] `ARCORE_COLMAP_DATA/` exists
- [ ] `SAM-2` import works
- [ ] `sam2_checkpoints/sam2.1_hiera_large.pt` exists
- [ ] `colmap` is installed
- [ ] GPU-enabled COLMAP is available if you want the existing scripts unchanged
