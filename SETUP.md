# Setup Guide

This repository is portable now, but a full working migration still has two parts:

1. clone the GitHub repo
2. copy the large local data that was intentionally not pushed

The code paths are repo-relative, so you do not need to recreate the old `/home/user/Desktop/...` layout.

## Recommended Machine

- Ubuntu `22.04` or `24.04`
- Python `3.10`
- NVIDIA GPU with CUDA support
- `12 GB` VRAM minimum, `24 GB` recommended
- `32 GB` RAM minimum, `64 GB` recommended
- `250 GB` free disk recommended

## Repository Layout

- `ARCORE_COLMAP_REPRODUCE/`
  - main training, inference, and evaluation code
- `DATASET GENERATION/`
  - COLMAP and SAM2 based dataset-generation utilities
- `OBJECTRON_OG/`
  - original Objectron TensorFlow code
- `Objectron.pdf`
  - paper reference

## What GitHub Contains

The GitHub repo already includes:

- training and inference scripts
- evaluation code
- compact annotation JSON files
- selected checkpoints and evaluation artifacts
- the paper-aligned two-stage branch

## What You Still Need To Copy From The Old Machine

For full training and video inference:

- `ARCORE_COLMAP_REPRODUCE/Blue/`
- `ARCORE_COLMAP_REPRODUCE/Red/`
- `ARCORE_COLMAP_REPRODUCE/Silver/`

Each video directory should contain:

- `objectron_prep/annotations.json`
- image folders such as `frames_rotated/` or `rgb/`

For full dataset regeneration from raw captures:

- `ARCORE_COLMAP_DATA/`
- `sam2_checkpoints/sam2.1_hiera_large.pt`

For keeping past training outputs intact:

- `ARCORE_COLMAP_REPRODUCE/posneg_run_001/`
- `ARCORE_COLMAP_REPRODUCE/fresh_from_run001_20260330_155107/`

## 1. Clone The Repo

```bash
git clone https://github.com/akshattrivedi-flam/COLMAP_COLA.git
cd COLMAP_COLA
```

## 2. Install System Packages

```bash
sudo apt update
sudo apt install -y $(tr '\n' ' ' < requirements-system.txt)
```

You also need the `colmap` CLI on your `PATH`.

Important:

- the current pipeline scripts call COLMAP with GPU flags
- for best reproducibility, install a CUDA-enabled COLMAP build
- a CPU-only COLMAP build will require script changes before full regeneration

## 3. Create The Main Environment

This environment is for:

- legacy single-stage training
- paper two-stage training
- inference
- evaluation
- optional SAM2 dataset generation

```bash
python3.10 -m venv .venv_objectron
source .venv_objectron/bin/activate

pip install --upgrade pip setuptools wheel

pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements-main.txt
```

If you want the full SAM2-based regeneration pipeline in the same environment:

```bash
pip install -r requirements-sam2.txt
```

If that package install fails on your new machine, use the upstream fallback:

```bash
pip install git+https://github.com/facebookresearch/sam2.git
```

## 4. Create The Separate OBJECTRON_OG Environment

Keep this separate from the main environment.

Reason:

- `OBJECTRON_OG` uses TensorFlow
- TensorFlow in that code path is not safe with `numpy 2.x`
- the main pipeline uses `numpy 2.x`

```bash
python3.10 -m venv .venv_objectron_og
source .venv_objectron_og/bin/activate

pip install --upgrade pip setuptools wheel
pip install -r requirements-objectron-og.txt

export PYTHONPATH="$PWD/OBJECTRON_OG/Objectron:$PYTHONPATH"
```

If you want that `PYTHONPATH` every time, add it to your shell startup file or activate it manually when using `OBJECTRON_OG`.

## 5. Copy The Missing Data From The Old Machine

Example `rsync` commands:

```bash
rsync -avh old_machine:/path/to/COLMAP_COLA/ARCORE_COLMAP_REPRODUCE/Blue/ ARCORE_COLMAP_REPRODUCE/Blue/
rsync -avh old_machine:/path/to/COLMAP_COLA/ARCORE_COLMAP_REPRODUCE/Red/ ARCORE_COLMAP_REPRODUCE/Red/
rsync -avh old_machine:/path/to/COLMAP_COLA/ARCORE_COLMAP_REPRODUCE/Silver/ ARCORE_COLMAP_REPRODUCE/Silver/
rsync -avh old_machine:/path/to/COLMAP_COLA/ARCORE_COLMAP_REPRODUCE/posneg_run_001/ ARCORE_COLMAP_REPRODUCE/posneg_run_001/
rsync -avh old_machine:/path/to/COLMAP_COLA/ARCORE_COLMAP_REPRODUCE/fresh_from_run001_20260330_155107/ ARCORE_COLMAP_REPRODUCE/fresh_from_run001_20260330_155107/
rsync -avh old_machine:/path/to/COLMAP_COLA/sam2_checkpoints/ sam2_checkpoints/
```

If you want the full raw-regeneration path too:

```bash
rsync -avh old_machine:/path/to/COLMAP_COLA/ARCORE_COLMAP_DATA/ ARCORE_COLMAP_DATA/
```

## 6. Sanity Checks

Main environment:

```bash
source .venv_objectron/bin/activate

nvidia-smi
python -c "import torch, torchvision, cv2, timm, pycolmap, open3d; print(torch.__version__, torchvision.__version__, cv2.__version__)"
python -c "import torch; print('cuda:', torch.cuda.is_available())"
python -c "import sam2; print('sam2 ok')"
colmap -h | head -n 1
ffmpeg -version | head -n 1
```

OBJECTRON_OG environment:

```bash
source .venv_objectron_og/bin/activate
export PYTHONPATH="$PWD/OBJECTRON_OG/Objectron:$PYTHONPATH"
python -c "import tensorflow as tf, frozendict, google.protobuf; print(tf.__version__)"
```

## 7. First Commands To Run

### Evaluate an existing checkpoint

```bash
source .venv_objectron/bin/activate

python ARCORE_COLMAP_REPRODUCE/eval_objectron_tracking.py \
  --checkpoint ARCORE_COLMAP_REPRODUCE/fresh_from_run001_20260330_155107/best_model.pt \
  --video_dirs ARCORE_COLMAP_REPRODUCE/Blue/video_08_blue \
  --out_dir /tmp/objectron_eval
```

### Legacy curated training

```bash
source .venv_objectron/bin/activate
bash ARCORE_COLMAP_REPRODUCE/run_train_posneg_curated.sh
```

### Paper two-stage training

```bash
source .venv_objectron/bin/activate
bash ARCORE_COLMAP_REPRODUCE/run_train_objectron_paper_twostage.sh
```

## 8. Full Regeneration Flow

This path needs:

- `ARCORE_COLMAP_DATA/`
- `sam2_checkpoints/sam2.1_hiera_large.pt`
- `SAM-2`
- `colmap`
- `ffmpeg`

Entry points:

- `ARCORE_COLMAP_REPRODUCE/run_pipeline.sh`
- `ARCORE_COLMAP_REPRODUCE/run_resume_fastseq.sh`
- `ARCORE_COLMAP_REPRODUCE/regenerate_blue_refined.sh`

## Notes

- `eval_objectron_tracking.py` supports both the legacy checkpoint format and the paper two-stage checkpoint format.
- Relative paths in the pushed run manifests are portable now.
- For strict alignment with the Objectron paper, keep the `OBJECTRON_OG` environment separate and use it only when you need the original code path.
