import os, sys, json
from pathlib import Path
import numpy as np
from PIL import Image
import torch

from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

root = Path('/home/user/Desktop/COLMAP_COLA')
frames_dir = root/'video_02_red'/'raw_frames'
mask_dir = root/'video_02_red'/'masks_sam2'
mask_dir.mkdir(parents=True, exist_ok=True)

ckpt = root/'sam2_checkpoints'/'sam2.1_hiera_large.pt'
# config is resolved from sam2 package
cfg_name = 'configs/sam2.1/sam2.1_hiera_l.yaml'

if not ckpt.exists():
    raise FileNotFoundError(ckpt)

# select first frame
frames = sorted(frames_dir.glob('frame_*.jpg'))
if not frames:
    raise RuntimeError('No frames found')
first_frame = frames[0]

print('Using first frame:', first_frame)

# Load first frame
img = Image.open(first_frame).convert('RGB')
img_np = np.array(img)
H, W = img_np.shape[:2]

# Build model for automatic mask generation
print('Building SAM2 model...')
model = build_sam2(cfg_name, str(ckpt), device='cuda')
model.eval()

# Automatic mask generator with high quality settings
mask_gen = SAM2AutomaticMaskGenerator(
    model,
    points_per_side=64,
    pred_iou_thresh=0.88,
    stability_score_thresh=0.96,
    box_nms_thresh=0.7,
    crop_n_layers=1,
    crop_nms_thresh=0.7,
    crop_overlap_ratio=0.4,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=500,
    output_mode='binary_mask',
    use_m2m=True,
)

print('Generating masks on first frame...')
masks = mask_gen.generate(img_np)
print('Masks generated:', len(masks))
if not masks:
    raise RuntimeError('No masks generated on first frame')

# Score masks for likely can: high red ratio, near center, medium size
center = np.array([W/2, H/2])

best = None
best_score = -1e9

for m in masks:
    mask = m['segmentation']
    area = mask.sum()
    area_frac = area / (H*W)
    if area_frac < 0.002 or area_frac > 0.3:
        continue

    ys, xs = np.where(mask)
    if len(xs) == 0:
        continue
    centroid = np.array([xs.mean(), ys.mean()])
    dist = np.linalg.norm((centroid - center) / np.array([W, H]))
    center_score = 1.0 - dist

    # color score
    masked_pixels = img_np[mask]
    if masked_pixels.size == 0:
        continue
    r = masked_pixels[:,0].mean()
    g = masked_pixels[:,1].mean()
    b = masked_pixels[:,2].mean()
    red_ratio = r / (g + b + 1e-6)

    # prefer medium area (around 7%)
    target = 0.07
    area_score = 1.0 - abs(area_frac - target) / target

    score = 2.0*red_ratio + 2.0*center_score + 1.0*area_score
    if score > best_score:
        best_score = score
        best = mask

if best is None:
    # fallback: largest mask
    best = max(masks, key=lambda m: m['segmentation'].sum())['segmentation']
    best_score = None

print('Selected mask score:', best_score)

# Save initial mask
first_mask_path = mask_dir / (first_frame.stem + '.png')
Image.fromarray((best.astype(np.uint8)*255)).save(first_mask_path)
print('Saved first mask:', first_mask_path)

# Build video predictor and propagate
print('Building video predictor...')
video_predictor = build_sam2_video_predictor(cfg_name, str(ckpt), device='cuda')
video_predictor.eval()

print('Initializing video state...')
state = video_predictor.init_state(str(frames_dir))

# add mask for frame 0, object id = 1
video_predictor.add_new_mask(state, frame_idx=0, obj_id=1, mask=best)

print('Propagating masks through video...')
for frame_idx, obj_ids, masks_out in video_predictor.propagate_in_video(state):
    # masks_out: [num_obj, H, W]
    for i, obj_id in enumerate(obj_ids):
        if obj_id != 1:
            continue
        mask = masks_out[i].cpu().numpy()
        mask = (mask > 0.0).astype(np.uint8)*255
        out_path = mask_dir / (frames[frame_idx].stem + '.png')
        Image.fromarray(mask).save(out_path)
    if frame_idx % 50 == 0:
        print('Saved mask for frame', frame_idx)

print('Done. Masks saved to', mask_dir)
