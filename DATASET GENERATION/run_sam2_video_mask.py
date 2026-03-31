import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import os

from sam2.build_sam import build_sam2_video_predictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.utils import amg as sam2_amg

def list_frames(frames_dir):
    exts = {'.jpg', '.jpeg', '.png'}
    frames = [p for p in frames_dir.iterdir() if p.suffix.lower() in exts]
    return sorted(frames)



def _clone_list_shallow(items):
    out = []
    for it in items:
        if isinstance(it, dict):
            out.append(dict(it))
        elif isinstance(it, list):
            out.append(list(it))
        else:
            out.append(it)
    return out


def _safe_cat(self, new_stats):
    for k, v in new_stats.items():
        if k not in self._stats or self._stats[k] is None:
            if isinstance(v, torch.Tensor):
                self._stats[k] = v.clone()
            elif isinstance(v, np.ndarray):
                self._stats[k] = v.copy()
            elif isinstance(v, list):
                self._stats[k] = _clone_list_shallow(v)
            else:
                self._stats[k] = v
        elif isinstance(v, torch.Tensor):
            self._stats[k] = torch.cat([self._stats[k], v], dim=0)
        elif isinstance(v, np.ndarray):
            self._stats[k] = np.concatenate([self._stats[k], v], axis=0)
        elif isinstance(v, list):
            self._stats[k] = self._stats[k] + _clone_list_shallow(v)
        else:
            raise TypeError(f"MaskData key {k} has an unsupported type {type(v)}.")


# Patch SAM2 MaskData.cat to avoid copy.deepcopy issues in some environments.
sam2_amg.MaskData.cat = _safe_cat


def _zoom_params(h, w, zoom):
    if zoom <= 1.0:
        return 0, 0, w, h
    crop_w = max(8, int(round(w / zoom)))
    crop_h = max(8, int(round(h / zoom)))
    x0 = (w - crop_w) // 2
    y0 = (h - crop_h) // 2
    return x0, y0, crop_w, crop_h


def _zoom_image_center(img_np, zoom):
    h, w = img_np.shape[:2]
    x0, y0, crop_w, crop_h = _zoom_params(h, w, zoom)
    if zoom <= 1.0:
        return img_np
    crop = img_np[y0 : y0 + crop_h, x0 : x0 + crop_w]
    zoomed = np.array(Image.fromarray(crop).resize((w, h), Image.BILINEAR))
    return zoomed


def _unzoom_mask_center(mask_zoomed, out_shape_hw, zoom):
    h, w = out_shape_hw
    x0, y0, crop_w, crop_h = _zoom_params(h, w, zoom)
    if zoom <= 1.0:
        return mask_zoomed.astype(np.uint8)
    small = np.array(
        Image.fromarray(mask_zoomed.astype(np.uint8) * 255).resize((crop_w, crop_h), Image.NEAREST)
    )
    out = np.zeros((h, w), dtype=np.uint8)
    out[y0 : y0 + crop_h, x0 : x0 + crop_w] = (small > 0).astype(np.uint8)
    return out


def _zoom_mask_center(mask_orig, out_shape_hw, zoom):
    h, w = out_shape_hw
    x0, y0, crop_w, crop_h = _zoom_params(h, w, zoom)
    if zoom <= 1.0:
        return mask_orig.astype(np.uint8)
    crop = mask_orig[y0 : y0 + crop_h, x0 : x0 + crop_w]
    big = np.array(
        Image.fromarray(crop.astype(np.uint8) * 255).resize((w, h), Image.NEAREST)
    )
    return (big > 0).astype(np.uint8)


def _color_score(img_np, mask, target_color):
    if target_color == "none":
        return 0.0
    pix = img_np[mask > 0]
    if pix.size == 0:
        return 0.0
    r = pix[:, 0].astype(np.int32)
    g = pix[:, 1].astype(np.int32)
    b = pix[:, 2].astype(np.int32)
    if target_color == "blue":
        sel = (b > g + 16) & (b > r + 16)
        return float(np.mean(sel))
    if target_color == "red":
        sel = (r > g + 16) & (r > b + 16)
        return float(np.mean(sel))
    if target_color == "silver":
        near_gray = (np.abs(r - g) < 22) & (np.abs(r - b) < 22) & (np.abs(g - b) < 22)
        bright = ((r + g + b) / 3.0) > 85.0
        return float(np.mean(near_gray & bright))
    return 0.0


def select_can_mask(img_np, masks, target_color="none"):
    # Generic can selector: center + slender aspect + reasonable area + solidity.
    H, W = img_np.shape[:2]
    center = np.array([W / 2, H / 2], dtype=np.float32)
    best = None
    best_score = -1e9

    for m in masks:
        mask = m['segmentation']
        area = mask.sum()
        area_frac = area / float(H * W)
        if area_frac < 0.002 or area_frac > 0.35:
            continue

        ys, xs = np.where(mask)
        if len(xs) == 0:
            continue

        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        w = float(x1 - x0 + 1)
        h = float(y1 - y0 + 1)
        if w <= 1 or h <= 1:
            continue

        centroid = np.array([xs.mean(), ys.mean()], dtype=np.float32)
        dist = np.linalg.norm((centroid - center) / np.array([W, H], dtype=np.float32))
        center_score = 1.0 - dist

        # Orientation-agnostic slenderness: vertical cans (h>w) and horizontal cans (w>h)
        # should both score high.
        aspect = h / w
        elong = max(aspect, 1.0 / max(aspect, 1e-6))
        elong_score = 1.0 - min(abs(elong - 2.2) / 2.2, 1.0)

        # Prefer compact, solid masks.
        bbox_area = w * h
        solidity = float(area) / (bbox_area + 1e-6)

        # Target area (roughly can size in frame)
        target = 0.06
        area_score = 1.0 - min(abs(area_frac - target) / target, 1.0)

        color_score = _color_score(img_np, mask, target_color)
        score = 2.0 * center_score + 2.0 * elong_score + 1.5 * solidity + 1.0 * area_score + 2.5 * color_score
        if score > best_score:
            best_score = score
            best = mask

    if best is None:
        best = max(masks, key=lambda m: m['segmentation'].sum())['segmentation']
    return best


def make_numeric_symlinks(frames, tmp_dir: Path, center_zoom: float = 1.0):
    tmp_dir.mkdir(parents=True, exist_ok=True)
    mapping = []  # (idx, orig_path, tmp_path)
    for i, p in enumerate(frames):
        tmp_name = f"{i:06d}.jpg"
        tmp_path = tmp_dir / tmp_name
        if center_zoom > 1.0 and tmp_path.is_symlink():
            tmp_path.unlink()
        if not tmp_path.exists():
            if center_zoom > 1.0:
                arr = np.array(Image.open(p).convert("RGB"))
                zoomed = _zoom_image_center(arr, center_zoom)
                Image.fromarray(zoomed).save(tmp_path, format="JPEG", quality=95)
            else:
                os.symlink(p, tmp_path)
        # Preserve the original filename when input frames are symlinks.
        # This keeps mask names aligned with COLMAP image names.
        orig_path = p.resolve() if p.is_symlink() else p
        mapping.append((i, orig_path, tmp_path))
    return mapping

def _mask_out_path(mapping, out_dir: Path, frame_idx: int) -> Path:
    # Keep mask names aligned with original frame stems used by COLMAP.
    return out_dir / (mapping[frame_idx][1].stem + '.png')


def _first_missing_mask_idx(mapping, out_dir: Path) -> int:
    n = len(mapping)
    i = 0
    while i < n and _mask_out_path(mapping, out_dir, i).exists():
        i += 1
    return i


def _read_binary_mask(mask_path: Path) -> np.ndarray:
    m = np.array(Image.open(mask_path).convert('L'))
    return (m > 0).astype(np.uint8)


def _find_latest_nonempty_seed(mapping, out_dir: Path, start_idx: int):
    for idx in range(start_idx - 1, -1, -1):
        p = _mask_out_path(mapping, out_dir, idx)
        if not p.exists():
            continue
        m = _read_binary_mask(p)
        if m.sum() > 0:
            return idx, m
    return None, None



def _repair_module_registries(root_module):
    """Repair rare runtime corruption where _modules becomes an iterator."""
    stack = [root_module]
    while stack:
        mod = stack.pop()
        mods = getattr(mod, "_modules", None)
        if not isinstance(mods, dict):
            try:
                mod._modules = dict(mods)
            except Exception:
                mod._modules = {}
            mods = mod._modules
        for child in mods.values():
            if isinstance(child, torch.nn.Module):
                stack.append(child)


def _set_eval_safely(module, name: str):
    try:
        module.eval()
    except AttributeError as e:
        if "dict_itemiterator" not in str(e):
            raise
        print(f"Warning: repairing module registry for {name}: {e}")
        _repair_module_registries(module)
        module.eval()


def _sanitize_torch_refs():
    """Repair rare torch module reference corruption across internal submodules."""
    import torch.autograd.grad_mode as grad_mode
    import torch.nn.init as nn_init

    if not hasattr(grad_mode.torch, "is_grad_enabled"):
        grad_mode.torch = torch
    if not hasattr(nn_init.torch, "no_grad"):
        nn_init.torch = torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames_dir', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--cfg', default='configs/sam2.1/sam2.1_hiera_l.yaml')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--points_per_side', type=int, default=64)
    ap.add_argument('--pred_iou_thresh', type=float, default=0.88)
    ap.add_argument('--stability_score_thresh', type=float, default=0.96)
    ap.add_argument('--crop_n_layers', type=int, default=1)
    ap.add_argument('--min_mask_region_area', type=int, default=500)
    ap.add_argument('--safe_cuda', action='store_true', help='Disable fast SDPA kernels for stability')
    ap.add_argument('--max_track_frames', type=int, default=0, help='If >0, propagate at most this many frames per run')
    ap.add_argument('--target_color', choices=['none', 'blue', 'red', 'silver'], default='none')
    ap.add_argument('--center_zoom', type=float, default=1.0, help='If >1, run SAM2 on center-zoomed frames and unzoom masks back')
    args = ap.parse_args()

    frames_dir = Path(args.frames_dir)
    frames = list_frames(frames_dir)
    if not frames:
        raise RuntimeError('No frames found')
    center_zoom = max(1.0, float(args.center_zoom))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = Path(args.ckpt)
    cfg_name = args.cfg if args.cfg else "configs/sam2.1/sam2.1_hiera_l.yaml"
    cfg_path = Path(cfg_name)
    if cfg_path.is_file():
        parts = cfg_path.parts
        if "configs" in parts:
            idx = parts.index("configs")
            cfg_name = "/".join(parts[idx:])
        else:
            raise RuntimeError(
                f"SAM2 config path must be under a 'configs' directory. Got: {cfg_path}. "
                "Use configs/sam2.1/sam2.1_hiera_l.yaml"
            )


    # Create numeric symlink dir for SAM2 video loader
    tmp_dir = frames_dir.parent / '_sam2_numeric_frames'
    mapping = make_numeric_symlinks(frames, tmp_dir, center_zoom=center_zoom)

    first_frame = frames[0]

    print('Building SAM2 video model/predictor...')
    if args.device.startswith("cuda") and args.safe_cuda:
        # Force safe CUDA settings to avoid kernel crashes on some setups.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(False)
        if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            torch.backends.cuda.enable_mem_efficient_sdp(False)
        if hasattr(torch.backends.cuda, "enable_math_sdp"):
            torch.backends.cuda.enable_math_sdp(True)
    _sanitize_torch_refs()
    video_predictor = build_sam2_video_predictor(cfg_name, str(ckpt), device=args.device)
    _set_eval_safely(video_predictor, "video predictor")

    start_idx = _first_missing_mask_idx(mapping, out_dir)
    if start_idx >= len(mapping):
        print('All masks already present. Nothing to do.')
        print('Numeric symlink dir:', tmp_dir)
        return

    if start_idx == 0:
        img = Image.open(first_frame).convert('RGB')
        img_np = np.array(img)
        img_for_sam = _zoom_image_center(img_np, center_zoom)

        mask_gen = SAM2AutomaticMaskGenerator(
            video_predictor,
            points_per_side=args.points_per_side,
            pred_iou_thresh=args.pred_iou_thresh,
            stability_score_thresh=args.stability_score_thresh,
            box_nms_thresh=0.7,
            crop_n_layers=args.crop_n_layers,
            crop_nms_thresh=0.7,
            crop_overlap_ratio=0.4,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=args.min_mask_region_area,
            output_mode='binary_mask',
            use_m2m=True,
        )

        print('Generating masks on first frame...')
        masks = mask_gen.generate(img_for_sam)
        if not masks:
            raise RuntimeError('No masks generated on first frame')

        seed_mask_pred = select_can_mask(img_for_sam, masks, target_color=args.target_color).astype(np.uint8)
        seed_mask = _unzoom_mask_center(seed_mask_pred, img_np.shape[:2], center_zoom)
        seed_idx = 0
        first_mask_path = _mask_out_path(mapping, out_dir, 0)
        Image.fromarray(seed_mask * 255).save(first_mask_path)
        print('Saved first mask:', first_mask_path)
    else:
        seed_idx, seed_mask = _find_latest_nonempty_seed(mapping, out_dir, start_idx)
        if seed_idx is None:
            print('Warning: no non-empty previous mask found; restarting from frame 0')
            start_idx = 0
            img = Image.open(first_frame).convert('RGB')
            img_np = np.array(img)
            img_for_sam = _zoom_image_center(img_np, center_zoom)

            mask_gen = SAM2AutomaticMaskGenerator(
                video_predictor,
                points_per_side=args.points_per_side,
                pred_iou_thresh=args.pred_iou_thresh,
                stability_score_thresh=args.stability_score_thresh,
                box_nms_thresh=0.7,
                crop_n_layers=args.crop_n_layers,
                crop_nms_thresh=0.7,
                crop_overlap_ratio=0.4,
                crop_n_points_downscale_factor=2,
                min_mask_region_area=args.min_mask_region_area,
                output_mode='binary_mask',
                use_m2m=True,
            )

            print('Generating masks on first frame (resume fallback)...')
            masks = mask_gen.generate(img_for_sam)
            if not masks:
                raise RuntimeError('No masks generated on first frame during fallback')
            seed_mask_pred = select_can_mask(img_for_sam, masks, target_color=args.target_color).astype(np.uint8)
            seed_mask = _unzoom_mask_center(seed_mask_pred, img_np.shape[:2], center_zoom)
            seed_idx = 0
            first_mask_path = _mask_out_path(mapping, out_dir, 0)
            Image.fromarray(seed_mask * 255).save(first_mask_path)
            print('Saved first mask:', first_mask_path)
        else:
            print(f'Resuming from frame {start_idx} using seed frame {seed_idx}')
            seed_mask_pred = _zoom_mask_center(seed_mask, seed_mask.shape[:2], center_zoom)

    if start_idx == 0:
        # Ensure predictor receives mask in its own frame coordinates.
        seed_mask_pred = _zoom_mask_center(seed_mask, seed_mask.shape[:2], center_zoom)

    print('Using same predictor for video propagation...')
    print('Initializing video state...')
    state = video_predictor.init_state(str(tmp_dir))
    video_predictor.add_new_mask(state, frame_idx=seed_idx, obj_id=1, mask=seed_mask_pred)

    max_track = args.max_track_frames if args.max_track_frames and args.max_track_frames > 0 else None
    if max_track is not None:
        print(f'Propagating masks through video in chunk mode: max_track_frames={max_track}')
    else:
        print('Propagating masks through video...')
    for frame_idx, obj_ids, masks_out in video_predictor.propagate_in_video(
        state, start_frame_idx=seed_idx, max_frame_num_to_track=max_track
    ):
        if frame_idx < start_idx:
            continue
        for i, obj_id in enumerate(obj_ids):
            if obj_id != 1:
                continue
            mask = masks_out[i].cpu().numpy()
            # Ensure 2D HxW
            if mask.ndim == 3 and mask.shape[0] == 1:
                mask = mask[0]
            mask = (mask > 0.0).astype(np.uint8)
            mask = _unzoom_mask_center(mask, (mask.shape[0], mask.shape[1]), center_zoom)
            mask = (mask > 0).astype(np.uint8) * 255
            out_path = _mask_out_path(mapping, out_dir, frame_idx)
            Image.fromarray(mask).save(out_path)
        if frame_idx % 50 == 0:
            print('Saved mask for frame', frame_idx)

    print('Done. Masks saved to', out_dir)
    print('Numeric symlink dir:', tmp_dir)


if __name__ == '__main__':
    main()
