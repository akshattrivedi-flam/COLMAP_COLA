import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


OBJECTRON_LOCAL_TEMPLATE = np.array(
    [
        [0.0, 0.5, 0.0],   # 0 center
        [-0.5, 0.0, +0.5], # 1 front-bottom-left
        [+0.5, 0.0, +0.5], # 2 front-bottom-right
        [+0.5, 1.0, +0.5], # 3 front-top-right
        [-0.5, 1.0, +0.5], # 4 front-top-left
        [-0.5, 0.0, -0.5], # 5 back-bottom-left
        [+0.5, 0.0, -0.5], # 6 back-bottom-right
        [+0.5, 1.0, -0.5], # 7 back-top-right
        [-0.5, 1.0, -0.5], # 8 back-top-left
    ],
    dtype=np.float32,
)

CUBOID_EDGES = [
    (1, 2), (2, 3), (3, 4), (4, 1),
    (5, 6), (6, 7), (7, 8), (8, 5),
    (1, 5), (2, 6), (3, 7), (4, 8),
]

SYM_PERMS_YAW4 = np.array(
    [
        [0, 1, 2, 3, 4, 5, 6, 7, 8],  # identity
        [0, 2, 6, 7, 3, 1, 5, 8, 4],  # +90 deg yaw
        [0, 6, 5, 8, 7, 2, 1, 4, 3],  # +180 deg yaw
        [0, 5, 1, 4, 8, 6, 2, 3, 7],  # +270 deg yaw
    ],
    dtype=np.int64,
)


class SimpleBackbone(nn.Module):
    def __init__(self, width: int = 32):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, width, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(width * 2, width * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(width * 4, width * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width * 4),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.out_dim = width * 4

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return torch.flatten(x, 1)


class MultiHeadRegressor(nn.Module):
    def __init__(self, backbone: str = "simple"):
        super().__init__()
        if backbone != "simple":
            raise ValueError(
                f"Unsupported backbone: {backbone}. "
                "This script uses a built-in simple CNN to avoid torchvision import issues."
            )
        self.backbone = SimpleBackbone(width=32)
        in_features = self.backbone.out_dim
        self.kpt_head = nn.Linear(in_features, 18)
        self.cls_head = nn.Linear(in_features, 1)

    def forward(self, x):
        feat = self.backbone(x)
        if feat.ndim > 2:
            feat = torch.flatten(feat, 1)
        kpt = self.kpt_head(feat).view(-1, 9, 2)
        cls = self.cls_head(feat).view(-1)
        return kpt, cls


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--images_dir", type=Path, required=True)
    ap.add_argument("--annotations_json", type=Path, default=None, help="Optional, used for intrinsics")
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--canonical_3d", type=Path, default=None)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--max_frames", type=int, default=-1)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--smooth_alpha", type=float, default=0.7, help="EMA smoothing factor (0 disables)")
    ap.add_argument("--threshold_on", type=float, default=None, help="Hysteresis: turn POS on above this")
    ap.add_argument("--threshold_off", type=float, default=None, help="Hysteresis: turn POS off below this")
    ap.add_argument("--pnp_max_err", type=float, default=8.0, help="Reject PnP if reproj err (px) exceeds this; <=0 disables")
    ap.add_argument("--pose_smooth_alpha", type=float, default=0.6, help="EMA for PnP reprojection points (0 disables)")
    ap.add_argument("--flow_blend", type=float, default=0.0, help="Blend tracked keypoints from optical flow (0 disables, 1=all model)")
    ap.add_argument("--flow_win", type=int, default=21, help="Optical flow window size (odd int)")
    ap.add_argument("--flow_max_err", type=float, default=25.0, help="Max LK error per keypoint to accept tracked point")
    ap.add_argument("--flow_min_valid", type=int, default=5, help="Min valid tracked keypoints required to apply flow fusion")
    ap.add_argument("--roi_refine", action="store_true", help="Run a second-stage ROI keypoint refinement pass")
    ap.add_argument("--roi_refine_weight", type=float, default=0.8, help="Blend weight for refined ROI keypoints")
    ap.add_argument("--roi_context", type=float, default=1.45, help="ROI expansion factor around predicted cuboid")
    ap.add_argument("--roi_min_size", type=int, default=96, help="Minimum ROI side length in source pixels")
    ap.add_argument("--roi_score_gate", type=float, default=0.2, help="Apply ROI refinement only when score is above this")
    ap.add_argument("--max_center_jump_frac", type=float, default=0.0, help="Reject keypoints if center jump exceeds this fraction of image diagonal (0 disables)")
    ap.add_argument("--min_area_ratio", type=float, default=0.0, help="Reject keypoints if cuboid 2D bbox area is below running area * ratio (0 disables)")
    ap.add_argument("--max_area_ratio", type=float, default=0.0, help="Reject keypoints if cuboid 2D bbox area is above running area * ratio (0 disables)")
    ap.add_argument("--area_ema", type=float, default=0.9, help="EMA momentum for running cuboid area when area gating is enabled")
    ap.add_argument("--symmetry_pose_search", action="store_true", help="Try yaw-symmetric keypoint permutations in PnP and keep temporally consistent pose")
    ap.add_argument("--symmetry_temporal_weight", type=float, default=0.25, help="Weight for temporal consistency in symmetry pose selection")
    ap.add_argument("--pipeline_mode", choices=["batched", "search_track"], default="batched", help="batched=legacy full-frame pass; search_track=Objectron-like SEARCH/TRACK state machine")
    ap.add_argument("--search_scales", type=str, default="0.30,0.40,0.55,0.70", help="Comma-separated ROI side scales (fraction of min(H,W)) used in SEARCH mode")
    ap.add_argument("--search_stride_frac", type=float, default=0.35, help="SEARCH grid stride as fraction of ROI side")
    ap.add_argument("--search_max_rois", type=int, default=180, help="Cap number of SEARCH ROIs per frame")
    ap.add_argument("--search_every", type=int, default=0, help="Run global SEARCH every N frames even in TRACK mode (0 disables)")
    ap.add_argument("--track_context", type=float, default=1.55, help="TRACK ROI expansion factor around predicted cuboid")
    ap.add_argument("--track_min_size", type=int, default=96, help="Minimum TRACK ROI side length in pixels")
    ap.add_argument("--track_box_ema", type=float, default=0.80, help="EMA momentum for updating track ROI")
    ap.add_argument("--lost_patience", type=int, default=8, help="Consecutive low-score frames before forcing SEARCH mode")
    return ap.parse_args()


def preprocess_rgb(image_bgr: np.ndarray, image_size: int) -> torch.Tensor:
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    image = image.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
    image = (image - mean) / std
    return torch.from_numpy(image).permute(2, 0, 1)


def draw_points(img, pts, color=(255, 0, 0)):
    for i, p in enumerate(pts):
        c = (0, 0, 255) if i == 0 else color
        cv2.circle(img, (int(p[0]), int(p[1])), 4, c, -1, cv2.LINE_AA)
        cv2.putText(img, str(i), (int(p[0]) + 4, int(p[1]) + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)


def draw_cuboid(img, pts, color=(0, 255, 0)):
    for i, j in CUBOID_EDGES:
        p0 = pts[i]
        p1 = pts[j]
        cv2.line(img, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), color, 2, cv2.LINE_AA)


def build_intrinsics_dict(annotations: list) -> Tuple[Dict[str, dict], dict]:
    by_name = {}
    fallback = None
    for ann in annotations:
        name = ann.get("image")
        intr = ann.get("camera_intrinsics", None)
        if name and intr is not None:
            by_name[name] = intr
            if fallback is None:
                fallback = intr
    if fallback is None:
        raise ValueError("No camera_intrinsics found in annotations JSON.")
    return by_name, fallback


def canonical_from_data(canonical_3d: Path, annotations: list):
    if canonical_3d is not None:
        if not canonical_3d.exists():
            raise ValueError(f"canonical_3d file not found: {canonical_3d}")
        pts = np.load(str(canonical_3d)).astype(np.float32)
        if pts.shape != (9, 3):
            raise ValueError(f"canonical_3d must have shape (9,3), got {pts.shape}")
        return pts

    scale = np.asarray(annotations[0]["pose_9dof"]["scale"], dtype=np.float32)
    center_shift = np.array([0.0, 0.5, 0.0], dtype=np.float32)
    return (OBJECTRON_LOCAL_TEMPLATE - center_shift[None, :]) * scale[None, :]


def solve_pose(canonical_3d: np.ndarray, uv_px: np.ndarray, intr: dict):
    fx, fy, cx, cy = float(intr["fx"]), float(intr["fy"]), float(intr["cx"]), float(intr["cy"])
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    dist = np.array(intr.get("dist", [0.0, 0.0, 0.0, 0.0]), dtype=np.float64).reshape(-1, 1)

    obj = canonical_3d[1:].astype(np.float64)
    img = uv_px[1:].astype(np.float64)

    ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_EPNP)
    if not ok:
        return False, None, None, None, None

    reproj, _ = cv2.projectPoints(canonical_3d.astype(np.float64), rvec, tvec, K, dist)
    reproj = reproj.reshape(-1, 2)
    reproj_err = float(np.mean(np.linalg.norm(reproj[1:] - uv_px[1:], axis=1)))
    return True, rvec.reshape(3), tvec.reshape(3), reproj, reproj_err


def solve_pose_with_symmetry(
    canonical_3d: np.ndarray,
    uv_px: np.ndarray,
    intr: dict,
    last_reproj: np.ndarray | None,
    pnp_max_err: float,
    use_symmetry: bool,
    temporal_weight: float,
):
    perms = SYM_PERMS_YAW4 if use_symmetry else SYM_PERMS_YAW4[:1]
    best = None
    best_score = None
    for perm in perms:
        uvp = uv_px[perm]
        ok, rvec, tvec, reproj, reproj_err = solve_pose(canonical_3d, uvp, intr)
        if not ok:
            continue
        if pnp_max_err > 0 and reproj_err > pnp_max_err:
            continue
        temporal_cost = 0.0
        if last_reproj is not None:
            temporal_cost = float(np.mean(np.linalg.norm(reproj - last_reproj, axis=1)))
        score = float(reproj_err + max(0.0, float(temporal_weight)) * temporal_cost)
        if best_score is None or score < best_score:
            best_score = score
            best = (rvec, tvec, reproj, reproj_err, perm)
    return best


def cuboid_bbox_area(uv_px: np.ndarray) -> float:
    x = uv_px[1:, 0]
    y = uv_px[1:, 1]
    w = float(np.max(x) - np.min(x))
    h = float(np.max(y) - np.min(y))
    return max(1.0, w * h)


def roi_bbox_from_uv_px(uv_px: np.ndarray, W: int, H: int, context: float, min_size: int) -> np.ndarray:
    pts = uv_px[1:] if uv_px.shape[0] >= 9 else uv_px
    xs = np.clip(pts[:, 0], 0.0, float(W - 1))
    ys = np.clip(pts[:, 1], 0.0, float(H - 1))
    x0, x1 = float(xs.min()), float(xs.max())
    y0, y1 = float(ys.min()), float(ys.max())
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    bw = max(2.0, x1 - x0)
    bh = max(2.0, y1 - y0)
    side = max(bw, bh) * max(1.0, float(context))
    side = max(side, float(min_size))

    x0 = cx - 0.5 * side
    y0 = cy - 0.5 * side
    x1 = cx + 0.5 * side
    y1 = cy + 0.5 * side

    if x0 < 0:
        x1 -= x0
        x0 = 0.0
    if y0 < 0:
        y1 -= y0
        y0 = 0.0
    if x1 > W:
        x0 -= (x1 - W)
        x1 = float(W)
    if y1 > H:
        y0 -= (y1 - H)
        y1 = float(H)
    x0 = max(0.0, x0)
    y0 = max(0.0, y0)
    x1 = min(float(W), x1)
    y1 = min(float(H), y1)
    if x1 - x0 < 2.0:
        x1 = min(float(W), x0 + 2.0)
    if y1 - y0 < 2.0:
        y1 = min(float(H), y0 + 2.0)
    return np.array([x0, y0, x1, y1], dtype=np.float32)


def predict_uv_in_roi(
    image_bgr: np.ndarray,
    bbox_xyxy: np.ndarray,
    model: nn.Module,
    image_size: int,
    device: torch.device,
    use_amp: bool,
):
    H, W = image_bgr.shape[:2]
    x0, y0, x1, y1 = bbox_xyxy.tolist()
    x0i = int(np.floor(x0))
    y0i = int(np.floor(y0))
    x1i = int(np.ceil(x1))
    y1i = int(np.ceil(y1))
    x0i = max(0, min(W - 2, x0i))
    y0i = max(0, min(H - 2, y0i))
    x1i = max(x0i + 2, min(W, x1i))
    y1i = max(y0i + 2, min(H, y1i))
    crop = image_bgr[y0i:y1i, x0i:x1i]
    if crop.size == 0:
        return None, None
    crop = cv2.resize(crop, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    t = preprocess_rgb(crop, image_size).unsqueeze(0).to(device)
    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            uv, cls_logits = model(t)
    uv = uv[0].detach().cpu().numpy()
    score = float(torch.sigmoid(cls_logits)[0].detach().cpu().item())
    crop_w = max(1.0, float(x1i - x0i))
    crop_h = max(1.0, float(y1i - y0i))
    uv_px = np.stack([x0i + uv[:, 0] * crop_w, y0i + uv[:, 1] * crop_h], axis=1)
    return uv_px, score


def parse_float_csv(s: str, fallback: List[float]) -> List[float]:
    vals = []
    for tok in str(s).split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            vals.append(float(tok))
        except Exception:
            pass
    vals = [v for v in vals if np.isfinite(v) and v > 0.0]
    return vals if vals else fallback


def build_search_rois(
    W: int,
    H: int,
    scales: List[float],
    stride_frac: float,
    min_size: int,
    max_rois: int,
) -> List[np.ndarray]:
    rois: List[np.ndarray] = []
    base = float(min(W, H))
    stride_frac = float(np.clip(stride_frac, 0.10, 1.00))
    for s in scales:
        side = max(float(min_size), float(s) * base)
        side = min(side, float(min(W, H)))
        if side < 2.0:
            continue
        stride = max(8.0, side * stride_frac)
        x_centers = np.arange(side * 0.5, W - side * 0.5 + 1e-3, stride, dtype=np.float32)
        y_centers = np.arange(side * 0.5, H - side * 0.5 + 1e-3, stride, dtype=np.float32)
        if x_centers.size == 0:
            x_centers = np.array([W * 0.5], dtype=np.float32)
        if y_centers.size == 0:
            y_centers = np.array([H * 0.5], dtype=np.float32)
        for cy in y_centers:
            for cx in x_centers:
                x0 = float(np.clip(cx - side * 0.5, 0.0, float(W - 2)))
                y0 = float(np.clip(cy - side * 0.5, 0.0, float(H - 2)))
                x1 = float(np.clip(x0 + side, 2.0, float(W)))
                y1 = float(np.clip(y0 + side, 2.0, float(H)))
                rois.append(np.array([x0, y0, x1, y1], dtype=np.float32))

    # Always include full-frame fallback.
    rois.append(np.array([0.0, 0.0, float(W), float(H)], dtype=np.float32))

    if len(rois) <= int(max_rois):
        return rois

    # Uniformly subsample while preserving coverage.
    keep_idx = np.linspace(0, len(rois) - 1, int(max_rois), dtype=np.int32)
    return [rois[int(i)] for i in keep_idx]


def search_best_roi(
    image_bgr: np.ndarray,
    model: nn.Module,
    image_size: int,
    device: torch.device,
    use_amp: bool,
    scales: List[float],
    stride_frac: float,
    min_size: int,
    max_rois: int,
) -> Tuple[np.ndarray | None, float, np.ndarray | None]:
    H, W = image_bgr.shape[:2]
    rois = build_search_rois(W, H, scales, stride_frac, min_size, max_rois)
    best_uv = None
    best_roi = None
    best_score = -1.0
    for roi in rois:
        uv_px, score = predict_uv_in_roi(
            image_bgr,
            roi,
            model,
            image_size=image_size,
            device=device,
            use_amp=use_amp,
        )
        if uv_px is None or score is None:
            continue
        if float(score) > float(best_score):
            best_uv = uv_px
            best_score = float(score)
            best_roi = roi
    return best_uv, float(best_score), best_roi


def run_search_track_pipeline(
    args,
    model: nn.Module,
    image_size: int,
    device: torch.device,
    use_amp: bool,
    image_paths: List[Path],
    intr_by_name: Dict[str, dict],
    fallback_intr: dict | None,
    canonical_3d: np.ndarray | None,
    overlay_dir: Path,
):
    th_on = args.threshold_on if args.threshold_on is not None else args.threshold
    th_off = args.threshold_off if args.threshold_off is not None else args.threshold
    if th_off > th_on:
        th_off = th_on

    search_scales = parse_float_csv(args.search_scales, fallback=[0.30, 0.40, 0.55, 0.70])

    results: List[dict] = []
    prev_is_pos = False
    prev_gray = None
    prev_uv_px = None
    area_ref = None
    last_reproj = None
    sm_uv = None
    sm_score = None

    is_tracking = False
    track_bbox = None
    low_score_streak = 0

    pbar = tqdm(image_paths, desc="search_track")
    for frame_idx, img_path in enumerate(pbar):
        name = img_path.name
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        H, W = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        uv_px_raw = None
        score_raw = -1.0
        cand_bbox = None
        state_before = "TRACK" if is_tracking else "SEARCH"

        if is_tracking and track_bbox is not None:
            uv_t, score_t = predict_uv_in_roi(
                img,
                track_bbox,
                model,
                image_size=image_size,
                device=device,
                use_amp=use_amp,
            )
            if uv_t is not None and score_t is not None:
                uv_px_raw = uv_t
                score_raw = float(score_t)
                cand_bbox = roi_bbox_from_uv_px(
                    uv_t, W, H, context=args.track_context, min_size=args.track_min_size
                )

        periodic_search = bool(args.search_every > 0 and (frame_idx % int(args.search_every) == 0))
        need_search = (not is_tracking) or periodic_search or (score_raw < th_off) or (uv_px_raw is None)
        if need_search:
            uv_s, score_s, roi_s = search_best_roi(
                img,
                model,
                image_size=image_size,
                device=device,
                use_amp=use_amp,
                scales=search_scales,
                stride_frac=args.search_stride_frac,
                min_size=args.track_min_size,
                max_rois=args.search_max_rois,
            )
            if uv_s is not None and score_s >= score_raw:
                uv_px_raw = uv_s
                score_raw = float(score_s)
                cand_bbox = roi_bbox_from_uv_px(
                    uv_s, W, H, context=args.track_context, min_size=args.track_min_size
                )

        if uv_px_raw is None:
            uv_px_raw = prev_uv_px.copy() if prev_uv_px is not None else np.zeros((9, 2), dtype=np.float32)
            score_raw = 0.0

        if score_raw >= th_on:
            is_tracking = True
            low_score_streak = 0
        elif score_raw <= th_off:
            low_score_streak += 1
            if low_score_streak >= max(1, int(args.lost_patience)):
                is_tracking = False
                track_bbox = None
        else:
            low_score_streak = 0 if is_tracking else low_score_streak

        if is_tracking and cand_bbox is not None:
            if track_bbox is None:
                track_bbox = cand_bbox.copy()
            else:
                m = float(np.clip(args.track_box_ema, 0.0, 0.999))
                track_bbox = m * track_bbox + (1.0 - m) * cand_bbox

        smooth_alpha = float(args.smooth_alpha)
        if smooth_alpha <= 0.0 or sm_uv is None:
            sm_uv = uv_px_raw.copy()
            sm_score = float(score_raw)
        else:
            sm_uv = smooth_alpha * sm_uv + (1.0 - smooth_alpha) * uv_px_raw
            sm_score = smooth_alpha * float(sm_score) + (1.0 - smooth_alpha) * float(score_raw)
        uv_px = sm_uv.copy()
        score = float(sm_score)

        if args.roi_refine and score >= float(args.roi_score_gate):
            roi_bbox = roi_bbox_from_uv_px(uv_px, W, H, args.roi_context, args.roi_min_size)
            uv_roi_px, score_roi = predict_uv_in_roi(
                img, roi_bbox, model, image_size=image_size, device=device, use_amp=use_amp
            )
            if uv_roi_px is not None:
                wr = float(np.clip(args.roi_refine_weight, 0.0, 1.0))
                uv_px = (1.0 - wr) * uv_px + wr * uv_roi_px
                if score_roi is not None:
                    score = 0.5 * score + 0.5 * float(score_roi)

        if (
            args.flow_blend > 0.0
            and prev_gray is not None
            and prev_uv_px is not None
            and prev_uv_px.shape == (9, 2)
        ):
            flow_win = max(3, int(args.flow_win))
            if flow_win % 2 == 0:
                flow_win += 1
            p0 = prev_uv_px.astype(np.float32).reshape(-1, 1, 2)
            p1, st, err = cv2.calcOpticalFlowPyrLK(
                prev_gray,
                gray,
                p0,
                None,
                winSize=(flow_win, flow_win),
                maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
            )
            if p1 is not None and st is not None and err is not None:
                p1 = p1.reshape(-1, 2)
                st = st.reshape(-1).astype(bool)
                err = err.reshape(-1)
                valid = st & np.isfinite(err) & (err <= float(args.flow_max_err))
                if int(valid.sum()) >= int(args.flow_min_valid):
                    b = float(np.clip(args.flow_blend, 0.0, 1.0))
                    uv_px[valid] = b * uv_px[valid] + (1.0 - b) * p1[valid]

        if prev_uv_px is not None:
            diag = float(np.hypot(W, H))
            use_gate = (
                float(args.max_center_jump_frac) > 0.0
                or float(args.min_area_ratio) > 0.0
                or float(args.max_area_ratio) > 0.0
            )
            if use_gate:
                reject = False
                if float(args.max_center_jump_frac) > 0.0:
                    c_prev = prev_uv_px[0]
                    c_cur = uv_px[0]
                    jump = float(np.linalg.norm(c_cur - c_prev) / max(1.0, diag))
                    if jump > float(args.max_center_jump_frac):
                        reject = True

                cur_area = cuboid_bbox_area(uv_px)
                if area_ref is None:
                    area_ref = cuboid_bbox_area(prev_uv_px)
                if not reject and float(args.min_area_ratio) > 0.0:
                    if cur_area < area_ref * float(args.min_area_ratio):
                        reject = True
                if not reject and float(args.max_area_ratio) > 0.0:
                    if cur_area > area_ref * float(args.max_area_ratio):
                        reject = True

                if reject:
                    uv_px = prev_uv_px.copy()
                else:
                    m = float(np.clip(args.area_ema, 0.0, 0.999))
                    area_ref = cur_area if area_ref is None else (m * area_ref + (1.0 - m) * cur_area)

        if score >= th_on:
            is_pos = True
        elif score <= th_off:
            is_pos = False
        else:
            is_pos = prev_is_pos
        prev_is_pos = is_pos

        overlay = img.copy()
        if is_pos:
            if canonical_3d is not None and fallback_intr is not None:
                intr = intr_by_name.get(name, fallback_intr)
                pose = solve_pose_with_symmetry(
                    canonical_3d,
                    uv_px,
                    intr,
                    last_reproj=last_reproj,
                    pnp_max_err=args.pnp_max_err,
                    use_symmetry=args.symmetry_pose_search,
                    temporal_weight=args.symmetry_temporal_weight,
                )
                if pose is not None:
                    rvec, tvec, reproj, reproj_err, perm = pose
                    if perm is not None:
                        uv_px = uv_px[perm]
                    if args.pose_smooth_alpha > 0 and last_reproj is not None:
                        reproj = args.pose_smooth_alpha * last_reproj + (1.0 - args.pose_smooth_alpha) * reproj
                    last_reproj = reproj
                    draw_cuboid(overlay, reproj, color=(0, 255, 0))
                elif last_reproj is not None:
                    draw_cuboid(overlay, last_reproj, color=(0, 255, 0))
            draw_points(overlay, uv_px, color=(255, 0, 0))

        state_after = "TRACK" if is_tracking else "SEARCH"
        label = "POS" if is_pos else "NEG"
        cv2.putText(
            overlay,
            f"{label} {score:.2f} | {state_after}",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0) if is_pos else (0, 0, 255),
            2,
        )
        cv2.imwrite(str(overlay_dir / name), overlay)

        results.append(
            {
                "image": name,
                "score": score,
                "score_raw": float(score_raw),
                "is_positive": bool(is_pos),
                "state_before": state_before,
                "state_after": state_after,
            }
        )

        prev_gray = gray
        prev_uv_px = uv_px.copy()

    return results


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = args.out_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    # Reduce native-thread usage to avoid OpenCV/Torch segfaults on some systems
    try:
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)
    except Exception:
        pass
    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda but CUDA is not available.")
    device = torch.device(args.device)
    use_amp = bool(args.amp and device.type == "cuda")

    # PyTorch 2.6+ defaults to weights_only=True; our checkpoint stores config/history too.
    checkpoint = torch.load(str(args.checkpoint), map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    backbone = config.get("backbone", "simple")
    image_size = int(config.get("image_size", 320))

    model = MultiHeadRegressor(backbone=backbone).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    intr_by_name = {}
    fallback_intr = None
    canonical_3d = None
    if args.annotations_json is not None:
        annotations = json.loads(args.annotations_json.read_text())
        intr_by_name, fallback_intr = build_intrinsics_dict(annotations)
        canonical_3d = canonical_from_data(args.canonical_3d, annotations)

    image_paths = sorted(list(args.images_dir.glob("*.jpg")) + list(args.images_dir.glob("*.png")))
    if args.max_frames > 0:
        image_paths = image_paths[: args.max_frames]
    if not image_paths:
        raise ValueError("No images found for inference.")

    print(
        f"Inference frames: {len(image_paths)} | backbone={backbone} | image_size={image_size} "
        f"| mode={args.pipeline_mode}"
    )

    if args.pipeline_mode == "search_track":
        results = run_search_track_pipeline(
            args=args,
            model=model,
            image_size=image_size,
            device=device,
            use_amp=use_amp,
            image_paths=image_paths,
            intr_by_name=intr_by_name,
            fallback_intr=fallback_intr,
            canonical_3d=canonical_3d,
            overlay_dir=overlay_dir,
        )
        (args.out_dir / "predictions.json").write_text(json.dumps(results, indent=2))
        print(f"Wrote overlays to {overlay_dir}")
        return

    # Pass 1: run model and collect predictions in temporal order
    all_pred_uv: List[np.ndarray] = []
    all_scores: List[float] = []
    all_names: List[str] = []
    pbar = tqdm(range(0, len(image_paths), args.batch_size))
    for start in pbar:
        batch_paths = image_paths[start:start + args.batch_size]
        batch_tensors = []
        batch_names = []
        for p in batch_paths:
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is None:
                continue
            batch_names.append(p.name)
            batch_tensors.append(preprocess_rgb(img, image_size))

        if not batch_tensors:
            continue

        batch = torch.stack(batch_tensors, dim=0).to(device)
        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                pred_uv, cls_logits = model(batch)
        probs = torch.sigmoid(cls_logits).detach().cpu().numpy()
        pred_uv = pred_uv.detach().cpu().numpy()

        for i, name in enumerate(batch_names):
            all_pred_uv.append(pred_uv[i].copy())
            all_scores.append(float(probs[i]))
            all_names.append(name)

    # Smooth predictions with EMA to reduce jitter
    smooth_alpha = float(args.smooth_alpha)
    if smooth_alpha <= 0.0:
        sm_uv = np.array(all_pred_uv, dtype=np.float32)
        sm_scores = np.array(all_scores, dtype=np.float32)
    else:
        sm_uv = np.zeros((len(all_pred_uv), 9, 2), dtype=np.float32)
        sm_scores = np.zeros((len(all_scores),), dtype=np.float32)
        if len(all_pred_uv) > 0:
            sm_uv[0] = all_pred_uv[0]
            sm_scores[0] = all_scores[0]
            for i in range(1, len(all_pred_uv)):
                sm_uv[i] = smooth_alpha * sm_uv[i - 1] + (1.0 - smooth_alpha) * all_pred_uv[i]
                sm_scores[i] = smooth_alpha * sm_scores[i - 1] + (1.0 - smooth_alpha) * all_scores[i]

    # Pass 2: draw overlays using smoothed predictions + hysteresis + pose smoothing
    results: List[dict] = []
    prev_is_pos = False
    last_reproj = None
    prev_gray = None
    prev_uv_px = None
    area_ref = None
    for i, name in enumerate(all_names):
        img_path = args.images_dir / name
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        H, W = img.shape[:2]
        uv = sm_uv[i]
        uv_px = np.stack([uv[:, 0] * W, uv[:, 1] * H], axis=1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        score = float(sm_scores[i])

        # Optional second-stage refinement on an object-centered ROI.
        if args.roi_refine and score >= float(args.roi_score_gate):
            roi_bbox = roi_bbox_from_uv_px(uv_px, W, H, args.roi_context, args.roi_min_size)
            uv_roi_px, score_roi = predict_uv_in_roi(
                img, roi_bbox, model, image_size=image_size, device=device, use_amp=use_amp
            )
            if uv_roi_px is not None:
                wr = float(np.clip(args.roi_refine_weight, 0.0, 1.0))
                uv_px = (1.0 - wr) * uv_px + wr * uv_roi_px
                if score_roi is not None:
                    score = 0.5 * score + 0.5 * float(score_roi)

        # Optional temporal fusion with optical-flow tracked keypoints.
        # This reduces frame-to-frame keypoint jumps when the regressor is unstable.
        if (
            args.flow_blend > 0.0
            and prev_gray is not None
            and prev_uv_px is not None
            and prev_uv_px.shape == (9, 2)
        ):
            flow_win = max(3, int(args.flow_win))
            if flow_win % 2 == 0:
                flow_win += 1
            p0 = prev_uv_px.astype(np.float32).reshape(-1, 1, 2)
            p1, st, err = cv2.calcOpticalFlowPyrLK(
                prev_gray,
                gray,
                p0,
                None,
                winSize=(flow_win, flow_win),
                maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
            )
            if p1 is not None and st is not None and err is not None:
                p1 = p1.reshape(-1, 2)
                st = st.reshape(-1).astype(bool)
                err = err.reshape(-1)
                valid = st & np.isfinite(err) & (err <= float(args.flow_max_err))
                if int(valid.sum()) >= int(args.flow_min_valid):
                    b = float(np.clip(args.flow_blend, 0.0, 1.0))
                    uv_px[valid] = b * uv_px[valid] + (1.0 - b) * p1[valid]

        # Optional temporal gating to reject sudden geometric collapses.
        if prev_uv_px is not None:
            diag = float(np.hypot(W, H))
            use_gate = (
                float(args.max_center_jump_frac) > 0.0
                or float(args.min_area_ratio) > 0.0
                or float(args.max_area_ratio) > 0.0
            )
            if use_gate:
                reject = False
                if float(args.max_center_jump_frac) > 0.0:
                    c_prev = prev_uv_px[0]
                    c_cur = uv_px[0]
                    jump = float(np.linalg.norm(c_cur - c_prev) / max(1.0, diag))
                    if jump > float(args.max_center_jump_frac):
                        reject = True

                cur_area = cuboid_bbox_area(uv_px)
                if area_ref is None:
                    area_ref = cuboid_bbox_area(prev_uv_px)
                if not reject and float(args.min_area_ratio) > 0.0:
                    if cur_area < area_ref * float(args.min_area_ratio):
                        reject = True
                if not reject and float(args.max_area_ratio) > 0.0:
                    if cur_area > area_ref * float(args.max_area_ratio):
                        reject = True

                if reject:
                    uv_px = prev_uv_px.copy()
                else:
                    m = float(np.clip(args.area_ema, 0.0, 0.999))
                    area_ref = cur_area if area_ref is None else (m * area_ref + (1.0 - m) * cur_area)

        th_on = args.threshold_on if args.threshold_on is not None else args.threshold
        th_off = args.threshold_off if args.threshold_off is not None else args.threshold
        if th_off > th_on:
            th_off = th_on
        if score >= th_on:
            is_pos = True
        elif score <= th_off:
            is_pos = False
        else:
            is_pos = prev_is_pos
        prev_is_pos = is_pos

        overlay = img.copy()
        if is_pos:
            if canonical_3d is not None and fallback_intr is not None:
                intr = intr_by_name.get(name, fallback_intr)
                pose = solve_pose_with_symmetry(
                    canonical_3d,
                    uv_px,
                    intr,
                    last_reproj=last_reproj,
                    pnp_max_err=args.pnp_max_err,
                    use_symmetry=args.symmetry_pose_search,
                    temporal_weight=args.symmetry_temporal_weight,
                )
                if pose is not None:
                    rvec, tvec, reproj, reproj_err, perm = pose
                    if perm is not None:
                        uv_px = uv_px[perm]
                    # Smooth reprojection points to reduce pose jitter
                    if args.pose_smooth_alpha > 0 and last_reproj is not None:
                        reproj = args.pose_smooth_alpha * last_reproj + (1.0 - args.pose_smooth_alpha) * reproj
                    last_reproj = reproj
                    draw_cuboid(overlay, reproj, color=(0, 255, 0))
                elif last_reproj is not None:
                    # Fallback to last stable pose if current frame is unreliable
                    draw_cuboid(overlay, last_reproj, color=(0, 255, 0))
            draw_points(overlay, uv_px, color=(255, 0, 0))
        label = "POS" if is_pos else "NEG"
        cv2.putText(
            overlay, f"{label} {score:.2f}",
            (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (0, 255, 0) if is_pos else (0, 0, 255), 2
        )

        out_path = overlay_dir / name
        cv2.imwrite(str(out_path), overlay)
        results.append({"image": name, "score": score, "is_positive": bool(is_pos)})
        prev_gray = gray
        prev_uv_px = uv_px.copy()

    (args.out_dir / "predictions.json").write_text(json.dumps(results, indent=2))
    print(f"Wrote overlays to {overlay_dir}")


if __name__ == "__main__":
    main()
