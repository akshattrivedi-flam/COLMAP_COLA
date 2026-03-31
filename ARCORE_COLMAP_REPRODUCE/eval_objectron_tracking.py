import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm

# Our annotation/inference keypoint order is:
# 0 center, 1..4 front face (bottom-left, bottom-right, top-right, top-left),
# 5..8 rear face  (bottom-left, bottom-right, top-right, top-left).
# Objectron IoU helper (objectron.dataset.box.Box) expects UNIT_BOX order:
# 1[-,-,-],2[-,-,+],3[-,+,-],4[-,+,+],5[+,-,-],6[+,-,+],7[+,+,-],8[+,+,+].
# Reorder indices to match Box expectations before IoU.
OUR_TO_BOXLIB = np.array([0, 5, 1, 8, 4, 6, 2, 7, 3], dtype=np.int64)
PAPER_MAX_PIXEL_ERROR = 20.0
PAPER_MAX_AZIMUTH_ERROR = 30.0
PAPER_MAX_POLAR_ERROR = 20.0
PAPER_MAX_DISTANCE = 1.0
PAPER_NUM_BINS = 21


class HitMiss:
    def __init__(self, thresholds: np.ndarray):
        self.thresholds = np.asarray(thresholds, dtype=np.float64)
        self.hit = np.zeros((self.thresholds.shape[0],), dtype=np.float64)
        self.miss = np.zeros((self.thresholds.shape[0],), dtype=np.float64)

    def record(self, metric: float, greater: bool = True) -> None:
        for i, threshold in enumerate(self.thresholds):
            hit = (greater and metric >= threshold) or ((not greater) and metric <= threshold)
            if hit:
                self.hit[i] += 1.0
            else:
                self.miss[i] += 1.0


class AveragePrecision:
    def __init__(self, thresholds: np.ndarray):
        self.thresholds = np.asarray(thresholds, dtype=np.float64)
        self.true_positive: List[List[float]] = [[] for _ in range(self.thresholds.shape[0])]
        self.false_positive: List[List[float]] = [[] for _ in range(self.thresholds.shape[0])]
        self.total_instances = 0.0

    def append(self, hit_miss: HitMiss, num_instances: int) -> None:
        for i in range(self.thresholds.shape[0]):
            self.true_positive[i].append(float(hit_miss.hit[i]))
            self.false_positive[i].append(float(hit_miss.miss[i]))
        self.total_instances += float(num_instances)

    @staticmethod
    def _compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
        recall = np.insert(recall, 0, [0.0])
        recall = np.append(recall, [1.0])
        precision = np.insert(precision, 0, [0.0])
        precision = np.append(precision, [0.0])
        monotonic_precision = precision.copy()
        for i in range(len(monotonic_precision) - 2, -1, -1):
            monotonic_precision[i] = max(monotonic_precision[i], monotonic_precision[i + 1])
        ap = 0.0
        for i in range(1, len(recall)):
            if recall[i] != recall[i - 1]:
                ap += (recall[i] - recall[i - 1]) * monotonic_precision[i]
        return float(ap)

    def compute_curve(self) -> List[float]:
        if self.total_instances <= 0:
            return [float("nan")] * self.thresholds.shape[0]
        aps: List[float] = []
        for i in range(self.thresholds.shape[0]):
            tp = np.cumsum(np.asarray(self.true_positive[i], dtype=np.float64))
            fp = np.cumsum(np.asarray(self.false_positive[i], dtype=np.float64))
            denom = tp + fp
            recall = tp / max(self.total_instances, 1e-6)
            precision = np.divide(tp, denom, out=np.zeros_like(tp), where=denom != 0)
            aps.append(self._compute_ap(recall, precision))
        return aps


def load_infer_module(script_path: Path):
    spec = importlib.util.spec_from_file_location(script_path.stem, str(script_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to import inference module: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args():
    ap = argparse.ArgumentParser(
        description="Objectron-style tracking evaluation (per-video 3D IoU + reprojection + jitter)"
    )
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)

    ap.add_argument("--video_dirs", nargs="*", default=[])
    ap.add_argument("--video_dirs_txt", nargs="*", default=[], help="Text file(s), one video dir per line")
    ap.add_argument("--class_roots", nargs="*", default=[], help="Class root dirs (e.g. .../Blue .../Red)")
    ap.add_argument("--run_dir", type=Path, default=None, help="Reads included_*_dirs.txt from run dir")

    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_frames", type=int, default=-1)

    # Match inference controls for fair evaluation
    ap.add_argument("--threshold", type=float, default=0.8)
    ap.add_argument("--smooth_alpha", type=float, default=0.4)
    ap.add_argument("--threshold_on", type=float, default=0.85)
    ap.add_argument("--threshold_off", type=float, default=0.6)
    ap.add_argument("--pnp_max_err", type=float, default=8.0)
    ap.add_argument("--pose_smooth_alpha", type=float, default=0.55)
    ap.add_argument("--flow_blend", type=float, default=0.1)
    ap.add_argument("--flow_win", type=int, default=21)
    ap.add_argument("--flow_max_err", type=float, default=25.0)
    ap.add_argument("--flow_min_valid", type=int, default=5)
    ap.add_argument("--roi_refine", action="store_true")
    ap.add_argument("--roi_refine_weight", type=float, default=0.8)
    ap.add_argument("--roi_context", type=float, default=1.45)
    ap.add_argument("--roi_min_size", type=int, default=96)
    ap.add_argument("--roi_score_gate", type=float, default=0.2)
    ap.add_argument("--max_center_jump_frac", type=float, default=0.06)
    ap.add_argument("--min_area_ratio", type=float, default=0.65)
    ap.add_argument("--max_area_ratio", type=float, default=1.7)
    ap.add_argument("--area_ema", type=float, default=0.95)
    ap.add_argument("--symmetry_pose_search", action="store_true", help="Try yaw-symmetric keypoint permutations in PnP")
    ap.add_argument("--symmetry_temporal_weight", type=float, default=0.25)
    ap.add_argument(
        "--symmetry_iou_steps",
        type=int,
        default=72,
        help="Discrete yaw samples used for symmetry-aware 3D IoU on cylindrical objects",
    )
    return ap.parse_args()


def read_video_dirs_from_txt(path: Path) -> List[Path]:
    out = []
    if not path.exists():
        return out
    base = path.parent
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        p = Path(s)
        if not p.is_absolute():
            p = (base / p).resolve()
        out.append(p)
    return out


def collect_video_dirs(args) -> List[Path]:
    videos: List[Path] = []

    for s in args.video_dirs:
        videos.append(Path(s))
    for s in args.video_dirs_txt:
        videos.extend(read_video_dirs_from_txt(Path(s)))

    if args.run_dir is not None:
        for name in [
            "included_blue_dirs.txt",
            "included_red_dirs.txt",
            "included_silver_dirs.txt",
            "included_negative_dirs.txt",
        ]:
            videos.extend(read_video_dirs_from_txt(args.run_dir / name))

    for root in args.class_roots:
        root = Path(root)
        if not root.exists():
            continue
        for p in sorted(root.glob("video_*")):
            if (p / "objectron_prep" / "annotations.json").exists():
                videos.append(p)

    # de-duplicate while preserving order
    uniq = []
    seen = set()
    for p in videos:
        rp = p.resolve()
        if str(rp) in seen:
            continue
        seen.add(str(rp))
        uniq.append(rp)
    return uniq


def infer_expected_label(video_dir: Path):
    s = str(video_dir).lower()
    if "/blue/" in s or s.endswith("_blue"):
        return 1
    if "/red/" in s or "/silver/" in s or s.endswith("_red") or s.endswith("_silver"):
        return 0
    return None


def get_objectron_iou_lib(repo_root: Path):
    objron_root = repo_root / "OBJECTRON_OG" / "Objectron"
    if str(objron_root) not in sys.path:
        sys.path.insert(0, str(objron_root))
    from objectron.dataset import box as objectron_box  # type: ignore
    from objectron.dataset import iou as objectron_iou  # type: ignore

    return objectron_box, objectron_iou


def cuboid_bbox_area(uv_px: np.ndarray) -> float:
    x = uv_px[1:, 0]
    y = uv_px[1:, 1]
    w = float(np.max(x) - np.min(x))
    h = float(np.max(y) - np.min(y))
    return max(1.0, w * h)


def reorder_kpts_for_boxlib(k3d_9x3: np.ndarray) -> np.ndarray:
    k = np.asarray(k3d_9x3, dtype=np.float64)
    if k.shape != (9, 3):
        raise ValueError(f"Expected (9,3) keypoints, got {k.shape}")
    return k[OUR_TO_BOXLIB]


def objectron_pixel_error(pred_uv_px: np.ndarray, gt_uv_px: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(pred_uv_px[1:] - gt_uv_px[1:], axis=1)))


def compute_average_distance(pred_k3d: np.ndarray, gt_k3d: np.ndarray) -> Tuple[float, float]:
    pred = np.asarray(pred_k3d, dtype=np.float64)
    gt = np.asarray(gt_k3d, dtype=np.float64)
    add = float(np.mean(np.linalg.norm(pred - gt, axis=1)))

    sym_dists = []
    for i in range(pred.shape[0]):
        d = np.linalg.norm(pred[i][None, :] - gt, axis=1)
        sym_dists.append(float(np.min(d)))
    adds = float(np.mean(sym_dists))
    return add, adds


def compute_ray(box_k3d: np.ndarray) -> np.ndarray:
    box = np.asarray(box_k3d, dtype=np.float64)
    size_x = np.linalg.norm(box[5] - box[1])
    size_y = np.linalg.norm(box[3] - box[1])
    size_z = np.linalg.norm(box[2] - box[1])
    size = np.asarray([size_x, size_y, size_z], dtype=np.float64)
    unit_box = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [-0.5, -0.5, -0.5],
            [-0.5, -0.5, 0.5],
            [-0.5, 0.5, -0.5],
            [-0.5, 0.5, 0.5],
            [0.5, -0.5, -0.5],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, -0.5],
            [0.5, 0.5, 0.5],
        ],
        dtype=np.float64,
    )
    box_o = unit_box * size[None, :]
    box_oh = np.ones((4, 9), dtype=np.float64)
    box_oh[:3] = box_o.T
    box_ch = np.ones((4, 9), dtype=np.float64)
    box_ch[:3] = box.T
    box_oct = box_oh @ box_ch.T
    box_cct_inv = np.linalg.inv(box_ch @ box_ch.T)
    transform = box_oct @ box_cct_inv
    return transform[:3, 3].reshape((3,))


def compute_viewpoint(box_k3d: np.ndarray) -> Tuple[float, float]:
    x, y, z = compute_ray(box_k3d)
    azimuth = math.degrees(math.atan2(z, x))
    polar = math.degrees(math.atan2(y, math.hypot(x, z)))
    return float(azimuth), float(polar)


def compute_viewpoint_errors(pred_k3d: np.ndarray, gt_k3d: np.ndarray) -> Tuple[float, float]:
    pred_azimuth, pred_polar = compute_viewpoint(pred_k3d)
    gt_azimuth, gt_polar = compute_viewpoint(gt_k3d)
    polar_error = abs(pred_polar - gt_polar)
    azimuth_error = abs(pred_azimuth - gt_azimuth)
    if azimuth_error > 180.0:
        azimuth_error = 360.0 - azimuth_error
    return float(azimuth_error), float(polar_error)


def rotation_matrix_y(angle_rad: float) -> np.ndarray:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.asarray(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=np.float64,
    )


def maximize_symmetric_iou(
    pred_boxlib: np.ndarray,
    gt_boxlib: np.ndarray,
    objectron_box,
    objectron_iou,
    steps: int = 72,
) -> float:
    pred_box = objectron_box.Box(vertices=np.asarray(pred_boxlib, dtype=np.float64))
    gt_box = objectron_box.Box(vertices=np.asarray(gt_boxlib, dtype=np.float64))
    rotation = np.asarray(pred_box.rotation, dtype=np.float64)
    translation = np.asarray(pred_box.translation, dtype=np.float64).reshape(3)
    scale = np.asarray(pred_box.scale, dtype=np.float64).reshape(3)
    best = float("nan")
    for angle in np.linspace(0.0, 2.0 * math.pi, num=max(1, int(steps)), endpoint=False):
        yaw = rotation_matrix_y(float(angle))
        cand_box = objectron_box.Box.from_transformation(rotation @ yaw, translation, scale)
        iou = float(objectron_iou.IoU(cand_box, gt_box).iou())
        if not np.isfinite(iou):
            continue
        if not np.isfinite(best) or iou > best:
            best = iou
    return best


def is_paper_eval_frame(ann: dict, gt_k2d: np.ndarray) -> bool:
    if gt_k2d.shape[0] != 9 or gt_k2d.shape[1] < 2:
        return False
    center_visible = bool(0.0 < float(gt_k2d[0, 0]) < 1.0 and 0.0 < float(gt_k2d[0, 1]) < 1.0)
    vis = np.asarray(ann.get("keypoints_2d_visibility", []), dtype=np.float32)
    if vis.shape[0] == 9:
        visibility = float(np.mean(np.clip(vis, 0.0, 1.0)))
    else:
        visibility = 1.0
    return center_visible and visibility > 0.1


def threshold_value(thresholds: np.ndarray, values: List[float], target: float) -> float | None:
    if not values:
        return None
    idx = int(np.argmin(np.abs(np.asarray(thresholds, dtype=np.float64) - float(target))))
    return float(values[idx])


def detect_checkpoint_kind(checkpoint: dict) -> str:
    if isinstance(checkpoint, dict) and "detector_state" in checkpoint and "regressor_state" in checkpoint:
        return "paper_twostage"
    return "legacy"


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda but CUDA is not available.")
    device = torch.device(args.device)
    use_amp = bool(args.amp and device.type == "cuda")

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    objectron_box, objectron_iou = get_objectron_iou_lib(repo_root)

    checkpoint = torch.load(str(args.checkpoint), map_location=device, weights_only=False)
    checkpoint_kind = detect_checkpoint_kind(checkpoint)
    if checkpoint_kind == "paper_twostage":
        infer_mod = load_infer_module(script_dir / "infer_objectron_paper_twostage.py")
        detector, regressor, _, config = infer_mod.load_models(args.checkpoint, device=device)
        model = None
        image_size = int(config.get("regressor_image_size", 224))
        print(f"Checkpoint kind: {checkpoint_kind} | detector+regressor | regressor_image_size={image_size}")
    else:
        infer_mod = load_infer_module(script_dir / "infer_objectron_can_posneg.py")
        config = checkpoint.get("config", {})
        backbone = config.get("backbone", "simple")
        image_size = int(config.get("image_size", 320))
        model = infer_mod.MultiHeadRegressor(backbone=backbone).to(device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        detector = None
        regressor = None
        print(f"Checkpoint kind: {checkpoint_kind} | backbone={backbone} | image_size={image_size}")

    video_dirs = collect_video_dirs(args)
    if not video_dirs:
        raise ValueError("No video directories found. Provide --video_dirs/--video_dirs_txt/--class_roots/--run_dir")

    print(f"Evaluating videos: {len(video_dirs)}")
    rows = []
    global_cls = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    paper_iou_thresholds = np.linspace(0.0, 1.0, num=PAPER_NUM_BINS)
    paper_pixel_thresholds = np.linspace(0.0, PAPER_MAX_PIXEL_ERROR, num=PAPER_NUM_BINS)
    paper_azimuth_thresholds = np.linspace(0.0, PAPER_MAX_AZIMUTH_ERROR, num=PAPER_NUM_BINS)
    paper_polar_thresholds = np.linspace(0.0, PAPER_MAX_POLAR_ERROR, num=PAPER_NUM_BINS)
    paper_add_thresholds = np.linspace(0.0, PAPER_MAX_DISTANCE, num=PAPER_NUM_BINS)
    paper_adds_thresholds = np.linspace(0.0, PAPER_MAX_DISTANCE, num=PAPER_NUM_BINS)
    paper_iou_ap = AveragePrecision(paper_iou_thresholds)
    paper_pixel_ap = AveragePrecision(paper_pixel_thresholds)
    paper_azimuth_ap = AveragePrecision(paper_azimuth_thresholds)
    paper_polar_ap = AveragePrecision(paper_polar_thresholds)
    paper_add_ap = AveragePrecision(paper_add_thresholds)
    paper_adds_ap = AveragePrecision(paper_adds_thresholds)
    paper_eval_frames_total = 0
    paper_detected_frames_total = 0

    for video_dir in video_dirs:
        ann_path = video_dir / "objectron_prep" / "annotations.json"
        img_dir = video_dir / "frames_rotated"
        if not ann_path.exists() or not img_dir.exists():
            print(f"SKIP (missing data): {video_dir}")
            continue

        annotations = json.loads(ann_path.read_text())
        ann_by_name: Dict[str, dict] = {a.get("image", ""): a for a in annotations}
        names = [a.get("image", "") for a in annotations if (img_dir / a.get("image", "")).exists()]
        if args.max_frames > 0:
            names = names[: args.max_frames]
        if not names:
            print(f"SKIP (no frames): {video_dir}")
            continue

        intr_by_name, fallback_intr = infer_mod.build_intrinsics_dict(annotations)
        canonical_3d = infer_mod.canonical_from_data(None, annotations)

        # Pass 1 (model)
        eval_names: List[str] = []
        all_pred_uv_px: List[np.ndarray] = []
        all_scores: List[float] = []
        if checkpoint_kind == "paper_twostage":
            for start in tqdm(range(0, len(names), args.batch_size), leave=False, desc=f"{video_dir.name} pass1"):
                batch_names = names[start:start + args.batch_size]
                rgb_images = []
                batch_tensors = []
                kept_names = []
                for name in batch_names:
                    img = cv2.imread(str(img_dir / name), cv2.IMREAD_COLOR)
                    if img is None:
                        continue
                    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    rgb_images.append(rgb)
                    batch_tensors.append(infer_mod.detector_image_to_tensor(rgb))
                    kept_names.append(name)
                if not batch_tensors:
                    continue

                det_inputs = [t.to(device) for t in batch_tensors]
                with torch.no_grad():
                    outputs = detector(det_inputs)

                reg_tensors = []
                reg_boxes = []
                reg_names = []
                reg_scores = []
                for name, rgb, output in zip(kept_names, rgb_images, outputs):
                    scores = output["scores"].detach().cpu().numpy()
                    boxes = output["boxes"].detach().cpu().numpy()
                    if scores.size > 0:
                        best_idx = int(np.argmax(scores))
                        score = float(scores[best_idx])
                        box_xyxy = boxes[best_idx].astype(np.float32)
                    else:
                        score = 0.0
                        box_xyxy = np.array([0.0, 0.0, float(rgb.shape[1]), float(rgb.shape[0])], dtype=np.float32)
                    crop, clipped_box, _ = infer_mod.crop_and_resize(rgb, box_xyxy=box_xyxy, output_size=image_size)
                    reg_tensors.append(infer_mod.regressor_image_to_tensor(crop))
                    reg_boxes.append(clipped_box)
                    reg_names.append(name)
                    reg_scores.append(score)

                if not reg_tensors:
                    continue

                reg_batch = torch.stack(reg_tensors, dim=0).to(device)
                with torch.no_grad():
                    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                        pred_crop_uv = regressor(reg_batch).detach().cpu().numpy()

                for i, name in enumerate(reg_names):
                    pred_uv_px = infer_mod.crop_uv_to_keypoints_px(pred_crop_uv[i], reg_boxes[i])
                    eval_names.append(name)
                    all_pred_uv_px.append(pred_uv_px.astype(np.float32))
                    all_scores.append(float(reg_scores[i]))
        else:
            for start in tqdm(range(0, len(names), args.batch_size), leave=False, desc=f"{video_dir.name} pass1"):
                batch_names = names[start:start + args.batch_size]
                batch_tensors = []
                kept_names = []
                kept_wh = []
                for name in batch_names:
                    img = cv2.imread(str(img_dir / name), cv2.IMREAD_COLOR)
                    if img is None:
                        continue
                    H, W = img.shape[:2]
                    batch_tensors.append(infer_mod.preprocess_rgb(img, image_size))
                    kept_names.append(name)
                    kept_wh.append((W, H))
                if not batch_tensors:
                    continue
                batch = torch.stack(batch_tensors, dim=0).to(device)
                with torch.no_grad():
                    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                        pred_uv, cls_logits = model(batch)
                probs = torch.sigmoid(cls_logits).detach().cpu().numpy()
                pred_uv = pred_uv.detach().cpu().numpy()
                for i, name in enumerate(kept_names):
                    W, H = kept_wh[i]
                    pred_uv_px = np.stack([pred_uv[i][:, 0] * W, pred_uv[i][:, 1] * H], axis=1)
                    eval_names.append(name)
                    all_pred_uv_px.append(pred_uv_px.astype(np.float32))
                    all_scores.append(float(probs[i]))

        n = len(eval_names)
        if n == 0:
            print(f"SKIP (model produced no predictions): {video_dir}")
            continue

        # EMA smoothing
        smooth_alpha = float(args.smooth_alpha)
        if smooth_alpha <= 0.0:
            sm_uv_px = np.array(all_pred_uv_px, dtype=np.float32)
            sm_scores = np.array(all_scores, dtype=np.float32)
        else:
            sm_uv_px = np.zeros((n, 9, 2), dtype=np.float32)
            sm_scores = np.zeros((n,), dtype=np.float32)
            sm_uv_px[0] = all_pred_uv_px[0]
            sm_scores[0] = all_scores[0]
            for i in range(1, n):
                sm_uv_px[i] = smooth_alpha * sm_uv_px[i - 1] + (1.0 - smooth_alpha) * all_pred_uv_px[i]
                sm_scores[i] = smooth_alpha * sm_scores[i - 1] + (1.0 - smooth_alpha) * all_scores[i]

        # Pass 2 (temporal + geometry)
        th_on = args.threshold_on if args.threshold_on is not None else args.threshold
        th_off = args.threshold_off if args.threshold_off is not None else args.threshold
        if th_off > th_on:
            th_off = th_on

        prev_is_pos = False
        prev_gray = None
        prev_uv_px = None
        area_ref = None
        last_reproj = None

        gt_uv_seq = []
        pred_uv_seq = []
        gt_3d_seq = []
        pred_3d_seq = []
        iou_3d_vals = []
        iou_3d_sym_vals = []
        reproj_px_vals = []
        pixel_error_objron_vals = []
        pnp_reproj_vals = []
        azimuth_err_vals = []
        polar_err_vals = []
        add_vals = []
        adds_vals = []
        score_vals = []
        is_pos_vals = []
        valid_3d_count = 0
        paper_eval_frames = 0
        paper_detected_frames = 0

        expected_label = infer_expected_label(video_dir)

        for i, name in enumerate(eval_names):
            img = cv2.imread(str(img_dir / name), cv2.IMREAD_COLOR)
            if img is None:
                continue
            H, W = img.shape[:2]
            ann = ann_by_name.get(name)
            if ann is None:
                continue

            uv_px = sm_uv_px[i].copy()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            score = float(sm_scores[i])

            if checkpoint_kind == "legacy" and args.roi_refine and score >= float(args.roi_score_gate):
                roi_bbox = infer_mod.roi_bbox_from_uv_px(uv_px, W, H, args.roi_context, args.roi_min_size)
                uv_roi_px, score_roi = infer_mod.predict_uv_in_roi(
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
                    prev_gray, gray, p0, None,
                    winSize=(flow_win, flow_win), maxLevel=3,
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
                use_gate = (
                    float(args.max_center_jump_frac) > 0.0
                    or float(args.min_area_ratio) > 0.0
                    or float(args.max_area_ratio) > 0.0
                )
                if use_gate:
                    reject = False
                    diag = float(np.hypot(W, H))
                    if float(args.max_center_jump_frac) > 0.0:
                        jump = float(np.linalg.norm(uv_px[0] - prev_uv_px[0]) / max(1.0, diag))
                        if jump > float(args.max_center_jump_frac):
                            reject = True
                    cur_area = cuboid_bbox_area(uv_px)
                    if area_ref is None:
                        area_ref = cuboid_bbox_area(prev_uv_px)
                    if not reject and float(args.min_area_ratio) > 0.0 and cur_area < area_ref * float(args.min_area_ratio):
                        reject = True
                    if not reject and float(args.max_area_ratio) > 0.0 and cur_area > area_ref * float(args.max_area_ratio):
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

            if expected_label is not None:
                if expected_label == 1 and is_pos:
                    global_cls["tp"] += 1
                elif expected_label == 1 and not is_pos:
                    global_cls["fn"] += 1
                elif expected_label == 0 and is_pos:
                    global_cls["fp"] += 1
                else:
                    global_cls["tn"] += 1

            gt_k2d = np.asarray(ann.get("keypoints_2d", []), dtype=np.float32)
            gt_k3d = np.asarray(ann.get("keypoints_3d", []), dtype=np.float32)
            if gt_k2d.shape[0] != 9 or gt_k2d.shape[1] < 2:
                continue
            gt_uv_px = np.stack([gt_k2d[:, 0] * W, gt_k2d[:, 1] * H], axis=1)
            pred_uv_eval = uv_px.copy()
            reproj_px = float(np.mean(np.linalg.norm(uv_px - gt_uv_px, axis=1)))
            reproj_px_vals.append(reproj_px)
            pixel_err_objron = objectron_pixel_error(pred_uv_eval, gt_uv_px)
            pixel_error_objron_vals.append(pixel_err_objron)

            gt_uv_seq.append(gt_uv_px)
            pred_uv_seq.append(uv_px.copy())
            score_vals.append(score)
            is_pos_vals.append(bool(is_pos))

            intr = intr_by_name.get(name, fallback_intr)
            pose = infer_mod.solve_pose_with_symmetry(
                canonical_3d,
                uv_px,
                intr,
                last_reproj=last_reproj,
                pnp_max_err=args.pnp_max_err,
                use_symmetry=args.symmetry_pose_search,
                temporal_weight=args.symmetry_temporal_weight,
            )
            paper_iou = 0.0
            paper_iou_sym = float("nan")
            paper_add = PAPER_MAX_DISTANCE
            paper_adds = PAPER_MAX_DISTANCE
            paper_azimuth_error = PAPER_MAX_AZIMUTH_ERROR
            paper_polar_error = PAPER_MAX_POLAR_ERROR
            if pose is not None:
                rvec, tvec, reproj, pnp_err, perm = pose
                if perm is not None:
                    uv_px = uv_px[perm]
                if args.pose_smooth_alpha > 0 and last_reproj is not None:
                    reproj = args.pose_smooth_alpha * last_reproj + (1.0 - args.pose_smooth_alpha) * reproj
                last_reproj = reproj
                pnp_reproj_vals.append(float(pnp_err))

                R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))
                pred_k3d = (R @ canonical_3d.T).T + np.asarray(tvec, dtype=np.float64).reshape(1, 3)
                if gt_k3d.shape == (9, 3):
                    try:
                        pred_boxlib = reorder_kpts_for_boxlib(pred_k3d)
                        gt_boxlib = reorder_kpts_for_boxlib(gt_k3d)
                        b_pred = objectron_box.Box(vertices=pred_boxlib)
                        b_gt = objectron_box.Box(vertices=gt_boxlib)
                        iou3d = float(objectron_iou.IoU(b_pred, b_gt).iou())
                        if np.isfinite(iou3d):
                            iou_3d_vals.append(iou3d)
                            paper_iou = iou3d
                        paper_iou_sym = maximize_symmetric_iou(
                            pred_boxlib,
                            gt_boxlib,
                            objectron_box,
                            objectron_iou,
                            steps=args.symmetry_iou_steps,
                        )
                        if np.isfinite(paper_iou_sym):
                            iou_3d_sym_vals.append(paper_iou_sym)
                        paper_add, paper_adds = compute_average_distance(pred_k3d, gt_k3d)
                        add_vals.append(paper_add)
                        adds_vals.append(paper_adds)
                        paper_azimuth_error, paper_polar_error = compute_viewpoint_errors(pred_k3d, gt_k3d)
                        azimuth_err_vals.append(paper_azimuth_error)
                        polar_err_vals.append(paper_polar_error)
                        pred_3d_seq.append(pred_k3d.astype(np.float32))
                        gt_3d_seq.append(gt_k3d.astype(np.float32))
                        valid_3d_count += 1
                    except Exception:
                        pass

            if expected_label == 1 and is_paper_eval_frame(ann, gt_k2d):
                paper_eval_frames += 1
                paper_eval_frames_total += 1
                detected = bool(is_pos)
                if detected:
                    paper_detected_frames += 1
                    paper_detected_frames_total += 1

                pixel_metric = pixel_err_objron if detected else PAPER_MAX_PIXEL_ERROR
                azimuth_metric = paper_azimuth_error if detected else PAPER_MAX_AZIMUTH_ERROR
                polar_metric = paper_polar_error if detected else PAPER_MAX_POLAR_ERROR
                iou_metric = paper_iou if detected else 0.0
                add_metric = paper_add if detected else PAPER_MAX_DISTANCE
                adds_metric = paper_adds if detected else PAPER_MAX_DISTANCE

                hm_iou = HitMiss(paper_iou_thresholds)
                hm_iou.record(iou_metric, greater=True)
                paper_iou_ap.append(hm_iou, 1)

                hm_pixel = HitMiss(paper_pixel_thresholds)
                hm_pixel.record(pixel_metric, greater=False)
                paper_pixel_ap.append(hm_pixel, 1)

                hm_azimuth = HitMiss(paper_azimuth_thresholds)
                hm_azimuth.record(azimuth_metric, greater=False)
                paper_azimuth_ap.append(hm_azimuth, 1)

                hm_polar = HitMiss(paper_polar_thresholds)
                hm_polar.record(polar_metric, greater=False)
                paper_polar_ap.append(hm_polar, 1)

                hm_add = HitMiss(paper_add_thresholds)
                hm_add.record(add_metric, greater=False)
                paper_add_ap.append(hm_add, 1)

                hm_adds = HitMiss(paper_adds_thresholds)
                hm_adds.record(adds_metric, greater=False)
                paper_adds_ap.append(hm_adds, 1)

            prev_gray = gray
            prev_uv_px = uv_px.copy()

        gt_uv_arr = np.asarray(gt_uv_seq, dtype=np.float32)
        pred_uv_arr = np.asarray(pred_uv_seq, dtype=np.float32)

        jitter_px = None
        center_jitter_px = None
        if len(pred_uv_arr) >= 2 and len(gt_uv_arr) == len(pred_uv_arr):
            pred_d = pred_uv_arr[1:] - pred_uv_arr[:-1]
            gt_d = gt_uv_arr[1:] - gt_uv_arr[:-1]
            res = pred_d - gt_d
            jitter_px = float(np.linalg.norm(res, axis=2).mean())
            center_jitter_px = float(np.linalg.norm(res[:, 0, :], axis=1).mean())

        jitter_3d = None
        if len(pred_3d_seq) >= 2 and len(pred_3d_seq) == len(gt_3d_seq):
            pred_3d_arr = np.asarray(pred_3d_seq, dtype=np.float32)
            gt_3d_arr = np.asarray(gt_3d_seq, dtype=np.float32)
            res3d = (pred_3d_arr[1:] - pred_3d_arr[:-1]) - (gt_3d_arr[1:] - gt_3d_arr[:-1])
            jitter_3d = float(np.linalg.norm(res3d, axis=2).mean())

        def smean(x):
            return float(np.mean(x)) if len(x) > 0 else None

        def sq(x, q):
            return float(np.quantile(x, q)) if len(x) > 0 else None

        row = {
            "video_dir": str(video_dir),
            "frames_eval": int(len(pred_uv_seq)),
            "paper_eval_frames": int(paper_eval_frames),
            "paper_detected_frames": int(paper_detected_frames),
            "score_mean": smean(score_vals),
            "score_min": float(np.min(score_vals)) if score_vals else None,
            "score_max": float(np.max(score_vals)) if score_vals else None,
            "pos_rate": smean(is_pos_vals),
            "reproj_px_mean": smean(reproj_px_vals),
            "reproj_px_p50": sq(reproj_px_vals, 0.5),
            "reproj_px_p90": sq(reproj_px_vals, 0.9),
            "pixel_error_objron_mean": smean(pixel_error_objron_vals),
            "pixel_error_objron_p50": sq(pixel_error_objron_vals, 0.5),
            "pixel_error_objron_p90": sq(pixel_error_objron_vals, 0.9),
            "pnp_reproj_px_mean": smean(pnp_reproj_vals),
            "azimuth_err_mean": smean(azimuth_err_vals),
            "polar_err_mean": smean(polar_err_vals),
            "add_mean": smean(add_vals),
            "adds_mean": smean(adds_vals),
            "iou3d_mean": smean(iou_3d_vals),
            "iou3d_p50": sq(iou_3d_vals, 0.5),
            "iou3d_p90": sq(iou_3d_vals, 0.9),
            "iou3d_sym_mean": smean(iou_3d_sym_vals),
            "iou3d_sym_p50": sq(iou_3d_sym_vals, 0.5),
            "iou3d_sym_p90": sq(iou_3d_sym_vals, 0.9),
            "jitter_px": jitter_px,
            "center_jitter_px": center_jitter_px,
            "jitter_3d": jitter_3d,
            "valid_3d_frames": int(valid_3d_count),
            "expected_label": expected_label,
        }
        rows.append(row)
        print(
            f"{video_dir.name:20s} | frames={row['frames_eval']:4d} "
            f"reproj={row['reproj_px_mean'] if row['reproj_px_mean'] is not None else float('nan'):.2f}px "
            f"iou3d={row['iou3d_mean'] if row['iou3d_mean'] is not None else float('nan'):.3f} "
            f"iou3d_sym={row['iou3d_sym_mean'] if row['iou3d_sym_mean'] is not None else float('nan'):.3f} "
            f"jitter={row['jitter_px'] if row['jitter_px'] is not None else float('nan'):.2f}px"
        )

    # Aggregate
    def agg_mean(key: str):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    agg = {
        "videos": len(rows),
        "paper_eval_frames": int(paper_eval_frames_total),
        "paper_detected_frames": int(paper_detected_frames_total),
        "reproj_px_mean": agg_mean("reproj_px_mean"),
        "pixel_error_objron_mean": agg_mean("pixel_error_objron_mean"),
        "azimuth_err_mean": agg_mean("azimuth_err_mean"),
        "polar_err_mean": agg_mean("polar_err_mean"),
        "add_mean": agg_mean("add_mean"),
        "adds_mean": agg_mean("adds_mean"),
        "iou3d_mean": agg_mean("iou3d_mean"),
        "iou3d_sym_mean": agg_mean("iou3d_sym_mean"),
        "jitter_px_mean": agg_mean("jitter_px"),
        "center_jitter_px_mean": agg_mean("center_jitter_px"),
        "jitter_3d_mean": agg_mean("jitter_3d"),
        "cls_tp": global_cls["tp"],
        "cls_fp": global_cls["fp"],
        "cls_tn": global_cls["tn"],
        "cls_fn": global_cls["fn"],
    }
    denom = global_cls["tp"] + global_cls["tn"] + global_cls["fp"] + global_cls["fn"]
    if denom > 0:
        acc = (global_cls["tp"] + global_cls["tn"]) / denom
        prec = global_cls["tp"] / max(1, (global_cls["tp"] + global_cls["fp"]))
        rec = global_cls["tp"] / max(1, (global_cls["tp"] + global_cls["fn"]))
        f1 = 2 * prec * rec / max(1e-9, (prec + rec))
        agg.update({
            "cls_acc": float(acc),
            "cls_prec": float(prec),
            "cls_rec": float(rec),
            "cls_f1": float(f1),
        })

    paper_curves = {
        "iou_thresholds": paper_iou_thresholds.tolist(),
        "pixel_thresholds": paper_pixel_thresholds.tolist(),
        "azimuth_thresholds": paper_azimuth_thresholds.tolist(),
        "polar_thresholds": paper_polar_thresholds.tolist(),
        "add_thresholds": paper_add_thresholds.tolist(),
        "adds_thresholds": paper_adds_thresholds.tolist(),
        "iou_ap": paper_iou_ap.compute_curve(),
        "pixel_ap": paper_pixel_ap.compute_curve(),
        "azimuth_ap": paper_azimuth_ap.compute_curve(),
        "polar_ap": paper_polar_ap.compute_curve(),
        "add_ap": paper_add_ap.compute_curve(),
        "adds_ap": paper_adds_ap.compute_curve(),
    }
    agg.update(
        {
            "paper_iou_ap_0_5": threshold_value(paper_iou_thresholds, paper_curves["iou_ap"], 0.5),
            "paper_azimuth_ap_15": threshold_value(paper_azimuth_thresholds, paper_curves["azimuth_ap"], 15.0),
            "paper_polar_ap_10": threshold_value(paper_polar_thresholds, paper_curves["polar_ap"], 10.0),
            "paper_pixel_ap_5": threshold_value(paper_pixel_thresholds, paper_curves["pixel_ap"], 5.0),
            "paper_add_ap_0_1": threshold_value(paper_add_thresholds, paper_curves["add_ap"], 0.1),
            "paper_adds_ap_0_1": threshold_value(paper_adds_thresholds, paper_curves["adds_ap"], 0.1),
        }
    )

    out_json = args.out_dir / "per_video_metrics.json"
    out_json.write_text(json.dumps({"aggregate": agg, "paper_curves": paper_curves, "videos": rows}, indent=2))

    out_csv = args.out_dir / "per_video_metrics.csv"
    keys = [
        "video_dir", "frames_eval", "paper_eval_frames", "paper_detected_frames",
        "score_mean", "score_min", "score_max", "pos_rate",
        "reproj_px_mean", "reproj_px_p50", "reproj_px_p90",
        "pixel_error_objron_mean", "pixel_error_objron_p50", "pixel_error_objron_p90",
        "pnp_reproj_px_mean",
        "azimuth_err_mean", "polar_err_mean", "add_mean", "adds_mean",
        "iou3d_mean", "iou3d_p50", "iou3d_p90",
        "iou3d_sym_mean", "iou3d_sym_p50", "iou3d_sym_p90",
        "jitter_px", "center_jitter_px", "jitter_3d", "valid_3d_frames", "expected_label",
    ]
    lines = [",".join(keys)]
    for r in rows:
        vals = []
        for k in keys:
            v = r.get(k, None)
            vals.append("" if v is None else str(v))
        lines.append(",".join(vals))
    out_csv.write_text("\n".join(lines) + "\n")

    agg_txt = args.out_dir / "summary.txt"
    summary_lines = [
        f"videos={agg['videos']}",
        f"paper_eval_frames={agg.get('paper_eval_frames')}",
        f"paper_detected_frames={agg.get('paper_detected_frames')}",
        f"reproj_px_mean={agg.get('reproj_px_mean')}",
        f"pixel_error_objron_mean={agg.get('pixel_error_objron_mean')}",
        f"azimuth_err_mean={agg.get('azimuth_err_mean')}",
        f"polar_err_mean={agg.get('polar_err_mean')}",
        f"add_mean={agg.get('add_mean')}",
        f"adds_mean={agg.get('adds_mean')}",
        f"iou3d_mean={agg.get('iou3d_mean')}",
        f"iou3d_sym_mean={agg.get('iou3d_sym_mean')}",
        f"jitter_px_mean={agg.get('jitter_px_mean')}",
        f"center_jitter_px_mean={agg.get('center_jitter_px_mean')}",
        f"jitter_3d_mean={agg.get('jitter_3d_mean')}",
        f"paper_iou_ap_0_5={agg.get('paper_iou_ap_0_5')}",
        f"paper_azimuth_ap_15={agg.get('paper_azimuth_ap_15')}",
        f"paper_polar_ap_10={agg.get('paper_polar_ap_10')}",
        f"paper_pixel_ap_5={agg.get('paper_pixel_ap_5')}",
        f"paper_add_ap_0_1={agg.get('paper_add_ap_0_1')}",
        f"paper_adds_ap_0_1={agg.get('paper_adds_ap_0_1')}",
        f"cls_tp={agg.get('cls_tp')} cls_fp={agg.get('cls_fp')} cls_tn={agg.get('cls_tn')} cls_fn={agg.get('cls_fn')}",
        f"cls_acc={agg.get('cls_acc')} cls_prec={agg.get('cls_prec')} cls_rec={agg.get('cls_rec')} cls_f1={agg.get('cls_f1')}",
    ]
    agg_txt.write_text("\n".join(summary_lines) + "\n")
    print(f"Wrote: {out_json}")
    print(f"Wrote: {out_csv}")
    print(f"Wrote: {agg_txt}")


if __name__ == "__main__":
    main()
