from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import timm
import torch
import torch.nn as nn


OBJECTRON_LOCAL_TEMPLATE = np.array(
    [
        [0.0, 0.5, 0.0],
        [-0.5, 0.0, +0.5],
        [+0.5, 0.0, +0.5],
        [+0.5, 1.0, +0.5],
        [-0.5, 1.0, +0.5],
        [-0.5, 0.0, -0.5],
        [+0.5, 0.0, -0.5],
        [+0.5, 1.0, -0.5],
        [-0.5, 1.0, -0.5],
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
        [0, 1, 2, 3, 4, 5, 6, 7, 8],
        [0, 2, 6, 7, 3, 1, 5, 8, 4],
        [0, 6, 5, 8, 7, 2, 1, 4, 3],
        [0, 5, 1, 4, 8, 6, 2, 3, 7],
    ],
    dtype=np.int64,
)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


@dataclass
class ObjectronFrameSample:
    image_path: Path
    image_name: str
    source_id: str
    label: int
    keypoints_uv: np.ndarray | None
    keypoints_w: np.ndarray | None
    intrinsics: dict | None
    width: int
    height: int


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_images_dir(base: Path) -> Path:
    candidates = [
        base / "frames_rotated",
        base / "rgb",
        base / "overlays",
        base / "objectron_prep" / "overlays",
        base,
    ]
    for c in candidates:
        if c.exists():
            if list(c.glob("*.jpg")) or list(c.glob("*.png")):
                return c
    raise ValueError(f"Could not find images in {base}")


def load_annotations(annotations_json: Path) -> list:
    if not annotations_json.exists():
        raise FileNotFoundError(f"Missing annotations file: {annotations_json}")
    return json.loads(annotations_json.read_text())


def load_positive_samples(pos_dir: Path) -> List[ObjectronFrameSample]:
    annotations_json = pos_dir / "objectron_prep" / "annotations.json"
    annotations = load_annotations(annotations_json)
    images_dir = resolve_images_dir(pos_dir)
    samples: List[ObjectronFrameSample] = []
    missing = 0
    malformed = 0
    for ann in annotations:
        image_name = ann.get("image", "")
        image_path = images_dir / image_name
        if not image_path.exists():
            missing += 1
            continue
        intr = ann.get("camera_intrinsics", {})
        k2d = ann.get("keypoints_2d")
        if (
            k2d is None
            or len(k2d) != 9
            or "image_width" not in intr
            or "image_height" not in intr
        ):
            malformed += 1
            continue
        kvis = ann.get("keypoints_2d_visibility", ann.get("visibility"))
        if kvis is not None and len(kvis) == 9:
            keypoints_w = np.asarray(kvis, dtype=np.float32)
        else:
            keypoints_w = np.ones((9,), dtype=np.float32)
        samples.append(
            ObjectronFrameSample(
                image_path=image_path,
                image_name=image_name,
                source_id=pos_dir.name,
                label=1,
                keypoints_uv=np.asarray(k2d, dtype=np.float32)[:, :2],
                keypoints_w=np.clip(keypoints_w, 0.0, 1.0),
                intrinsics=intr,
                width=int(intr["image_width"]),
                height=int(intr["image_height"]),
            )
        )
    if not samples:
        raise ValueError(f"No valid positive samples in {pos_dir}")
    print(f"[POS] {pos_dir.name}: {len(samples)} samples (missing={missing}, malformed={malformed})")
    return samples


def load_negative_samples(neg_dir: Path) -> List[ObjectronFrameSample]:
    annotations_json = None
    if (neg_dir / "objectron_prep" / "annotations.json").exists():
        annotations_json = neg_dir / "objectron_prep" / "annotations.json"
    elif (neg_dir / "annotations.json").exists():
        annotations_json = neg_dir / "annotations.json"

    images_dir = resolve_images_dir(neg_dir)
    image_names: List[str] = []
    if annotations_json is not None:
        for ann in load_annotations(annotations_json):
            name = ann.get("image", "")
            if name:
                image_names.append(name)
    else:
        image_names = [p.name for p in sorted(images_dir.glob("*.jpg"))]
        image_names += [p.name for p in sorted(images_dir.glob("*.png"))]

    samples: List[ObjectronFrameSample] = []
    for image_name in image_names:
        image_path = images_dir / image_name
        if not image_path.exists():
            continue
        samples.append(
            ObjectronFrameSample(
                image_path=image_path,
                image_name=image_name,
                source_id=neg_dir.name,
                label=0,
                keypoints_uv=None,
                keypoints_w=None,
                intrinsics=None,
                width=0,
                height=0,
            )
        )
    if not samples:
        raise ValueError(f"No valid negative samples in {neg_dir}")
    print(f"[NEG] {neg_dir.name}: {len(samples)} samples")
    return samples


def split_indices_stratified_by_source(
    samples: List[ObjectronFrameSample],
    pos_idx: List[int],
    neg_idx: List[int],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[List[int], List[int], List[int]]:
    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1.0:
        raise ValueError("Require train_ratio > 0, val_ratio > 0 and train_ratio + val_ratio < 1.")

    rng = random.Random(seed)

    def split_grouped(indices: List[int]) -> Tuple[List[int], List[int], List[int]]:
        groups: Dict[str, List[int]] = {}
        for idx in indices:
            groups.setdefault(samples[idx].source_id, []).append(idx)
        group_ids = list(groups.keys())
        rng.shuffle(group_ids)
        n = len(group_ids)
        if n == 0:
            return [], [], []
        if n == 1:
            return groups[group_ids[0]], [], []

        n_train = max(1, int(round(train_ratio * n)))
        n_val = int(round(val_ratio * n))
        if n >= 3:
            n_train = min(n_train, n - 2)
            n_val = max(1, n_val)
        else:
            n_val = max(0, n_val)

        if n_train + n_val >= n:
            n_val = max(0, n - n_train - 1)
        if n_train >= n:
            n_train = n - 1

        train_g = group_ids[:n_train]
        val_g = group_ids[n_train:n_train + n_val]
        test_g = group_ids[n_train + n_val:]
        if len(test_g) == 0 and len(val_g) > 1:
            test_g = [val_g[-1]]
            val_g = val_g[:-1]
        if len(test_g) == 0 and len(train_g) > 1:
            test_g = [train_g[-1]]
            train_g = train_g[:-1]

        train = [idx for gid in train_g for idx in groups[gid]]
        val = [idx for gid in val_g for idx in groups[gid]]
        test = [idx for gid in test_g for idx in groups[gid]]
        return train, val, test

    pos_train, pos_val, pos_test = split_grouped(pos_idx)
    neg_train, neg_val, neg_test = split_grouped(neg_idx)
    train_idx = pos_train + neg_train
    val_idx = pos_val + neg_val
    test_idx = pos_test + neg_test
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx


def summarize_split(name: str, samples: Sequence[ObjectronFrameSample], indices: Sequence[int]) -> None:
    n = len(indices)
    n_pos = sum(1 for idx in indices if samples[idx].label == 1)
    n_neg = n - n_pos
    videos = len(set(samples[idx].source_id for idx in indices))
    print(f"{name:>5} | samples={n} pos={n_pos} neg={n_neg} videos={videos}")


def clip_box_xyxy(box_xyxy: np.ndarray, width: int, height: int) -> np.ndarray:
    x0, y0, x1, y1 = box_xyxy.astype(np.float32).tolist()
    x0 = float(np.clip(x0, 0.0, max(0.0, width - 2.0)))
    y0 = float(np.clip(y0, 0.0, max(0.0, height - 2.0)))
    x1 = float(np.clip(x1, x0 + 2.0, float(width)))
    y1 = float(np.clip(y1, y0 + 2.0, float(height)))
    return np.array([x0, y0, x1, y1], dtype=np.float32)


def square_crop_box_from_keypoints(
    keypoints_uv: np.ndarray,
    width: int,
    height: int,
    context: float,
    min_size: int,
) -> np.ndarray:
    points = np.asarray(keypoints_uv, dtype=np.float32)
    if points.shape[0] >= 9:
        points = points[1:]
    xs = np.clip(points[:, 0] * width, 0.0, float(width - 1))
    ys = np.clip(points[:, 1] * height, 0.0, float(height - 1))
    x0, x1 = float(xs.min()), float(xs.max())
    y0, y1 = float(ys.min()), float(ys.max())
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    bw = max(2.0, x1 - x0)
    bh = max(2.0, y1 - y0)
    side = max(bw, bh) * max(1.0, float(context))
    side = max(side, float(min_size))
    box = np.array(
        [cx - 0.5 * side, cy - 0.5 * side, cx + 0.5 * side, cy + 0.5 * side],
        dtype=np.float32,
    )
    return clip_box_xyxy(box, width=width, height=height)


def crop_and_resize(
    image_rgb: np.ndarray,
    box_xyxy: np.ndarray,
    output_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = image_rgb.shape[:2]
    x0, y0, x1, y1 = clip_box_xyxy(box_xyxy, width=width, height=height).tolist()
    x0i = int(np.floor(x0))
    y0i = int(np.floor(y0))
    x1i = int(np.ceil(x1))
    y1i = int(np.ceil(y1))
    x0i = max(0, min(width - 2, x0i))
    y0i = max(0, min(height - 2, y0i))
    x1i = max(x0i + 2, min(width, x1i))
    y1i = max(y0i + 2, min(height, y1i))
    crop = image_rgb[y0i:y1i, x0i:x1i]
    crop = cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_LINEAR)
    box = np.array([x0i, y0i, x1i, y1i], dtype=np.float32)
    scale = np.array([max(1.0, float(x1i - x0i)), max(1.0, float(y1i - y0i))], dtype=np.float32)
    return crop, box, scale


def keypoints_uv_to_px(keypoints_uv: np.ndarray, width: int, height: int) -> np.ndarray:
    pts = np.asarray(keypoints_uv, dtype=np.float32)
    return np.stack([pts[:, 0] * width, pts[:, 1] * height], axis=1)


def keypoints_px_to_crop_uv(
    keypoints_px: np.ndarray,
    box_xyxy: np.ndarray,
) -> np.ndarray:
    x0, y0, x1, y1 = box_xyxy.astype(np.float32).tolist()
    crop_w = max(1.0, x1 - x0)
    crop_h = max(1.0, y1 - y0)
    uv = keypoints_px.copy().astype(np.float32)
    uv[:, 0] = (uv[:, 0] - x0) / crop_w
    uv[:, 1] = (uv[:, 1] - y0) / crop_h
    return np.clip(uv, 0.0, 1.0)


def crop_uv_to_keypoints_px(
    crop_uv: np.ndarray,
    box_xyxy: np.ndarray,
) -> np.ndarray:
    x0, y0, x1, y1 = box_xyxy.astype(np.float32).tolist()
    crop_w = max(1.0, x1 - x0)
    crop_h = max(1.0, y1 - y0)
    uv = np.asarray(crop_uv, dtype=np.float32)
    return np.stack([x0 + uv[:, 0] * crop_w, y0 + uv[:, 1] * crop_h], axis=1)


def detector_image_to_tensor(image_rgb: np.ndarray) -> torch.Tensor:
    image = image_rgb.astype(np.float32) / 255.0
    return torch.from_numpy(image).permute(2, 0, 1)


def regressor_image_to_tensor(image_rgb: np.ndarray) -> torch.Tensor:
    image = image_rgb.astype(np.float32) / 255.0
    image = (image - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(image).permute(2, 0, 1)


class EfficientNetLiteRegressor(nn.Module):
    def __init__(self, backbone: str = "efficientnet_lite0", pretrained: bool = False, dropout: float = 0.0):
        super().__init__()
        self.encoder = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        num_features = getattr(self.encoder, "num_features", None)
        if num_features is None:
            raise RuntimeError(f"Could not infer feature size for backbone: {backbone}")
        self.head = nn.Sequential(
            nn.Dropout(p=float(max(0.0, dropout))),
            nn.Linear(int(num_features), 18),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x)
        if feat.ndim > 2:
            feat = torch.flatten(feat, 1)
        return self.head(feat).view(-1, 9, 2)


def draw_points(image_bgr: np.ndarray, pts_px: np.ndarray, color: Tuple[int, int, int] = (255, 0, 0)) -> None:
    for idx, pt in enumerate(pts_px):
        pt_color = (0, 0, 255) if idx == 0 else color
        cv2.circle(image_bgr, (int(pt[0]), int(pt[1])), 4, pt_color, -1, cv2.LINE_AA)
        cv2.putText(
            image_bgr,
            str(idx),
            (int(pt[0]) + 4, int(pt[1]) + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def draw_cuboid(image_bgr: np.ndarray, pts_px: np.ndarray, color: Tuple[int, int, int] = (0, 255, 0)) -> None:
    for i, j in CUBOID_EDGES:
        p0 = pts_px[i]
        p1 = pts_px[j]
        cv2.line(
            image_bgr,
            (int(p0[0]), int(p0[1])),
            (int(p1[0]), int(p1[1])),
            color,
            2,
            cv2.LINE_AA,
        )


def build_intrinsics_dict(annotations: list) -> Tuple[Dict[str, dict], dict]:
    by_name = {}
    fallback = None
    for ann in annotations:
        image_name = ann.get("image")
        intr = ann.get("camera_intrinsics")
        if image_name and intr is not None:
            by_name[image_name] = intr
            if fallback is None:
                fallback = intr
    if fallback is None:
        raise ValueError("No camera_intrinsics found in annotations.")
    return by_name, fallback


def canonical_from_data(canonical_3d: Path | None, annotations: list) -> np.ndarray:
    if canonical_3d is not None:
        if not canonical_3d.exists():
            raise FileNotFoundError(f"canonical_3d file not found: {canonical_3d}")
        pts = np.load(str(canonical_3d)).astype(np.float32)
        if pts.shape != (9, 3):
            raise ValueError(f"canonical_3d must have shape (9, 3), got {pts.shape}")
        return pts

    scale = np.asarray(annotations[0]["pose_9dof"]["scale"], dtype=np.float32)
    center_shift = np.array([0.0, 0.5, 0.0], dtype=np.float32)
    return (OBJECTRON_LOCAL_TEMPLATE - center_shift[None, :]) * scale[None, :]


def solve_pose_epnp(canonical_3d: np.ndarray, uv_px: np.ndarray, intr: dict):
    fx, fy = float(intr["fx"]), float(intr["fy"])
    cx, cy = float(intr["cx"]), float(intr["cy"])
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    dist = np.array(intr.get("dist", [0.0, 0.0, 0.0, 0.0]), dtype=np.float64).reshape(-1, 1)
    obj = canonical_3d[1:].astype(np.float64)
    img = uv_px[1:].astype(np.float64)
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_EPNP)
    if not ok:
        return False, None, None, None, None
    reproj, _ = cv2.projectPoints(canonical_3d.astype(np.float64), rvec, tvec, K, dist)
    reproj = reproj.reshape(-1, 2)
    reproj_err = float(np.mean(np.linalg.norm(reproj[1:] - uv_px[1:], axis=1)))
    return True, rvec.reshape(3), tvec.reshape(3), reproj.astype(np.float32), reproj_err


def solve_pose_with_symmetry(
    canonical_3d: np.ndarray,
    uv_px: np.ndarray,
    intr: dict,
    last_reproj: np.ndarray | None = None,
    pnp_max_err: float = 0.0,
    use_symmetry: bool = True,
    temporal_weight: float = 0.0,
):
    perms = SYM_PERMS_YAW4 if use_symmetry else SYM_PERMS_YAW4[:1]
    best = None
    best_score = None
    for perm in perms:
        permuted = uv_px[perm]
        ok, rvec, tvec, reproj, reproj_err = solve_pose_epnp(canonical_3d, permuted, intr)
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


def load_twostage_checkpoint(checkpoint_path: Path, device: torch.device | str = "cpu") -> dict:
    checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unexpected checkpoint format in {checkpoint_path}")
    if "detector_state" not in checkpoint or "regressor_state" not in checkpoint:
        raise ValueError(f"Checkpoint does not look like a paper two-stage checkpoint: {checkpoint_path}")
    return checkpoint


def write_lines(path: Path, values: Iterable[str]) -> None:
    path.write_text("".join(f"{v}\n" for v in values))
