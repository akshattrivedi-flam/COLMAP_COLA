import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm


@dataclass
class SamplePN:
    image_path: Path
    image_name: str
    source_id: str  # usually video folder name, used for leakage-safe split
    label: int  # 1=positive, 0=negative
    keypoints_uv: np.ndarray | None  # (9, 2) normalized [0,1] for positives
    keypoints_w: np.ndarray | None  # (9,) visibility/weight for positives
    width: int
    height: int


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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


def load_positive_samples(pos_dir: Path) -> List[SamplePN]:
    annotations_json = pos_dir / "objectron_prep" / "annotations.json"
    if not annotations_json.exists():
        raise ValueError(f"Missing annotations.json in {pos_dir}")
    images_dir = resolve_images_dir(pos_dir)
    annotations = json.loads(annotations_json.read_text())
    samples: List[SamplePN] = []
    missing = 0
    malformed = 0
    for ann in annotations:
        image_name = ann.get("image", "")
        image_path = images_dir / image_name
        if not image_path.exists():
            missing += 1
            continue
        k2d = ann.get("keypoints_2d", None)
        kvis = ann.get("keypoints_2d_visibility", None)
        intr = ann.get("camera_intrinsics", {})
        if k2d is None or len(k2d) != 9 or "image_width" not in intr or "image_height" not in intr:
            malformed += 1
            continue
        k2d = np.asarray(k2d, dtype=np.float32)
        keypoints_uv = k2d[:, :2]
        if kvis is not None and len(kvis) == 9:
            keypoints_w = np.asarray(kvis, dtype=np.float32)
        else:
            keypoints_w = np.ones((9,), dtype=np.float32)
        keypoints_w = np.clip(keypoints_w, 0.0, 1.0)
        samples.append(
            SamplePN(
                image_path=image_path,
                image_name=image_name,
                source_id=pos_dir.name,
                label=1,
                keypoints_uv=keypoints_uv,
                keypoints_w=keypoints_w,
                width=int(intr["image_width"]),
                height=int(intr["image_height"]),
            )
        )
    if not samples:
        raise ValueError(f"No valid positive samples in {pos_dir}")
    print(f"[POS] {pos_dir.name}: {len(samples)} samples (missing={missing}, malformed={malformed})")
    return samples


def load_negative_samples(neg_dir: Path) -> List[SamplePN]:
    annotations_json = None
    if (neg_dir / "objectron_prep" / "annotations.json").exists():
        annotations_json = neg_dir / "objectron_prep" / "annotations.json"
    elif (neg_dir / "annotations.json").exists():
        annotations_json = neg_dir / "annotations.json"

    images_dir = resolve_images_dir(neg_dir)

    image_names = []
    if annotations_json is not None:
        annotations = json.loads(annotations_json.read_text())
        for ann in annotations:
            name = ann.get("image", "")
            if name:
                image_names.append(name)
    else:
        image_names = [p.name for p in sorted(images_dir.glob("*.jpg"))]
        image_names += [p.name for p in sorted(images_dir.glob("*.png"))]

    samples: List[SamplePN] = []
    for name in image_names:
        p = images_dir / name
        if not p.exists():
            continue
        samples.append(
            SamplePN(
                image_path=p,
                image_name=name,
                source_id=neg_dir.name,
                label=0,
                keypoints_uv=None,
                keypoints_w=None,
                width=0,
                height=0,
            )
        )
    if not samples:
        raise ValueError(f"No valid negative samples in {neg_dir}")
    print(f"[NEG] {neg_dir.name}: {len(samples)} samples")
    return samples


def split_indices_stratified(pos_idx: List[int], neg_idx: List[int], train_ratio: float, val_ratio: float, seed: int):
    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1.0:
        raise ValueError("Require train_ratio > 0, val_ratio > 0 and train_ratio + val_ratio < 1.")

    rng = random.Random(seed)
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)

    def split_group(idx: List[int]):
        n = len(idx)
        n_train = max(1, int(round(train_ratio * n)))
        n_val = max(1, int(round(val_ratio * n)))
        n_train = min(n_train, n - 2)
        n_val = min(n_val, n - n_train - 1)
        n_test = n - n_train - n_val
        if n_test <= 0:
            n_val = max(1, n_val - 1)
            n_test = n - n_train - n_val
        return idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]

    pos_train, pos_val, pos_test = split_group(pos_idx)
    neg_train, neg_val, neg_test = split_group(neg_idx)

    train_idx = pos_train + neg_train
    val_idx = pos_val + neg_val
    test_idx = pos_test + neg_test
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx


def split_indices_stratified_by_source(
    samples: List[SamplePN],
    pos_idx: List[int],
    neg_idx: List[int],
    train_ratio: float,
    val_ratio: float,
    seed: int,
):
    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1.0:
        raise ValueError("Require train_ratio > 0, val_ratio > 0 and train_ratio + val_ratio < 1.")

    rng = random.Random(seed)

    def split_grouped(idx: List[int]):
        groups = {}
        for i in idx:
            sid = samples[i].source_id
            groups.setdefault(sid, []).append(i)
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

        train = [i for g in train_g for i in groups[g]]
        val = [i for g in val_g for i in groups[g]]
        test = [i for g in test_g for i in groups[g]]
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


class CanPosNegDataset(Dataset):
    def __init__(
        self,
        samples: List[SamplePN],
        image_size: int,
        augment: bool = False,
        roi_train: bool = False,
        roi_context: float = 1.45,
        roi_jitter: float = 0.08,
        roi_min_size: int = 96,
        roi_prob: float = 0.75,
        rot_max_deg: float = 180.0,
    ):
        self.samples = samples
        self.image_size = image_size
        self.augment = augment
        self.roi_train = roi_train
        self.roi_context = float(roi_context)
        self.roi_jitter = float(roi_jitter)
        self.roi_min_size = int(roi_min_size)
        self.roi_prob = float(np.clip(roi_prob, 0.0, 1.0))
        self.rot_max_deg = float(max(0.0, rot_max_deg))
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

    def _apply_augment(
        self,
        image: np.ndarray,
        keypoints_uv: np.ndarray | None,
    ) -> Tuple[np.ndarray, np.ndarray | None]:
        h, w = image.shape[:2]
        k = None if keypoints_uv is None else keypoints_uv.copy().astype(np.float32)

        # Color jitter (RGB gain/bias + HSV hue/sat/value variation)
        if random.random() < 0.8:
            gain = random.uniform(0.80, 1.25)
            bias = random.uniform(-0.10, 0.10)
            image = np.clip(image.astype(np.float32) / 255.0 * gain + bias, 0.0, 1.0)
            image = (image * 255.0).astype(np.uint8)
        if random.random() < 0.7:
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[..., 0] = (hsv[..., 0] + random.uniform(-12.0, 12.0)) % 180.0
            hsv[..., 1] = np.clip(hsv[..., 1] * random.uniform(0.75, 1.35), 0.0, 255.0)
            hsv[..., 2] = np.clip(hsv[..., 2] * random.uniform(0.75, 1.35), 0.0, 255.0)
            image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        # Mild Gaussian blur
        if random.random() < 0.25:
            ksz = random.choice([3, 5])
            image = cv2.GaussianBlur(image, (ksz, ksz), sigmaX=0.0)

        # Random affine (rotation + scale + translation)
        if random.random() < 0.7:
            angle = random.uniform(-self.rot_max_deg, self.rot_max_deg)
            scale = random.uniform(0.85, 1.15)
            tx = random.uniform(-0.12, 0.12) * w
            ty = random.uniform(-0.12, 0.12) * h
            M = cv2.getRotationMatrix2D((w * 0.5, h * 0.5), angle, scale)
            M[0, 2] += tx
            M[1, 2] += ty
            image = cv2.warpAffine(
                image,
                M,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )
            if k is not None:
                pts = np.stack([k[:, 0] * w, k[:, 1] * h, np.ones((9,), dtype=np.float32)], axis=1)
                pts2 = (M @ pts.T).T
                k[:, 0] = np.clip(pts2[:, 0] / max(1.0, float(w)), 0.0, 1.0)
                k[:, 1] = np.clip(pts2[:, 1] / max(1.0, float(h)), 0.0, 1.0)

        # Horizontal flip + keypoint re-indexing for left/right corners
        if random.random() < 0.5:
            image = np.ascontiguousarray(image[:, ::-1, :])
            if k is not None:
                k[:, 0] = 1.0 - k[:, 0]
                # center unchanged, swap left/right cuboid corners
                remap = np.array([0, 2, 1, 4, 3, 6, 5, 8, 7], dtype=np.int64)
                k = k[remap]

        return image, k

    def _build_roi_bbox(self, keypoints_uv: np.ndarray, w: int, h: int) -> np.ndarray:
        pts = keypoints_uv[1:] if keypoints_uv.shape[0] >= 9 else keypoints_uv
        xs = np.clip(pts[:, 0] * w, 0.0, float(w - 1))
        ys = np.clip(pts[:, 1] * h, 0.0, float(h - 1))
        x0, x1 = float(xs.min()), float(xs.max())
        y0, y1 = float(ys.min()), float(ys.max())
        cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        bw = max(2.0, x1 - x0)
        bh = max(2.0, y1 - y0)
        side = max(bw, bh) * max(1.0, self.roi_context)
        side = max(side, float(self.roi_min_size))

        if self.augment and self.roi_jitter > 0.0:
            jitter = self.roi_jitter
            cx += np.random.uniform(-jitter, jitter) * side
            cy += np.random.uniform(-jitter, jitter) * side
            side *= np.random.uniform(1.0 - jitter, 1.0 + jitter)

        x0 = cx - 0.5 * side
        y0 = cy - 0.5 * side
        x1 = cx + 0.5 * side
        y1 = cy + 0.5 * side

        # Keep crop inside image bounds by translation.
        if x0 < 0:
            x1 -= x0
            x0 = 0.0
        if y0 < 0:
            y1 -= y0
            y0 = 0.0
        if x1 > w:
            x0 -= (x1 - w)
            x1 = float(w)
        if y1 > h:
            y0 -= (y1 - h)
            y1 = float(h)
        x0 = max(0.0, x0)
        y0 = max(0.0, y0)
        x1 = min(float(w), x1)
        y1 = min(float(h), y1)
        if x1 - x0 < 2.0:
            x1 = min(float(w), x0 + 2.0)
        if y1 - y0 < 2.0:
            y1 = min(float(h), y0 + 2.0)
        return np.array([x0, y0, x1, y1], dtype=np.float32)

    def _crop_roi(self, image: np.ndarray, keypoints_uv: np.ndarray, bbox_xyxy: np.ndarray):
        h, w = image.shape[:2]
        x0, y0, x1, y1 = bbox_xyxy.tolist()
        x0i = int(np.floor(x0))
        y0i = int(np.floor(y0))
        x1i = int(np.ceil(x1))
        y1i = int(np.ceil(y1))
        x0i = max(0, min(w - 2, x0i))
        y0i = max(0, min(h - 2, y0i))
        x1i = max(x0i + 2, min(w, x1i))
        y1i = max(y0i + 2, min(h, y1i))
        crop = image[y0i:y1i, x0i:x1i]
        crop_h, crop_w = crop.shape[:2]
        crop = cv2.resize(crop, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)

        kp = keypoints_uv.copy().astype(np.float32)
        kp_px = np.stack([kp[:, 0] * w, kp[:, 1] * h], axis=1)
        kp_px[:, 0] = (kp_px[:, 0] - x0i) / max(1.0, float(crop_w))
        kp_px[:, 1] = (kp_px[:, 1] - y0i) / max(1.0, float(crop_h))
        kp_px = np.clip(kp_px, 0.0, 1.0)
        return crop, kp_px

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        image = cv2.imread(str(s.image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to read image: {s.image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h0, w0 = image.shape[:2]
        keypoints_uv = None if s.keypoints_uv is None else s.keypoints_uv.copy().astype(np.float32)
        keypoints_w = None if s.keypoints_w is None else s.keypoints_w.copy().astype(np.float32)
        if self.augment:
            image, keypoints_uv = self._apply_augment(image, keypoints_uv)
        use_roi = (
            self.roi_train
            and keypoints_uv is not None
            and (random.random() < self.roi_prob)
        )
        if use_roi:
            h, w = image.shape[:2]
            bbox = self._build_roi_bbox(keypoints_uv, w=w, h=h)
            image, keypoints_uv = self._crop_roi(image, keypoints_uv, bbox)
            dims = torch.tensor([self.image_size, self.image_size], dtype=torch.float32)
        else:
            image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
            dims = torch.tensor([w0, h0], dtype=torch.float32)
        image = image.astype(np.float32) / 255.0
        image = (image - self.mean) / self.std
        image = torch.from_numpy(image).permute(2, 0, 1)

        if keypoints_uv is None:
            target_uv = torch.zeros((9, 2), dtype=torch.float32)
            target_w = torch.zeros((9,), dtype=torch.float32)
            has_kpt = torch.tensor(0.0, dtype=torch.float32)
        else:
            target_uv = torch.from_numpy(keypoints_uv.astype(np.float32))
            if keypoints_w is None:
                target_w = torch.ones((9,), dtype=torch.float32)
            else:
                target_w = torch.from_numpy(np.clip(keypoints_w, 0.0, 1.0).astype(np.float32))
            has_kpt = torch.tensor(1.0, dtype=torch.float32)

        label = torch.tensor(float(s.label), dtype=torch.float32)
        return {
            "image": image,
            "target_uv": target_uv,
            "target_w": target_w,
            "dims_wh": dims,
            "has_kpt": has_kpt,
            "label": label,
            "image_name": s.image_name,
        }


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
    def __init__(self, backbone: str = "simple", pretrained: bool = False):
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


SYM_PERMS_YAW4 = np.array(
    [
        [0, 1, 2, 3, 4, 5, 6, 7, 8],  # identity
        [0, 2, 6, 7, 3, 1, 5, 8, 4],  # +90 deg yaw
        [0, 6, 5, 8, 7, 2, 1, 4, 3],  # +180 deg yaw
        [0, 5, 1, 4, 8, 6, 2, 3, 7],  # +270 deg yaw
    ],
    dtype=np.int64,
)


def _symmetry_aware_kpt_terms(pred_pos, target_pos, dims_pos, w_pos):
    perms = torch.as_tensor(SYM_PERMS_YAW4, dtype=torch.long, device=pred_pos.device)
    # [P, B, 9, 2]
    target_perm = target_pos[:, perms, :].permute(1, 0, 2, 3)

    # SmoothL1 per point, average over xy
    per_point_l1 = F.smooth_l1_loss(
        pred_pos.unsqueeze(0).expand(target_perm.shape[0], -1, -1, -1),
        target_perm,
        beta=0.02,
        reduction="none",
    ).mean(dim=-1)  # [P, B, 9]

    if w_pos is not None:
        w = torch.clamp(w_pos, 0.0, 1.0).unsqueeze(0)  # [1, B, 9]
        denom = torch.clamp(w.sum(dim=-1), min=1e-6)  # [1, B]
        per_perm = (per_point_l1 * w).sum(dim=-1) / denom  # [P, B]
    else:
        per_perm = per_point_l1.mean(dim=-1)  # [P, B]

    best_perm = torch.argmin(per_perm, dim=0)  # [B]
    best_loss = per_perm[best_perm, torch.arange(pred_pos.shape[0], device=pred_pos.device)]  # [B]
    kp_loss = best_loss.mean()

    # Gather aligned target for reporting UV/PX errors.
    chosen = target_perm[best_perm, torch.arange(pred_pos.shape[0], device=pred_pos.device)]  # [B, 9, 2]
    diff = pred_pos - chosen
    err = torch.linalg.norm(diff, dim=-1)  # [B, 9]
    if w_pos is not None:
        w = torch.clamp(w_pos, 0.0, 1.0)
        denom = torch.clamp(w.sum(dim=-1), min=1e-6)
        err_uv = ((err * w).sum(dim=-1) / denom).mean()
        err_px = (
            (torch.linalg.norm(diff * dims_pos.unsqueeze(1), dim=-1) * w).sum(dim=-1) / denom
        ).mean()
    else:
        err_uv = err.mean()
        err_px = torch.linalg.norm(diff * dims_pos.unsqueeze(1), dim=-1).mean()
    return kp_loss, err_uv, err_px


def compute_losses(
    pred_uv,
    target_uv,
    target_w,
    dims_wh,
    has_kpt,
    cls_logits,
    labels,
    kp_weight,
    bce_loss,
    symmetry_aware=False,
):
    cls_loss = bce_loss(cls_logits, labels)

    pos_mask = has_kpt > 0.5
    if pos_mask.any():
        pred_pos = pred_uv[pos_mask]
        tgt_pos = target_uv[pos_mask]
        w_pos = target_w[pos_mask]
        dims_pos = dims_wh[pos_mask]
        if symmetry_aware:
            kp_loss, err_uv, err_px = _symmetry_aware_kpt_terms(pred_pos, tgt_pos, dims_pos, w_pos)
        else:
            per_point = F.smooth_l1_loss(pred_pos, tgt_pos, beta=0.02, reduction="none").mean(dim=-1)  # [B, 9]
            w = torch.clamp(w_pos, 0.0, 1.0)
            denom = torch.clamp(w.sum(dim=-1), min=1e-6)
            kp_loss = ((per_point * w).sum(dim=-1) / denom).mean()

            diff = pred_pos - tgt_pos
            err = torch.linalg.norm(diff, dim=-1)
            err_uv = ((err * w).sum(dim=-1) / denom).mean()
            err_px = (
                (torch.linalg.norm(diff * dims_pos.unsqueeze(1), dim=-1) * w).sum(dim=-1) / denom
            ).mean()
    else:
        kp_loss = torch.tensor(0.0, device=pred_uv.device)
        err_uv = torch.tensor(0.0, device=pred_uv.device)
        err_px = torch.tensor(0.0, device=pred_uv.device)

    total = cls_loss + kp_weight * kp_loss
    return total, cls_loss, kp_loss, err_uv, err_px


def run_epoch(
    model,
    loader,
    device,
    optimizer=None,
    use_amp=False,
    scaler=None,
    kp_weight=1.0,
    bce_loss=None,
    symmetry_aware=False,
):
    is_train = optimizer is not None
    model.train(is_train)

    totals = {"loss": 0.0, "cls": 0.0, "kp": 0.0, "uv": 0.0, "px": 0.0, "acc": 0.0, "prec": 0.0, "rec": 0.0}
    count = 0

    iterator = tqdm(loader, leave=False)
    for batch in iterator:
        image = batch["image"].to(device, non_blocking=True)
        target_uv = batch["target_uv"].to(device, non_blocking=True)
        target_w = batch["target_w"].to(device, non_blocking=True)
        dims_wh = batch["dims_wh"].to(device, non_blocking=True)
        has_kpt = batch["has_kpt"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        with torch.set_grad_enabled(is_train):
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                pred_uv, cls_logits = model(image)
                loss, cls_loss, kp_loss, err_uv, err_px = compute_losses(
                    pred_uv,
                    target_uv,
                    target_w,
                    dims_wh,
                    has_kpt,
                    cls_logits,
                    labels,
                    kp_weight,
                    bce_loss,
                    symmetry_aware=symmetry_aware,
                )

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        probs = torch.sigmoid(cls_logits)
        preds = (probs >= 0.5).float()
        tp = ((preds == 1) & (labels == 1)).sum().item()
        fp = ((preds == 1) & (labels == 0)).sum().item()
        tn = ((preds == 0) & (labels == 0)).sum().item()
        fn = ((preds == 0) & (labels == 1)).sum().item()

        acc = (tp + tn) / max(1.0, tp + tn + fp + fn)
        prec = tp / max(1.0, tp + fp)
        rec = tp / max(1.0, tp + fn)

        bs = image.shape[0]
        totals["loss"] += float(loss.item()) * bs
        totals["cls"] += float(cls_loss.item()) * bs
        totals["kp"] += float(kp_loss.item()) * bs
        totals["uv"] += float(err_uv.item()) * bs
        totals["px"] += float(err_px.item()) * bs
        totals["acc"] += acc * bs
        totals["prec"] += prec * bs
        totals["rec"] += rec * bs
        count += bs

    for k in totals:
        totals[k] /= max(1, count)
    return totals


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos_dirs", nargs="+", required=True)
    ap.add_argument("--neg_dirs", nargs="+", required=True)
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--image_size", type=int, default=320)
    ap.add_argument("--backbone", type=str, default="simple", help="Only 'simple' is supported in this script")
    ap.add_argument("--pretrained", action="store_true", help="Ignored (kept for CLI compatibility)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--kp_weight", type=float, default=5.0)
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--split_mode", choices=["sample", "video"], default="video")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--balance", action="store_true", help="Use WeightedRandomSampler to balance classes")
    ap.add_argument("--auto_pos_weight", action="store_true", help="Set BCE pos_weight = n_neg / n_pos")
    ap.add_argument("--augment", action="store_true", help="Apply geometric/photometric augmentation on train split")
    ap.add_argument("--roi_train", action="store_true", help="Train positives on object-centered ROI crops")
    ap.add_argument("--roi_context", type=float, default=1.45, help="ROI expansion factor around 2D keypoints")
    ap.add_argument("--roi_jitter", type=float, default=0.08, help="ROI center/scale jitter during training")
    ap.add_argument("--roi_min_size", type=int, default=96, help="Minimum ROI side length in source pixels")
    ap.add_argument("--roi_prob", type=float, default=0.75, help="Probability of using ROI crop on positive train sample")
    ap.add_argument("--rot_max_deg", type=float, default=180.0, help="Max absolute random rotation in augmentation")
    ap.add_argument("--symmetry_aware", action="store_true", help="Use yaw-symmetry-invariant keypoint loss (recommended for cylindrical cans)")
    ap.add_argument("--init_checkpoint", type=Path, default=None, help="Optional checkpoint to continue training from")
    return ap.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)

    pos_samples: List[SamplePN] = []
    for p in args.pos_dirs:
        pos_samples += load_positive_samples(Path(p))

    neg_samples: List[SamplePN] = []
    for n in args.neg_dirs:
        neg_samples += load_negative_samples(Path(n))

    all_samples = pos_samples + neg_samples
    pos_idx = [i for i, s in enumerate(all_samples) if s.label == 1]
    neg_idx = [i for i, s in enumerate(all_samples) if s.label == 0]

    if not pos_idx or not neg_idx:
        raise ValueError("Need both positive and negative samples.")

    if args.split_mode == "video":
        train_idx, val_idx, test_idx = split_indices_stratified_by_source(
            all_samples, pos_idx, neg_idx, args.train_ratio, args.val_ratio, args.seed
        )
    else:
        train_idx, val_idx, test_idx = split_indices_stratified(
            pos_idx, neg_idx, args.train_ratio, args.val_ratio, args.seed
        )

    if not train_idx or not val_idx or not test_idx:
        raise ValueError(
            f"Split produced empty set(s): train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}. "
            "Adjust exclusions/ratios or use --split_mode sample."
        )

    def summarize_split(name: str, idx: List[int]):
        n = len(idx)
        n_pos = sum(1 for i in idx if all_samples[i].label == 1)
        n_neg = n - n_pos
        vids = len(set(all_samples[i].source_id for i in idx))
        print(f"{name:>5} | samples={n} pos={n_pos} neg={n_neg} videos={vids}")

    print(f"Split mode: {args.split_mode}")
    summarize_split("train", train_idx)
    summarize_split("val", val_idx)
    summarize_split("test", test_idx)

    train_set = CanPosNegDataset(
        [all_samples[i] for i in train_idx],
        args.image_size,
        augment=args.augment,
        roi_train=args.roi_train,
        roi_context=args.roi_context,
        roi_jitter=args.roi_jitter,
        roi_min_size=args.roi_min_size,
        roi_prob=args.roi_prob,
        rot_max_deg=args.rot_max_deg,
    )
    val_set = CanPosNegDataset([all_samples[i] for i in val_idx], args.image_size)
    test_set = CanPosNegDataset([all_samples[i] for i in test_idx], args.image_size)

    if args.balance:
        labels = np.array([s.label for s in train_set.samples], dtype=np.int64)
        n_pos = max(1, int((labels == 1).sum()))
        n_neg = max(1, int((labels == 0).sum()))
        pos_w = 0.5 / n_pos
        neg_w = 0.5 / n_neg
        weights = np.where(labels == 1, pos_w, neg_w).astype(np.float64)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        train_loader = DataLoader(
            train_set, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=True
        )
    else:
        train_loader = DataLoader(
            train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True
        )
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda but CUDA is not available.")

    model = MultiHeadRegressor(backbone=args.backbone, pretrained=args.pretrained).to(device)
    if args.init_checkpoint is not None:
        if not args.init_checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {args.init_checkpoint}")
        ckpt = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"Loaded init checkpoint: {args.init_checkpoint}")
        if missing:
            print(f"Missing keys while loading: {len(missing)}")
        if unexpected:
            print(f"Unexpected keys while loading: {len(unexpected)}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    pos_weight = None
    if args.auto_pos_weight:
        n_pos = len([s for s in train_set.samples if s.label == 1])
        n_neg = len([s for s in train_set.samples if s.label == 0])
        pos_weight = torch.tensor([max(1.0, n_neg / max(1, n_pos))], device=device)
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model, train_loader, device, optimizer=optimizer, use_amp=args.amp, scaler=scaler,
            kp_weight=args.kp_weight, bce_loss=bce_loss, symmetry_aware=args.symmetry_aware
        )
        val_metrics = run_epoch(
            model, val_loader, device, optimizer=None, use_amp=False, scaler=None,
            kp_weight=args.kp_weight, bce_loss=bce_loss, symmetry_aware=args.symmetry_aware
        )

        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()},
               **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        print(
            f"Epoch {epoch:03d} | "
            f"train loss {train_metrics['loss']:.4f} cls {train_metrics['cls']:.4f} kp {train_metrics['kp']:.4f} "
            f"acc {train_metrics['acc']:.3f} | "
            f"val loss {val_metrics['loss']:.4f} cls {val_metrics['cls']:.4f} kp {val_metrics['kp']:.4f} "
            f"acc {val_metrics['acc']:.3f}"
        )

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": vars(args),
                    "history": history,
                },
                args.output_dir / "best_model.pt",
            )

        torch.save(
            {
                "model_state": model.state_dict(),
                "config": vars(args),
                "history": history,
            },
            args.output_dir / "last_model.pt",
        )

    test_metrics = run_epoch(
        model, test_loader, device, optimizer=None, use_amp=False, scaler=None,
        kp_weight=args.kp_weight, bce_loss=bce_loss, symmetry_aware=args.symmetry_aware
    )
    print(
        f"Test | loss {test_metrics['loss']:.4f} cls {test_metrics['cls']:.4f} "
        f"kp {test_metrics['kp']:.4f} acc {test_metrics['acc']:.3f}"
    )

    history_path = args.output_dir / "history.json"
    history_path.write_text(json.dumps(history, indent=2))
    print(f"Saved history to {history_path}")


if __name__ == "__main__":
    main()
