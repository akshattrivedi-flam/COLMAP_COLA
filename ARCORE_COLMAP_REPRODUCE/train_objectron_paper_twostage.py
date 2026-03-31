from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.ops import box_iou
from tqdm import tqdm

from objectron_paper_twostage import (
    SYM_PERMS_YAW4,
    EfficientNetLiteRegressor,
    ObjectronFrameSample,
    crop_and_resize,
    detector_image_to_tensor,
    keypoints_px_to_crop_uv,
    keypoints_uv_to_px,
    load_negative_samples,
    load_positive_samples,
    regressor_image_to_tensor,
    seed_everything,
    split_indices_stratified_by_source,
    square_crop_box_from_keypoints,
    summarize_split,
    write_lines,
)


def parse_args():
    ap = argparse.ArgumentParser(description="Paper-aligned Objectron two-stage training")
    ap.add_argument("--pos_dirs", nargs="+", required=True)
    ap.add_argument("--neg_dirs", nargs="+", required=True)
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--split_mode", choices=["video"], default="video")
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--num_workers", type=int, default=4)

    ap.add_argument("--detector_arch", type=str, default="ssdlite320_mobilenet_v3_large")
    ap.add_argument("--detector_epochs", type=int, default=60)
    ap.add_argument("--detector_batch_size", type=int, default=8)
    ap.add_argument("--detector_lr", type=float, default=3e-4)
    ap.add_argument("--detector_weight_decay", type=float, default=1e-4)
    ap.add_argument("--detector_score_thresh", type=float, default=0.35)
    ap.add_argument("--detector_context", type=float, default=1.35)
    ap.add_argument("--detector_min_size", type=int, default=48)
    ap.add_argument("--detector_augment", action="store_true")

    ap.add_argument("--regressor_backbone", type=str, default="efficientnet_lite0")
    ap.add_argument("--regressor_image_size", type=int, default=224)
    ap.add_argument("--regressor_epochs", type=int, default=250)
    ap.add_argument("--regressor_batch_size", type=int, default=64)
    ap.add_argument("--regressor_lr", type=float, default=1e-2)
    ap.add_argument("--regressor_lr_final", type=float, default=1e-6)
    ap.add_argument("--regressor_weight_decay", type=float, default=1e-4)
    ap.add_argument("--regressor_dropout", type=float, default=0.0)
    ap.add_argument("--crop_context", type=float, default=1.35)
    ap.add_argument("--crop_min_size", type=int, default=96)
    ap.add_argument("--crop_jitter", type=float, default=0.08)
    ap.add_argument("--rot_max_deg", type=float, default=180.0)
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--symmetry_aware", action="store_true")
    ap.add_argument("--pretrained_regressor", action="store_true")
    return ap.parse_args()


class DetectorDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[ObjectronFrameSample],
        box_context: float,
        min_box_size: int,
        augment: bool = False,
    ):
        self.samples = list(samples)
        self.box_context = float(box_context)
        self.min_box_size = int(min_box_size)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image = cv2.imread(str(sample.image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to read image: {sample.image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]

        box = None
        if sample.label == 1 and sample.keypoints_uv is not None:
            box = square_crop_box_from_keypoints(
                sample.keypoints_uv,
                width=width,
                height=height,
                context=self.box_context,
                min_size=self.min_box_size,
            )

        if self.augment:
            if random.random() < 0.8:
                gain = random.uniform(0.85, 1.20)
                bias = random.uniform(-0.08, 0.08)
                image = np.clip(image.astype(np.float32) / 255.0 * gain + bias, 0.0, 1.0)
                image = (image * 255.0).astype(np.uint8)
            if random.random() < 0.5:
                image = np.ascontiguousarray(image[:, ::-1, :])
                if box is not None:
                    x0, y0, x1, y1 = box.tolist()
                    box = np.array([width - x1, y0, width - x0, y1], dtype=np.float32)

        image_tensor = detector_image_to_tensor(image)
        if box is None:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.from_numpy(box.reshape(1, 4))
            labels = torch.ones((1,), dtype=torch.int64)
            area = torch.tensor([(box[2] - box[0]) * (box[3] - box[1])], dtype=torch.float32)
            iscrowd = torch.zeros((1,), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "area": area,
            "iscrowd": iscrowd,
            "image_id": torch.tensor([idx], dtype=torch.int64),
        }
        return image_tensor, target


class RegressorDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[ObjectronFrameSample],
        image_size: int,
        crop_context: float,
        crop_min_size: int,
        crop_jitter: float = 0.0,
        augment: bool = False,
        rot_max_deg: float = 180.0,
    ):
        self.samples = [s for s in samples if s.label == 1 and s.keypoints_uv is not None]
        self.image_size = int(image_size)
        self.crop_context = float(crop_context)
        self.crop_min_size = int(crop_min_size)
        self.crop_jitter = float(max(0.0, crop_jitter))
        self.augment = augment
        self.rot_max_deg = float(max(0.0, rot_max_deg))

    def __len__(self) -> int:
        return len(self.samples)

    def _apply_augment(
        self,
        image: np.ndarray,
        keypoints_uv: np.ndarray,
        keypoints_w: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        height, width = image.shape[:2]
        k = keypoints_uv.copy().astype(np.float32)
        w = keypoints_w.copy().astype(np.float32)

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
        if random.random() < 0.25:
            ksz = random.choice([3, 5])
            image = cv2.GaussianBlur(image, (ksz, ksz), sigmaX=0.0)

        if random.random() < 0.7:
            angle = random.uniform(-self.rot_max_deg, self.rot_max_deg)
            scale = random.uniform(0.85, 1.15)
            tx = random.uniform(-0.12, 0.12) * width
            ty = random.uniform(-0.12, 0.12) * height
            M = cv2.getRotationMatrix2D((width * 0.5, height * 0.5), angle, scale)
            M[0, 2] += tx
            M[1, 2] += ty
            image = cv2.warpAffine(
                image,
                M,
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )
            pts = np.stack(
                [k[:, 0] * width, k[:, 1] * height, np.ones((k.shape[0],), dtype=np.float32)],
                axis=1,
            )
            pts2 = (M @ pts.T).T
            k[:, 0] = np.clip(pts2[:, 0] / max(1.0, float(width)), 0.0, 1.0)
            k[:, 1] = np.clip(pts2[:, 1] / max(1.0, float(height)), 0.0, 1.0)

        if random.random() < 0.5:
            remap = np.array([0, 2, 1, 4, 3, 6, 5, 8, 7], dtype=np.int64)
            image = np.ascontiguousarray(image[:, ::-1, :])
            k[:, 0] = 1.0 - k[:, 0]
            k = k[remap]
            w = w[remap]

        return image, k, w

    def _maybe_jitter_box(self, box_xyxy: np.ndarray, width: int, height: int) -> np.ndarray:
        if not self.augment or self.crop_jitter <= 0.0:
            return box_xyxy
        x0, y0, x1, y1 = box_xyxy.astype(np.float32).tolist()
        side = max(x1 - x0, y1 - y0)
        cx = 0.5 * (x0 + x1)
        cy = 0.5 * (y0 + y1)
        jitter = self.crop_jitter
        cx += np.random.uniform(-jitter, jitter) * side
        cy += np.random.uniform(-jitter, jitter) * side
        side *= np.random.uniform(1.0 - jitter, 1.0 + jitter)
        box = np.array(
            [cx - 0.5 * side, cy - 0.5 * side, cx + 0.5 * side, cy + 0.5 * side],
            dtype=np.float32,
        )
        return np.array(
            [
                np.clip(box[0], 0.0, max(0.0, width - 2.0)),
                np.clip(box[1], 0.0, max(0.0, height - 2.0)),
                np.clip(box[2], 2.0, float(width)),
                np.clip(box[3], 2.0, float(height)),
            ],
            dtype=np.float32,
        )

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image = cv2.imread(str(sample.image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to read image: {sample.image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        keypoints_uv = sample.keypoints_uv.copy().astype(np.float32)
        keypoints_w = sample.keypoints_w.copy().astype(np.float32)

        if self.augment:
            image, keypoints_uv, keypoints_w = self._apply_augment(image, keypoints_uv, keypoints_w)

        height, width = image.shape[:2]
        box = square_crop_box_from_keypoints(
            keypoints_uv,
            width=width,
            height=height,
            context=self.crop_context,
            min_size=self.crop_min_size,
        )
        box = self._maybe_jitter_box(box, width=width, height=height)
        crop, clipped_box, _ = crop_and_resize(image, box_xyxy=box, output_size=self.image_size)
        keypoints_px = keypoints_uv_to_px(keypoints_uv, width=width, height=height)
        target_uv = keypoints_px_to_crop_uv(keypoints_px, clipped_box)

        return {
            "image": regressor_image_to_tensor(crop),
            "target_uv": torch.from_numpy(target_uv.astype(np.float32)),
            "target_w": torch.from_numpy(np.clip(keypoints_w, 0.0, 1.0).astype(np.float32)),
            "image_name": sample.image_name,
        }


def collate_detection(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


def build_detector(args) -> nn.Module:
    if args.detector_arch != "ssdlite320_mobilenet_v3_large":
        raise ValueError(
            f"Unsupported detector_arch: {args.detector_arch}. "
            "Only ssdlite320_mobilenet_v3_large is currently implemented."
        )
    return ssdlite320_mobilenet_v3_large(
        weights=None,
        weights_backbone=None,
        num_classes=2,
    )


def detector_loss_step(model, images, targets, device, use_amp: bool):
    images = [img.to(device, non_blocking=True) for img in images]
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())
    scalar_losses = {k: float(v.detach().cpu().item()) for k, v in loss_dict.items()}
    scalar_losses["loss"] = float(loss.detach().cpu().item())
    return loss, scalar_losses


def train_detector_epoch(model, loader, device, optimizer, use_amp: bool, scaler):
    model.train()
    totals: Dict[str, float] = {}
    count = 0
    for images, targets in tqdm(loader, leave=False, desc="det_train"):
        loss, loss_dict = detector_loss_step(model, images, targets, device=device, use_amp=use_amp)
        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        bs = len(images)
        count += bs
        for key, value in loss_dict.items():
            totals[key] = totals.get(key, 0.0) + value * bs

    for key in list(totals.keys()):
        totals[key] /= max(1, count)
    return totals


@torch.no_grad()
def evaluate_detector(model, loader, device, score_thresh: float):
    model.eval()
    tp = fp = tn = fn = 0
    mean_iou = 0.0
    pos_count = 0
    for images, targets in tqdm(loader, leave=False, desc="det_eval"):
        images_device = [img.to(device, non_blocking=True) for img in images]
        outputs = model(images_device)
        for output, target in zip(outputs, targets):
            gt_boxes = target["boxes"]
            scores = output["scores"].detach().cpu()
            boxes = output["boxes"].detach().cpu()
            keep = scores >= float(score_thresh)
            is_positive = gt_boxes.shape[0] > 0

            if is_positive:
                pos_count += 1
                best_iou = 0.0
                if keep.any():
                    ious = box_iou(boxes[keep], gt_boxes)
                    best_iou = float(ious.max().item())
                mean_iou += best_iou
                if best_iou >= 0.5:
                    tp += 1
                else:
                    fn += 1
            else:
                if keep.any():
                    fp += 1
                else:
                    tn += 1

    precision = tp / max(1.0, tp + fp)
    recall = tp / max(1.0, tp + fn)
    f1 = 2.0 * precision * recall / max(1e-6, precision + recall)
    neg_total = fp + tn
    return {
        "precision_50": precision,
        "recall_50": recall,
        "f1_50": f1,
        "mean_iou": mean_iou / max(1, pos_count),
        "neg_fp_rate": fp / max(1.0, neg_total),
    }


def _symmetry_aware_vertex_mse(pred_uv, target_uv, target_w, image_size: int):
    perms = torch.as_tensor(SYM_PERMS_YAW4, dtype=torch.long, device=pred_uv.device)
    target_perm = target_uv[:, perms, :].permute(1, 0, 2, 3)
    weight_perm = target_w[:, perms].permute(1, 0, 2)
    per_vertex = ((pred_uv.unsqueeze(0) - target_perm) ** 2).sum(dim=-1) / 2.0
    denom = torch.clamp(weight_perm.sum(dim=-1), min=1e-6)
    per_perm = (per_vertex * weight_perm).sum(dim=-1) / denom
    best_perm = torch.argmin(per_perm, dim=0)
    batch_index = torch.arange(pred_uv.shape[0], device=pred_uv.device)
    best_loss = per_perm[best_perm, batch_index]
    chosen_target = target_perm[best_perm, batch_index]
    chosen_weight = weight_perm[best_perm, batch_index]

    diff = pred_uv - chosen_target
    err_uv = torch.linalg.norm(diff, dim=-1)
    err_px = torch.linalg.norm(diff * float(image_size), dim=-1)
    denom = torch.clamp(chosen_weight.sum(dim=-1), min=1e-6)
    return (
        best_loss.mean(),
        ((err_uv * chosen_weight).sum(dim=-1) / denom).mean(),
        ((err_px * chosen_weight).sum(dim=-1) / denom).mean(),
    )


def vertex_mse_loss(pred_uv, target_uv, target_w, image_size: int, symmetry_aware: bool):
    if symmetry_aware:
        return _symmetry_aware_vertex_mse(pred_uv, target_uv, target_w, image_size=image_size)

    weight = torch.clamp(target_w, 0.0, 1.0)
    per_vertex = ((pred_uv - target_uv) ** 2).sum(dim=-1) / 2.0
    denom = torch.clamp(weight.sum(dim=-1), min=1e-6)
    loss = ((per_vertex * weight).sum(dim=-1) / denom).mean()

    diff = pred_uv - target_uv
    err_uv = ((torch.linalg.norm(diff, dim=-1) * weight).sum(dim=-1) / denom).mean()
    err_px = ((torch.linalg.norm(diff * float(image_size), dim=-1) * weight).sum(dim=-1) / denom).mean()
    return loss, err_uv, err_px


def run_regressor_epoch(model, loader, device, optimizer, scheduler, use_amp: bool, scaler, image_size: int, symmetry_aware: bool):
    is_train = optimizer is not None
    model.train(is_train)
    totals = {"loss": 0.0, "uv": 0.0, "px": 0.0}
    count = 0
    for batch in tqdm(loader, leave=False, desc="reg_train" if is_train else "reg_eval"):
        image = batch["image"].to(device, non_blocking=True)
        target_uv = batch["target_uv"].to(device, non_blocking=True)
        target_w = batch["target_w"].to(device, non_blocking=True)

        with torch.set_grad_enabled(is_train):
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                pred_uv = model(image)
                loss, err_uv, err_px = vertex_mse_loss(
                    pred_uv,
                    target_uv,
                    target_w,
                    image_size=image_size,
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
                if scheduler is not None:
                    scheduler.step()

        bs = image.shape[0]
        totals["loss"] += float(loss.detach().cpu().item()) * bs
        totals["uv"] += float(err_uv.detach().cpu().item()) * bs
        totals["px"] += float(err_px.detach().cpu().item()) * bs
        count += bs

    for key in totals:
        totals[key] /= max(1, count)
    return totals


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda but CUDA is not available.")
    use_amp = bool(args.amp and device.type == "cuda")

    pos_samples: List[ObjectronFrameSample] = []
    for pos_dir in args.pos_dirs:
        pos_samples += load_positive_samples(Path(pos_dir))

    neg_samples: List[ObjectronFrameSample] = []
    for neg_dir in args.neg_dirs:
        neg_samples += load_negative_samples(Path(neg_dir))

    all_samples = pos_samples + neg_samples
    pos_idx = [idx for idx, sample in enumerate(all_samples) if sample.label == 1]
    neg_idx = [idx for idx, sample in enumerate(all_samples) if sample.label == 0]
    if not pos_idx or not neg_idx:
        raise ValueError("Need both positive and negative samples for two-stage training.")

    train_idx, val_idx, test_idx = split_indices_stratified_by_source(
        all_samples,
        pos_idx,
        neg_idx,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    if not train_idx or not val_idx or not test_idx:
        raise ValueError(
            f"Split produced empty set(s): train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}"
        )

    print("Split mode: video")
    summarize_split("train", all_samples, train_idx)
    summarize_split("val", all_samples, val_idx)
    summarize_split("test", all_samples, test_idx)

    write_lines(args.output_dir / "included_positive_dirs.txt", args.pos_dirs)
    write_lines(args.output_dir / "included_negative_dirs.txt", args.neg_dirs)

    det_train_set = DetectorDataset(
        [all_samples[i] for i in train_idx],
        box_context=args.detector_context,
        min_box_size=args.detector_min_size,
        augment=args.detector_augment,
    )
    det_val_set = DetectorDataset(
        [all_samples[i] for i in val_idx],
        box_context=args.detector_context,
        min_box_size=args.detector_min_size,
        augment=False,
    )
    det_test_set = DetectorDataset(
        [all_samples[i] for i in test_idx],
        box_context=args.detector_context,
        min_box_size=args.detector_min_size,
        augment=False,
    )

    reg_train_samples = [all_samples[i] for i in train_idx if all_samples[i].label == 1]
    reg_val_samples = [all_samples[i] for i in val_idx if all_samples[i].label == 1]
    reg_test_samples = [all_samples[i] for i in test_idx if all_samples[i].label == 1]
    if not reg_train_samples or not reg_val_samples or not reg_test_samples:
        raise ValueError("Positive splits for regressor are empty. Adjust splits or dataset.")

    reg_train_set = RegressorDataset(
        reg_train_samples,
        image_size=args.regressor_image_size,
        crop_context=args.crop_context,
        crop_min_size=args.crop_min_size,
        crop_jitter=args.crop_jitter,
        augment=args.augment,
        rot_max_deg=args.rot_max_deg,
    )
    reg_val_set = RegressorDataset(
        reg_val_samples,
        image_size=args.regressor_image_size,
        crop_context=args.crop_context,
        crop_min_size=args.crop_min_size,
        crop_jitter=0.0,
        augment=False,
        rot_max_deg=args.rot_max_deg,
    )
    reg_test_set = RegressorDataset(
        reg_test_samples,
        image_size=args.regressor_image_size,
        crop_context=args.crop_context,
        crop_min_size=args.crop_min_size,
        crop_jitter=0.0,
        augment=False,
        rot_max_deg=args.rot_max_deg,
    )

    det_train_loader = DataLoader(
        det_train_set,
        batch_size=args.detector_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_detection,
    )
    det_val_loader = DataLoader(
        det_val_set,
        batch_size=args.detector_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_detection,
    )
    det_test_loader = DataLoader(
        det_test_set,
        batch_size=args.detector_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_detection,
    )

    reg_train_loader = DataLoader(
        reg_train_set,
        batch_size=args.regressor_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    reg_val_loader = DataLoader(
        reg_val_set,
        batch_size=args.regressor_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    reg_test_loader = DataLoader(
        reg_test_set,
        batch_size=args.regressor_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    detector = build_detector(args).to(device)
    det_optimizer = torch.optim.Adam(
        detector.parameters(),
        lr=args.detector_lr,
        weight_decay=args.detector_weight_decay,
    )
    det_scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    history: List[dict] = []
    best_det_f1 = -float("inf")
    best_detector_state = None
    for epoch in range(1, args.detector_epochs + 1):
        train_metrics = train_detector_epoch(
            detector,
            det_train_loader,
            device=device,
            optimizer=det_optimizer,
            use_amp=use_amp,
            scaler=det_scaler,
        )
        val_metrics = evaluate_detector(
            detector,
            det_val_loader,
            device=device,
            score_thresh=args.detector_score_thresh,
        )
        row = {
            "stage": "detector",
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)
        print(
            f"[detector] epoch {epoch:03d} | "
            f"train loss {train_metrics.get('loss', 0.0):.4f} | "
            f"val f1@0.5 {val_metrics['f1_50']:.3f} recall {val_metrics['recall_50']:.3f} "
            f"precision {val_metrics['precision_50']:.3f} mean_iou {val_metrics['mean_iou']:.3f}"
        )
        if val_metrics["f1_50"] > best_det_f1:
            best_det_f1 = val_metrics["f1_50"]
            best_detector_state = {k: v.detach().cpu().clone() for k, v in detector.state_dict().items()}
            torch.save(
                {
                    "detector_state": best_detector_state,
                    "config": vars(args),
                    "history": history,
                    "val_metrics": val_metrics,
                },
                args.output_dir / "best_detector.pt",
            )

        torch.save(
            {
                "detector_state": detector.state_dict(),
                "config": vars(args),
                "history": history,
                "val_metrics": val_metrics,
            },
            args.output_dir / "last_detector.pt",
        )

    last_detector_state = {k: v.detach().cpu().clone() for k, v in detector.state_dict().items()}
    if best_detector_state is not None:
        detector.load_state_dict(best_detector_state)
    detector_test_metrics = evaluate_detector(
        detector,
        det_test_loader,
        device=device,
        score_thresh=args.detector_score_thresh,
    )
    print(
        f"[detector] test | f1@0.5 {detector_test_metrics['f1_50']:.3f} "
        f"recall {detector_test_metrics['recall_50']:.3f} precision {detector_test_metrics['precision_50']:.3f}"
    )

    regressor = EfficientNetLiteRegressor(
        backbone=args.regressor_backbone,
        pretrained=args.pretrained_regressor,
        dropout=args.regressor_dropout,
    ).to(device)
    reg_optimizer = torch.optim.Adam(
        regressor.parameters(),
        lr=args.regressor_lr,
        weight_decay=args.regressor_weight_decay,
    )
    reg_scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    total_steps = max(1, args.regressor_epochs * len(reg_train_loader))
    gamma = math.exp(math.log(args.regressor_lr_final / args.regressor_lr) / max(1, total_steps - 1))
    reg_scheduler = torch.optim.lr_scheduler.ExponentialLR(reg_optimizer, gamma=gamma)

    best_reg_loss = float("inf")
    best_regressor_state = None
    for epoch in range(1, args.regressor_epochs + 1):
        train_metrics = run_regressor_epoch(
            regressor,
            reg_train_loader,
            device=device,
            optimizer=reg_optimizer,
            scheduler=reg_scheduler,
            use_amp=use_amp,
            scaler=reg_scaler,
            image_size=args.regressor_image_size,
            symmetry_aware=args.symmetry_aware,
        )
        val_metrics = run_regressor_epoch(
            regressor,
            reg_val_loader,
            device=device,
            optimizer=None,
            scheduler=None,
            use_amp=False,
            scaler=None,
            image_size=args.regressor_image_size,
            symmetry_aware=args.symmetry_aware,
        )
        row = {
            "stage": "regressor",
            "epoch": epoch,
            "lr": reg_optimizer.param_groups[0]["lr"],
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)
        print(
            f"[regressor] epoch {epoch:03d} | "
            f"train loss {train_metrics['loss']:.6f} uv {train_metrics['uv']:.4f} px {train_metrics['px']:.2f} | "
            f"val loss {val_metrics['loss']:.6f} uv {val_metrics['uv']:.4f} px {val_metrics['px']:.2f}"
        )

        if val_metrics["loss"] < best_reg_loss:
            best_reg_loss = val_metrics["loss"]
            best_regressor_state = {k: v.detach().cpu().clone() for k, v in regressor.state_dict().items()}
            torch.save(
                {
                    "regressor_state": best_regressor_state,
                    "config": vars(args),
                    "history": history,
                    "val_metrics": val_metrics,
                },
                args.output_dir / "best_regressor.pt",
            )

        torch.save(
            {
                "regressor_state": regressor.state_dict(),
                "config": vars(args),
                "history": history,
                "val_metrics": val_metrics,
            },
            args.output_dir / "last_regressor.pt",
        )

    last_regressor_state = {k: v.detach().cpu().clone() for k, v in regressor.state_dict().items()}
    if best_regressor_state is not None:
        regressor.load_state_dict(best_regressor_state)
    regressor_test_metrics = run_regressor_epoch(
        regressor,
        reg_test_loader,
        device=device,
        optimizer=None,
        scheduler=None,
        use_amp=False,
        scaler=None,
        image_size=args.regressor_image_size,
        symmetry_aware=args.symmetry_aware,
    )
    print(
        f"[regressor] test | loss {regressor_test_metrics['loss']:.6f} "
        f"uv {regressor_test_metrics['uv']:.4f} px {regressor_test_metrics['px']:.2f}"
    )

    if best_detector_state is None:
        best_detector_state = {k: v.detach().cpu().clone() for k, v in detector.state_dict().items()}
    if best_regressor_state is None:
        best_regressor_state = {k: v.detach().cpu().clone() for k, v in regressor.state_dict().items()}
    if "last_detector_state" not in locals():
        last_detector_state = {k: v.detach().cpu().clone() for k, v in detector.state_dict().items()}
    if "last_regressor_state" not in locals():
        last_regressor_state = {k: v.detach().cpu().clone() for k, v in regressor.state_dict().items()}

    torch.save(
        {
            "stage": "paper_twostage",
            "detector_state": best_detector_state,
            "regressor_state": best_regressor_state,
            "config": vars(args),
            "history": history,
            "detector_test_metrics": detector_test_metrics,
            "regressor_test_metrics": regressor_test_metrics,
        },
        args.output_dir / "best_model.pt",
    )
    torch.save(
        {
            "stage": "paper_twostage",
            "detector_state": last_detector_state,
            "regressor_state": last_regressor_state,
            "config": vars(args),
            "history": history,
            "detector_test_metrics": detector_test_metrics,
            "regressor_test_metrics": regressor_test_metrics,
        },
        args.output_dir / "last_model.pt",
    )

    history_path = args.output_dir / "history.json"
    history_path.write_text(json.dumps(history, indent=2))
    print(f"Saved history to {history_path}")


if __name__ == "__main__":
    main()
