from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from tqdm import tqdm

from objectron_paper_twostage import (
    EfficientNetLiteRegressor,
    build_intrinsics_dict,
    canonical_from_data,
    crop_and_resize,
    crop_uv_to_keypoints_px,
    detector_image_to_tensor,
    draw_cuboid,
    draw_points,
    load_annotations,
    load_twostage_checkpoint,
    regressor_image_to_tensor,
    solve_pose_epnp,
    solve_pose_with_symmetry,
)


def parse_args():
    ap = argparse.ArgumentParser(description="Paper-aligned Objectron two-stage inference")
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--images_dir", type=Path, required=True)
    ap.add_argument("--annotations_json", type=Path, default=None)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--canonical_3d", type=Path, default=None)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--max_frames", type=int, default=-1)
    ap.add_argument("--detector_score_thresh", type=float, default=0.35)
    ap.add_argument("--pnp_max_err", type=float, default=8.0)
    ap.add_argument("--symmetry_pose_search", action="store_true")
    ap.add_argument("--symmetry_temporal_weight", type=float, default=0.25)
    ap.add_argument("--box_smooth_alpha", type=float, default=0.0)
    ap.add_argument("--keypoint_smooth_alpha", type=float, default=0.0)
    ap.add_argument("--score_smooth_alpha", type=float, default=0.0)
    return ap.parse_args()


def build_detector(detector_arch: str):
    if detector_arch != "ssdlite320_mobilenet_v3_large":
        raise ValueError(
            f"Unsupported detector_arch: {detector_arch}. "
            "Only ssdlite320_mobilenet_v3_large is currently implemented."
        )
    return ssdlite320_mobilenet_v3_large(
        weights=None,
        weights_backbone=None,
        num_classes=2,
    )


def load_models(checkpoint_path: Path, device: torch.device):
    checkpoint = load_twostage_checkpoint(checkpoint_path, device=device)
    config = checkpoint.get("config", {})
    detector_arch = config.get("detector_arch", "ssdlite320_mobilenet_v3_large")
    regressor_backbone = config.get("regressor_backbone", "efficientnet_lite0")
    regressor_dropout = float(config.get("regressor_dropout", 0.0))

    detector = build_detector(detector_arch).to(device)
    detector.load_state_dict(checkpoint["detector_state"])
    detector.eval()

    regressor = EfficientNetLiteRegressor(
        backbone=regressor_backbone,
        pretrained=False,
        dropout=regressor_dropout,
    ).to(device)
    regressor.load_state_dict(checkpoint["regressor_state"])
    regressor.eval()

    return detector, regressor, checkpoint, config


@torch.no_grad()
def detect_box(
    image_rgb: np.ndarray,
    detector,
    device: torch.device,
    score_thresh: float,
):
    image_tensor = detector_image_to_tensor(image_rgb).to(device)
    output = detector([image_tensor])[0]
    scores = output["scores"].detach().cpu().numpy()
    boxes = output["boxes"].detach().cpu().numpy()
    if scores.size == 0:
        return None, 0.0
    idx = int(np.argmax(scores))
    score = float(scores[idx])
    if score < float(score_thresh):
        return None, score
    return boxes[idx].astype(np.float32), score


@torch.no_grad()
def regress_keypoints(
    image_rgb: np.ndarray,
    box_xyxy: np.ndarray,
    regressor,
    image_size: int,
    device: torch.device,
    use_amp: bool,
):
    crop, clipped_box, _ = crop_and_resize(image_rgb, box_xyxy=box_xyxy, output_size=image_size)
    tensor = regressor_image_to_tensor(crop).unsqueeze(0).to(device)
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
        crop_uv = regressor(tensor)[0].detach().cpu().numpy()
    keypoints_px = crop_uv_to_keypoints_px(crop_uv, clipped_box)
    return crop_uv, keypoints_px, clipped_box


def list_images(images_dir: Path) -> List[Path]:
    images = sorted(images_dir.glob("*.png"))
    images += sorted(images_dir.glob("*.jpg"))
    return images


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = args.out_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda but CUDA is not available.")
    use_amp = bool(args.amp and device.type == "cuda")

    detector, regressor, checkpoint, config = load_models(args.checkpoint, device=device)
    regressor_image_size = int(config.get("regressor_image_size", 224))
    images = list_images(args.images_dir)
    if args.max_frames > 0:
        images = images[: args.max_frames]
    if not images:
        raise ValueError(f"No images found in {args.images_dir}")

    annotations = None
    ann_by_name: Dict[str, dict] = {}
    intr_by_name = {}
    fallback_intr = None
    canonical_3d = None
    if args.annotations_json is not None and args.annotations_json.exists():
        annotations = load_annotations(args.annotations_json)
        ann_by_name = {ann.get("image", ""): ann for ann in annotations}
        intr_by_name, fallback_intr = build_intrinsics_dict(annotations)
        canonical_3d = canonical_from_data(args.canonical_3d, annotations)

    results = []
    prev_box = None
    prev_keypoints_px = None
    prev_score = None
    last_reproj = None

    for image_path in tqdm(images, desc="infer", leave=False):
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        height, width = image_bgr.shape[:2]
        ann = ann_by_name.get(image_path.name)
        intr = intr_by_name.get(image_path.name, fallback_intr)

        box_xyxy, score = detect_box(
            image_rgb,
            detector=detector,
            device=device,
            score_thresh=args.detector_score_thresh,
        )
        if box_xyxy is not None and prev_box is not None and args.box_smooth_alpha > 0.0:
            alpha = float(np.clip(args.box_smooth_alpha, 0.0, 1.0))
            box_xyxy = alpha * prev_box + (1.0 - alpha) * box_xyxy
        if prev_score is not None and args.score_smooth_alpha > 0.0:
            alpha = float(np.clip(args.score_smooth_alpha, 0.0, 1.0))
            score = alpha * prev_score + (1.0 - alpha) * score

        detected = box_xyxy is not None
        pred_crop_uv = None
        pred_image_uv = None
        pred_uv_px = None
        pose = None
        pose_reproj = None
        pose_reproj_err = None

        if detected:
            pred_crop_uv, pred_uv_px, box_xyxy = regress_keypoints(
                image_rgb,
                box_xyxy=box_xyxy,
                regressor=regressor,
                image_size=regressor_image_size,
                device=device,
                use_amp=use_amp,
            )
            if prev_keypoints_px is not None and args.keypoint_smooth_alpha > 0.0:
                alpha = float(np.clip(args.keypoint_smooth_alpha, 0.0, 1.0))
                pred_uv_px = alpha * prev_keypoints_px + (1.0 - alpha) * pred_uv_px
            pred_image_uv = np.stack(
                [pred_uv_px[:, 0] / max(1.0, float(width)), pred_uv_px[:, 1] / max(1.0, float(height))],
                axis=1,
            )
            prev_box = box_xyxy.copy()
            prev_keypoints_px = pred_uv_px.copy()
            prev_score = float(score)

            if intr is not None and canonical_3d is not None:
                if args.symmetry_pose_search:
                    pose_candidate = solve_pose_with_symmetry(
                        canonical_3d,
                        pred_uv_px,
                        intr,
                        last_reproj=last_reproj,
                        pnp_max_err=args.pnp_max_err,
                        use_symmetry=True,
                        temporal_weight=args.symmetry_temporal_weight,
                    )
                    if pose_candidate is not None:
                        rvec, tvec, pose_reproj, pose_reproj_err, perm = pose_candidate
                        pose = {
                            "rotation": rvec.tolist(),
                            "translation": tvec.tolist(),
                            "perm": perm.tolist(),
                        }
                else:
                    ok, rvec, tvec, pose_reproj, pose_reproj_err = solve_pose_epnp(canonical_3d, pred_uv_px, intr)
                    if ok and (args.pnp_max_err <= 0.0 or pose_reproj_err <= args.pnp_max_err):
                        pose = {
                            "rotation": rvec.tolist(),
                            "translation": tvec.tolist(),
                        }
                if pose_reproj is not None:
                    last_reproj = pose_reproj.copy()
        else:
            prev_score = float(score)

        overlay = image_bgr.copy()
        if detected and box_xyxy is not None:
            x0, y0, x1, y1 = box_xyxy.astype(int).tolist()
            cv2.rectangle(overlay, (x0, y0), (x1, y1), (255, 255, 0), 2, cv2.LINE_AA)
            draw_points(overlay, pred_uv_px)
            draw_cuboid(overlay, pose_reproj if pose_reproj is not None else pred_uv_px)
        cv2.putText(
            overlay,
            f"score={score:.3f}" if detected else f"score={score:.3f} no_det",
            (12, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imwrite(str(overlay_dir / image_path.name), overlay)

        record = {
            "image": image_path.name,
            "detected": bool(detected),
            "score": float(score),
            "box_xyxy": None if box_xyxy is None else [float(x) for x in box_xyxy.tolist()],
            "pred_keypoints_crop_uv": None if pred_crop_uv is None else pred_crop_uv.tolist(),
            "pred_keypoints_image_uv": None if pred_image_uv is None else pred_image_uv.tolist(),
            "pred_keypoints_px": None if pred_uv_px is None else pred_uv_px.tolist(),
            "pose": pose,
            "pnp_reproj_error": None if pose_reproj_err is None else float(pose_reproj_err),
        }
        if ann is not None:
            record["gt_present"] = True
            record["gt_keypoints_2d"] = ann.get("keypoints_2d")
        results.append(record)

    summary = {
        "checkpoint": str(args.checkpoint),
        "num_frames": len(results),
        "num_detected": int(sum(1 for r in results if r["detected"])),
        "config": config,
        "predictions": results,
    }
    (args.out_dir / "predictions.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
