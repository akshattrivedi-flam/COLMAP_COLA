import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn


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


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--draw", action="store_true", help="Draw keypoints/cuboid on overlay")
    ap.add_argument("--scale_uv", type=float, default=1.0, help="Scale 2D cuboid around center (1.0 = no scale)")
    return ap.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

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

    image_bgr = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Failed to read image: {args.image}")
    H, W = image_bgr.shape[:2]

    tensor = preprocess_rgb(image_bgr, image_size).unsqueeze(0).to(device)
    with torch.no_grad():
        if use_amp:
            with torch.autocast(device_type="cuda"):
                pred_uv, cls_logits = model(tensor)
        else:
            pred_uv, cls_logits = model(tensor)

    pred_uv = pred_uv.squeeze(0).detach().cpu().numpy()
    cls_logit = float(cls_logits.squeeze(0).detach().cpu().numpy())
    score = float(1.0 / (1.0 + np.exp(-cls_logit)))
    label = int(score >= args.threshold)

    uv_px = np.stack([pred_uv[:, 0] * W, pred_uv[:, 1] * H], axis=1)
    if args.scale_uv != 1.0:
        center = uv_px[0].copy()
        uv_px = center[None, :] + (uv_px - center[None, :]) * float(args.scale_uv)

    overlay_path = args.out_dir / f"overlay_{args.image.stem}.png"
    result_path = args.out_dir / f"result_{args.image.stem}.json"

    if args.draw:
        overlay = image_bgr.copy()
        draw_points(overlay, uv_px, color=(255, 0, 0))
        draw_cuboid(overlay, uv_px, color=(0, 255, 0))
        cv2.putText(overlay, f"score={score:.3f}", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if label else (0, 0, 255), 2)
        cv2.imwrite(str(overlay_path), overlay)

    payload = {
        "image": str(args.image),
        "score": score,
        "label": label,
        "threshold": args.threshold,
        "image_size": image_size,
        "uv_norm": pred_uv.tolist(),
        "uv_px": uv_px.tolist(),
    }
    result_path.write_text(json.dumps(payload, indent=2))

    print(f"score={score:.4f} label={label} threshold={args.threshold}")
    print(f"saved: {result_path}")
    if args.draw:
        print(f"saved: {overlay_path}")


if __name__ == "__main__":
    main()
