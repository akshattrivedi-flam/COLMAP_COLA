import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn


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
    def __init__(self):
        super().__init__()
        self.backbone = SimpleBackbone(width=32)
        in_features = self.backbone.out_dim
        self.kpt_head = nn.Linear(in_features, 18)
        self.cls_head = nn.Linear(in_features, 1)

    def forward(self, x):
        feat = self.backbone(x)
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


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--include_preprocess", action="store_true", help="Include preprocess+H2D in timing")
    return ap.parse_args()


def main():
    args = parse_args()

    try:
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)
    except Exception:
        pass

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda but CUDA is not available.")
    device = torch.device(args.device)
    use_amp = bool(args.amp and device.type == "cuda")

    checkpoint = torch.load(str(args.checkpoint), map_location="cpu", weights_only=False)
    cfg = checkpoint.get("config", {})
    image_size = int(cfg.get("image_size", 320))

    model = MultiHeadRegressor().to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    image_bgr = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Failed to read image: {args.image}")

    def run_once(tensor):
        with torch.no_grad():
            if use_amp:
                with torch.autocast(device_type="cuda"):
                    model(tensor)
            else:
                model(tensor)

    if not args.include_preprocess:
        tensor = preprocess_rgb(image_bgr, image_size).unsqueeze(0).to(device)

    # Warmup
    for _ in range(args.warmup):
        if args.include_preprocess:
            tensor = preprocess_rgb(image_bgr, image_size).unsqueeze(0).to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        run_once(tensor)
        if device.type == "cuda":
            torch.cuda.synchronize()

    # Timed runs
    times = []
    for _ in range(args.iters):
        if args.include_preprocess:
            tensor = preprocess_rgb(image_bgr, image_size).unsqueeze(0).to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        run_once(tensor)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    times_np = np.array(times, dtype=np.float32)
    mean_ms = float(times_np.mean())
    p50 = float(np.percentile(times_np, 50))
    p95 = float(np.percentile(times_np, 95))
    fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0

    mode = "full (preprocess+H2D+model)" if args.include_preprocess else "model only"
    print(f"Mode: {mode}")
    print(f"Device: {device} | image_size: {image_size} | iters: {args.iters}")
    print(f"Mean: {mean_ms:.3f} ms | P50: {p50:.3f} ms | P95: {p95:.3f} ms | FPS: {fps:.2f}")


if __name__ == "__main__":
    main()
