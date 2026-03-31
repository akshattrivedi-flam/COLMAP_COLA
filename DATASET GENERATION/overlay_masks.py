import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_dir", type=Path, required=True)
    ap.add_argument("--masks_dir", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--max_frames", type=int, default=50)
    ap.add_argument("--alpha", type=float, default=0.5)
    return ap.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    frames = sorted(args.frames_dir.glob("frame_*.jpg"))
    if not frames:
        raise RuntimeError("No frame_*.jpg images found.")

    frames = frames[: args.max_frames]
    for p in tqdm(frames):
        mask_path = args.masks_dir / (p.stem + ".png")
        if not mask_path.exists():
            continue
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            continue
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        color = np.zeros_like(img)
        color[:, :, 1] = 255
        m = (mask > 127).astype(np.uint8)[:, :, None]
        overlay = img.copy()
        overlay = (overlay * (1 - args.alpha) + (color * args.alpha) * m + overlay * args.alpha * (1 - m)).astype(
            np.uint8
        )
        cv2.imwrite(str(args.out_dir / p.name), overlay)

    print(f"Wrote overlays to {args.out_dir}")


if __name__ == "__main__":
    main()
