import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np


def draw_points(img, pts, color=(0, 255, 0), radius=4):
    for i, (x, y) in enumerate(pts):
        cv2.circle(img, (int(round(x)), int(round(y))), radius, color, -1)
        cv2.putText(img, str(i), (int(round(x)) + 4, int(round(y)) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)


def draw_cuboid_edges(img, pts, color=(0, 255, 0), thickness=2):
    # Keypoint 0 is center; cuboid corners are 1..8.
    edges = [
        (1, 2), (1, 3), (1, 5),
        (2, 4), (2, 6),
        (3, 4), (3, 7),
        (4, 8),
        (5, 6), (5, 7),
        (6, 8),
        (7, 8),
    ]
    for i, j in edges:
        p0 = pts[i]
        p1 = pts[j]
        cv2.line(
            img,
            (int(round(p0[0])), int(round(p0[1]))),
            (int(round(p1[0])), int(round(p1[1]))),
            color,
            thickness,
            cv2.LINE_AA,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", required=True, type=Path)
    ap.add_argument("--images", required=True, type=Path)
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--num", type=int, default=12)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    ann_files = sorted(args.annotations.glob("*.json"))
    if not ann_files:
        raise ValueError("No annotation JSONs found")

    rng = random.Random(args.seed)
    sample = rng.sample(ann_files, min(args.num, len(ann_files)))

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for ann_path in sample:
        ann = json.loads(ann_path.read_text())
        img_name = ann["image"]
        img_path = args.images / img_name
        if not img_path.exists():
            # skip if image missing
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            continue

        pts_norm_depth = np.array(ann["keypoints"]["points_2d_norm_depth"], dtype=np.float64)
        W, H = ann["image_size"]
        us = pts_norm_depth[:, 0] * W
        vs = pts_norm_depth[:, 1] * H
        pts_px = np.stack([us, vs], axis=1)

        vis = ann.get("visibility", 1.0)
        overlay = img.copy()
        draw_cuboid_edges(overlay, pts_px, color=(0, 255, 0), thickness=2)
        draw_points(overlay, pts_px, color=(0, 255, 0), radius=4)
        cv2.putText(overlay, f"vis={vis:.2f}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

        out_path = args.out_dir / img_path.name
        cv2.imwrite(str(out_path), overlay)

    print(f"Wrote overlays to {args.out_dir}")


if __name__ == "__main__":
    main()
