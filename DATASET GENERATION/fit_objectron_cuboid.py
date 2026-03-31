import argparse
import json
from pathlib import Path
import numpy as np


def load_points3d_txt(path: Path) -> np.ndarray:
    pts = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            x, y, z = map(float, parts[1:4])
            pts.append((x, y, z))
    if not pts:
        raise ValueError(f"No points found in {path}")
    return np.asarray(pts, dtype=np.float64)


def pca_obb(points: np.ndarray, extent_percentile: float = 2.0):
    # Center by centroid first
    centroid = points.mean(axis=0)
    centered = points - centroid
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]

    # Ensure right-handed coordinate system
    if np.linalg.det(eigvecs) < 0:
        eigvecs[:, 2] *= -1.0

    # Transform points into PCA frame and get robust extents
    points_obj = centered @ eigvecs
    if extent_percentile < 0.0 or extent_percentile >= 50.0:
        raise ValueError("extent_percentile must be in [0, 50)")

    if extent_percentile == 0.0:
        minv = points_obj.min(axis=0)
        maxv = points_obj.max(axis=0)
    else:
        lo = extent_percentile
        hi = 100.0 - extent_percentile
        minv = np.percentile(points_obj, lo, axis=0)
        maxv = np.percentile(points_obj, hi, axis=0)

    # Use symmetric extents around the centroid to keep center stable
    half = np.maximum(np.abs(minv), np.abs(maxv))

    # Box center in world coordinates (centroid)
    translation = centroid

    return eigvecs, translation, half


def build_keypoints(half: np.ndarray) -> np.ndarray:
    # Objectron unit box ordering (center + 8 corners)
    w, h, d = half
    corners = np.array([
        [0.0, 0.0, 0.0],
        [-w, -h, -d],
        [-w, -h, +d],
        [-w, +h, -d],
        [-w, +h, +d],
        [+w, -h, -d],
        [+w, -h, +d],
        [+w, +h, -d],
        [+w, +h, +d],
    ], dtype=np.float64)
    return corners


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--points3d", required=True, type=Path)
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--extent_percentile", type=float, default=2.0,
                    help="Per-axis percentile trim in PCA frame for box extents (0 disables).")
    args = ap.parse_args()

    points = load_points3d_txt(args.points3d)

    R_wo, t_wo, half = pca_obb(points, extent_percentile=args.extent_percentile)
    keypoints_obj = build_keypoints(half)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    np.save(args.out_dir / "cuboid_3d_keypoints.npy", keypoints_obj)

    keypoint_order = [
        "CENTER",
        "X- Y- Z-",
        "X- Y- Z+",
        "X- Y+ Z-",
        "X- Y+ Z+",
        "X+ Y- Z-",
        "X+ Y- Z+",
        "X+ Y+ Z-",
        "X+ Y+ Z+",
    ]

    with (args.out_dir / "keypoint_order.json").open("w") as f:
        json.dump({"order": keypoint_order}, f, indent=2)

    with (args.out_dir / "object_frame.json").open("w") as f:
        json.dump({
            "rotation_world_from_object": R_wo.tolist(),
            "translation_world_from_object": t_wo.tolist(),
            "scale_full": (half * 2.0).tolist(),
            "scale_half": half.tolist()
        }, f, indent=2)

    with (args.out_dir / "cuboid_3d_keypoints.json").open("w") as f:
        json.dump({"keypoints_object": keypoints_obj.tolist()}, f, indent=2)

    print("Wrote:")
    print(args.out_dir / "cuboid_3d_keypoints.npy")
    print(args.out_dir / "cuboid_3d_keypoints.json")
    print(args.out_dir / "keypoint_order.json")
    print(args.out_dir / "object_frame.json")


if __name__ == "__main__":
    main()
