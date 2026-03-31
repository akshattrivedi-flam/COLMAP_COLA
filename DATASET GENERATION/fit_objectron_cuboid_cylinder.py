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
            pts.append(tuple(map(float, parts[1:4])))
    if not pts:
        raise ValueError(f"No points in {path}")
    return np.asarray(pts, dtype=np.float64)


def load_images_txt(path: Path):
    rows = []
    lines = [ln.rstrip("\n") for ln in path.read_text().splitlines()]
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith('#'):
            i += 1
            continue
        p = line.split()
        if len(p) < 10:
            i += 1
            continue
        rows.append({
            "qvec": np.array(list(map(float, p[1:5])), dtype=np.float64),
            "tvec": np.array(list(map(float, p[5:8])), dtype=np.float64),
            "name": p[9],
        })
        i += 2
    return rows


def qvec_to_rotmat(qvec):
    qw, qx, qy, qz = qvec
    return np.array([
        [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy],
    ], dtype=np.float64)


def build_keypoints(half):
    w, h, d = half
    return np.array([
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--points3d", required=True, type=Path)
    ap.add_argument("--images", required=True, type=Path)
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--height_lo", type=float, default=2.0)
    ap.add_argument("--height_hi", type=float, default=98.0)
    ap.add_argument("--radius_pct", type=float, default=90.0)
    ap.add_argument("--radius_scale", type=float, default=1.0)
    ap.add_argument("--height_scale", type=float, default=1.0)
    args = ap.parse_args()

    pts = load_points3d_txt(args.points3d)
    c = pts.mean(axis=0)
    X = pts - c

    # Can axis from PCA major component.
    cov = np.cov(X.T)
    vals, vecs = np.linalg.eigh(cov)
    u = vecs[:, np.argsort(vals)[::-1][0]]
    u = u / np.linalg.norm(u)

    # Height from robust percentiles along axis.
    z = X @ u
    z_lo = np.percentile(z, args.height_lo)
    z_hi = np.percentile(z, args.height_hi)
    half_h = 0.5 * (z_hi - z_lo) * args.height_scale

    # Radius from perpendicular distance to axis.
    X_perp = X - np.outer(z, u)
    r = np.linalg.norm(X_perp, axis=1)
    radius = np.percentile(r, args.radius_pct) * args.radius_scale

    # Resolve yaw using average camera viewing direction projected on axis-orthogonal plane.
    cams = load_images_txt(args.images)
    v_sum = np.zeros(3, dtype=np.float64)
    for row in cams:
        Rcw = qvec_to_rotmat(row["qvec"])
        tcw = row["tvec"]
        Cw = -Rcw.T @ tcw  # camera center in world
        v = Cw - c
        v = v - (v @ u) * u
        n = np.linalg.norm(v)
        if n > 1e-12:
            v_sum += v / n

    if np.linalg.norm(v_sum) < 1e-12:
        # Fallback axis in orthogonal plane.
        x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        x_axis = x_axis - (x_axis @ u) * u
        if np.linalg.norm(x_axis) < 1e-12:
            x_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            x_axis = x_axis - (x_axis @ u) * u
    else:
        x_axis = v_sum

    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(u, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, u)
    x_axis = x_axis / np.linalg.norm(x_axis)

    R_wo = np.stack([x_axis, y_axis, u], axis=1)
    if np.linalg.det(R_wo) < 0:
        y_axis = -y_axis
        R_wo = np.stack([x_axis, y_axis, u], axis=1)

    half = np.array([radius, radius, half_h], dtype=np.float64)
    key_obj = build_keypoints(half)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.out_dir / "cuboid_3d_keypoints.npy", key_obj)

    with (args.out_dir / "cuboid_3d_keypoints.json").open("w") as f:
        json.dump({"keypoints_object": key_obj.tolist()}, f, indent=2)

    with (args.out_dir / "keypoint_order.json").open("w") as f:
        json.dump({"order": [
            "CENTER",
            "X- Y- Z-", "X- Y- Z+", "X- Y+ Z-", "X- Y+ Z+",
            "X+ Y- Z-", "X+ Y- Z+", "X+ Y+ Z-", "X+ Y+ Z+",
        ]}, f, indent=2)

    with (args.out_dir / "object_frame.json").open("w") as f:
        json.dump({
            "rotation_world_from_object": R_wo.tolist(),
            "translation_world_from_object": c.tolist(),
            "scale_full": (half * 2.0).tolist(),
            "scale_half": half.tolist(),
            "fit_method": "cylinder_principal_axis",
            "height_percentiles": [args.height_lo, args.height_hi],
            "radius_percentile": args.radius_pct,
        }, f, indent=2)

    print("half", half.tolist())
    print("wrote", args.out_dir)


if __name__ == "__main__":
    main()
