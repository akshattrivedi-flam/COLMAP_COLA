import argparse
import json
from pathlib import Path

import numpy as np


def qvec_to_rotmat(qvec):
    qw, qx, qy, qz = qvec
    return np.array([
        [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
        [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qx * qw],
        [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx * qx - 2 * qy * qy],
    ], dtype=np.float64)


def normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-12:
        return None
    return v / n


def axis_angle_deg(a, b):
    c = np.clip(float(np.dot(a, b)), -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def rotate_about_axis(v, axis, angle_deg):
    # Rodrigues rotation.
    th = np.deg2rad(float(angle_deg))
    k = normalize(axis)
    if k is None:
        return v
    return v * np.cos(th) + np.cross(k, v) * np.sin(th) + k * (k @ v) * (1.0 - np.cos(th))


def load_images_txt(path: Path):
    rows = []
    lines = [ln.rstrip("\n") for ln in path.read_text().splitlines()]
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith("#"):
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


def load_colmap_points3d_txt(path: Path) -> np.ndarray:
    pts = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        p = line.split()
        if len(p) < 4:
            continue
        pts.append([float(p[1]), float(p[2]), float(p[3])])
    if not pts:
        raise ValueError(f"No points found in {path}")
    return np.asarray(pts, dtype=np.float64)


def load_points(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() == ".txt":
        return load_colmap_points3d_txt(path)

    if path.suffix.lower() == ".npy":
        pts = np.load(path)
        pts = np.asarray(pts, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(f"Expected Nx3 points in {path}, got {pts.shape}")
        return pts

    if path.suffix.lower() == ".npz":
        data = np.load(path, allow_pickle=True)
        if "xyz" in data.files:
            pts = data["xyz"]
        else:
            pts = None
            for k in data.files:
                arr = np.asarray(data[k])
                if arr.ndim == 2 and arr.shape[1] == 3:
                    pts = arr
                    break
            if pts is None:
                raise ValueError(f"No Nx3 array in {path}. Keys: {data.files}")
        pts = np.asarray(pts, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(f"Expected Nx3 points in {path}, got {pts.shape}")
        return pts

    raise ValueError(f"Unsupported points format: {path.suffix}")


def build_keypoints(half):
    hx, hy, hz = half
    return np.array([
        [0.0, 0.0, 0.0],
        [-hx, -hy, -hz], [-hx, -hy, +hz], [-hx, +hy, -hz], [-hx, +hy, +hz],
        [+hx, -hy, -hz], [+hx, -hy, +hz], [+hx, +hy, -hz], [+hx, +hy, +hz],
    ], dtype=np.float64)


def robust_half_extents(points_world, center_world, R_world_from_object, lo=2.0, hi=98.0):
    # Transform world -> object.
    Xo = (R_world_from_object.T @ (points_world - center_world).T).T
    q_lo = np.percentile(Xo, lo, axis=0)
    q_hi = np.percentile(Xo, hi, axis=0)
    half = 0.5 * (q_hi - q_lo)
    return np.maximum(half, 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--object_frame", required=True, type=Path)
    ap.add_argument("--images", required=True, type=Path)
    ap.add_argument("--points", required=True, type=Path)
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--preserve_scale", action="store_true")
    ap.add_argument("--preserve_translation", action="store_true")
    ap.add_argument("--enforce_cylinder", action="store_true")
    ap.add_argument("--extent_lo", type=float, default=2.0)
    ap.add_argument("--extent_hi", type=float, default=98.0)
    ap.add_argument("--max_z_change_deg", type=float, default=8.0)
    ap.add_argument("--max_yaw_change_deg", type=float, default=15.0)
    ap.add_argument("--yaw_source", choices=["camera", "pca2"], default="camera")
    args = ap.parse_args()

    obj = json.loads(args.object_frame.read_text())
    R_prev = np.array(obj["rotation_world_from_object"], dtype=np.float64)
    t_prev = np.array(obj["translation_world_from_object"], dtype=np.float64)
    half_prev = np.array(obj["scale_half"], dtype=np.float64)

    points = load_points(args.points)
    rows = load_images_txt(args.images)

    center = t_prev.copy() if args.preserve_translation else np.median(points, axis=0)

    # 1) Lock can axis to PCA major axis, but preserve sign consistency with previous z-axis.
    X = points - center
    cov = np.cov(X.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    z_axis_raw = normalize(eigvecs[:, 0])
    if z_axis_raw is None:
        raise RuntimeError("Failed to estimate PCA axis from points.")
    if np.dot(z_axis_raw, R_prev[:, 2]) < 0:
        z_axis_raw = -z_axis_raw

    z_delta = axis_angle_deg(z_axis_raw, R_prev[:, 2])
    if z_delta > args.max_z_change_deg:
        # Point-cloud PCA can drift/noise; avoid large sudden axis jumps.
        z_axis = R_prev[:, 2].copy()
        z_clamped = True
    else:
        z_axis = z_axis_raw
        z_clamped = False

    # 2) Resolve yaw around can axis.
    if args.yaw_source == "camera":
        v_sum = np.zeros(3, dtype=np.float64)
        for row in rows:
            Rcw = qvec_to_rotmat(row["qvec"])
            tcw = row["tvec"]
            Cw = -Rcw.T @ tcw
            v = Cw - center
            v = v - (v @ z_axis) * z_axis
            nv = np.linalg.norm(v)
            if nv > 1e-12:
                v_sum += v / nv
        x_target = normalize(v_sum)
    else:
        # Secondary PCA axis can be more stable than camera-mean for symmetric objects.
        pca2 = eigvecs[:, 1]
        pca2 = pca2 - (pca2 @ z_axis) * z_axis
        x_target = normalize(pca2)
    if x_target is None:
        x_target = R_prev[:, 0] - (R_prev[:, 0] @ z_axis) * z_axis
        x_target = normalize(x_target)
    if x_target is None:
        trial = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        x_target = normalize(trial - (trial @ z_axis) * z_axis)
    if x_target is None:
        trial = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        x_target = normalize(trial - (trial @ z_axis) * z_axis)
    if x_target is None:
        raise RuntimeError("Failed to resolve stable x-axis.")

    prev_x_proj = R_prev[:, 0] - (R_prev[:, 0] @ z_axis) * z_axis
    prev_x_proj = normalize(prev_x_proj)
    if prev_x_proj is None:
        prev_x_proj = x_target

    # Clamp yaw update around can axis to prevent hard flips.
    # Signed angle from prev_x_proj -> x_target around z_axis.
    cross_term = np.cross(prev_x_proj, x_target)
    sinv = float(np.dot(cross_term, z_axis))
    cosv = float(np.clip(np.dot(prev_x_proj, x_target), -1.0, 1.0))
    yaw_delta = float(np.degrees(np.arctan2(sinv, cosv)))
    yaw_delta_clamped = float(np.clip(yaw_delta, -args.max_yaw_change_deg, args.max_yaw_change_deg))
    x_axis = normalize(rotate_about_axis(prev_x_proj, z_axis, yaw_delta_clamped))
    if x_axis is None:
        x_axis = x_target

    y_axis = normalize(np.cross(z_axis, x_axis))
    if y_axis is None:
        raise RuntimeError("Degenerate y-axis during stabilization.")
    x_axis = normalize(np.cross(y_axis, z_axis))

    # 3) Symmetry-safe sign locking: choose between [x,y] and [-x,-y] using previous frame.
    score_same = float(np.dot(x_axis, R_prev[:, 0]) + np.dot(y_axis, R_prev[:, 1]))
    score_flip = float(np.dot(-x_axis, R_prev[:, 0]) + np.dot(-y_axis, R_prev[:, 1]))
    if score_flip > score_same:
        x_axis = -x_axis
        y_axis = -y_axis

    R_new = np.stack([x_axis, y_axis, z_axis], axis=1)
    if np.linalg.det(R_new) < 0:
        y_axis = -y_axis
        R_new = np.stack([x_axis, y_axis, z_axis], axis=1)

    if args.preserve_scale:
        half = half_prev.copy()
    else:
        half = robust_half_extents(points, center, R_new, lo=args.extent_lo, hi=args.extent_hi)
        if args.enforce_cylinder:
            r = 0.5 * (half[0] + half[1])
            half[0] = r
            half[1] = r

    key_obj = build_keypoints(half)

    # Diagnostics.
    diag = {
        "preserve_scale": bool(args.preserve_scale),
        "preserve_translation": bool(args.preserve_translation),
        "enforce_cylinder": bool(args.enforce_cylinder),
        "points_count": int(points.shape[0]),
        "pca_eigenvalues": eigvals.tolist(),
        "yaw_source": args.yaw_source,
        "z_axis_raw_change_deg": z_delta,
        "z_axis_clamped": bool(z_clamped),
        "yaw_delta_raw_deg": yaw_delta,
        "yaw_delta_applied_deg": yaw_delta_clamped,
        "angle_change_deg": {
            "x": axis_angle_deg(R_prev[:, 0], R_new[:, 0]),
            "y": axis_angle_deg(R_prev[:, 1], R_new[:, 1]),
            "z": axis_angle_deg(R_prev[:, 2], R_new[:, 2]),
        },
    }

    obj["rotation_world_from_object"] = R_new.tolist()
    obj["translation_world_from_object"] = center.tolist()
    obj["scale_half"] = half.tolist()
    obj["scale_full"] = (2.0 * half).tolist()
    obj["stabilized_cylinder_frame"] = diag

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "object_frame.json").write_text(json.dumps(obj, indent=2))
    np.save(args.out_dir / "cuboid_3d_keypoints.npy", key_obj)
    (args.out_dir / "cuboid_3d_keypoints.json").write_text(
        json.dumps({"keypoints_object": key_obj.tolist()}, indent=2)
    )

    print("stabilized_frame", json.dumps(diag, indent=2))


if __name__ == "__main__":
    main()
