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


def load_images_txt(path: Path):
    images = []
    lines = [ln.rstrip('\n') for ln in path.read_text().splitlines()]
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith('#'):
            i += 1
            continue
        parts = line.split()
        if len(parts) < 10:
            i += 1
            continue
        image_id = int(parts[0])
        qvec = np.array(list(map(float, parts[1:5])), dtype=np.float64)
        tvec = np.array(list(map(float, parts[5:8])), dtype=np.float64)
        cam_id = int(parts[8])
        name = parts[9]
        pts_line = lines[i + 1].strip() if i + 1 < len(lines) else ''
        pts2d = None
        if pts_line:
            try:
                pts = np.array(list(map(float, pts_line.split())), dtype=np.float64).reshape(-1, 3)
                pts2d = pts[:, :2]
            except Exception:
                pts2d = None
        images.append({
            "image_id": image_id,
            "qvec": qvec,
            "tvec": tvec,
            "camera_id": cam_id,
            "name": name,
            "pts2d": pts2d,
        })
        i += 2
    return images


def load_cameras_txt(path: Path):
    cameras = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            cam_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = list(map(float, parts[4:]))
            cameras[cam_id] = {
                "model": model,
                "width": width,
                "height": height,
                "params": params,
            }
    return cameras


def camera_intrinsics(cam):
    model = cam["model"]
    p = cam["params"]
    if model in ("SIMPLE_PINHOLE",):
        f, cx, cy = p
        fx = fy = f
        dist = np.zeros(4, dtype=np.float64)
    elif model in ("PINHOLE",):
        fx, fy, cx, cy = p
        dist = np.zeros(4, dtype=np.float64)
    elif model in ("SIMPLE_RADIAL", "RADIAL", "OPENCV", "OPENCV_FISHEYE"):
        fx, fy, cx, cy = p[:4]
        if model == "SIMPLE_RADIAL":
            k1 = p[4] if len(p) > 4 else 0.0
            dist = np.array([k1, 0.0, 0.0, 0.0], dtype=np.float64)
        elif model == "RADIAL":
            k1 = p[4] if len(p) > 4 else 0.0
            k2 = p[5] if len(p) > 5 else 0.0
            dist = np.array([k1, k2, 0.0, 0.0], dtype=np.float64)
        else:
            k1 = p[4] if len(p) > 4 else 0.0
            k2 = p[5] if len(p) > 5 else 0.0
            p1 = p[6] if len(p) > 6 else 0.0
            p2 = p[7] if len(p) > 7 else 0.0
            dist = np.array([k1, k2, p1, p2], dtype=np.float64)
    else:
        raise ValueError(f"Unsupported camera model: {model}")
    return fx, fy, cx, cy, dist


def qvec_to_rotmat(qvec):
    qw, qx, qy, qz = qvec
    return np.array([
        [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw,     1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw,     2*qy*qz + 2*qx*qw,     1 - 2*qx*qx - 2*qy*qy],
    ], dtype=np.float64)


def build_keypoints(half: np.ndarray) -> np.ndarray:
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


def project_bbox(R_wo, t_wo, key_obj, R_cw, t_cw, fx, fy, cx, cy, dist):
    Xw = (R_wo @ key_obj.T) + t_wo.reshape(3, 1)
    Xc = (R_cw @ Xw) + t_cw.reshape(3, 1)
    Xc = Xc.T
    if np.any(Xc[:, 2] <= 0):
        return None
    k1, k2, p1, p2 = dist
    x = Xc[:, 0] / Xc[:, 2]
    y = Xc[:, 1] / Xc[:, 2]
    r2 = x * x + y * y
    r4 = r2 * r2
    radial = 1.0 + k1 * r2 + k2 * r4
    x_dist = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    y_dist = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    u = fx * x_dist + cx
    v = fy * y_dist + cy
    return np.array([u.min(), v.min(), u.max(), v.max()])


def bbox_iou(b1, b2):
    # b: [xmin, ymin, xmax, ymax]
    ixmin = max(b1[0], b2[0])
    iymin = max(b1[1], b2[1])
    ixmax = min(b1[2], b2[2])
    iymax = min(b1[3], b2[3])
    iw = max(0.0, ixmax - ixmin)
    ih = max(0.0, iymax - iymin)
    inter = iw * ih
    a1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
    a2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
    union = a1 + a2 - inter
    if union <= 0.0:
        return 0.0
    return inter / union


EDGES = (
    (1, 5), (2, 6), (3, 7), (4, 8),
    (1, 3), (5, 7), (2, 4), (6, 8),
    (1, 2), (3, 4), (5, 6), (7, 8),
)


def point_segment_dist(p, a, b):
    # p, a, b: (2,)
    ab = b - a
    denom = ab @ ab
    if denom <= 1e-12:
        return np.linalg.norm(p - a)
    t = np.clip(((p - a) @ ab) / denom, 0.0, 1.0)
    proj = a + t * ab
    return np.linalg.norm(p - proj)


def edge_fit_cost(pts2d, proj_pts):
    # pts2d: (N,2), proj_pts: (9,2)
    # Use distances to nearest edge, robust median
    dists = []
    for p in pts2d:
        best = 1e9
        for s, e in EDGES:
            a = proj_pts[s]
            b = proj_pts[e]
            d = point_segment_dist(p, a, b)
            if d < best:
                best = d
        dists.append(best)
    if not dists:
        return 1e9
    dists = np.array(dists)
    return float(np.median(dists))


def bbox_error(b1, b2):
    # L2 on bbox corners
    return np.linalg.norm(b1 - b2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--points3d", required=True, type=Path)
    ap.add_argument("--images", required=True, type=Path)
    ap.add_argument("--cameras", required=True, type=Path)
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--extent_percentile", type=float, default=2.0)
    ap.add_argument("--radius_percentile", type=float, default=98.0)
    ap.add_argument("--yaw_step_deg", type=float, default=5.0)
    ap.add_argument("--scale_min", type=float, default=0.6)
    ap.add_argument("--scale_max", type=float, default=1.4)
    ap.add_argument("--scale_steps", type=int, default=15)
    ap.add_argument("--max_frames", type=int, default=120)
    ap.add_argument("--max_points", type=int, default=200)
    ap.add_argument("--cost", choices=["bbox_iou", "edge"], default="edge")
    args = ap.parse_args()

    points = load_points3d_txt(args.points3d)
    centroid = points.mean(axis=0)
    centered = points - centroid

    # PCA
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]
    if np.linalg.det(eigvecs) < 0:
        eigvecs[:, 2] *= -1.0

    # PCA axes (unordered)
    axes = [eigvecs[:, 0], eigvecs[:, 1], eigvecs[:, 2]]

    # Use PCA axis 0 as default height to estimate extents
    proj_z = centered @ axes[0]
    lo = args.extent_percentile
    hi = 100.0 - args.extent_percentile
    z_min = np.percentile(proj_z, lo)
    z_max = np.percentile(proj_z, hi)
    half_h = max(abs(z_min), abs(z_max))

    # Radius in plane
    proj_x = centered @ axes[1]
    proj_y = centered @ axes[2]
    r = np.sqrt(proj_x**2 + proj_y**2)
    radius = np.percentile(r, args.radius_percentile)

    base_half = np.array([radius, radius, half_h], dtype=np.float64)

    cameras = load_cameras_txt(args.cameras)
    images = load_images_txt(args.images)

    # Precompute 2D bboxes from observed points
    frame_data = []
    for im in images:
        if im["pts2d"] is None:
            continue
        pts2d = im["pts2d"]
        if len(pts2d) < 10:
            continue
        cam = cameras[im["camera_id"]]
        fx, fy, cx, cy, dist = camera_intrinsics(cam)
        if len(pts2d) > args.max_points:
            idx = np.random.choice(len(pts2d), size=args.max_points, replace=False)
            pts2d = pts2d[idx]
        b2d = np.array([pts2d[:, 0].min(), pts2d[:, 1].min(), pts2d[:, 0].max(), pts2d[:, 1].max()])
        frame_data.append((im, fx, fy, cx, cy, dist, b2d, pts2d))
        if len(frame_data) >= args.max_frames:
            break

    if not frame_data:
        raise ValueError("No usable frames with 2D points for optimization")

    best = None
    best_theta = None
    best_sr = None
    best_sh = None

    scales = np.linspace(args.scale_min, args.scale_max, args.scale_steps)

    # Try all axis permutations and sign flips (det > 0)
    perms = [
        (0, 1, 2),
        (0, 2, 1),
        (1, 0, 2),
        (1, 2, 0),
        (2, 0, 1),
        (2, 1, 0),
    ]
    sign_sets = [
        (1, 1, 1),
        (1, 1, -1),
        (1, -1, 1),
        (1, -1, -1),
        (-1, 1, 1),
        (-1, 1, -1),
        (-1, -1, 1),
        (-1, -1, -1),
    ]

    for perm in perms:
        base_axes = [axes[perm[0]], axes[perm[1]], axes[perm[2]]]
        for sx, sy, sz in sign_sets:
            axis_x0 = base_axes[0] * sx
            axis_y0 = base_axes[1] * sy
            axis_z = base_axes[2] * sz
            R0 = np.stack([axis_x0, axis_y0, axis_z], axis=1)
            if np.linalg.det(R0) < 0:
                continue

            for deg in np.arange(0.0, 180.0, args.yaw_step_deg):
                theta = np.deg2rad(deg)
                c, s = np.cos(theta), np.sin(theta)
                axis_x = c * axis_x0 + s * axis_y0
                axis_y = -s * axis_x0 + c * axis_y0
                R_wo = np.stack([axis_x, axis_y, axis_z], axis=1)

                for sr in scales:
                    for sh in scales:
                        half = base_half * np.array([sr, sr, sh], dtype=np.float64)
                        key_obj = build_keypoints(half)
                        total = 0.0
                        count = 0
                        for im, fx, fy, cx, cy, dist, b2d, pts2d in frame_data:
                            R_cw = qvec_to_rotmat(im["qvec"])
                            t_cw = im["tvec"]
                            bproj = project_bbox(R_wo, centroid, key_obj, R_cw, t_cw, fx, fy, cx, cy, dist)
                            if bproj is None:
                                continue
                            if args.cost == "bbox_iou":
                                total += 1.0 - bbox_iou(bproj, b2d)
                            else:
                                # project full points for edge fit
                                Xw = (R_wo @ key_obj.T) + centroid.reshape(3, 1)
                                Xc = (R_cw @ Xw) + t_cw.reshape(3, 1)
                                Xc = Xc.T
                                k1, k2, p1, p2 = dist
                                x = Xc[:, 0] / Xc[:, 2]
                                y = Xc[:, 1] / Xc[:, 2]
                                r2 = x * x + y * y
                                r4 = r2 * r2
                                radial = 1.0 + k1 * r2 + k2 * r4
                                x_dist = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
                                y_dist = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
                                u = fx * x_dist + cx
                                v = fy * y_dist + cy
                                proj_pts = np.stack([u, v], axis=1)
                                total += edge_fit_cost(pts2d, proj_pts)
                            count += 1
                        if count == 0:
                            continue
                        avg = total / count
                        if best is None or avg < best:
                            best = avg
                            best_theta = (perm, (sx, sy, sz), deg)
                            best_sr = sr
                            best_sh = sh

    # Build final R_wo with best yaw
    # Build final R_wo from best parameters
    if best_theta is None:
        perm = (0, 1, 2)
        sx, sy, sz = (1, 1, 1)
        yaw_deg = 0.0
    else:
        perm, (sx, sy, sz), yaw_deg = best_theta

    axis_x0 = axes[perm[0]] * sx
    axis_y0 = axes[perm[1]] * sy
    axis_z = axes[perm[2]] * sz
    theta = np.deg2rad(yaw_deg)
    c, s = np.cos(theta), np.sin(theta)
    axis_x = c * axis_x0 + s * axis_y0
    axis_y = -s * axis_x0 + c * axis_y0
    R_wo = np.stack([axis_x, axis_y, axis_z], axis=1)
    if best_sr is None:
        best_sr = 1.0
    if best_sh is None:
        best_sh = 1.0
    half = base_half * np.array([best_sr, best_sr, best_sh], dtype=np.float64)
    key_obj = build_keypoints(half)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    np.save(args.out_dir / "cuboid_3d_keypoints.npy", key_obj)

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
            "translation_world_from_object": centroid.tolist(),
            "scale_full": (half * 2.0).tolist(),
            "scale_half": half.tolist(),
            "best_axes_perm": list(perm),
            "best_axes_sign": [sx, sy, sz],
            "best_yaw_deg": yaw_deg,
            "best_scale_radius": best_sr,
            "best_scale_height": best_sh
        }, f, indent=2)

    with (args.out_dir / "cuboid_3d_keypoints.json").open("w") as f:
        json.dump({"keypoints_object": key_obj.tolist()}, f, indent=2)

    print("best_perm", perm, "best_sign", (sx, sy, sz), "best_yaw_deg", yaw_deg, "best_sr", best_sr, "best_sh", best_sh)
    print("Wrote:")
    print(args.out_dir / "cuboid_3d_keypoints.npy")
    print(args.out_dir / "cuboid_3d_keypoints.json")
    print(args.out_dir / "keypoint_order.json")
    print(args.out_dir / "object_frame.json")


if __name__ == "__main__":
    main()
