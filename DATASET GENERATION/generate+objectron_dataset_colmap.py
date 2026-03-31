import argparse
import io
import json
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

try:
    from scipy.optimize import least_squares
except Exception:
    least_squares = None


@dataclass
class Camera:
    camera_id: int
    model: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    dist: np.ndarray


@dataclass
class ImagePose:
    image_id: int
    qvec: np.ndarray
    tvec: np.ndarray
    camera_id: int
    name: str
    line_index: int


@dataclass
class MaskObservation:
    pose: ImagePose
    cam: Camera
    bbox_xyxy: np.ndarray


def qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = qvec
    return np.array(
        [
            [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
            [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qx * qw],
            [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx * qx - 2 * qy * qy],
        ],
        dtype=np.float64,
    )


def read_cameras(path: Path):
    cams = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        p = line.split()
        camera_id = int(p[0])
        model = p[1]
        width = int(p[2])
        height = int(p[3])
        params = list(map(float, p[4:]))
        dist = np.zeros(4, dtype=np.float64)

        if model == "SIMPLE_PINHOLE":
            f, cx, cy = params[:3]
            fx = fy = f
        elif model == "PINHOLE":
            fx, fy, cx, cy = params[:4]
        elif model == "SIMPLE_RADIAL":
            f, cx, cy, k1 = params[:4]
            fx = fy = f
            dist = np.array([k1, 0.0, 0.0, 0.0], dtype=np.float64)
        elif model == "RADIAL":
            f, cx, cy, k1, k2 = params[:5]
            fx = fy = f
            dist = np.array([k1, k2, 0.0, 0.0], dtype=np.float64)
        elif model in ("OPENCV", "FULL_OPENCV"):
            fx, fy, cx, cy = params[:4]
            if len(params) >= 8:
                dist = np.array([params[4], params[5], params[6], params[7]], dtype=np.float64)
        elif model == "OPENCV_FISHEYE":
            fx, fy, cx, cy = params[:4]
            if len(params) >= 8:
                dist = np.array([params[4], params[5], params[6], params[7]], dtype=np.float64)
        else:
            raise ValueError(f"Unsupported camera model: {model}")

        cams[camera_id] = Camera(
            camera_id=camera_id,
            model=model,
            width=width,
            height=height,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            dist=dist,
        )

    if not cams:
        raise ValueError(f"No cameras found in {path}")
    return cams


def read_images(path: Path):
    poses = []
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

        poses.append(
            ImagePose(
                image_id=int(p[0]),
                qvec=np.array(list(map(float, p[1:5])), dtype=np.float64),
                tvec=np.array(list(map(float, p[5:8])), dtype=np.float64),
                camera_id=int(p[8]),
                name=p[9],
                line_index=len(poses),
            )
        )

        # IMPORTANT: images.txt has two lines per image.
        i += 2

    if not poses:
        raise ValueError(f"No image poses found in {path}")
    return poses


def frame_number_from_name(name: str):
    stem = Path(name).stem
    m = re.search(r"(\d+)$", stem)
    if m:
        return int(m.group(1))
    return None


def sort_poses(poses, mode: str):
    if mode == "image_id":
        return sorted(poses, key=lambda r: r.image_id)
    if mode == "line":
        return sorted(poses, key=lambda r: r.line_index)

    # mode == "name"
    def key_fn(r):
        n = frame_number_from_name(r.name)
        if n is None:
            return (1, r.name)
        return (0, n)

    return sorted(poses, key=key_fn)


def camera_center_world(pose: ImagePose) -> np.ndarray:
    Rcw = qvec2rotmat(pose.qvec)
    return -Rcw.T @ pose.tvec.reshape(3, 1)


def estimate_world_up(poses):
    # COLMAP/OpenCV camera coordinates have +y downward, so camera-up is -Y axis.
    ups = []
    for pose in poses:
        Rcw = qvec2rotmat(pose.qvec)
        up = -Rcw.T[:, 1]
        n = float(np.linalg.norm(up))
        if n > 1e-12:
            ups.append(up / n)

    if not ups:
        return np.array([0.0, 0.0, -1.0], dtype=np.float64)

    world_up = np.mean(np.asarray(ups, dtype=np.float64), axis=0)
    n = float(np.linalg.norm(world_up))
    if n < 1e-12:
        return np.array([0.0, 0.0, -1.0], dtype=np.float64)
    return world_up / n


# Objectron mapping requested by user:
# 0 center at (0,0.5,0), then 1..8 as front/back + bottom/top.
def objectron_unit_box_mapping():
    return np.array(
        [
            [0.0, 0.5, 0.0],   # 0 center
            [-0.5, 0.0, +0.5], # 1 front-bottom-left
            [+0.5, 0.0, +0.5], # 2 front-bottom-right
            [+0.5, 1.0, +0.5], # 3 front-top-right
            [-0.5, 1.0, +0.5], # 4 front-top-left
            [-0.5, 0.0, -0.5], # 5 rear-bottom-left
            [+0.5, 0.0, -0.5], # 6 rear-bottom-right
            [+0.5, 1.0, -0.5], # 7 rear-top-right
            [-0.5, 1.0, -0.5], # 8 rear-top-left
        ],
        dtype=np.float64,
    )


def compute_cuboid(points_world: np.ndarray, world_up: np.ndarray, front_hint: np.ndarray):
    points_world = np.asarray(points_world, dtype=np.float64)
    if points_world.shape[0] < 3:
        raise ValueError("Not enough points to compute cuboid")

    mean_xyz = points_world.mean(axis=0)
    X = points_world - mean_xyz
    cov = np.cov(X.T)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    R_obb = vecs[:, order]
    proj = X @ R_obb
    minp = proj.min(axis=0)
    maxp = proj.max(axis=0)
    extent_obb = maxp - minp
    center = mean_xyz + R_obb @ ((minp + maxp) * 0.5)

    # Map OBB axes into semantic object axes:
    # y = height (largest extent), x/z = remaining two.
    idx_y = int(np.argmax(extent_obb))
    rem = [i for i in range(3) if i != idx_y]
    idx_a, idx_b = rem[0], rem[1]

    a = R_obb[:, idx_a].copy()
    y = R_obb[:, idx_y].copy()
    b = R_obb[:, idx_b].copy()
    ea = float(extent_obb[idx_a])
    ey = float(extent_obb[idx_y])
    eb = float(extent_obb[idx_b])

    # Keep top direction consistent with estimated global up.
    world_up = np.asarray(world_up, dtype=np.float64).reshape(3)
    if float(np.linalg.norm(world_up)) < 1e-12:
        world_up = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    world_up = world_up / (np.linalg.norm(world_up) + 1e-12)
    if float(np.dot(y, world_up)) < 0:
        y = -y

    # Pick front axis (+z) to point toward the first selected camera.
    z_hint = None
    if front_hint is not None:
        front_hint = np.asarray(front_hint, dtype=np.float64).reshape(3)
        v = front_hint - center
        v = v - float(np.dot(v, y)) * y
        nv = float(np.linalg.norm(v))
        if nv > 1e-12:
            z_hint = v / nv

    if z_hint is None:
        x = a
        z = b
        ex, ez = ea, eb
    else:
        if abs(float(np.dot(a, z_hint))) >= abs(float(np.dot(b, z_hint))):
            z = a
            x = b
            ez, ex = ea, eb
        else:
            z = b
            x = a
            ez, ex = eb, ea
        if float(np.dot(z, z_hint)) < 0:
            z = -z

    # Rebuild right-handed frame.
    y = y / (np.linalg.norm(y) + 1e-12)
    x = np.cross(y, z)
    x = x / (np.linalg.norm(x) + 1e-12)
    z = np.cross(x, y)
    z = z / (np.linalg.norm(z) + 1e-12)

    R = np.stack([x, y, z], axis=1)
    extent = np.array([ex, ey, ez], dtype=np.float64)
    return center, R, extent


def build_world_keypoints(center: np.ndarray, R: np.ndarray, extent: np.ndarray):
    tpl = objectron_unit_box_mapping()
    # Template is defined in [0,1] for y with center at y=0.5.
    # Recenter before scaling so world center remains at OBB center.
    tpl_centered = tpl - np.array([0.0, 0.5, 0.0], dtype=np.float64)
    local = tpl_centered * extent.reshape(1, 3)
    return (R @ local.T).T + center.reshape(1, 3)


def build_K(cam: Camera):
    return np.array(
        [[cam.fx, 0.0, cam.cx], [0.0, cam.fy, cam.cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def read_ply_ascii_points(path: Path):
    with path.open("r") as f:
        format_ascii = False
        num_verts = None
        props = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Unexpected EOF while reading PLY header: {path}")
            s = line.strip()
            if s.startswith("format"):
                if "ascii" in s:
                    format_ascii = True
            elif s.startswith("element vertex"):
                num_verts = int(s.split()[-1])
            elif s.startswith("property"):
                parts = s.split()
                if len(parts) >= 3:
                    props.append(parts[2])
            elif s == "end_header":
                break

        if not format_ascii:
            raise ValueError(f"PLY is not ascii. Re-export with ascii format: {path}")
        if num_verts is None:
            raise ValueError(f"PLY missing vertex count: {path}")

        lines = [f.readline() for _ in range(num_verts)]
        data = np.loadtxt(io.StringIO("".join(lines)), dtype=np.float64)
        if data.ndim == 1:
            data = data[None, :]

    def find_idx(name):
        return props.index(name) if name in props else None

    xi = find_idx("x")
    yi = find_idx("y")
    zi = find_idx("z")
    if xi is None or yi is None or zi is None:
        raise ValueError(f"PLY missing x/y/z properties: {path}")

    return data[:, [xi, yi, zi]].astype(np.float64)


def project(points_world: np.ndarray, cam: Camera, pose: ImagePose):
    Rcw = qvec2rotmat(pose.qvec)
    tvec = pose.tvec.reshape(3, 1)

    # camera-space coordinates and depth
    Xc = (Rcw @ points_world.T) + tvec
    Xc = Xc.T
    depth = Xc[:, 2]

    K = build_K(cam)
    rvec, _ = cv2.Rodrigues(Rcw)

    if cam.model == "OPENCV_FISHEYE":
        uv, _ = cv2.fisheye.projectPoints(
            points_world.reshape(-1, 1, 3).astype(np.float64),
            rvec,
            tvec,
            K,
            cam.dist.reshape(4, 1),
        )
    else:
        uv, _ = cv2.projectPoints(
            points_world.reshape(-1, 1, 3).astype(np.float64),
            rvec,
            tvec,
            K,
            cam.dist,
        )

    return uv.reshape(-1, 2), Xc, depth


def compute_visibility(uv: np.ndarray, depth: np.ndarray, width: int, height: int):
    vis = []
    for (u, v), z in zip(uv, depth):
        if z > 1e-8 and 0 <= u < width and 0 <= v < height:
            vis.append(1.0)
        else:
            vis.append(0.0)
    return vis


CUBOID_EDGES = [
    # Front face
    (1, 2), (2, 3), (3, 4), (4, 1),
    # Rear face
    (5, 6), (6, 7), (7, 8), (8, 5),
    # Side connectors
    (1, 5), (2, 6), (3, 7), (4, 8),
]


def draw_overlay(image: np.ndarray, uv: np.ndarray):
    uv = uv.astype(np.int32)

    for i, j in CUBOID_EDGES:
        p1 = tuple(uv[i])
        p2 = tuple(uv[j])
        cv2.line(image, p1, p2, (0, 255, 0), 2, cv2.LINE_AA)

    for i, p in enumerate(uv):
        color = (0, 0, 255) if i == 0 else (255, 0, 0)
        pp = tuple(p)
        cv2.circle(image, pp, 4, color, -1, cv2.LINE_AA)
        cv2.putText(image, str(i), (pp[0] + 5, pp[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return image


def trim_points_along_axis(points: np.ndarray, axis: np.ndarray, trim: float, arg_name: str = "--height_trim"):
    if trim <= 0.0:
        return points
    trim = float(trim)
    if trim >= 0.49:
        raise ValueError(f"{arg_name} must be < 0.49")
    axis = np.asarray(axis, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(axis))
    if n < 1e-12:
        return points
    axis = axis / n
    proj = points @ axis
    lo = np.quantile(proj, trim)
    hi = np.quantile(proj, 1.0 - trim)
    mask = (proj >= lo) & (proj <= hi)
    kept = points[mask]
    if kept.shape[0] < max(100, int(0.1 * points.shape[0])):
        # Don't over-trim if it would discard too much.
        return points
    return kept


def trim_points_along_up(points: np.ndarray, up: np.ndarray, trim: float):
    return trim_points_along_axis(points, axis=up, trim=trim, arg_name="--height_trim")


def trim_points_lateral(points: np.ndarray, center: np.ndarray, R: np.ndarray, trim: float):
    """Trim lateral (x/z) tails in object coordinates to tighten can width/depth."""
    if trim <= 0.0:
        return points
    trim = float(trim)
    if trim >= 0.49:
        raise ValueError("--horizontal_trim must be < 0.49")

    points = np.asarray(points, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64).reshape(1, 3)
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)

    local = (R.T @ (points - center).T).T
    x = local[:, 0]
    z = local[:, 2]

    x_lo, x_hi = np.quantile(x, trim), np.quantile(x, 1.0 - trim)
    z_lo, z_hi = np.quantile(z, trim), np.quantile(z, 1.0 - trim)

    mask = (x >= x_lo) & (x <= x_hi) & (z >= z_lo) & (z <= z_hi)
    kept = points[mask]

    if kept.shape[0] < max(100, int(0.1 * points.shape[0])):
        return points
    return kept


def mask_bbox_from_file(mask_path: Path):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    ys, xs = np.where(mask > 0)
    if xs.size < 16:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    if x1 <= x0 or y1 <= y0:
        return None
    return np.array([x0, y0, x1, y1], dtype=np.float64)


def bbox_from_uv(uv: np.ndarray):
    uv = np.asarray(uv, dtype=np.float64)
    if uv.ndim != 2 or uv.shape[0] == 0:
        return None
    x0, y0 = np.min(uv[:, 0]), np.min(uv[:, 1])
    x1, y1 = np.max(uv[:, 0]), np.max(uv[:, 1])
    if not np.isfinite([x0, y0, x1, y1]).all():
        return None
    if x1 <= x0 or y1 <= y0:
        return None
    return np.array([x0, y0, x1, y1], dtype=np.float64)


def bbox_iou(a: np.ndarray, b: np.ndarray):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    iw = max(0.0, ix1 - ix0)
    ih = max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    den = area_a + area_b - inter
    if den <= 1e-12:
        return 0.0
    return float(inter / den)


def collect_mask_observations(
    poses,
    cams,
    masks_dir: Path,
    sample_stride: int,
    max_frames: int,
    min_area_frac: float = 0.002,
):
    obs = []
    if masks_dir is None or not masks_dir.exists():
        return obs

    stride = max(1, int(sample_stride))
    max_frames = max(1, int(max_frames))

    for idx, pose in enumerate(poses):
        if idx % stride != 0:
            continue
        if len(obs) >= max_frames:
            break
        cam = cams.get(pose.camera_id)
        if cam is None:
            continue
        mask_path = masks_dir / f"{Path(pose.name).stem}.png"
        bbox = mask_bbox_from_file(mask_path)
        if bbox is None:
            continue
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        if area < (min_area_frac * cam.width * cam.height):
            continue
        obs.append(MaskObservation(pose=pose, cam=cam, bbox_xyxy=bbox))
    return obs


def orthonormalize_rotation(R: np.ndarray):
    U, _, Vt = np.linalg.svd(R)
    Rn = U @ Vt
    if np.linalg.det(Rn) < 0:
        U[:, -1] *= -1.0
        Rn = U @ Vt
    return Rn


def apply_cuboid_delta(base_center, base_R, base_extent, params):
    dt = params[0:3]
    drot = params[3:6]
    dscale = params[6:9]

    dR, _ = cv2.Rodrigues(drot.reshape(3, 1).astype(np.float64))
    R = orthonormalize_rotation(dR @ base_R)
    center = base_center + dt
    extent = base_extent * np.exp(dscale)
    extent = np.clip(extent, 1e-4, 1e6)
    return center, R, extent


def evaluate_mask_fit(obs, center, R, extent):
    if not obs:
        return {"count": 0, "median_iou": 0.0, "mean_iou": 0.0}
    kpts = build_world_keypoints(center, R, extent)
    ious = []
    for ob in obs:
        uv, _, depth = project(kpts, ob.cam, ob.pose)
        if np.sum(depth[1:] > 1e-8) < 4:
            continue
        pred_bbox = bbox_from_uv(uv[1:])
        if pred_bbox is None:
            continue
        ious.append(bbox_iou(pred_bbox, ob.bbox_xyxy))
    if not ious:
        return {"count": 0, "median_iou": 0.0, "mean_iou": 0.0}
    ious = np.asarray(ious, dtype=np.float64)
    return {
        "count": int(ious.size),
        "median_iou": float(np.median(ious)),
        "mean_iou": float(np.mean(ious)),
    }


def refine_cuboid_with_masks(
    poses,
    cams,
    masks_dir: Path,
    center: np.ndarray,
    R: np.ndarray,
    extent: np.ndarray,
    sample_stride: int,
    max_frames: int,
):
    obs = collect_mask_observations(
        poses=poses,
        cams=cams,
        masks_dir=masks_dir,
        sample_stride=sample_stride,
        max_frames=max_frames,
    )
    if len(obs) < 12:
        print(f"Mask-fit skipped: only {len(obs)} valid mask observations")
        return center, R, extent

    base_center = center.astype(np.float64).reshape(3)
    base_R = orthonormalize_rotation(R.astype(np.float64).reshape(3, 3))
    base_extent = extent.astype(np.float64).reshape(3)
    extent_scale = max(float(np.mean(base_extent)), 1e-6)

    def residual(params, reg_cfg):
        c, Rn, en = apply_cuboid_delta(base_center, base_R, base_extent, params)
        kpts = build_world_keypoints(c, Rn, en)
        res = []
        for ob in obs:
            uv, _, depth = project(kpts, ob.cam, ob.pose)
            if np.sum(depth[1:] > 1e-8) < 4:
                continue
            pred_bbox = bbox_from_uv(uv[1:])
            if pred_bbox is None:
                continue

            pb = pred_bbox
            mb = ob.bbox_xyxy
            pw = max(1e-6, pb[2] - pb[0])
            ph = max(1e-6, pb[3] - pb[1])
            mw = max(1e-6, mb[2] - mb[0])
            mh = max(1e-6, mb[3] - mb[1])
            pcx, pcy = 0.5 * (pb[0] + pb[2]), 0.5 * (pb[1] + pb[3])
            mcx, mcy = 0.5 * (mb[0] + mb[2]), 0.5 * (mb[1] + mb[3])

            res.append((pcx - mcx) / float(ob.cam.width))
            res.append((pcy - mcy) / float(ob.cam.height))
            res.append(np.log(pw / mw))
            res.append(np.log(ph / mh))

        if not res:
            return np.array([1e3], dtype=np.float64)

        # Regularization profile (stable vs flexible).
        reg = [
            params[0] / (reg_cfg["trans"] * extent_scale),
            params[1] / (reg_cfg["trans"] * extent_scale),
            params[2] / (reg_cfg["trans"] * extent_scale),
            params[3] / reg_cfg["rot"],
            params[4] / reg_cfg["rot"],
            params[5] / reg_cfg["rot"],
            params[6] / reg_cfg["scale"],
            params[7] / reg_cfg["scale"],
            params[8] / reg_cfg["scale"],
        ]
        return np.asarray(res + reg, dtype=np.float64)

    def cost_fn(params, reg_cfg):
        r = residual(params, reg_cfg)
        return float(np.dot(r, r))

    def optimize_fallback(p0, reg_cfg, n_iters=80):
        # Numpy-only coordinate-descent fallback when scipy is unavailable.
        p = p0.astype(np.float64).copy()
        steps = np.array(
            [
                0.08 * extent_scale,
                0.08 * extent_scale,
                0.08 * extent_scale,
                0.18,
                0.18,
                0.18,
                0.12,
                0.12,
                0.12,
            ],
            dtype=np.float64,
        )
        best_cost = cost_fn(p, reg_cfg)
        for _ in range(int(n_iters)):
            improved = False
            for j in range(9):
                for sgn in (+1.0, -1.0):
                    cand = p.copy()
                    cand[j] += sgn * steps[j]
                    c = cost_fn(cand, reg_cfg)
                    if c < best_cost:
                        p = cand
                        best_cost = c
                        improved = True
            if not improved:
                steps *= 0.65
                if float(np.max(steps)) < 1e-4:
                    break
        return {"x": p, "cost": best_cost}

    init_rotvecs = [
        np.array([0.0, 0.0, 0.0], dtype=np.float64),
        np.array([np.pi * 0.5, 0.0, 0.0], dtype=np.float64),
        np.array([-np.pi * 0.5, 0.0, 0.0], dtype=np.float64),
        np.array([0.0, np.pi * 0.5, 0.0], dtype=np.float64),
        np.array([0.0, -np.pi * 0.5, 0.0], dtype=np.float64),
        np.array([0.0, 0.0, np.pi * 0.5], dtype=np.float64),
        np.array([0.0, 0.0, -np.pi * 0.5], dtype=np.float64),
        np.array([np.pi, 0.0, 0.0], dtype=np.float64),
        np.array([0.0, np.pi, 0.0], dtype=np.float64),
        np.array([0.0, 0.0, np.pi], dtype=np.float64),
    ]

    before = evaluate_mask_fit(obs, base_center, base_R, base_extent)

    profiles = [
        {
            "name": "stable",
            "reg": {"trans": 0.25, "rot": 0.7, "scale": 0.8},
            "max_nfev": 140,
            "fallback_iters": 80,
        },
        {
            "name": "flex",
            "reg": {"trans": 1.5, "rot": 1.2, "scale": 1.2},
            "max_nfev": 220,
            "fallback_iters": 120,
        },
    ]

    def solve_profile(profile):
        reg_cfg = profile["reg"]
        best = None
        for rv in init_rotvecs:
            p0 = np.zeros(9, dtype=np.float64)
            p0[3:6] = rv
            if least_squares is None:
                sol = optimize_fallback(p0, reg_cfg, n_iters=profile["fallback_iters"])
                cost = float(sol["cost"])
                x = sol["x"]
            else:
                try:
                    sol = least_squares(
                        lambda p: residual(p, reg_cfg),
                        p0,
                        method="trf",
                        loss="huber",
                        f_scale=0.25,
                        max_nfev=profile["max_nfev"],
                    )
                except Exception:
                    continue
                cost = float(sol.cost)
                x = sol.x

            if best is None or cost < best["cost"]:
                best = {"cost": cost, "x": x}

        if best is None:
            return None

        c_ref, R_ref, e_ref = apply_cuboid_delta(base_center, base_R, base_extent, best["x"])
        after = evaluate_mask_fit(obs, c_ref, R_ref, e_ref)
        return {
            "profile": profile["name"],
            "center": c_ref,
            "R": R_ref,
            "extent": e_ref,
            "stats": after,
        }

    candidates = []
    first = solve_profile(profiles[0])
    if first is not None:
        s = first["stats"]
        print(
            "Mask-fit stats:",
            f"profile={first['profile']}",
            f"obs={s['count']}",
            f"median_iou {before['median_iou']:.3f}->{s['median_iou']:.3f}",
            f"mean_iou {before['mean_iou']:.3f}->{s['mean_iou']:.3f}",
        )
        candidates.append(first)

    run_flex = (
        first is None
        or first["stats"]["median_iou"] < 0.50
        or first["stats"]["mean_iou"] < 0.45
    )
    if run_flex:
        second = solve_profile(profiles[1])
        if second is not None:
            s = second["stats"]
            print(
                "Mask-fit stats:",
                f"profile={second['profile']}",
                f"obs={s['count']}",
                f"median_iou {before['median_iou']:.3f}->{s['median_iou']:.3f}",
                f"mean_iou {before['mean_iou']:.3f}->{s['mean_iou']:.3f}",
            )
            candidates.append(second)

    if not candidates:
        print("Mask-fit failed: optimizer did not converge")
        return center, R, extent

    def fit_score(cand):
        s = cand["stats"]
        return (s["median_iou"], s["mean_iou"], s["count"])

    best = max(candidates, key=fit_score)
    after = best["stats"]
    if after["count"] > 0 and before["count"] > 0:
        if (after["median_iou"] + 0.02) < before["median_iou"] and (after["mean_iou"] + 0.02) < before["mean_iou"]:
            print("Mask-fit rejected: quality decreased; keeping pre-fit cuboid")
            return base_center, base_R, base_extent
    print(f"Mask-fit selected profile={best['profile']}")
    return best["center"], best["R"], best["extent"]


def refine_pose_for_frame_mask(
    center: np.ndarray,
    R: np.ndarray,
    extent: np.ndarray,
    cam,
    pose,
    mask_bbox: np.ndarray,
    max_nfev: int = 40,
):
    base_center = center.astype(np.float64).reshape(3)
    base_R = orthonormalize_rotation(R.astype(np.float64).reshape(3, 3))
    extent = extent.astype(np.float64).reshape(3)
    extent_scale = max(float(np.mean(extent)), 1e-6)

    def apply_delta(params):
        dR, _ = cv2.Rodrigues(params[3:6].reshape(3, 1).astype(np.float64))
        Rn = orthonormalize_rotation(dR @ base_R)
        c = base_center + params[0:3]
        return c, Rn

    def project_bbox(c, Rn):
        kpts = build_world_keypoints(c, Rn, extent)
        uv, _, depth = project(kpts, cam, pose)
        if np.sum(depth[1:] > 1e-8) < 4:
            return None
        return bbox_from_uv(uv[1:])

    base_bbox = project_bbox(base_center, base_R)
    if base_bbox is None:
        return base_center, base_R
    base_iou = bbox_iou(base_bbox, mask_bbox)

    def residual(params):
        c, Rn = apply_delta(params)
        pred_bbox = project_bbox(c, Rn)
        if pred_bbox is None:
            return np.array([1e3], dtype=np.float64)

        pb = pred_bbox
        mb = mask_bbox
        pw = max(1e-6, pb[2] - pb[0])
        ph = max(1e-6, pb[3] - pb[1])
        mw = max(1e-6, mb[2] - mb[0])
        mh = max(1e-6, mb[3] - mb[1])
        pcx, pcy = 0.5 * (pb[0] + pb[2]), 0.5 * (pb[1] + pb[3])
        mcx, mcy = 0.5 * (mb[0] + mb[2]), 0.5 * (mb[1] + mb[3])

        res = [
            (pcx - mcx) / float(cam.width),
            (pcy - mcy) / float(cam.height),
            np.log(pw / mw),
            np.log(ph / mh),
        ]
        # Keep per-frame corrections bounded to avoid unstable jumps.
        reg = [
            params[0] / (0.8 * extent_scale),
            params[1] / (0.8 * extent_scale),
            params[2] / (0.8 * extent_scale),
            params[3] / 0.8,
            params[4] / 0.8,
            params[5] / 0.8,
        ]
        return np.asarray(res + reg, dtype=np.float64)

    p0 = np.zeros(6, dtype=np.float64)
    p_best = p0
    if least_squares is not None:
        try:
            sol = least_squares(
                residual,
                p0,
                method="trf",
                loss="huber",
                f_scale=0.25,
                max_nfev=max_nfev,
            )
            p_best = sol.x
        except Exception:
            p_best = p0

    c_ref, R_ref = apply_delta(p_best)
    ref_bbox = project_bbox(c_ref, R_ref)
    if ref_bbox is None:
        return base_center, base_R
    ref_iou = bbox_iou(ref_bbox, mask_bbox)
    if ref_iou + 1e-4 >= base_iou:
        return c_ref, R_ref
    return base_center, base_R


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--point_cloud", type=Path, required=True)
    parser.add_argument("--cameras_txt", type=Path, required=True)
    parser.add_argument("--images_txt", type=Path, required=True)
    parser.add_argument("--image_folder", type=Path, required=True)

    parser.add_argument("--overlay_dir", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--object_frame", type=Path, default=None, help="Use a precomputed object_frame.json")
    parser.add_argument("--object_frame_out", type=Path, default=None, help="Write initial object_frame.json")

    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--sort_by", choices=["name", "image_id", "line"], default="name")
    parser.add_argument("--height_trim", type=float, default=0.0, help="Trim top/bottom percent of points to tighten can height (e.g., 0.03)")
    parser.add_argument("--height_trim_mode", choices=["auto", "world_up", "object_axis"], default="auto", help="Axis for height trim: auto selects object axis when can is horizontal")
    parser.add_argument("--horizontal_trim", type=float, default=0.0, help="Trim left/right tails in object x/z axes (e.g., 0.05)")
    parser.add_argument("--masks_dir", type=Path, default=None, help="SAM2 masks directory with <image_stem>.png files")
    parser.add_argument("--fit_to_masks", action="store_true", help="Refine global cuboid pose/scale using 2D mask bboxes")
    parser.add_argument("--fit_sample_stride", type=int, default=3, help="Use every Nth frame for mask fitting")
    parser.add_argument("--fit_max_frames", type=int, default=320, help="Max frames to use for mask fitting")
    parser.add_argument("--per_frame_mask_refine", action="store_true", help="Refine per-frame pose against that frame mask bbox")
    parser.add_argument("--per_frame_refine_max_nfev", type=int, default=40, help="Max optimizer iterations for per-frame mask refine")
    parser.add_argument("--snap_bbox_to_mask", action="store_true", help="Snap projected 2D keypoints to per-frame mask bbox when misaligned")
    parser.add_argument("--snap_iou_threshold", type=float, default=0.65, help="Apply snap only when predicted-vs-mask bbox IoU is below this threshold")

    args = parser.parse_args()

    if args.stride <= 0:
        raise ValueError("--stride must be > 0")
    if args.start < 0:
        raise ValueError("--start must be >= 0")

    args.overlay_dir.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)

    print("Loading point cloud")
    points = read_ply_ascii_points(args.point_cloud)
    if points.size == 0:
        raise ValueError(f"Point cloud is empty: {args.point_cloud}")

    cams = read_cameras(args.cameras_txt)
    poses = sort_poses(read_images(args.images_txt), args.sort_by)
    poses = poses[args.start :: args.stride]

    print(f"Total selected poses: {len(poses)}")
    if not poses:
        raise ValueError("No poses selected with current --start / --stride")

    print("Fitting cuboid")
    if args.object_frame is not None:
        obj = json.loads(args.object_frame.read_text())
        R = np.array(obj["rotation_world_from_object"], dtype=np.float64)
        center = np.array(obj["translation_world_from_object"], dtype=np.float64).reshape(3)
        if "scale_half" in obj:
            half = np.array(obj["scale_half"], dtype=np.float64).reshape(3)
            extent = 2.0 * half
        else:
            extent = np.array(obj["scale_full"], dtype=np.float64).reshape(3)
    else:
        world_up = estimate_world_up(poses)
        front_hint = camera_center_world(poses[0]).reshape(3)

        # Initial cuboid fit before any trims.
        center, R, extent = compute_cuboid(points, world_up=world_up, front_hint=front_hint)

        if args.height_trim > 0.0:
            trim_axis = world_up
            trim_label = "up axis"
            if args.height_trim_mode == "object_axis":
                trim_axis = R[:, 1]
                trim_label = "object axis"
            elif args.height_trim_mode == "auto":
                # If object main axis is far from world up, the can is likely horizontal.
                align = abs(float(np.dot(R[:, 1] / (np.linalg.norm(R[:, 1]) + 1e-12), world_up / (np.linalg.norm(world_up) + 1e-12))))
                if align < 0.6:
                    trim_axis = R[:, 1]
                    trim_label = "object axis (auto-horizontal)"

            before_n = points.shape[0]
            points_h = trim_points_along_axis(points, axis=trim_axis, trim=args.height_trim, arg_name="--height_trim")
            if points_h.shape[0] != before_n:
                print(f"Trimmed points along {trim_label}: {before_n} -> {points_h.shape[0]}")
                points = points_h
                center, R, extent = compute_cuboid(points, world_up=world_up, front_hint=front_hint)

        if args.horizontal_trim > 0.0:
            before_n = points.shape[0]
            points_lat = trim_points_lateral(points, center=center, R=R, trim=args.horizontal_trim)
            if points_lat.shape[0] != before_n:
                print(f"Trimmed points laterally: {before_n} -> {points_lat.shape[0]}")
                points = points_lat
                center, R, extent = compute_cuboid(points, world_up=world_up, front_hint=front_hint)

    if args.fit_to_masks:
        if args.masks_dir is None:
            raise ValueError("--fit_to_masks requires --masks_dir")
        center, R, extent = refine_cuboid_with_masks(
            poses=poses,
            cams=cams,
            masks_dir=args.masks_dir,
            center=center,
            R=R,
            extent=extent,
            sample_stride=args.fit_sample_stride,
            max_frames=args.fit_max_frames,
        )

    world_kpts = build_world_keypoints(center, R, extent)
    if args.object_frame_out is not None:
        obj = {
            "rotation_world_from_object": R.tolist(),
            "translation_world_from_object": center.tolist(),
            "scale_half": (extent * 0.5).tolist(),
            "scale_full": extent.tolist(),
        }
        args.object_frame_out.parent.mkdir(parents=True, exist_ok=True)
        args.object_frame_out.write_text(json.dumps(obj, indent=2))

    annotations = []

    for frame_id, pose in enumerate(tqdm(poses)):
        if pose.camera_id not in cams:
            continue
        cam = cams[pose.camera_id]
        frame_center = center
        frame_R = R
        frame_world_kpts = world_kpts

        if args.per_frame_mask_refine and args.masks_dir is not None:
            mask_path = args.masks_dir / f"{Path(pose.name).stem}.png"
            mask_bbox = mask_bbox_from_file(mask_path)
            if mask_bbox is not None:
                frame_center, frame_R = refine_pose_for_frame_mask(
                    center=center,
                    R=R,
                    extent=extent,
                    cam=cam,
                    pose=pose,
                    mask_bbox=mask_bbox,
                    max_nfev=args.per_frame_refine_max_nfev,
                )
                frame_world_kpts = build_world_keypoints(frame_center, frame_R, extent)

        uv, cam_pts, depth = project(frame_world_kpts, cam, pose)

        if args.snap_bbox_to_mask and args.masks_dir is not None:
            mask_path = args.masks_dir / f"{Path(pose.name).stem}.png"
            mask_bbox = mask_bbox_from_file(mask_path)
            pred_bbox = bbox_from_uv(uv[1:])
            if mask_bbox is not None and pred_bbox is not None:
                cur_iou = bbox_iou(pred_bbox, mask_bbox)
                if cur_iou < float(args.snap_iou_threshold):
                    pb_w = max(1e-6, float(pred_bbox[2] - pred_bbox[0]))
                    pb_h = max(1e-6, float(pred_bbox[3] - pred_bbox[1]))
                    mb_w = max(1e-6, float(mask_bbox[2] - mask_bbox[0]))
                    mb_h = max(1e-6, float(mask_bbox[3] - mask_bbox[1]))
                    pcx = 0.5 * float(pred_bbox[0] + pred_bbox[2])
                    pcy = 0.5 * float(pred_bbox[1] + pred_bbox[3])
                    mcx = 0.5 * float(mask_bbox[0] + mask_bbox[2])
                    mcy = 0.5 * float(mask_bbox[1] + mask_bbox[3])
                    sx = float(np.clip(mb_w / pb_w, 0.4, 2.5))
                    sy = float(np.clip(mb_h / pb_h, 0.4, 2.5))

                    uv_adj = uv.copy()
                    uv_adj[:, 0] = (uv[:, 0] - pcx) * sx + mcx
                    uv_adj[:, 1] = (uv[:, 1] - pcy) * sy + mcy
                    uv = uv_adj

        uv_norm = np.zeros_like(uv, dtype=np.float64)
        uv_norm[:, 0] = uv[:, 0] / float(cam.width)
        uv_norm[:, 1] = uv[:, 1] / float(cam.height)

        keypoints_2d = []
        keypoints_3d = []
        for i in range(9):
            keypoints_2d.append([float(uv_norm[i, 0]), float(uv_norm[i, 1]), float(depth[i])])
            keypoints_3d.append([float(cam_pts[i, 0]), float(cam_pts[i, 1]), float(cam_pts[i, 2])])

        visibility = compute_visibility(uv, depth, cam.width, cam.height)

        Rcw = qvec2rotmat(pose.qvec)
        view = np.eye(4, dtype=np.float64)
        view[:3, :3] = Rcw
        view[:3, 3] = pose.tvec

        model = np.eye(4, dtype=np.float64)
        model[:3, :3] = frame_R
        model[:3, 3] = frame_center

        img_path = args.image_folder / pose.name
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is not None:
            if (image.shape[1], image.shape[0]) != (cam.width, cam.height):
                image = cv2.resize(image, (cam.width, cam.height), interpolation=cv2.INTER_LINEAR)
            overlay = draw_overlay(image.copy(), uv)
            cv2.imwrite(str(args.overlay_dir / pose.name), overlay)

        ann = {
            "frame_id": frame_id,
            "image": pose.name,
            "keypoints_2d": keypoints_2d,
            "keypoints_3d": keypoints_3d,
            "model_matrix": model.flatten().tolist(),
            "view_matrix": view.flatten().tolist(),
            "camera_intrinsics": {
                "fx": cam.fx,
                "fy": cam.fy,
                "cx": cam.cx,
                "cy": cam.cy,
                "image_width": cam.width,
                "image_height": cam.height,
                "model": cam.model,
                "dist": cam.dist.tolist(),
            },
            "visibility": visibility,
            "keypoints_2d_visibility": visibility,
            "pose_6dof": {
                "translation": frame_center.tolist(),
                "rotation": frame_R.flatten().tolist(),
            },
            "pose_9dof": {
                "translation": frame_center.tolist(),
                "rotation": frame_R.flatten().tolist(),
                "scale": extent.tolist(),
            },
        }
        annotations.append(ann)

    print("Writing JSON")
    with args.output_json.open("w") as f:
        json.dump(annotations, f, indent=2)

    print(f"Done. overlays={len(list(args.overlay_dir.glob('*')))} annotations={len(annotations)}")


if __name__ == "__main__":
    main()
