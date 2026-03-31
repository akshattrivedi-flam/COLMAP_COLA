import argparse
import io
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


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
            # Keep first 4 fish-eye coeffs separately in dist.
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

        image_id = int(p[0])
        qvec = np.array(list(map(float, p[1:5])), dtype=np.float64)
        tvec = np.array(list(map(float, p[5:8])), dtype=np.float64)
        camera_id = int(p[8])
        name = p[9]
        poses.append(
            ImagePose(
                image_id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera_id,
                name=name,
                line_index=len(poses),
            )
        )

        # IMPORTANT: images.txt has 2 lines per image; skip POINTS2D line.
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


def build_K(cam: Camera):
    return np.array(
        [[cam.fx, 0.0, cam.cx], [0.0, cam.fy, cam.cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def read_ply_ascii(path: Path):
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

    def find_idx(names):
        for n in names:
            if n in props:
                return props.index(n)
        return None

    xi = find_idx(["x"])
    yi = find_idx(["y"])
    zi = find_idx(["z"])
    if xi is None or yi is None or zi is None:
        raise ValueError(f"PLY missing x/y/z properties: {path}")

    points = data[:, [xi, yi, zi]].astype(np.float64)

    ri = find_idx(["red", "r"])
    gi = find_idx(["green", "g"])
    bi = find_idx(["blue", "b"])
    colors_bgr = None
    if ri is not None and gi is not None and bi is not None:
        rgb = data[:, [ri, gi, bi]]
        if rgb.max() <= 1.0:
            rgb = rgb * 255.0
        colors_bgr = np.clip(rgb[:, ::-1], 0, 255).astype(np.uint8)

    return points, colors_bgr


def project_points_colmap(points_world: np.ndarray, cam: Camera, row: ImagePose):
    Rcw = qvec2rotmat(row.qvec)
    tcw = row.tvec.reshape(3, 1)
    K = build_K(cam)

    # Depth for visibility/z-buffer.
    Xc = (Rcw @ points_world.T) + tcw
    z = Xc[2, :]

    rvec, _ = cv2.Rodrigues(Rcw)
    tvec = row.tvec.reshape(3, 1)

    if cam.model == "OPENCV_FISHEYE":
        pts = points_world.reshape(-1, 1, 3).astype(np.float64)
        # cv2.fisheye expects shape (N,1,3)
        uv, _ = cv2.fisheye.projectPoints(pts, rvec, tvec, K, cam.dist.reshape(4, 1))
    else:
        uv, _ = cv2.projectPoints(
            points_world.reshape(-1, 1, 3).astype(np.float64),
            rvec,
            tvec,
            K,
            cam.dist,
        )
    uv = uv.reshape(-1, 2)
    return uv, z


def render_point_cloud_image(uv: np.ndarray, z: np.ndarray, colors_bgr: np.ndarray, width: int, height: int, point_size: int):
    img = np.zeros((height, width, 3), dtype=np.uint8)

    valid = (
        (z > 1e-8)
        & (uv[:, 0] >= 0)
        & (uv[:, 0] < width)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < height)
    )
    if not np.any(valid):
        return img

    uvi = uv[valid]
    zvi = z[valid]
    cvi = colors_bgr[valid]

    # Draw far to near so close points stay visible.
    order = np.argsort(zvi)[::-1]
    uvi = uvi[order]
    cvi = cvi[order]

    r = max(1, int(point_size))
    for p, c in zip(uvi, cvi):
        u = int(round(float(p[0])))
        v = int(round(float(p[1])))
        cv2.circle(img, (u, v), r, (int(c[0]), int(c[1]), int(c[2])), -1, lineType=cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--point_cloud", type=Path, required=True)
    ap.add_argument("--cameras_txt", type=Path, required=True)
    ap.add_argument("--images_txt", type=Path, required=True)
    ap.add_argument("--image_folder", type=Path, required=True)
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--stride", type=int, default=50)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--sort_by", choices=["name", "image_id", "line"], default="name")
    ap.add_argument("--point_size", type=int, default=1)
    ap.add_argument("--overlay_alpha", type=float, default=0.5)
    ap.add_argument("--max_points", type=int, default=200000)
    args = ap.parse_args()

    if args.stride <= 0:
        raise ValueError("--stride must be > 0")
    if args.start < 0:
        raise ValueError("--start must be >= 0")
    if not (0.0 <= args.overlay_alpha <= 1.0):
        raise ValueError("--overlay_alpha must be in [0,1]")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    cams = read_cameras(args.cameras_txt)
    poses = sort_poses(read_images(args.images_txt), args.sort_by)
    selected = poses[args.start :: args.stride]
    if not selected:
        raise ValueError("No poses selected with current --start / --stride")

    points_world, colors_bgr = read_ply_ascii(args.point_cloud)
    if points_world.size == 0:
        raise ValueError(f"Point cloud is empty: {args.point_cloud}")
    if colors_bgr is None:
        colors_bgr = np.full((points_world.shape[0], 3), 220, dtype=np.uint8)

    if points_world.shape[0] > args.max_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(points_world.shape[0], size=args.max_points, replace=False)
        points_world = points_world[idx]
        colors_bgr = colors_bgr[idx]

    print(f"Loaded points: {points_world.shape[0]}")
    print(f"Total poses: {len(poses)}")
    print(f"Selected poses: {len(selected)} (stride={args.stride}, start={args.start}, sort={args.sort_by})")

    for k, row in enumerate(selected):
        if row.camera_id not in cams:
            print(f"[skip] {row.name}: camera_id={row.camera_id} missing in cameras.txt")
            continue
        cam = cams[row.camera_id]
        img_path = args.image_folder / row.name
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[skip] {row.name}: image not found at {img_path}")
            continue

        # Ensure target dimensions match camera intrinsics dimensions.
        if (img.shape[1], img.shape[0]) != (cam.width, cam.height):
            img = cv2.resize(img, (cam.width, cam.height), interpolation=cv2.INTER_LINEAR)

        uv, z = project_points_colmap(points_world, cam, row)
        cloud = render_point_cloud_image(
            uv=uv,
            z=z,
            colors_bgr=colors_bgr,
            width=cam.width,
            height=cam.height,
            point_size=args.point_size,
        )

        a = float(args.overlay_alpha)
        overlay = cv2.addWeighted(img, a, cloud, 1.0 - a, 0.0)

        stem = Path(row.name).stem
        out_cloud = args.output_dir / f"{k:04d}_img{row.image_id:06d}_{stem}_cloud.png"
        out_overlay = args.output_dir / f"{k:04d}_img{row.image_id:06d}_{stem}_overlay.png"
        cv2.imwrite(str(out_cloud), cloud)
        cv2.imwrite(str(out_overlay), overlay)
        print(f"[{k + 1}/{len(selected)}] {row.name} -> {out_overlay.name}")

    print("Done.")


if __name__ == "__main__":
    main()
