import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d
import cv2


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


def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = qvec
    return np.array(
        [
            [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
            [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qx * qw],
            [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx * qx - 2 * qy * qy],
        ],
        dtype=np.float64,
    )


def parse_camera_line(parts):
    camera_id = int(parts[0])
    model = parts[1]
    width = int(parts[2])
    height = int(parts[3])
    p = list(map(float, parts[4:]))
    dist = np.zeros(4, dtype=np.float64)

    if model == "SIMPLE_PINHOLE":
        f, cx, cy = p[:3]
        fx = fy = f
    elif model == "PINHOLE":
        fx, fy, cx, cy = p[:4]
    elif model in ("SIMPLE_RADIAL", "RADIAL"):
        f, cx, cy = p[:3]
        fx = fy = f
        if model == "SIMPLE_RADIAL" and len(p) >= 4:
            dist = np.array([p[3], 0.0, 0.0, 0.0], dtype=np.float64)
        elif model == "RADIAL" and len(p) >= 5:
            dist = np.array([p[3], p[4], 0.0, 0.0], dtype=np.float64)
    elif model in ("OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV"):
        fx, fy, cx, cy = p[:4]
        if len(p) >= 8:
            dist = np.array([p[4], p[5], p[6], p[7]], dtype=np.float64)
    else:
        raise ValueError(f"Unsupported COLMAP camera model: {model}")

    return Camera(camera_id, model, width, height, fx, fy, cx, cy, dist)


def load_cameras_txt(path: Path):
    cams = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        cam = parse_camera_line(line.split())
        cams[cam.camera_id] = cam
    if not cams:
        raise ValueError(f"No cameras found in {path}")
    return cams


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
        rows.append(
            ImagePose(
                image_id=int(p[0]),
                qvec=np.array(list(map(float, p[1:5])), dtype=np.float64),
                tvec=np.array(list(map(float, p[5:8])), dtype=np.float64),
                camera_id=int(p[8]),
                name=p[9],
            )
        )
        i += 2  # Skip POINTS2D line.
    if not rows:
        raise ValueError(f"No image poses found in {path}")
    rows.sort(key=lambda r: r.image_id)
    return rows


def load_points3d_txt(path: Path):
    xyz = []
    rgb = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        p = line.split()
        if len(p) < 7:
            continue
        xyz.append([float(p[1]), float(p[2]), float(p[3])])
        rgb.append([float(p[4]), float(p[5]), float(p[6])])
    if not xyz:
        raise ValueError(f"No 3D points found in {path}")
    xyz = np.asarray(xyz, dtype=np.float64)
    rgb = np.asarray(rgb, dtype=np.float64) / 255.0
    return xyz, rgb


def load_pointcloud(path: Path):
    ext = path.suffix.lower()
    if ext == ".txt":
        return load_points3d_txt(path)

    if ext in (".ply", ".pcd", ".xyz", ".xyzn", ".xyzrgb"):
        pcd = o3d.io.read_point_cloud(str(path))
        if pcd.is_empty():
            raise ValueError(f"Point cloud is empty: {path}")
        xyz = np.asarray(pcd.points, dtype=np.float64)
        if pcd.has_colors():
            rgb = np.asarray(pcd.colors, dtype=np.float64)
        else:
            rgb = np.full((xyz.shape[0], 3), 0.9, dtype=np.float64)
        return xyz, rgb

    raise ValueError(f"Unsupported point cloud format: {path}")


def build_intrinsic_for_window(cam: Camera, out_w: int, out_h: int):
    sx = out_w / float(cam.width)
    sy = out_h / float(cam.height)
    fx = cam.fx * sx
    fy = cam.fy * sy
    cx = cam.cx * sx
    cy = cam.cy * sy
    return o3d.camera.PinholeCameraIntrinsic(out_w, out_h, fx, fy, cx, cy)


def build_K_for_image(cam: Camera):
    return np.array(
        [[cam.fx, 0.0, cam.cx], [0.0, cam.fy, cam.cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def load_original_image(image_dir: Path, row: ImagePose):
    p = image_dir / row.name
    if not p.exists():
        return None
    return cv2.imread(str(p), cv2.IMREAD_COLOR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--points3d",
        required=True,
        type=Path,
        help="Point cloud file (.txt COLMAP points3D, .ply, .pcd, .xyz)",
    )
    ap.add_argument("--images", required=True, type=Path, help="COLMAP images.txt")
    ap.add_argument("--cameras", required=True, type=Path, help="COLMAP cameras.txt")
    ap.add_argument("--out_dir", required=True, type=Path, help="Output folder for rendered views")
    ap.add_argument("--stride", type=int, default=50, help="Take every Nth registered image pose")
    ap.add_argument("--start_index", type=int, default=0, help="Start index in sorted registered images")
    ap.add_argument("--max_points", type=int, default=200000, help="Randomly sample point cloud if larger than this")
    ap.add_argument("--point_size", type=float, default=2.0, help="Open3D render point size")
    ap.add_argument("--image_dir", type=Path, default=None, help="Optional original image folder for overlay output")
    ap.add_argument("--overlay_alpha", type=float, default=0.5, help="Overlay alpha for original image")
    ap.add_argument("--undistort_images", action="store_true", help="Undistort original images before overlay")
    ap.add_argument("--visible_window", action="store_true", help="Show viewer window while rendering")
    ap.add_argument("--dry_run", action="store_true", help="Parse inputs and print selected frames without rendering")
    args = ap.parse_args()

    if args.stride <= 0:
        raise ValueError("--stride must be > 0")
    if args.start_index < 0:
        raise ValueError("--start_index must be >= 0")
    if not (0.0 <= args.overlay_alpha <= 1.0):
        raise ValueError("--overlay_alpha must be in [0,1]")

    xyz, rgb = load_pointcloud(args.points3d)
    if xyz.shape[0] > args.max_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(xyz.shape[0], size=args.max_points, replace=False)
        xyz = xyz[idx]
        rgb = rgb[idx]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)

    cameras = load_cameras_txt(args.cameras)
    poses = load_images_txt(args.images)
    selected = poses[args.start_index :: args.stride]
    if not selected:
        raise ValueError("No poses selected. Check --start_index / --stride.")

    first_cam = cameras[selected[0].camera_id]
    out_w, out_h = first_cam.width, first_cam.height

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Total registered images: {len(poses)}")
    print(f"Selected (stride={args.stride}, start={args.start_index}): {len(selected)}")
    print(f"Rendering size: {out_w}x{out_h}")
    print(f"Output: {args.out_dir}")
    if args.image_dir is not None:
        print(f"Overlay images from: {args.image_dir}")

    if args.dry_run:
        for k, row in enumerate(selected[:10]):
            print(f"[dry-run {k + 1}] image_id={row.image_id} name={row.name} camera_id={row.camera_id}")
        print("Dry run complete.")
        return

    vis = o3d.visualization.Visualizer()
    ok = vis.create_window(
        window_name="COLMAP stride views",
        width=out_w,
        height=out_h,
        visible=args.visible_window,
    )
    if not ok:
        raise RuntimeError(
            "Failed to create Open3D window. If headless, run with xvfb-run or on a desktop session."
        )

    vis.add_geometry(pcd)
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.0, 0.0, 0.0], dtype=np.float64)
    opt.point_size = float(args.point_size)

    ctr = vis.get_view_control()

    for k, row in enumerate(selected):
        cam = cameras[row.camera_id]
        intrinsic = build_intrinsic_for_window(cam, out_w, out_h)

        extrinsic = np.eye(4, dtype=np.float64)
        extrinsic[:3, :3] = qvec_to_rotmat(row.qvec)  # world -> camera
        extrinsic[:3, 3] = row.tvec

        params = o3d.camera.PinholeCameraParameters()
        params.intrinsic = intrinsic
        params.extrinsic = extrinsic

        ctr.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)
        vis.poll_events()
        vis.update_renderer()

        stem = Path(row.name).stem
        out_img = args.out_dir / f"{k:04d}_img{row.image_id:06d}_{stem}.png"
        vis.capture_screen_image(str(out_img), do_render=True)

        if args.image_dir is not None:
            raw = load_original_image(args.image_dir, row)
            if raw is not None:
                if args.undistort_images and np.linalg.norm(cam.dist) > 0:
                    K = build_K_for_image(cam)
                    raw = cv2.undistort(raw, K, cam.dist)
                if (raw.shape[1], raw.shape[0]) != (out_w, out_h):
                    raw = cv2.resize(raw, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
                cloud = cv2.imread(str(out_img), cv2.IMREAD_COLOR)
                if cloud is not None and cloud.shape == raw.shape:
                    a = float(args.overlay_alpha)
                    overlay = cv2.addWeighted(raw, a, cloud, 1.0 - a, 0.0)
                    out_overlay = args.out_dir / f"{k:04d}_img{row.image_id:06d}_{stem}_overlay.png"
                    cv2.imwrite(str(out_overlay), overlay)

        print(f"[{k + 1}/{len(selected)}] {row.name} -> {out_img.name}")

    vis.destroy_window()
    print("Done.")


if __name__ == "__main__":
    main()
