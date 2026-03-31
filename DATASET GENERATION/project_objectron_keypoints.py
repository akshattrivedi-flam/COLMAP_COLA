import argparse
import json
import cv2
from pathlib import Path
import numpy as np


def qvec_to_rotmat(qvec):
    # COLMAP qvec: [qw, qx, qy, qz]
    qw, qx, qy, qz = qvec
    return np.array([
        [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw,     1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw,     2*qy*qz + 2*qx*qw,     1 - 2*qx*qx - 2*qy*qy],
    ], dtype=np.float64)


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
        # Use first 4 for intrinsics; ignore distortion in projection
        fx, fy, cx, cy = p[:4]
        if model == "SIMPLE_RADIAL":
            k1 = p[4] if len(p) > 4 else 0.0
            dist = np.array([k1, 0.0, 0.0, 0.0], dtype=np.float64)
        elif model == "RADIAL":
            k1 = p[4] if len(p) > 4 else 0.0
            k2 = p[5] if len(p) > 5 else 0.0
            dist = np.array([k1, k2, 0.0, 0.0], dtype=np.float64)
        elif model == "OPENCV":
            k1 = p[4] if len(p) > 4 else 0.0
            k2 = p[5] if len(p) > 5 else 0.0
            p1 = p[6] if len(p) > 6 else 0.0
            p2 = p[7] if len(p) > 7 else 0.0
            dist = np.array([k1, k2, p1, p2], dtype=np.float64)
        else:
            dist = np.zeros(4, dtype=np.float64)
    else:
        raise ValueError(f"Unsupported camera model: {model}")
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    return K, dist, (fx, fy, cx, cy)


def load_images_txt(path: Path):
    images = []
    with path.open() as f:
        lines = [ln.strip() for ln in f]
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line or line.startswith('#'):
            i += 1
            continue
        parts = line.split()
        if len(parts) < 9:
            i += 1
            continue
        image_id = int(parts[0])
        qvec = np.array(list(map(float, parts[1:5])), dtype=np.float64)
        tvec = np.array(list(map(float, parts[5:8])), dtype=np.float64)
        cam_id = int(parts[8])
        name = parts[9]
        images.append({
            "image_id": image_id,
            "qvec": qvec,
            "tvec": tvec,
            "camera_id": cam_id,
            "name": name,
        })
        i += 2  # skip the next line (2D points)
    return images


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cameras", required=True, type=Path)
    ap.add_argument("--images", required=True, type=Path)
    ap.add_argument("--object_frame", required=True, type=Path)
    ap.add_argument("--keypoints", required=True, type=Path)
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--visibility_min", type=float, default=0.1)
    args = ap.parse_args()

    cameras = load_cameras_txt(args.cameras)
    images = load_images_txt(args.images)
    keypoints_obj = np.load(args.keypoints)

    with args.object_frame.open() as f:
        obj = json.load(f)
    R_wo = np.array(obj["rotation_world_from_object"], dtype=np.float64)
    t_wo = np.array(obj["translation_world_from_object"], dtype=np.float64)

    out_ann = args.out_dir / "annotations"
    out_ann.mkdir(parents=True, exist_ok=True)

    written = 0
    for im in images:
        cam = cameras[im["camera_id"]]
        K, dist, (fx, fy, cx, cy) = camera_intrinsics(cam)
        W, H = cam["width"], cam["height"]

        R_cw = qvec_to_rotmat(im["qvec"])  # world -> camera
        t_cw = im["tvec"].reshape(3, 1)

        # object -> world -> camera
        X_world = (R_wo @ keypoints_obj.T) + t_wo.reshape(3, 1)
        X_cam = (R_cw @ X_world) + t_cw
        X_cam = X_cam.T  # (9,3)

        # project (with distortion if available)
        rvec, _ = cv2.Rodrigues(R_cw)
        tvec = t_cw.reshape(3, 1)
        img_pts, _ = cv2.projectPoints(
            X_world.T.reshape(-1, 1, 3), rvec, tvec, K, dist
        )
        img_pts = img_pts.reshape(-1, 2)
        us = img_pts[:, 0]
        vs = img_pts[:, 1]

        zs = X_cam[:, 2]

        u_norm = us / W
        v_norm = vs / H

        visible = (zs > 0) & (us >= 0) & (us < W) & (vs >= 0) & (vs < H)
        visibility = float(visible.sum()) / float(len(visible))

        if visibility < args.visibility_min:
            continue

        ann = {
            "image": im["name"],
            "image_size": [W, H],
            "intrinsics": K.tolist(),
            "extrinsics": {
                "R_cw": R_cw.tolist(),
                "t_cw": t_cw.flatten().tolist(),
            },
            "keypoints": {
                "points_3d_object": keypoints_obj.tolist(),
                "points_3d_camera": X_cam.tolist(),
                "points_2d_norm_depth": np.stack([u_norm, v_norm, zs], axis=1).tolist(),
            },
            "visibility": visibility,
        }

        out_path = out_ann / f"{im['image_id']:06d}.json"
        with out_path.open("w") as f:
            json.dump(ann, f, indent=2)
        written += 1

    print(f"Wrote {written} annotations to {out_ann}")


if __name__ == "__main__":
    main()
