import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
from PIL import Image


def load_mask_for_image(masks_dir: Path, name: str):
    mask_path = masks_dir / (Path(name).stem + ".png")
    if not mask_path.exists():
        return None
    mask = Image.open(mask_path).convert("L")
    mask_np = np.array(mask) > 127
    return mask_np


def parse_points3d(points_txt: Path):
    points = {}
    with points_txt.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            pid = int(parts[0])
            xyz = np.array(list(map(float, parts[1:4])), dtype=np.float64)
            error = float(parts[7])
            track = parts[8:]
            track_len = len(track) // 2
            points[pid] = (xyz, error, track_len, parts)
    return points


def parse_images(images_txt: Path):
    image_entries = []
    with images_txt.open() as f:
        lines = [ln.rstrip("\n") for ln in f]
    idx = 0
    while idx < len(lines):
        line = lines[idx].strip()
        if not line or line.startswith("#"):
            idx += 1
            continue
        parts = line.split()
        if len(parts) < 10:
            idx += 1
            continue
        image_id = int(parts[0])
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])
        camera_id = int(parts[8])
        name = parts[9]
        idx += 1
        pts_line = lines[idx].strip() if idx < len(lines) else ""
        pts = []
        if pts_line:
            vals = pts_line.split()
            if isinstance(vals, type) or not isinstance(vals, (list, tuple)):
                vals = str(pts_line).split()
            for i in range(0, len(vals), 3):
                try:
                    x = float(vals[i])
                    y = float(vals[i + 1])
                    pid = int(vals[i + 2])
                except Exception:
                    # Skip malformed triples safely
                    continue
                pts.append((x, y, pid))

        image_entries.append(
            {
                "image_id": image_id,
                "camera_id": camera_id,
                "name": name,
                "q": (qw, qx, qy, qz),
                "t": (tx, ty, tz),
                "points2D": pts,
            }
        )
        idx += 1
    return image_entries


def compute_scale_from_points(kept_xyz: np.ndarray, real_height: float):
    if kept_xyz.size == 0:
        return 1.0, 0.0
    mean_xyz = kept_xyz.mean(axis=0)
    X = kept_xyz - mean_xyz
    cov = np.cov(X.T)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vecs = vecs[:, order]
    proj = X @ vecs
    extent = proj.max(axis=0) - proj.min(axis=0)
    recon_height = float(extent[0])
    scale = real_height / recon_height if recon_height > 0 else 1.0
    return scale, recon_height


def write_ply(points_xyz: np.ndarray, out_path: Path):
    points_xyz = np.asarray(points_xyz, dtype=np.float64)
    n = points_xyz.shape[0]
    header = [
        "ply\n",
        "format ascii 1.0\n",
        f"element vertex {n}\n",
        "property float x\n",
        "property float y\n",
        "property float z\n",
        "end_header\n",
    ]
    with out_path.open("w") as f:
        f.writelines(header)
        for p in points_xyz:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--colmap_txt", required=True, type=Path)
    ap.add_argument("--masks_dir", required=True, type=Path)
    ap.add_argument("--database_db", required=True, type=Path)
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--real_height", type=float, default=0.13)
    ap.add_argument("--min_total_obs", type=int, default=3)
    ap.add_argument("--min_inmask_ratio", type=float, default=0.6)
    ap.add_argument("--skip_desc_db", action="store_true", help="Skip descriptor aggregation from the COLMAP database")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    points_txt = args.colmap_txt / "points3D.txt"
    images_txt = args.colmap_txt / "images.txt"
    cameras_txt = args.colmap_txt / "cameras.txt"

    points = parse_points3d(points_txt)
    image_entries = parse_images(images_txt)

    def ensure_counts(d, key):
        v = d.get(key)
        if not isinstance(v, list) or len(v) != 2:
            d[key] = [0, 0]
        return d[key]

    # Filter 2D points by mask.
    valid_obs = {}
    for e in image_entries:
        mask = load_mask_for_image(args.masks_dir, e["name"])
        filtered_pts = []
        for x, y, pid in e["points2D"]:
            pid = int(pid)
            if pid < 0:
                continue
            if mask is None:
                continue
            xi, yi = int(round(x)), int(round(y))
            if 0 <= yi < mask.shape[0] and 0 <= xi < mask.shape[1] and mask[yi, xi]:
                filtered_pts.append((x, y, pid))
                counts = ensure_counts(valid_obs, pid)
                counts[0] += 1
                counts[1] += 1
            else:
                counts = ensure_counts(valid_obs, pid)
                counts[1] += 1
        e["points2D"] = filtered_pts

    kept_pids = set()
    for pid, (inm, total) in valid_obs.items():
        if total >= args.min_total_obs and inm / max(total, 1) >= args.min_inmask_ratio:
            kept_pids.add(pid)

    kept_xyz = np.array([points[pid][0] for pid in kept_pids if pid in points], dtype=np.float64)
    if kept_xyz.size == 0:
        kept_xyz = np.array([xyz for (xyz, _, _, _) in points.values()], dtype=np.float64)

    scale, recon_height = compute_scale_from_points(kept_xyz, args.real_height)

    # Write scaled points3D.txt
    with (args.out_dir / "points3D.txt").open("w") as f:
        f.write(f"# Masked+Scaled points3D.txt (scale={scale:.6f})\n")
        f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
        with points_txt.open() as fin:
            for line in fin:
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.split()
                pid = int(parts[0])
                if pid not in kept_pids:
                    continue
                x, y, z = map(float, parts[1:4])
                x, y, z = x * scale, y * scale, z * scale
                parts[1] = f"{x:.6f}"
                parts[2] = f"{y:.6f}"
                parts[3] = f"{z:.6f}"
                f.write(" ".join(parts) + "\n")

    # Write scaled images.txt
    with (args.out_dir / "images.txt").open("w") as f:
        f.write(f"# Masked+Scaled images.txt (scale={scale:.6f})\n")
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for e in image_entries:
            qw, qx, qy, qz = e["q"]
            tx, ty, tz = e["t"]
            tx, ty, tz = tx * scale, ty * scale, tz * scale
            f.write(
                f"{e['image_id']} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {e['camera_id']} {e['name']}\n"
            )
            pts = [(x, y, pid) for x, y, pid in e["points2D"] if pid in kept_pids]
            if pts:
                flat = " ".join([f"{x} {y} {pid}" for x, y, pid in pts])
                f.write(flat + "\n")
            else:
                f.write("\n")

    # Copy cameras.txt unchanged
    (args.out_dir / "cameras.txt").write_text(cameras_txt.read_text())

    # Write PLY for visualization
    scaled_xyz = np.array([points[pid][0] * scale for pid in kept_pids if pid in points], dtype=np.float64)
    if scaled_xyz.size == 0:
        scaled_xyz = np.array([xyz * scale for (xyz, _, _, _) in points.values()], dtype=np.float64)
    write_ply(scaled_xyz, args.out_dir / "points3D.ply")

    # Optionally compute a simple descriptor DB for PnP (useful later).
    if args.skip_desc_db:
        print("Skipping descriptor DB aggregation (--skip_desc_db)")
    else:
        conn = sqlite3.connect(str(args.database_db))
        cur = conn.cursor()
        cur.execute("SELECT image_id, name FROM images")
        name_to_id = {name: image_id for image_id, name in cur.fetchall()}

        def fetch_keypoints_and_desc(img_id):
            cur.execute("SELECT data FROM keypoints WHERE image_id=?", (img_id,))
            row = cur.fetchone()
            if row is None:
                return None, None
            kp_data = np.frombuffer(row[0], dtype=np.float32)
            kp = kp_data.reshape(-1, 6)
            cur.execute("SELECT data FROM descriptors WHERE image_id=?", (img_id,))
            row = cur.fetchone()
            if row is None:
                return kp, None
            desc = np.frombuffer(row[0], dtype=np.uint8)
            desc = desc.reshape(-1, 128)
            return kp, desc

        acc = {}
        count = {}
        for e in image_entries:
            img_id_db = name_to_id.get(e["name"])
            if img_id_db is None:
                continue
            kp, desc = fetch_keypoints_and_desc(img_id_db)
            if kp is None or desc is None:
                continue
            for idx, (_, _, pid) in enumerate(e["points2D"]):
                if pid not in kept_pids:
                    continue
                if idx >= desc.shape[0]:
                    continue
                d = desc[idx].astype(np.float32)
                acc[pid] = acc.get(pid, 0) + d
                count[pid] = count.get(pid, 0) + 1

        pids = sorted(acc.keys())
        xyz = []
        desc = []
        track = []
        for pid in pids:
            xyz.append(points[pid][0] * scale)
            d = acc[pid] / count[pid]
            norm = np.linalg.norm(d)
            if norm > 0:
                d = d / norm
            desc.append(d)
            track.append(points[pid][2])

        if xyz:
            xyz = np.array(xyz, dtype=np.float32)
            desc = np.array(desc, dtype=np.float32)
            track = np.array(track, dtype=np.int32)
            np.savez_compressed(
                args.out_dir / "ref_sift_db_masked.npz",
                point3d_id=np.array(pids, dtype=np.int64),
                xyz=xyz,
                desc=desc,
                track_len=track,
                scale=scale,
                recon_height=recon_height,
                real_height=args.real_height,
            )

    stats = {
        "num_images": len(image_entries),
        "num_points3d_total": len(points),
        "num_points3d_kept": len(kept_pids),
        "scale": scale,
        "recon_height": recon_height,
        "real_height": args.real_height,
    }
    with (args.out_dir / "stats_masked.json").open("w") as f:
        json.dump(stats, f, indent=2)

    print("KEPT 3D points:", len(kept_pids))
    print("SCALE", scale, "recon_height", recon_height)
    print("WROTE", args.out_dir)


if __name__ == "__main__":
    main()
