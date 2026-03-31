import argparse
import json
from pathlib import Path

import numpy as np


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

        data = np.loadtxt([f.readline() for _ in range(num_verts)], dtype=np.float64)
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


def write_ply_ascii(points_xyz: np.ndarray, out_path: Path):
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


def compute_height(points: np.ndarray):
    if points.size == 0:
        return 0.0
    mean_xyz = points.mean(axis=0)
    X = points - mean_xyz
    cov = np.cov(X.T)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vecs = vecs[:, order]
    proj = X @ vecs
    extent = proj.max(axis=0) - proj.min(axis=0)
    return float(extent[0])


def rescale_points3d_txt(path: Path, scale: float):
    lines = []
    with path.open() as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                lines.append(line)
                continue
            parts = line.split()
            x, y, z = map(float, parts[1:4])
            x, y, z = x * scale, y * scale, z * scale
            parts[1] = f"{x:.6f}"
            parts[2] = f"{y:.6f}"
            parts[3] = f"{z:.6f}"
            lines.append(" ".join(parts) + "\n")
    path.write_text("".join(lines))


def rescale_images_txt(path: Path, scale: float):
    lines = [ln.rstrip("\n") for ln in path.read_text().splitlines()]
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line or line.startswith("#"):
            out.append(line)
            i += 1
            continue
        parts = line.split()
        if len(parts) >= 10:
            tx, ty, tz = map(float, parts[5:8])
            tx, ty, tz = tx * scale, ty * scale, tz * scale
            parts[5] = f"{tx:.6f}"
            parts[6] = f"{ty:.6f}"
            parts[7] = f"{tz:.6f}"
            out.append(" ".join(parts))
            # next line is points2D
            if i + 1 < len(lines):
                out.append(lines[i + 1])
            i += 2
        else:
            out.append(line)
            i += 1
    path.write_text("\n".join(out) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scaled_dir", type=Path, required=True)
    ap.add_argument("--clean_ply", type=Path, required=True)
    ap.add_argument("--real_height", type=float, default=0.13)
    ap.add_argument("--out_clean_ply", type=Path, required=True)
    args = ap.parse_args()

    points_clean = read_ply_ascii_points(args.clean_ply)
    recon_height = compute_height(points_clean)
    if recon_height <= 0:
        raise RuntimeError("Clean point cloud has zero height.")

    factor = args.real_height / recon_height
    print(f"Rescale factor: {factor:.6f} (recon_height={recon_height:.6f}, real_height={args.real_height})")

    # Rescale text model
    rescale_points3d_txt(args.scaled_dir / "points3D.txt", factor)
    rescale_images_txt(args.scaled_dir / "images.txt", factor)

    # Rescale PLYs
    points_all = read_ply_ascii_points(args.scaled_dir / "points3D.ply")
    write_ply_ascii(points_all * factor, args.scaled_dir / "points3D.ply")
    write_ply_ascii(points_clean * factor, args.out_clean_ply)

    # Update stats if present
    stats_path = args.scaled_dir / "stats_masked.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text())
        stats["rescale_factor"] = factor
        stats["recon_height_after_clean"] = recon_height
        stats["real_height"] = args.real_height
        stats_path.write_text(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
