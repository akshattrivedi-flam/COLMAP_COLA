import argparse
from collections import deque, defaultdict
from pathlib import Path

import numpy as np


def load_points_from_npz(npz_path: Path):
    data = np.load(npz_path, allow_pickle=True)
    if "xyz" in data.files:
        points = data["xyz"]
    else:
        points = None
        for k in data.files:
            arr = np.asarray(data[k])
            if arr.ndim == 2 and arr.shape[1] == 3:
                points = arr
                break
        if points is None:
            raise ValueError(f"No Nx3 point array found in {npz_path}. Keys: {data.files}")
    return np.asarray(points, dtype=np.float64)


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


def dbscan_grid(points: np.ndarray, eps: float, min_points: int):
    points = np.asarray(points, dtype=np.float64)
    n = points.shape[0]
    if n == 0:
        return np.array([], dtype=np.int32)

    eps = float(eps)
    min_points = int(min_points)
    if eps <= 0:
        raise ValueError("--eps must be > 0")

    inv = 1.0 / eps
    keys = np.floor(points * inv).astype(np.int64)
    grid = defaultdict(list)
    for i, k in enumerate(keys):
        grid[(k[0], k[1], k[2])].append(i)

    eps2 = eps * eps

    def region_query(i):
        k = keys[i]
        neigh = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    cell = (k[0] + dx, k[1] + dy, k[2] + dz)
                    for j in grid.get(cell, []):
                        if j == i:
                            neigh.append(j)
                            continue
                        d = points[j] - points[i]
                        if d.dot(d) <= eps2:
                            neigh.append(j)
        return neigh

    labels = np.full(n, -1, dtype=np.int32)
    visited = np.zeros(n, dtype=bool)
    cluster_id = 0

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        neighbors = region_query(i)
        if len(neighbors) < min_points:
            labels[i] = -1
            continue
        labels[i] = cluster_id
        queue = deque(neighbors)
        in_queue = np.zeros(n, dtype=bool)
        for j in neighbors:
            in_queue[j] = True
        while queue:
            j = queue.popleft()
            if not visited[j]:
                visited[j] = True
                neighbors2 = region_query(j)
                if len(neighbors2) >= min_points:
                    for k in neighbors2:
                        if not in_queue[k]:
                            queue.append(k)
                            in_queue[k] = True
            if labels[j] == -1:
                labels[j] = cluster_id
        cluster_id += 1

    return labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_npz", type=Path, default=None)
    ap.add_argument("--input_ply", type=Path, default=None)
    ap.add_argument("--output_ply", type=Path, required=True)
    ap.add_argument("--eps", type=float, default=0.002)
    ap.add_argument("--min_points", type=int, default=10)
    ap.add_argument("--keep_top_k", type=int, default=0, help="Keep top-k largest clusters (0 = keep all non-noise)")
    ap.add_argument("--min_cluster_size", type=int, default=0, help="Drop clusters smaller than this size")
    ap.add_argument("--no_view", action="store_true")
    args = ap.parse_args()

    if args.input_npz is None and args.input_ply is None:
        raise ValueError("Provide --input_npz or --input_ply")

    if args.input_ply is not None:
        points = read_ply_ascii_points(args.input_ply)
    else:
        points = load_points_from_npz(args.input_npz)

    print("Total points before filtering:", points.shape[0])

    labels = dbscan_grid(points, eps=args.eps, min_points=args.min_points)

    max_label = labels.max() if labels.size else -1
    print("Clusters found:", max_label + 1)

    clean_indices = np.where(labels != -1)[0]
    if max_label >= 0:
        # Cluster size stats (excluding noise)
        counts = np.bincount(labels[labels >= 0])
        order = np.argsort(counts)[::-1]
        top_counts = counts[order][:10]
        print("Top cluster sizes:", top_counts.tolist())

        keep_labels = None
        if args.keep_top_k and args.keep_top_k > 0:
            keep_labels = set(order[: args.keep_top_k].tolist())
        if args.min_cluster_size and args.min_cluster_size > 0:
            keep_by_size = set(np.where(counts >= args.min_cluster_size)[0].tolist())
            keep_labels = keep_by_size if keep_labels is None else (keep_labels & keep_by_size)

        if keep_labels is not None:
            mask = np.array([lab in keep_labels for lab in labels], dtype=bool)
            clean_indices = np.where(mask)[0]
    pcd_clean = points[clean_indices]

    print("Points after removing outliers:", pcd_clean.shape[0])
    write_ply_ascii(pcd_clean, args.output_ply)

    if not args.no_view:
        try:
            import open3d as o3d
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pcd_clean)
            o3d.visualization.draw_geometries([pcd])
        except Exception as e:
            print("Open3D visualization skipped:", e)


if __name__ == "__main__":
    main()
