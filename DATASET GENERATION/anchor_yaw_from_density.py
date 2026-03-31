import argparse
import json
from pathlib import Path

import numpy as np


def normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-12:
        return None
    return v / n


def rotate_about_axis(v, axis, angle_deg):
    th = np.deg2rad(float(angle_deg))
    k = normalize(axis)
    if k is None:
        return v
    return v * np.cos(th) + np.cross(k, v) * np.sin(th) + k * (k @ v) * (1.0 - np.cos(th))


def load_points_with_weights(path: Path, weight_mode: str = "track"):
    ext = path.suffix.lower()
    if ext == ".txt":
        pts = []
        tr_ws = []
        rgbs = []
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = line.split()
            if len(p) < 8:
                continue
            pts.append([float(p[1]), float(p[2]), float(p[3])])
            # RGB present in COLMAP points3D.txt at tokens [4:7].
            rgbs.append([float(p[4]), float(p[5]), float(p[6])])
            # COLMAP points3D track entries start at token index 8: (IMAGE_ID, POINT2D_IDX)...
            if len(p) > 8:
                tr = max(1, (len(p) - 8) // 2)
            else:
                tr = 1
            tr_ws.append(float(tr))
        if not pts:
            raise ValueError(f"No points found in {path}")
        pts = np.asarray(pts, dtype=np.float64)
        tr_ws = np.asarray(tr_ws, dtype=np.float64)
        rgbs = np.asarray(rgbs, dtype=np.float64)
        if weight_mode == "track":
            ws = tr_ws
        else:
            # Dark-text weighting for logo-facing anchor.
            lum = 0.2126 * rgbs[:, 0] + 0.7152 * rgbs[:, 1] + 0.0722 * rgbs[:, 2]
            dark = np.clip((255.0 - lum) / 255.0, 0.0, 1.0)
            if weight_mode == "dark":
                ws = 1.0 + 3.0 * dark
            elif weight_mode == "track_dark":
                ws = tr_ws * (1.0 + 2.5 * dark)
            else:
                raise ValueError(f"Unsupported weight_mode: {weight_mode}")
        return pts, ws

    if ext == ".npz":
        d = np.load(path, allow_pickle=True)
        if "xyz" in d.files:
            pts = np.asarray(d["xyz"], dtype=np.float64)
        else:
            pts = None
            for k in d.files:
                a = np.asarray(d[k])
                if a.ndim == 2 and a.shape[1] == 3:
                    pts = np.asarray(a, dtype=np.float64)
                    break
            if pts is None:
                raise ValueError(f"No Nx3 points in {path}; keys={d.files}")
        if "track_len" in d.files:
            ws = np.asarray(d["track_len"], dtype=np.float64)
            if ws.shape[0] != pts.shape[0]:
                ws = np.ones((pts.shape[0],), dtype=np.float64)
        else:
            ws = np.ones((pts.shape[0],), dtype=np.float64)
        return pts, ws

    if ext == ".npy":
        pts = np.asarray(np.load(path), dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(f"Expected Nx3 points in {path}; got {pts.shape}")
        ws = np.ones((pts.shape[0],), dtype=np.float64)
        return pts, ws

    raise ValueError(f"Unsupported points format: {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--object_frame", required=True, type=Path)
    ap.add_argument("--points", required=True, type=Path)
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--max_yaw_change_deg", type=float, default=12.0)
    ap.add_argument("--min_confidence", type=float, default=0.015)
    ap.add_argument("--radial_min_quantile", type=float, default=20.0)
    ap.add_argument("--weight_mode", choices=["track", "dark", "track_dark"], default="track")
    args = ap.parse_args()

    obj = json.loads(args.object_frame.read_text())
    R_prev = np.array(obj["rotation_world_from_object"], dtype=np.float64)
    t = np.array(obj["translation_world_from_object"], dtype=np.float64)

    z = normalize(R_prev[:, 2])
    x_prev = normalize(R_prev[:, 0] - (R_prev[:, 0] @ z) * z)
    y_prev = normalize(np.cross(z, x_prev))
    x_prev = normalize(np.cross(y_prev, z))

    pts, w = load_points_with_weights(args.points, args.weight_mode)
    rel = pts - t.reshape(1, 3)
    rel_perp = rel - np.outer(rel @ z, z)
    rr = np.linalg.norm(rel_perp, axis=1)

    rmin = float(np.percentile(rr, args.radial_min_quantile))
    keep = rr > max(1e-8, rmin)
    rel_perp = rel_perp[keep]
    rr = rr[keep]
    w = w[keep]
    if rel_perp.shape[0] < 50:
        raise RuntimeError("Too few usable radial points for yaw anchoring.")

    u = rel_perp / rr.reshape(-1, 1)
    # Weighted circular first moment; density asymmetry from texture/track lengths anchors yaw.
    moment = (w.reshape(-1, 1) * u).sum(axis=0)
    x_target = normalize(moment)
    if x_target is None:
        raise RuntimeError("Degenerate moment for yaw anchoring.")

    conf = float(np.linalg.norm(moment) / (np.sum(w) + 1e-12))
    if conf < args.min_confidence:
        # Very symmetric distribution; do not change orientation.
        x_target = x_prev.copy()

    # Signed yaw from x_prev to x_target around z.
    cross_term = np.cross(x_prev, x_target)
    sinv = float(np.dot(cross_term, z))
    cosv = float(np.clip(np.dot(x_prev, x_target), -1.0, 1.0))
    yaw_raw = float(np.degrees(np.arctan2(sinv, cosv)))
    yaw_applied = float(np.clip(yaw_raw, -args.max_yaw_change_deg, args.max_yaw_change_deg))

    x_new = normalize(rotate_about_axis(x_prev, z, yaw_applied))
    y_new = normalize(np.cross(z, x_new))
    x_new = normalize(np.cross(y_new, z))
    R_new = np.stack([x_new, y_new, z], axis=1)
    if np.linalg.det(R_new) < 0:
        y_new = -y_new
        R_new = np.stack([x_new, y_new, z], axis=1)

    obj["rotation_world_from_object"] = R_new.tolist()
    obj["yaw_anchor_from_density"] = {
        "points_path": str(args.points),
        "num_points_total": int(pts.shape[0]),
        "num_points_used": int(rel_perp.shape[0]),
        "radial_min_quantile": float(args.radial_min_quantile),
        "weight_mode": args.weight_mode,
        "confidence": conf,
        "min_confidence": float(args.min_confidence),
        "yaw_delta_raw_deg": yaw_raw,
        "yaw_delta_applied_deg": yaw_applied,
        "max_yaw_change_deg": float(args.max_yaw_change_deg),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "object_frame.json").write_text(json.dumps(obj, indent=2))
    print("yaw_anchor_from_density", json.dumps(obj["yaw_anchor_from_density"], indent=2))


if __name__ == "__main__":
    main()
