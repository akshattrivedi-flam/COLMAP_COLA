import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def qvec_to_rotmat(qvec):
    qw, qx, qy, qz = qvec
    return np.array([
        [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy],
    ], dtype=np.float64)


def load_cameras(path):
    cams = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        p = line.split()
        cid = int(p[0])
        model = p[1]
        w = int(p[2])
        h = int(p[3])
        prm = list(map(float, p[4:]))
        if model == 'SIMPLE_PINHOLE':
            f, cx, cy = prm
            fx = fy = f
            dist = np.zeros(4)
        elif model == 'PINHOLE':
            fx, fy, cx, cy = prm
            dist = np.zeros(4)
        else:
            fx, fy, cx, cy = prm[:4]
            k1 = prm[4] if len(prm) > 4 else 0.0
            k2 = prm[5] if len(prm) > 5 else 0.0
            p1 = prm[6] if len(prm) > 6 else 0.0
            p2 = prm[7] if len(prm) > 7 else 0.0
            dist = np.array([k1, k2, p1, p2], dtype=np.float64)
        cams[cid] = dict(w=w, h=h, fx=fx, fy=fy, cx=cx, cy=cy, dist=dist)
    return cams


def load_images(path):
    rows = []
    lines = [l.rstrip('\n') for l in path.read_text().splitlines()]
    i = 0
    while i < len(lines):
        ln = lines[i].strip()
        if not ln or ln.startswith('#'):
            i += 1
            continue
        p = ln.split()
        if len(p) < 10:
            i += 1
            continue
        rows.append(dict(
            iid=int(p[0]),
            qvec=np.array(list(map(float, p[1:5])), dtype=np.float64),
            tvec=np.array(list(map(float, p[5:8])), dtype=np.float64),
            cid=int(p[8]),
            name=p[9],
        ))
        i += 2
    return rows


def build_keypoints(half):
    w, h, d = half
    return np.array([
        [0.0, 0.0, 0.0],
        [-w, -h, -d], [-w, -h, +d], [-w, +h, -d], [-w, +h, +d],
        [+w, -h, -d], [+w, -h, +d], [+w, +h, -d], [+w, +h, +d],
    ], dtype=np.float64)


def project_points(Rwo, two, key_obj, row, cam):
    Rcw = qvec_to_rotmat(row['qvec'])
    tcw = row['tvec']
    Xw = (Rwo @ key_obj.T) + two.reshape(3, 1)
    Xc = (Rcw @ Xw) + tcw.reshape(3, 1)
    Xc = Xc.T
    if np.any(Xc[:, 2] <= 1e-8):
        return None
    x = Xc[:, 0] / Xc[:, 2]
    y = Xc[:, 1] / Xc[:, 2]
    k1, k2, p1, p2 = cam['dist']
    r2 = x * x + y * y
    radial = 1.0 + k1 * r2 + k2 * r2 * r2
    xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    u = cam['fx'] * xd + cam['cx']
    v = cam['fy'] * yd + cam['cy']
    return np.stack([u, v], axis=1)


def iou(mask_a, mask_b):
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    return float(inter) / float(union)


def load_masks(mask_dir, names, downsample):
    feats = {}
    for n in names:
        p = mask_dir / (Path(n).stem + '.png')
        if not p.exists():
            continue
        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        bw = (m > 0).astype(np.uint8)
        if bw.sum() < 30:
            continue
        h, w = bw.shape
        dh = max(1, h // downsample)
        dw = max(1, w // downsample)
        bwd = cv2.resize(bw, (dw, dh), interpolation=cv2.INTER_NEAREST)
        ys, xs = np.where(bwd > 0)
        if len(xs) < 10:
            continue
        ctr = np.array([xs.mean(), ys.mean()], dtype=np.float64)
        feats[n] = dict(mask=bwd, center=ctr, w=dw, h=dh)
    return feats


def rasterize_hull(uv, w, h):
    mask = np.zeros((h, w), dtype=np.uint8)
    hull = cv2.convexHull(uv.astype(np.float32)).astype(np.int32)
    if hull.shape[0] >= 3:
        cv2.fillConvexPoly(mask, hull.reshape(-1, 2), 1)
    return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cameras', required=True, type=Path)
    ap.add_argument('--images', required=True, type=Path)
    ap.add_argument('--masks', required=True, type=Path)
    ap.add_argument('--object_frame', required=True, type=Path)
    ap.add_argument('--out_dir', required=True, type=Path)
    ap.add_argument('--sample', type=int, default=220)
    ap.add_argument('--downsample', type=int, default=3)
    ap.add_argument('--target_center', type=float, default=7.0)
    ap.add_argument('--trans_step', type=float, default=0.0008)
    ap.add_argument('--trans_levels', type=int, default=3)
    args = ap.parse_args()

    cams = load_cameras(args.cameras)
    rows = load_images(args.images)
    masks = load_masks(args.masks, [r['name'] for r in rows], args.downsample)
    rows = [r for r in rows if r['name'] in masks]

    rng = np.random.default_rng(42)
    if len(rows) > args.sample:
        idx = rng.choice(len(rows), size=args.sample, replace=False)
        rows = [rows[i] for i in idx]

    obj = json.loads(args.object_frame.read_text())
    R0 = np.array(obj['rotation_world_from_object'], dtype=np.float64)
    t0 = np.array(obj['translation_world_from_object'], dtype=np.float64)
    half0 = np.array(obj['scale_half'], dtype=np.float64)

    base = obj.get('silhouette_joint_refined_with_sam2', {})
    yaw0 = float(base.get('yaw_deg', 0.0))
    sr0 = float(base.get('scale_radius', 1.0))
    sh0 = float(base.get('scale_height', 1.0))

    x0 = R0[:, 0]
    y0 = R0[:, 1]
    z0 = R0[:, 2]

    def make_R(yaw_deg):
        th = np.deg2rad(yaw_deg)
        c = np.cos(th)
        s = np.sin(th)
        x = c * x0 + s * y0
        y = -s * x0 + c * y0
        return np.stack([x, y, z0], axis=1)

    def eval_metrics(yaw, sr, sh, t_local):
        R = make_R(yaw)
        half = half0 * np.array([sr, sr, sh], dtype=np.float64)
        key = build_keypoints(half)
        tw = t0 + R @ t_local

        ious = []
        centers = []
        for r in rows:
            cam = cams[r['cid']]
            uv = project_points(R, tw, key, r, cam)
            if uv is None:
                continue
            f = masks[r['name']]
            uvd = uv / float(args.downsample)
            pred = rasterize_hull(uvd, f['w'], f['h'])
            i = iou(pred > 0, f['mask'] > 0)
            ious.append(i)

            ys, xs = np.where(pred > 0)
            if len(xs) == 0:
                centers.append(1e6)
            else:
                pc = np.array([xs.mean(), ys.mean()], dtype=np.float64)
                err = np.linalg.norm(pc - f['center']) * float(args.downsample)
                centers.append(err)

        if len(ious) < max(20, len(rows) // 4):
            return None
        ious = np.array(ious)
        centers = np.array(centers)
        return dict(
            iou_mean=float(ious.mean()),
            iou_median=float(np.median(ious)),
            center_mean=float(centers.mean()),
            center_median=float(np.median(centers)),
            n=len(ious),
            R=R,
            t=tw,
            half=half,
        )

    # Local Pareto neighborhood search around current best.
    yaw_grid = yaw0 + np.array([-6, -4, -2, 0, 2, 4, 6], dtype=np.float64)
    sr_grid = sr0 * np.array([0.90, 0.95, 1.00, 1.05, 1.10], dtype=np.float64)
    sh_grid = sh0 * np.array([0.90, 0.95, 1.00, 1.05, 1.10], dtype=np.float64)
    if args.trans_levels < 1:
        raise ValueError('trans_levels must be >= 1')
    if args.trans_levels == 1:
        t_vals = np.array([0.0], dtype=np.float64)
    else:
        k = args.trans_levels // 2
        t_vals = np.linspace(-args.trans_step * k, args.trans_step * k, args.trans_levels, dtype=np.float64)

    best = None
    best_params = None

    # Priority: satisfy center constraint, then maximize IoU.
    for yaw in yaw_grid:
        for sr in sr_grid:
            for sh in sh_grid:
                for tx in t_vals:
                    for ty in t_vals:
                        for tz in t_vals:
                            m = eval_metrics(yaw, sr, sh, np.array([tx, ty, tz], dtype=np.float64))
                            if m is None:
                                continue

                            feasible = (m['center_median'] <= args.target_center)
                            key = (
                                1 if feasible else 0,
                                m['iou_mean'],
                                -m['center_median'],
                                m['iou_median'],
                                -m['center_mean'],
                            )
                            if best is None or key > best:
                                best = key
                                best_params = (yaw, sr, sh, np.array([tx, ty, tz], dtype=np.float64), m)

    if best_params is None:
        raise RuntimeError('Pareto search found no valid candidate')

    yaw, sr, sh, t_local, m = best_params
    R = m['R']
    tw = m['t']
    half = m['half']
    key = build_keypoints(half)

    obj['rotation_world_from_object'] = R.tolist()
    obj['translation_world_from_object'] = tw.tolist()
    obj['scale_half'] = half.tolist()
    obj['scale_full'] = (2.0 * half).tolist()
    obj['pareto_refined_with_sam2'] = {
        'yaw_deg': float(yaw),
        'scale_radius': float(sr),
        'scale_height': float(sh),
        'translation_offset_world': (tw - t0).tolist(),
        'iou_mean': m['iou_mean'],
        'iou_median': m['iou_median'],
        'center_mean_px': m['center_mean'],
        'center_median_px': m['center_median'],
        'n_frames': m['n'],
        'target_center': float(args.target_center),
    }

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / 'object_frame.json').write_text(json.dumps(obj, indent=2))
    np.save(out / 'cuboid_3d_keypoints.npy', key)
    (out / 'cuboid_3d_keypoints.json').write_text(json.dumps({'keypoints_object': key.tolist()}, indent=2))

    print('best', obj['pareto_refined_with_sam2'])


if __name__ == '__main__':
    main()
