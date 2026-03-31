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


def rodrigues_to_rotmat(rvec):
    R, _ = cv2.Rodrigues(rvec.astype(np.float64))
    return R


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
            dist = np.zeros(4, dtype=np.float64)
        elif model == 'PINHOLE':
            fx, fy, cx, cy = prm
            dist = np.zeros(4, dtype=np.float64)
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


EDGES = [
    (1, 2), (1, 3), (1, 5),
    (2, 4), (2, 6),
    (3, 4), (3, 7),
    (4, 8),
    (5, 6), (5, 7),
    (6, 8),
    (7, 8),
]


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


def rasterize_hull(uv, w, h):
    m = np.zeros((h, w), dtype=np.uint8)
    hull = cv2.convexHull(uv.astype(np.float32)).astype(np.int32)
    if hull.shape[0] >= 3:
        cv2.fillConvexPoly(m, hull.reshape(-1, 2), 1)
    return m


def iou(a, b):
    inter = np.logical_and(a, b).sum()
    uni = np.logical_or(a, b).sum()
    return float(inter) / float(uni) if uni > 0 else 0.0


def sample_wire_points(uv, n=14):
    pts = []
    for i, j in EDGES:
        p0 = uv[i]
        p1 = uv[j]
        for t in np.linspace(0.0, 1.0, n):
            pts.append((1.0 - t) * p0 + t * p1)
    return np.array(pts, dtype=np.float64)


def load_mask_feats(mask_dir, names, downsample):
    feats = {}
    for n in names:
        mp = mask_dir / (Path(n).stem + '.png')
        if not mp.exists():
            continue
        m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        bw = (m > 0).astype(np.uint8)
        if bw.sum() < 40:
            continue

        h, w = bw.shape
        dh = max(1, h // downsample)
        dw = max(1, w // downsample)
        bwd = cv2.resize(bw, (dw, dh), interpolation=cv2.INTER_NEAREST)

        ys, xs = np.where(bwd > 0)
        if len(xs) < 12:
            continue

        ctr = np.array([xs.mean(), ys.mean()], dtype=np.float64)
        edge = cv2.Canny((bwd * 255).astype(np.uint8), 40, 120)
        non_edge = (edge == 0).astype(np.uint8)
        dt = cv2.distanceTransform(non_edge, cv2.DIST_L2, 3)

        feats[n] = dict(mask=bwd, center=ctr, w=dw, h=dh, dt=dt)
    return feats


def evaluate(rows, cams, feats, R, t, half, downsample):
    key = build_keypoints(half)
    ious = []
    centers = []
    edge_dists = []

    for r in rows:
        cam = cams[r['cid']]
        uv = project_points(R, t, key, r, cam)
        if uv is None:
            continue

        f = feats[r['name']]
        uvd = uv / float(downsample)

        pred = rasterize_hull(uvd, f['w'], f['h'])
        i = iou(pred > 0, f['mask'] > 0)
        ious.append(i)

        ys, xs = np.where(pred > 0)
        if len(xs) > 0:
            cp = np.array([xs.mean(), ys.mean()], dtype=np.float64)
            centers.append(np.linalg.norm(cp - f['center']) * float(downsample))

        wire = sample_wire_points(uvd, n=12)
        wx = np.clip(np.round(wire[:, 0]).astype(np.int32), 0, f['w'] - 1)
        wy = np.clip(np.round(wire[:, 1]).astype(np.int32), 0, f['h'] - 1)
        d = f['dt'][wy, wx]
        edge_dists.append(float(np.mean(d)) * float(downsample))

    if len(ious) < max(25, len(rows) // 4):
        return None

    ious = np.array(ious, dtype=np.float64)
    centers = np.array(centers if centers else [1e6], dtype=np.float64)
    edge_dists = np.array(edge_dists, dtype=np.float64)

    # Lower is better
    loss = (1.0 - float(np.mean(ious))) + 0.22 * float(np.median(centers) / 10.0) + 0.30 * float(np.mean(edge_dists) / 10.0)

    return dict(
        loss=loss,
        iou_mean=float(np.mean(ious)),
        iou_median=float(np.median(ious)),
        center_mean=float(np.mean(centers)),
        center_median=float(np.median(centers)),
        edge_mean=float(np.mean(edge_dists)),
        edge_median=float(np.median(edge_dists)),
        n=len(ious),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cameras', required=True, type=Path)
    ap.add_argument('--images', required=True, type=Path)
    ap.add_argument('--masks', required=True, type=Path)
    ap.add_argument('--object_frame', required=True, type=Path)
    ap.add_argument('--out_dir', required=True, type=Path)
    ap.add_argument('--sample', type=int, default=260)
    ap.add_argument('--downsample', type=int, default=2)
    ap.add_argument('--rotation_only', action='store_true')
    ap.add_argument('--max_rot_deg', type=float, default=20.0)
    ap.add_argument('--start_rot_deg', type=float, default=8.0)
    args = ap.parse_args()

    cams = load_cameras(args.cameras)
    rows = load_images(args.images)
    feats = load_mask_feats(args.masks, [r['name'] for r in rows], args.downsample)
    rows = [r for r in rows if r['name'] in feats]
    if not rows:
        raise RuntimeError('No overlap between frames and SAM2 masks')

    rng = np.random.default_rng(42)
    if len(rows) > args.sample:
        idx = rng.choice(len(rows), size=args.sample, replace=False)
        rows = [rows[i] for i in idx]

    obj = json.loads(args.object_frame.read_text())
    R_base = np.array(obj['rotation_world_from_object'], dtype=np.float64)
    t_base = np.array(obj['translation_world_from_object'], dtype=np.float64)
    half = np.array(obj['scale_half'], dtype=np.float64)

    # params: [rx_deg, ry_deg, rz_deg, tx, ty, tz]
    def clamp(p):
        q = p.copy()
        q[:3] = np.clip(q[:3], -args.max_rot_deg, args.max_rot_deg)
        if args.rotation_only:
            q[3:] = 0.0
        else:
            q[3:] = np.clip(q[3:], -0.002, 0.002)
        return q

    def compose(p):
        rx, ry, rz, tx, ty, tz = clamp(p)
        R_delta = rodrigues_to_rotmat(np.deg2rad(np.array([rx, ry, rz], dtype=np.float64)))
        R = R_base @ R_delta
        t = t_base + R @ np.array([tx, ty, tz], dtype=np.float64)
        return R, t

    def score(p):
        R, t = compose(p)
        return evaluate(rows, cams, feats, R, t, half, args.downsample)

    s = float(args.start_rot_deg)
    starts = [
        np.array([0, 0, 0, 0, 0, 0], dtype=np.float64),
        np.array([s, 0, 0, 0, 0, 0], dtype=np.float64),
        np.array([-s, 0, 0, 0, 0, 0], dtype=np.float64),
        np.array([0, s, 0, 0, 0, 0], dtype=np.float64),
        np.array([0, -s, 0, 0, 0, 0], dtype=np.float64),
        np.array([0, 0, s, 0, 0, 0], dtype=np.float64),
        np.array([0, 0, -s, 0, 0, 0], dtype=np.float64),
        np.array([1.2 * s, 1.0 * s, -0.5 * s, 0, 0, 0], dtype=np.float64),
        np.array([-1.2 * s, -1.0 * s, 0.5 * s, 0, 0, 0], dtype=np.float64),
    ]

    steps = [
        np.array([2.2, 2.2, 2.2, 0.0009, 0.0009, 0.0009], dtype=np.float64),
        np.array([0.9, 0.9, 0.9, 0.00035, 0.00035, 0.00035], dtype=np.float64),
        np.array([0.35, 0.35, 0.35, 0.00015, 0.00015, 0.00015], dtype=np.float64),
    ]
    if args.rotation_only:
        steps = [
            np.array([2.2, 2.2, 2.2, 0.0, 0.0, 0.0], dtype=np.float64),
            np.array([0.9, 0.9, 0.9, 0.0, 0.0, 0.0], dtype=np.float64),
            np.array([0.35, 0.35, 0.35, 0.0, 0.0, 0.0], dtype=np.float64),
        ]

    best_m = None
    best_p = None

    for s in starts:
        p = clamp(s)
        m = score(p)
        if m is None:
            continue

        for st in steps:
            improved = True
            while improved:
                improved = False
                for i in range(len(p)):
                    for sign in (-1.0, 1.0):
                        c = p.copy()
                        c[i] += sign * st[i]
                        c = clamp(c)
                        cm = score(c)
                        if cm is None:
                            continue
                        # prioritize lower loss, then higher IoU, then lower edge error
                        kc = (-cm['loss'], cm['iou_mean'], -cm['edge_mean'], -cm['center_median'])
                        km = (-m['loss'], m['iou_mean'], -m['edge_mean'], -m['center_median'])
                        if kc > km:
                            p = c
                            m = cm
                            improved = True

        if best_m is None:
            best_m = m
            best_p = p.copy()
        else:
            kb = (-best_m['loss'], best_m['iou_mean'], -best_m['edge_mean'], -best_m['center_median'])
            km = (-m['loss'], m['iou_mean'], -m['edge_mean'], -m['center_median'])
            if km > kb:
                best_m = m
                best_p = p.copy()

    if best_p is None:
        raise RuntimeError('No valid rotation refinement result')

    R_new, t_new = compose(best_p)
    key = build_keypoints(half)

    obj['rotation_world_from_object'] = R_new.tolist()
    obj['translation_world_from_object'] = t_new.tolist()
    obj['rotation_refined_with_sam2_edges_v2'] = {
        'params': {
            'rx_deg': float(best_p[0]),
            'ry_deg': float(best_p[1]),
            'rz_deg': float(best_p[2]),
            't_local': [float(best_p[3]), float(best_p[4]), float(best_p[5])],
        },
        'metrics': best_m,
        'num_frames': int(len(rows)),
        'downsample': int(args.downsample),
        'rotation_only': bool(args.rotation_only),
    }

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / 'object_frame.json').write_text(json.dumps(obj, indent=2))
    np.save(out / 'cuboid_3d_keypoints.npy', key)
    (out / 'cuboid_3d_keypoints.json').write_text(json.dumps({'keypoints_object': key.tolist()}, indent=2))

    print('best_params', best_p.tolist())
    print('best_metrics', json.dumps(best_m, indent=2))


if __name__ == '__main__':
    main()
