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
    mask = np.zeros((h, w), dtype=np.uint8)
    hull = cv2.convexHull(uv.astype(np.float32)).astype(np.int32)
    if hull.shape[0] >= 3:
        cv2.fillConvexPoly(mask, hull.reshape(-1, 2), 1)
    return mask


def iou(mask_a, mask_b):
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    return float(inter) / float(union)


def load_mask_features(mask_dir, names, downsample):
    feats = {}
    for n in names:
        p = mask_dir / (Path(n).stem + '.png')
        if not p.exists():
            continue
        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
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
        if len(xs) < 10:
            continue
        ctr = np.array([xs.mean(), ys.mean()], dtype=np.float64)
        x0, x1 = float(xs.min()), float(xs.max())
        y0, y1 = float(ys.min()), float(ys.max())
        bwid = max(1.0, x1 - x0 + 1.0)
        bhei = max(1.0, y1 - y0 + 1.0)

        feats[n] = dict(mask=bwd, center=ctr, w=dw, h=dh, bw=bwid, bh=bhei)
    return feats


def evaluate(rows, cams, feats, R, t, half, downsample):
    key = build_keypoints(half)
    losses = []
    ious = []
    center_px = []
    used = 0

    for r in rows:
        cam = cams[r['cid']]
        uv = project_points(R, t, key, r, cam)
        if uv is None:
            continue

        f = feats[r['name']]
        uvd = uv / float(downsample)
        pred = rasterize_hull(uvd, f['w'], f['h'])

        i = iou(pred > 0, f['mask'] > 0)

        ys, xs = np.where(pred > 0)
        if len(xs) < 5:
            losses.append(3.0)
            continue

        pctr = np.array([xs.mean(), ys.mean()], dtype=np.float64)
        perr = np.linalg.norm(pctr - f['center'])
        p_x0, p_x1 = float(xs.min()), float(xs.max())
        p_y0, p_y1 = float(ys.min()), float(ys.max())
        pbw = max(1.0, p_x1 - p_x0 + 1.0)
        pbh = max(1.0, p_y1 - p_y0 + 1.0)

        c_norm = perr / (np.hypot(f['w'], f['h']) + 1e-8)
        w_err = abs(pbw - f['bw']) / f['bw']
        h_err = abs(pbh - f['bh']) / f['bh']

        # IoU primary, plus center/size terms for scale anchoring.
        loss = (1.0 - i) + 0.55 * c_norm + 0.45 * (w_err + h_err)
        losses.append(loss)
        ious.append(i)
        center_px.append(perr * float(downsample))
        used += 1

    if used < max(20, len(rows) // 4):
        return None

    losses = np.array(losses, dtype=np.float64)
    ious = np.array(ious, dtype=np.float64)
    center_px = np.array(center_px, dtype=np.float64)

    return dict(
        loss=float(np.mean(losses)),
        iou_mean=float(np.mean(ious)),
        iou_median=float(np.median(ious)),
        center_mean=float(np.mean(center_px)),
        center_median=float(np.median(center_px)),
        n=int(used),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cameras', required=True, type=Path)
    ap.add_argument('--images', required=True, type=Path)
    ap.add_argument('--masks', required=True, type=Path)
    ap.add_argument('--object_frame', required=True, type=Path)
    ap.add_argument('--out_dir', required=True, type=Path)
    ap.add_argument('--sample', type=int, default=220)
    ap.add_argument('--downsample', type=int, default=3)
    args = ap.parse_args()

    cams = load_cameras(args.cameras)
    rows = load_images(args.images)
    feats = load_mask_features(args.masks, [r['name'] for r in rows], args.downsample)
    rows = [r for r in rows if r['name'] in feats]

    if not rows:
        raise RuntimeError('No overlap between COLMAP frames and SAM2 masks')

    rng = np.random.default_rng(42)
    if len(rows) > args.sample:
        idx = rng.choice(len(rows), size=args.sample, replace=False)
        rows = [rows[i] for i in idx]

    obj = json.loads(args.object_frame.read_text())
    R0 = np.array(obj['rotation_world_from_object'], dtype=np.float64)
    t0 = np.array(obj['translation_world_from_object'], dtype=np.float64)
    half0 = np.array(obj['scale_half'], dtype=np.float64)

    # params = [rx_deg, ry_deg, rz_deg, sr, sh, tx, ty, tz]
    def clamp(p):
        q = p.copy()
        q[:3] = np.clip(q[:3], -12.0, 12.0)
        q[3] = np.clip(q[3], 0.70, 1.30)
        q[4] = np.clip(q[4], 0.70, 1.30)
        q[5:] = np.clip(q[5:], -0.0035, 0.0035)
        return q

    def compose(p):
        rx, ry, rz, sr, sh, tx, ty, tz = clamp(p)
        R_delta = rodrigues_to_rotmat(np.deg2rad(np.array([rx, ry, rz], dtype=np.float64)))
        R = R0 @ R_delta
        half = half0 * np.array([sr, sr, sh], dtype=np.float64)
        t = t0 + R @ np.array([tx, ty, tz], dtype=np.float64)
        return R, t, half

    def score(p):
        R, t, half = compose(p)
        m = evaluate(rows, cams, feats, R, t, half, args.downsample)
        if m is None:
            return None
        return m

    # Multi-start coordinate descent.
    starts = [
        np.array([0, 0, 0, 1, 1, 0, 0, 0], dtype=np.float64),
        np.array([4, 0, 0, 1, 1, 0, 0, 0], dtype=np.float64),
        np.array([-4, 0, 0, 1, 1, 0, 0, 0], dtype=np.float64),
        np.array([0, 4, 0, 1, 1, 0, 0, 0], dtype=np.float64),
        np.array([0, -4, 0, 1, 1, 0, 0, 0], dtype=np.float64),
        np.array([0, 0, 4, 1, 1, 0, 0, 0], dtype=np.float64),
        np.array([0, 0, -4, 1, 1, 0, 0, 0], dtype=np.float64),
        np.array([0, 0, 0, 0.92, 1.08, 0, 0, 0], dtype=np.float64),
        np.array([0, 0, 0, 1.08, 0.92, 0, 0, 0], dtype=np.float64),
    ]

    steps = [
        np.array([2.5, 2.5, 2.5, 0.06, 0.06, 0.0012, 0.0012, 0.0012], dtype=np.float64),
        np.array([1.0, 1.0, 1.0, 0.025, 0.025, 0.0005, 0.0005, 0.0005], dtype=np.float64),
        np.array([0.4, 0.4, 0.4, 0.01, 0.01, 0.0002, 0.0002, 0.0002], dtype=np.float64),
    ]

    best = None
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
                        cand = p.copy()
                        cand[i] += sign * st[i]
                        cand = clamp(cand)
                        cm = score(cand)
                        if cm is None:
                            continue
                        # Primary: IoU mean; secondary: lower center median/mean; tertiary: iou median.
                        key_cand = (cm['iou_mean'], -cm['center_median'], -cm['center_mean'], cm['iou_median'])
                        key_cur = (m['iou_mean'], -m['center_median'], -m['center_mean'], m['iou_median'])
                        if key_cand > key_cur:
                            p = cand
                            m = cm
                            improved = True

        key_best = None if best is None else (best['iou_mean'], -best['center_median'], -best['center_mean'], best['iou_median'])
        key_cur = (m['iou_mean'], -m['center_median'], -m['center_mean'], m['iou_median'])
        if best is None or key_cur > key_best:
            best = m
            best_p = p.copy()

    if best_p is None:
        raise RuntimeError('No valid candidate found for full-pose refinement')

    R, t, half = compose(best_p)
    key = build_keypoints(half)

    obj['rotation_world_from_object'] = R.tolist()
    obj['translation_world_from_object'] = t.tolist()
    obj['scale_half'] = half.tolist()
    obj['scale_full'] = (2.0 * half).tolist()
    obj['fullpose_refined_with_sam2'] = {
        'params': {
            'rx_deg': float(best_p[0]),
            'ry_deg': float(best_p[1]),
            'rz_deg': float(best_p[2]),
            'scale_radius': float(best_p[3]),
            'scale_height': float(best_p[4]),
            't_local': [float(best_p[5]), float(best_p[6]), float(best_p[7])],
        },
        'metrics': best,
        'num_frames': int(len(rows)),
        'downsample': int(args.downsample),
    }

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / 'object_frame.json').write_text(json.dumps(obj, indent=2))
    np.save(out / 'cuboid_3d_keypoints.npy', key)
    (out / 'cuboid_3d_keypoints.json').write_text(json.dumps({'keypoints_object': key.tolist()}, indent=2))

    print('best_metrics', json.dumps(best, indent=2))
    print('best_params', best_p.tolist())


if __name__ == '__main__':
    main()
