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


def rasterize_silhouette(uv, w, h):
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


def load_mask_features(mask_dir, names, ds):
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
        dh = max(1, h // ds)
        dw = max(1, w // ds)
        bwd = cv2.resize(bw, (dw, dh), interpolation=cv2.INTER_NEAREST)
        ys, xs = np.where(bwd > 0)
        if len(xs) < 10:
            continue
        ctr = np.array([xs.mean(), ys.mean()], dtype=np.float64)
        feats[n] = dict(mask=bwd, center=ctr, w=dw, h=dh)
    return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cameras', required=True, type=Path)
    ap.add_argument('--images', required=True, type=Path)
    ap.add_argument('--object_frame', required=True, type=Path)
    ap.add_argument('--out_dir', required=True, type=Path)
    ap.add_argument('--masks', required=True, type=Path)
    ap.add_argument('--sample', type=int, default=120)
    ap.add_argument('--downsample', type=int, default=4)
    args = ap.parse_args()

    cams = load_cameras(args.cameras)
    rows = load_images(args.images)
    feats = load_mask_features(args.masks, [r['name'] for r in rows], args.downsample)
    rows = [r for r in rows if r['name'] in feats]
    if not rows:
        raise RuntimeError('No overlapping rows with SAM2 masks')

    rng = np.random.default_rng(42)
    if len(rows) > args.sample:
        idx = rng.choice(len(rows), size=args.sample, replace=False)
        rows = [rows[i] for i in idx]

    obj = json.loads(args.object_frame.read_text())
    R0 = np.array(obj['rotation_world_from_object'], dtype=np.float64)
    t0 = np.array(obj['translation_world_from_object'], dtype=np.float64)
    half0 = np.array(obj['scale_half'], dtype=np.float64)

    base = obj.get('joint_refined_with_sam2', obj.get('refined_with_sam2_edges', obj.get('calibrated_from_sam2_masks', {})))
    yaw0 = float(base.get('yaw_deg', 0.0))
    sr0 = float(base.get('scale_radius', 1.0))
    sh0 = float(base.get('scale_height', 1.0))

    x0 = R0[:, 0]; y0 = R0[:, 1]; z0 = R0[:, 2]

    def make_R(yaw_deg):
        th = np.deg2rad(yaw_deg)
        c = np.cos(th); s = np.sin(th)
        x = c * x0 + s * y0
        y = -s * x0 + c * y0
        return np.stack([x, y, z0], axis=1)

    def clamp(p):
        # [yaw, sr, sh, tx, ty, tz]
        p[1] = np.clip(p[1], 0.45, 2.0)
        p[2] = np.clip(p[2], 0.35, 2.0)
        p[3:] = np.clip(p[3:], -0.008, 0.008)
        return p

    def eval_cost(p):
        p = clamp(p.copy())
        yaw, sr, sh, tx, ty, tz = p
        R = make_R(yaw)
        half = half0 * np.array([sr, sr, sh], dtype=np.float64)
        key = build_keypoints(half)
        tw = t0 + R @ np.array([tx, ty, tz], dtype=np.float64)

        total = 0.0
        cnt = 0
        for r in rows:
            cam = cams[r['cid']]
            uv = project_points(R, tw, key, r, cam)
            if uv is None:
                continue
            f = feats[r['name']]
            uvd = uv / float(args.downsample)
            pred = rasterize_silhouette(uvd, f['w'], f['h'])
            i = iou(pred > 0, f['mask'] > 0)

            ys, xs = np.where(pred > 0)
            if len(xs) == 0:
                total += 2.0
                cnt += 1
                continue
            ctr_p = np.array([xs.mean(), ys.mean()], dtype=np.float64)
            diag = np.hypot(f['w'], f['h']) + 1e-8
            c_err = np.linalg.norm(ctr_p - f['center']) / diag

            # silhouette IoU dominates; center keeps it anchored.
            total += (1.0 - i) + 0.85 * c_err
            cnt += 1

        if cnt < max(12, len(rows) // 3):
            return None
        return total / cnt

    # Multi-start coordinate descent
    starts = []
    for dy in (-24, -12, 0, 12, 24):
        for msr in (0.85, 1.0, 1.15):
            for msh in (0.85, 1.0, 1.15):
                starts.append(np.array([yaw0 + dy, sr0 * msr, sh0 * msh, 0.0, 0.0, 0.0], dtype=np.float64))

    stages = [
        np.array([8.0, 0.18, 0.18, 0.0022, 0.0022, 0.0022], dtype=np.float64),
        np.array([3.0, 0.08, 0.08, 0.0009, 0.0009, 0.0009], dtype=np.float64),
        np.array([1.0, 0.03, 0.03, 0.00035, 0.00035, 0.00035], dtype=np.float64),
    ]

    best_c = None
    best_p = None

    for s in starts:
        p = clamp(s.copy())
        c = eval_cost(p)
        if c is None:
            continue
        for step in stages:
            improved = True
            while improved:
                improved = False
                for i in range(6):
                    for sign in (-1.0, 1.0):
                        cand = p.copy()
                        cand[i] += sign * step[i]
                        cand = clamp(cand)
                        cc = eval_cost(cand)
                        if cc is None:
                            continue
                        if cc + 1e-10 < c:
                            p = cand
                            c = cc
                            improved = True
        if best_c is None or c < best_c:
            best_c = c
            best_p = p.copy()

    if best_p is None:
        raise RuntimeError('No valid solution from silhouette optimization')

    yaw, sr, sh, tx, ty, tz = best_p
    R = make_R(yaw)
    half = half0 * np.array([sr, sr, sh], dtype=np.float64)
    key = build_keypoints(half)
    tw = t0 + R @ np.array([tx, ty, tz], dtype=np.float64)

    obj['rotation_world_from_object'] = R.tolist()
    obj['translation_world_from_object'] = tw.tolist()
    obj['scale_half'] = half.tolist()
    obj['scale_full'] = (2.0 * half).tolist()
    obj['silhouette_joint_refined_with_sam2'] = {
        'yaw_deg': float(yaw),
        'scale_radius': float(sr),
        'scale_height': float(sh),
        'translation_offset_world': (tw - t0).tolist(),
        'score': float(best_c),
        'num_frames': int(len(rows)),
        'downsample': int(args.downsample),
    }

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / 'object_frame.json').write_text(json.dumps(obj, indent=2))
    np.save(out / 'cuboid_3d_keypoints.npy', key)
    (out / 'cuboid_3d_keypoints.json').write_text(json.dumps({'keypoints_object': key.tolist()}, indent=2))

    print('best', obj['silhouette_joint_refined_with_sam2'])


if __name__ == '__main__':
    main()
