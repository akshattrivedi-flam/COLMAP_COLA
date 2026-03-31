import argparse
import json
from pathlib import Path

import cv2
import numpy as np

EDGES = (
    (1, 5), (2, 6), (3, 7), (4, 8),
    (1, 3), (5, 7), (2, 4), (6, 8),
    (1, 2), (3, 4), (5, 6), (7, 8),
)


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


def bbox_from_points(uv):
    return np.array([uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max()], dtype=np.float64)


def iou(a, b):
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1]); ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    uu = aa + bb - inter
    return inter / uu if uu > 1e-8 else 0.0


def sample_edge_points(uv, n_per_edge=14):
    pts = []
    for s, e in EDGES:
        a = uv[s]
        b = uv[e]
        for t in np.linspace(0.0, 1.0, n_per_edge):
            pts.append(a * (1.0 - t) + b * t)
    return np.array(pts, dtype=np.float64)


def load_mask_features(mask_dir, names):
    features = {}
    for n in names:
        mp = mask_dir / (Path(n).stem + '.png')
        if not mp.exists():
            continue
        m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        bw = (m > 0).astype(np.uint8)
        ys, xs = np.where(bw > 0)
        if len(xs) < 40:
            continue
        bb = np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float64)
        ctr = np.array([xs.mean(), ys.mean()], dtype=np.float64)

        edge = cv2.Canny((bw * 255).astype(np.uint8), 50, 150)
        inv_edge = (edge == 0).astype(np.uint8)
        dist = cv2.distanceTransform(inv_edge, cv2.DIST_L2, 3)

        features[n] = dict(bbox=bb, center=ctr, dist=dist, mask=bw)
    return features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cameras', required=True, type=Path)
    ap.add_argument('--images', required=True, type=Path)
    ap.add_argument('--object_frame', required=True, type=Path)
    ap.add_argument('--out_dir', required=True, type=Path)
    ap.add_argument('--masks', required=True, type=Path)
    ap.add_argument('--sample', type=int, default=120)
    ap.add_argument('--coarse_range', type=float, default=0.003)
    ap.add_argument('--coarse_steps', type=int, default=7)
    ap.add_argument('--fine_range', type=float, default=0.0012)
    ap.add_argument('--fine_steps', type=int, default=7)
    ap.add_argument('--edge_w', type=float, default=0.7)
    ap.add_argument('--bbox_w', type=float, default=0.35)
    ap.add_argument('--center_w', type=float, default=1.2)
    args = ap.parse_args()

    cams = load_cameras(args.cameras)
    rows = load_images(args.images)
    mfeat = load_mask_features(args.masks, [r['name'] for r in rows])
    rows = [r for r in rows if r['name'] in mfeat]
    if not rows:
        raise RuntimeError('No overlapping frames with SAM2 masks')

    rng = np.random.default_rng(42)
    if len(rows) > args.sample:
        idx = rng.choice(len(rows), size=args.sample, replace=False)
        rows = [rows[i] for i in idx]

    obj = json.loads(args.object_frame.read_text())
    R = np.array(obj['rotation_world_from_object'], dtype=np.float64)
    t0 = np.array(obj['translation_world_from_object'], dtype=np.float64)
    half = np.array(obj['scale_half'], dtype=np.float64)
    key = build_keypoints(half)

    def eval_cost(tw):
        total = 0.0
        count = 0
        for r in rows:
            cam = cams[r['cid']]
            uv = project_points(R, tw, key, r, cam)
            if uv is None:
                continue
            feat = mfeat[r['name']]
            bb_m = feat['bbox']
            ctr_m = feat['center']
            dist = feat['dist']
            mask = feat['mask']

            edge_pts = sample_edge_points(uv, n_per_edge=12)
            h, w = dist.shape
            dsum = 0.0
            for p in edge_pts:
                x = int(round(p[0]))
                y = int(round(p[1]))
                if x < 0 or x >= w or y < 0 or y >= h:
                    dsum += 12.0
                else:
                    dsum += float(dist[y, x])
            edge_cost = dsum / max(1, len(edge_pts))

            bb_p = bbox_from_points(uv)
            bbox_cost = 1.0 - iou(bb_p, bb_m)

            ctr_p = np.array([(bb_p[0] + bb_p[2]) * 0.5, (bb_p[1] + bb_p[3]) * 0.5], dtype=np.float64)
            diag = np.hypot(cam['w'], cam['h']) + 1e-8
            center_cost = np.linalg.norm(ctr_p - ctr_m) / diag

            cx = int(round(uv[0, 0]))
            cy = int(round(uv[0, 1]))
            center_out = 0.0
            if cx < 0 or cx >= mask.shape[1] or cy < 0 or cy >= mask.shape[0] or mask[cy, cx] == 0:
                center_out = 0.35

            total += args.edge_w * edge_cost + args.bbox_w * bbox_cost + args.center_w * center_cost + center_out
            count += 1

        if count < max(10, len(rows) // 3):
            return None
        return total / count

    def search(center_t, rng_val, steps):
        grid = np.linspace(-rng_val, rng_val, steps)
        best_t = center_t.copy()
        best_c = eval_cost(best_t)
        for dx in grid:
            for dy in grid:
                for dz in grid:
                    off_w = R @ np.array([dx, dy, dz], dtype=np.float64)
                    tw = center_t + off_w
                    c = eval_cost(tw)
                    if c is None:
                        continue
                    if best_c is None or c < best_c:
                        best_c = c
                        best_t = tw
        return best_t, best_c

    # Coarse then fine translation refinement.
    t1, c1 = search(t0, args.coarse_range, args.coarse_steps)
    t2, c2 = search(t1, args.fine_range, args.fine_steps)

    obj['translation_world_from_object'] = t2.tolist()
    obj['refined_translation_with_sam2_edges'] = {
        'offset_world': (t2 - t0).tolist(),
        'score': float(c2 if c2 is not None else c1),
        'num_frames': int(len(rows)),
        'coarse_range': float(args.coarse_range),
        'fine_range': float(args.fine_range),
    }

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / 'object_frame.json').write_text(json.dumps(obj, indent=2))

    print('best', obj['refined_translation_with_sam2_edges'])


if __name__ == '__main__':
    main()
