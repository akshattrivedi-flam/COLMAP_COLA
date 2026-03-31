import argparse
import json
from pathlib import Path
import numpy as np
import cv2


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
        cid = int(p[0]); model = p[1]
        w = int(p[2]); h = int(p[3])
        prm = list(map(float, p[4:]))
        if model == 'SIMPLE_PINHOLE':
            f, cx, cy = prm; fx = fy = f; dist = np.zeros(4)
        elif model == 'PINHOLE':
            fx, fy, cx, cy = prm; dist = np.zeros(4)
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


def load_mask_stats(mask_dir, names):
    stats = {}
    for n in names:
        mp = mask_dir / (Path(n).stem + '.png')
        if not mp.exists():
            continue
        m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        ys, xs = np.where(m > 0)
        if len(xs) < 40:
            continue
        bb = np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float64)
        c = np.array([xs.mean(), ys.mean()], dtype=np.float64)
        stats[n] = (bb, c)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cameras', required=True, type=Path)
    ap.add_argument('--images', required=True, type=Path)
    ap.add_argument('--object_frame', required=True, type=Path)
    ap.add_argument('--out_dir', required=True, type=Path)
    ap.add_argument('--masks', required=True, type=Path)
    ap.add_argument('--sample', type=int, default=120)
    ap.add_argument('--yaw_step', type=float, default=5.0)
    ap.add_argument('--scale_min', type=float, default=0.50)
    ap.add_argument('--scale_max', type=float, default=1.30)
    ap.add_argument('--scale_steps', type=int, default=17)
    ap.add_argument('--trans_range', type=float, default=0.006)
    ap.add_argument('--trans_steps', type=int, default=7)
    args = ap.parse_args()

    cams = load_cameras(args.cameras)
    rows = load_images(args.images)
    names = [r['name'] for r in rows]
    mstats = load_mask_stats(args.masks, names)
    rows = [r for r in rows if r['name'] in mstats]

    if len(rows) == 0:
        raise RuntimeError('No frames overlap between images.txt and SAM2 masks')

    rng = np.random.default_rng(42)
    if len(rows) > args.sample:
        idx = rng.choice(len(rows), size=args.sample, replace=False)
        rows = [rows[i] for i in idx]

    obj = json.loads(args.object_frame.read_text())
    R0 = np.array(obj['rotation_world_from_object'], dtype=np.float64)
    t0 = np.array(obj['translation_world_from_object'], dtype=np.float64)
    half0 = np.array(obj['scale_half'], dtype=np.float64)

    x0 = R0[:, 0]; y0 = R0[:, 1]; z0 = R0[:, 2]

    def make_R(yaw_deg):
        t = np.deg2rad(yaw_deg)
        c = np.cos(t); s = np.sin(t)
        x = c * x0 + s * y0
        y = -s * x0 + c * y0
        R = np.stack([x, y, z0], axis=1)
        return R

    best = None
    best_par = None

    yaw_grid = np.arange(0.0, 180.0, args.yaw_step)
    s_grid = np.linspace(args.scale_min, args.scale_max, args.scale_steps)

    def eval_cost(R, tvec, key):
        cst = 0.0
        cnt = 0
        for r in rows:
            cam = cams[r['cid']]
            uv = project_points(R, tvec, key, r, cam)
            if uv is None:
                continue
            pb = bbox_from_points(uv)
            mb, mc = mstats[r['name']]
            i = iou(pb, mb)
            pc = np.array([(pb[0] + pb[2]) * 0.5, (pb[1] + pb[3]) * 0.5])
            diag = np.hypot(cam['w'], cam['h']) + 1e-8
            c_err = np.linalg.norm(pc - mc) / diag
            pw, ph = pb[2] - pb[0], pb[3] - pb[1]
            mw, mh = mb[2] - mb[0], mb[3] - mb[1]
            s_err = abs(np.log((pw + 1e-8) / (mw + 1e-8))) + abs(np.log((ph + 1e-8) / (mh + 1e-8)))
            cst += (1.0 - i) + 0.35 * c_err + 0.35 * s_err
            cnt += 1
        if cnt < max(10, len(rows) // 4):
            return None
        return cst / cnt

    # Stage 1: optimize yaw + scale with fixed translation.
    for yaw in yaw_grid:
        R = make_R(yaw)
        for sr in s_grid:
            for sh in s_grid:
                half = half0 * np.array([sr, sr, sh], dtype=np.float64)
                key = build_keypoints(half)
                cst = eval_cost(R, t0, key)
                if cst is None:
                    continue
                if best is None or cst < best:
                    best = cst
                    best_par = (yaw, sr, sh)

    if best_par is None:
        raise RuntimeError('Mask calibration failed to find a valid solution')

    yaw, sr, sh = best_par
    R = make_R(yaw)
    half = half0 * np.array([sr, sr, sh], dtype=np.float64)
    key = build_keypoints(half)

    # Stage 2: refine translation in object local frame around current center.
    # Search translation offsets along object axes and apply in world coordinates.
    d = np.linspace(-args.trans_range, args.trans_range, args.trans_steps)
    best_t = t0.copy()
    best_t_cost = eval_cost(R, best_t, key)
    for dx in d:
        for dy in d:
            for dz in d:
                off_w = R @ np.array([dx, dy, dz], dtype=np.float64)
                tw = t0 + off_w
                cst = eval_cost(R, tw, key)
                if cst is None:
                    continue
                if best_t_cost is None or cst < best_t_cost:
                    best_t_cost = cst
                    best_t = tw

    t_final = best_t
    key = build_keypoints(half)

    obj['rotation_world_from_object'] = R.tolist()
    obj['translation_world_from_object'] = t_final.tolist()
    obj['scale_half'] = half.tolist()
    obj['scale_full'] = (2.0 * half).tolist()
    obj['calibrated_from_sam2_masks'] = {
        'yaw_deg': float(yaw),
        'scale_radius': float(sr),
        'scale_height': float(sh),
        'score': float(best_t_cost if best_t_cost is not None else best),
        'num_frames': int(len(rows)),
        'translation_offset_world': (t_final - t0).tolist(),
    }

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / 'object_frame.json').write_text(json.dumps(obj, indent=2))
    np.save(out / 'cuboid_3d_keypoints.npy', key)
    (out / 'cuboid_3d_keypoints.json').write_text(json.dumps({'keypoints_object': key.tolist()}, indent=2))

    print('best', obj['calibrated_from_sam2_masks'])


if __name__ == '__main__':
    main()
