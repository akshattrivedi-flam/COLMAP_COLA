import argparse
import json
from pathlib import Path
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


def load_images_and_bboxes(path):
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
        iid = int(p[0]); qvec = np.array(list(map(float, p[1:5]))); tvec = np.array(list(map(float, p[5:8]))); cid = int(p[8]); name = p[9]
        pts_line = lines[i+1].strip() if i + 1 < len(lines) else ''
        try:
            arr = np.array(list(map(float, pts_line.split())), dtype=np.float64).reshape(-1, 3)
            uv = arr[:, :2]
            bb = np.array([uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max()], dtype=np.float64)
            rows.append(dict(iid=iid, name=name, qvec=qvec, tvec=tvec, cid=cid, bbox=bb))
        except Exception:
            pass
        i += 2
    return rows


def build_keypoints(half):
    w, h, d = half
    return np.array([
        [0.0, 0.0, 0.0],
        [-w, -h, -d], [-w, -h, +d], [-w, +h, -d], [-w, +h, +d],
        [+w, -h, -d], [+w, -h, +d], [+w, +h, -d], [+w, +h, +d],
    ], dtype=np.float64)


def project_bbox(Rwo, two, keyobj, row, cam):
    Rcw = qvec_to_rotmat(row['qvec'])
    tcw = row['tvec']
    Xw = (Rwo @ keyobj.T) + two.reshape(3, 1)
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
    return np.array([u.min(), v.min(), u.max(), v.max()], dtype=np.float64)


def iou(a, b):
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1]); ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    uu = aa + bb - inter
    return inter / uu if uu > 1e-8 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cameras', required=True, type=Path)
    ap.add_argument('--images', required=True, type=Path)
    ap.add_argument('--object_frame', required=True, type=Path)
    ap.add_argument('--out_dir', required=True, type=Path)
    ap.add_argument('--sample', type=int, default=120)
    ap.add_argument('--yaw_step', type=float, default=5.0)
    ap.add_argument('--sr_min', type=float, default=0.08)
    ap.add_argument('--sr_max', type=float, default=0.80)
    ap.add_argument('--sr_steps', type=int, default=19)
    ap.add_argument('--sh_min', type=float, default=0.30)
    ap.add_argument('--sh_max', type=float, default=1.20)
    ap.add_argument('--sh_steps', type=int, default=19)
    args = ap.parse_args()

    cams = load_cameras(args.cameras)
    rows = load_images_and_bboxes(args.images)
    rng = np.random.default_rng(42)
    if len(rows) > args.sample:
        idx = rng.choice(len(rows), size=args.sample, replace=False)
        rows = [rows[i] for i in idx]

    obj = json.loads(args.object_frame.read_text())
    Rwo0 = np.array(obj['rotation_world_from_object'], dtype=np.float64)
    two = np.array(obj['translation_world_from_object'], dtype=np.float64)
    half0 = np.array(obj['scale_half'], dtype=np.float64)

    x0 = Rwo0[:, 0].copy(); y0 = Rwo0[:, 1].copy(); z0 = Rwo0[:, 2].copy()

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
    sr_grid = np.linspace(args.sr_min, args.sr_max, args.sr_steps)
    sh_grid = np.linspace(args.sh_min, args.sh_max, args.sh_steps)

    for yaw in yaw_grid:
        Rwo = make_R(yaw)
        for sr in sr_grid:
            for sh in sh_grid:
                half = half0 * np.array([sr, sr, sh], dtype=np.float64)
                key = build_keypoints(half)
                cost = 0.0
                cnt = 0
                for r in rows:
                    cam = cams[r['cid']]
                    pb = project_bbox(Rwo, two, key, r, cam)
                    if pb is None:
                        continue
                    ob = r['bbox']
                    i = iou(pb, ob)
                    # mix IoU and size ratio stability
                    pw, ph = pb[2]-pb[0], pb[3]-pb[1]
                    ow, oh = ob[2]-ob[0], ob[3]-ob[1]
                    if ow < 1 or oh < 1:
                        continue
                    sw = abs(np.log((pw + 1e-8)/(ow + 1e-8)))
                    shh = abs(np.log((ph + 1e-8)/(oh + 1e-8)))
                    cost += (1.0 - i) + 0.35 * (sw + shh)
                    cnt += 1
                if cnt < 10:
                    continue
                c = cost / cnt
                if best is None or c < best:
                    best = c
                    best_par = (yaw, sr, sh)

    if best_par is None:
        raise RuntimeError('optimization failed')

    yaw, sr, sh = best_par
    Rwo = make_R(yaw)
    half = half0 * np.array([sr, sr, sh], dtype=np.float64)
    key = build_keypoints(half)

    obj['rotation_world_from_object'] = Rwo.tolist()
    obj['scale_half'] = half.tolist()
    obj['scale_full'] = (2.0 * half).tolist()
    obj['calibrated_from_2d'] = dict(yaw_deg=float(yaw), scale_radius=float(sr), scale_height=float(sh), score=float(best))

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'object_frame.json').write_text(json.dumps(obj, indent=2))
    np.save(out / 'cuboid_3d_keypoints.npy', key)
    (out / 'cuboid_3d_keypoints.json').write_text(json.dumps({'keypoints_object': key.tolist()}, indent=2))

    print('best', obj['calibrated_from_2d'])


if __name__ == '__main__':
    main()
