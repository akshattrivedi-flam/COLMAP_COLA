import json, numpy as np, sqlite3
from pathlib import Path
from PIL import Image

root = Path('/home/user/Desktop/COLMAP_COLA')
colmap_txt = root/'video_02_red/colmap_recon_best/txt'
images_txt = colmap_txt/'images.txt'
points_txt = colmap_txt/'points3D.txt'
masks_dir = root/'video_02_red/masks_sam2'

scaled_masked_dir = root/'video_02_red/colmap_recon_best/scaled_masked'
scaled_masked_dir.mkdir(parents=True, exist_ok=True)

mask_cache = {}

def load_mask_for_image(name):
    if name in mask_cache:
        return mask_cache[name]
    mask_path = masks_dir / (Path(name).stem + '.png')
    if not mask_path.exists():
        mask_cache[name] = None
        return None
    mask = Image.open(mask_path).convert('L')
    mask_np = np.array(mask) > 127
    mask_cache[name] = mask_np
    return mask_np

# Parse points3D
points = {}
errors = []
with points_txt.open() as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        pid = int(parts[0])
        xyz = np.array(list(map(float, parts[1:4])), dtype=np.float64)
        error = float(parts[7])
        track = parts[8:]
        track_len = len(track)//2
        points[pid] = (xyz, error, track_len)
        errors.append(error)
errors = np.array(errors)

# Parse images.txt and filter 2D points by mask
image_entries = []
valid_obs = {}

with images_txt.open() as f:
    lines = [ln.rstrip('\n') for ln in f]

idx = 0
while idx < len(lines):
    line = lines[idx].strip()
    if not line or line.startswith('#'):
        idx += 1
        continue
    parts = line.split()
    image_id = int(parts[0])
    qw,qx,qy,qz = map(float, parts[1:5])
    tx,ty,tz = map(float, parts[5:8])
    camera_id = int(parts[8])
    name = parts[9]
    idx += 1
    pts_line = lines[idx].strip() if idx < len(lines) else ''
    pts = []
    if pts_line:
        vals = pts_line.split()
        for i in range(0, len(vals), 3):
            x = float(vals[i]); y = float(vals[i+1]); pid = int(vals[i+2])
            pts.append((x,y,pid))

    mask = load_mask_for_image(name)
    filtered_pts = []
    for x,y,pid in pts:
        if pid < 0:
            continue
        if mask is None:
            continue
        xi, yi = int(round(x)), int(round(y))
        if 0 <= yi < mask.shape[0] and 0 <= xi < mask.shape[1] and mask[yi, xi]:
            filtered_pts.append((x,y,pid))
            if pid not in valid_obs:
                valid_obs[pid] = [1,1]
            else:
                valid_obs[pid][0] += 1
                valid_obs[pid][1] += 1
        else:
            if pid not in valid_obs:
                valid_obs[pid] = [0,1]
            else:
                valid_obs[pid][1] += 1

    image_entries.append({
        'image_id': image_id,
        'camera_id': camera_id,
        'name': name,
        'q': (qw,qx,qy,qz),
        't': (tx,ty,tz),
        'points2D': filtered_pts,
    })
    idx += 1

kept_pids = set()
for pid, (inm, total) in valid_obs.items():
    if total >= 3 and inm / max(total,1) >= 0.6:
        kept_pids.add(pid)

kept_xyz = np.array([points[pid][0] for pid in kept_pids if pid in points])
if kept_xyz.size == 0:
    kept_xyz = np.array([xyz for (xyz, _, _) in points.values()])

mean_xyz = kept_xyz.mean(axis=0)
X = kept_xyz - mean_xyz
cov = np.cov(X.T)
vals, vecs = np.linalg.eigh(cov)
order = np.argsort(vals)[::-1]
vecs = vecs[:, order]
proj = X @ vecs
extent = proj.max(axis=0) - proj.min(axis=0)
recon_height = float(extent[0])
real_height = 0.13
scale = real_height / recon_height if recon_height > 0 else 1.0

with (scaled_masked_dir/'points3D.txt').open('w') as f:
    f.write('# Masked+Scaled points3D.txt (scale={:.6f}, can-only)\n'.format(scale))
    f.write('# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n')
    with points_txt.open() as fin:
        for line in fin:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            pid = int(parts[0])
            if pid not in kept_pids:
                continue
            x,y,z = map(float, parts[1:4])
            x,y,z = x*scale, y*scale, z*scale
            parts[1] = f'{x:.6f}'; parts[2] = f'{y:.6f}'; parts[3] = f'{z:.6f}'
            f.write(' '.join(parts)+'\n')

with (scaled_masked_dir/'images.txt').open('w') as f:
    f.write('# Masked+Scaled images.txt (scale={:.6f})\n'.format(scale))
    f.write('# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n')
    f.write('# POINTS2D[] as (X, Y, POINT3D_ID)\n')
    for e in image_entries:
        qw,qx,qy,qz = e['q']
        tx,ty,tz = e['t']
        tx,ty,tz = tx*scale, ty*scale, tz*scale
        f.write(f"{e['image_id']} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {e['camera_id']} {e['name']}\n")
        pts = [(x,y,pid) for x,y,pid in e['points2D'] if pid in kept_pids]
        if pts:
            flat = ' '.join([f"{x} {y} {pid}" for x,y,pid in pts])
            f.write(flat+'\n')
        else:
            f.write('\n')

with (colmap_txt/'cameras.txt').open() as fin, (scaled_masked_dir/'cameras.txt').open('w') as fout:
    fout.write('# Masked+Scaled cameras.txt (unchanged intrinsics)\n')
    fout.write(fin.read())

DB_PATH = root/'video_02_red/colmap_recon_best/database.db'
conn = sqlite3.connect(str(DB_PATH))
cur = conn.cursor()
cur.execute('SELECT image_id, name FROM images')
name_to_id = {name: image_id for image_id, name in cur.fetchall()}

def fetch_keypoints_and_desc(img_id):
    cur.execute('SELECT data FROM keypoints WHERE image_id=?', (img_id,))
    row = cur.fetchone()
    if row is None:
        return None, None
    kp_data = np.frombuffer(row[0], dtype=np.float32)
    kp = kp_data.reshape(-1, 6)
    cur.execute('SELECT data FROM descriptors WHERE image_id=?', (img_id,))
    row = cur.fetchone()
    if row is None:
        return kp, None
    desc = np.frombuffer(row[0], dtype=np.uint8)
    desc = desc.reshape(-1, 128)
    return kp, desc

acc = {}
count = {}
for e in image_entries:
    img_name = e['name']
    img_id_db = name_to_id.get(img_name)
    if img_id_db is None:
        continue
    kp, desc = fetch_keypoints_and_desc(img_id_db)
    if kp is None or desc is None:
        continue
    pts = e['points2D']
    for idx, (_, _, pid) in enumerate(pts):
        if pid not in kept_pids:
            continue
        if idx >= desc.shape[0]:
            continue
        d = desc[idx].astype(np.float32)
        if pid not in acc:
            acc[pid] = d
            count[pid] = 1
        else:
            acc[pid] += d
            count[pid] += 1

pids = sorted(acc.keys())
xyz = []
desc = []
track = []
for pid in pids:
    xyz.append(points[pid][0]*scale)
    d = acc[pid] / count[pid]
    norm = np.linalg.norm(d)
    if norm > 0:
        d = d / norm
    desc.append(d)
    track.append(points[pid][2])

xyz = np.array(xyz, dtype=np.float32)
desc = np.array(desc, dtype=np.float32)
track = np.array(track, dtype=np.int32)

ref_path = root/'video_02_red/colmap_recon_best/ref_sift_db_masked.npz'
np.savez_compressed(ref_path, point3d_id=np.array(pids, dtype=np.int64), xyz=xyz, desc=desc, track_len=track, scale=scale, recon_height=recon_height, real_height=real_height)

stats = {
    'num_images': len(image_entries),
    'num_points3d_total': len(points),
    'num_points3d_kept': len(kept_pids),
    'scale': scale,
    'recon_height': recon_height,
    'real_height': real_height,
}

stats_path = root/'video_02_red/colmap_recon_best/stats_masked.json'
with stats_path.open('w') as f:
    json.dump(stats, f, indent=2)

print('KEPT 3D points:', len(kept_pids))
print('SCALE', scale, 'recon_height', recon_height)
print('WROTE', scaled_masked_dir)
print('REF_DB', ref_path)
print('STATS_PATH', stats_path)
