import argparse
from pathlib import Path
import numpy as np
import cv2
import open3d as o3d
import re

###############################################
# Quaternion → rotation
###############################################
def qvec2rotmat(q):
    qw,qx,qy,qz=q
    return np.array([
        [1-2*qy*qy-2*qz*qz,2*qx*qy-2*qz*qw,2*qx*qz+2*qy*qw],
        [2*qx*qy+2*qz*qw,1-2*qx*qx-2*qz*qz,2*qy*qz-2*qx*qw],
        [2*qx*qz-2*qy*qw,2*qy*qz+2*qx*qw,1-2*qx*qx-2*qy*qy]
    ])

###############################################
# Read COLMAP cameras
###############################################
def read_cameras(path):

    cams={}
    for line in open(path):

        if line.startswith("#") or not line.strip():
            continue

        p=line.split()

        cam_id=int(p[0])
        model=p[1]
        width=int(p[2])
        height=int(p[3])

        params=list(map(float,p[4:]))

        dist=np.zeros(4,dtype=np.float64)

        if model=="PINHOLE":
            fx,fy,cx,cy=params[:4]
        elif model=="SIMPLE_PINHOLE":
            f,cx,cy=params[:3]
            fx=fy=f
        elif model=="SIMPLE_RADIAL":
            f,cx,cy,k1=params[:4]
            fx=fy=f
            dist=np.array([k1,0.0,0.0,0.0],dtype=np.float64)
        elif model=="RADIAL":
            f,cx,cy,k1,k2=params[:5]
            fx=fy=f
            dist=np.array([k1,k2,0.0,0.0],dtype=np.float64)
        elif model in ("OPENCV","FULL_OPENCV"):
            fx,fy,cx,cy=params[:4]
            if len(params)>=8:
                dist=np.array([params[4],params[5],params[6],params[7]],dtype=np.float64)
        elif model=="OPENCV_FISHEYE":
            fx,fy,cx,cy=params[:4]
            if len(params)>=8:
                # Stored separately and handled in project().
                dist=np.array([params[4],params[5],params[6],params[7]],dtype=np.float64)
        else:
            raise RuntimeError(f"Unsupported camera model: {model}")

        cams[cam_id]=(model,width,height,fx,fy,cx,cy,dist)

    return cams


###############################################
# Read COLMAP poses
###############################################
def read_images(path):

    poses=[]

    lines=open(path).read().splitlines()
    i=0

    while i<len(lines):

        line=lines[i]

        if line.startswith("#") or not line.strip():
            i+=1
            continue

        p=line.split()

        image_id=int(p[0])
        q=np.array(list(map(float,p[1:5])))
        t=np.array(list(map(float,p[5:8])))
        cam_id=int(p[8])
        name=p[9]

        poses.append((image_id,q,t,cam_id,name))

        i+=2

    return poses


def frame_number(name):
    m=re.search(r"(\d+)",Path(name).stem)
    return int(m.group(1)) if m else 10**9


###############################################
# Projection
###############################################
def project(points,cam,row):

    model,width,height,fx,fy,cx,cy,dist=cam
    _,q,t,_,_=row

    R=qvec2rotmat(q)

    rvec,_=cv2.Rodrigues(R)

    K=np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])

    if model=="OPENCV_FISHEYE":
        uv,_=cv2.fisheye.projectPoints(
            points.reshape(-1,1,3).astype(np.float64),
            rvec,
            t.reshape(3,1).astype(np.float64),
            K,
            dist.reshape(4,1),
        )
    else:
        uv,_=cv2.projectPoints(
            points.reshape(-1,1,3).astype(np.float64),
            rvec,
            t.reshape(3,1).astype(np.float64),
            K,
            dist,
        )

    return uv.reshape(-1,2)


###############################################
# Stable cuboid for cylindrical objects
###############################################
def compute_cylinder_cuboid(points):

    center=points.mean(axis=0)

    pts=points-center

    cov=np.cov(pts.T)

    eigvals,eigvecs=np.linalg.eig(cov)

    axis=eigvecs[:,np.argmax(eigvals)]
    axis=axis/np.linalg.norm(axis)

    h=pts@axis

    hmin=h.min()
    hmax=h.max()

    radial=pts-np.outer(h,axis)

    r=np.linalg.norm(radial,axis=1)
    radius=np.percentile(r,95)

    world_up=np.array([0,0,1])

    if abs(axis@world_up)>0.9:
        world_up=np.array([1,0,0])

    x=np.cross(axis,world_up)
    x=x/np.linalg.norm(x)

    y=np.cross(axis,x)

    corners=[]

    for sx in [-1,1]:
        for sy in [-1,1]:

            b=center+axis*hmin+sx*radius*x+sy*radius*y
            t=center+axis*hmax+sx*radius*x+sy*radius*y

            corners.append(b)
            corners.append(t)

    corners=np.array(corners)

    cuboid_center=center

    return corners,cuboid_center


###############################################
# Draw cuboid
###############################################
def draw_cuboid(img,pts):

    pts=pts.astype(int)

    edges=[
        (0,2),(2,6),(6,4),(4,0),
        (1,3),(3,7),(7,5),(5,1),
        (0,1),(2,3),(4,5),(6,7)
    ]

    for a,b in edges:
        cv2.line(img,tuple(pts[a]),tuple(pts[b]),(0,255,0),2)

    return img


###############################################
# Main
###############################################
def main():

    ap=argparse.ArgumentParser()

    ap.add_argument("--point_cloud",required=True)
    ap.add_argument("--cameras_txt",required=True)
    ap.add_argument("--images_txt",required=True)
    ap.add_argument("--image_folder",required=True)
    ap.add_argument("--output_dir",required=True)
    ap.add_argument("--stride",type=int,default=50)

    args=ap.parse_args()

    out=Path(args.output_dir)
    out.mkdir(parents=True,exist_ok=True)

    cams=read_cameras(args.cameras_txt)
    poses=read_images(args.images_txt)
    poses=sorted(poses,key=lambda r: frame_number(r[4]))

    pcd=o3d.io.read_point_cloud(args.point_cloud)

    pts=np.asarray(pcd.points)

    if pcd.has_colors():
        colors=(np.asarray(pcd.colors)[:,::-1]*255).astype(np.uint8)
    else:
        colors=np.full((pts.shape[0],3),220,dtype=np.uint8)

    corners,center=compute_cylinder_cuboid(pts)

    cuboid_points=np.vstack([corners,center])

    for i in range(0,len(poses),args.stride):

        row=poses[i]

        img_path=Path(args.image_folder)/row[4]
        img=cv2.imread(str(img_path))
        if img is None:
            print("skip_missing_image",img_path)
            continue

        cam=cams[row[3]]

        uv=project(cuboid_points,cam,row)

        cuboid_uv=uv[:8]
        center_uv=uv[8]

        img=draw_cuboid(img,cuboid_uv)

        cv2.circle(img,tuple(center_uv.astype(int)),5,(0,0,255),-1)

        cv2.imwrite(str(out/f"cuboid_{i:04d}_{Path(row[4]).stem}.png"),img)

        print("saved",i,row[4])


if __name__=="__main__":
    main()
