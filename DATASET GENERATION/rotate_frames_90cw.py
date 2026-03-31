import argparse
from pathlib import Path

import cv2
from tqdm import tqdm


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, type=Path)
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--ext", default="png", help="Input extension (png/jpg)")
    ap.add_argument("--quality", type=int, default=95)
    ap.add_argument("--no_rotate", action="store_true", help="Do not rotate, just convert/rename")
    return ap.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    frames = sorted(args.in_dir.glob(f"*.{args.ext}"))
    if not frames:
        raise RuntimeError(f"No *.{args.ext} files in {args.in_dir}")

    for p in tqdm(frames, desc="rotate"):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue
        rotated = img if args.no_rotate else cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        # Use frame_XXXXXX.jpg naming.
        stem = p.stem
        out_name = f"frame_{stem}.jpg"
        out_path = args.out_dir / out_name
        cv2.imwrite(str(out_path), rotated, [int(cv2.IMWRITE_JPEG_QUALITY), args.quality])

    action = "Converted" if args.no_rotate else "Rotated"
    print(f"{action} {len(frames)} frames -> {args.out_dir}")


if __name__ == "__main__":
    main()
