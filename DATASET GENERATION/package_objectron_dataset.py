import argparse
import json
import random
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True, type=Path)
    ap.add_argument("--annotations_dir", required=True, type=Path)
    ap.add_argument("--metadata_dir", required=True, type=Path)
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--copy_images", action="store_true")
    args = ap.parse_args()

    if args.train_ratio + args.val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1.0")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "images").mkdir(parents=True, exist_ok=True)
    (args.out_dir / "annotations").mkdir(parents=True, exist_ok=True)
    (args.out_dir / "metadata").mkdir(parents=True, exist_ok=True)
    (args.out_dir / "splits").mkdir(parents=True, exist_ok=True)

    # Copy metadata files if they exist
    if args.metadata_dir.exists():
        for p in args.metadata_dir.iterdir():
            if p.is_file() and p.suffix in (".json", ".npy"):
                shutil.copy2(p, args.out_dir / "metadata" / p.name)

    # Collect annotations
    ann_files = sorted(args.annotations_dir.glob("*.json"))
    if not ann_files:
        raise ValueError("No annotation JSONs found")

    # Shuffle and split
    rng = random.Random(args.seed)
    rng.shuffle(ann_files)

    n = len(ann_files)
    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)

    train_files = ann_files[:n_train]
    val_files = ann_files[n_train:n_train + n_val]
    test_files = ann_files[n_train + n_val:]

    def write_split(name, files):
        with (args.out_dir / "splits" / f"{name}.txt").open("w") as f:
            for p in files:
                f.write(p.stem + "\n")

    write_split("train", train_files)
    write_split("val", val_files)
    write_split("test", test_files)

    # Copy annotations
    for p in ann_files:
        shutil.copy2(p, args.out_dir / "annotations" / p.name)

    # Optionally copy images
    if args.copy_images:
        for p in ann_files:
            ann = json.loads(p.read_text())
            img_name = ann["image"]
            src = args.images_dir / img_name
            if src.exists():
                shutil.copy2(src, args.out_dir / "images" / img_name)

    summary = {
        "total": n,
        "train": len(train_files),
        "val": len(val_files),
        "test": len(test_files),
        "copy_images": args.copy_images,
    }

    with (args.out_dir / "metadata" / "split_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
