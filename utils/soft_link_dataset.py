import argparse
import glob
import os
from pathlib import Path

def link_files(src_image_dir, dst_dataset_dir):
    src_image_dir = Path(src_image_dir)
    dst_dataset_dir = Path(dst_dataset_dir)
    source_is_txt = src_image_dir.suffix == ".txt"

    if source_is_txt:
        with src_image_dir.open('r') as f:
            src_image_paths = [l.strip() for l in f.readlines()]
    else:
        src_image_paths = glob.glob(os.path.join(src_image_dir, "**", "images", "*.jpg"), recursive=True)

    for src_image_path in src_image_paths:
        if "/train/" in src_image_path:
            rel_path = src_image_path.split("/train/")[1]
        elif "/val/" in src_image_path:
            rel_path = src_image_path.split("/val/")[1]
        else:
            raise ValueError(f"路径 {src_image_path} 中没有包含train、val")

        src_image_path = Path(src_image_path)
        if not src_image_path.exists():
            print(f"[WARN] Source file does not exist: {src_image_path}")
            continue

        dst_image_path = dst_dataset_dir / rel_path
        dst_image_path.parent.mkdir(parents=True, exist_ok=True)

        if dst_image_path.exists():
            if dst_image_path.is_symlink() and os.readlink(dst_image_path) == str(src_image_path):
                print(f"[SKIP] Destination already exists and is the correct symlink: {dst_image_path}")
                continue  # already linked correctly
            else:
                print(f"[SKIP] Destination already exists and is not the correct symlink: {dst_image_path}")
                continue
        dst_image_path.symlink_to(src_image_path)
        # print(f"[OK] Linked: {dst_image_path} -> {src_image_path}")
        src_label_path = str(src_image_path).replace("/images/", "/labels/").replace(".jpg", ".png")
        dst_label_path = str(dst_image_path).replace("/images/", "/labels/").replace(".jpg", ".png")
        src_label_path = Path(src_label_path)
        dst_label_path = Path(dst_label_path)
        dst_label_path.parent.mkdir(parents=True, exist_ok=True)
        dst_label_path.symlink_to(src_label_path)
        # print(f"[OK] Linked: {dst_label_path} -> {src_label_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Link listed files to target directory while preserving structure.")
    parser.add_argument("src_image_dir", help="Path to text file with relative paths (one per line).")
    parser.add_argument("dst_dataset_dir", help="Destination root directory to place symlinks.")
    args = parser.parse_args()

    link_files(args.src_image_dir, args.dst_dataset_dir)
