import os
import random
import shutil
import time
import cv2
import numpy as np
import glob
import argparse
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate calibration data from source images")
    parser.add_argument("src_dir", help="源图像目录路径")
    parser.add_argument("export_type", choices=["rgb", "gray"], help="导出类型: rgb 或 gray")
    parser.add_argument("h", type=int, help="输出图像高度")
    parser.add_argument("w", type=int, help="输出图像宽度")
    args = parser.parse_args()
    args.dst_dir = os.path.join(args.src_dir.rstrip("/") + f"_{args.export_type}_h{args.h}_w{args.w}_f32")
    print(f"Exporting {args.export_type} images to {args.dst_dir}")
    
    src_image_paths = glob.glob(os.path.join(args.src_dir, "**", "*.*"), recursive=True)
    print(f"Found {len(src_image_paths)} images in {args.src_dir}")

    for src_image_path in tqdm(src_image_paths):
        src_image_name = os.path.splitext(os.path.basename(src_image_path))[0]
        if args.export_type == "rgb":
            dst_image_path = os.path.join(args.dst_dir, src_image_name + ".rgb")
        else:
            dst_image_path = os.path.join(args.dst_dir, src_image_name + ".gray")
        os.makedirs(os.path.dirname(dst_image_path), exist_ok=True)
        src_image_np = cv2.imread(src_image_path)
        src_image_np = cv2.resize(src_image_np, (args.w, args.h))
        if args.export_type == "rgb":
            vis_image_np = cv2.cvtColor(src_image_np, cv2.COLOR_BGR2RGB)
        else:
            vis_image_np = cv2.cvtColor(src_image_np, cv2.COLOR_BGR2GRAY)
        dtype = np.float32
        vis_image_np.astype(dtype).tofile(dst_image_path)
    print(f"Exported {len(src_image_paths)} images to {args.dst_dir}")