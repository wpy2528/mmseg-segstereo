import numpy as np
from tqdm import tqdm
import os
import argparse
import glob
import cv2

index_name_map = [["front_left", "front_right", "seg"],
                   ["left", "right", "seg_reconstructed"],
                   ["point_cloud", "point_cloud_top_down", "point_cloud_seg_top_down"]]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("src_image_dir")
    parser.add_argument("dst_image_dir")
    parser.add_argument("--only_fl", action="store_true")

    args = parser.parse_args()
    os.makedirs(args.dst_image_dir, exist_ok=True)
    
    src_image_paths = glob.glob(os.path.join(args.src_image_dir, "**", "*.png"), recursive=True)

    for src_image_path in tqdm(src_image_paths):
        src_image_np = cv2.imread(src_image_path)
        sub_image_nps = []
        H, W = src_image_np.shape[:2]
        # 横竖分成3*3
        h, w = H//3, W//3
        for i in range(3):
            for j in range(3):
                dst_image_np = src_image_np[i*h:(i+1)*h, j*w:(j+1)*w]
                sub_image_nps.append(dst_image_np)
        fl, fr, vis_seg, sl, sr, ai_map, depth, depth_map, prob_map = sub_image_nps
        cv2.imwrite(os.path.join(args.dst_image_dir, os.path.basename(src_image_path).replace(".png", "_fl.png")), fl)
        if not args.only_fl:
            cv2.imwrite(os.path.join(args.dst_image_dir, os.path.basename(src_image_path).replace(".png", "_fr.png")), fr)
            cv2.imwrite(os.path.join(args.dst_image_dir, os.path.basename(src_image_path).replace(".png", "_sl.png")), sl)
            cv2.imwrite(os.path.join(args.dst_image_dir, os.path.basename(src_image_path).replace(".png", "_sr.png")), sr)
        
