import cv2
import numpy as np
import argparse
import glob
import os
from tqdm import tqdm

'''
生成一个gt全0的负样本数据集
'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("src_image_dir")
    parser.add_argument("dst_dataset_dir")
    args = parser.parse_args()
    src_drc_3x3_paths = glob.glob(os.path.join(args.src_image_dir, "**", "*.png"), recursive=True)
    for src_drc_3x3_path in tqdm(src_drc_3x3_paths):
        src_image_name = os.path.basename(src_drc_3x3_path)
        src_image_np = cv2.imread(src_drc_3x3_path)
        h, w = src_image_np.shape[:2]
        assert h == 816 and w == 960, f"src_image_np.shape: {src_image_np.shape}"
        fl_image_np = src_image_np[0:h//3, 0:w//3, :]
        fr_image_np = src_image_np[0:h//3, w//3:2*w//3, :]
        src_mask_np = np.zeros(fl_image_np.shape[:2], dtype=np.uint8)

        dst_fl_path = os.path.join(args.dst_dataset_dir, "images", src_image_name.replace(".png", "_fl.jpg"))
        dst_fr_path = os.path.join(args.dst_dataset_dir, "images", src_image_name.replace(".png", "_fr.jpg"))
        dst_fl_mask_path = os.path.join(args.dst_dataset_dir, "labels", src_image_name.replace(".png", "_fl.png"))
        dst_fr_mask_path = os.path.join(args.dst_dataset_dir, "labels", src_image_name.replace(".png", "_fr.png"))
        os.makedirs(os.path.dirname(dst_fl_path), exist_ok=True)
        os.makedirs(os.path.dirname(dst_fr_path), exist_ok=True)
        os.makedirs(os.path.dirname(dst_fl_mask_path), exist_ok=True)
        os.makedirs(os.path.dirname(dst_fr_mask_path), exist_ok=True)
        cv2.imwrite(dst_fl_path, fl_image_np)
        cv2.imwrite(dst_fr_path, fr_image_np)
        cv2.imwrite(dst_fl_mask_path, src_mask_np)
        cv2.imwrite(dst_fr_mask_path, src_mask_np)

        
