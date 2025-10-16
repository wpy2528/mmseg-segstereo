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
    parser = argparse.ArgumentParser()
    parser.add_argument("src_dir", type=str, help="源图像目录路径")
    parser.add_argument("dst_dir", type=str, help="导出数据集所在文件夹")
    args = parser.parse_args()
    
    h, w = 352, 640
    print("注意：图像大小必须为352x640")
    src_left_paths = glob.glob(os.path.join(args.src_dir, "**", "*left*.*"), recursive=True)
    for i, src_left_path in tqdm(enumerate(src_left_paths)):
        src_right_path = src_left_path.replace("left", "right")
        src_left_np = cv2.imread(src_left_path)
        src_right_np = cv2.imread(src_right_path)
        
        src_left_np = cv2.resize(src_left_np, (w, h))
        src_right_np = cv2.resize(src_right_np, (w, h))
        # left_cropped = batch['left_img_yuv'][0].transpose(2, 0, 1)
        left_cropped = src_left_np.transpose(2, 0, 1) # batch['left_img'][0] 正常imread出来的图像
        left_cropped = np.ascontiguousarray(left_cropped)
        # right_cropped = batch['right_img_yuv'][0].transpose(2, 0, 1)
        right_cropped = src_right_np.transpose(2, 0, 1)
        right_cropped = np.ascontiguousarray(right_cropped)
        print(left_cropped.shape, right_cropped.shape)
        os.makedirs(os.path.join(args.dst_dir, "infra1"), exist_ok=True)
        os.makedirs(os.path.join(args.dst_dir, "infra2"), exist_ok=True)
        left_cropped.tofile(os.path.join(args.dst_dir, "infra1", "%d.npy" % i))
        right_cropped.tofile(os.path.join(args.dst_dir, "infra2", "%d.npy" % i))