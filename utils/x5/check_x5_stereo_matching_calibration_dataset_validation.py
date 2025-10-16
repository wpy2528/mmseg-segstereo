import os
import glob
import numpy as np
import cv2
import argparse
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("src_dir", type=str, help="源图像目录路径")
    args = parser.parse_args()
    src_dir = args.src_dir
    assert os.path.isdir(src_dir), f"源图像目录路径不存在: {src_dir}"
    src_left_paths = glob.glob(os.path.join(src_dir, "infra1", "*.npy"), recursive=True)
    assert len(src_left_paths) > 0, f"左目图像不存在: {src_dir}"
    for src_left_path in tqdm(src_left_paths):
        src_right_path = src_left_path.replace("infra1", "infra2")
        assert os.path.exists(src_right_path), f"右目图像不存在: {src_right_path}"
        assert src_left_path.endswith(".npy") and src_right_path.endswith(".npy"), f"左目图像或右目图像不是npy文件: {src_left_path} 或 {src_right_path}"
    print(f"标定数据集验证完成，共 {len(src_left_paths)} 对图像")

    # src_left_paths = glob.glob(os.path.join(src_dir, "**", "*left*.*"), recursive=True)
    # for src_left_path in tqdm(src_left_paths):
    #     src_right_path = src_left_path.replace("left", "right")
    #     assert os.path.exists(src_right_path), f"右目图像不存在: {src_right_path}"
    #     assert src_left_path.endswith(".npy") and src_right_path.endswith(".npy"), f"左目图像或右目图像不是npy文件: {src_left_path} 或 {src_right_path}"
    # print(f"标定数据集验证完成，共 {len(src_left_paths)} 对图像")