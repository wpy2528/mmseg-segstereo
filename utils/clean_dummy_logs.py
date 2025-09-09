import os
import json
import numpy as np
import cv2
import glob
import requests
import argparse
from tqdm import tqdm
import shutil


if __name__ == "__main__":
    root_log_dir = "work_dirs"
    to_be_deleted_log_dirs = []
    for src_log_path in glob.glob(os.path.join(root_log_dir, "**", "*.log"), recursive=True):
        folder = src_log_path.split('/')[-2]
        if folder != os.path.basename(src_log_path)[:-4]:
            print(f"Error: {src_log_path} is not in the correct folder")
            continue
        src_log_dir = os.path.dirname(src_log_path)
        with open(src_log_path, "r") as f:
            content = f.read()
        if content.count("mmengine - INFO - Epoch(train)") < 10:
            print(f"{src_log_dir} 中基本不包含训练内容，视为无效log目录")
            to_be_deleted_log_dirs.append(src_log_dir)
    print()
    print()
    print()
    [print(log_dir) for log_dir in to_be_deleted_log_dirs]
    print(f"将删除 {len(to_be_deleted_log_dirs)} 个无效log目录")
    res = input("是否继续？(y/n)")
    if res != "y":
        exit()
    
    for log_dir in tqdm(to_be_deleted_log_dirs):
        shutil.rmtree(log_dir)
