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
    parser.add_argument("dst_root_dir", help="导出标定数据集所在目录")
    parser.add_argument("export_color_type", choices=["rgb", "gray"], help="导出类型: rgb 或 gray")
    parser.add_argument("h", type=int, help="输出图像高度")
    parser.add_argument("w", type=int, help="输出图像宽度")
    parser.add_argument("--export_data_type", default="f32")
    args = parser.parse_args()
    args.src_dir = args.src_dir.rstrip("/")
    assert args.export_data_type == "f32", "目前只支持f32格式，别急，我正在开发"
    assert os.path.isdir(args.src_dir), f"源图像目录路径不存在: {args.src_dir}"
    assert any(e not in args.dst_root_dir for e in ["rgb", "gray"]), "你搞错了，这个目录是标定数据集文件夹所在的目录，这个程序会自动给你的文件夹加上类别标注"
    assert os.path.exists(args.dst_root_dir), f"导出数据集所在文件夹不存在: {args.dst_root_dir}"
    args.dst_dir = os.path.join(args.dst_root_dir, f"{os.path.basename(args.src_dir)}_h{args.h}_w{args.w}_{args.export_data_type}")
    assert not os.path.exists(args.dst_dir), f"导出数据集已存在，请删除后重新运行: {args.dst_dir}"
    print(f"标定数据集保存在 {args.dst_dir}")
    
    src_image_paths = glob.glob(os.path.join(args.src_dir, "**", "*.*"), recursive=True)
    print(f"Found {len(src_image_paths)} images in {args.src_dir}")
    assert len(src_image_paths) > 0, f"源图像目录路径不存在: {args.src_dir}"

    for src_image_path in tqdm(src_image_paths):
        src_image_name = os.path.splitext(os.path.basename(src_image_path))[0]
        dst_image_path = os.path.join(args.dst_dir, src_image_name + f".{args.export_color_type}")
        os.makedirs(os.path.dirname(dst_image_path), exist_ok=True)
        src_image_np = cv2.imread(src_image_path)
        src_image_np = cv2.resize(src_image_np, (args.w, args.h))
        if args.export_color_type == "rgb":
            vis_image_np = cv2.cvtColor(src_image_np, cv2.COLOR_BGR2RGB)
        else:
            vis_image_np = cv2.cvtColor(src_image_np, cv2.COLOR_BGR2GRAY)
        dtype = np.float32
        vis_image_np.astype(dtype).tofile(dst_image_path)
    print(f"Exported {len(src_image_paths)} images to {args.dst_dir}")