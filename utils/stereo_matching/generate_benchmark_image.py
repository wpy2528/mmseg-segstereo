import numpy as np
import torch
import shutil
import os
import glob
import cv2
import os

def generate_rgb_image():
    # 生成h544w960的RGB图像，三个通道分别为0, 50, 80
    height, width = 544, 960
    vis_image_np = np.zeros((height, width, 3), dtype=np.uint8)
    vis_image_np[..., 0] = 20    # R通道
    vis_image_np[..., 1] = 60   # G通道
    vis_image_np[..., 2] = 80   # B通道

    # 保存图像到cursor_utils目录
    cv2.imwrite("rdk_model_zoo/resource/stereo_benchmark/rgb_544_960_left.png", vis_image_np)

    # 生成h544w960的RGB图像，三个通道分别为0, 50, 80
    height, width = 544, 960
    vis_image_np = np.zeros((height, width, 3), dtype=np.uint8)
    vis_image_np[..., 0] = 50    # R通道
    vis_image_np[..., 1] = 80   # G通道
    vis_image_np[..., 2] = 10   # B通道

    # 保存图像到cursor_utils目录
    cv2.imwrite("rdk_model_zoo/resource/stereo_benchmark/rgb_544_960_right.png", vis_image_np)

def copy_sceneflow_to_left_right_fomat():
    dst_stereo_dir = "rdk_model_zoo/resource/stereo_benchmark"
    src_sceneflow_dir = "sceneflow/SceneFlow_flyingthings3d/val/images"
    src_left_paths = sorted(glob.glob(os.path.join(src_sceneflow_dir, "left", "*.png")))[:10]
    for src_left_path in src_left_paths:
        src_right_path = src_left_path.replace("left", "right")
        shutil.copy(src_left_path, os.path.join(dst_stereo_dir, os.path.basename(src_left_path).replace(".png", "_left.png")))
        shutil.copy(src_right_path, os.path.join(dst_stereo_dir, os.path.basename(src_right_path).replace(".png", "_right.png")))


if __name__ == "__main__":
    generate_rgb_image()