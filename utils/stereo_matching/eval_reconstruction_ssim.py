import cv2
import time
import numpy as np
from skimage.metrics import structural_similarity as ssim
if __name__ == "__main__":
    from visualize_disp import pfm_imread, visualize_pfm
    from visualize_x5_stereo import get_right_and_disp_path_by_left_path
else:
    from utils.visualize_disp import pfm_imread, visualize_pfm
    from utils.visualize_x5_stereo import get_right_and_disp_path_by_left_path
import argparse
import glob
import os

def load_data(left_path, right_path, disp_path):
    left_img = cv2.imread(left_path)
    right_img = cv2.imread(right_path)
    disp = pfm_imread(disp_path)  # shape: [H, W]
    if len(disp.shape) == 3:
        disp = disp.squeeze()
    return left_img, right_img, disp

# def warp_left_to_right_remap(left_img, disp):
#     h, w = disp.shape

#     # 构造右图坐标网格（目标图）
#     grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h), indexing='xy')

#     # 对于右图每个像素，想知道它在左图中该采样哪个位置：
#     # 反向查找：x_left = x_right + disp_right(x, y)
#     map_x = (grid_x + disp).astype(np.float32)
#     map_y = grid_y.astype(np.float32)

#     # 执行 remap（从左图采样，生成右图）
#     remapped = cv2.remap(left_img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

#     cv2.imwrite("pgs/kk_remap_corrected.png", remapped)

#     return remapped

def warp_left_to_right(left_img, disp):
    h, w = disp.shape
    start_time = time.time()
    # 构建网格坐标
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))

    # 计算目标坐标
    dst_x = (grid_x - disp).astype(np.int32)
    dst_y = grid_y

    # 创建一个空图像
    remapped = np.zeros_like(left_img)

    # 合法坐标掩码
    valid_mask = (dst_x >= 0) & (dst_x < w) & (dst_y >= 0) & (dst_y < h)

    # 使用掩码过滤合法坐标
    src_y = grid_y[valid_mask]
    src_x = grid_x[valid_mask]
    dst_y = dst_y[valid_mask]
    dst_x = dst_x[valid_mask]

    remapped[dst_y, dst_x] = left_img[src_y, src_x]
    end_time = time.time()
    # print(f"warp_left_to_right time: {end_time - start_time} seconds")
    return remapped


def compute_metrics(recons_img, right_img):
    # 转为灰度图计算 SSIM
    gray1 = cv2.cvtColor(recons_img, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
    ssim_val = ssim(gray1, gray2, data_range=255)

    # 计算像素级误差（L1）
    abs_diff = np.abs(gray1.astype(np.float32) - gray2.astype(np.float32))
    mae = np.mean(abs_diff)
    return ssim_val, mae


def calculate_consistent_metrics(left_img, right_img, disp):
    assert left_img.shape == right_img.shape
    assert left_img.shape[:2] == disp.shape
    recons_img = warp_left_to_right(left_img, disp)
    ssim_val, mae = compute_metrics(recons_img, right_img)
    return ssim_val, mae, recons_img

    
def visualize_reconstruct_images(left_img, right_img, recons_img, disp_np):
    if isinstance(disp_np, str):
        disp_np = visualize_pfm(disp_np)
    vis_np = cv2.hconcat([left_img, right_img, recons_img, disp_np])
    return vis_np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir', help='数据集目录，如 pgs/x5_stereo')
    args = parser.parse_args()

    src_left_paths = sorted(glob.glob(os.path.join(args.dataset_dir, 'left*.png')))
    ssim_list = []
    mae_list = []
    for left_path in src_left_paths:
        print("======================")
        right_path, disp_path, idx = get_right_and_disp_path_by_left_path(left_path)
        print(left_path, right_path, disp_path)
        if not (os.path.exists(right_path) and os.path.exists(disp_path)):
            print(f"跳过 {left_path}，缺少对应文件")
            continue
        left_img, right_img, disp = load_data(left_path, right_path, disp_path)
        ssim_val, mae, recons_img = calculate_consistent_metrics(left_img, right_img, disp)
        ssim_list.append(ssim_val)
        mae_list.append(mae)
        print(f"{idx}: SSIM={ssim_val:.4f}, MAE={mae:.2f}")
        
        vis_np = visualize_reconstruct_images(left_img, right_img, recons_img, disp_path)
        vis_save_path = os.path.join("pgs/vis_x5", f'recons_right{idx}.png')
        os.makedirs(os.path.dirname(vis_save_path), exist_ok=True)
        cv2.imwrite(vis_save_path, vis_np)
        print(f"Saved reconstructed right image as {vis_save_path}")
        
        # 可选：保存重建图像
        # cv2.imwrite(os.path.join(args.dataset_dir, f'recons_right{idx}.png'), recons_img)
    if ssim_list:
        print(f"\n全部样本统计：")
        print(f"平均SSIM: {np.mean(ssim_list):.4f}")
        print(f"平均MAE: {np.mean(mae_list):.2f}")
    else:
        print("未找到有效样本！")

if __name__ == "__main__":
    main()
