import argparse
import glob
import os
import cv2
import numpy as np

category_map = {
    "background": 0,
    "occu": 1,
    "glare": 2,
    "pure_background": 3
}

DIALATE_WIDTH = 7

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="遍历目录中的所有png文件")
    parser.add_argument("src_image_dir", type=str, help="待遍历的目录")
    parser.add_argument("ambiguous_category", type=int, help="模糊类别")
    args = parser.parse_args()

    src_image_dir = args.src_image_dir
    png_path_list = glob.glob(os.path.join(src_image_dir, "**", "*.png"), recursive=True)
    print(f"共找到{len(png_path_list)}个png文件")
    for src_label_path in png_path_list:
        src_image_path = src_label_path.replace(".png", ".jpg").replace("labels", "images")
        src_image_np = cv2.imread(src_image_path)
        src_mask_np = cv2.imread(src_label_path.replace(".png", ".png"), cv2.IMREAD_UNCHANGED)
        if len(src_mask_np.shape) == 3:
            src_mask_np = src_mask_np[..., 0]
        # 找到类别为ambiguous_category的区域
        ambiguous_mask_np = (src_mask_np == args.ambiguous_category).astype("uint8")
        # 获取边缘
        contours, _ = cv2.findContours(ambiguous_mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        edge_mask_np = np.zeros_like(src_mask_np, dtype="uint8")
        cv2.drawContours(edge_mask_np, contours, -1, 1, thickness=1)
        # 图像边缘保持不变
        edge_mask_np[:5, :] = 0
        edge_mask_np[-5:, :] = 0
        edge_mask_np[:, :5] = 0
        edge_mask_np[:, -5:] = 0
        # 对边缘进行膨胀
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (DIALATE_WIDTH, DIALATE_WIDTH))
        dilated_edge_mask_np = cv2.dilate(edge_mask_np, kernel, iterations=1)
        # 将膨胀后的区域置为255
        src_mask_np[dilated_edge_mask_np > 0] = 255
        vis_np = src_mask_np.copy() * 50
        vis_np = cv2.cvtColor(vis_np, cv2.COLOR_GRAY2BGR)
        vis_np = np.concatenate([src_image_np, vis_np], axis=1)
        # 保存修改后的mask
        vis_save_path = os.path.join("pgs", "vis_255", os.path.basename(src_label_path))
        os.makedirs(os.path.dirname(vis_save_path), exist_ok=True)
        cv2.imwrite(vis_save_path, vis_np)
        print(f"保存到 {vis_save_path}")
