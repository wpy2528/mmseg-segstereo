import numpy as np
from tqdm import tqdm
import os
import argparse
import glob
import cv2


import cv2
import numpy as np

def concat_front_and_side_images(front_image, side_image, dst_size):
    # 确保输出图像初始化好
    dst_image = np.zeros((dst_size[1], dst_size[0], front_image.shape[2]), dtype=front_image.dtype)

    upper_height_proportion = 1.0 / 3 + 1.0 / 10

    src_upper_tly = round(side_image.shape[0] * (1.0 - upper_height_proportion))
    src_upper_height = round(side_image.shape[0] * upper_height_proportion)

    src_lower_tly = round(front_image.shape[0] * 1.0 / 3)
    src_lower_height = round(front_image.shape[0] * (1.0 - upper_height_proportion))

    # 从 side_image 获取下部分
    src_upper_roi = side_image[src_upper_tly : src_upper_tly + src_upper_height, :, :]

    # 从 front_image 获取上部分
    src_lower_roi = front_image[src_lower_tly : src_lower_tly + src_lower_height, :, :]

    dst_upper_height = round(dst_size[1] * upper_height_proportion)
    dst_lower_height = dst_size[1] - dst_upper_height

    dst_upper_roi = cv2.resize(src_upper_roi, (dst_size[0], dst_upper_height))
    dst_lower_roi = cv2.resize(src_lower_roi, (dst_size[0], dst_lower_height))

    # 将两部分填入输出图像
    dst_image[0 : dst_upper_height, :, :] = dst_upper_roi
    dst_image[dst_upper_height : dst_size[1], :, :] = dst_lower_roi

    return dst_image


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
        if H == 272 and W == 2880:
            # 水平分成9份
            sub_width = W // 9
            for i in range(9):
                dst_image_np = src_image_np[:, i*sub_width:(i+1)*sub_width]
                sub_image_nps.append(dst_image_np)
            fl, fr, vis_seg, sl, sr, ai_map, depth, depth_map, prob_map = sub_image_nps
        elif H == 816 and W == 960:
            # 横竖分成3*3
            h, w = H//3, W//3
            for i in range(3):
                for j in range(3):
                    dst_image_np = src_image_np[i*h:(i+1)*h, j*w:(j+1)*w]
                    sub_image_nps.append(dst_image_np)
            fl, fr, vis_seg, sl, sr, ai_map, depth, depth_map, prob_map = sub_image_nps
        else:
            print(f"src_image_path: {src_image_path} 的尺寸不符合要求, {H}x{W}")
            continue
        
        concat_image = concat_front_and_side_images(fl, sr, (320, 320))
        
        
        cv2.imwrite(os.path.join(args.dst_image_dir, os.path.basename(src_image_path).replace(".png", "_concat.png")), concat_image)
        
