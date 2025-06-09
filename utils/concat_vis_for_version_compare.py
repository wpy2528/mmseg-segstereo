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
    stdc_result_paths = glob.glob(os.path.join("pgs/leaf2_3x3", "**", "*.jpg"), recursive=True)
    for stdc_result_path in tqdm(stdc_result_paths):
        # yolov5ds_result_path = stdc_result_path.replace("pgs/leaf2_3x3", "pgs/vis").replace(".jpg", ".png")
        yolov5ds_result_path = os.path.join("pgs/vis", os.path.basename(stdc_result_path)).replace(".jpg", ".png")
        if not os.path.exists(yolov5ds_result_path):
            print(f"yolov5ds_result_path: {yolov5ds_result_path} 不存在")
            continue
        stdc_result_np = cv2.imread(stdc_result_path)
        h, w = stdc_result_np.shape[:2]
        stdc_result_seg_np = stdc_result_np[: h//3, w//3*2:]
        yolov5ds_result_np = cv2.imread(yolov5ds_result_path)
        if yolov5ds_result_np is None:
            continue
        yolov5ds_result_np = yolov5ds_result_np[: h//3, w//3*2:]
        cv2.putText(yolov5ds_result_np, "yolov5ds", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(stdc_result_seg_np, "stdc", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        res = np.hstack([yolov5ds_result_np, stdc_result_seg_np])
        os.makedirs("pgs/vis_cmp_drc2", exist_ok=True)
        cv2.imwrite(os.path.join("pgs/vis_cmp_drc2", os.path.basename(stdc_result_path).replace("_concat.png", "_res.png")), res)
    
    exit(0)
    stdc_result = glob.glob(os.path.join("pgs/vis_cases_c", "*_concat.png"))
    yolov5ds_result = glob.glob(os.path.join("pgs/vis_leaf_concat", "*_concat.png"))

    for stdc_result_path in tqdm(stdc_result):
        stdc_result_name = os.path.basename(stdc_result_path)
        yolov5ds_result_path = os.path.join("pgs/vis_leaf_concat", stdc_result_name)
        if not os.path.exists(yolov5ds_result_path):
            print(f"stdc_result_path: {stdc_result_path} 不存在")
            continue
        stdc_image = cv2.imread(stdc_result_path).copy()
        width = stdc_image.shape[1] // 2
        src_image_np = stdc_image[:, :width]
        yolov5ds_image = cv2.imread(yolov5ds_result_path).copy()
        yolov5ds_result_np = yolov5ds_image[:, width:]
        stdc_result_np = stdc_image[:, width:]
        cv2.putText(yolov5ds_result_np, "yolov5ds", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(stdc_result_np, "stdc", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        res = np.hstack([src_image_np, yolov5ds_result_np, stdc_result_np])
        os.makedirs("pgs/vis_cmp", exist_ok=True)
        cv2.imwrite(os.path.join("pgs/vis_cmp", os.path.basename(stdc_result_path).replace("_concat.png", "_res.png")), res)
    
    exit(0)
    src_front_image_paths = glob.glob(os.path.join("pgs/vis_cases", "*_front_left.png"))

    for src_front_image_path in src_front_image_paths:
        src_side_image_path = src_front_image_path.replace("_front_left.png", "_side_right.png")
        src_front_image = cv2.imread(src_front_image_path)
        src_side_image = cv2.imread(src_side_image_path)
        concat_image = concat_front_and_side_images(src_front_image, src_side_image, (2 * 320, 320))
        os.makedirs("pgs/vis_cases_c", exist_ok=True)
        cv2.imwrite(os.path.join("pgs/vis_cases_c", os.path.basename(src_front_image_path).replace("_front_left.png", "_concat.png")), concat_image)

    exit(0)
    
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
        
