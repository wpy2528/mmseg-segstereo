import json
import shutil
import glob
import argparse
import os
import numpy as np
import cv2
from tqdm import tqdm
from pathlib import Path


def parse_labelstudio_to_coco(
    src_anno_path,
    dst_dataset_dir,
    category_config,
    label_studio_prefix="/data/local-files/?d=data_pool/",
    real_prefix="/data/playground/label_anything/data_pool/"
):
    if isinstance(src_anno_path, str):
        assert os.path.isfile(src_anno_path), f"anno file {src_anno_path} not found"
        src_anno_jd = json.load(open(src_anno_path))
    elif isinstance(src_anno_path, (list, dict)):
        src_anno_jd = src_anno_path
    else:
        raise ValueError(f"src_anno_path must be a string or a dict, but got {type(src_anno_path)}")

    assert isinstance(category_config, dict), f"category_config must be a dict, but got {type(category_config)}"
    category_fill_priority = category_config['category_fill_priority']
    category_map = category_config['category_map']
    ignored_categories = category_config['ignored_categories']
    same_category_map = category_config['same_category_map']
    allow_nonexisted_category = category_config['allow_nonexisted_category']

    for anno in tqdm(src_anno_jd):
        src_image_path = anno['data']['image']
        assert label_studio_prefix in src_image_path, f"image {src_image_path} not found in {label_studio_prefix}"
        src_image_path = src_image_path.replace(label_studio_prefix, real_prefix)
        if not os.path.exists(src_image_path):
            print(f"🚨 图片 {src_image_path} 不存在")
            continue

        src_image_np = cv2.imread(src_image_path)
        src_mask_np = np.zeros(src_image_np.shape[:2], dtype=np.uint8)

        polygon_catgory_pairs = []
        for e in anno['annotations'][0]['result']:
            polygon_catogry = e['value']['polygonlabels'][0]
            if polygon_catogry in ignored_categories:
                continue
            if polygon_catogry in same_category_map:
                polygon_catogry = same_category_map[polygon_catogry]
            
            if polygon_catogry not in category_map:
                if allow_nonexisted_category:
                    continue
                else:
                    raise ValueError(f"{src_image_path}  polygon category {polygon_catogry} not found in {category_map}")
            polygon = e['value']['points']
            polygon_np = np.array(polygon) / 100 * (src_image_np.shape[1], src_image_np.shape[0])
            polygon_np = polygon_np.round().astype(np.int32)
            polygon_catgory_pairs.append((polygon_np, polygon_catogry))

        polygon_catgory_pairs.sort(key=lambda x: category_fill_priority.index(x[1]))

        for polygon_np, polygon_catogry in polygon_catgory_pairs:
            # # 如果当前多边形类别为glare，则将高度方向上35%以下的多边形区域删除
            # if polygon_catogry == "glare":
            #     h = src_image_np.shape[0]
            #     cutoff = int(h * 0.35)
            #     # 创建一个与图像同样大小的mask
            #     temp_mask_np = np.zeros(src_image_np.shape[:2], dtype=np.uint8)
            #     cv2.fillPoly(temp_mask_np, [polygon_np], 1)
            #     # 将高度方向上35%以下的区域置为0
            #     temp_mask_np[cutoff:, :] = 0
            #     # 重新提取剩余的多边形轮廓
            #     contours, _ = cv2.findContours(temp_mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            #     for cnt in contours:
            #         if cnt.shape[0] >= 3:
            #             cv2.fillPoly(src_mask_np, [cnt], category_map[polygon_catogry])
            #     continue  # 已经处理完glare，跳过后续的fillPoly
            cv2.fillPoly(src_mask_np, [polygon_np], category_map[polygon_catogry])

        dst_image_path = os.path.join(dst_dataset_dir, "images", os.path.basename(src_image_path)).replace(".png", ".jpg")
        dst_label_path = dst_image_path.replace("/images/", "/labels/").replace(".jpg", ".png")
        os.makedirs(os.path.dirname(dst_image_path), exist_ok=True)
        os.makedirs(os.path.dirname(dst_label_path), exist_ok=True)

        cv2.imwrite(dst_image_path, src_image_np)
        cv2.imwrite(dst_label_path, src_mask_np)

    if isinstance(src_anno_path, str):
        shutil.copy(src_anno_path, dst_dataset_dir)
        print("✅ 标注json文件已经保存到", os.path.join(dst_dataset_dir, os.path.basename(src_anno_path)))
    elif isinstance(src_anno_path, (list, dict)):
        with open(os.path.join(dst_dataset_dir, "annotations.json"), "w") as f:
            json.dump(src_anno_path, f)
            print(f"✅ 标注json文件已经保存到 {os.path.join(dst_dataset_dir, 'annotations.json')}")
    else:
        raise ValueError(f"src_anno_path must be a string or a dict, but got {type(src_anno_path)}")


# 允许作为脚本独立执行
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("src_anno_path", type=str)
    parser.add_argument("src_image_dir", type=str)
    parser.add_argument("dst_dataset_dir", type=str)
    parser.add_argument("--label_studio_prefix", type=str, default="/data/local-files/?d=data_pool/")
    parser.add_argument("--real_prefix", type=str, default="/data/playground/label_anything/data_pool/")
    args = parser.parse_args()

    parse_labelstudio_to_coco(
        src_anno_path=args.src_anno_path,
        dst_dataset_dir=args.dst_dataset_dir,
        label_studio_prefix=args.label_studio_prefix,
        real_prefix=args.real_prefix
    )
