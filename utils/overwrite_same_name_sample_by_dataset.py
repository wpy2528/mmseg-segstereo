import cv2
import shutil
import glob
import os
import numpy as np
import argparse
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from pathlib import Path

'''
这个脚本的作用通常是将筛选出的错误标注组成一个新数据集进行标注后，
将新数据集中的样本覆盖到原数据集中，从而达到提升标注质量的目的。
'''

def process_dataset(src_dataset_dir, filter_dir):
    """
    处理数据集中的重复样本
    
    Args:
        src_dataset_dir (str): 数据集根目录
        filter_dir (str): 优先使用的目录名（如 'soil_filter'）
    """
    src_image_paths = glob.glob(os.path.join(src_dataset_dir, "**", "images", "*.jpg"), recursive=True)
    filter_paths = [p for p in src_image_paths if f"/{filter_dir}/" in str(p)]
    non_filter_paths = [p for p in src_image_paths if f"/{filter_dir}/" not in str(p)]

    # 获取filter_paths中的文件名到路径的映射
    filter_map = {Path(p).name: p for p in filter_paths}
    
    # 获取non_filter_paths中的文件名到路径的映射
    non_filter_map = {Path(p).name: p for p in non_filter_paths}
    
    # 检查每个filter文件是否在non_filter中存在
    for name in filter_map:
        src_image_path = filter_map[name]
        src_label_path = src_image_path.replace("/images", "/labels/").replace(".jpg", ".png")
        if name in non_filter_map:
            print(f"文件 {name} 存在对应关系:")
            print(f"{filter_dir}路径: {filter_map[name]}")
            print(f"non_{filter_dir}路径: {non_filter_map[name]}")
            print("---")
            dst_image_path = non_filter_map[name]
            dst_label_path = dst_image_path.replace("/images", "/labels/").replace(".jpg", ".png")

            shutil.copy(src_image_path, dst_image_path)
            shutil.copy(src_label_path, dst_label_path)
        else:
            # 删除
            os.remove(src_image_path)
            os.remove(src_label_path)

def main():
    parser = argparse.ArgumentParser(description='处理数据集中的重复样本')
    parser.add_argument('--src_dir', type=str, default="/home/mck/datasets/grass_seg_data_c3/",
                        help='数据集根目录')
    parser.add_argument('--filter_dir', type=str, required=True,
                        help='重新标注数据组成的新数据集的目录名称')
    
    args = parser.parse_args()
    assert '/' not in args.filter_dir, "filter_dir不能包含斜杠"
    
    process_dataset(args.src_dir, args.filter_dir)

if __name__ == "__main__":
    main()
