import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm

# 定义类别颜色
PALETTE = np.array([
    [128, 0, 128],  # 紫色
    [0, 255, 0],    # 绿色
    [255, 0, 0],    # 蓝色
    [0, 255, 255],  # 黄色
    [0, 0, 255]     # 红色
], dtype=np.uint8)





def visualize_segmentation(image_path, seg_path, concat=False):
    """
    可视化分割结果
    """
    # 读取原图和分割结果
    image = cv2.imread(image_path)
    seg = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)
    
    # 创建彩色分割图
    h, w = seg.shape
    color_seg = np.zeros((h, w, 3), dtype=np.uint8)
    
    # 为每个类别填充对应的颜色
    for i in range(len(PALETTE)):
        color_seg[seg == i] = PALETTE[i]
    
    # 将彩色分割图与原图进行混合
    alpha = 0.5
    beta = 1 - alpha
    gamma = 0
    result = cv2.addWeighted(image, alpha, color_seg, beta, gamma)
    if concat:
        result = np.concatenate([image, result], axis=1)
    
    return result


def get_image_paths(input_path):
    """
    根据输入路径获取所有需要处理的图像路径
    """
    image_paths = []
    
    if os.path.isfile(input_path):
        if input_path.endswith('.txt'):
            # 从txt文件读取图像路径
            with open(input_path, 'r') as f:
                image_paths = [line.strip() for line in f.readlines()]
        else:
            # 单张图像
            image_paths = [input_path]
    elif os.path.isdir(input_path):
        # 遍历目录中的所有jpg文件
        for root, _, files in os.walk(input_path):
            for file in files:
                if file.endswith('.jpg'):
                    image_paths.append(os.path.join(root, file))
    
    return image_paths

def main():
    parser = argparse.ArgumentParser(description='Visualize segmentation results')
    parser.add_argument('input', help='Input image path, directory path, or txt file path')
    parser.add_argument('output_dir', help='Output directory path')
    parser.add_argument("--concat", action="store_true", help="Concatenate images horizontally")
    args = parser.parse_args()
    # 获取所有需要处理的图像路径
    image_paths = get_image_paths(args.input)
    if not image_paths:
        print(f"Error: No valid images found in {args.input}")
        return
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 处理每张图像
    for image_path in tqdm(image_paths):
        # 获取图像名称
        image_name = os.path.basename(image_path)
        # 生成分割结果路径
        seg_path = image_path.replace("/images/", "/labels/").replace(".jpg", ".png")
        # 生成输出路径
        output_path = os.path.join(args.output_dir, image_name)
        
        if not os.path.exists(seg_path):
            print(f"Warning: Segmentation file not found for {image_path}")
            continue
        
        vis_np = visualize_segmentation(image_path, seg_path, args.concat)
        cv2.imwrite(output_path, vis_np)
if __name__ == "__main__":
    main() 