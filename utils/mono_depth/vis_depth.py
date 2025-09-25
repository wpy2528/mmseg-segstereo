#!/usr/bin/env python3
"""
Orbbec深度图像数据集可视化工具
遍历Color和Depth目录中的图像对进行可视化
"""

import os
import cv2
import numpy as np
import argparse
import glob


def quantize_depth_to_uint8(depth_np, min_depth=0.0, max_depth=10000.0):
    """
    将16位深度图量化到uint8
    
    Args:
        depth_np (np.ndarray): 16位深度图
        min_depth (float): 最小深度值
        max_depth (float): 最大深度值
        
    Returns:
        np.ndarray: 量化后的uint8深度图
    """
    # 将深度值限制在指定范围内
    depth_clipped = np.clip(depth_np, min_depth, max_depth)
    
    # 归一化到0-255范围
    depth_normalized = (depth_clipped - min_depth) / (max_depth - min_depth)
    depth_uint8 = (depth_normalized * 255).astype(np.uint8)
    
    return depth_uint8


def find_image_pairs(orbbec_dir):
    """
    在orbbec目录中查找Color和Depth图像对
    
    Args:
        orbbec_dir (str): orbbec根目录路径
        
    Returns:
        list: 图像对列表，每个元素为(color_path, depth_path)
    """
    image_pairs = []
    
    src_image_paths = sorted(glob.glob(os.path.join(orbbec_dir, '**', 'Color', '*.jpg'), recursive=True))
    for src_image_path in src_image_paths:
        color_path = src_image_path
        depth_path = src_image_path.replace('Color', 'Depth').replace('.jpg', '.png')
        image_pairs.append((color_path, depth_path))
    
    return image_pairs

def visualize_image_pair(color_path, depth_path, min_depth=0.0, max_depth=10000.0):
    """
    可视化单个图像对
    
    Args:
        color_path (str): 彩色图像路径
        depth_path (str): 深度图像路径
        min_depth (float): 最小深度值
        max_depth (float): 最大深度值
    """
    print(color_path, depth_path)
    # 读取彩色图像
    color_img = cv2.imread(color_path)
    if color_img is None:
        print(f"无法读取彩色图像: {color_path}")
        return None
    
    # 读取16位深度图
    depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth_img is None:
        print(f"无法读取深度图像: {depth_path}")
        return None
    
    # 量化深度图到uint8
    depth_uint8 = quantize_depth_to_uint8(depth_img, min_depth, max_depth)
    
    # 应用伪彩色映射
    depth_color = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)
    
    # 调整图像大小使其一致
    target_height = min(color_img.shape[0], depth_img.shape[0])
    target_width = min(color_img.shape[1], depth_img.shape[1])
    
    color_resized = cv2.resize(color_img, (target_width, target_height))
    print(color_resized.shape)
    depth_resized = cv2.resize(depth_img, (target_width, target_height))
    depth_uint8_resized = cv2.resize(depth_uint8, (target_width, target_height))
    depth_color_resized = cv2.resize(depth_color, (target_width, target_height))
    
    # 水平拼接图像
    vis_img = np.concatenate([
        color_resized,
        # cv2.cvtColor(depth_resized, cv2.COLOR_GRAY2BGR),
        # cv2.cvtColor(depth_uint8_resized, cv2.COLOR_GRAY2BGR),
        depth_color_resized
    ], axis=1)
    
    return vis_img


def main():
    parser = argparse.ArgumentParser(description='Orbbec深度图像数据集可视化工具')
    parser.add_argument('orbbec_dir', help='Orbbec数据集目录路径')
    parser.add_argument('output_dir', help='输出目录路径')
    parser.add_argument('--min_depth', type=float, default=0.0, help='最小深度值（默认: 0.0）')
    parser.add_argument('--max_depth', type=float, default=10000.0, help='最大深度值（默认: 10000.0）')
    parser.add_argument('--start_idx', type=int, default=0, help='开始索引（默认: 0）')
    parser.add_argument('--end_idx', type=int, default=None, help='结束索引（默认: 处理所有）')
    
    args = parser.parse_args()
    
    # 检查目录
    if not os.path.exists(args.orbbec_dir):
        print(f"错误: 目录 {args.orbbec_dir} 不存在")
        return
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 查找图像对
    print("正在查找图像对...")
    image_pairs = find_image_pairs(args.orbbec_dir)
    
    if not image_pairs:
        print("未找到任何图像对")
        return
    
    print(f"找到 {len(image_pairs)} 个图像对")
    
    # 确定处理范围
    start_idx = min(args.start_idx, len(image_pairs) - 1)
    end_idx = args.end_idx if args.end_idx is not None else len(image_pairs)
    end_idx = min(end_idx, len(image_pairs))
    
    print(f"处理范围: {start_idx} 到 {end_idx-1}")
    
    # 处理图像对
    for i in range(start_idx, end_idx):
        color_path, depth_path = image_pairs[i]
        print(f"处理 {i+1}/{end_idx}: {os.path.basename(color_path)}")
        
        # 可视化图像对
        vis_img = visualize_image_pair(color_path, depth_path, args.min_depth, args.max_depth)
        
        if vis_img is not None:
            # 添加标题
            title_height = 30
            title_img = np.zeros((title_height, vis_img.shape[1], 3), dtype=np.uint8)
            cv2.putText(title_img, f"Color | Depth (16bit) | Depth (8bit) | Depth (Colormap)", 
                       (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            vis_img = np.concatenate([title_img, vis_img], axis=0)
            
            # 保存图像
            output_filename = f"vis_{os.path.basename(color_path).replace('.jpg', '.png')}"
            output_path = os.path.join(args.output_dir, output_filename)
            cv2.imwrite(output_path, vis_img)
            print(f"  保存到: {output_path}")
        else:
            print(f"跳过无效的图像对: {color_path}")
    
    print("处理完成！")


if __name__ == '__main__':
    main()
