import cv2
import numpy as np
import argparse
import os
from tqdm import tqdm

def pfm_imread(filename):
    """读取PFM文件"""
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    import re
    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data

def visualize_pfm(pfm_path, colormap=cv2.COLORMAP_JET):
    """可视化PFM文件"""
    # 读取PFM文件
    if isinstance(pfm_path, str):
        disp_data = pfm_imread(pfm_path)
    else:
        disp_data = pfm_path
    
    # 归一化到0-255
    disp_min = np.min(disp_data)
    disp_max = np.max(disp_data)
    disp_normalized = ((disp_data - disp_min) / (disp_max - disp_min) * 255).astype(np.uint8)
    
    # 应用颜色映射
    disp_colored = cv2.applyColorMap(disp_normalized, colormap)
    
    # 保存或显示结果
    # if output_path:
    #     cv2.imwrite(output_path, disp_colored)
    #     print(f"可视化结果已保存到: {output_path}")
    # else:
    #     cv2.imshow('PFM Visualization', disp_colored)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    
    return disp_colored

def batch_visualize_pfm(input_dir, output_dir, colormap=cv2.COLORMAP_JET):
    """批量可视化目录中的PFM文件"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 递归查找所有PFM文件
    pfm_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.pfm'):
                pfm_files.append(os.path.join(root, file))
    
    print(f"找到 {len(pfm_files)} 个PFM文件")
    
    for pfm_file in tqdm(pfm_files, desc="处理PFM文件"):
        # 计算相对路径
        rel_path = os.path.relpath(pfm_file, input_dir)
        output_file = os.path.join(output_dir, rel_path.replace('.pfm', '.png'))
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # 可视化并保存
        visualize_pfm(pfm_file, output_file, colormap)

def main():
    parser = argparse.ArgumentParser(description='PFM文件可视化工具')
    parser.add_argument('input', help='输入PFM文件路径或包含PFM文件的目录')
    parser.add_argument('output', nargs='?', help='输出图像路径或目录（可选）')
    parser.add_argument('--colormap', default='jet', choices=['jet', 'viridis', 'plasma', 'inferno', 'magma', 'hot', 'cool'],
                       help='颜色映射类型')
    
    args = parser.parse_args()
    
    # 颜色映射字典
    colormap_dict = {
        'jet': cv2.COLORMAP_JET,
        'viridis': cv2.COLORMAP_VIRIDIS,
        'plasma': cv2.COLORMAP_PLASMA,
        'inferno': cv2.COLORMAP_INFERNO,
        'magma': cv2.COLORMAP_MAGMA,
        'hot': cv2.COLORMAP_HOT,
        'cool': cv2.COLORMAP_COOL
    }
    
    colormap = colormap_dict[args.colormap]
    
    if os.path.isfile(args.input):
        # 单个文件处理
        if args.output is None:
            visualize_pfm(args.input, colormap=colormap)
        else:
            visualize_pfm(args.input, args.output, colormap)
    elif os.path.isdir(args.input):
        # 目录批量处理
        if args.output is None:
            args.output = os.path.join(os.path.dirname(args.input), 'pfm_visualization')
        batch_visualize_pfm(args.input, args.output, colormap)
    else:
        print(f"错误: 输入路径 '{args.input}' 不存在")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
