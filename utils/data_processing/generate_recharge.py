import os
import cv2
import numpy as np
import json
import urllib.parse
from pathlib import Path
import shutil


CATEGORY_LABELS = {
    'ver': 1,
    'triangle': 3,
    'electrode': 2, 
    'background': 0
}

def extract_combined_regions(json_file, image_folder, output_folder, categories=('ver', 'triangle', 'electrode')):

    # 读取JSON文件
    with open(json_file, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    os.makedirs(output_folder, exist_ok=True)
    extracted_data = []
    
    for task in json_data:
        if 'data' not in task or 'image' not in task['data']:
            continue
            
        # 解析图片路径
        image_uri = task['data']['image']
        parsed_uri = urllib.parse.urlparse(image_uri)
        query_params = urllib.parse.parse_qs(parsed_uri.query)
        image_path = query_params.get('d', [''])[0]
        
        if not image_path:
            continue
            
        # 加载图片
        full_image_path = os.path.join(image_folder, image_path)
        img = cv2.imread(full_image_path)
        if img is None:
            continue
        
        # 获取图片尺寸
        height, width = img.shape[:2]
        
        # 创建透明图像和掩码
        transparent_img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        transparent_img[:, :, 3] = 0  # 设置全透明
        mask_img = np.zeros((height, width), dtype=np.uint8)
        
        # 存储提取的区域坐标
        extracted_regions = []
        
        # 检查是否有标注
        if not task.get('annotations'):
            continue
            
        # 遍历所有标注
        for annotation in task['annotations']:
            if not annotation.get('result'):
                continue
                
            # 遍历所有形状标注
            for shape in annotation['result']:
                category = shape['value'].get('polygonlabels', [''])[0]
                
                # 只处理指定类别
                if category in categories:
                    points = shape['value']['points']
                    original_width = shape['original_width']
                    original_height = shape['original_height']
                    
                    # 转换百分比坐标为像素坐标
                    points_pixels = np.array([(int(x/100 * original_width), int(y/100 * original_height)) 
                                             for x, y in points], np.int32)
                    
                    # 创建多边形遮罩
                    mask = np.zeros((height, width), dtype=np.uint8)
                    cv2.fillPoly(mask, [points_pixels], 255)
                    
                    # 计算区域面积
                    area = cv2.contourArea(points_pixels)
                    
                    # 将类别区域添加到透明图像
                    transparent_img[:, :, 3] = np.maximum(transparent_img[:, :, 3], mask)
                    
                    # 更新掩码图像
                    mask_img = np.where(mask > 0, CATEGORY_LABELS[category], mask_img)
                    
                    # 存储区域坐标
                    extracted_regions.append({
                        'category': category,
                        'points': points_pixels.tolist(),
                        'area': area
                    })
        
        # 检查是否有有效区域
        if extracted_regions:
            # 生成输出文件名
            image_name = Path(image_path).stem
            transparent_filename = f"{image_name}_transparent.png"
            mask_filename = f"{image_name}_mask.png"
            
            # 保存透明PNG
            transparent_path = os.path.join(output_folder, transparent_filename)
            cv2.imwrite(transparent_path, transparent_img)
            
            # 保存掩码文件
            mask_path = os.path.join(output_folder, mask_filename)
            cv2.imwrite(mask_path, mask_img)
            
            # 存储元数据
            extracted_data.append({
                'transparent_path': transparent_path,
                'mask_path': mask_path,
                'width': width,
                'height': height,
                'original_image': image_path
            })
    
    print(f"提取完成: 共提取 {len(extracted_data)} 个区域 (包含ver, triangle和electrode)")
    return extracted_data

def fixed_position_paste(extracted_data, target_folder, output_folder, mask_output_folder):

    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(mask_output_folder, exist_ok=True)
    
    # 获取所有目标图片
    target_images = [f for f in os.listdir(target_folder) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not target_images:
        print(f"未找到目标图片: {target_folder}")
        return
    
    print(f"找到 {len(target_images)} 张目标图片和 {len(extracted_data)} 张抠图")
    
    # 确保目标图片数量足够（如果不够，循环使用）
    for i, data in enumerate(extracted_data):
        # 选择对应的目标图片
        target_index = i % len(target_images)
        target_file = target_images[target_index]
        
        # 读取目标图片
        target_path = os.path.join(target_folder, target_file)
        target_img = cv2.imread(target_path)
        if target_img is None:
            print(f"无法加载目标图片: {target_path}")
            continue
        
        # 获取目标图片尺寸
        target_height, target_width = target_img.shape[:2]
        
        # 检查尺寸是否匹配
        if target_width != data['width'] or target_height != data['height']:
            print(f"尺寸不匹配: 原始尺寸={data['width']}x{data['height']}, 目标尺寸={target_width}x{target_height}")
            continue
        
        # 读取透明图片
        transparent_img = cv2.imread(data['transparent_path'], cv2.IMREAD_UNCHANGED)
        if transparent_img is None:
            print(f"无法加载透明图片: {data['transparent_path']}")
            continue
        
        # 创建结果图片副本
        result_img = target_img.copy()
        
        # 分离RGB和alpha通道
        if transparent_img.shape[2] == 4:
            paste_img = transparent_img[:, :, :3]
            alpha = transparent_img[:, :, 3] / 255.0
        else:
            paste_img = transparent_img
            alpha = np.ones(transparent_img.shape[:2], dtype=np.float32)
        
        # 直接在原始位置粘贴（固定位置）
        for c in range(3):
            result_img[:, :, c] = result_img[:, :, c] * (1 - alpha) + paste_img[:, :, c] * alpha
        
        # 生成输出文件名
        output_filename = f"merged_{Path(target_file).stem}.png"
        output_path = os.path.join(output_folder, output_filename)
        cv2.imwrite(output_path, result_img)
        
        # 生成掩码文件名
        mask_output_filename = f"mask_{Path(target_file).stem}.png"
        mask_output_path = os.path.join(mask_output_folder, mask_output_filename)
        shutil.copy2(data['mask_path'], mask_output_path)
        
        print(f"粘贴: 抠图 {i} -> {target_file} (保存为: {output_filename})")

def main():
    # 1. 设置路径
    json_file = "project-346-at-2025-08-20-12-45-1e0ff804.json"  # 原始JSON文件,从ls上导出
    image_folder = "/data/playground/label_anything"  # 原始图片文件夹
    extracted_output_folder = "extracted_regions"  # 存放抠出来的充电桩
    target_folder = "/data/playground/label_anything/data_pool/0107_edge_data"  # 目标图片文件夹
    final_output_folder = "merged_images"  # 最终输出图片文件夹
    mask_output_folder = "masks"  # 标签掩码输出文件夹
    
    # 2. 提取区域 - 新增"electrode"类别
    print("提取 'ver', 'triangle' 和 'electrode' 区域...")
    extracted_data = extract_combined_regions(
        json_file, 
        image_folder, 
        extracted_output_folder,
        categories=('ver', 'triangle', 'electrode')
    )
    
    # 3. 固定位置粘贴到目标图片
    fixed_position_paste(
        extracted_data,
        target_folder,
        final_output_folder,
        mask_output_folder
    )
    


if __name__ == "__main__":
    main()
