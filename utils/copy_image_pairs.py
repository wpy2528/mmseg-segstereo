import os
import shutil
from pathlib import Path

# 源目录和目标目录
source_dir = "/data/share/filter_hedgehog"
target_dir = "/data/share/hed"
target_images_dir = "/data/user_device_drc/0520_hedgehog"


'''
根据给定的源文件路径列表，从给定目录中找出对应的图片和mask，并复制到目标目录中
'''

# 确保目标目录存在
os.makedirs(target_dir, exist_ok=True)
os.makedirs(os.path.join(target_dir, "images"), exist_ok=True)
os.makedirs(os.path.join(target_dir, "labels"), exist_ok=True)

def find_image_in_source(image_name):
    """在源目录中递归查找图片"""
    for root, _, files in os.walk(source_dir):
        if "images" in root:
            if image_name in files:
                return os.path.join(root, image_name)
    return None

def find_mask_in_source(image_path):
    """根据图片路径找到对应的mask路径"""
    if image_path:
        mask_path = image_path.replace("/images/", "/labels/").replace(".jpg", ".png")
        if os.path.exists(mask_path):
            return mask_path
    return None

# 获取目标图片列表
target_images = os.listdir(target_images_dir)

# 处理每张图片
for image_name in target_images:
    if not image_name.endswith('.jpg'):
        continue
        
    # 在源目录中查找图片
    source_image_path = find_image_in_source(image_name)
    if source_image_path:
        # 查找对应的mask
        source_mask_path = find_mask_in_source(source_image_path)
        
        if source_mask_path:
            # 复制图片和mask到目标目录
            target_image_path = os.path.join(target_dir, "images", image_name)
            target_mask_path = os.path.join(target_dir, "labels", image_name.replace(".jpg", ".png"))
            
            shutil.copy2(source_image_path, target_image_path)
            shutil.copy2(source_mask_path, target_mask_path)
            print(f"Copied: {image_name} and its mask")
        else:
            print(f"Mask not found for: {image_name}")
    else:
        print(f"Image not found in source: {image_name}")

print("Copy process completed!") 