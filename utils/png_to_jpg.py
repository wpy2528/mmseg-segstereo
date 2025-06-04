import os
import cv2
from tqdm import tqdm

def convert_png_to_jpg(input_dir):
    """
    将指定目录及其子目录下的所有PNG图像转换为JPG格式
    """
    # 确保输入目录存在
    if not os.path.exists(input_dir):
        print(f"错误：目录 {input_dir} 不存在")
        return

    # 获取所有PNG文件（包括子目录）
    png_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.png'):
                png_files.append(os.path.join(root, file))
    
    if not png_files:
        print(f"警告：在 {input_dir} 及其子目录中没有找到PNG文件")
        return

    # 处理每个文件
    for png_path in tqdm(png_files, desc="转换文件"):
        jpg_path = png_path.replace('.png', '.jpg')
        
        # 读取PNG图像
        img = cv2.imread(png_path)
        
        if img is None:
            print(f"警告：无法读取文件 {png_path}")
            continue
            
        # 保存为JPG格式
        cv2.imwrite(jpg_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        # 删除原PNG文件
        os.remove(png_path)

if __name__ == "__main__":
    input_dir = "pgs/leaf2_3x3"
    convert_png_to_jpg(input_dir) 