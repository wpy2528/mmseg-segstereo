import os
import argparse
import cv2
import numpy as np
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='将标注图像中的像素值2和4映射为0,3映射为2')
    parser.add_argument('src_dataset_dir', help='数据集根目录')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    # 递归遍历所有labels目录
    for root, dirs, files in os.walk(args.src_dataset_dir):
        if os.path.basename(root) == 'labels':
            print(f'处理目录: {root}')
            for file in tqdm(files):
                if not file.endswith('.png'):
                    continue
                    
                file_path = os.path.join(root, file)
                
                # 读取图像
                mask = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                
                # 将2和4映射为0,3映射为2
                mask[mask == 2] = 0
                mask[mask == 4] = 0  
                mask[mask == 3] = 2
                
                # 保存修改后的图像
                cv2.imwrite(file_path, mask)

if __name__ == '__main__':
    main()
