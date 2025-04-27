import os
import random
from pathlib import Path

def split_dataset(data_dir, train_ratio=0.8):
    """随机分配训练集和验证集"""
    # 获取所有jpg文件
    jpg_files = list(Path(data_dir).rglob('*.jpg'))
    print(f"找到 {len(jpg_files)} 个jpg文件")
    
    # 随机打乱文件列表
    random.shuffle(jpg_files)
    
    # 计算分割点
    split_idx = int(len(jpg_files) * train_ratio)
    train_files = jpg_files[:split_idx]
    val_files = jpg_files[split_idx:]
    
    # 创建输出目录
    output_dir = Path("data/perception_segmentation/0427")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 写入训练集
    with open(output_dir / "train.txt", 'w') as f:
        for file in train_files:
            f.write(f"{file}\n")
    
    # 写入验证集
    with open(output_dir / "val.txt", 'w') as f:
        for file in val_files:
            f.write(f"{file}\n")
    
    print(f"训练集: {len(train_files)} 个文件")
    print(f"验证集: {len(val_files)} 个文件")
    print(f"训练集已保存到: {output_dir}/train.txt")
    print(f"验证集已保存到: {output_dir}/val.txt")

if __name__ == "__main__":
    data_dir = "/data/mck/grass_seg_data/"
    split_dataset(data_dir) 