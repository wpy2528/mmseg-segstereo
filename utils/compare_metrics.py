import json
import numpy as np
import argparse
from collections import defaultdict

def load_metrics(file_path):
    """加载指标文件"""
    with open(file_path, 'r') as f:
        return json.load(f)

def compare_iou_only(metrics1, metrics2):
    """只对比IoU指标"""
    # 创建以img_path为键的字典
    metrics1_dict = {item['img_path']: item for item in metrics1}
    metrics2_dict = {item['img_path']: item for item in metrics2}
    
    # 获取共同图片
    common_paths = set(metrics1_dict.keys()) & set(metrics2_dict.keys())
    
    print(f"共同图片数: {len(common_paths)}")
    print()
    
    # 按类别分析IoU差异
    for class_idx in range(len(metrics1[0]['IoU'])):
        print(f"类别 {class_idx} IoU 对比:")
        print("=" * 60)
        
        better_in_first = []  # 第一个文件更好的图片
        better_in_second = []  # 第二个文件更好的图片
        
        for path in common_paths:
            m1 = metrics1_dict[path]
            m2 = metrics2_dict[path]
            
            iou1 = m1['IoU'][class_idx]
            iou2 = m2['IoU'][class_idx]
            
            # 跳过None值
            if iou1 is None or iou2 is None:
                continue
                
            img_path = path
            diff = iou2 - iou1
            
            if diff > 0:
                better_in_second.append((img_path, iou1, iou2, diff))
            elif diff < 0:
                better_in_first.append((img_path, iou1, iou2, diff))
        
        # 打印第一个文件更好的图片
        if better_in_first:
            print(f"第一个文件更好的图片 ({len(better_in_first)}张):")
            for img_path, iou1, iou2, diff in sorted(better_in_first, key=lambda x: x[3]):
                print(f"  {img_path} : {iou1:.6f} -> {iou2:.6f} (差异: {diff:+.6f})")
        else:
            print("第一个文件没有更好的图片")
        
        print()
        
        # 打印第二个文件更好的图片
        if better_in_second:
            print(f"第二个文件更好的图片 ({len(better_in_second)}张):")
            for img_path, iou1, iou2, diff in sorted(better_in_second, key=lambda x: x[3], reverse=True):
                print(f"  {img_path} : {iou1:.6f} -> {iou2:.6f} (差异: {diff:+.6f})")
        else:
            print("第二个文件没有更好的图片")
        
        print()
        
        # 统计信息
        total_comparable = len(better_in_first) + len(better_in_second)
        if total_comparable > 0:
            print(f"统计信息:")
            print(f"  可比较样本数: {total_comparable}")
            print(f"  第一个文件更好: {len(better_in_first)} ({len(better_in_first)/total_comparable*100:.1f}%)")
            print(f"  第二个文件更好: {len(better_in_second)} ({len(better_in_second)/total_comparable*100:.1f}%)")
        
        print("\n" + "="*80 + "\n")

def main():
    parser = argparse.ArgumentParser(description='对比两个指标文件的IoU')
    parser.add_argument('file1', help='第一个指标文件路径')
    parser.add_argument('file2', help='第二个指标文件路径')
    
    args = parser.parse_args()
    
    # 加载指标文件
    print(f"加载文件1: {args.file1}")
    metrics1 = load_metrics(args.file1)
    print(f"加载文件2: {args.file2}")
    metrics2 = load_metrics(args.file2)
    
    # 对比IoU
    compare_iou_only(metrics1, metrics2)

if __name__ == '__main__':
    main()

