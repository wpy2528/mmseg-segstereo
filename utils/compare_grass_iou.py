import json
import os
import argparse
from collections import defaultdict

parser = argparse.ArgumentParser(description='比较两个JSON文件中第1类别的IoU值')
parser.add_argument('sota_file', type=str, help='SOTA结果JSON文件路径')
parser.add_argument('metrics_file', type=str, help='Metrics结果JSON文件路径')
parser.add_argument("--threshold", type=float, default=0.1, help='阈值')
args = parser.parse_args()
    

def load_json(file_path):
    """加载JSON文件"""
    with open(file_path, 'r') as f:
        return json.load(f)

def compare_iou(sota_file, metrics_file):
    """比较两个JSON文件中第1类别的IoU值，列出SOTA结果更高的样本"""
    # 加载两个JSON文件
    sota_data = load_json(sota_file)
    metrics_data = load_json(metrics_file)
    
    # 创建字典存储结果
    results = defaultdict(dict)
    
    # 处理sota结果
    for item in sota_data:
        img_path = item['img_path']
        iou = item['IoU'][1]  # 获取第1类别的IoU
        results[img_path]['sota'] = iou
    
    # 处理metrics结果
    for item in metrics_data:
        img_path = item['img_path']
        iou = item['IoU'][1]  # 获取第1类别的IoU
        results[img_path]['metrics'] = iou
    
    # 找出SOTA结果更高的样本
    better_samples = []
    for img_path, values in results.items():
        if 'sota' in values and 'metrics' in values:
            if values['sota'] - values['metrics'] > args.threshold:
                better_samples.append((img_path, values))
    
    # 按差异大小排序
    better_samples.sort(key=lambda x: x[1]['sota'] - x[1]['metrics'], reverse=True)
    
    if better_samples:
        print("\nSOTA结果更好的样本（按差异大小排序）:")
        for img_path, values in better_samples:
            print(f"图像: {img_path}")
            print(f"SOTA IoU: {values['sota']:.4f}")
            print(f"Metrics IoU: {values['metrics']:.4f}")
            print(f"差异: {values['sota'] - values['metrics']:.4f}")
            print()
    
    print(f"总图像数: {len(results)}")
    print(f"SOTA结果更好的样本数: {len(better_samples)}")
    

def main():
    if not os.path.exists(args.sota_file):
        print(f"错误: 文件 {args.sota_file} 不存在")
    elif not os.path.exists(args.metrics_file):
        print(f"错误: 文件 {args.metrics_file} 不存在")
    else:
        compare_iou(args.sota_file, args.metrics_file)

if __name__ == "__main__":
    main() 