import json
import argparse
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict

def read_metrics_json(json_path):
    """
    读取per_sample_metrics.json文件
    
    Args:
        json_path (str): json文件路径
        
    Returns:
        list: 包含所有样本指标的列表
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"文件不存在: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        metrics_data = json.load(f)
    
    print(f"成功读取 {len(metrics_data)} 个样本的指标数据")
    return metrics_data

def analyze_metrics(metrics_data):
    """
    分析指标数据
    
    Args:
        metrics_data (list): 指标数据列表
        
    Returns:
        dict: 分析结果
    """
    # 提取所有指标
    aacc_values = [item['aAcc'] for item in metrics_data if item['aAcc'] is not None]
    iou_values = []
    acc_values = []
    
    for item in metrics_data:
        if item['IoU'] is not None:
            iou_values.extend([val for val in item['IoU'] if val is not None])
        if item['Acc'] is not None:
            acc_values.extend([val for val in item['Acc'] if val is not None])
    
    analysis = {
        'total_samples': len(metrics_data),
        'valid_aacc_samples': len(aacc_values),
        'valid_iou_samples': len(iou_values),
        'valid_acc_samples': len(acc_values),
        'aacc_stats': {
            'mean': np.mean(aacc_values),
            'std': np.std(aacc_values),
            'min': np.min(aacc_values),
            'max': np.max(aacc_values),
            'median': np.median(aacc_values)
        },
        'iou_stats': {
            'mean': np.mean(iou_values),
            'std': np.std(iou_values),
            'min': np.min(iou_values),
            'max': np.max(iou_values),
            'median': np.median(iou_values)
        },
        'acc_stats': {
            'mean': np.mean(acc_values),
            'std': np.std(acc_values),
            'min': np.min(acc_values),
            'max': np.max(acc_values),
            'median': np.median(acc_values)
        }
    }
    
    return analysis

def plot_metrics_distribution(metrics_data, output_dir='cursor_utils'):
    """
    绘制指标分布图
    
    Args:
        metrics_data (list): 指标数据列表
        output_dir (str): 输出目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 提取数据
    aacc_values = [item['aAcc'] for item in metrics_data if item['aAcc'] is not None]
    iou_values = []
    acc_values = []
    
    for item in metrics_data:
        if item['IoU'] is not None:
            iou_values.extend([val for val in item['IoU'] if val is not None])
        if item['Acc'] is not None:
            acc_values.extend([val for val in item['Acc'] if val is not None])
    
    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('指标分布分析', fontsize=16)
    
    # aAcc分布
    axes[0, 0].hist(aacc_values, bins=30, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].set_title('aAcc分布')
    axes[0, 0].set_xlabel('aAcc值')
    axes[0, 0].set_ylabel('频次')
    axes[0, 0].axvline(np.mean(aacc_values), color='red', linestyle='--', label=f'均值: {np.mean(aacc_values):.4f}')
    axes[0, 0].legend()
    
    # IoU分布
    axes[0, 1].hist(iou_values, bins=30, alpha=0.7, color='green', edgecolor='black')
    axes[0, 1].set_title('IoU分布')
    axes[0, 1].set_xlabel('IoU值')
    axes[0, 1].set_ylabel('频次')
    axes[0, 1].axvline(np.mean(iou_values), color='red', linestyle='--', label=f'均值: {np.mean(iou_values):.4f}')
    axes[0, 1].legend()
    
    # Acc分布
    axes[1, 0].hist(acc_values, bins=30, alpha=0.7, color='orange', edgecolor='black')
    axes[1, 0].set_title('Acc分布')
    axes[1, 0].set_xlabel('Acc值')
    axes[1, 0].set_ylabel('频次')
    axes[1, 0].axvline(np.mean(acc_values), color='red', linestyle='--', label=f'均值: {np.mean(acc_values):.4f}')
    axes[1, 0].legend()
    
    # 箱线图
    data_for_box = [aacc_values, iou_values, acc_values]
    labels = ['aAcc', 'IoU', 'Acc']
    axes[1, 1].boxplot(data_for_box, labels=labels)
    axes[1, 1].set_title('指标箱线图')
    axes[1, 1].set_ylabel('指标值')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"分布图已保存到: {os.path.join(output_dir, 'metrics_distribution.png')}")

def save_analysis_report(analysis, output_dir='cursor_utils'):
    """
    保存分析报告
    
    Args:
        analysis (dict): 分析结果
        output_dir (str): 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    report_path = os.path.join(output_dir, 'analysis_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("指标分析报告\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"总样本数: {analysis['total_samples']}\n")
        f.write(f"有效aAcc样本数: {analysis['valid_aacc_samples']}\n")
        f.write(f"有效IoU样本数: {analysis['valid_iou_samples']}\n")
        f.write(f"有效Acc样本数: {analysis['valid_acc_samples']}\n\n")
        
        f.write("aAcc统计信息:\n")
        for key, value in analysis['aacc_stats'].items():
            f.write(f"  {key}: {value:.6f}\n")
        f.write("\n")
        
        f.write("IoU统计信息:\n")
        for key, value in analysis['iou_stats'].items():
            f.write(f"  {key}: {value:.6f}\n")
        f.write("\n")
        
        f.write("Acc统计信息:\n")
        for key, value in analysis['acc_stats'].items():
            f.write(f"  {key}: {value:.6f}\n")
    
    print(f"分析报告已保存到: {report_path}")

def main():
    parser = argparse.ArgumentParser(description='分析per_sample_metrics.json文件')
    parser.add_argument('json_path', help='per_sample_metrics.json文件路径')
    parser.add_argument('--output_dir', default='cursor_utils', help='输出目录')
    
    args = parser.parse_args()
    
    # 读取数据
    print(f"正在读取文件: {args.json_path}")
    metrics_data = read_metrics_json(args.json_path)

    # 分析IoU值小于0.95的样本
    print("正在分析IoU值小于0.95的样本...")
    low_iou_samples = defaultdict(list)
    
    for item in metrics_data:
        if item['IoU'] is not None:
            for i, iou_val in enumerate(item['IoU']):
                if iou_val is not None and iou_val < 0.95:
                    low_iou_samples[i].append(item['img_path'])
    
    for i, samples in low_iou_samples.items():
        print(f"类别 {i}: 发现 {len(samples)} 个IoU值小于0.95的样本")
        with open(os.path.join(args.output_dir, f'low_iou_class_{i}.txt'), 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(sample + '\n')
    


        
if __name__ == "__main__":
    main()
