import json
import os
import argparse
from pathlib import Path

def find_annotation_by_image(json_path, image_name):
    """
    根据图像名称查找对应的标注信息
    
    Args:
        json_path: JSON文件路径
        image_name: 图像文件名
    
    Returns:
        list: 包含标注信息的列表，每个元素是一个字典，包含id和标注人员信息
    """
    # 读取JSON文件
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = []
    
    # 遍历所有标注项
    for item in data:
        # 获取图像路径
        image_path = item.get('data', {}).get('image', '')
        # 从路径中提取文件名
        current_image_name = os.path.basename(image_path)
        
        # 如果找到匹配的图像
        if image_name in current_image_name:
            # 获取标注信息
            annotations = item.get('annotations', [])
            # print(annotations)
            for anno in annotations:
                result = {
                    'id': item.get('id'),
                    'annotator': anno.get('created_username', 'unknown'),
                    'image_path': image_path
                }
                results.append(result)
    
    return results

def main():
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='查找图像标注信息')
    parser.add_argument('json_path', type=str, help='标注JSON文件的路径')
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.json_path):
        print(f"错误：文件 {args.json_path} 不存在")
        return
    
    print(f"已加载标注文件：{args.json_path}")
    print("输入图像名称进行查询（输入'q'退出）：")
    
    # 交互式查询循环
    while True:
        # 获取用户输入
        image_name = input("\n请输入图像名称: ").strip()
        
        # 检查是否退出
        if image_name.lower() == 'q':
            print("程序已退出")
            break
        
        # 如果输入为空，继续下一次循环
        if not image_name:
            continue
        
        # 查找标注信息
        results = find_annotation_by_image(args.json_path, image_name)
        
        # 打印结果
        if results:
            print(f"\n找到 {len(results)} 条标注信息：")
            for result in results:
                print(f"ID: {result['id']}")
                print(f"标注人员: {result['annotator']}")
                print(f"图像路径: {result['image_path']}")
                print("-" * 50)
        else:
            print(f"\n未找到图像 {image_name} 的标注信息")

if __name__ == '__main__':
    main() 