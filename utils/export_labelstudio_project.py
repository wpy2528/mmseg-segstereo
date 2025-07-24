import argparse
import time
import json
import os
import requests
import sys
from datetime import datetime

from parse_labelstudio_anno import parse_labelstudio_to_coco

# ---------------------------
# 配置（可根据实际情况修改）
LABEL_STUDIO_URL = "http://localhost:8080"
API_TOKEN = "49438e22558ae996d008a9cbc2cb6f0f1685760b"
HEADERS = {
    "Authorization": f"Token {API_TOKEN}"
}
# ---------------------------

def get_all_projects(i=1):
    url = f"{LABEL_STUDIO_URL}/api/projects/?page={i}"
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    return response.json()

def get_project_id_by_name(projects, name):
    for project in projects["results"]:
        if project.get("title") == name:
            return project.get("id")
    return None

def export_project_annotations(project_id):
    url = f"{LABEL_STUDIO_URL}/api/projects/{project_id}/export?exportType=JSON"
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    return response.content

def main():
    parser = argparse.ArgumentParser(description="Export Label Studio project annotations by project name")
    parser.add_argument("project_name", type=str, help="项目名称（Label Studio 中显示的名字）")
    parser.add_argument("dst_dataset_dir", type=str, help="导出保存目录")
    parser.add_argument("category_config_name", type=str, help="类别配置名称")

    args = parser.parse_args()
    
    category_config = json.load(open("utils/category_configs.json"))[args.category_config_name]
    print(f"使用 {args.category_config_name} 类别配置")
    time.sleep(2)
    
    project_name = args.project_name
    dst_dataset_dir = args.dst_dataset_dir.rstrip("/")
    if os.path.exists(dst_dataset_dir):
        # 判断目录中是否有images文件夹
        if not (os.path.exists(os.path.join(dst_dataset_dir, "images")) and os.path.exists(os.path.join(dst_dataset_dir, "labels"))):
            raise ValueError(f"目录 {dst_dataset_dir} 已经存在，但是里头没有images文件夹或者labels文件夹，说明这个目录不是一个独立的数据集目录，好好检查检查是不是放到大数据集根目录了")
        else:
            print(f"目录 {dst_dataset_dir} 已经存在，想中止的话抓紧啦！")
            time.sleep(5)

    print("🔍 获取项目列表...")
    for i in range(1, 4):
        projects = get_all_projects(i)
        project_id = get_project_id_by_name(projects, project_name)
        if project_id:
            break
    if not project_id:
        print(f"❌ 找不到项目：{project_name}")
        sys.exit(1)

    print(f"📦 正在导出项目（ID: {project_id}）的标注...")
    data = export_project_annotations(project_id)
    dst_anno_path = os.path.join(dst_dataset_dir, f"{project_name}.json")
    labelstudio_jd = json.loads(data)
    parse_labelstudio_to_coco(labelstudio_jd, dst_dataset_dir, category_config)
    print(f"✅ 导出完成，保存到 {dst_dataset_dir}")

if __name__ == "__main__":
    main()
