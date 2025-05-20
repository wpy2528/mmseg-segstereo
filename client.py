import os
import json
import numpy as np
import cv2
import glob
import requests
import argparse
from tqdm import tqdm

from utils.convert_png_label_to_labelstudio import convert_mask_to_labelme, convert_labelme_to_labelstudio

SERVER_URL = 'http://192.168.160.52:5000/infer'

parser = argparse.ArgumentParser()
parser.add_argument("src_image_dir")
parser.add_argument("vis_save_dir")
parser.add_argument("--save_labelstudio_json", action="store_true", help="是否保存labelstudio的json文件")
args = parser.parse_args()


if __name__ == "__main__":
    src_image_dir = args.src_image_dir
    src_image_dir = src_image_dir.rstrip("/")
    src_image_folder_name = os.path.basename(src_image_dir)
    vis_save_dir = args.vis_save_dir
    if os.path.isfile(src_image_dir):
        src_image_paths = [src_image_dir]
    else:
        src_image_paths = glob.glob(os.path.join(src_image_dir, "**", "*.jpg"), recursive=True) + glob.glob(os.path.join(src_image_dir, "**", "*.png"), recursive=True) 
    
    
    os.makedirs(vis_save_dir, exist_ok=True)
    combined_data = []
    for i, src_image_path in enumerate(tqdm(src_image_paths)):

        files = {'image': open(src_image_path, 'rb')}
        print(f"📤 正在上传: {src_image_path} ...")

        response = requests.post(SERVER_URL, files=files, timeout=10)
        if response.status_code == 200:
            save_path = os.path.join(vis_save_dir, os.path.basename(src_image_path)).replace(".jpg", ".png")
            with open(save_path, 'wb') as f:
                f.write(response.content)
            # response.content转opencv
            if args.save_labelstudio_json:
                img = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
                mask_np = img[:, img.shape[1] // 3 * 2:, :]
                mask_np = cv2.cvtColor(mask_np, cv2.COLOR_BGR2GRAY)
                labelme_json = convert_mask_to_labelme(mask_np, os.path.basename(src_image_path))
                combined_entry = convert_labelme_to_labelstudio(labelme_json, src_image_folder_name, i)
                combined_data.append(combined_entry)
            
            print(f"✅ 已保存: {save_path}")
        else:
            print(f"❌ 错误: {response.status_code}, 内容: {response.text}")

    if args.save_labelstudio_json:
        # 将JSON数据保存到文件
        output_json_path = f"{src_image_folder_name}.json"
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(combined_data, f, indent=2)
        print(f"已将JSON数据保存至: {output_json_path}")