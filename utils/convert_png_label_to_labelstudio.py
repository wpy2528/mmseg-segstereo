import os
import argparse
import glob
import cv2
import json
import uuid
from datetime import datetime
import numpy as np
from tqdm import tqdm

# 定义类别映射
CLASS_NAMES = {
    0: "background",
    1: "grassland",
    2: "soil", 
    # 2: "water", 
    # 3: "soil",
    # 4: "animal"
}

# 从CLASS_NAMES中获取soil对应的class_id
SOIL_CLASS_ID = [key for key, value in CLASS_NAMES.items() if value == "soil"][0]

LABELME_TEMPLATE = {
"version": "5.1.1",
"flags": {},
"shapes": [
    {
    "label": "grassland",
    "points": [
        [1.0, 1.0],
        [2.0, 2.0],
    ],
    "group_id": None,
    "shape_type": "polygon",
    "flags": {}
    }
],
"imagePath": "../images/2025-04-24-173756-465_concat.png",
"imageData": None,
"imageHeight": 320,
"imageWidth": 320
}

def add_border_points(approx, contour, height, width):
    for i in range(len(contour)):
        if (contour[i] == approx[0]).all():
            break
    if i != 0:
        contour = contour.tolist()
        contour = contour[i:] + contour[:i]
        contour = np.array(contour)
    result = []
    j = 0
    for i in range(len(contour)):
        pt = contour[i]
        is_border = (pt[0,0] <= 0 or pt[0,0] >= width - 1 or pt[0,1] <= 0 or pt[0,1] >= height - 1)
        if j < len(approx) and (pt == approx[j]).all():
            result.append(pt)
            j += 1
        else:
            if is_border:
                result.append(pt)
    result = np.array(result)
    return result
    
def convert_mask_to_labelme(src_mask_np, src_image_name):
    """Convert a PNG label to a LabelMe JSON format."""
    # 读取PNG标签图像
    label_img = src_mask_np
    # 获取图像尺寸
    height, width = label_img.shape
    
    # 创建LabelMe格式的JSON结构
    labelme_json = LABELME_TEMPLATE.copy()
    labelme_json["shapes"] = []
    
    # 对每个类别进行处理
    for class_id in CLASS_NAMES:
        # 创建二值掩码
        binary_mask = (label_img == class_id).astype(np.uint8)

        if class_id == 1:
            # 进行连通域分析
            # 如果连通域面积小于50，则删除
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask)
            # for i in range(1, num_labels):
            #     if stats[i, cv2.CC_STAT_AREA] < 50:
            #         binary_mask[labels == i] = 0
            # 对剩余的每个连通域遍历，并膨胀
            kernel = np.ones((5, 5), np.uint8)
            binary_mask_eroded = cv2.dilate(binary_mask, kernel, iterations=1)
            # 找出膨胀后的草地mask与泥土mask的交集
            binary_mask_with_soil = cv2.bitwise_and(binary_mask_eroded, (label_img == SOIL_CLASS_ID).astype(np.uint8))
            # 原始草地mask与交集合并（从而防止草地和泥土之间出现间隙）
            binary_mask = cv2.bitwise_or(binary_mask, binary_mask_with_soil)
        
        # 查找轮廓
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 检查面积
        contours = [contour for contour in contours if cv2.contourArea(contour) > 50]
        # 如果是背景类别(class_id=0),过滤掉与图像边界接触的轮廓
        if class_id == 0:
            filtered_contours = []
            for contour in contours:
                # 创建掩码并绘制轮廓
                mask = np.zeros(binary_mask.shape, dtype=np.uint8)
                cv2.drawContours(mask, [contour], -1, 1, -1)
                
                # 检查轮廓是否接触图像边界
                border_contact = False
                if (mask[0,:].any() or  # 上边界
                    mask[-1,:].any() or  # 下边界
                    mask[:,0].any() or   # 左边界
                    mask[:,-1].any()):   # 右边界
                    border_contact = True
                    
                if not border_contact:
                    filtered_contours.append(contour)
            contours = filtered_contours
        
        # 处理每个轮廓
        for contour in contours:
            # 简化轮廓点
            epsilon = 0.001 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            approx = add_border_points(approx, contour, height, width)
            # 转换轮廓点格式
            points = []
            for point in approx:
                x, y = point[0]
                points.append([float(x), float(y)])
            
            # 创建形状字典
            shape = {
                "label": CLASS_NAMES[class_id],
                "points": points,
                "group_id": None,
                "shape_type": "polygon",
                "flags": {}
            }
            
            # 添加到shapes列表
            labelme_json["shapes"].append(shape)
    
    # 更新图像尺寸信息
    labelme_json["imageHeight"] = height
    labelme_json["imageWidth"] = width
    labelme_json["imagePath"] = src_image_name if '.' in src_image_name else src_image_name + '.jpg'
    
    return labelme_json

def convert_labelme_to_labelstudio(labelme_json, dst_image_folder_name, i):
    original_width = labelme_json.get("imageWidth", 640)  # Default width
    original_height = labelme_json.get("imageHeight", 544)  # Default height
    src_image_name = labelme_json.get("imagePath", "")
    
    combined_entry = {
        "id": labelme_json.get("id", i),
        "annotations": [],
        "drafts": [],
        "predictions": [],
        "data": {
            "image": f"/data/local-files/?d=data_pool/{dst_image_folder_name}/{src_image_name}",
        },
        "meta": {},
        "created_at": datetime.utcnow().isoformat() + "Z",
        "updated_at": datetime.utcnow().isoformat() + "Z",
        "inner_id": 1,
        "total_annotations": len(labelme_json.get("shapes", [])),
        "cancelled_annotations": 0,
        "total_predictions": 0,
        "comment_count": 0,
        "unresolved_comment_count": 0,
        "last_comment_updated_at": None,
        "project": 28,
        "updated_by": 1,
        "comment_authors": []
    }

    prediction_result = []  # To collect all polygon labels under one result

    for shape in labelme_json.get("shapes", []):
        normalized_points = [
            [point[0] / (original_width - 1) * 100, point[1] / (original_height - 1) * 100] for point in shape["points"]
        ]
        
        prediction_result.append({
            "original_width": original_width,
            "original_height": original_height,
            "image_rotation": 0,
            "value": {
                "points": normalized_points,
                "closed": True,
                "polygonlabels": [shape["label"]]
            },
            "id": str(uuid.uuid4()),
            "from_name": "label",
            "to_name": "image",
            "type": "polygonlabels",
            "origin": "manual"
        })
    
    # Single prediction with all polygon labels in one result
    predictions = {
        "id": int(uuid.uuid4().int) % 100000,
        "completed_by": 1,
        "result": prediction_result,
        "was_cancelled": False,
        "ground_truth": False,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "updated_at": datetime.utcnow().isoformat() + "Z",
        "draft_created_at": datetime.utcnow().isoformat() + "Z",
        "lead_time": 0,
        "prediction": {},
        "result_count": len(prediction_result),
        "unique_id": str(uuid.uuid4()),
        "import_id": None,
        "last_action": None,
        "task": combined_entry["id"],
        "project": 28,
        "updated_by": 1,
        "parent_prediction": None,
        "parent_annotation": None,
        "last_created_by": None
    }
    
    combined_entry["predictions"].append(predictions)

    return combined_entry
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate LabelStudio JSON from PNG labels')
    parser.add_argument('src_label_dir', type=str)
    parser.add_argument('dst_image_folder_name', type=str)
    args = parser.parse_args()

    os.makedirs(args.dst_image_folder_name, exist_ok=True)

    src_label_paths = []
    if os.path.isdir(args.src_label_dir):
        src_label_paths = glob.glob(os.path.join(args.src_label_dir, "**", "*.png"), recursive=True)
    elif args.src_label_dir.endswith(".txt"):
        with open(args.src_label_dir, "r") as f:
            src_label_paths = [line.strip() for line in f.readlines()]
    else:
        raise ValueError(f"Invalid source label directory: {args.src_label_dir}")
    
    for src_label_path in src_label_paths:
        assert "/labels/" in src_label_path, f"Label path must contain '/labels/': {src_label_path}"
        assert os.path.exists(src_label_path), f"Label path does not exist: {src_label_path}"
    
    combined_data = []
    for i, src_label_path in enumerate(tqdm(src_label_paths)):
        src_mask_np = cv2.imread(src_label_path, cv2.IMREAD_GRAYSCALE)
        src_image_name = os.path.basename(src_label_path).replace(".png", ".jpg")
        labelme_json = convert_mask_to_labelme(src_mask_np, src_image_name)
        combined_entry = convert_labelme_to_labelstudio(labelme_json, args.dst_image_folder_name, i)
        combined_data.append(combined_entry)

    # 将JSON数据保存到文件
    output_json_path = os.path.join("pgs", "combined.json")
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(combined_data, f, indent=2)
    print(f"已将JSON数据保存至: {output_json_path}")
