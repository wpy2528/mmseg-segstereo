# Copyright (c) OpenMMLab. All rights reserved.
import time
from tqdm import tqdm
import os
import cv2
import numpy as np
from argparse import ArgumentParser
import glob
from mmengine.model import revert_sync_batchnorm
from mmengine.structures import PixelData
from mmseg.apis import inference_model, init_model, show_result_pyplot
from flask import Flask, request, jsonify, send_file
from io import BytesIO
import hashlib

def get_color_mask(mask):
    color_mask = np.zeros_like(mask, dtype=np.uint8)
    color_mask = cv2.cvtColor(color_mask, cv2.COLOR_GRAY2BGR)
    color_mask[mask == 0] = (255, 0, 255)
    color_mask[mask == 1] = (0, 255, 0)
    color_mask[mask == 2] = (0, 255, 255)
    color_mask[mask == 3] = (0, 0, 255)
    return color_mask


def infer_image(model, src_image_path, src_image_np, args):
    result = inference_model(model, src_image_np)
    gt_image_path = src_image_path.replace("/images/", "/labels/").replace(".jpg", ".png")
    if (gt_image_path != src_image_path and os.path.exists(gt_image_path)):
        draw_gt = True
        gt_image = cv2.imread(gt_image_path, cv2.IMREAD_GRAYSCALE)
        vis_gt = cv2.addWeighted(src_image_np, 1, get_color_mask(gt_image), 0.5, 0)
        cv2.putText(vis_gt, "gt", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        result.gt_sem_seg = PixelData(data=gt_image)
    else:
        draw_gt = False
    # 保存预测mask
    if args.save_pred_mask:
        mask_t = result._pred_sem_seg.data
        mask_np = mask_t.cpu().numpy().astype(np.uint8)
        cv2.imwrite(os.path.join(args.out_file, os.path.basename(src_image_path).replace(".jpg", "_mask.png")), mask_np)

    if args.pred_is_image:
        vis = result._seg_logits.data
        vis = vis.cpu().numpy().clip(0, 255).astype(np.uint8).transpose(1, 2, 0)
    else:
        mask_t = result._pred_sem_seg.data
        mask_np = mask_t.cpu().numpy().astype(np.uint8)[0]
        vis = cv2.addWeighted(src_image_np, 1, get_color_mask(mask_np), 0.5, 0)
        cv2.putText(vis, "pred", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    res = np.hstack([src_image_np, vis])
    if draw_gt:
        res = np.hstack([res, vis_gt])
    cv2.imwrite(os.path.join(args.out_file, os.path.basename(src_image_path).replace(".jpg", ".png")), res)


def main():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--out-file', default=None, help='Path to output file')
    parser.add_argument('--save-pred-mask', action='store_true', help='Save predicted mask')
    parser.add_argument('--pred-is-image', action='store_true')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    parser.add_argument(
        '--server',
        action='store_true',
        default=False,
        help='Whether to use server mode')
    parser.add_argument(
        '--with-labels',
        action='store_true',
        default=False,
        help='Whether to display the class labels.')
    parser.add_argument(
        '--title', default='result', help='The image identifier.')
    args = parser.parse_args()

    if '/' not in args.checkpoint:
        config_name = os.path.basename(args.config)[:-3]
        args.checkpoint = os.path.join("work_dirs", config_name, args.checkpoint)
        print(f"给定的checkpoint不是完整路径，拓展为 {args.checkpoint}")
        time.sleep(1)
    if not args.checkpoint.endswith(".pth"):
        with open(args.checkpoint, "r") as f:
            args.checkpoint = f.read().strip()

    # 输出 pth 文件的 md5sum
    def get_md5(file_path):
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    if os.path.isfile(args.checkpoint):
        print(f"{args.checkpoint} 的 md5sum: {get_md5(args.checkpoint)}")

    os.makedirs(args.out_file, exist_ok=True)
    assert os.path.isdir(args.out_file), f"{args.out_file} is not a directory"


    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)
    if args.device == 'cpu':
        model = revert_sync_batchnorm(model)
    
    # 服务模式
    if args.server:
        app = Flask(__name__)
        
        @app.route('/infer', methods=['POST'])
        def infer():
            if 'image' not in request.files:
                return {'error': 'No image uploaded'}, 400

            file = request.files['image']
            file_bytes = file.read()
            np_arr = np.frombuffer(file_bytes, np.uint8)
            src_image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if src_image_np is None:
                return {'error': 'Invalid image'}, 400

            result = inference_model(model, src_image_np)
            mask_t = result._pred_sem_seg.data
            mask_np = mask_t.cpu().numpy().astype(np.uint8)[0]

            vis = cv2.addWeighted(src_image_np, 1, get_color_mask(mask_np), 0.5, 0)
            mask_np = cv2.cvtColor(mask_np, cv2.COLOR_GRAY2BGR)
            res = np.hstack([src_image_np, vis, mask_np])
            _, buffer = cv2.imencode('.png', res)
            return send_file(BytesIO(buffer.tobytes()), mimetype='image/png')
            
        app.run(host='0.0.0.0', port=5000)
        return
    
    src_image_paths = [args.img]
    if args.img.endswith(".txt"):
        with open(args.img, "r") as f:
            src_image_paths = [line.strip().split()[0] for line in f.readlines()]
    elif os.path.isdir(args.img):
        src_image_paths = glob.glob(os.path.join(args.img, "**", "*.png"), recursive=True) + glob.glob(os.path.join(args.img, "**", "*.jpg"), recursive=True)
        # 排除包含/labels/的图片
        src_image_paths = [path for path in src_image_paths if "/labels/" not in path]
            
    for src_image_path in tqdm(src_image_paths):
        src_image_np = cv2.imread(src_image_path)
        src_image_name = os.path.splitext(os.path.basename(src_image_path))[0]
        if src_image_np.shape[1] in [2880, 960]:
            # 横着分9份
            src_image_np_patches = []
            if src_image_np.shape[1] == 2880:
                for i in range(9):
                    src_image_np_patches.append(src_image_np[:, i*src_image_np.shape[1]//9:(i+1)*src_image_np.shape[1]//9, :])
                front_left, side_left, side_right = src_image_np_patches[0], src_image_np_patches[3], src_image_np_patches[4]
            else:
                for i in range(3):
                    for j in range(3):
                        src_image_np_patches.append(src_image_np[i*src_image_np.shape[0]//3:(i+1)*src_image_np.shape[0]//3, j*src_image_np.shape[1]//3:(j+1)*src_image_np.shape[1]//3, :])
                front_left, side_left, side_right = src_image_np_patches[0], src_image_np_patches[3], src_image_np_patches[4]
            infer_image(model, src_image_name + "_front_left.jpg", front_left, args)
            infer_image(model, src_image_name + "_side_left.jpg", side_left, args)
            infer_image(model, src_image_name + "_side_right.jpg", side_right, args)
        else:
            infer_image(model, src_image_path, src_image_np, args)

if __name__ == '__main__':
    main()
