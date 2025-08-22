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


def infer_image(model, left_image_path, args):
    right_image_path = left_image_path.replace("left", "right")
    left_image_np = cv2.imread(left_image_path)
    right_image_np = cv2.imread(right_image_path)
    result = dict(left_img_path=left_image_path, right_img_path=right_image_path)
    gt_image_path = left_image_path.replace("/images/", "/labels/").replace(".jpg", ".png")
    result = inference_model(model, result)
    # if os.path.exists(gt_image_path):
    #     result['left_disp_path'] = gt_image_path
    if (gt_image_path != left_image_path and os.path.exists(gt_image_path)):
        draw_gt = True
        gt_image = cv2.imread(gt_image_path, cv2.IMREAD_GRAYSCALE)
        vis_gt = cv2.addWeighted(left_image_np, 1, get_color_mask(gt_image), 0.5, 0)
        cv2.putText(vis_gt, "gt", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        result.gt_sem_seg = PixelData(data=gt_image)
    else:
        draw_gt = False

    if args.pred_is_image:
        vis = result._seg_logits.data
        vis = vis.cpu().numpy().clip(0, 255).astype(np.uint8).transpose(1, 2, 0)
    else:
        mask_t = result._pred_disp.data
        mask_np = mask_t.cpu().numpy()[0]
        # 归一化到0-255
        disp_min = np.min(mask_np)
        disp_max = np.max(mask_np)
        disp_normalized = ((mask_np - disp_min) / (disp_max - disp_min) * 255).astype(np.uint8)
        
        # 应用颜色映射
        vis_np = cv2.applyColorMap(disp_normalized, cv2.COLORMAP_JET)
        cv2.putText(vis_np, "pred", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    res = np.hstack([left_image_np, right_image_np, vis_np])
    if args.save_pred_mask:
        mask_t = result._pred_sem_seg.data
        mask_np = mask_t.cpu().numpy().astype(np.uint8)[0]
        mask_np = cv2.cvtColor(mask_np, cv2.COLOR_GRAY2BGR)
        res = np.hstack([res, mask_np])
    postfix = ".jpg"
    if draw_gt:
        res = np.hstack([res, vis_gt])
        postfix = ".png"
    cv2.imwrite(os.path.join(args.out_file, os.path.basename(left_image_path).replace(".jpg", postfix)), res)


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
    parser.add_argument("--port", type=int, default=5000, help='The port of the server.')
    args = parser.parse_args()

    if '/' not in args.checkpoint:
        args.checkpoint = os.path.join("work_dirs", os.path.splitext(args.config.split("configs/")[-1])[0], args.checkpoint)
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
            
        app.run(host='0.0.0.0', port=args.port)
        return
    
    left_image_paths = [args.img]
    if args.img.endswith(".txt"):
        with open(args.img, "r") as f:
            left_image_paths = [line.strip().split()[0] for line in f.readlines()]
    elif os.path.isdir(args.img):
        if "x5_stereo" in args.img:
            left_image_paths = glob.glob(os.path.join(args.img, "**", "left*.png"), recursive=True)
        else:
            left_image_paths = glob.glob(os.path.join(args.img, "**", "*.png"), recursive=True) + glob.glob(os.path.join(args.img, "**", "*.jpg"), recursive=True) + glob.glob(os.path.join(args.img, "**", "*.bmp"), recursive=True)
        # 排除包含/labels/的图片
        left_image_paths = [path for path in left_image_paths if "/labels/" not in path]
            
    for left_image_path in tqdm(left_image_paths):
        infer_image(model, left_image_path, args)

if __name__ == '__main__':
    main()
