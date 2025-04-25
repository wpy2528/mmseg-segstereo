# Copyright (c) OpenMMLab. All rights reserved.
from tqdm import tqdm
import os
import cv2
import numpy as np
from argparse import ArgumentParser

from mmengine.model import revert_sync_batchnorm
from mmengine.structures import PixelData
from mmseg.apis import inference_model, init_model, show_result_pyplot


def main():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--out-file', default=None, help='Path to output file')
    parser.add_argument('--save-pred-mask', action='store_true', help='Save predicted mask')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    parser.add_argument(
        '--with-labels',
        action='store_true',
        default=False,
        help='Whether to display the class labels.')
    parser.add_argument(
        '--title', default='result', help='The image identifier.')
    args = parser.parse_args()

    if not args.checkpoint.endswith(".pth"):
        with open(args.checkpoint, "r") as f:
            args.checkpoint = f.read().strip()

    os.makedirs(args.out_file, exist_ok=True)
    assert os.path.isdir(args.out_file), f"{args.out_file} is not a directory"


    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)
    if args.device == 'cpu':
        model = revert_sync_batchnorm(model)
    
    src_image_paths = [args.img]
    if args.img.endswith(".txt"):
        with open(args.img, "r") as f:
            src_image_paths = [line.strip() for line in f.readlines()]
            
    for src_image_path in tqdm(src_image_paths):
        result = inference_model(model, src_image_path)
        gt_image_path = src_image_path.replace("/images/", "/labels/").replace(".jpg", ".png")
        if (gt_image_path != src_image_path and os.path.exists(gt_image_path)):
            draw_gt = True
            gt_image = cv2.imread(gt_image_path, cv2.IMREAD_GRAYSCALE).astype(np.int64)[None, ...]
            result.gt_sem_seg = PixelData(data=gt_image)
        else:
            draw_gt = False
        # 保存预测mask
        if args.save_pred_mask:
            mask_t = result._pred_sem_seg.data
            mask_np = mask_t.cpu().numpy()
            mask_np = mask_t.astype(np.uint8)
            cv2.imwrite(os.path.join(args.out_file, os.path.basename(src_image_path).replace(".jpg", ".png")), mask_np)
        # show the results
        show_result_pyplot(
            model,
            src_image_path,
            result,
            title=args.title,
            opacity=args.opacity,
            with_labels=args.with_labels,
            draw_gt=True,
            show=False if args.out_file is not None else True,
            out_file=os.path.join(args.out_file, os.path.basename(src_image_path)))

if __name__ == '__main__':
    main()
