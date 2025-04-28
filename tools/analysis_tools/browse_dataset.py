# Copyright (c) OpenMMLab. All rights reserved.
import cv2
import numpy as np
import time
import os
import argparse
import os.path as osp
import random
from mmengine.config import Config, DictAction
from mmengine.utils import ProgressBar

from mmseg.registry import DATASETS, VISUALIZERS
from mmseg.utils import register_all_modules


def parse_args():
    parser = argparse.ArgumentParser(description='Browse a dataset')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--output-dir',
        default=None,
        type=str,
        help='If there is no display interface, you can save it')
    parser.add_argument('--not-show', default=False, action='store_true')
    parser.add_argument(
        '--show-interval',
        type=float,
        default=2,
        help='the interval of show (s)')
    parser.add_argument("--not_random", default=False, action='store_true')
    parser.add_argument("--create_augmented_dataset", default=False, action='store_true')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # register all modules in mmdet into the registries
    register_all_modules()

    dataset = DATASETS.build(cfg.train_dataloader.dataset)
    visualizer = VISUALIZERS.build(cfg.visualizer)
    visualizer.dataset_meta = dataset.metainfo

    progress_bar = ProgressBar(len(dataset))

    indexes = list(range(len(dataset)))
    if not args.not_random:
        random.shuffle(indexes)
    
    for index in indexes:
        item = dataset[index]
        img = item['inputs'].permute(1, 2, 0).numpy()
        img = img[..., [2, 1, 0]]  # bgr to rgb
        data_sample = item['data_samples'].numpy()
        src_image_name = osp.basename(item['data_samples'].img_path)

        out_file = osp.join(
            args.output_dir,
            src_image_name) if args.output_dir is not None else None

        if args.create_augmented_dataset:
            src_image_np = img
            src_image_np = src_image_np[..., ::-1]
            src_gt_np = data_sample.gt_sem_seg.data.astype(np.uint8)[0]
            if hasattr(dataset, "reverse_label_map"):
                src_gt_np = np.vectorize(dataset.reverse_label_map.get)(src_gt_np)

            dst_image_path = osp.join(args.output_dir, "images", src_image_name)
            dst_gt_path = osp.join(args.output_dir, "labels", src_image_name).replace(".jpg", ".png")
            os.makedirs(osp.dirname(dst_image_path), exist_ok=True)
            os.makedirs(osp.dirname(dst_gt_path), exist_ok=True)
            cv2.imwrite(dst_image_path, src_image_np)
            cv2.imwrite(dst_gt_path, src_gt_np)
            
        else:
            visualizer.add_datasample(
                name=osp.basename(src_image_name),
                image=img,
                data_sample=data_sample,
                draw_gt=True,
                draw_pred=False,
                wait_time=args.show_interval,
                out_file=out_file,
                show=False)
        progress_bar.update()


if __name__ == '__main__':
    main()
