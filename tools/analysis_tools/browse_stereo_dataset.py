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
from mmseg.utils.class_names import get_palette
from mmseg.datasets.ld_perception_stereo_matching_dataset import LDPerceptionStereoMatchingDataset

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
        args.output_dir = args.output_dir.rstrip("/")
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # register all modules in mmdet into the registries
    register_all_modules()

    print("注意可视化的是训练数据集，不是验证数据集")
    time.sleep(1)
    dataset = DATASETS.build(cfg.train_dataloader.dataset)
    visualizer = VISUALIZERS.build(cfg.visualizer)
    visualizer.dataset_meta = dataset.metainfo

    progress_bar = ProgressBar(len(dataset))

    indexes = list(range(len(dataset)))
    if not args.not_random:
        random.shuffle(indexes)
    
    dataset.data_root = dataset.data_root.rstrip("/")
    assert isinstance(dataset, LDPerceptionStereoMatchingDataset), "必须是双目数据集"
    for index in indexes:
        item = dataset[index]
        img = item['inputs'].permute(1, 2, 0).numpy()
        data_sample = item['data_samples'].numpy()
        left_image_path = item['data_samples'].img_path
        src_image_name = osp.basename(left_image_path)
        right_image_path = left_image_path.replace("left", "right")

        out_file = osp.join(
            args.output_dir,
            src_image_name) if args.output_dir is not None else None

        if args.create_augmented_dataset:
            src_image_np = img
            src_image_np = src_image_np[..., ::-1]
            src_gt_np = data_sample.gt_sem_seg.data.astype(np.uint8)[0]
            # if hasattr(dataset, "reverse_label_map"):
            #     src_gt_np = np.vectorize(dataset.reverse_label_map.get)(src_gt_np)

            assert dataset.data_root in src_image_path, src_image_path
            dst_image_path = src_image_path.replace(dataset.data_root, args.output_dir)
            assert "/images/" in dst_image_path and ".jpg" in dst_image_path, dst_image_path
            dst_gt_path = dst_image_path.replace("/images/", "/labels/").replace(".jpg", ".png")
            os.makedirs(osp.dirname(dst_image_path), exist_ok=True)
            os.makedirs(osp.dirname(dst_gt_path), exist_ok=True)
            cv2.imwrite(dst_image_path, src_image_np)
            cv2.imwrite(dst_gt_path, src_gt_np)
            
        else:
            src_image_np = img
            gt_np = data_sample.gt_sem_seg.data.astype(np.uint8)[0]

            # 将src_image_np的最后一个轴拆分为left_image_np和right_image_np
            left_image_np = src_image_np[..., 0]
            right_image_np = src_image_np[..., 1]

            # 将单通道灰度图转为3通道以便可视化
            left_image_np_vis = cv2.cvtColor(left_image_np, cv2.COLOR_GRAY2BGR)
            right_image_np_vis = cv2.cvtColor(right_image_np, cv2.COLOR_GRAY2BGR)

            # 用jet色图可视化gt_np
            gt_norm = cv2.normalize(gt_np.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX)
            gt_color_np = cv2.applyColorMap(gt_norm.astype(np.uint8), cv2.COLORMAP_JET)

            # 拼接三张图像
            vis_np = np.concatenate([left_image_np_vis, right_image_np_vis, gt_color_np], axis=1)
            
            cv2.imwrite(out_file, vis_np)
        progress_bar.update()


if __name__ == '__main__':
    main()
