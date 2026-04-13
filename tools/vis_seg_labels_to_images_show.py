#!/usr/bin/env python3
# Copyright (c) OpenMMLab. All rights reserved.
"""将 ``images`` 与 ``labels`` 拼成「原图 | 彩色 GT | 叠加」保存到 ``images_show``。

默认调色板与 ``LDPerceptionSegDataset.PALETTE``（RGB）及 grass 四分类一致；255 为 ignore，显示为灰色。
"""

from __future__ import annotations

import argparse
import os
import os.path as osp

import cv2
import numpy as np

# 与 mmseg/datasets/ld_perception_seg_dataset.py PALETTE 一致（RGB）
_PALETTE_RGB = np.asarray(
    [
        [128, 0, 128],
        [0, 255, 0],
        [0, 255, 255],
        [0, 0, 255],
    ],
    dtype=np.uint8,
)


def _label_to_color_bgr(lab: np.ndarray, num_classes: int, pal_rgb: np.ndarray) -> np.ndarray:
    k = pal_rgb.shape[0]
    out = np.zeros((*lab.shape, 3), dtype=np.uint8)
    for c in range(min(num_classes, k)):
        m = lab == c
        if not np.any(m):
            continue
        bgr = cv2.cvtColor(pal_rgb[c : c + 1, None, :], cv2.COLOR_RGB2BGR)[0, 0]
        out[m] = bgr
    m255 = lab == 255
    if np.any(m255):
        out[m255] = (64, 64, 64)
    return out


def _hstack_panels(panels: list, sep_w: int) -> np.ndarray:
    oh = panels[0].shape[0]
    if sep_w <= 0:
        return np.hstack(panels)
    sep = np.full((oh, sep_w, 3), 255, dtype=np.uint8)
    out = panels[0]
    for p in panels[1:]:
        out = np.hstack([out, sep, p])
    return out


def main():
    parser = argparse.ArgumentParser(
        description='images + labels -> images_show 拼图可视化')
    parser.add_argument(
        '--base',
        type=str,
        default='/data_SSD2/datasets/segstereo/20260408',
        help='含 images/、labels/ 的数据目录')
    parser.add_argument(
        '--out-subdir',
        type=str,
        default='images_show',
        help='输出子目录名（在 base 下）')
    parser.add_argument(
        '--overlay-alpha',
        type=float,
        default=0.5,
        help='叠加列：彩色 GT 权重（默认 0.5）')
    parser.add_argument(
        '--sep',
        type=int,
        default=4,
        help='列间白边宽度，0 表示无')
    parser.add_argument(
        '--num-classes',
        type=int,
        default=4,
        help='语义类别数（与标注 0~C-1 一致）')
    args = parser.parse_args()

    base = osp.abspath(osp.expanduser(args.base))
    img_dir = osp.join(base, 'images')
    lab_dir = osp.join(base, 'labels')
    out_dir = osp.join(base, args.out_subdir.strip() or 'images_show')
    os.makedirs(out_dir, exist_ok=True)

    pal = _PALETTE_RGB[: args.num_classes]
    alpha = float(np.clip(args.overlay_alpha, 0.0, 1.0))
    sep = max(0, int(args.sep))

    exts = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG', '.bmp')
    files = sorted(
        f for f in os.listdir(img_dir)
        if f.lower().endswith(exts))
    n_ok = 0
    for fn in files:
        stem = osp.splitext(fn)[0]
        ip = osp.join(img_dir, fn)
        lp = osp.join(lab_dir, stem + '.png')
        if not osp.isfile(lp):
            print(f'跳过（无 label）: {fn}')
            continue
        bgr = cv2.imread(ip, cv2.IMREAD_COLOR)
        if bgr is None:
            print(f'跳过（读图失败）: {ip}')
            continue
        lab = cv2.imread(lp, cv2.IMREAD_UNCHANGED)
        if lab is None:
            print(f'跳过（读 label 失败）: {lp}')
            continue
        if lab.ndim == 3:
            lab = lab[:, :, 0]
        if lab.shape[:2] != bgr.shape[:2]:
            lab = cv2.resize(
                lab,
                (bgr.shape[1], bgr.shape[0]),
                interpolation=cv2.INTER_NEAREST)
        color = _label_to_color_bgr(lab, args.num_classes, pal)
        blend = cv2.addWeighted(bgr, 1.0 - alpha, color, alpha, 0)
        strip = _hstack_panels([bgr, color, blend], sep)
        out_path = osp.join(out_dir, f'{stem}_vis.png')
        cv2.imwrite(out_path, strip)
        n_ok += 1

    print(f'完成：{n_ok} 张 -> {out_dir}')


if __name__ == '__main__':
    main()
