#!/usr/bin/env python3
# Copyright (c) OpenMMLab. All rights reserved.
"""从 ``infer_onnx_segstereo.py`` 生成的五联图 ``*_vis.png`` 中裁出视差列，存为 ``*_disp.png``。

拼接顺序为：原图 | 分割彩图 | 叠加 | **灰度视差** | JET 伪彩。默认 **只导出灰度视差** 一列（无热力图）。

用法::

    python utils/stereo_matching/extract_disp_from_vis_strip.py \\
        work_dirs/.../onnx_val_output
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import cv2
import numpy as np


def _split_panels(
        strip_bgr: np.ndarray,
        num_panels: int,
        sep_w: int,
) -> list:
    """与 ``infer_onnx_segstereo._hstack_panels`` 一致：等高 ``H``，列宽相同（除 sep）。"""
    h, w = strip_bgr.shape[:2]
    if sep_w <= 0:
        # 无余白分隔：总宽 = num_panels * pw（浮点取整边界）
        pw = w / float(num_panels)
        out = []
        for i in range(num_panels):
            x0 = int(round(i * pw))
            x1 = int(round((i + 1) * pw))
            out.append(strip_bgr[:, x0:x1].copy())
        return out
    inner = w - (num_panels - 1) * sep_w
    if inner <= 0 or inner % num_panels != 0:
        raise ValueError(
            f'宽 {w}、{num_panels} 列、sep={sep_w} 无法整除列宽，请检查 --sep 或手工指定')
    pw = inner // num_panels
    out = []
    x = 0
    for i in range(num_panels):
        out.append(strip_bgr[:, x:x + pw].copy())
        x += pw
        if i < num_panels - 1:
            x += sep_w
    return out


def main():
    p = argparse.ArgumentParser(description='从 *_vis.png 裁出视差列 → *_disp.png')
    p.add_argument('vis_dir', type=str, help='含 *_vis.png 的目录')
    p.add_argument(
        '--sep',
        type=int,
        default=0,
        help='列间白边宽度（像素），与推理时 --vis-sep 一致，默认 0')
    p.add_argument(
        '--num-panels',
        type=int,
        default=5,
        help='拼接总列数，默认 5（seg+disp 全流程图）')
    p.add_argument(
        '--with-jet',
        action='store_true',
        help='附加拼接 JET 伪彩列（灰度|JET）；默认仅灰度视差')
    p.add_argument(
        '--suffix-in',
        type=str,
        default='_vis.png',
        help='输入文件名后缀')
    p.add_argument(
        '--suffix-out',
        type=str,
        default='_disp.png',
        help='输出文件名后缀')
    args = p.parse_args()

    pattern = os.path.join(args.vis_dir, f'*{args.suffix_in}')
    paths = sorted(glob.glob(pattern))
    if not paths:
        print(f'未找到: {pattern}', file=sys.stderr)
        sys.exit(1)

    n_out = 0
    for path in paths:
        img = cv2.imread(path)
        if img is None:
            print(f'跳过（无法读图）: {path}', file=sys.stderr)
            continue
        try:
            panels = _split_panels(img, args.num_panels, args.sep)
        except ValueError as e:
            print(f'{path}: {e}', file=sys.stderr)
            sys.exit(1)
        if args.with_jet:
            disp = np.hstack([panels[-2], panels[-1]])
        else:
            disp = panels[-2]
        base = os.path.basename(path)
        if not base.endswith(args.suffix_in):
            continue
        stem = base[: -len(args.suffix_in)]
        out_path = os.path.join(args.vis_dir, stem + args.suffix_out)
        cv2.imwrite(out_path, disp)
        n_out += 1
    print(f'已写入 {n_out} 个 *{args.suffix_out} 至 {os.path.abspath(args.vis_dir)}')


if __name__ == '__main__':
    main()
