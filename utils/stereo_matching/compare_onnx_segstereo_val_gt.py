#!/usr/bin/env python3
# Copyright (c) OpenMMLab. All rights reserved.
"""SegStereo ONNX 与验证集真值对比：视差 EPE / Bad-X + 分割 mIoU，并输出简要分析。

数据目录须与 ``LDPerceptionStereoSegDataset`` 一致。支持两种布局：

- **扁平**：``data_root/images``、``images_right``、``disparity``、``labels`` 同级；
- **多批次子目录**：``data_root/<batch>/images/*.jpg``，右图/视差/标签在同名 ``<batch>`` 下。

示例::

    python utils/stereo_matching/compare_onnx_segstereo_val_gt.py \\
        work_dirs/.../epoch_70.onnx /data_SSD2/datasets/segstereo_val \\
        --input-h 272 --input-w 320 --mean '0 0 0' --std '255 255 255'
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import infer_onnx_segstereo as inf


def _parse_float_list(s: str, n: int) -> list:
    parts = [float(x) for x in s.replace(',', ' ').split()]
    if len(parts) != n:
        raise ValueError(f'需要 {n} 个数: {s!r}')
    return parts


def _load_gt_disp_png(path: str) -> np.ndarray:
    """与 ``LoadStereoMatchingAnnotations`` 中 ``.png`` 分支一致：灰度 uint8 → float。"""
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if g is None:
        raise FileNotFoundError(path)
    return np.abs(g.astype(np.float32))


def _remap_seg(gt_u8: np.ndarray, mapping: Dict[int, int],
               ignore: int) -> np.ndarray:
    out = np.full(gt_u8.shape, ignore, dtype=np.int32)
    for raw, tid in mapping.items():
        out[gt_u8 == raw] = tid
    return out


def _collect_left_paths(data_root: str, max_samples: int) -> List[str]:
    """``data_root/images/*.jpg`` 或 ``data_root/**/images/*.jpg``（多批次子目录）。"""
    flat_img = os.path.join(data_root, 'images')
    paths: List[str] = []
    if os.path.isdir(flat_img):
        paths = sorted(glob.glob(os.path.join(flat_img, '*.jpg')))
    if not paths:
        paths = sorted(
            glob.glob(os.path.join(data_root, '**', 'images', '*.jpg'),
                     recursive=True))
    if max_samples > 0:
        paths = paths[:max_samples]
    return paths


def _stereo_paths_from_left(left_path: str) -> Tuple[str, str, str]:
    """由左图路径得到同批次下的右图、视差、语义标注路径。"""
    img_dir = os.path.dirname(left_path)
    batch_root = os.path.dirname(img_dir)
    base = os.path.basename(left_path)
    stem = os.path.splitext(base)[0]
    rp = os.path.join(batch_root, 'images_right', base)
    dp = os.path.join(batch_root, 'disparity', stem + '.png')
    sp = os.path.join(batch_root, 'labels', stem + '.png')
    return rp, dp, sp


def _is_disp_arr(arr: np.ndarray) -> bool:
    if arr.ndim == 4:
        return arr.shape[1] == 1
    if arr.ndim == 3:
        return arr.shape[0] == 1
    return False


def _run_onnx(
    session: ort.InferenceSession,
    inps,
    outs,
    left_bgr: np.ndarray,
    right_bgr: np.ndarray,
    h: int,
    w: int,
    mean: np.ndarray,
    std: np.ndarray,
    bgr_to_rgb: bool,
    swap_outputs: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """返回 (pred_disp 2D float, pred_seg_logits NCHW 或 seg 已 argmax — 返回 logits)。"""
    x6, _ = inf.preprocess_pair(left_bgr, right_bgr, h, w, mean, std, bgr_to_rgb)
    ort_inputs = {}
    if len(inps) == 1:
        ort_inputs[inps[0].name] = x6
    elif len(inps) == 2:
        ort_inputs[inps[0].name] = x6[:, :3]
        ort_inputs[inps[1].name] = x6[:, 3:6]
    else:
        raise RuntimeError(f'不支持的 ONNX 输入数: {len(inps)}')
    names = [o.name for o in outs]
    ort_outs = session.run(names, ort_inputs)
    a, b = ort_outs[0], ort_outs[1]
    if swap_outputs:
        a, b = b, a
    if _is_disp_arr(a) and not _is_disp_arr(b):
        disp_np, seg_np = a, b
    elif _is_disp_arr(b) and not _is_disp_arr(a):
        seg_np, disp_np = a, b
    else:
        seg_np, disp_np = a, b
    disp_2d = np.squeeze(disp_np).astype(np.float32)
    seg_logits = seg_np
    if seg_logits.ndim == 4:
        pass
    else:
        raise ValueError(f'期望 seg logits (1,C,H,W)，得到 {seg_logits.shape}')
    return disp_2d, seg_logits


def _disp_accumulate(
    pred: np.ndarray,
    gt: np.ndarray,
    max_disp: float,
    acc: Optional[dict] = None,
) -> dict:
    """双线性对齐 GT 尺寸后，在 valid=(gt>0)&(gt<max_disp) 上累计与 DisparityMetric 一致的量。"""
    if acc is None:
        acc = {
            'n': 0,
            'abs_sum': 0.0,
            'sq_sum': 0.0,
            'bad': [0, 0, 0],
            'thresholds': (1.0, 3.0, 5.0),
        }
    h, w = gt.shape
    pr = cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR).astype(
        np.float64)
    g = gt.astype(np.float64)
    valid = (g > 0) & (g < float(max_disp))
    if not valid.any():
        return acc
    err = np.abs(pr - g)
    vn = valid
    acc['n'] += int(vn.sum())
    acc['abs_sum'] += float(err[vn].sum())
    acc['sq_sum'] += float((err[vn]**2).sum())
    for i, t in enumerate(acc['thresholds']):
        acc['bad'][i] += int((err[vn] > t).sum())
    return acc


def _disp_finalize(acc: dict) -> Dict[str, float]:
    n = acc['n']
    if n == 0:
        return {}
    epe = acc['abs_sum'] / n
    rms = (acc['sq_sum'] / n) ** 0.5
    out = {'epe': epe, 'rms': rms, 'valid_pixels': n}
    for i, t in enumerate(acc['thresholds']):
        key = f'bad_{int(t)}' if abs(t - round(t)) < 1e-6 else f'bad_{t}'
        out[key] = 100.0 * acc['bad'][i] / n
    return out


def _seg_iou_accumulate(
    logits: np.ndarray,
    gt: np.ndarray,
    num_classes: int,
    ignore: int,
    inter: np.ndarray,
    union: np.ndarray,
) -> None:
    """logits (1,C,h,w)；将 argmax 最近邻放大到 gt 尺寸后在 valid 上累计 inter/union。"""
    log = logits[0] if logits.ndim == 4 else logits
    pred_small = np.argmax(log, axis=0).astype(np.int32)
    h, w = gt.shape
    pred = cv2.resize(
        pred_small.astype(np.uint8),
        (w, h),
        interpolation=cv2.INTER_NEAREST).astype(np.int32)
    valid = gt != ignore
    if not valid.any():
        return
    for c in range(num_classes):
        pc = (pred == c) & valid
        gc = (gt == c) & valid
        inter[c] += int((pc & gc).sum())
        union[c] += int((pc | gc).sum())


def _seg_miou(inter: np.ndarray, union: np.ndarray,
              class_names: Tuple[str, ...]) -> Tuple[float, List[float]]:
    ious = []
    for c in range(len(class_names)):
        if union[c] == 0:
            ious.append(float('nan'))
        else:
            ious.append(inter[c] / union[c])
    miou = float(np.nanmean(ious))
    return miou, ious


def main():
    p = argparse.ArgumentParser(
        description='ONNX SegStereo 与 val GT 对比（视差 + 分割）')
    p.add_argument('onnx_path', type=str)
    p.add_argument(
        'data_root',
        type=str,
        help='含 images、images_right、disparity、labels')
    p.add_argument('--input-h', type=int, default=272)
    p.add_argument('--input-w', type=int, default=320)
    p.add_argument('--mean', type=str, default='0 0 0')
    p.add_argument('--std', type=str, default='255 255 255')
    p.add_argument('--no-bgr-to-rgb', action='store_true')
    p.add_argument('--swap-outputs', action='store_true')
    p.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
    p.add_argument('--max-disp', type=float, default=192.0)
    p.add_argument(
        '--class-names',
        type=str,
        default='background,grass,soil,animal',
        help='逗号分隔，与训练一致')
    p.add_argument(
        '--label-map',
        type=str,
        default='1:0,50:1,200:2,250:3',
        help='原始标签 id:train_id，逗号分隔')
    p.add_argument('--ignore-label', type=int, default=255)
    p.add_argument('--max-samples', type=int, default=0, help='0 表示全部')
    p.add_argument('--json-out', type=str, default=None)
    args = p.parse_args()

    mean = np.array(_parse_float_list(args.mean, 3), np.float32)
    std = np.array(_parse_float_list(args.std, 3), np.float32)
    bgr_to_rgb = not args.no_bgr_to_rgb

    label_map = {}
    for part in args.label_map.split(','):
        part = part.strip()
        if not part:
            continue
        a, b = part.split(':')
        label_map[int(a.strip())] = int(b.strip())

    class_names = tuple(x.strip() for x in args.class_names.split(',') if x.strip())
    num_classes = len(class_names)

    if not os.path.isdir(args.data_root):
        print(f'数据根目录不存在: {args.data_root}', file=sys.stderr)
        sys.exit(1)
    left_dir = os.path.join(args.data_root, 'images')
    right_dir = os.path.join(args.data_root, 'images_right')
    disp_dir = os.path.join(args.data_root, 'disparity')
    lab_dir = os.path.join(args.data_root, 'labels')
    if os.path.isdir(left_dir):
        for d in (right_dir, disp_dir, lab_dir):
            if not os.path.isdir(d):
                print(f'缺少目录: {d}', file=sys.stderr)
                sys.exit(1)

    session = inf._build_session(args.onnx_path, args.device)
    inps = session.get_inputs()
    outs = session.get_outputs()

    left_paths = _collect_left_paths(args.data_root, args.max_samples)

    disp_acc = {
        'n': 0,
        'abs_sum': 0.0,
        'sq_sum': 0.0,
        'bad': [0, 0, 0],
        'thresholds': (1.0, 3.0, 5.0),
    }
    inter = np.zeros(num_classes, dtype=np.int64)
    union = np.zeros(num_classes, dtype=np.int64)
    n_ok = 0
    n_skip = 0
    skip_reasons: List[str] = []

    if not left_paths:
        print(f'未找到左图: {args.data_root} 下 images 或 **/images', file=sys.stderr)
        sys.exit(1)

    for lp in left_paths:
        stem = os.path.splitext(os.path.basename(lp))[0]
        rp, dp, sp = _stereo_paths_from_left(lp)
        if not os.path.isfile(rp):
            n_skip += 1
            skip_reasons.append(f'{stem}: 缺右图')
            continue
        if not os.path.isfile(dp) or not os.path.isfile(sp):
            n_skip += 1
            skip_reasons.append(f'{stem}: 缺 disp 或 label')
            continue
        left_bgr = cv2.imread(lp)
        right_bgr = cv2.imread(rp)
        if left_bgr is None or right_bgr is None:
            n_skip += 1
            continue
        try:
            pred_disp, seg_logits = _run_onnx(
                session,
                inps,
                outs,
                left_bgr,
                right_bgr,
                args.input_h,
                args.input_w,
                mean,
                std,
                bgr_to_rgb,
                args.swap_outputs,
            )
            gt_disp = _load_gt_disp_png(dp)
            gt_raw = cv2.imread(sp, cv2.IMREAD_UNCHANGED)
            if gt_raw is None:
                n_skip += 1
                continue
            if gt_raw.ndim == 3:
                gt_raw = gt_raw[:, :, 0]
            gt_seg = _remap_seg(gt_raw, label_map, args.ignore_label)
        except Exception as e:
            n_skip += 1
            skip_reasons.append(f'{stem}: {e}')
            continue

        if gt_disp.shape != gt_seg.shape:
            n_skip += 1
            skip_reasons.append(f'{stem}: disp/label 尺寸不一致')
            continue

        _disp_accumulate(pred_disp, gt_disp, args.max_disp, disp_acc)
        _seg_iou_accumulate(seg_logits, gt_seg, num_classes, args.ignore_label,
                            inter, union)
        n_ok += 1

    disp_metrics = _disp_finalize(disp_acc)
    miou, ious = _seg_miou(inter, union, class_names)

    lines = []
    lines.append('=' * 60)
    lines.append(f'SegStereo ONNX vs GT  （样本数 {n_ok}，跳过 {n_skip}）')
    lines.append('=' * 60)
    if disp_metrics:
        lines.append('[双目视差] 有效像素内（GT: 0 < d < max_disp）与训练 DisparityMetric 一致：')
        lines.append(
            f"  EPE (px): {disp_metrics['epe']:.4f}   RMS (px): {disp_metrics['rms']:.4f}"
        )
        for k in sorted(disp_metrics.keys()):
            if k.startswith('bad_'):
                lines.append(f"  {k} (%): {disp_metrics[k]:.4f}")
        lines.append(
            f"  说明：EPE 为平均绝对误差；bad_k 为误差 > k 像素的有效点比例。"
        )
    else:
        lines.append('[双目视差] 无有效 GT 像素，未统计。')

    lines.append('[语义分割] 全局 IoU（忽略 label=255）：')
    lines.append(f'  mIoU: {miou:.4f}')
    for name, iou in zip(class_names, ious):
        iv = f'{iou:.4f}' if iou == iou else 'nan'
        lines.append(f'    {name}: {iv}')
    lines.append('=' * 60)

    # 简要文字分析
    analysis = []
    analysis.append('【分析摘要】')
    if disp_metrics:
        epe = disp_metrics['epe']
        b3 = disp_metrics.get('bad_3', 100.0)
        if epe < 5:
            analysis.append(
                '- 视差：EPE 较低，整体上深度/视差回归与真值较接近。')
        elif epe < 15:
            analysis.append(
                '- 视差：EPE 中等，草地纹理、遮挡或标定误差可能导致局部偏差。')
        else:
            analysis.append(
                '- 视差：EPE 偏高，模型在部分区域与真值差异大，可结合 bad_3 看大误差占比。')
        if b3 < 20:
            analysis.append(
                f'- 视差：Bad-3 为 {b3:.1f}% ，多数有效像素误差在 3px 内。')
        elif b3 < 50:
            analysis.append(
                f'- 视差：Bad-3 为 {b3:.1f}% ，约一半有效点存在明显视差误差。')
        else:
            analysis.append(
                f'- 视差：Bad-3 为 {b3:.1f}% ，大误差像素占比较高，双目匹配难度或标注噪声需关注。')
    if miou == miou:
        if miou >= 0.75:
            analysis.append('- 分割：mIoU 较高，语义与真值整体一致性好。')
        elif miou >= 0.5:
            analysis.append('- 分割：mIoU 中等，部分类别边界或易混类仍有提升空间。')
        else:
            analysis.append('- 分割：mIoU 偏低，建议检查类别映射与难例。')
    lines.extend([''] + analysis)

    report = '\n'.join(lines)
    print(report)

    if args.json_out:
        payload = {
            'n_samples': n_ok,
            'n_skip': n_skip,
            'disparity': disp_metrics,
            'segmentation': {
                'miou': miou,
                'per_class_iou': {
                    n: (float(i) if i == i else None)
                    for n, i in zip(class_names, ious)
                },
            },
        }
        with open(args.json_out, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f'已写 JSON: {args.json_out}')

    if skip_reasons and len(skip_reasons) <= 20:
        print('\n跳过明细:', file=sys.stderr)
        for s in skip_reasons:
            print(' ', s, file=sys.stderr)
    elif skip_reasons:
        print(f'\n跳过 {len(skip_reasons)} 条（仅列前 10）:', file=sys.stderr)
        for s in skip_reasons[:10]:
            print(' ', s, file=sys.stderr)


if __name__ == '__main__':
    main()
