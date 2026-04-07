#!/usr/bin/env python3
# Copyright (c) OpenMMLab. All rights reserved.
"""将已对齐的 SegStereo 数据按随机比例划分为训练目录与验证目录。

默认 **8:2**：约 80% 样本保留在 ``train_root``，约 20% **移动** 到 ``val_root``（四模态各一份）。

目录结构（两侧一致）::

    train_root/images/*.jpg
    train_root/images_right/*.jpg
    train_root/disparity/*.png
    train_root/labels/*.png

``val_root`` 下子目录名相同。划分使用固定随机种子，便于复现。

验证集配置示例：将 ``val_dataloader.dataset.data_root`` 设为 ``val_root`` 的父目录，并用
``include=['/segstereo_val/']``（路径中需包含该片段）；或单独把 ``data_root`` 指到 ``val_root``。

用法::

    python utils/stereo_matching/split_segstereo_train_val.py \\
        /data_SSD2/datasets/segstereo --val-root /data_SSD2/datasets/segstereo_val --dry-run

    python utils/stereo_matching/split_segstereo_train_val.py \\
        /data_SSD2/datasets/segstereo --val-root /data_SSD2/datasets/segstereo_val --seed 42
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

SPECS = [
    ('images', '.jpg'),
    ('images_right', '.jpg'),
    ('disparity', '.png'),
    ('labels', '.png'),
]


def _common_stems(root: Path) -> list:
    sets = []
    for sub, ext in SPECS:
        d = root / sub
        if not d.is_dir():
            print(f'缺少目录: {d}', file=sys.stderr)
            sys.exit(1)
        sets.append({p.stem for p in d.glob(f'*{ext}')})
    common = sets[0]
    for s in sets[1:]:
        common &= s
    return sorted(common)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        'train_root',
        type=str,
        help='当前训练数据根目录（含 images / images_right / disparity / labels）')
    p.add_argument(
        '--val-root',
        type=str,
        default=None,
        help='验证集根目录，默认与 train_root 同级的 segstereo_val')
    p.add_argument(
        '--val-ratio',
        type=float,
        default=0.2,
        help='验证集比例，默认 0.2（八二开中「二」）')
    p.add_argument('--seed', type=int, default=42, help='随机种子')
    p.add_argument(
        '--dry-run',
        action='store_true',
        help='只打印划分数量与示例，不移动文件')
    p.add_argument(
        '--manifest',
        type=str,
        default=None,
        help='可选：将验证集主名列表写入该 txt（每行一个 stem）')
    args = p.parse_args()

    train_root = Path(args.train_root).resolve()
    val_root = Path(args.val_root).resolve() if args.val_root else (
        train_root.parent / 'segstereo_val')

    stems = _common_stems(train_root)
    n = len(stems)
    if n == 0:
        print('无有效交集样本', file=sys.stderr)
        sys.exit(1)
    if n < 2:
        print('至少需要 2 个交集样本才能划分训练/验证', file=sys.stderr)
        sys.exit(1)

    n_val = round(n * args.val_ratio)
    n_val = max(1, min(n_val, n - 1))
    rng = random.Random(args.seed)
    rng.shuffle(stems)
    val_stems = set(stems[:n_val])
    train_stems = set(stems[n_val:])

    print(f'总样本数: {n}')
    print(f'训练集: {len(train_stems)}  验证集: {len(val_stems)}  (val_ratio={args.val_ratio}, seed={args.seed})')
    print(f'train_root: {train_root}')
    print(f'val_root:   {val_root}')

    if args.dry_run:
        print('验证集主名示例:', sorted(val_stems)[:min(10, len(val_stems))])
        return

    for sub, ext in SPECS:
        (val_root / sub).mkdir(parents=True, exist_ok=True)

    moved = 0
    for stem in sorted(val_stems):
        for sub, ext in SPECS:
            src = train_root / sub / f'{stem}{ext}'
            dst = val_root / sub / f'{stem}{ext}'
            if not src.is_file():
                print(f'缺失: {src}', file=sys.stderr)
                sys.exit(1)
            shutil.move(str(src), str(dst))
            moved += 1

    print(f'已移动 {moved} 个文件（{len(val_stems)} 个样本 × 4 模态）到 {val_root}')

    if args.manifest:
        mp = Path(args.manifest)
        mp.parent.mkdir(parents=True, exist_ok=True)
        with open(mp, 'w') as f:
            for st in sorted(val_stems):
                f.write(st + '\n')
        print(f'验证集 stem 列表: {mp}')


if __name__ == '__main__':
    main()
