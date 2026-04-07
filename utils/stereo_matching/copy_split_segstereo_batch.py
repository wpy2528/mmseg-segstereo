#!/usr/bin/env python3
# Copyright (c) OpenMMLab. All rights reserved.
"""将一批四模态数据（images / images_right / disparity / labels）按比例拆成 train/val，
**复制**到 ``<train_parent>/<batch_name>/`` 与 ``<val_parent>/<batch_name>/``。

``batch_name`` 默认同源目录末级名（如 ``20260407`` -> ``segstereo/20260407``）。

用法::

    python utils/stereo_matching/copy_split_segstereo_batch.py \\
        /data_SSD2/datasets/20260407 \\
        --train-parent /data_SSD2/datasets/segstereo \\
        --val-parent /data_SSD2/datasets/segstereo_val

    # 先预览
    python .../copy_split_segstereo_batch.py /path/to/batch --dry-run
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
        'source',
        type=str,
        help='源批次根目录（其下含 images / images_right / disparity / labels）')
    p.add_argument(
        '--train-parent',
        type=str,
        default='/data_SSD2/datasets/segstereo',
        help='训练集父目录，默认 .../segstereo')
    p.add_argument(
        '--val-parent',
        type=str,
        default='/data_SSD2/datasets/segstereo_val',
        help='验证集父目录，默认 .../segstereo_val')
    p.add_argument(
        '--batch-name',
        type=str,
        default=None,
        help='目标子目录名，默认与 source 末级目录名相同')
    p.add_argument('--val-ratio', type=float, default=0.2)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()

    src = Path(args.source).resolve()
    if not src.is_dir():
        print(f'源不存在: {src}', file=sys.stderr)
        sys.exit(1)

    batch = args.batch_name or src.name
    train_root = Path(args.train_parent).resolve() / batch
    val_root = Path(args.val_parent).resolve() / batch

    stems = _common_stems(src)
    n = len(stems)
    if n < 2:
        print(f'交集样本不足 2（当前 {n}）', file=sys.stderr)
        sys.exit(1)

    n_val = max(1, min(round(n * args.val_ratio), n - 1))
    rng = random.Random(args.seed)
    rng.shuffle(stems)
    val_stems = set(stems[:n_val])
    train_stems = set(stems[n_val:])

    print(f'源: {src}')
    print(f'训练 -> {train_root}  ({len(train_stems)} 样本)')
    print(f'验证 -> {val_root}    ({len(val_stems)} 样本)')
    print(f'val_ratio={args.val_ratio}, seed={args.seed}')

    if args.dry_run:
        print('dry-run，未写入磁盘')
        print('验证集 stem 示例:', sorted(val_stems)[:8])
        return

    for root in (train_root, val_root):
        for sub, _ in SPECS:
            (root / sub).mkdir(parents=True, exist_ok=True)

    def copy_stem(stem: str, dest_root: Path) -> None:
        for sub, ext in SPECS:
            sfp = src / sub / f'{stem}{ext}'
            dfp = dest_root / sub / f'{stem}{ext}'
            if not sfp.is_file():
                print(f'缺失源文件: {sfp}', file=sys.stderr)
                sys.exit(1)
            shutil.copy2(sfp, dfp)

    for stem in sorted(train_stems):
        copy_stem(stem, train_root)
    for stem in sorted(val_stems):
        copy_stem(stem, val_root)

    print('复制完成。')


if __name__ == '__main__':
    main()
