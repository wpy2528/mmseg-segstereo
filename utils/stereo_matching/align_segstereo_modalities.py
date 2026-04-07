#!/usr/bin/env python3
# Copyright (c) OpenMMLab. All rights reserved.
"""对齐并校验 SegStereo 四模态目录。

要求子目录：``images``、``images_right``、``disparity``、``labels``（主名相同，扩展名分别为 .jpg/.jpg/.png/.png）。

- 仅保留在四个目录中**都存在**同一主名的样本。
- 默认将无效文件**移走**到 ``<父目录>/<根名>_orphans_quarantine/<子目录>/``（勿放在 data_root 内，避免 glob 误匹配）。
- ``--delete-unmatched``：不移动，**直接删除**无效文件。
- ``--verify-read``：用 OpenCV 尝试读取；读失败或 **H×W 不一致** 的样本从有效集中剔除，其四份文件按上两项规则处理。

用法::

    # 只看统计
    python utils/stereo_matching/align_segstereo_modalities.py /data/segstereo --dry-run

    # 验证可读与尺寸一致后，删除所有无效文件
    python utils/stereo_matching/align_segstereo_modalities.py /data/segstereo --verify-read --delete-unmatched
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def _rename_if_needed(root: Path, dry_run: bool) -> None:
    if (root / 'image').is_dir() and not (root / 'images').is_dir():
        print('重命名: image -> images')
        if not dry_run:
            (root / 'image').rename(root / 'images')
    if (root / 'image_right').is_dir() and not (root / 'images_right').is_dir():
        print('重命名: image_right -> images_right')
        if not dry_run:
            (root / 'image_right').rename(root / 'images_right')


def _hw(img) -> tuple:
    if img is None:
        return None
    if img.ndim == 2:
        return img.shape[0], img.shape[1]
    return img.shape[0], img.shape[1]


def _verify_stem(root: Path, stem: str) -> bool:
    import cv2

    left = root / 'images' / f'{stem}.jpg'
    right = root / 'images_right' / f'{stem}.jpg'
    disp = root / 'disparity' / f'{stem}.png'
    lab = root / 'labels' / f'{stem}.png'

    l = cv2.imread(str(left), cv2.IMREAD_COLOR)
    r = cv2.imread(str(right), cv2.IMREAD_COLOR)
    d = cv2.imread(str(disp), cv2.IMREAD_UNCHANGED)
    s = cv2.imread(str(lab), cv2.IMREAD_UNCHANGED)

    if l is None or r is None or d is None or s is None:
        return False

    hl, wl = _hw(l)
    hr, wr = _hw(r)
    hd, wd = _hw(d)
    hs, ws = _hw(s)

    if not (hl == hr == hd == hs and wl == wr == wd == ws):
        return False
    return True


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        'data_root', type=str, help='含 images、images_right、disparity、labels 的数据根目录')
    p.add_argument(
        '--dry-run',
        action='store_true',
        help='只打印统计，不改文件')
    p.add_argument(
        '--delete-unmatched',
        action='store_true',
        help='对无效样本直接删除文件（默认改为移到 orphans_quarantine）')
    p.add_argument(
        '--verify-read',
        action='store_true',
        help='用 OpenCV 读取并检查四张图 H×W 一致；不合格者从有效集中剔除')
    args = p.parse_args()
    root = Path(args.data_root).resolve()

    _rename_if_needed(root, args.dry_run)

    specs = [
        ('images', '*.jpg'),
        ('images_right', '*.jpg'),
        ('disparity', '*.png'),
        ('labels', '*.png'),
    ]
    stems = []
    for sub, pat in specs:
        d = root / sub
        if not d.is_dir():
            print(f'缺少目录: {d}', file=sys.stderr)
            sys.exit(1)
        s = {x.stem for x in d.glob(pat)}
        stems.append(s)
        print(f'{sub}: {len(s)} 个文件')

    common = stems[0]
    for s in stems[1:]:
        common &= s
    print(f'四者交集主名数量（文件名层）: {len(common)}')

    if args.verify_read:
        try:
            import cv2  # noqa: F401
        except ImportError:
            print('需要 OpenCV：pip install opencv-python', file=sys.stderr)
            sys.exit(1)
        bad = []
        for stem in sorted(common):
            if not _verify_stem(root, stem):
                bad.append(stem)
        for b in bad:
            common.discard(b)
        if bad:
            print(f'校验失败或尺寸不一致，剔除 {len(bad)} 个主名（示例: {bad[:5]}...）')
        print(f'校验后主名数量: {len(common)}')

    if args.dry_run:
        only_in = []
        for (sub, pat), sset in zip(specs, stems):
            for st in sorted(sset - common):
                only_in.append(f'{sub}/{st}')
        if only_in:
            print(f'将处理（删除或移走）的文件数: {len(only_in)}')
            for line in only_in[:30]:
                print('  ', line)
            if len(only_in) > 30:
                print(f'  ... 共 {len(only_in)}')
        return

    total = 0
    if args.delete_unmatched:
        for (sub, pat), sset in zip(specs, stems):
            d = root / sub
            n = 0
            for f in list(d.glob(pat)):
                if f.stem not in common:
                    f.unlink()
                    n += 1
            total += n
            remain = len(list(d.glob(pat)))
            print(f'{sub}: 删除 {n} 个，剩余 {remain}')
        print(f'共删除 {total} 个无效文件')
    else:
        qroot = root.parent / (root.name + '_orphans_quarantine')
        qroot.mkdir(parents=True, exist_ok=True)
        for (sub, pat), sset in zip(specs, stems):
            q = qroot / sub
            q.mkdir(parents=True, exist_ok=True)
            d = root / sub
            n = 0
            for f in list(d.glob(pat)):
                if f.stem not in common:
                    dest = q / f.name
                    if dest.exists():
                        dest.unlink()
                    shutil.move(str(f), str(dest))
                    n += 1
            total += n
            remain = len(list(d.glob(pat)))
            print(f'{sub}: 保留 {remain}，移走 {n} 个 -> {qroot}/{sub}')
        print(f'共移走 {total} 个文件')


if __name__ == '__main__':
    main()
