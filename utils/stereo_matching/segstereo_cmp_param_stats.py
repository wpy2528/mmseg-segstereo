#!/usr/bin/env python3
"""对比两份 SegStereo 配置的全网与视差分支参数量。

用法:
  python utils/stereo_matching/segstereo_cmp_param_stats.py \\
    configs/segstereo/..._disp_stereofeat.py \\
    configs/segstereo/..._disp_igev.py
"""

from __future__ import annotations

import argparse
import os.path as osp
import sys


def _count(module) -> tuple[int, int]:
    n = sum(p.numel() for p in module.parameters())
    ng = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return n, ng


def main() -> None:
    parser = argparse.ArgumentParser(description='SegStereo 配置参数量对比')
    parser.add_argument(
        'configs',
        nargs='+',
        help='一个或多个训练配置文件路径',
    )
    args = parser.parse_args()

    repo_root = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from mmengine.config import Config

    from mmseg.registry import MODELS
    from mmseg.utils import register_all_modules

    register_all_modules(init_default_scope=True)

    rows = []
    for path in args.configs:
        cfg = Config.fromfile(path)
        model = MODELS.build(cfg.model)
        disp = model.disparity_branch
        t_all, tg_all = _count(model)
        t_disp, tg_disp = _count(disp)
        name = osp.basename(path)
        rows.append((name, t_all, tg_all, t_disp, tg_disp, type(disp).__name__))

    colw = max(len(r[0]) for r in rows)
    hdr = f"{'config':<{colw}}  {'total(M)':>12}  {'disp(M)':>12}  disp_type"
    print(hdr)
    print('-' * len(hdr))
    for name, t_all, tg_all, t_disp, tg_disp, cls_name in rows:
        print(
            f'{name:<{colw}}  {t_all / 1e6:12.3f}  {t_disp / 1e6:12.3f}  {cls_name}'
        )
    print(
        '\n说明: total 为 SegStereo 全部参数；disp 为 disparity_branch 子模块。'
        '训练后对比双目效果请看验证集 DisparityMetric（EPE、bad1/3/5）与 mIoU。'
    )


if __name__ == '__main__':
    main()
