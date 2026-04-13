# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.runner import Runner

from mmseg.engine.hooks.val_prediction_save_hook import ValPredictionSaveHook
from mmseg.registry import RUNNERS


def _inject_val_prediction_save_hook(
        cfg,
        *,
        out_subdir: str,
        max_images: int,
        overlay_alpha: float,
        vis_sep: int,
) -> None:
    """在 ``work_dir/<out_subdir>`` 下每轮 val 保存「原图 | 原图+掩码叠加」拼图。"""
    ch = cfg.get('custom_hooks', None)
    if ch is None:
        ch = []
    else:
        ch = list(ch)

    def _is_val_pred_hook(item) -> bool:
        if not isinstance(item, dict):
            return False
        t = item.get('type')
        return t is ValPredictionSaveHook or t == 'ValPredictionSaveHook'

    if any(_is_val_pred_hook(h) for h in ch):
        return
    # 须用注册名字符串，勿写 type=类对象；否则 cfg.pretty_text 无法用 YAPF 格式化而崩溃。
    ch.append(
        dict(
            type='ValPredictionSaveHook',
            out_subdir=out_subdir,
            max_images=max_images,
            overlay_alpha=overlay_alpha,
            vis_sep=vis_sep,
        ))
    cfg.custom_hooks = ch


def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume',
        action='store_true',
        default=False,
        help='resume from the latest checkpoint in the work_dir automatically')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config file, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    parser.add_argument(
        '--no-val-pred-vis',
        action='store_true',
        help='关闭验证阶段在 work_dir 下保存预测可视化（默认开启，每轮 val 保存 10 张）')
    parser.add_argument(
        '--val-pred-vis-dir',
        type=str,
        default='val_vis',
        help='验证可视化子目录名（相对 work_dir，默认 val_vis）')
    parser.add_argument(
        '--val-pred-vis-num',
        type=int,
        default=10,
        help='每次验证最多保存的图片数（默认 10）')
    parser.add_argument(
        '--val-pred-vis-alpha',
        type=float,
        default=0.5,
        help='验证可视化：彩色掩码权重，原图为 1-α（默认 0.5 即 5:5，同 infer_onnx overlay）')
    parser.add_argument(
        '--val-pred-vis-sep',
        type=int,
        default=0,
        help='验证可视化两列间白色间隔宽度（像素），默认 0')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        assert "configs/" in args.config, "work_dir must be in configs/ directory"
        cfg.work_dir = os.path.join("work_dirs", os.path.dirname(args.config).split("configs/")[-1], osp.splitext(osp.basename(args.config))[0])

    # enable automatic-mixed-precision training
    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == 'AmpOptimWrapper':
            print_log(
                'AMP training is already enabled in your config.',
                logger='current',
                level=logging.WARNING)
        else:
            assert optim_wrapper == 'OptimWrapper', (
                '`--amp` is only supported when the optimizer wrapper type is '
                f'`OptimWrapper` but got {optim_wrapper}.')
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

    # resume training
    cfg.resume = args.resume

    if not args.no_val_pred_vis:
        n = max(1, int(args.val_pred_vis_num))
        a = float(args.val_pred_vis_alpha)
        alpha = min(1.0, max(0.0, a))
        sep = max(0, int(args.val_pred_vis_sep))
        _inject_val_prediction_save_hook(
            cfg,
            out_subdir=args.val_pred_vis_dir.strip() or 'val_vis',
            max_images=n,
            overlay_alpha=alpha,
            vis_sep=sep,
        )

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # start training
    runner.train()


if __name__ == '__main__':
    main()
