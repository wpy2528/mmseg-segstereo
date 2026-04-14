# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging
import os
import os.path as osp
from contextlib import contextmanager

from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.runner import Runner

from mmseg.engine.hooks.val_prediction_save_hook import ValPredictionSaveHook
from mmseg.registry import RUNNERS


def _patch_mmengine_autocast_device_type():
    """``torch.autocast`` only accepts ``device_type`` ``'cuda'`` or ``'cpu'``, not ``'cuda:0'``.

    MMEngine's ``autocast`` (PyTorch >= 1.10 branch) forwards ``get_device()`` when
    ``device_type`` is omitted. If that string is ``cuda:N`` (or a ``torch.device`` with
    index), PyTorch raises
    ``ValueError: User specified autocast device_type must be cuda or cpu, but got cuda:0``.
    ``AmpOptimWrapper`` and val/test loops hit this when enabling AMP.

    We wrap the original context manager and normalize CUDA device strings before
    delegating. Also normalize ``mmengine.device.utils.DEVICE`` when it looks like
    ``cuda:N`` so the upstream implementation stays consistent.
    """
    try:
        import torch
        import mmengine.device.utils as device_utils
        import mmengine.runner as runner_pkg
        import mmengine.runner.amp as amp_mod
        import mmengine.runner.loops as loops_mod
    except ImportError:
        return

    dev = getattr(device_utils, 'DEVICE', None)
    if isinstance(dev, str) and dev.startswith('cuda:'):
        device_utils.DEVICE = 'cuda'

    _orig = amp_mod.autocast

    @contextmanager
    def _autocast_normalized(device_type=None, dtype=None, enabled=True,
                             cache_enabled=None):
        if isinstance(device_type, torch.device) and device_type.type == 'cuda':
            device_type = 'cuda'
        elif isinstance(device_type, str) and device_type.startswith('cuda:'):
            device_type = 'cuda'
        if device_type is None:
            from mmengine.device import get_device
            d = get_device()
            if isinstance(d, str) and d.startswith('cuda:'):
                device_type = 'cuda'
        with _orig(
                device_type=device_type,
                dtype=dtype,
                enabled=enabled,
                cache_enabled=cache_enabled):
            yield

    amp_mod.autocast = _autocast_normalized
    loops_mod.autocast = _autocast_normalized
    runner_pkg.autocast = _autocast_normalized


def _inject_val_prediction_save_hook(
        cfg,
        *,
        out_subdir: str,
        max_images: int,
        overlay_alpha: float,
        vis_sep: int,
) -> None:
    """在 ``work_dir/<out_subdir>`` 下每轮 val 保存「原图 | 原图+掩码叠加」拼图。

    若 ``cfg.custom_hooks`` 里已有 ``ValPredictionSaveHook``（例如用 ``work_dirs/.../config.py``
    启动时 dump 里已写入），则用本次合并后的参数 **原地覆盖** 该条，避免忽略
    ``--val-pred-vis-num`` 等命令行覆盖。
    """
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

    # 须用注册名字符串，勿写 type=类对象；否则 cfg.pretty_text 无法用 YAPF 格式化而崩溃。
    spec = dict(
        type='ValPredictionSaveHook',
        out_subdir=out_subdir,
        max_images=max_images,
        overlay_alpha=overlay_alpha,
        vis_sep=vis_sep,
    )
    for i, h in enumerate(ch):
        if _is_val_pred_hook(h):
            merged = dict(h)
            merged.update(spec)
            ch[i] = merged
            cfg.custom_hooks = ch
            return
    ch.append(spec)
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
        '--val-pred-vis',
        action='store_true',
        help='强制开启验证预测可视化；未传时是否开启由配置文件 ``val_pred_vis.enable`` 决定')
    parser.add_argument(
        '--val-pred-vis-dir',
        type=str,
        default=None,
        help='覆盖配置 ``val_pred_vis.out_subdir``（相对 work_dir）；未传则用配置，缺省为 val_vis')
    parser.add_argument(
        '--val-pred-vis-num',
        type=int,
        default=None,
        help='覆盖配置 ``val_pred_vis.max_images``；未传则用配置，缺省为 10')
    parser.add_argument(
        '--val-pred-vis-alpha',
        type=float,
        default=None,
        help='覆盖配置 ``val_pred_vis.overlay_alpha``；未传则用配置，缺省为 0.5')
    parser.add_argument(
        '--val-pred-vis-sep',
        type=int,
        default=None,
        help='覆盖配置 ``val_pred_vis.vis_sep``；未传则用配置，缺省为 0')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()
    _patch_mmengine_autocast_device_type()

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

    vpc = cfg.get('val_pred_vis', None) or {}
    if not isinstance(vpc, dict):
        print_log(
            '配置项 val_pred_vis 应为 dict，已忽略。',
            logger='current',
            level=logging.WARNING)
        vpc = {}
    cfg_enable = bool(vpc.get('enable', False))
    val_vis_on = bool(args.val_pred_vis) or cfg_enable
    if val_vis_on:
        out_subdir = (
            args.val_pred_vis_dir if args.val_pred_vis_dir is not None else
            vpc.get('out_subdir', 'val_vis'))
        out_subdir = (out_subdir or 'val_vis').strip() or 'val_vis'
        n_raw = (args.val_pred_vis_num if args.val_pred_vis_num is not None else
                 vpc.get('max_images', 10))
        n = max(1, int(n_raw))
        a_raw = (args.val_pred_vis_alpha if args.val_pred_vis_alpha is not None
                 else vpc.get('overlay_alpha', 0.5))
        a = float(a_raw)
        alpha = min(1.0, max(0.0, a))
        sep_raw = (args.val_pred_vis_sep if args.val_pred_vis_sep is not None else
                   vpc.get('vis_sep', 0))
        sep = max(0, int(sep_raw))
        _inject_val_prediction_save_hook(
            cfg,
            out_subdir=out_subdir,
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
