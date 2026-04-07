# Copyright (c) OpenMMLab. All rights reserved.
"""SegStereo：按 epoch 预热或验证集指标平台自动打开语义→视差融合。"""
from typing import Dict, Optional

from mmengine.hooks import Hook
from mmengine.logging import print_log
from mmengine.model import is_model_wrapper

from mmseg.registry import HOOKS


@HOOKS.register_module()
class SegStereoSemanticFuseHook(Hook):
    """在训练中更新 :class:`SegStereo` 的 ``_semantic_fuse_active``。

    **与 ``SegStereo.fuse_semantic_warmup_epochs`` 的关系**：每个 ``before_train_epoch``
    会设置

    .. code-block:: text

        active = (runner.epoch >= warmup_epochs) or plateau_triggered

    其中 ``warmup_epochs`` 本 Hook 的 ``warmup_epochs`` 若不为 ``None`` 则优先，
    否则使用模型上的 ``fuse_semantic_warmup_epochs``。

    **平台检测**：``auto_plateau=True`` 时，在 ``after_val_epoch`` 根据 ``metrics``
    中指定指标判断若连续 ``patience`` 次验证提升不足 ``min_delta``（与指标同量纲，
    IoUMetric 的 ``mIoU`` 为百分制小数，如 ``98.5``），则 ``plateau_triggered=True``，
    语义融合提前开启（与 warmup **或** 关系）。可选 ``plateau_only_after_epoch`` 仅在该
    epoch 之后才检测平台（避免前期波动误触发）。

    Args:
        warmup_epochs (int, optional): 覆盖模型配置中的预热 epoch 数；``None`` 则用
            ``SegStereo.fuse_semantic_warmup_epochs``。默认 ``None``。
        auto_plateau (bool): 是否根据验证指标开启融合。默认 ``False``。
        plateau_metric (str): 指标名，需与 ``IoUMetric`` 等返回的 ``metrics`` 键一致，
            一般为 ``mIoU``。默认 ``mIoU``。
        plateau_patience (int): 连续多少次验证“无足够提升”后触发。默认 ``2``。
        plateau_min_delta (float): 视为有提升的最小增幅（与 ``plateau_metric`` 同单位）。
            默认 ``0.05``（对百分制 mIoU 即 0.05 个百分点）。
        plateau_only_after_epoch (int): 仅当 ``runner.epoch >=`` 该值时才做平台判断；
            默认 ``0``（从第一次验证起即可）。
    """

    priority = 'NORMAL'

    def __init__(self,
                 warmup_epochs: Optional[int] = None,
                 auto_plateau: bool = False,
                 plateau_metric: str = 'mIoU',
                 plateau_patience: int = 2,
                 plateau_min_delta: float = 0.05,
                 plateau_only_after_epoch: int = 0) -> None:
        super().__init__()
        self.warmup_epochs_override = warmup_epochs
        self.auto_plateau = auto_plateau
        self.plateau_metric = plateau_metric
        self.plateau_patience = max(1, int(plateau_patience))
        self.plateau_min_delta = float(plateau_min_delta)
        self.plateau_only_after_epoch = max(0, int(plateau_only_after_epoch))
        self._plateau_triggered = False
        self._best_metric: Optional[float] = None
        self._epochs_no_improve = 0
        self._last_fuse_log: Optional[bool] = None

    def _get_segstereo(self, runner):
        m = runner.model
        if is_model_wrapper(m):
            m = m.module
        if type(m).__name__ == 'SegStereo':
            return m
        return None

    def before_train_epoch(self, runner) -> None:
        seg = self._get_segstereo(runner)
        if seg is None or not seg.fuse_semantic_to_disp:
            return
        warmup = (
            self.warmup_epochs_override
            if self.warmup_epochs_override is not None else
            seg.fuse_semantic_warmup_epochs)
        warmup = max(0, int(warmup))
        epoch = int(runner.epoch)
        warmup_ok = epoch >= warmup
        active = warmup_ok or self._plateau_triggered
        seg.set_semantic_fuse_active(active)
        if active != self._last_fuse_log:
            self._last_fuse_log = active
            if active:
                reason = []
                if warmup_ok:
                    reason.append(f'epoch>={warmup}')
                if self._plateau_triggered:
                    reason.append('plateau')
                print_log(
                    f'SegStereo 语义融合已开启（{"+".join(reason)}），epoch={epoch}',
                    logger='current')
            else:
                print_log(
                    f'SegStereo 语义融合已关闭，epoch={epoch}',
                    logger='current')

    def after_val_epoch(self,
                        runner,
                        metrics: Optional[Dict[str, float]] = None) -> None:
        seg = self._get_segstereo(runner)
        if seg is None or not seg.fuse_semantic_to_disp or not self.auto_plateau:
            return
        if self._plateau_triggered:
            return
        if metrics is None:
            return
        epoch = int(runner.epoch)
        if epoch < self.plateau_only_after_epoch:
            return
        if self.plateau_metric not in metrics:
            print_log(
                f'SegStereoSemanticFuseHook: metrics 无键 {self.plateau_metric}，'
                f'跳过平台检测。现有键: {list(metrics.keys())}',
                logger='current',
                level='WARNING')
            return
        cur = float(metrics[self.plateau_metric])
        if self._best_metric is None:
            self._best_metric = cur
            return
        if cur > self._best_metric + self.plateau_min_delta:
            self._best_metric = cur
            self._epochs_no_improve = 0
        else:
            self._epochs_no_improve += 1
        if self._epochs_no_improve >= self.plateau_patience:
            self._plateau_triggered = True
            seg.set_semantic_fuse_active(True)
            print_log(
                f'SegStereo 语义融合由验证平台触发：{self.plateau_metric} '
                f'best={self._best_metric:.4f}, cur={cur:.4f}, epoch={epoch}',
                logger='current')
