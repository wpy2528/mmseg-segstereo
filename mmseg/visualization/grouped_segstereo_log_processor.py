# Copyright (c) OpenMMLab. All rights reserved.
"""SegStereo 训练日志：将 total / 语义 / 视差 相关标量分行打印，便于阅读。

说明：日志中的 ``loss_disp``、``decode.loss_ce`` 等为**训练损失**，不是验证集上的
视差 EPE / Bad-X（后者由 ``DisparityMetric`` 在 val epoch 输出）。"""
import re
from typing import List, Tuple

from mmengine.registry import LOG_PROCESSORS
from mmengine.runner.log_processor import LogProcessor


def _parse_metric_tail(log_str: str) -> Tuple[str, List[Tuple[str, str]]]:
    """从 ``LogProcessor`` 生成的单行里拆出 header 与 ``name: value`` 列表。"""
    # 指标段通常从 ``loss:`` 开始（总 loss）；若无则整段视为 header
    m = re.search(r'\b(loss:\s)', log_str)
    if not m:
        return log_str, []
    i = m.start()
    header = log_str[:i]
    body = log_str[i:]
    pairs: List[Tuple[str, str]] = []
    for seg in body.split('  '):
        seg = seg.strip()
        if not seg or ': ' not in seg:
            continue
        name, val = seg.split(': ', 1)
        name = name.strip()
        val = val.strip()
        if name:
            pairs.append((name, val))
    return header, pairs


def _group_pairs(
        pairs: List[Tuple[str, str]]
) -> Tuple[List[Tuple[str, str]], ...]:
    total: List[Tuple[str, str]] = []
    semantic: List[Tuple[str, str]] = []
    stereo: List[Tuple[str, str]] = []
    other: List[Tuple[str, str]] = []
    for k, v in pairs:
        if k == 'loss':
            total.append((k, v))
        elif k.startswith('loss_disp') or k.startswith('loss_sem'):
            stereo.append((k, v))
        elif k.startswith('decode.') or k.startswith('aux_'):
            semantic.append((k, v))
        else:
            other.append((k, v))
    return total, semantic, stereo, other


def _join_pairs(pairs: List[Tuple[str, str]], sep: str = '  ') -> str:
    return sep.join(f'{k}: {v}' for k, v in pairs)


@LOG_PROCESSORS.register_module()
class GroupedSegStereoLogProcessor(LogProcessor):
    """在默认 :class:`~mmengine.runner.log_processor.LogProcessor` 基础上，将 train iter
    日志按 ``[total]`` / ``[semantic]`` / ``[stereo]`` / ``[other]`` 分行排版。

    val / test 的日志格式与父类一致。
    """

    def get_log_after_iter(self, runner, batch_idx: int, mode: str):
        tag, log_str = super().get_log_after_iter(runner, batch_idx, mode)
        if mode != 'train':
            return tag, log_str
        header, pairs = _parse_metric_tail(log_str)
        if not pairs:
            return tag, log_str
        total, semantic, stereo, other = _group_pairs(pairs)
        lines = [header.rstrip()]
        if total:
            lines.append('  [total]    ' + _join_pairs(total))
        if semantic:
            lines.append('  [semantic] ' + _join_pairs(semantic))
        if stereo:
            lines.append('  [stereo]   ' + _join_pairs(stereo))
        if other:
            lines.append('  [other]    ' + _join_pairs(other))
        return tag, '\n'.join(lines)
