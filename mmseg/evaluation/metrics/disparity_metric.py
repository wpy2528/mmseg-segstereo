# Copyright (c) OpenMMLab. All rights reserved.
"""双目视差评估：与 SegStereo ``pred_disp`` / ``gt_disp`` 对齐（Scene Flow / KITTI 风格 EPE、Bad-X）。"""
from collections import OrderedDict
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log
from prettytable import PrettyTable

from mmseg.registry import METRICS


def _bad_key(th: float) -> str:
    if abs(th - round(th)) < 1e-6:
        return f'bad_{int(round(th))}'
    return f'bad_{th}'


@METRICS.register_module()
class DisparityMetric(BaseMetric):
    """Evaluate disparity (stereo) predictions against ``gt_disp``.

    Valid pixels follow common stereo practice and SegStereo loss:
    ``(gt > 0) & (gt < max_disp)``.

    Metrics (micro-averaged over all valid pixels in the val set):

        - ``epe``: mean absolute error (px)
        - ``rms``: root mean square error (px)
        - ``bad_{t}``: percentage of valid pixels with ``|pred - gt| > t`` for
          each ``t`` in ``bad_thresholds``

    Args:
        max_disp (float): Upper bound on valid GT disparity (exclusive).
            Should match ``disparity_branch.max_disp`` in the model config.
        bad_thresholds (tuple): Absolute error thresholds in pixels for bad
            pixel rates. Default ``(1.0, 3.0, 5.0)``.
        collect_device (str): Device for distributed gather. Default ``cpu``.
        prefix (str, optional): Log prefix for metric names.
    """

    default_prefix: Optional[str] = 'disparity'

    def __init__(self,
                 max_disp: float = 192.0,
                 bad_thresholds: Tuple[float, ...] = (1.0, 3.0, 5.0),
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 **kwargs) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.max_disp = float(max_disp)
        self.bad_thresholds = tuple(float(t) for t in bad_thresholds)

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        for data_sample in data_samples:
            if 'pred_disp' not in data_sample or 'gt_disp' not in data_sample:
                continue
            pred = data_sample['pred_disp']['data']
            gt = data_sample['gt_disp']['data']
            if isinstance(pred, np.ndarray):
                pred = torch.from_numpy(pred)
            if isinstance(gt, np.ndarray):
                gt = torch.from_numpy(gt)
            pred = pred.float().squeeze()
            gt = gt.float().squeeze()
            if pred.dim() != 2 or gt.dim() != 2:
                continue
            if pred.shape != gt.shape:
                pred = F.interpolate(
                    pred.unsqueeze(0).unsqueeze(0),
                    size=gt.shape,
                    mode='bilinear',
                    align_corners=False).squeeze()
            valid = (gt > 0) & (gt < self.max_disp)
            if not valid.any():
                continue
            err = (pred - gt).abs()
            v = valid
            abs_sum = err[v].sum()
            sq_sum = (err[v]**2).sum()
            n = int(v.sum().item())
            bad_counts = [
                int((err[v] > t).sum().item()) for t in self.bad_thresholds
            ]
            self.results.append((n, abs_sum.item(), sq_sum.item(), bad_counts))

    def compute_metrics(self, results: list):
        logger: MMLogger = MMLogger.get_current_instance()
        if not results:
            print_log(
                'DisparityMetric: no valid disparity pairs '
                '(need 6-ch val + pred_disp & gt_disp).',
                logger=logger,
                level='WARNING')
            return OrderedDict(), []

        total_n = sum(r[0] for r in results)
        total_abs = sum(r[1] for r in results)
        total_sq = sum(r[2] for r in results)
        num_t = len(self.bad_thresholds)
        total_bad = [sum(r[3][i] for r in results) for i in range(num_t)]

        epe = total_abs / total_n
        rms = (total_sq / total_n) ** 0.5
        metrics: Dict[str, float] = OrderedDict()
        metrics['epe'] = round(float(epe), 4)
        metrics['rms'] = round(float(rms), 4)
        for t, bc in zip(self.bad_thresholds, total_bad):
            metrics[_bad_key(t)] = round(100.0 * float(bc) / float(total_n), 4)

        table = PrettyTable()
        for k, v in metrics.items():
            table.add_column(k, [v])
        print_log('disparity metrics:', logger)
        print_log('\n' + table.get_string(), logger=logger)

        # 第二个返回值须可迭代，避免作为最后一个 metric 时 ValLoop 写 per_sample json 报错
        return metrics, []
