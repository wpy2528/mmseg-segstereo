# Copyright (c) OpenMMLab. All rights reserved.
"""验证阶段保存「原图 | 原图与预测掩码叠加」拼图（与 ONNX 推理可视化一致）。"""

import logging
import os.path as osp
from typing import Optional, Sequence

import cv2
import mmcv
import numpy as np
import torch
from mmengine.dist import is_main_process
from mmengine.logging import print_log
from mmengine.fileio import get
from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmengine.utils import mkdir_or_exist

from mmseg.registry import HOOKS
from mmseg.structures import SegDataSample


def _pred_to_color_bgr(pred: np.ndarray, pal_rgb: np.ndarray) -> np.ndarray:
    """``pred`` 类别 id；``pal_rgb`` (K,3) RGB uint8 → BGR 彩色图。"""
    k = pal_rgb.shape[0]
    safe = np.clip(pred, 0, k - 1)
    rgb = pal_rgb[safe.reshape(-1)].reshape(*pred.shape, 3)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _make_seg_overlay_bgr(
        img_bgr: np.ndarray,
        pred_hw: np.ndarray,
        pal_rgb: np.ndarray,
        alpha: float,
) -> np.ndarray:
    """与 ``utils/stereo_matching/infer_onnx_segstereo.make_seg_overlay_bgr`` 一致：
    ``addWeighted(原图, 1-a, 彩色掩码, a, 0)``；``pred`` 会先最近邻缩放到与 ``img_bgr`` 一致。
    """
    oh, ow = img_bgr.shape[:2]
    pred_u8 = pred_hw.astype(np.uint8)
    if pred_u8.shape[0] != oh or pred_u8.shape[1] != ow:
        pred_u8 = cv2.resize(
            pred_u8, (ow, oh), interpolation=cv2.INTER_NEAREST)
    color = _pred_to_color_bgr(pred_u8, pal_rgb)
    a = float(np.clip(alpha, 0.0, 1.0))
    return cv2.addWeighted(img_bgr, 1.0 - a, color, a, 0)


def _hstack_panels(panels: list, sep_w: int) -> np.ndarray:
    oh = panels[0].shape[0]
    if sep_w <= 0:
        return np.hstack(panels)
    sep = np.full((oh, sep_w, 3), 255, dtype=np.uint8)
    out = panels[0]
    for p in panels[1:]:
        out = np.hstack([out, sep, p])
    return out


def _resolve_palette_rgb(runner: Runner) -> np.ndarray:
    """dataset_meta.palette → (K,3) uint8 RGB。"""
    model = runner.model
    if hasattr(model, 'module'):
        model = model.module
    meta = getattr(model, 'dataset_meta', None) or {}
    pal = meta.get('palette', None)
    if pal is None:
        vm = getattr(runner.visualizer, 'dataset_meta', None) or {}
        pal = vm.get('palette', None)
    if pal is None:
        raise ValueError(
            'ValPredictionSaveHook: 无 palette，请在数据集 metainfo 或 visualizer 中提供')
    return np.asarray(pal, dtype=np.uint8)


def _resolve_source_image_path(sample: SegDataSample) -> Optional[str]:
    """原图磁盘路径：单目 pipeline 多为 ``img_path``；SegStereo 为 ``left_img_path``。"""
    for key in ('img_path', 'left_img_path'):
        if key in sample:
            p = getattr(sample, key, None)
            if p is not None and str(p):
                return str(p)
    return None


def _pred_sem_seg_to_numpy(sample: SegDataSample) -> np.ndarray:
    if 'pred_sem_seg' not in sample:
        raise KeyError('pred_sem_seg')
    t = sample.pred_sem_seg.data
    if isinstance(t, torch.Tensor):
        t = t.detach().cpu().numpy()
    return np.squeeze(t).astype(np.uint8)


@HOOKS.register_module()
class ValPredictionSaveHook(Hook):
    """验证阶段保存「原图 | 叠加预测」拼图。

    **张数上限**：``max_images`` 限制的是 **每一次完整验证**（跑完一遍 val
    dataloader）内写入的最多张数，**不是**整个训练过程的总数；每个 ``val_interval``
    都会最多再写 ``max_images`` 张，目录下文件会随 epoch 累积。

    **文件名里的 epoch**：使用 MMEngine 的 ``runner.epoch``（即
    ``train_loop._epoch``），表示 **已完成训练 epoch 计数**（含 ``--resume``
    从 checkpoint 恢复后的连续计数），与 checkpoint ``meta['epoch']`` 语义一致，
    不一定等于「本次启动后从零数起的第几轮」。

    布局：**原图（BGR）| 原图与预测语义掩码按权重叠加**（默认 0.5:0.5，同
    ``infer_onnx_segstereo.py`` 的 ``--overlay-alpha 0.5``）。

    Args:
        out_subdir (str): 相对 ``runner.work_dir`` 的子目录。
        max_images (int): **单次验证**保存张数上限（至少为 1）。
        overlay_alpha (float): 彩色掩码权重，原图为 ``1-alpha``（默认 0.5 即 5:5）。
        vis_sep (int): 两列之间白色间隔宽度，0 表示紧贴。
        backend_args (dict, optional): ``fileio.get`` 参数。
    """

    priority = 'NORMAL'

    def __init__(
            self,
            out_subdir: str = 'val_vis',
            max_images: int = 10,
            overlay_alpha: float = 0.5,
            vis_sep: int = 0,
            backend_args: Optional[dict] = None):
        self.out_subdir = out_subdir
        self.max_images = max(1, int(max_images))
        self.overlay_alpha = float(overlay_alpha)
        self.vis_sep = int(vis_sep)
        self.backend_args = backend_args.copy() if backend_args else None
        self._saved = 0
        self._out_root = ''
        self._logged_quota = False

    def _reset_val_state(self, runner: Runner) -> None:
        self._saved = 0
        self._out_root = osp.join(runner.work_dir, self.out_subdir)
        self._logged_quota = False
        if is_main_process():
            mkdir_or_exist(self._out_root)

    def before_val(self, runner: Runner) -> None:
        self._reset_val_state(runner)

    def before_val_epoch(self, runner: Runner) -> None:
        # 与 before_val 一致：部分流程下仅保证 before_val_epoch 被调用时也能重置计数
        self._reset_val_state(runner)

    def _name_prefix(self, runner: Runner) -> str:
        tl = runner.train_loop
        if tl is not None and getattr(tl, 'max_epochs', 1) > 1:
            return f'epoch_{runner.epoch:04d}'
        if runner.max_iters > 0:
            return f'iter_{runner.iter:07d}'
        return f'epoch_{runner.epoch:04d}'

    def after_val_iter(
            self,
            runner: Runner,
            batch_idx: int,
            data_batch: dict,
            outputs: Sequence[SegDataSample],
    ) -> None:
        if not is_main_process() or self._saved >= self.max_images:
            return

        try:
            pal_rgb = _resolve_palette_rgb(runner)
        except ValueError as exc:
            print_log(str(exc), logger='current', level=logging.WARNING)
            return

        if not self._logged_quota and is_main_process():
            print_log(
                f'ValPredictionSaveHook: 本轮验证最多保存 {self.max_images} 张 '
                f'(runner.epoch={runner.epoch}，为 MMEngine 训练 epoch 计数，含 resume)',
                logger='current',
                level=logging.INFO)
            self._logged_quota = True

        prefix = self._name_prefix(runner)

        if not isinstance(outputs, (list, tuple)):
            print_log(
                f'ValPredictionSaveHook: 期望 outputs 为 list/tuple 样本序列，收到 '
                f'{type(outputs)}，跳过',
                logger='current',
                level=logging.WARNING)
            return

        for sample in outputs:
            if self._saved >= self.max_images:
                break
            if 'pred_sem_seg' not in sample:
                continue
            img_path = _resolve_source_image_path(sample)
            if not img_path:
                continue
            img_bytes = get(img_path, backend_args=self.backend_args)
            # 与磁盘原图一致，BGR，便于与 cv2.addWeighted 一致
            img_bgr = mmcv.imfrombytes(img_bytes, channel_order='bgr')
            try:
                pred_hw = _pred_sem_seg_to_numpy(sample)
            except Exception:
                continue

            overlay_bgr = _make_seg_overlay_bgr(
                img_bgr, pred_hw, pal_rgb, self.overlay_alpha)
            strip = _hstack_panels(
                [img_bgr, overlay_bgr], max(0, self.vis_sep))

            stem = osp.splitext(osp.basename(img_path))[0]
            # slot：当次验证内第几张（0..max_images-1），勿与 batch_idx 混淆
            out_file = osp.join(
                self._out_root,
                f'{prefix}_slot{self._saved:02d}_{stem}.png',
            )
            mmcv.imwrite(strip, out_file)
            self._saved += 1
