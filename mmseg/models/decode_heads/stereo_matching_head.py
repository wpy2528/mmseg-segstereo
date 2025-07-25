# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import MODELS
from .decode_head import BaseDecodeHead
# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from abc import ABCMeta, abstractmethod
from typing import List, Tuple

import torch
import torch.nn as nn
from mmengine.model import BaseModule
from torch import Tensor

from mmseg.registry import MODELS
from mmseg.structures import build_pixel_sampler
from mmseg.utils import ConfigType, SampleList
from ..losses import accuracy
from ..utils import resize

@MODELS.register_module()
class StereoMatchingHead(BaseDecodeHead):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, inputs):
        return inputs

    def loss_by_feat(self, seg_logits: Tensor,
                     batch_data_samples: SampleList) -> dict:
        """Compute segmentation loss.

        Args:
            seg_logits (Tensor): The output from decode head forward function.
            batch_data_samples (List[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        seg_label = self._stack_batch_gt(batch_data_samples)
        loss = dict()
        assert seg_logits.shape == seg_label.shape, \
            f"seg_logits.shape: {seg_logits.shape}, seg_label.shape: {seg_label.shape}"
        # seg_logits = resize(
        #     input=seg_logits,
        #     size=seg_label.shape[2:],
        #     mode='bilinear',
        #     align_corners=self.align_corners)
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logits, seg_label)
        else:
            seg_weight = None
        # ! 由于要计算L1损失，所以需要保持[B 1 H W]的形状
        # seg_label = seg_label.squeeze(1) 

        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode
        for loss_decode in losses_decode:
            if loss_decode.loss_name not in loss:
                loss[loss_decode.loss_name] = loss_decode(
                    seg_logits,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)
            else:
                loss[loss_decode.loss_name] += loss_decode(
                    seg_logits,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)
        if len(seg_label.shape) == 3:
            loss['acc_seg'] = accuracy(
                seg_logits, seg_label, ignore_index=self.ignore_index)
        return loss

    def predict_by_feat(self, seg_logits: Tensor,
                        batch_img_metas: List[dict]) -> Tensor:
        if isinstance(batch_img_metas[0]['img_shape'], torch.Size):
            # slide inference
            size = batch_img_metas[0]['img_shape']
        elif 'pad_shape' in batch_img_metas[0]:
            size = batch_img_metas[0]['pad_shape'][:2]
        else:
            size = batch_img_metas[0]['img_shape']
        
        # 检查seg_logits的形状是否与size一致
        assert seg_logits.shape[2] == size[0] and seg_logits.shape[3] == size[1], \
            f"seg_logits.shape: {seg_logits.shape}, size: {size}"
        
        return seg_logits