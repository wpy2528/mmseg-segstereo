# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.registry import MODELS
from .utils import weighted_loss, get_class_weight

@weighted_loss
def l1_loss(pred, target, ignore_index=None):
    """
    逐元素L1损失，支持ignore_index。
    Args:
        pred (Tensor): 预测 (N, C, ...)
        target (Tensor): 标签 (N, ...)
        ignore_index (int|None): 忽略的标签值
    Returns:
        Tensor: 损失 (与 pred 相同 shape)
    """
    if ignore_index is not None:
        valid_mask = (target != ignore_index)
        loss = torch.abs(pred - target)
        loss = loss * valid_mask
    else:
        loss = torch.abs(pred - target)
    return loss

@MODELS.register_module()
class L1Loss(nn.Module):
    """L1 损失 (MAE)。
    Args:
        reduction (str): 损失归约方式。默认 'mean'。
        loss_weight (float): 损失权重。默认 1.0。
        ignore_index (int|None): 忽略的标签值。默认 None。
        loss_name (str): 损失名称。默认 'loss_l1'。
    """
    def __init__(self,
                 reduction='mean',
                 loss_weight=1.0,
                 ignore_index=None,
                 loss_name='loss_l1'):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index
        self._loss_name = loss_name

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None, **kwargs):
        reduction = reduction_override if reduction_override else self.reduction
        loss = self.loss_weight * l1_loss(
            pred, target, weight=weight, reduction=reduction, avg_factor=avg_factor, ignore_index=self.ignore_index)
        return loss

    @property
    def loss_name(self):
        return self._loss_name 