# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.registry import MODELS
from .utils import weighted_loss


@weighted_loss
def stereo_sequence_loss(pred, target, valid_mask, loss_gamma=0.9):
    """
    DStereoPlus序列损失函数，用于立体匹配的多尺度预测。
    
    Args:
        pred (list[Tensor]): 预测列表，包含初始预测和迭代预测
            - pred[0]: 初始预测 (N, 1, H, W)
            - pred[1:]: 迭代预测列表，每个元素 (N, 1, H, W)
        target (Tensor): 真实视差图 (N, 1, H, W)
        valid_mask (Tensor): 有效像素掩码 (N, 1, H, W)
        loss_gamma (float): 损失权重衰减因子，默认0.9
        
    Returns:
        list[Tensor]: 损失列表，包含初始损失和迭代损失
    """
    if not isinstance(pred, (list, tuple)):
        pred = [pred]
    
    n_predictions = len(pred)
    assert n_predictions >= 1, "至少需要一个预测"
    
    disp_loss = []
    
    # 确保有效掩码形状正确
    assert valid_mask.shape == target.shape, f"有效掩码形状 {valid_mask.shape} 与目标形状 {target.shape} 不匹配"
    assert not torch.isinf(target[valid_mask.bool()]).any(), "目标视差图中包含无穷大值"
    
    # 初始预测损失
    agg_pred = pred[0]
    assert agg_pred.shape == valid_mask.shape, f"初始预测形状 {agg_pred.shape} 与有效掩码形状 {valid_mask.shape} 不匹配"
    
    # 使用Smooth L1损失计算初始预测损失
    initial_loss = F.smooth_l1_loss(
        agg_pred[valid_mask.bool()], 
        target[valid_mask.bool()], 
        reduction='mean'
    )
    disp_loss.append(1.0 * initial_loss)
    
    # 迭代预测损失
    for i in range(1, n_predictions):
        iter_pred = pred[i]
        assert iter_pred.shape == valid_mask.shape, f"迭代预测 {i} 形状 {iter_pred.shape} 与有效掩码形状 {valid_mask.shape} 不匹配"
        
        # 计算调整后的损失权重
        adjusted_loss_gamma = loss_gamma ** (15 / (n_predictions - 1))
        i_weight = adjusted_loss_gamma ** (n_predictions - i - 1)
        
        # 计算L1损失
        i_loss = torch.abs(iter_pred - target)
        assert i_loss.shape == valid_mask.shape, f"损失形状 {i_loss.shape} 与有效掩码形状 {valid_mask.shape} 不匹配"
        
        # 只计算有效像素的损失
        iter_loss = i_loss[valid_mask.bool()].mean()
        disp_loss.append(i_weight * iter_loss)
    
    return disp_loss


@MODELS.register_module()
class StereoSequenceLoss(nn.Module):
    """DStereoPlus序列损失函数。
    
    用于立体匹配任务的多尺度预测损失，支持初始预测和迭代预测的加权组合。
    
    Args:
        loss_gamma (float): 损失权重衰减因子，用于调整迭代预测的权重。默认0.9。
        max_disp (int): 最大视差值，用于生成有效像素掩码。默认192。
        loss_weight (float): 损失权重。默认1.0。
        loss_name (str): 损失名称。默认'loss_stereo_sequence'。
    """
    
    def __init__(self,
                 loss_gamma=0.9,
                 max_disp=192,
                 loss_weight=1.0,
                 loss_name='loss_stereo_sequence'):
        super().__init__()
        self.loss_gamma = loss_gamma
        self.max_disp = max_disp
        self.loss_weight = loss_weight
        self._loss_name = loss_name
    
    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None, **kwargs):
        """
        前向传播计算序列损失。
        
        Args:
            pred (list[Tensor]): 预测列表
            target (Tensor): 真实视差图 (N, 1, H, W)
            weight (Tensor, optional): 像素权重
            avg_factor (float, optional): 平均因子
            reduction_override (str, optional): 归约方式覆盖
            **kwargs: 其他参数
            
        Returns:
            list[Tensor]: 损失列表
        """
        # 生成有效像素掩码
        valid_mask = ((target > 0.) & (target < self.max_disp))
        
        # 计算序列损失
        losses = stereo_sequence_loss(
            pred, 
            target, 
            valid_mask, 
            loss_gamma=self.loss_gamma
        )
        
        # 应用损失权重
        losses = [self.loss_weight * loss for loss in losses]
        
        return losses
    
    @property
    def loss_name(self):
        """返回损失名称"""
        return self._loss_name


@MODELS.register_module()
class StereoSequenceLossWithValidMask(nn.Module):
    """带有效掩码的DStereoPlus序列损失函数。
    
    当输入已经包含有效掩码时使用此版本。
    
    Args:
        loss_gamma (float): 损失权重衰减因子。默认0.9。
        loss_weight (float): 损失权重。默认1.0。
        loss_name (str): 损失名称。默认'loss_stereo_sequence_with_mask'。
    """
    
    def __init__(self,
                 loss_gamma=0.9,
                 loss_weight=1.0,
                 loss_name='loss_stereo_sequence_with_mask'):
        super().__init__()
        self.loss_gamma = loss_gamma
        self.loss_weight = loss_weight
        self._loss_name = loss_name
    
    def forward(self, pred, target, valid_mask, weight=None, avg_factor=None, reduction_override=None, **kwargs):
        """
        前向传播计算序列损失。
        
        Args:
            pred (list[Tensor]): 预测列表
            target (Tensor): 真实视差图 (N, 1, H, W)
            valid_mask (Tensor): 有效像素掩码 (N, 1, H, W)
            weight (Tensor, optional): 像素权重
            avg_factor (float, optional): 平均因子
            reduction_override (str, optional): 归约方式覆盖
            **kwargs: 其他参数
            
        Returns:
            list[Tensor]: 损失列表
        """
        # 计算序列损失
        losses = stereo_sequence_loss(
            pred, 
            target, 
            valid_mask, 
            loss_gamma=self.loss_gamma
        )
        
        # 应用损失权重
        losses = [self.loss_weight * loss for loss in losses]
        
        return losses
    
    @property
    def loss_name(self):
        """返回损失名称"""
        return self._loss_name