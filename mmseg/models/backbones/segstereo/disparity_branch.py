# Copyright (c) OpenMMLab. All rights reserved.
"""SegStereo 视差分支：孪生特征、代价体、二维卷积聚合、soft-argmin 视差回归。"""

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule

from mmseg.registry import MODELS
from ...utils import resize


def _build_correlation_volume(ref_fea: torch.Tensor, tgt_fea: torch.Tensor,
                              max_disp: int) -> torch.Tensor:
    """按通道求均值的相关代价体，形状 (B, D, H, W)，视差维作为通道。"""
    b, c, h, w = ref_fea.shape
    cost = ref_fea.new_zeros(b, max_disp, h, w)
    for d in range(max_disp):
        if d > 0:
            prod = ref_fea[:, :, :, d:] * tgt_fea[:, :, :, :-d]
            cost[:, d, :, d:] = prod.mean(dim=1)
        else:
            cost[:, d] = (ref_fea * tgt_fea).mean(dim=1)
    return cost


def _disparity_regression_softmax(logits: torch.Tensor, disp_values: torch.Tensor) -> torch.Tensor:
    """在视差维度（dim=1）上做 soft-argmin。

    Args:
        logits: (B, D, H, W)
        disp_values: (D,) 每个下标对应的视差取值（当前尺度下，如 0..D-1 映射到全分辨率像素）
    """
    prob = torch.softmax(logits, dim=1)
    disp = torch.sum(prob * disp_values.view(1, -1, 1, 1), dim=1, keepdim=True)
    return disp


class _StereoFeatNet(nn.Module):
    """轻量孪生骨干：输出为输入尺寸的 1/4，特征通道数为 ``out_channels``。"""

    def __init__(self, in_channels: int, out_channels: int, norm_cfg: dict,
                 act_cfg: dict):
        super().__init__()
        self.conv0 = ConvModule(
            in_channels, 32, 7, stride=2, padding=3, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv1 = ConvModule(
            32, 32, 3, stride=1, padding=1, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv2 = ConvModule(
            32, out_channels, 3, stride=2, padding=1, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv3 = ConvModule(
            out_channels, out_channels, 3, stride=1, padding=1, norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class _CostAgg2D(nn.Module):
    """在代价体上用二维卷积聚合：视差维 D 作为通道，混合空间邻域与跨视差信息。"""

    def __init__(self, num_disp: int, hidden: int):
        super().__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(num_disp, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.classify = nn.Conv2d(hidden, num_disp, kernel_size=3, padding=1, bias=True)

    def forward(self, cost: torch.Tensor) -> torch.Tensor:
        """cost: (B, D, H, W) -> logits (B, D, H, W)。"""
        x = self.conv0(cost)
        x = self.conv1(x)
        return self.classify(x)


@MODELS.register_module()
class SegStereoDisparityBranch(BaseModule):
    """SegStereo 联合训练中的视差子网络。

    流程（典型 GC-Net / PSM 类）：
    1/4 分辨率共享孪生特征 -> 相关代价体 -> 二维卷积在 (H,W) 上聚合（视差作通道）->
    softmax + soft-argmin -> 上采样回输入分辨率。

    语义融合（可选）：传入 ``left_semantic``（如分割分支的 logits 或特征，
    空间尺寸可与原图一致，内部会 resize）。先经投影再与左右特征图在通道维拼接后建代价体，
    使立体路径能利用分割线索，与 SegStereo 论文思路一致。

    Args:
        max_disp (int): **全分辨率**下的最大视差。代价体在 1/4 尺度构建，
            视差档位数为 ``max_disp // 4``（至少为 1）。
        feature_channels (int): 孪生特征通道数，默认 64。
        semantic_channels (int): 大于 0 时启用融合，此时 ``left_semantic`` 通道数须与此一致。
            默认 0 表示不融合。
        fused_channels (int): 融合时语义经 1x1 投影后的通道数。
        cost_hidden (int): 代价聚合模块中二维卷积的隐藏通道数。
        norm_cfg (dict): 二维卷积的归一化配置。
        act_cfg (dict): 二维卷积的激活配置。
        align_corners (bool): resize 语义图时是否 align_corners。
    """

    def __init__(self,
                 max_disp: int = 192,
                 in_channels: int = 3,
                 feature_channels: int = 64,
                 semantic_channels: int = 0,
                 fused_channels: int = 32,
                 cost_hidden: int = 16,
                 norm_cfg=None,
                 act_cfg=None,
                 align_corners: bool = False,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        if norm_cfg is None:
            norm_cfg = dict(type='BN', requires_grad=True)
        if act_cfg is None:
            act_cfg = dict(type='ReLU')

        self.max_disp = max_disp
        self.feature_channels = feature_channels
        self.semantic_channels = semantic_channels
        self.fused_channels = fused_channels
        self.align_corners = align_corners

        md_q = max(max_disp // 4, 1)
        self.max_disp_quarter = md_q

        self.feature = _StereoFeatNet(in_channels, feature_channels, norm_cfg, act_cfg)
        if semantic_channels > 0:
            self.sem_proj = ConvModule(
                semantic_channels,
                fused_channels,
                1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
        else:
            self.sem_proj = None

        self.cost_agg = _CostAgg2D(num_disp=md_q, hidden=cost_hidden)

        # 1/4 尺度下标 i 对应全分辨率视差 i*4（像素）
        disp_vals = torch.arange(
            0, md_q, dtype=torch.float32) * 4.0
        self.register_buffer('disp_values_quarter', disp_vals)

    def extract_features(self, left: torch.Tensor,
                         right: torch.Tensor) -> tuple:
        """提取 1/4 分辨率的孪生特征（若需语义融合请在 forward 中拼接）。"""
        return self.feature(left), self.feature(right)

    def forward(self,
                left: torch.Tensor,
                right: torch.Tensor,
                left_semantic: torch.Tensor = None):
        """计算左视图视差图。

        Args:
            left (Tensor): 左图 (B, 3, H, W)
            right (Tensor): 右图 (B, 3, H, W)
            left_semantic (Tensor, optional): 左视图语义 (B, C_sem, H, W)，
                启用融合时需满足 C_sem == semantic_channels。

        Returns:
            Tensor: 预测视差 (B, 1, H, W)，与输入同分辨率。
        """
        fl = self.feature(left)
        fr = self.feature(right)

        if self.semantic_channels > 0:
            if left_semantic is None:
                raise ValueError(
                    'semantic_channels>0 时必须在 forward 中传入 left_semantic')
            if left_semantic.shape[1] != self.semantic_channels:
                raise ValueError(
                    f'left_semantic 通道数为 {left_semantic.shape[1]}，'
                    f'期望为 {self.semantic_channels}')
            sem = resize(
                left_semantic,
                size=fl.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            sem = self.sem_proj(sem)
            fl = torch.cat([fl, sem], dim=1)
            fr = torch.cat([fr, sem], dim=1)

        cost = _build_correlation_volume(fl, fr, self.max_disp_quarter)
        logits = self.cost_agg(cost)
        disp_q = _disparity_regression_softmax(
            logits, self.disp_values_quarter.to(device=logits.device))
        disp = resize(
            disp_q,
            size=left.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return disp
