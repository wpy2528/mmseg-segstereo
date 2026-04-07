# Copyright (c) OpenMMLab. All rights reserved.
"""SegStereo 视差分支：复用 IGEV-Stereo 完整匹配器（与 DepthEstimator 中 backbone 一致）。"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule

from mmseg.models.utils import resize
from mmseg.registry import MODELS


def _pad_spatial_to_multiple(x: torch.Tensor, m: int) -> tuple:
    """将 (B,C,H,W) 的 H、W pad 到 m 的整数倍。返回 (x_pad, (orig_h, orig_w))。"""
    _, _, h, w = x.shape
    pad_h = (m - h % m) % m
    pad_w = (m - w % m) % m
    if pad_h == 0 and pad_w == 0:
        return x, (h, w)
    # F.pad: (left, right, top, bottom)
    x = F.pad(x, (0, pad_w, 0, pad_h), mode='constant', value=0.0)
    return x, (h, w)


# IGEV match 特征经 desc 后为 96 通道；GWC 的 groupwise 要求 C % num_groups == 0（num_groups=8）
_BASE_MATCH_CH = 96

# hourglass 内 FeatureAtt 对 IGEV 图像特征的通道数（见 rt_igev_stereo.hourglass）
_IGEV_FEAT_HG8 = 64
_IGEV_FEAT_HG16 = 192


@MODELS.register_module()
class SegStereoIGEVDisparityBranch(BaseModule):
    """IGEV-Stereo 作为视差子网络；可选三级语义相关嵌入。

    1. **match**：分割 logits 经 ``sem_proj`` 后与 ``match`` 拼接（与单级融合一致）。
    2. **hourglass 1/8**：STDC 特征（默认 ``x[2]``）经 ``sem_add_hg8`` 映射到 64 维，与
       ``features_left[1]`` **残差相加**（不改变通道数，兼容 ``FeatureAtt``）。
    3. **hourglass 1/16**：STDC 特征（默认 ``x[1]``）映射到 192 维，与 ``features_left[2]`` 残差相加。

    三级可分别用配置 ``fuse_semantic_level_*`` 关闭。

    Args:
        fuse_level_match (bool): 是否启用 match 级 logits 拼接。
        fuse_level_hg8 (bool): 是否启用 ``features[1]`` 残差注入。
        fuse_level_hg16 (bool): 是否启用 ``features[2]`` 残差注入。
        stdc_in_channels_hg8 (int): STDC 特征 ``x[stdc_index_hg8]`` 的通道数（默认 128，与 ARM 一致）。
        stdc_in_channels_hg16 (int): STDC 特征 ``x[stdc_index_hg16]`` 的通道数。
    """

    def __init__(self,
                 max_disp: int = 192,
                 mixed_precision: bool = False,
                 precision_dtype: str = 'float32',
                 multiple: int = 32,
                 align_corners: bool = False,
                 use_gru: bool = True,
                 gru_iters: Optional[int] = None,
                 semantic_channels: int = 0,
                 fused_channels: int = 32,
                 fuse_level_match: bool = True,
                 fuse_level_hg8: bool = True,
                 fuse_level_hg16: bool = True,
                 stdc_in_channels_hg8: int = 128,
                 stdc_in_channels_hg16: int = 128,
                 norm_cfg=None,
                 act_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        from mmseg.models.backbones.igevpp_rt.rt_igev_stereo import IGEVStereo

        if norm_cfg is None:
            norm_cfg = dict(type='BN', requires_grad=True)
        if act_cfg is None:
            act_cfg = dict(type='ReLU')

        self.fuse_level_match = bool(fuse_level_match)
        self.fuse_level_hg8 = bool(fuse_level_hg8)
        self.fuse_level_hg16 = bool(fuse_level_hg16)

        self.semantic_channels = int(semantic_channels)
        self.fused_channels = int(fused_channels)
        if self.semantic_channels > 0:
            total_ch = _BASE_MATCH_CH + self.fused_channels
            if total_ch % 8 != 0:
                raise ValueError(
                    'IGEV 语义融合要求 (96 + fused_channels) 能被 8 整除，当前 '
                    f'fused_channels={self.fused_channels}，合计通道为 {total_ch}')
            self.sem_proj = ConvModule(
                self.semantic_channels,
                self.fused_channels,
                1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
        else:
            self.sem_proj = None

        cin8 = int(stdc_in_channels_hg8)
        cin16 = int(stdc_in_channels_hg16)
        # 残差支路：无 ReLU，避免破坏原特征分布
        self.sem_add_hg8 = nn.Sequential(
            nn.Conv2d(cin8, _IGEV_FEAT_HG8, 1, bias=False),
            nn.BatchNorm2d(_IGEV_FEAT_HG8)) if cin8 > 0 else None
        self.sem_add_hg16 = nn.Sequential(
            nn.Conv2d(cin16, _IGEV_FEAT_HG16, 1, bias=False),
            nn.BatchNorm2d(_IGEV_FEAT_HG16)) if cin16 > 0 else None

        self.igev = IGEVStereo()
        self.igev.args.max_disp = max_disp
        self.igev.args.mixed_precision = mixed_precision
        self.igev.args.precision_dtype = precision_dtype
        if gru_iters is not None:
            self.igev.args.gru_iters = int(gru_iters)

        self.use_gru = bool(use_gru)
        self.multiple = multiple
        self.max_disp = max_disp
        self.align_corners = align_corners

    def forward(self,
                left: torch.Tensor,
                right: torch.Tensor,
                left_semantic: torch.Tensor = None,
                stdc_feat_hg8: torch.Tensor = None,
                stdc_feat_hg16: torch.Tensor = None):
        if self.semantic_channels > 0:
            if left_semantic is not None and left_semantic.shape[
                    1] != self.semantic_channels:
                raise ValueError(
                    f'left_semantic 通道数为 {left_semantic.shape[1]}，'
                    f'期望为 {self.semantic_channels}')
        elif left_semantic is not None:
            raise ValueError(
                'semantic_channels=0 时不要传入 left_semantic（SegStereoIGEVDisparityBranch）')

        _, _, h0, w0 = left.shape
        left_p, _ = _pad_spatial_to_multiple(left, self.multiple)
        right_p, _ = _pad_spatial_to_multiple(right, self.multiple)

        sem_p = None
        if self.semantic_channels > 0 and left_semantic is not None:
            sem_p = left_semantic
            if sem_p.shape[2:] != left_p.shape[2:]:
                sem_p = resize(
                    sem_p,
                    size=left_p.shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners)

        iters = None if self.use_gru else 0
        disp = self.igev.forward_inner(
            (left_p, right_p),
            iters=iters,
            test_mode=False,
            left_semantic=sem_p,
            sem_proj=self.sem_proj,
            align_corners=self.align_corners,
            fuse_level_match=self.fuse_level_match,
            fuse_level_hg8=self.fuse_level_hg8,
            fuse_level_hg16=self.fuse_level_hg16,
            stdc_feat_hg8=stdc_feat_hg8,
            stdc_feat_hg16=stdc_feat_hg16,
            sem_add_hg8=self.sem_add_hg8,
            sem_add_hg16=self.sem_add_hg16)
        disp = disp[:, :, :h0, :w0]
        return disp
