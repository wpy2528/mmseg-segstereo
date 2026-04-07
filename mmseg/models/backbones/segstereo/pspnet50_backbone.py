# Copyright (c) OpenMMLab. All rights reserved.
"""SegStereo 语义分支编码器（PSPNet-50 设定），基于 mmseg 的 ResNet 实现。"""

from mmseg.registry import MODELS
from ..resnet import ResNetV1c


@MODELS.register_module()
class SegStereoPSPNet50Backbone(ResNetV1c):
    """SegStereo **语义**分支所用的 ResNet-50 骨干（PSPNet-50）。

    与 ``configs/_base_/models/pspnet_r50-d8.py`` 中的空洞 ResNet 一致：
    第四阶段 stride 为 1，空洞为 (1, 1, 2, 4)，多尺度输出 ``out_indices=(0,1,2,3)``，
    供 PSP 头使用（下标 3，2048 通道），可选 FCN 辅助头（下标 2，1024 通道）。

    若之后将语义编码器换为 STDC：将配置中的 ``backbone`` 改为 ``STDCContextPathNet``
    （或你的 STDC 变体），并把 ``decode_head`` / ``auxiliary_head`` 的 ``in_channels``、
    ``in_index`` 对齐到 STDC 各层特征；通道匹配时解码头仍可继续使用 ``PSPHead``。
    """

    def __init__(self,
                 depth=50,
                 num_stages=4,
                 out_indices=(0, 1, 2, 3),
                 dilations=(1, 1, 2, 4),
                 strides=(1, 2, 1, 1),
                 style='pytorch',
                 contract_dilation=True,
                 **kwargs):
        super().__init__(
            depth=depth,
            num_stages=num_stages,
            out_indices=out_indices,
            dilations=dilations,
            strides=strides,
            style=style,
            contract_dilation=contract_dilation,
            **kwargs)
