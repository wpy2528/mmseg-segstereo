# Copyright (c) Horizon Robotics. All rights reserved.
import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from .models.backbones.mixvargenet import IdentityConfig, MixVarGENet
from .models.base_modules.basic_mixvargenet_module import (
    BasicMixVarGEBlock,
    MixVarGEBlock,
)
from .models.base_modules.conv_module import ConvModule2d
from .models.base_modules.extend_container import ExtSequential
from .registry import OBJECT_REGISTRY

__all__ = ["SNDRMixDWNet"]

BLOCK_CONFIG = {
    "mixdw_k3k3_f1": {"conv3_kernel_size": 3, "padding": 1, "factor": 1},
    "mixdw_k3k3_f2": {"conv3_kernel_size": 3, "padding": 1, "factor": 2},
    "mixdw_k3k3_f4": {"conv3_kernel_size": 3, "padding": 1, "factor": 4},
    "mixdw_f1": {"padding": 0, "factor": 1},
    "mixdw_f2": {"padding": 0, "factor": 2},
    "mixdw_f4": {"padding": 0, "factor": 4},
    "mixdw_f1_dw": {
        "padding": 0,
        "factor": 1,
        "depthwise_conv": True,
    },
    "mixdw_f2_dw": {
        "padding": 0,
        "factor": 2,
        "depthwise_conv": True,
    },
    "mixdw_f4_dw": {
        "padding": 0,
        "factor": 4,
        "depthwise_conv": True,
    },
    "mixdw_f1_r": {
        "padding": 0,
        "factor": 1,
        "merge_branch": True,
    },
    "mixdw_f1_r_dw": {
        "padding": 0,
        "factor": 1,
        "depthwise_conv": True,
        "merge_branch": True,
    },
    "mixdw_f2_r": {
        "padding": 0,
        "factor": 2,
        "merge_branch": True,
    },
    "mixdw_f2_r_dw": {
        "padding": 0,
        "factor": 2,
        "depthwise_conv": True,
        "merge_branch": True,
    },
    "mixdw_f4_r": {
        "padding": 0,
        "factor": 4,
        "merge_branch": True,
    },
    "mixdw_f4_r_dw": {
        "padding": 0,
        "factor": 4,
        "depthwise_conv": True,
        "merge_branch": True,
    },
}


@OBJECT_REGISTRY.register
class SNDRMixDWNet(MixVarGENet):
    """Module of MixVarGENet.

    Inherit from MixVarGENet, the parameters are the same as it.
    This model is based on a design using full convolution for small channels
    and deep convolution for large channels. Because full convolution has the
    same speed and better performance as deep convolution on small channels,
    and deep convolution has faster speed on large channels.
    """

    def _make_stage(self, stage_config):
        def _get_fusion_channels(fusion_strides):
            if len(fusion_strides) == 0:
                return []
            strides_ids = map(
                lambda stride: int(math.log2(stride) - 1), fusion_strides
            )
            fusion_channels = map(
                lambda idx: self.net_config[idx][0].out_channels, strides_ids
            )
            return list(fusion_channels)

        layers = []
        for config_i in stage_config:
            if isinstance(config_i, IdentityConfig):
                layers.append(nn.Identity())
            else:
                layers.append(
                    MixDWBlock(
                        in_ch=config_i.in_channels,
                        block_ch=config_i.out_channels,
                        head_op=config_i.head_op,
                        stack_ops=config_i.stack_ops,
                        stack_factor=config_i.stack_factor,
                        stride=config_i.stride,
                        bias=self.bias,
                        fusion_channels=_get_fusion_channels(
                            config_i.fusion_strides
                        ),
                        downsample_num=config_i.extra_downsample_num,
                        bn_kwargs=self.bn_kwargs,
                    )
                )
        return ExtSequential(layers)


class MixDWBlock(MixVarGEBlock):
    """
    A block for MixDWBlock.

    Args:
        in_ch: The in_channels for the block.
        block_ch: The out_channels for the block.
        head_op: One key of the BLOCK_CONFIG.
        stack_ops: a list consisting the keys of the
            BLOCK_CONFIG, or be None.
        stack_factor: channel factor of middle stack ops.
            Input and output channels of stack remains the same.
        stride: Stride of basic block.
        bias: Whether to use bias in basic block.
        bn_kwargs: Dict for BN layer.
        fusion_channels: A list of fusion layer input channels,
            which should be former block's downsampled feature map channel.
            Eg, fusion_channels of stride16's head block
            could contains channels of stride4 or stride8.
        downsample_num: Downsampled feature maps of current block's
            output feature map,
            which could be next few blocks' shortcut input.
            Currently 0, 1, 2, 3 are supported.
            0 means no backbone fusion,
            1 means fusion 2x scales,
            2 means fusion 2x and 4x scales.
            3 means fusion 2x and 4x and 8x scales.
        output_downsample: To controll whether ouput downsample feature list.
    """

    def __init__(
        self,
        in_ch: int,
        block_ch: int,
        head_op: str,
        stack_ops: List[str],
        stride: int,
        bias: bool,
        bn_kwargs: dict,
        stack_factor: Optional[int] = 1,
        fusion_channels: Optional[Union[Tuple[int], List[int]]] = (),
        downsample_num: Optional[int] = 0,
        output_downsample: Optional[bool] = True,
    ):
        assert downsample_num in [0, 1, 2, 3]
        super(MixDWBlock, self).__init__(
            in_ch,
            block_ch,
            "mixvarge_f1",
            ["mixvarge_f1"],
            stride,
            bias,
            bn_kwargs,
            stack_factor,
            fusion_channels,
            downsample_num,
            output_downsample,
        )
        self.ouput_downsample = output_downsample
        self.head_layer = BasicMixDWBlock(
            in_channels=in_ch,
            out_channels=block_ch,
            stride=stride,
            bias=bias,
            bn_kwargs=bn_kwargs,
            fusion_channels=fusion_channels,
            **BLOCK_CONFIG[head_op],
        )

        modules = []

        stack_num = len(stack_ops)
        for i, stack_op in enumerate(stack_ops):
            if stack_num > 1:
                if i == 0:
                    input_channel = block_ch
                    output_channel = stack_factor * block_ch
                elif i == stack_num - 1:
                    input_channel = stack_factor * block_ch
                    output_channel = block_ch
                else:
                    input_channel = stack_factor * block_ch
                    output_channel = stack_factor * block_ch
            else:
                input_channel = block_ch
                output_channel = block_ch

            modules.append(
                BasicMixDWBlock(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    stride=1,
                    bias=bias,
                    bn_kwargs=bn_kwargs,
                    **BLOCK_CONFIG[stack_op],
                )
            )

        self.stack_layers = ExtSequential(modules)

        downsample_fusion_layers = []
        for _idx in range(downsample_num):
            downsample_fusion_layers.append(
                ConvModule2d(
                    block_ch,
                    block_ch,
                    kernel_size=3,
                    padding=1,
                    stride=2,
                    norm_layer=nn.BatchNorm2d(block_ch, **bn_kwargs),
                )
            )
        self.downsample_fusion_layers = (
            ExtSequential(downsample_fusion_layers) if downsample_num else None
        )


class BasicMixDWBlock(BasicMixVarGEBlock):
    """
    A basic block for MixDWBlock.

    Args:
        in_channels (int): The in_channels for the block.
        out_channels (int): The out_channels for the block.
        stride (int): Stride of basic block.
        bias (bool): Whether to use bias in basic block.
        bn_kwargs (dict): Dict for BN layer.
        kernel_size (int): Kernel size of basic block.
        padding (int): Padding of basic block.
        factor (int): Factor for channels expansion.
        depthwise_conv (bool): Whether use depthwise conv.
        fusion_channels (list): Channels of input fusion layer.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        stride,
        bias,
        bn_kwargs,
        conv1_kernel_size=1,
        conv2_kernel_size=3,
        conv3_kernel_size=1,
        padding=0,
        factor=1,
        depthwise_conv=False,
        fusion_channels=(),
        merge_branch=False,
        fuse_2x=False,
    ):
        super(BasicMixDWBlock, self).__init__(
            in_channels,
            out_channels,
            stride,
            bias,
            bn_kwargs,
            conv2_kernel_size,
            conv3_kernel_size,
            padding,
            factor,
        )
        self.depthwise_conv = depthwise_conv
        self.fuse_2x = fuse_2x and stride == 2

        mid_channle = (
            out_channels if factor == 1 else int(in_channels * factor)
        )

        block_type = ConvModule2d
        conv2_pad = (conv2_kernel_size - 1) // 2

        if depthwise_conv:
            self.conv = nn.Sequential(
                block_type(
                    in_channels,
                    mid_channle,
                    conv1_kernel_size,
                    padding=padding,
                    stride=1,
                    groups=1,
                    bias=bias,
                    norm_layer=nn.BatchNorm2d(mid_channle, **bn_kwargs),
                    act_layer=nn.ReLU(inplace=True),
                ),
                block_type(
                    mid_channle,
                    mid_channle,
                    conv2_kernel_size,
                    padding=conv2_pad,
                    stride=stride,
                    groups=mid_channle,
                    bias=bias,
                    norm_layer=nn.BatchNorm2d(mid_channle, **bn_kwargs),
                    act_layer=nn.ReLU(inplace=True),
                ),
                block_type(
                    mid_channle,
                    out_channels,
                    conv3_kernel_size,
                    padding=padding,
                    stride=1,
                    groups=1,
                    bias=bias,
                    norm_layer=nn.BatchNorm2d(out_channels, **bn_kwargs),
                ),
            )
        else:
            self.conv = nn.Sequential(
                block_type(
                    in_channels,
                    mid_channle,
                    conv2_kernel_size,
                    padding=conv2_pad,
                    stride=stride,
                    groups=1,
                    bias=bias,
                    norm_layer=nn.BatchNorm2d(mid_channle, **bn_kwargs),
                    act_layer=nn.ReLU(inplace=True),
                ),
                block_type(
                    mid_channle,
                    out_channels,
                    conv3_kernel_size,
                    padding=padding,
                    stride=1,
                    groups=1,
                    bias=bias,
                    norm_layer=nn.BatchNorm2d(out_channels, **bn_kwargs),
                ),
            )

        fusion_layers = []
        for input_channel in fusion_channels:
            layer = ConvModule2d(
                input_channel,
                out_channels,
                kernel_size=1,
                padding=0,
                stride=1,
                norm_layer=nn.BatchNorm2d(out_channels, **bn_kwargs),
            )
            fusion_layers.append(layer)
        self.fusion_layers = (
            ExtSequential(fusion_layers) if len(fusion_channels) else None
        )

        self.fusion_relu = (
            nn.ReLU(inplace=True) if len(fusion_channels) else None
        )

        self.fusion_adds = ExtSequential(
            [
                nn.quantized.FloatFunctional()
                for i in range(len(fusion_channels))
            ]
        )

        self.downsample = None
        if merge_branch or not (stride == 1 and in_channels == out_channels):
            self.downsample = ConvModule2d(
                in_channels,
                out_channels,
                1,
                stride=stride,
                norm_layer=nn.BatchNorm2d(out_channels, **bn_kwargs),
            )
        self.relu = nn.ReLU(inplace=True)
        self.short_add = nn.quantized.FloatFunctional()

    def fuse_model(self):
        # for qat mappings
        from horizon_plugin_pytorch import quantization

        if self.downsample is not None:
            self.downsample.fuse_model()

        if self.depthwise_conv:
            self.conv[0].fuse_model()
            self.conv[1].fuse_model()
            fuse_list = ["conv.2.0", "conv.2.1"]
        else:
            self.conv[0].fuse_model()
            fuse_list = ["conv.1.0", "conv.1.1"]
        fuse_list.extend(["short_add", "relu"])

        torch.quantization.fuse_modules(
            self,
            fuse_list,
            inplace=True,
            fuser_func=quantization.fuse_known_modules,
        )

        if self.fusion_layers is not None:
            for idx in range(len(self.fusion_layers) - 1):
                fusion_list = [
                    f"fusion_layers.{idx}.0",
                    f"fusion_layers.{idx}.1",
                    f"fusion_adds.{idx}",
                ]
                torch.quantization.fuse_modules(
                    self,
                    fusion_list,
                    inplace=True,
                    fuser_func=quantization.fuse_known_modules,
                )
            idx = len(self.fusion_layers) - 1
            fusion_list = [
                f"fusion_layers.{idx}.0",
                f"fusion_layers.{idx}.1",
                f"fusion_adds.{idx}",
                "fusion_relu",
            ]
            torch.quantization.fuse_modules(
                self,
                fusion_list,
                inplace=True,
                fuser_func=quantization.fuse_known_modules,
            )
