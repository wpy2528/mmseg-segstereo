# Copyright (c) Horizon Robotics. All rights reserved.

import copy
from typing import Dict, Sequence

import torch.nn as nn
from horizon_plugin_pytorch.quantization import QuantStub
from torch.quantization import DeQuantStub

from .models.backbones.efficientnet import (
    DEFAULT_BLOCKS_ARGS,
    BlockArgs,
    EfficientNet,
    round_filters,
)
from .models.base_modules.conv_module import ConvModule2d
from .registry import OBJECT_REGISTRY, build_from_registry

__all__ = ["SNDREfficientnet"]


@OBJECT_REGISTRY.register
class SNDREfficientnet(EfficientNet):
    """
    A module of efficientnet for snapdragon SoC.

    Args:
        model_type (str): Select to use which EfficientNet(B0-B7 or lite0-4), \
            for EfficientNet model, model_type must be one of: \
              ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7'], \
            for EfficientNet-lite model, model_type must be one of: \
              ['lite0', 'lite1', 'lite2', 'lite3', 'lite4'].
        coefficient_params (tuple): Parameter coefficients of EfficientNet, \
            include: \
              width_coefficient(float): scaling coefficient for net width. \
              depth_coefficient(float): scaling coefficient for net depth. \
              default_resolution(int): default input image size. \
              dropout_rate(float): dropout rate for final classifier layer. \
        num_classes (int): Num classes of output layer.
        mod1_channels (int): Default to 32.
            The out_channels of the first conv \
            (mod1 of efficientnet). If 'lite' is not in model_type, \
            the mod1_channels will be multiplied by width_coefficient.
        mod3_channels (int): Default to 1280.
            The out_channels of the mod3.
            If 'lite' is not in model_type, \
            the value will be multiplied by width_coefficient.
        downsample_times (int): The total number of downsamples, \
            is equal to the number of stride to 2 in the config plus 1
        bn_kwargs (dict): Dict for Bn layer.
        bias (bool): Whether to use bias in module.
        drop_connect_rate (float): Dropout rate at skip connections.
        depth_division (int): Depth division, Defaults to 8.
        activation (str): Activation layer, defaults to 'relu'.
        use_se_block (bool): Whether to use SEBlock in module.
        blocks_args (list): A list of BlockArgs to MBConvBlock modules.
        include_top (bool): Whether to include output layer.
        flat_output (bool): Whether to view the output tensor.
        input_channels (int): Input channels of first conv.
        split_expand_conv (bool): Whether split expand conv into two conv. Set
            to true when expand conv is too large to deploy on xj3.
    """

    def __init__(
        self,
        model_type: str,
        coefficient_params: tuple,
        num_classes: int,
        mod1_channels: int = 32,
        mod3_channels: int = 1280,
        downsample_times: int = 5,
        bn_kwargs: dict = None,
        bias: bool = False,
        drop_connect_rate: float = None,
        depth_division: int = 8,
        activation: str = "relu",
        use_se_block: bool = False,
        blocks_args: Sequence[Dict] = DEFAULT_BLOCKS_ARGS,
        include_top: bool = True,
        flat_output: bool = True,
        input_channels: int = 3,
        resolution: int = 0,
        split_expand_conv: bool = False,
    ):
        super(SNDREfficientnet, self).__init__(
            model_type, coefficient_params, num_classes
        )
        assert activation in ["relu", "relu6", "swish"], (
            f'activation must be one of ["relu", "relu6", "swish"], but '
            f"get {activation}"
        )

        self.model_type = model_type
        self.use_lite = True if "lite" in self.model_type else False
        if self.use_lite:
            assert (
                use_se_block is False and not activation == "swish"
            ), '"Swish" activation and "Squeeze-and-excitation" block \
                    cannot be used in EfficientNet-lite model'

        (
            self.width_coefficient,
            self.depth_coefficient,
            self.default_resolution,
            self.dropout_rate,
        ) = coefficient_params
        self.drop_connect_rate = drop_connect_rate
        self.depth_division = depth_division
        self.num_classes = num_classes
        self.include_top = include_top
        self.flat_output = flat_output
        self.split_expand_conv = split_expand_conv
        self.mod1_channels = mod1_channels
        self.mod3_channels = mod3_channels
        self.downsample_times = downsample_times
        self.blocks_args = [
            BlockArgs(**block_args)
            if isinstance(block_args, dict)
            else block_args
            for block_args in blocks_args
        ]
        self.activation = activation
        act_layer = build_from_registry(
            dict(type=activation, inplace=True)  # noqa
        )

        if bn_kwargs is not None:
            self.bn_kwargs = bn_kwargs
        else:
            batch_norm_momentum = 0.99
            batch_norm_epsilon = 1e-3
            self.bn_kwargs = {
                "momentum": 1.0 - batch_norm_momentum,
                "eps": batch_norm_epsilon,
            }
        self.use_se_block = use_se_block

        if self.use_lite:
            out_planes = self.mod1_channels
        else:
            out_planes = round_filters(
                self.mod1_channels, self.width_coefficient, self.depth_division
            )

        self.mod1 = nn.Sequential(
            ConvModule2d(
                in_channels=input_channels,
                out_channels=out_planes,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=bias,
                norm_layer=nn.BatchNorm2d(out_planes, **self.bn_kwargs),
                act_layer=copy.deepcopy(act_layer),
            )
        )

        self.mod2 = self._make_stage(act_layer)

        in_planes = round_filters(
            self.blocks_args[-1].out_filters,
            self.width_coefficient,
            self.depth_division,
        )
        if self.use_lite:
            out_planes = self.mod3_channels
        else:
            out_planes = round_filters(
                self.mod3_channels, self.width_coefficient, self.depth_division
            )

        if self.include_top:
            self.mod3 = nn.Sequential(
                ConvModule2d(
                    in_channels=in_planes,
                    out_channels=out_planes,
                    kernel_size=1,
                    bias=bias,
                    norm_layer=nn.BatchNorm2d(out_planes, **self.bn_kwargs),
                    act_layer=copy.deepcopy(act_layer),
                )
            )

            self.output = nn.Sequential(
                nn.AvgPool2d(
                    self.default_resolution // (2 ** self.downsample_times)
                ),
                nn.Dropout2d(self.dropout_rate),
                ConvModule2d(
                    in_channels=out_planes,
                    out_channels=num_classes,
                    kernel_size=1,
                    bias=bias,
                    norm_layer=nn.BatchNorm2d(num_classes, **self.bn_kwargs),
                ),
            )
        else:
            self.output = None

        self.quant = QuantStub(scale=1.0 / 128.0)
        self.dequant = DeQuantStub()
