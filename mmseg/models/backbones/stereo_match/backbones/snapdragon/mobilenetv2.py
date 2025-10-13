from typing import Dict, List, Union

import torch.nn as nn
from horizon_plugin_pytorch.quantization import QuantStub
from torch.quantization import DeQuantStub

from .models.backbones.mobilenetv2 import MobileNetV2
from .models.base_modules.conv_module import ConvModule2d
from .models.base_modules.inverted_residual import InvertedResidual
from .registry import OBJECT_REGISTRY

__all__ = ["SNDRMobileNetV2"]


@OBJECT_REGISTRY.register
class SNDRMobileNetV2(MobileNetV2):
    """A module of mobilenetv2 for snapdragon SoC.

    Default channels in mobilenetv2 1.0:
        >>> in_chls = [
        ...   [32],
        ...   [16, 24],
        ...   [24, 32, 32],
        ...   [32] + [64] * 4 + [96] * 2,
        ...   [96] + [160] * 3,
        ... ],
        ... out_chls = [
        ...   [16],
        ...   [24, 24],
        ...   [32, 32, 32],
        ...   [64] * 4 + [96] * 3,
        ...   [160] * 3 + [320],
        ... ]

    Depthwise Convolution on the DSP is not optimized for all cases.
    The following case is optimized:
    - Horizontal stride is <= 2.
    - Filter is 3x3.
    - Depth is a multiple of 32.

    Args:
        num_classes: Num classes of output layer.
        bn_kwargs: Dict of BN layer.
        in_chls: Input channel list.
        out_chls: Output channel list.
        input_channels: Input channels of first conv.
        feat_size: Last feature map size. feat_size = input_size // 32.
            Defaults to 4.
        expand_ratio: expand_ratio of inverse residual block. Defaults to 6.
        alpha: Alpha for mobilenetv2. Defaults to 1.
        bias: Whether to use bias in module. Defaults to True.
        include_top: Whether to incldue output layer. Defaults to True.
        flat_output: Whether to view the output tensor. Defaults to True.
        use_dw_as_avgpool: Whether to replace AvgPool with DepthWiseConv.
            Defaults to False. Not recommended on Snapdragon.
    """

    def __init__(
        self,
        num_classes: int,
        bn_kwargs: Dict,
        in_chls: List[List[int]],
        out_chls: List[List[int]],
        input_channels: int = 3,
        feat_size: Union[List[int], int] = 4,
        expand_ratio: int = 6,
        alpha: float = 1,
        bias: bool = True,
        include_top: bool = True,
        flat_output: bool = True,
        use_dw_as_avgpool: bool = False,
    ):
        self.input_channels = input_channels
        super(SNDRMobileNetV2, self).__init__(num_classes, bn_kwargs)
        # Just for QAT training on BPU in case.
        self.quant = QuantStub(scale=1.0 / 128.0)
        self.dequant = DeQuantStub()

        self.alpha = alpha
        self.bias = bias
        self.bn_kwargs = bn_kwargs

        self.num_classes = num_classes
        self.include_top = include_top
        self.flat_output = flat_output
        self.use_dw_as_avgpool = use_dw_as_avgpool
        self.mod1 = self._make_stage(in_chls[0], out_chls[0], 1, True)
        self.mod2 = self._make_stage(in_chls[1], out_chls[1], expand_ratio)
        self.mod3 = self._make_stage(in_chls[2], out_chls[2], expand_ratio)
        self.mod4 = self._make_stage(in_chls[3], out_chls[3], expand_ratio)
        self.mod5 = self._make_stage(in_chls[4], out_chls[4], expand_ratio)

        if self.include_top:
            if self.use_dw_as_avgpool:
                pool_layer = ConvModule2d(
                    in_channels=max(1280, int(1280 * alpha)),
                    out_channels=max(1280, int(1280 * alpha)),
                    kernel_size=feat_size,
                    stride=1,
                    padding=0,
                    groups=max(1280, int(1280 * alpha)),
                )
            else:
                pool_layer = nn.AvgPool2d(feat_size)

            self.output = nn.Sequential(
                ConvModule2d(
                    int(out_chls[4][-1] * alpha),
                    max(1280, int(1280 * alpha)),
                    1,
                    bias=self.bias,
                    norm_layer=nn.BatchNorm2d(
                        max(1280, int(1280 * alpha)), **bn_kwargs
                    ),
                    act_layer=nn.ReLU(inplace=True),
                ),
                pool_layer,
                ConvModule2d(
                    max(1280, int(1280 * alpha)),
                    num_classes,
                    1,
                    bias=self.bias,
                    norm_layer=nn.BatchNorm2d(num_classes, **bn_kwargs),
                ),
            )
        else:
            self.output = None

    def _make_stage(self, in_chls, out_chls, expand_t, first_layer=False):
        layers = []
        in_chls = [int(chl * self.alpha) for chl in in_chls]
        out_chls = [int(chl * self.alpha) for chl in out_chls]
        for i, in_chl, out_chl in zip(range(len(in_chls)), in_chls, out_chls):
            stride = 2 if i == 0 else 1
            if first_layer:
                layers.append(
                    ConvModule2d(
                        # Adapt to the task whose input channel is not 3.
                        self.input_channels,
                        in_chls[0],
                        3,
                        stride,
                        1,
                        bias=self.bias,
                        norm_layer=nn.BatchNorm2d(
                            in_chls[0], **self.bn_kwargs
                        ),
                        act_layer=nn.ReLU(inplace=True),
                    )
                )
                stride = 1
            layers.append(
                InvertedResidual(
                    in_chl,
                    out_chl,
                    stride,
                    expand_t,
                    self.bn_kwargs,
                    self.bias,
                )
            )
        return nn.Sequential(*layers)
