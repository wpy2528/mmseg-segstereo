from typing import Dict, List, Union

import torch.nn as nn
from horizon_plugin_pytorch.quantization import QuantStub
from torch.quantization import DeQuantStub

from .models.base_modules.conv_module import ConvModule2d
from .models.base_modules.sandglass import SandGlass
from .registry import OBJECT_REGISTRY

__all__ = ["SNDRMobileNeXt"]


@OBJECT_REGISTRY.register
class SNDRMobileNeXt(nn.Module):
    """A module of mobilenext for snapdragon SoC.

    Args:
        num_classes: Num classes of output layer.
        bn_kwargs: Dict of BN layer.
        in_chls: Input channel list.
        out_chls: Output channel list.
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
        feat_size: Union[List[int], int] = 4,
        expand_ratio: int = 6,
        alpha: float = 1,
        bias: bool = True,
        include_top: bool = True,
        flat_output: bool = True,
        use_dw_as_avgpool: bool = False,
    ):
        super(SNDRMobileNeXt, self).__init__()
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
        self.mod1 = self._make_stage(
            in_chls[0], out_chls[0], 1, first_layer=True
        )
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
                        3,
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
                SandGlass(
                    in_chl,
                    out_chl,
                    stride,
                    expand_t,
                    self.bn_kwargs,
                    self.bias,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        output = []
        x = self.quant(x)
        for module in [self.mod1, self.mod2, self.mod3, self.mod4, self.mod5]:
            x = module(x)
            output.append(x)
        if not self.include_top:
            return output
        x = self.output(x)
        x = self.dequant(x)
        if self.flat_output:
            x = x.view(-1, self.num_classes)
        return x

    def fuse_model(self):
        modules = [self.mod1, self.mod2, self.mod3, self.mod4, self.mod5]
        if self.include_top:
            modules += [self.output]
        for module in modules:
            for m in module:
                if hasattr(m, "fuse_model"):
                    m.fuse_model()

    def set_qconfig(self):
        from .utils import qconfig_manager

        if self.include_top:
            # disable output quantization for last quanti layer.
            getattr(
                self.output, "2"
            ).qconfig = qconfig_manager.get_default_qat_out_qconfig()
