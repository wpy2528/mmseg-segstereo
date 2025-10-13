# Copyright (c) Horizon Robotics. All rights reserved.
from collections.abc import Sequence

import horizon_plugin_pytorch.nn as hnn
import torch.nn as nn

from .models.backbones.vargnetv2 import VargNetV2
from .models.base_modules.conv_module import FixedConvModule2d, FusedConv2d
from .registry import OBJECT_REGISTRY

__all__ = ["VargNetV2ISP"]


def normal_init(module, mean=0, std=1.0, bias=0.0):
    nn.init.normal_(module.weight, mean, std)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class NeuralISPGroupBlock(nn.Module):
    """
    A module of Neural ISP group block module.

    Args:
        in_channels: Input channels of first conv.
        out_channels: Output channels of this block.
        use_bias: Whether to use bias in module.
        bn_kwargs: Dict for BN layer.
        group: group for the second fused conv.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        use_bias: bool = False,
        bn_kwargs: dict = None,
        group: int = 4,
    ):
        super(NeuralISPGroupBlock, self).__init__()

        self.head_conv = FusedConv2d(
            in_channels=in_channels,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=use_bias,
            bn_kwargs=bn_kwargs,
            with_relu=True,
        )

        self.conv1 = nn.Conv2d(
            16, out_channels, kernel_size=1, stride=1, padding=0
        )
        self.conv2 = FusedConv2d(
            in_channels=16,
            out_channels=16,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=use_bias,
            group=group,
            bn_kwargs=bn_kwargs,
            with_relu=True,
        )

        self.conv2_1x1 = FusedConv2d(
            in_channels=16,
            out_channels=16,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=use_bias,
            bn_kwargs=bn_kwargs,
            with_relu=True,
        )
        self.conv3 = nn.Conv2d(
            16, out_channels, kernel_size=1, stride=1, padding=0
        )

        self.upsampler = hnn.Interpolate(
            scale_factor=2, mode="bilinear", recompute_scale_factor=True
        )
        self.skip_add = nn.quantized.FloatFunctional()

        self.bnrelu = FixedConvModule2d(out_channels, bn_kwargs=bn_kwargs)

    def forward(self, x):
        x1 = self.head_conv(x)
        out = self.conv1(x1)
        x_down = self.conv2(x1)
        x_down = self.conv2_1x1(x_down)

        out_down = self.conv3(x_down)
        out_up = self.upsampler(out_down)
        out_fusion = self.skip_add.add(x, out)
        out_fusion = self.skip_add.add(out_fusion, out_up)
        output = self.bnrelu(out_fusion)

        return output

    def fuse_model(self):
        modules = [self.head_conv, self.conv2, self.conv2_1x1, self.bnrelu]
        for module in modules:
            module.fuse_model()


class NeuralISP(nn.Module):
    """
    NeuralISP Module.

    Args:
        in_channels: Input channels of NeuralISP module.
        out_channels: Output channels of NeuralISP module.
        use_bias: whether to use bias in conv.
        bn_kwargs: Dict for BN layer in NeuralISPGroupBlock.
        group: group for conv in NeuralISPGroupBlock.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        use_bias: bool = False,
        bn_kwargs: dict = None,
        group: int = 4,
    ):
        super(NeuralISP, self).__init__()
        self.model = NeuralISPGroupBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            use_bias=use_bias,
            bn_kwargs=bn_kwargs,
            group=group,
        )
        self._init_weights()

    def forward(self, img, **kwargs):
        out = self.model(img)
        return out

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, mean=0, std=0.01, bias=0)
        for m in self.modules():
            if isinstance(m, FixedConvModule2d):
                m._init_weights()

    def fuse_model(self):
        self.model.fuse_model()


@OBJECT_REGISTRY.register
class VargNetV2ISP(VargNetV2):
    """
    A module of vargnetv2 with NeuralISP module.

    Args:
        num_classes: Num classes of output layer.
        bn_kwargs: Dict for BN layer.
        model_type: Choose to use `VargNetV2` or `TinyVargNetV2`.
        alpha: Alpha for vargnetv2.
        group_base: Group base for vargnetv2.
        factor: Factor for channel expansion in basic block.
        bias: Whether to use bias in module.
        extend_features: Whether to extend features.
        include_top: Whether to include output layer.
        flat_output: Whether to view the output tensor.
        input_channels: Input channels of first conv.
        input_sequence_length: Length of input sequence.
        head_factor: Factor for channels expansion of stage1(mod2).
        input_resize_scale: Narrow_model need resize input 0.65 scale,
            While int_infer or visualize or eval
    """

    def __init__(
        self,
        num_classes: int,
        bn_kwargs: dict,
        model_type: str = "VargNetV2",
        alpha: float = 1.0,
        group_base: int = 8,
        factor: int = 2,
        bias: bool = True,
        extend_features: bool = False,
        disable_quanti_input: bool = False,
        include_top: bool = True,
        flat_output: bool = True,
        input_channels: int = 3,
        input_sequence_length: int = 1,
        head_factor: int = 1,
        input_resize_scale: int = None,
    ):
        super(VargNetV2ISP, self).__init__(
            num_classes,
            bn_kwargs,
            model_type,
            alpha,
            group_base,
            factor,
            bias,
            extend_features,
            disable_quanti_input,
            include_top,
            flat_output,
            input_channels,
            input_sequence_length,
            head_factor,
            input_resize_scale,
        )

        self.isp = NeuralISP(
            in_channels=input_channels,
            use_bias=self.bias,
            bn_kwargs=self.bn_kwargs,
        )

    def forward(self, x):
        if self.input_sequence_length > 1:
            x = self.process_sequence_input(x)
        else:
            if isinstance(x, Sequence) and len(x) == 1:
                x = x[0]
            x = x if self.disable_quanti_input else self.quant(x)

        x = self.isp(x)

        if self.input_resize_scale is not None:
            x = self.resize(x)

        output = []
        for module in [self.mod1, self.mod2, self.mod3, self.mod4, self.mod5]:
            x = module(x)
            output.append(x)

        if self.extend_features:
            output = self.ext(output)

        if not self.include_top:
            return output
        x = self.output(x)
        x = self.dequant(x)
        if self.flat_output:
            x = x.view(-1, self.num_classes)
        return x

    def fuse_model(self):
        super().fuse_model()
        self.isp.fuse_model()
