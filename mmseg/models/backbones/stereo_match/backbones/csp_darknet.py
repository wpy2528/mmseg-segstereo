import torch.nn as nn

from .models.base_modules.basic_cspdarknet_module import (
    CSPLayer,
    Focus,
    SPPBottleneck,
)
from .models.base_modules.conv_module import ConvModule2d
from .registry import OBJECT_REGISTRY

__all__ = ["CSPDarknet"]


@OBJECT_REGISTRY.register
class CSPDarknet(nn.Module):
    """CSP-Darknet backbone used in YOLOX.

    Args:
        dep_mul: Depth multiplier, multiply number of
            blocks in CSP layer by this amount.
        wid_mul: Width multiplier, multiply number of
            channels in each layer by this amount.
        frozen_stages: Stages to be frozen (stop grad and set eval
            mode). -1 means not freezing any parameters.
        depthwise: Whether to use depthwise separable convolution.
        act: Activation layer.
    """

    def __init__(
        self,
        dep_mul: float,
        wid_mul: float,
        frozen_stages: int = -1,
        depthwise: bool = False,
        act: str = "silu",
    ):
        super().__init__()
        if frozen_stages not in range(-1, 5):
            raise ValueError(
                "frozen_stages must be in range(-1, "
                "len(arch) + 1). But received "
                f"{frozen_stages}"
            )
        self.frozen_stages = frozen_stages

        base_channels = int(wid_mul * 64)  # 64
        base_depth = max(round(dep_mul * 3), 1)  # 3

        # stem
        self.stem = Focus(3, base_channels, ksize=3, act=act)

        # dark2
        self.dark2 = nn.Sequential(
            ConvModule2d(
                base_channels,
                base_channels * 2,
                3,
                stride=2,
                padding=1,
                bias=False,
                norm_layer=nn.BatchNorm2d(base_channels * 2),
                act_layer=nn.SiLU(inplace=True),
            ),
            CSPLayer(
                base_channels * 2,
                base_channels * 2,
                n=base_depth,
                depthwise=depthwise,
                act=act,
            ),
        )

        # dark3
        self.dark3 = nn.Sequential(
            ConvModule2d(
                base_channels * 2,
                base_channels * 4,
                3,
                stride=2,
                padding=1,
                bias=False,
                norm_layer=nn.BatchNorm2d(base_channels * 4),
                act_layer=nn.SiLU(inplace=True),
            ),
            CSPLayer(
                base_channels * 4,
                base_channels * 4,
                n=base_depth * 3,
                depthwise=depthwise,
                act=act,
            ),
        )

        # dark4
        self.dark4 = nn.Sequential(
            ConvModule2d(
                base_channels * 4,
                base_channels * 8,
                3,
                stride=2,
                padding=1,
                bias=False,
                norm_layer=nn.BatchNorm2d(base_channels * 8),
                act_layer=nn.SiLU(inplace=True),
            ),
            CSPLayer(
                base_channels * 8,
                base_channels * 8,
                n=base_depth * 3,
                depthwise=depthwise,
                act=act,
            ),
        )

        # dark5
        self.dark5 = nn.Sequential(
            ConvModule2d(
                base_channels * 8,
                base_channels * 16,
                3,
                stride=2,
                padding=1,
                bias=False,
                norm_layer=nn.BatchNorm2d(base_channels * 16),
                act_layer=nn.SiLU(inplace=True),
            ),
            SPPBottleneck(
                base_channels * 16, base_channels * 16, activation=act
            ),
            CSPLayer(
                base_channels * 16,
                base_channels * 16,
                n=base_depth,
                shortcut=False,
                depthwise=depthwise,
                act=act,
            ),
        )
        self.layers = ["stem", "dark2", "dark3", "dark4", "dark5"]

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for i in range(self.frozen_stages + 1):
                m = getattr(self, self.layers[i])
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def forward(self, x):
        output = []
        x = self.stem(x)
        for layer_name in self.layers[1:]:
            layer = getattr(self, layer_name)
            x = layer(x)
            output.append(x)
        return output

    def fuse_model(self):
        self.stem.fuse_model()
        for layer_name in self.layers[1:]:
            layer = getattr(self, layer_name)
            for m in layer:
                m.fuse_model()
