#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from horizon_plugin_pytorch.quantization import QuantStub
from torch.quantization import DeQuantStub

from .models.base_modules.basic_repvgg_module import (
    MultiBranchConvModule,
    RepBlock,
)
from .models.base_modules.conv_module import ConvModule2d
from .registry import OBJECT_REGISTRY

__all__ = [
    "MobileOneBaseV2",
    "MobileOneV2",
]


class MobileOneBaseV2(nn.Module):
    """MobileOne Model, but V2 changed BN place following QA-Repvgg.

        Pytorch implementation of `An Improved One millisecond Mobile Backbone`
        https://arxiv.org/pdf/2206.04040.pdf

    Args:
        num_blocks_per_stage: List of number of blocks per stage.
        num_classes: Number of classes in the dataset.
        width_multipliers: List of width multiplier for blocks in a stage.
        group_size: Group size.
        use_se: Whether to use SE-ReLU activations.
        num_conv_branches: Number of linear conv branches.
        include_top: Whether to include output layer.
        flat_output: Whether to view the output tensor.
        deploy: If True, instantiates model in inference mode.
    """

    def __init__(
        self,
        num_blocks_per_stage: Tuple[int] = (2, 8, 10, 1),
        num_classes: int = 1000,
        width_multipliers: Optional[List[float]] = None,
        group_size: int = 1,
        use_se: bool = False,
        num_conv_branches: int = 1,
        include_top: bool = True,
        flat_output: bool = True,
        deploy: bool = False,
    ) -> None:
        super(MobileOneBaseV2, self).__init__()

        assert len(width_multipliers) == 4
        self.deploy = deploy
        self.in_planes = min(64, int(64 * width_multipliers[0]))
        self.use_se = use_se
        self.num_conv_branches = num_conv_branches

        self.num_classes = num_classes
        self.include_top = include_top
        self.flat_output = flat_output

        self.quant = QuantStub(scale=1.0 / 128.0)
        self.dequant = DeQuantStub()
        # Build stages
        self.mod1 = RepBlock(
            module=MultiBranchConvModule(
                in_channels=3,
                out_channels=self.in_planes,
                k_size_list=[0, 1, 3],
                bn_flag_list=[False, False, True],
                stride=2,
            ),
            norm_layer=nn.BatchNorm2d(self.in_planes),
            act_layer=nn.ReLU(),
            deploy=self.deploy,
        )
        self.cur_layer_idx = 1
        self.mod2 = self._make_stage(
            int(64 * width_multipliers[0]),
            num_blocks_per_stage[0],
            num_se_blocks=0,
            group_size=group_size,
        )
        self.mod3 = self._make_stage(
            int(128 * width_multipliers[1]),
            num_blocks_per_stage[1],
            num_se_blocks=0,
            group_size=group_size,
        )
        self.mod4 = self._make_stage(
            int(256 * width_multipliers[2]),
            num_blocks_per_stage[2],
            num_se_blocks=int(num_blocks_per_stage[2] // 2) if use_se else 0,
            group_size=group_size,
        )
        self.mod5 = self._make_stage(
            int(512 * width_multipliers[3]),
            num_blocks_per_stage[3],
            num_se_blocks=num_blocks_per_stage[3] if use_se else 0,
            group_size=group_size,
        )
        if self.include_top:
            self.output = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                ConvModule2d(
                    int(512 * width_multipliers[3]),
                    num_classes,
                    1,
                    bias=True,
                ),
            )
        else:
            self.output = None

    def _make_stage(
        self,
        planes: int,
        num_blocks: int,
        num_se_blocks: int,
        group_size: int = 1,
    ) -> nn.Sequential:
        """Build a stage of MobileOne model.

        Args:
            planes: Number of output channels.
            num_blocks: Number of blocks in this stage.
            num_se_blocks: Number of SE blocks in this stage.
        """
        # Get strides for all layers
        strides = [2] + [1] * (num_blocks - 1)
        blocks = []
        for ix, stride in enumerate(strides):
            use_se = False
            if num_se_blocks > num_blocks:
                raise ValueError(
                    "Number of SE blocks cannot " "exceed number of layers."
                )
            if ix >= (num_blocks - num_se_blocks):
                use_se = True
            k_size_list = [0, 1] + [3] * self.num_conv_branches
            # Depthwise conv
            blocks.append(
                RepBlock(
                    module=MultiBranchConvModule(
                        in_channels=self.in_planes,
                        out_channels=self.in_planes,
                        k_size_list=k_size_list,
                        bn_flag_list=[False, False]
                        + [True for i in range(self.num_conv_branches)],
                        stride=stride,
                        groups=self.in_planes // group_size,
                    ),
                    norm_layer=nn.BatchNorm2d(self.in_planes),
                    act_layer=nn.ReLU(),
                    use_se=use_se,
                    deploy=self.deploy,
                )
            )
            # Pointwise conv
            blocks.append(
                RepBlock(
                    module=MultiBranchConvModule(
                        in_channels=self.in_planes,
                        out_channels=planes,
                        k_size_list=[0] + [1] * self.num_conv_branches,
                        bn_flag_list=[False] * (self.num_conv_branches + 1),
                        stride=1,
                        groups=1,
                    ),
                    norm_layer=nn.BatchNorm2d(planes),
                    act_layer=nn.ReLU(),
                    use_se=use_se,
                    deploy=self.deploy,
                )
            )
            self.in_planes = planes
            self.cur_layer_idx += 1
        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output_list = []
        x = self.quant(x)
        for module in [self.mod1, self.mod2, self.mod3, self.mod4, self.mod5]:
            x = module(x)
            output_list.append(x)
        if not self.include_top:
            return output_list
        x = self.output(x)
        x = self.dequant(x)
        if self.flat_output:
            x = x.view(-1, self.num_classes)
        return x

    def set_qconfig(self):
        from .utils import qconfig_manager

        if self.include_top:
            # disable output quantization for last quanti layer.
            getattr(
                self.output, "1"
            ).qconfig = qconfig_manager.get_default_qat_out_qconfig()

    def switch_to_deploy(self):
        for module in self.children():
            if hasattr(module, "switch_to_deploy"):
                module.switch_to_deploy()


PARAMS = {
    "s0": {"width_multipliers": (0.75, 1.0, 1.0, 2.0), "num_conv_branches": 4},
    "s1": {"width_multipliers": (1.5, 1.5, 2.0, 2.5)},
    "s2": {"width_multipliers": (1.5, 2.0, 2.5, 4.0)},
    "s3": {"width_multipliers": (2.0, 2.5, 3.0, 4.0)},
    "s4": {"width_multipliers": (3.0, 3.5, 3.5, 4.0), "use_se": True},
}


@OBJECT_REGISTRY.register
class MobileOneV2(MobileOneBaseV2):
    """
    MobileOneV2 model that specify the model type.

    Args:
        model_type: Model_type must be one of: [s0, s1, s2, s3, s4].
        num_classes: Number of classes in the dataset.
        include_top: Whether to include output layer.
        flat_output: Whether to view the output tensor.
        deploy: If True, instantiates model in inference mode.
    """

    def __init__(
        self,
        model_type: str,
        num_classes: int = 1000,
        group_size: int = 1,
        include_top: bool = True,
        flat_output: bool = True,
        deploy: bool = False,
    ) -> None:
        variant_params = PARAMS[model_type]
        super(MobileOneV2, self).__init__(
            num_classes=num_classes,
            group_size=group_size,
            include_top=include_top,
            flat_output=flat_output,
            deploy=deploy,
            **variant_params
        )
