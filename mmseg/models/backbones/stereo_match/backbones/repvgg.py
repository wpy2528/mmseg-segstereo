from typing import Dict, List

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
    "RepVGG",
]


class RepVGGBase(nn.Module):
    """RepVGG model template.

    Args:
        num_blocks: List of number of blocks for each stage.
        width_multiplier: Width multiplier for scaling the number of channels.
        override_groups_map: Dictionary for overriding the group configurations
        num_classes: Number of output classes.
        deploy: Whether the model is for deployment or training.
            When deploy=False, the model has 3 branches
            When deploy=True, the model is reparameterized.
        use_se: Whether to use Squeeze-and-Excitation blocks.
        include_top: Whether to include output layer.
            If True, used for classification; else used as backbone
        flat_output: Whether to flatten the output. Default is True.
    """

    def __init__(
        self,
        num_blocks: List[int] = None,
        width_multiplier: float = None,
        override_groups_map: Dict = None,
        num_classes: int = 1000,
        deploy: bool = False,
        use_se: bool = False,
        include_top: bool = True,
        flat_output: bool = True,
    ):

        super(RepVGGBase, self).__init__()
        assert len(width_multiplier) == 4
        self.deploy = deploy
        self.override_groups_map = override_groups_map or {}
        assert 0 not in self.override_groups_map
        self.use_se = use_se
        self.include_top = include_top
        self.flat_output = flat_output
        self.num_classes = num_classes

        self.quant = QuantStub(scale=1 / 128.0)
        self.dequant = DeQuantStub()

        self.in_planes = min(64, int(64 * width_multiplier[0]))
        self.stage0 = RepBlock(
            module=MultiBranchConvModule(
                in_channels=3,
                out_channels=self.in_planes,
                k_size_list=[0, 1, 3],
                bn_flag_list=[True, True, True],
                stride=2,
            ),
            use_se=self.use_se,
            act_layer=nn.ReLU(),
            deploy=self.deploy,
        )
        self.cur_layer_idx = 1
        self.stage1 = self._make_stage(
            int(64 * width_multiplier[0]), num_blocks[0], stride=2
        )
        self.stage2 = self._make_stage(
            int(128 * width_multiplier[1]), num_blocks[1], stride=2
        )
        self.stage3 = self._make_stage(
            int(256 * width_multiplier[2]), num_blocks[2], stride=2
        )
        self.stage4 = self._make_stage(
            int(512 * width_multiplier[3]), num_blocks[3], stride=2
        )
        if self.include_top:
            self.output = nn.Sequential(
                nn.AvgPool2d(7),
                ConvModule2d(
                    int(512 * width_multiplier[3]),
                    num_classes,
                    1,
                ),
            )

    def _make_stage(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for stride in strides:
            cur_groups = self.override_groups_map.get(self.cur_layer_idx, 1)
            multi_branches = MultiBranchConvModule(
                in_channels=self.in_planes,
                out_channels=planes,
                k_size_list=[0, 1, 3],
                bn_flag_list=[True, True, True],
                stride=stride,
                groups=cur_groups,
            )
            rep_block = RepBlock(
                multi_branches,
                norm_layer=nn.BatchNorm2d(planes),
                use_se=self.use_se,
                act_layer=nn.ReLU(),
                deploy=self.deploy,
            )
            blocks.append(rep_block)
            self.in_planes = planes
            self.cur_layer_idx += 1
        return nn.ModuleList(blocks)

    def forward(self, x):
        out_feats = []
        x = self.quant(x)
        x = self.stage0(x)
        for stage in (self.stage1, self.stage2, self.stage3, self.stage4):
            for block in stage:
                x = block(x)
            out_feats.append(x)
        if not self.include_top:
            return out_feats
        out = self.output(x)
        out = self.dequant(out)
        if self.flat_output:
            out = out.view(-1, self.num_classes)
        return out

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
        self.deploy = True


PARAMS = {
    "A0": {
        "num_blocks": [2, 4, 14, 1],
        "width_multiplier": [0.75, 0.75, 0.75, 2.5],
    },
    "A1": {"num_blocks": [2, 4, 14, 1], "width_multiplier": [1, 1, 1, 2.5]},
    "A2": {
        "num_blocks": [2, 4, 14, 1],
        "width_multiplier": [1.5, 1.5, 1.5, 2.75],
    },
    "B0": {"num_blocks": [4, 6, 16, 1], "width_multiplier": [1, 1, 1, 2.5]},
    "B1": {"num_blocks": [4, 6, 16, 1], "width_multiplier": [1, 1, 1, 4]},
}


@OBJECT_REGISTRY.register
class RepVGG(RepVGGBase):
    """RepVGG model with configuration.

    Args:
        model_type: Model type, available type: ["A0", "A1", "A2", "B0", "B1"]
        num_classes : Number of output classes. Default is 1000.
        deploy : Whether the model is for deployment or training.
            Default is False.
        use_checkpoint : Whether to use gradient checkpointing.
            Default is False.
        include_top: Whether to include output layer.
            If True, used for classification; else used as backbone
        flat_output: Whether to view the output tensor.
    """

    def __init__(
        self,
        model_type: str,
        num_classes: int = 1000,
        deploy: bool = False,
        include_top: bool = True,
        flat_output: bool = True,
    ):
        super(RepVGG, self).__init__(
            **PARAMS[model_type],
            num_classes=num_classes,
            override_groups_map=None,
            deploy=deploy,
            include_top=include_top,
            flat_output=flat_output,
        )
