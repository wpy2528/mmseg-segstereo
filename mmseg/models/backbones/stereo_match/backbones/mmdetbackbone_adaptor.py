# Copyright (c) Horizon Robotics. All rights reserved.
import logging
from collections.abc import Mapping
from typing import List, Union

from torch import Tensor, nn

from .registry import OBJECT_REGISTRY
from .utils.package_helper import require_packages

try:
    from mmdet.version import __version__ as mm_version

    if mm_version >= "3.0.0":
        from mmdet.registry import MODELS
        from mmengine.config.config import ConfigDict

        build_backbone = MODELS.build
    else:
        from mmcv.utils import ConfigDict
        from mmdet.models import build_backbone
except ImportError:
    pass

logger = logging.getLogger(__name__)

__all__ = ["MMDetBackboneAdaptor"]


@OBJECT_REGISTRY.register
class MMDetBackboneAdaptor(nn.Module):
    """
    Adaptor of mmdetection backbones to use in HAT.

    Args:
        backbone: either a backbone module or a mmdet config dict
        that defines a backbone.
        The backbone takes a 4D image tensor and returns a list of tensors.
    """

    @require_packages("mmdet", "mmcv")
    def __init__(
        self,
        backbone: Union[nn.Module, Mapping],
    ):
        super().__init__()
        if isinstance(backbone, Mapping):
            backbone = build_backbone(ConfigDict(backbone))
        self.backbone = backbone

    def forward(self, x) -> List[Tensor]:
        outs = self.backbone(x)
        assert isinstance(
            outs, (list, tuple)
        ), "mmdet backbone should return a list of tensors!"
        return outs
