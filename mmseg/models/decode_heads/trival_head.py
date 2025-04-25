# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import MODELS
from .decode_head import BaseDecodeHead


@MODELS.register_module()
class TrivalHead(BaseDecodeHead):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, inputs):
        return inputs
