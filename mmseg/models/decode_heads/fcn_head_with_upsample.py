import torch.nn as nn

from mmseg.registry import MODELS
from .fcn_head import FCNHead


@MODELS.register_module()
class FCNHeadWithUpsample(FCNHead):
    """Decoder head for STDC self-supervised image reconstruction."""

    def __init__(self, upsample_channels=3, **kwargs):
        super().__init__(**kwargs)

        self.upsample_head = nn.Sequential(
            nn.Conv2d(upsample_channels, self.channels, 3, padding=1),
            nn.BatchNorm2d(self.channels),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(self.channels, self.channels // 2, 3, padding=1),
            nn.BatchNorm2d(self.channels // 2),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(self.channels // 2, self.channels // 4, 3, padding=1),
            nn.BatchNorm2d(self.channels // 4),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(self.channels // 4, self.num_classes, 3, padding=1)
        )

    def forward(self, inputs):
        x = super().forward(inputs)
        x = self.upsample_head(x)
        return x
