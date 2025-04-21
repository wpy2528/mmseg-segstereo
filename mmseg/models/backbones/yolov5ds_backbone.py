# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule

from mmseg.registry import MODELS
from ..decode_heads.psp_head import PPM
from ..utils import resize


import os
from tqdm import tqdm
import cv2
import sys
sys.path.append("/home/mck/yolov5ds/")

from models.yolodhs import Model



@MODELS.register_module()
class Yolov5DSBackbone(BaseModule):
    """Yolov5DSBackbone for Real-Time Semantic Segmentation on High-Resolution Images.

    This backbone is the implementation of
    `Yolov5DSBackbone <https://arxiv.org/abs/1704.08545>`_.

    Args:
        backbone_cfg (dict): Config dict to build backbone. Usually it is
            ResNet but it can also be other backbones.
        in_channels (int): The number of input image channels. Default: 3.
        layer_channels (Sequence[int]): The numbers of feature channels at
            layer 2 and layer 4 in ResNet. It can also be other backbones.
            Default: (512, 2048).
        light_branch_middle_channels (int): The number of channels of the
            middle layer in light branch. Default: 32.
        psp_out_channels (int): The number of channels of the output of PSP
            module. Default: 512.
        out_channels (Sequence[int]): The numbers of output feature channels
            at each branches. Default: (64, 256, 256).
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module. Default: (1, 2, 3, 6).
        conv_cfg (dict): Dictionary to construct and config conv layer.
            Default: None.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Default: dict(type='BN').
        act_cfg (dict): Dictionary to construct and config act layer.
            Default: dict(type='ReLU').
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 backbone_cfg=None,
                 in_channels=3,
                 layer_channels=(512, 2048),
                 light_branch_middle_channels=32,
                 psp_out_channels=512,
                 out_channels=(64, 256, 256),
                 pool_scales=(1, 2, 3, 6),
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU'),
                 align_corners=False,
                 init_cfg=None):
        if init_cfg is None:
            init_cfg = [
                dict(type='Kaiming', mode='fan_out', layer='Conv2d'),
                dict(type='Constant', val=1, layer='_BatchNorm'),
                dict(type='Normal', mean=0.01, layer='Linear')
            ]
        super().__init__(init_cfg=init_cfg)
        cfg = 'models/my_yolov5s.yaml'
        segcfg = 'models/my_segheads.yaml'
        nc = 5
        segnc = 5
        # torch.Size([20, 3, 40, 40, 25]) torch.Size([20, 3, 20, 20, 25]) torch.Size([20, 3, 10, 10, 25])
        # [20, 3, 320, 320] -> [20, 5, 320, 320]
        self.backbone = Model(cfg, segcfg, ch=3, nc=nc, segnc=segnc, anchors=None)
        dummy_input_t = torch.randn(20, 3, 320, 320)
        dummy_output_t = self.backbone(dummy_input_t)
        feats, seg_pred = dummy_output_t
        print(feats[0].shape, feats[1].shape, feats[2].shape)
        print(seg_pred[0].shape)

    def forward(self, x):
        output = self.backbone(x)
        return output
        output = []

        # sub 1
        output.append(self.conv_sub1(x))

        # sub 2
        x = resize(
            x,
            scale_factor=0.5,
            mode='bilinear',
            align_corners=self.align_corners)
        x = self.backbone.stem(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        output.append(self.conv_sub2(x))

        # sub 4
        x = resize(
            x,
            scale_factor=0.5,
            mode='bilinear',
            align_corners=self.align_corners)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        psp_outs = self.psp_modules(x) + [x]
        psp_outs = torch.cat(psp_outs, dim=1)
        x = self.psp_bottleneck(psp_outs)

        output.append(self.conv_sub4(x))

        return output
