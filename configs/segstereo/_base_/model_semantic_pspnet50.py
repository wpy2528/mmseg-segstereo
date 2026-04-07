# 仅语义分支：PSPNet-50 = 空洞 ResNet-50 + PSPHead（+ FCN 辅助头），写法与 mmseg 一致。
# 骨干类型为 SegStereoPSPNet50Backbone（拓扑与 pspnet_r50-d8 的 ResNetV1c 相同）。
#
# 之后换 STDC：将 backbone 改为 STDCContextPathNet，并相应调整例如
#   decode_head.in_channels -> 最后一层特征通道（常见为 256），
#   auxiliary_head 的 in_channels / in_index -> STDC 中间层。
norm_cfg = dict(type='SyncBN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained='open-mmlab://resnet50_v1c',
    backbone=dict(
        type='SegStereoPSPNet50Backbone',
        norm_cfg=norm_cfg,
        norm_eval=False),
    decode_head=dict(
        type='PSPHead',
        in_channels=2048,
        in_index=3,
        channels=512,
        pool_scales=(1, 2, 3, 6),
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=1024,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
