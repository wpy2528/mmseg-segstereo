# SegStereo 联合训练：type=SegStereo = 语义 EncoderDecoder + SegStereoDisparityBranch。
# 输入为 6 通道 [左 RGB | 右 RGB]（PackSegStereoInputs），标注需 gt_sem_seg + gt_disp。
#
# 如何接到 tools/train.py：
#   1) 复制一份你已有的分割 config（保证 train_dataloader / optim_wrapper / schedule 可跑通）。
#   2) 将其中 model 的 _base_ 改为本文件，或直接把 model = dict(...) 嵌进你的 config。
#   3) train_pipeline：LoadStereoImages -> LoadAnnotations -> LoadStereoMatchingAnnotations
#      -> 同步几何增强 -> PackSegStereoInputs（勿用仅 3 通道的 PackSegInputs）。
#   4) 数据集每条样本提供：left_img_path, right_img_path, seg_map_path, left_disp_path。
#   5) SegDataPreProcessor 已对 6 通道做左右分别归一化（见 mmseg/models/data_preprocessor.py）。
#
# 视差平滑：``loss_disp_smooth_weight>0`` 时对 |∂d/∂x|+|∂d/∂y| 正则（与 loss_disp 同 gt 有效区）；
#   不增加参数量；权重可从 0.01～0.05 试起，过大易糊边。
# 论文 L_seg（语义正则视差）：``loss_semantic_warp_weight>0`` 时对右图再过一遍 backbone+decode，
#   用预测视差将右语义 logits warp 到左后与 gt_sem_seg 算 CE；默认 0 关闭（多一次右分支前向）。
# 语义融合视差：设 disparity_branch['semantic_channels']=num_classes，且 model['fuse_semantic_to_disp']=True。
# 延迟/自动开启：model 中 ``fuse_semantic_warmup_epochs``（runner.epoch 达到该值后开启）；
# ``default_hooks`` 注册 ``SegStereoSemanticFuseHook``（warmup_epochs 可覆盖模型配置，
# auto_plateau 可根据 val 的 mIoU 平台提前开启）。
norm_cfg = dict(type='SyncBN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)
model = dict(
    type='SegStereo',
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
    disparity_branch=dict(
        type='SegStereoDisparityBranch',
        max_disp=192,
        semantic_channels=0,
    ),
    fuse_semantic_to_disp=False,
    loss_disp_weight=0.5,
    loss_disp_smooth_weight=0.0,
    loss_semantic_warp_weight=0.0,
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
