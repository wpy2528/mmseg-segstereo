NUM_CLASSES = 5
BATCH_PAD_SIZE = (320, 320)
RESIZE_SIZE = (320, 320)


# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    size=BATCH_PAD_SIZE,
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)

'''
Input shape: (320, 320)
Flops: 68.948G
Params: 41.218M
'''
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained='open-mmlab://resnet50_v1c',
    backbone=dict(
        type='ResNetV1c',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=dict(
        type='DepthwiseSeparableASPPHead',
        in_channels=2048,
        in_index=3,
        channels=512,
        dilations=(1, 12, 24, 36),
        c1_in_channels=256,
        c1_channels=48,
        dropout_ratio=0.1,
        num_classes=NUM_CLASSES,
        norm_cfg=norm_cfg,
        align_corners=False,
        # loss_decode=[
        # dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0),
        # dict(type='DiceLoss', loss_name='loss_dice', loss_weight=3.0)
        # ]
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
        num_classes=NUM_CLASSES,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))


# dataset settings
dataset_type = 'LDPerceptionSegDataset'
data_root = 'data/perception_segmentation/0107'

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    
    # 随机缩放图像 & mask（多尺度增强）
    dict(
        type='RandomResize',
        scale=(320, 320),
        ratio_range=(1.0, 2.0),  # 尺寸随机缩放
        keep_ratio=False
    ),
    
    # 随机裁剪固定尺寸（控制输入大小 & 兼顾上下文）
    dict(
        type='RandomCrop',
        crop_size=(320, 320),
        cat_max_ratio=0.75  # 避免某类 dominate 整张图（比如全是背景）
    ),
    
    # 随机翻转
    dict(type='RandomFlip', prob=0.5),
    
    # 随机颜色扰动（亮度、饱和度、对比度等）
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18
    ),
    
    # Padding 补全到固定尺寸（保证输入张量一致）
    dict(
        type='Pad',
        size=(320, 320),
        pad_val=255,
        # seg_pad_val=255  # ignore_index
    ),
    
    # 转换为 tensor 并归一化
    dict(type='PackSegInputs')  # 替代 Normalize + ToTensor
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='Resize',
        scale=RESIZE_SIZE,
        keep_ratio=False
    ),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]
# img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
# tta_pipeline = [
#     dict(type='LoadImageFromFile', backend_args=None),
#     dict(
#         type='TestTimeAug',
#         transforms=[
#             [
#                 dict(type='Resize', scale_factor=r, keep_ratio=True)
#                 for r in img_ratios
#             ],
#             [
#                 dict(type='RandomFlip', prob=0., direction='horizontal'),
#                 dict(type='RandomFlip', prob=1., direction='horizontal')
#             ], [dict(type='LoadAnnotations')], [dict(type='PackSegInputs')]
#         ])
# ]
train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train.txt',
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='val.txt',
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'], output_dir="pgs", calc_per_sample_metric=True)
test_evaluator = val_evaluator


default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
vis_backends=[dict(type='LocalVisBackend'),
              dict(type='TensorboardVisBackend'),
              ]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(by_epoch=True)
log_level = 'INFO'
load_from = None
resume = False

tta_model = dict(type='SegTTAModel')


# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)
# learning policy
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-4,
        by_epoch=True,
        begin=0,
        end=3  # 前3个 epoch warmup
    ),
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        by_epoch=True,
        begin=3,
        end=50
    )
]
# training schedule for 240k
train_cfg = dict(by_epoch=True, max_epochs=50, val_interval=1)
# train_cfg = dict(
#     type='IterBasedTrainLoop', max_iters=240000, val_interval=24000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook', draw=True, interval=1))
