# ===== Global Constants =====
CLASS_NAMES = ("background", "grass", "soil", "animal")
NUM_CLASSES = len(CLASS_NAMES)
RESIZE_WH = (320, 272)
BATCH_PAD_HW = (272, 320)

norm_cfg = dict(type='BN', requires_grad=True)

data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[0.0, 0.0, 0.0],
    std=[255.0, 255.0, 255.0],
    size=BATCH_PAD_HW,
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255
)

# ===== Model Settings =====
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type='STDCContextPathNet',
        backbone_cfg=dict(
            type='STDCNet',
            stdc_type='STDCNet2',
            in_channels=3,
            channels=(32, 64, 256, 512, 1024),
            num_convs=4,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU'),
            bottleneck_type='cat',
            with_final_conv=False,
            init_cfg=dict(
                type='Pretrained',
                checkpoint='https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/stdc/stdc2_20220308-7dbd9127.pth'
            )
        ),
        last_in_channels=(1024, 512),
        out_channels=128,
        ffm_cfg=dict(in_channels=384, out_channels=256, scale_factor=4)
    ),
    decode_head=dict(
        type='FCNHead',
        in_channels=256,
        in_index=3,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=NUM_CLASSES,
        norm_cfg=norm_cfg,
        align_corners=True,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        sampler=dict(type='OHEMPixelSamplerWithSpecialClassPenalty', gt_class=2, penalty_pred_class=0, thresh=0.7, min_kept=10000)
    ),
    auxiliary_head=[
        dict(
            type='FCNHead',
            in_channels=128,
            in_index=2,
            channels=64,
            num_convs=1,
            concat_input=False,
            num_classes=NUM_CLASSES,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            sampler=dict(type='OHEMPixelSamplerWithSpecialClassPenalty', gt_class=2, penalty_pred_class=0, thresh=0.7, min_kept=10000)
        ),
        dict(
            type='FCNHead',
            in_channels=128,
            in_index=1,
            channels=64,
            num_convs=1,
            concat_input=False,
            num_classes=NUM_CLASSES,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            sampler=dict(type='OHEMPixelSamplerWithSpecialClassPenalty', gt_class=2, penalty_pred_class=0, thresh=0.7, min_kept=10000)
        ),
        dict(
            type='STDCHead',
            in_channels=256,
            in_index=0,
            channels=64,
            num_convs=1,
            concat_input=False,
            num_classes=2,
            norm_cfg=norm_cfg,
            align_corners=True,
            boundary_threshold=0.1,
            loss_decode=[
                dict(type='CrossEntropyLoss', loss_name='loss_ce', use_sigmoid=True, loss_weight=1.0),
                dict(type='DiceLoss', loss_name='loss_dice', loss_weight=1.0)
            ]
        )
    ],
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)

# ===== Dataset / Dataloader Settings =====
dataset_type = 'LDPerceptionSegDataset'

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='Resize',
        scale=RESIZE_WH,
        keep_ratio=False
    ),
    dict(type='RandomFlip', prob=0.5),
    dict(type='ImageQualityAug', prob=0.8),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=RESIZE_WH, keep_ratio=False),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

TEST_FOLDERS = ['misseg_common', 'misseg_20250521', 'hedgehog_data']
REPEAT_FOLDERS = {'2501AHGE000A0092': 2, 'misseg_leaf_24507HGD00070081': 1, '25062HGG00020016': 2, '0610_hedgehog': 1}

train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    drop_last=True,
    dataset=dict(
        type=dataset_type,
        data_root='/home/mck/datasets/grass_seg_data_c4',
        class_names=CLASS_NAMES,
        pipeline=train_pipeline,
        repeat=REPEAT_FOLDERS,
        test_mode=False
    )
)

val_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root='/home/mck/datasets/grass_seg_data_c4',
        class_names=CLASS_NAMES,
        pipeline=test_pipeline,
        include=TEST_FOLDERS,
        test_mode=True
    )
)

test_dataloader = val_dataloader

# ===== Evaluation =====
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator

# ===== Optimization & Scheduler =====
optimizer = dict(type='AdamW', lr=0.0005, weight_decay=0.01)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)

param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        by_epoch=True,
        begin=0,
        end=50
    )
]

# ===== Runtime & Logging =====
default_scope = 'mmseg'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook')
)

env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl')
)

visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend')
    ],
    name='visualizer'
)

log_processor = dict(by_epoch=True)
log_level = 'INFO'
load_from = "work_dirs/stdc2_grass-c4-320x272-penalty_ohem_noaug/last_checkpoint"
resume = False

train_cfg = dict(by_epoch=True, max_epochs=70, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

