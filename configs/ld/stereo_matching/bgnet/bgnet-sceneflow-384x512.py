# ===== Global Constants =====
NUM_CLASSES = 4
CROP_HW = (384, 512)
BATCH_PAD_HW = (384, 512)

norm_cfg = dict(type='BN', requires_grad=True)

data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[0.0, 0.0], # 输入的时候把grayscale的左目和右目沿着通道轴拼接，所以mean和std都是两个
    std=[255.0, 255.0],
    size=BATCH_PAD_HW,
    bgr_to_rgb=False, # 不能交换通道（不能交换左右目）
    pad_val=0,
    seg_pad_val=255
)

# ===== Model Settings =====
model = dict(
    type='DepthEstimator',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='BGNet_Plus'),
    decode_head=dict(
        type='StereoMatchingHead',
        in_channels=512,
        in_index=3,
        channels=512,
        num_classes=NUM_CLASSES,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=[
            dict(type='L1Loss', loss_name='loss_l1', loss_weight=1.0),
        ]),
    auxiliary_head=None,
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# ===== Dataset / Dataloader Settings =====
dataset_type = 'LDPerceptionStereoMatchingDataset'

train_pipeline = [
    dict(type='LoadStereoImages'),
    dict(type='LoadStereoMatchingAnnotations'),
    # dict(type='RandomCrop', crop_size=CROP_HW),
    dict(type='CREStereoAugmentor', image_hw=CROP_HW),
    # dict(
    #     type='Resize',
    #     scale=RESIZE_WH,
    #     keep_ratio=False
    # ),
    # dict(type='RandomFlip', prob=0.5),
    dict(type='PackStereoMatchingInputs')
]

test_pipeline = [
    dict(type='LoadStereoImages'),
    dict(type='LoadStereoMatchingAnnotations'),
    dict(type='PackStereoMatchingInputs')
]

train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    drop_last=True,
    dataset=dict(
        type=dataset_type,
        data_root='stereo_datasets/sceneflow/driving',
        num_classes=NUM_CLASSES,
        pipeline=train_pipeline,
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
        data_root='stereo_datasets/sceneflow/driving',
        num_classes=NUM_CLASSES,
        pipeline=test_pipeline,
        test_mode=True
    )
)

test_dataloader = val_dataloader

# ===== Evaluation =====
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'], output_dir='pgs', calc_per_sample_metric=True)
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
load_from = None
resume = False

train_cfg = dict(by_epoch=True, max_epochs=70, val_interval=100)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

