# 由 configs/ld/grass/stdc2_grass-c4-320x288-penalty_fp_bg_0919_concat.py 改写：
# - 语义分支为 STDC2 + FCN / 辅助头（与 concat 版一致），不再使用 PSPNet/ResNet50。
# - 模型类型为 SegStereo：在上述语义分支外增加 SegStereoIGEVDisparityBranch。
#
# 联合训练要求：batch 内 inputs 为 6 通道 [左|右]，且 SegDataSample 含 gt_sem_seg 与 gt_disp。
# 使用 ``LDPerceptionStereoSegDataset``：在原有 ``.../images/左图.jpg`` 枚举基础上，自动映射
# ``.../images_right/同名.jpg``、``.../disparity/同名.png``（目录名与视差后缀见 dataset 参数）。

# ===== Data roots（train=segstereo，val=segstereo_val）=====
DATA_ROOT_TRAIN = '/data_SSD2/datasets/segstereo'
DATA_ROOT_VAL = '/data_SSD2/datasets/segstereo_val'
# 标注 PNG 像素为 0～C-1，与 CLASS_NAMES 顺序一致；忽略类像素为 255（与 seg_pad_val 一致）

# ===== Global Constants =====
CLASS_NAMES = ("background", "grass", "soil", "animal")
NUM_CLASSES = len(CLASS_NAMES)
RESIZE_WH = (320, 288)
BATCH_PAD_HW = (288, 320)

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

# ===== Model：SegStereo + STDC（与 grass concat 语义结构一致）+ 视差分支 =====
model = dict(
    type='SegStereo',
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
        # False：与 bilinear 配合可避免 mmcv resize 在「非 (n*(in-1)+1)」尺寸下对 align_corners=True 的告警
        align_corners=False,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        sampler=dict(type='OHEMPixelSamplerWithSpecialClassPenalty', penalty_factor=3.0, gt_class=-1, penalty_pred_class=0, thresh=0.7, min_kept=10000)
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
            sampler=dict(type='OHEMPixelSamplerWithSpecialClassPenalty', penalty_factor=3.0, gt_class=-1, penalty_pred_class=0, thresh=0.7, min_kept=10000)
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
            sampler=dict(type='OHEMPixelSamplerWithSpecialClassPenalty', penalty_factor=3.0, gt_class=-1, penalty_pred_class=0, thresh=0.7, min_kept=10000)
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
            align_corners=False,
            boundary_threshold=0.1,
            loss_decode=[
                dict(type='CrossEntropyLoss', loss_name='loss_ce', use_sigmoid=True, loss_weight=1.0),
                dict(type='DiceLoss', loss_name='loss_dice', loss_weight=1.0)
            ]
        )
    ],
    disparity_branch=dict(
        type='SegStereoIGEVDisparityBranch',
        max_disp=192,
        mixed_precision=False,
        precision_dtype='float32',
        multiple=32,
        use_gru=True,
        gru_iters=4,
        # 与 decode_head num_classes 一致；在 match 特征上拼接 1x1 投影后的语义（96+fused 须能被 8 整除）
        semantic_channels=NUM_CLASSES,
        fused_channels=32,
    ),
    # 语义特征融合（fuse_semantic_to_disp=True 时生效；False 则无事发生）
    fuse_semantic_to_disp=True,
    # 三级嵌入独立开关：match=logits 拼 GWC；hg8/hg16=STDC x[index] 残差进 IGEV features[1]/[2]
    fuse_semantic_level_match=True,
    fuse_semantic_level_hg8=True,
    fuse_semantic_level_hg16=True,
    stdc_index_hg8=2,
    stdc_index_hg16=1,
    # 前若干 epoch 不传 left_sem 时，IGEV 视差分支会退化为无拼接语义（与 fuse_semantic_warmup_epochs /
    # SegStereoSemanticFuseHook 一致）。GC-Net 类 SegStereoDisparityBranch 仍要求预热期也传入语义或设 semantic_channels=0。
    fuse_semantic_warmup_epochs=0,
    loss_disp_weight=0.35,
    loss_disp_smooth_weight=0.01,
    loss_semantic_warp_weight=0.05,
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)

# ===== Dataset / Dataloader（双目 pipeline；数据字段需与之一致）=====
dataset_type = 'LDPerceptionStereoSegDataset'

# 双目阶段仅有 left_img/right_img，无 img：须用 ResizeStereoImages（同步左右图、语义图、视差；视差数值按水平缩放）
# 勿用 mmcv/mmseg 的 Resize（依赖 results['img']，且视差在 seg_fields 中仅几何缩放、未乘 w_scale）
train_pipeline = [
    dict(type='LoadStereoImages', color_type='color', imdecode_backend='cv2'),
    dict(type='LoadAnnotations'),
    dict(type='LoadStereoMatchingAnnotations'),
    dict(type='ResizeStereoImages', scale=RESIZE_WH, keep_ratio=False),
    dict(type='PackSegStereoInputs')
]

test_pipeline = [
    dict(type='LoadStereoImages', color_type='color', imdecode_backend='cv2'),
    dict(type='LoadAnnotations'),
    dict(type='LoadStereoMatchingAnnotations'),
    dict(type='ResizeStereoImages', scale=RESIZE_WH, keep_ratio=False),
    dict(type='PackSegStereoInputs')
]

train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    drop_last=True,
    dataset=dict(
        type=dataset_type,
        data_root=DATA_ROOT_TRAIN,
        class_names=CLASS_NAMES,
        pipeline=train_pipeline,
        repeat=None,
        exclude=None,
        test_mode=False,
    )
)

val_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=DATA_ROOT_VAL,
        class_names=CLASS_NAMES,
        pipeline=test_pipeline,
        include=None,
        exclude=None,
        test_mode=True,
    )
)

test_dataloader = val_dataloader

# ===== Evaluation =====
# 列表形式：同时输出分割 IoU 与视差 EPE / Bad-1,3,5（需 val 为 6 通道且含 gt_disp）
val_evaluator = [
    dict(
        type='IoUMetric',
        iou_metrics=['mIoU'],
        output_dir='pgs',
        calc_per_sample_metric=True),
    dict(
        type='DisparityMetric',
        max_disp=192,
        bad_thresholds=(1.0, 3.0, 5.0)),
]
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
    checkpoint=dict(type='SessionCheckpointHook', interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'),
    segstereo_fuse=dict(
        type='SegStereoSemanticFuseHook',
        warmup_epochs=None,
        auto_plateau=False,
        # 训练时建议微调
        # auto_plateau=True
        # plateau_metric='mIoU', 
        # plateau_patience=2, 
        # plateau_min_delta=0.05,
        # plateau_only_after_epoch=5,
        # warmup_epochs=999,
    ),
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

train_cfg = dict(by_epoch=True, max_epochs=70, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
