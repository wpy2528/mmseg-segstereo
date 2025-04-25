_base_ = [
    '../_base_/models/stdc.py', 
    '../_base_/datasets/ld_perception_seg_v1.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_240k.py'
]
crop_size = (320, 320)
data_preprocessor = dict(size=crop_size)
checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/stdc/stdc2_20220308-7dbd9127.pth'  # noqa
model = dict(data_preprocessor=data_preprocessor,
             backbone=dict(backbone_cfg=dict(stdc_type='STDCNet2',
                                             init_cfg=dict(type='Pretrained', checkpoint=checkpoint))))


param_scheduler = [
    dict(type='LinearLR', by_epoch=False, start_factor=0.1, begin=0, end=1000),
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=1000,
        end=240000,
        by_epoch=False,
    )
]


data_root = 'data/perception_segmentation/0704_cleaned'
train_dataloader = dict(dataset=dict(data_root=data_root), batch_size=16)
val_dataloader = dict(dataset=dict(data_root=data_root))
test_dataloader = val_dataloader
