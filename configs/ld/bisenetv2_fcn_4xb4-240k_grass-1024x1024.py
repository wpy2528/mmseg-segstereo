_base_ = [
    '../_base_/models/bisenetv2.py',
    '../_base_/datasets/ld_perception_seg_v1.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_240k.py'
]
crop_size = (320, 320)
data_preprocessor = dict(size=crop_size)
model = dict(data_preprocessor=data_preprocessor)

data_root = 'data/perception_segmentation/0704_cleaned'
train_dataloader = dict(dataset=dict(data_root=data_root), batch_size=16)
val_dataloader = dict(dataset=dict(data_root=data_root))

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
optimizer = dict(type='SGD', lr=0.05, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer)
train_dataloader = dict(batch_size=4, num_workers=4)
val_dataloader = dict(batch_size=1, num_workers=4)
test_dataloader = val_dataloader
