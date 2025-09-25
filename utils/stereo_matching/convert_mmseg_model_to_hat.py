import torch

hat_model = torch.load("/data_SSD2/mck/DStereoV23/work_dirs/tmp_models_zbh/DStereoV23/float-checkpoint-last.pth.tar")
print(hat_model)

mmseg_model = torch.load("/home/mck/mmsegmentation/work_dirs/ld/stereo_matching/dstereoplus/dstereoplus_sceneflow/epoch_70.pth")
print(mmseg_model)

hks = sorted(list(hat_model["state_dict"].keys() ))
mks = sorted(list(mmseg_model["state_dict"].keys() ))
# mks = ['.'.join(mk.split('.')[1:]) for mk in mks]

for hk, mk in zip(hks, mks):
    assert hk == '.'.join(mk.split('.')[1:])
    hat_model["state_dict"][hk] = mmseg_model["state_dict"][mk]

torch.save(hat_model, "/data_SSD2/mck/DStereoV23/work_dirs/tmp_models_zbh/DStereoV23/float-checkpoint-last.pth.tar")
