import torch
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("hat_model_path", type=str, help="HAT模型路径")
    parser.add_argument("mmseg_model_path", type=str, help="MMSEG模型路径")
    # parser.add_argument("output_model_path", type=str, help="输出模型路径")
    args = parser.parse_args()
    hat_model = torch.load("/data_SSD2/mck/DStereoV23/work_dirs/tmp_models_zbh/DStereoV23/float-checkpoint-last.pth.tar")
    # print(hat_model)

    mmseg_model = torch.load(args.mmseg_model_path)
    # print(mmseg_model)

    hks = sorted(list(hat_model["state_dict"].keys() ))
    mks = sorted(list(mmseg_model["state_dict"].keys() ))
    # mks = ['.'.join(mk.split('.')[1:]) for mk in mks]

    for hk, mk in zip(hks, mks):
        assert hk == '.'.join(mk.split('.')[1:])
        hat_model["state_dict"][hk] = mmseg_model["state_dict"][mk]

    torch.save(hat_model, "/data_SSD2/mck/DStereoV23/work_dirs/tmp_models_zbh/DStereoV23/float-checkpoint-last.pth.tar")
