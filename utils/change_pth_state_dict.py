import torch
from copy import deepcopy


if __name__ == "__main__":
    prefix = "backbone.backbone."

    pth_path = open("/home/mck/mmpretrain/work_dirs/stdc_imagenet/last_checkpoint").read()
    state_dict = torch.load(pth_path)
    new_state_dict = deepcopy(state_dict)
    new_state_dict["state_dict"].clear()
    for k, v in state_dict["state_dict"].items():
        print(k)
        if k.startswith(prefix):
            new_k = k[len(prefix):]
            print(new_k)
            new_state_dict["state_dict"][new_k] = v
    torch.save(new_state_dict, "checkpoints/stdc_cut1_pretrain.pth")

