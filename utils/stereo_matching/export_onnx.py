import os
import argparse
from functools import partial
import time
import torch

import onnx
import onnxsim

from mmengine.model import revert_sync_batchnorm
from mmseg.apis import init_model


def export_onnx(config, checkpoint, device, input_hw=(320, 320)):
    model = init_model(config, checkpoint, device=device)
    model.eval()
    assert device == 'cpu', "only cpu is supported"
    if device == 'cpu':
        model = revert_sync_batchnorm(model)
    
    
        
    assert checkpoint.endswith(".pth"), "checkpoint must be a pth file"
    dst_onnx_path = checkpoint.replace(".pth", ".onnx")
    dummy_left = torch.zeros(1, 3, input_hw[0], input_hw[1]).float()
    dummy_right = torch.zeros(1, 3, input_hw[0], input_hw[1]).float()
    model.backbone.forward = model.backbone.forward_inner
    torch.onnx.export(
        model, ((dummy_left, dummy_right), None, "export_stereo_matching"), dst_onnx_path, input_names=["left", "right"], output_names=["output"],
        dynamic_axes=None,
        opset_version=11
    )

    print(f"Model exported to {dst_onnx_path}")
    
    # **执行 onnxsim 进行简化**
    simplified_onnx_path = dst_onnx_path # .replace(".onnx", "_sim.onnx")
    model_onnx = onnx.load(dst_onnx_path)
    model_simp, check = onnxsim.simplify(model_onnx)

    if check:
        onnx.save(model_simp, simplified_onnx_path)
        print(f"Simplified ONNX model saved as {simplified_onnx_path}")
    else:
        print("ONNX simplification failed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export PyTorch model to ONNX")
    parser.add_argument("config", type=str)
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--input_hw", nargs="+", type=int, required=True)
    args = parser.parse_args()
    
    if args.checkpoint == "None":
        checkpoint_path = None
    elif '/' not in args.checkpoint:
        args.checkpoint = os.path.join("work_dirs", os.path.splitext(args.config.split("configs/")[-1])[0], args.checkpoint)
        print(f"给定的checkpoint不是完整路径，拓展为 {args.checkpoint}")
        time.sleep(1)
        if not args.checkpoint.endswith(".pth"):
            with open(args.checkpoint, "r") as f:
                args.checkpoint = f.read().strip()
    
        checkpoint_path = args.checkpoint
        if os.path.basename(checkpoint_path) == "last_checkpoint":
            with open(checkpoint_path, "r") as f:
                checkpoint_path = f.read().strip()
    else:
        raise ValueError("checkpoint must be a pth file")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    
    export_onnx(args.config, checkpoint_path, device, input_hw=args.input_hw)
    print(args.input_hw)
    