import os
import argparse
from functools import partial

import torch

import onnx
import onnxsim

from mmengine.model import revert_sync_batchnorm
from mmseg.apis import init_model


def export_onnx(config, checkpoint, device, input_hw=(320, 320)):
    model = init_model(config, checkpoint, device=device)
    assert device == 'cpu', "only cpu is supported"
    if device == 'cpu':
        model = revert_sync_batchnorm(model)
        
    assert checkpoint.endswith(".pth"), "checkpoint must be a pth file"
    dst_onnx_path = checkpoint.replace(".pth", ".onnx")
    dummy_input = torch.zeros(1, 3, input_hw[0], input_hw[1]).float()
    # output = model(dummy_input, mode="export_for_nb")
    torch.onnx.export(
        model, (dummy_input, None, "export_for_nb"), dst_onnx_path, input_names=["input"], output_names=["output"],
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
    checkpoint_path = args.checkpoint
    if os.path.basename(checkpoint_path) == "last_checkpoint":
        with open(checkpoint_path, "r") as f:
            checkpoint_path = f.read().strip()
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    
    export_onnx(args.config, checkpoint_path, device, input_hw=args.input_hw)
    print(args.input_hw)
    