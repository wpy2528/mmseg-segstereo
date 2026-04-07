"""将 SegStereo 训练权重导出为 ONNX（双输入 RGB 左/右，双输出分割 logits + 视差）。

与训练 ``SegDataPreProcessor`` 一致：请在推理时使用与 config 相同的 ``--mean`` / ``std`` /
``--no-bgr-to-rgb``（见 ``infer_onnx_segstereo.py``）。

示例::

    python utils/stereo_matching/export_onnx_segstereo.py \\
        configs/segstereo/stdc2_grass-c4-320x272-penalty_fp_bg_0919_segstereo.py \\
        work_dirs/.../last_checkpoint \\
        --input-hw 272 320
"""

import argparse
import os
import time

import onnx
import torch

try:
    import onnxsim
except ImportError:
    onnxsim = None
from mmengine.model import revert_sync_batchnorm

from mmseg.apis import init_model


def export_onnx_segstereo(config: str, checkpoint: str, input_hw: tuple, opset: int = 16):
    device = 'cpu'
    model = init_model(config, checkpoint, device=device)
    model.eval()
    model = revert_sync_batchnorm(model)

    assert str(checkpoint).endswith('.pth'), 'checkpoint 须为 .pth'
    dst_onnx_path = str(checkpoint).replace('.pth', '.onnx')

    h, w = input_hw[0], input_hw[1]
    dummy_left = torch.zeros(1, 3, h, w, dtype=torch.float32)
    dummy_right = torch.zeros(1, 3, h, w, dtype=torch.float32)
    export_args = ((dummy_left, dummy_right), None, 'export_segstereo')

    with torch.no_grad():
        test_out = model((dummy_left, dummy_right), None, 'export_segstereo')
        assert isinstance(test_out, tuple) and len(test_out) == 2, test_out

    torch.onnx.export(
        model,
        export_args,
        dst_onnx_path,
        input_names=['left', 'right'],
        output_names=['seg_logits', 'disp'],
        dynamic_axes=None,
        opset_version=opset,
    )
    print(f'Model exported to {dst_onnx_path}')

    if onnxsim is None:
        print('未安装 onnxsim，跳过图简化（可 pip install onnxsim）')
    else:
        model_onnx = onnx.load(dst_onnx_path)
        model_simp, check = onnxsim.simplify(model_onnx)
        if check:
            onnx.save(model_simp, dst_onnx_path)
            print(f'Simplified ONNX saved: {dst_onnx_path}')
        else:
            print('ONNX simplification failed, keeping unsimplified graph.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Export SegStereo checkpoint to ONNX (left/right -> seg_logits, disp)')
    parser.add_argument('config', type=str)
    parser.add_argument('checkpoint', type=str)
    parser.add_argument(
        '--input-hw',
        nargs=2,
        type=int,
        default=[272, 320],
        metavar=('H', 'W'),
        help='输入张量高、宽（与训练 Resize 一致；默认 272 320）')
    parser.add_argument(
        '--opset',
        type=int,
        default=16,
        help='ONNX opset（SegStereo/IGEV 含 grid_sample 时需 >=16，默认 16）')
    args = parser.parse_args()

    ckpt = args.checkpoint
    if '/' not in ckpt:
        cfg_rel = args.config.split('configs/')[-1]
        ckpt = os.path.join('work_dirs', os.path.splitext(cfg_rel)[0], ckpt)
        print(f'checkpoint 已扩展为: {ckpt}')
        time.sleep(0.3)
    if not ckpt.endswith('.pth'):
        with open(ckpt, 'r', encoding='utf-8') as f:
            ckpt = f.read().strip()
    if os.path.basename(ckpt) == 'last_checkpoint':
        with open(ckpt, 'r', encoding='utf-8') as f:
            ckpt = f.read().strip()

    export_onnx_segstereo(args.config, ckpt, tuple(args.input_hw), opset=args.opset)
    print('input_hw:', args.input_hw)
