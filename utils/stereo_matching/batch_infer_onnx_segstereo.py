#!/usr/bin/env python3
"""对双目数据根目录批量调用 ``infer_onnx_segstereo.py``（左图在 ``**/images/*.jpg``，右图为同名在 ``images_right``）。"""

import argparse
import glob
import os
import subprocess
import sys


def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    infer_py = os.path.join(root, 'utils', 'stereo_matching', 'infer_onnx_segstereo.py')

    parser = argparse.ArgumentParser(
        description='Batch ONNX inference for SegStereo val/test tree')
    parser.add_argument('onnx_path', type=str)
    parser.add_argument(
        'data_root',
        type=str,
        help='含 images / images_right 的数据根（与 LDPerceptionStereoSegDataset 一致）')
    parser.add_argument(
        '--out-dir',
        type=str,
        required=True,
        help='输出目录（每样本子目录或平铺，见 --flat）')
    parser.add_argument(
        '--flat',
        action='store_true',
        help='所有结果直接写在 out-dir 下（默认每图单独子目录）')
    parser.add_argument('--input-h', type=int, default=None)
    parser.add_argument('--input-w', type=int, default=None)
    parser.add_argument(
        '--mean',
        type=str,
        default='0 0 0',
        help='与训练 data_preprocessor 一致（本 grass 配置为 0 0 0）')
    parser.add_argument(
        '--std',
        type=str,
        default='255 255 255',
        help='与训练 data_preprocessor 一致（本 grass 配置为 255 255 255）')
    parser.add_argument(
        '--no-bgr-to-rgb',
        action='store_true',
        help='与训练关闭 bgr_to_rgb 时加此参数')
    parser.add_argument(
        '--swap-outputs',
        action='store_true',
        help='传给 infer_onnx_segstereo')
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'])
    parser.add_argument('--palette', type=str, default=None)
    parser.add_argument(
        '--overlay-alpha',
        type=float,
        default=None,
        help='传给 infer（默认用脚本内建默认 0.5）')
    parser.add_argument('--vis-sep', type=int, default=None)
    parser.add_argument('--vis-suffix', type=str, default=None)
    parser.add_argument(
        '--save-disp-npy',
        action='store_true',
        help='传给 infer，额外保存 disp.npy')
    parser.add_argument(
        '--disp-vis-mode',
        type=str,
        default=None,
        choices=['fixed', 'adaptive'],
        help='传给 infer，默认 infer 内 fixed')
    parser.add_argument('--disp-vis-vmin', type=float, default=None)
    parser.add_argument('--disp-vis-vmax', type=float, default=None)
    parser.add_argument('--disp-vis-mask-below', type=float, default=None)
    args = parser.parse_args()

    pattern = os.path.join(args.data_root, '**', 'images', '*.jpg')
    left_paths = sorted(glob.glob(pattern, recursive=True))
    if not left_paths:
        print(f'未找到左图: {pattern}', file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)
    needle = '/' + 'images' + '/'
    rep = '/' + 'images_right' + '/'
    n_ok = 0
    for left in left_paths:
        if needle not in left.replace('\\', '/'):
            continue
        right = left.replace('\\', '/').replace(needle, rep, 1)
        if not os.path.isfile(right):
            print(f'跳过（无右图）: {right}', file=sys.stderr)
            continue
        stem = os.path.splitext(os.path.basename(left))[0]
        if args.flat:
            out_sub = args.out_dir
        else:
            out_sub = os.path.join(args.out_dir, stem)
        os.makedirs(out_sub, exist_ok=True)

        cmd = [
            sys.executable,
            infer_py,
            args.onnx_path,
            left,
            right,
            '--out-dir',
            out_sub,
            '--mean',
            args.mean,
            '--std',
            args.std,
            '--device',
            args.device,
        ]
        if args.input_h is not None:
            cmd.extend(['--input-h', str(args.input_h)])
        if args.input_w is not None:
            cmd.extend(['--input-w', str(args.input_w)])
        if args.no_bgr_to_rgb:
            cmd.append('--no-bgr-to-rgb')
        if args.swap_outputs:
            cmd.append('--swap-outputs')
        if args.palette:
            cmd.extend(['--palette', args.palette])
        if args.overlay_alpha is not None:
            cmd.extend(['--overlay-alpha', str(args.overlay_alpha)])
        if args.vis_sep is not None:
            cmd.extend(['--vis-sep', str(args.vis_sep)])
        if args.vis_suffix is not None:
            cmd.extend(['--vis-suffix', args.vis_suffix])
        if args.save_disp_npy:
            cmd.append('--save-disp-npy')
        if args.disp_vis_mode is not None:
            cmd.extend(['--disp-vis-mode', args.disp_vis_mode])
        if args.disp_vis_vmin is not None:
            cmd.extend(['--disp-vis-vmin', str(args.disp_vis_vmin)])
        if args.disp_vis_vmax is not None:
            cmd.extend(['--disp-vis-vmax', str(args.disp_vis_vmax)])
        if args.disp_vis_mask_below is not None:
            cmd.extend([
                '--disp-vis-mask-below',
                str(args.disp_vis_mask_below),
            ])

        subprocess.run(cmd, check=True)
        n_ok += 1

    print(f'完成 {n_ok} / {len(left_paths)} 对')


if __name__ == '__main__':
    main()
