# Copyright (c) OpenMMLab. All rights reserved.
"""SegStereo ONNX（语义分割 + 左视差）。

``left`` / ``right`` 可为**单张图片路径**或**目录**：
  - 均为文件：推理一对图；
  - 均为目录：按**同名文件名**配对（默认仅当前目录；``--recursive`` 时按相对路径镜像配对）。

默认仅保存一张横向拼接图：``<stem>_vis.png``（原图 | 分割彩图 | 分割叠加 | 视差灰度 | 视差 JET，高度对齐原图）。可选 ``--save-disp-npy`` 额外写 ``<stem>_disp.npy``。
"""

import argparse
import os
import sys

import cv2
import numpy as np
import onnxruntime as ort

# LD grass 四类（RGB），与 configs 中 LDPerception 系列 palette 一致；无 --palette 时用于彩色掩码与叠加
_DEFAULT_SEG_PALETTE_RGB = np.array(
    [[128, 0, 128], [0, 255, 0], [0, 255, 255], [0, 0, 255]], dtype=np.uint8)

_IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def _parse_float_list(s: str, n: int) -> list:
    parts = [float(x) for x in s.replace(',', ' ').split()]
    if len(parts) != n:
        raise argparse.ArgumentTypeError(f'需要 {n} 个数，得到 {len(parts)}: {s!r}')
    return parts


def preprocess_pair(
    left_bgr: np.ndarray,
    right_bgr: np.ndarray,
    out_h: int,
    out_w: int,
    mean: np.ndarray,
    std: np.ndarray,
    bgr_to_rgb: bool,
) -> tuple:
    """返回 (input_6ch_nchw, left_bgr_resized) 供可视化对齐。"""
    left = cv2.resize(left_bgr, (out_w, out_h))
    right = cv2.resize(right_bgr, (out_w, out_h))
    if bgr_to_rgb:
        left = cv2.cvtColor(left, cv2.COLOR_BGR2RGB)
        right = cv2.cvtColor(right, cv2.COLOR_BGR2RGB)
    left = left.astype(np.float32)
    right = right.astype(np.float32)
    left_t = np.transpose(left, (2, 0, 1))[np.newaxis, ...]
    right_t = np.transpose(right, (2, 0, 1))[np.newaxis, ...]
    mean = mean.reshape(1, 3, 1, 1).astype(np.float32)
    std = std.reshape(1, 3, 1, 1).astype(np.float32)
    left_n = (left_t - mean) / std
    right_n = (right_t - mean) / std
    x6 = np.concatenate([left_n, right_n], axis=1)
    return x6.astype(np.float32), left


def _static_hw_from_onnx(session: ort.InferenceSession):
    shape = session.get_inputs()[0].shape
    if len(shape) < 4:
        return None, None
    h, w = shape[2], shape[3]
    h = int(h) if isinstance(h, int) else None
    w = int(w) if isinstance(w, int) else None
    return h, w


def _build_session(onnx_path: str, device: str) -> ort.InferenceSession:
    if device.lower() == 'cuda':
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    else:
        providers = ['CPUExecutionProvider']
    return ort.InferenceSession(onnx_path, providers=providers)


def resolve_stereo_pairs(left: str, right: str, recursive: bool) -> list:
    """返回 [(left_path, right_path), ...]。"""
    left = os.path.abspath(left)
    right = os.path.abspath(right)
    lf, rf = os.path.isfile(left), os.path.isfile(right)
    ld, rd = os.path.isdir(left), os.path.isdir(right)
    if lf and rf:
        return [(left, right)]
    if ld and rd:
        pairs = []
        if not recursive:
            for name in sorted(os.listdir(left)):
                lp = os.path.join(left, name)
                if not os.path.isfile(lp):
                    continue
                if os.path.splitext(name)[1].lower() not in _IMG_EXT:
                    continue
                rp = os.path.join(right, name)
                if os.path.isfile(rp):
                    pairs.append((lp, rp))
                else:
                    print(f'跳过（右目录无同名文件）: {rp}', file=sys.stderr)
        else:
            for root, _, files in os.walk(left):
                for name in sorted(files):
                    if os.path.splitext(name)[1].lower() not in _IMG_EXT:
                        continue
                    lp = os.path.join(root, name)
                    rel = os.path.relpath(lp, left)
                    rp = os.path.join(right, rel)
                    if os.path.isfile(rp):
                        pairs.append((lp, rp))
                    else:
                        print(f'跳过（右树无对应文件）: {rp}', file=sys.stderr)
        return pairs
    raise ValueError(
        'left / right 须同为**图片文件**或同为**目录**；混合文件与目录不支持。')


def _logits_to_pred(logits: np.ndarray) -> np.ndarray:
    log = logits
    if log.ndim == 4:
        log = log[0]
    return np.argmax(log, axis=0).astype(np.uint8)


def _pred_to_color_bgr(pred: np.ndarray, pal_rgb: np.ndarray) -> np.ndarray:
    k = pal_rgb.shape[0]
    safe = np.clip(pred, 0, k - 1)
    rgb = pal_rgb[safe.reshape(-1)].reshape(*pred.shape, 3)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _make_seg_overlay_bgr(left_bgr_orig, pred_small, pal_rgb, alpha) -> np.ndarray:
    oh, ow = left_bgr_orig.shape[:2]
    pred_full = cv2.resize(
        pred_small, (ow, oh), interpolation=cv2.INTER_NEAREST)
    color = _pred_to_color_bgr(pred_full, pal_rgb)
    a = float(np.clip(alpha, 0.0, 1.0))
    return cv2.addWeighted(left_bgr_orig, 1.0 - a, color, a, 0)


def _disp_to_jet_bgr(
        disp: np.ndarray,
        *,
        vis_mode: str = 'fixed',
        vmin: float = 0.0,
        vmax: float = 192.0,
        mask_below: float = None) -> np.ndarray:
    """视差伪彩：OpenCV ``COLORMAP_JET``，不做通道反转（低视差→色标低端偏蓝，高视差→偏红）。

    默认 ``fixed``：按 [vmin, vmax] 线性映射；``adaptive``：按当前张量 min/max 拉伸。
    ``mask_below``：小于该值的像素显示为灰色。
    """
    d = np.squeeze(disp).astype(np.float32)
    if d.ndim != 2:
        raise ValueError(f'视差可视化期望 2D，得到 shape={d.shape}')
    if vis_mode == 'adaptive':
        dmin, dmax = float(d.min()), float(d.max())
        if dmax - dmin < 1e-6:
            jet_idx = np.zeros_like(d, dtype=np.uint8)
        else:
            norm = (d - dmin) / (dmax - dmin) * 255.0
            jet_idx = np.clip(norm, 0.0, 255.0).astype(np.uint8)
    elif vis_mode == 'fixed':
        lo, hi = float(vmin), float(vmax)
        if hi <= lo:
            raise ValueError(f'disp-vis: vmax ({hi}) 须大于 vmin ({lo})')
        t = (np.clip(d, lo, hi) - lo) / (hi - lo) * 255.0
        jet_idx = np.clip(t, 0.0, 255.0).astype(np.uint8)
    else:
        raise ValueError(f'未知 disp-vis-mode: {vis_mode!r}')
    vis = cv2.applyColorMap(jet_idx, cv2.COLORMAP_JET)
    if mask_below is not None:
        m = d < float(mask_below)
        if m.any():
            vis[m] = (64, 64, 64)
    return vis


def _hstack_panels(panels: list, sep_w: int) -> np.ndarray:
    """``panels`` 均为同高 H×W×3 uint8 BGR。"""
    if not panels:
        raise ValueError('panels 为空')
    oh = panels[0].shape[0]
    for i, p in enumerate(panels):
        if p.shape[0] != oh or p.ndim != 3 or p.shape[2] != 3:
            raise ValueError(f'panel {i} 须为 H×W×3 且高度一致')
    if sep_w <= 0:
        return np.hstack(panels)
    sep = np.full((oh, sep_w, 3), 255, dtype=np.uint8)
    out = panels[0]
    for p in panels[1:]:
        out = np.hstack([out, sep, p])
    return out


def _disp_to_gray_bgr(disp_np: np.ndarray, vmax: float) -> np.ndarray:
    """模型分辨率下的视差 → H×W×3 灰度 BGR（``[0,vmax]``→0~255）。"""
    d = np.squeeze(disp_np).astype(np.float32)
    hi = float(vmax)
    if hi <= 0:
        raise ValueError('disp-vis-vmax 须 > 0')
    u8 = np.clip(d / hi * 255.0, 0.0, 255.0).astype(np.uint8)
    return cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)


def _concat_seg_disp_vis(
        left_bgr: np.ndarray,
        pred_s: np.ndarray,
        pal_rgb: np.ndarray,
        overlay_alpha: float,
        disp_np: np.ndarray,
        disp_jet_kw: dict,
        vmax_gray: float,
        sep_w: int) -> np.ndarray:
    oh, ow = left_bgr.shape[:2]
    pred_full = cv2.resize(pred_s, (ow, oh), interpolation=cv2.INTER_NEAREST)
    seg_bgr = _pred_to_color_bgr(pred_full, pal_rgb)
    overlay_bgr = _make_seg_overlay_bgr(
        left_bgr, pred_s, pal_rgb, overlay_alpha)
    gray_bgr = cv2.resize(
        _disp_to_gray_bgr(disp_np, vmax_gray),
        (ow, oh),
        interpolation=cv2.INTER_LINEAR)
    jet_bgr = cv2.resize(
        _disp_to_jet_bgr(disp_np, **disp_jet_kw),
        (ow, oh),
        interpolation=cv2.INTER_LINEAR)
    return _hstack_panels(
        [left_bgr, seg_bgr, overlay_bgr, gray_bgr, jet_bgr], sep_w)


def _concat_disp_only_vis(
        left_bgr: np.ndarray,
        disp_np: np.ndarray,
        disp_jet_kw: dict,
        vmax_gray: float,
        sep_w: int) -> np.ndarray:
    oh, ow = left_bgr.shape[:2]
    gray_bgr = cv2.resize(
        _disp_to_gray_bgr(disp_np, vmax_gray),
        (ow, oh),
        interpolation=cv2.INTER_LINEAR)
    jet_bgr = cv2.resize(
        _disp_to_jet_bgr(disp_np, **disp_jet_kw),
        (ow, oh),
        interpolation=cv2.INTER_LINEAR)
    return _hstack_panels([left_bgr, gray_bgr, jet_bgr], sep_w)


def _concat_seg_only_vis(
        left_bgr: np.ndarray,
        pred_s: np.ndarray,
        pal_rgb: np.ndarray,
        overlay_alpha: float,
        sep_w: int) -> np.ndarray:
    oh, ow = left_bgr.shape[:2]
    pred_full = cv2.resize(pred_s, (ow, oh), interpolation=cv2.INTER_NEAREST)
    seg_bgr = _pred_to_color_bgr(pred_full, pal_rgb)
    overlay_bgr = _make_seg_overlay_bgr(
        left_bgr, pred_s, pal_rgb, overlay_alpha)
    return _hstack_panels([left_bgr, seg_bgr, overlay_bgr], sep_w)


def _disp_jet_kwargs(args):
    return dict(
        vis_mode=args.disp_vis_mode,
        vmin=args.disp_vis_vmin,
        vmax=args.disp_vis_vmax,
        mask_below=args.disp_vis_mask_below)


def _process_ort_outputs(ort_outs, args, pal_rgb, base_path, left_bgr):
    """根据 ONNX 输出个数写盘并打印一行摘要。"""
    if len(ort_outs) == 2:
        a, b = ort_outs[0], ort_outs[1]
        if args.swap_outputs:
            a, b = b, a

        def is_disp_arr(arr):
            if arr.ndim == 4:
                return arr.shape[1] == 1
            if arr.ndim == 3:
                return arr.shape[0] == 1
            return False

        if is_disp_arr(a) and not is_disp_arr(b):
            disp_np, seg_np = a, b
        elif is_disp_arr(b) and not is_disp_arr(a):
            seg_np, disp_np = a, b
        else:
            seg_np, disp_np = a, b

        pred_s = _logits_to_pred(seg_np)
        strip = _concat_seg_disp_vis(
            left_bgr,
            pred_s,
            pal_rgb,
            args.overlay_alpha,
            disp_np,
            _disp_jet_kwargs(args),
            float(args.disp_vis_vmax),
            max(0, int(args.vis_sep)),
        )
        vis_path = f'{base_path}{args.vis_suffix}'
        cv2.imwrite(vis_path, strip)
        msg = vis_path
        if args.save_disp_npy:
            np.save(f'{base_path}_disp.npy', np.squeeze(disp_np))
            msg += f', {base_path}_disp.npy'
        print(f'已保存: {msg}')
    elif len(ort_outs) == 1:
        y = ort_outs[0]
        if y.ndim == 4 and y.shape[1] == 1:
            strip = _concat_disp_only_vis(
                left_bgr,
                y,
                _disp_jet_kwargs(args),
                float(args.disp_vis_vmax),
                max(0, int(args.vis_sep)),
            )
            vis_path = f'{base_path}{args.vis_suffix}'
            cv2.imwrite(vis_path, strip)
            msg = vis_path
            if args.save_disp_npy:
                np.save(f'{base_path}_disp.npy', np.squeeze(y))
                msg += f', {base_path}_disp.npy'
            print(f'单输出视差: {msg}')
        else:
            pred_s = _logits_to_pred(y)
            strip = _concat_seg_only_vis(
                left_bgr,
                pred_s,
                pal_rgb,
                args.overlay_alpha,
                max(0, int(args.vis_sep)),
            )
            vis_path = f'{base_path}{args.vis_suffix}'
            cv2.imwrite(vis_path, strip)
            print(f'单输出分割: {vis_path}')
    else:
        raise RuntimeError(f'不支持的 ONNX 输出个数: {len(ort_outs)}')


def infer_one_pair(
        args,
        left_path: str,
        right_path: str,
        session,
        inps,
        outs,
        h: int,
        w: int,
        mean: np.ndarray,
        std: np.ndarray,
        bgr_to_rgb: bool,
        pal_rgb: np.ndarray):
    left_bgr = cv2.imread(left_path)
    right_bgr = cv2.imread(right_path)
    if left_bgr is None or right_bgr is None:
        raise FileNotFoundError(f'读取失败: {left_path} / {right_path}')

    x6, _ = preprocess_pair(left_bgr, right_bgr, h, w, mean, std, bgr_to_rgb)
    ort_inputs = {}
    if len(inps) == 1:
        ort_inputs[inps[0].name] = x6
    elif len(inps) == 2:
        ort_inputs[inps[0].name] = x6[:, :3]
        ort_inputs[inps[1].name] = x6[:, 3:6]
    else:
        raise RuntimeError(f'不支持的 ONNX 输入个数: {len(inps)}（仅支持 1 或 2）。')

    output_names = [o.name for o in outs]
    ort_outs = session.run(output_names, ort_inputs)

    stem = os.path.splitext(os.path.basename(left_path))[0]
    base_path = os.path.join(args.out_dir, stem)
    _process_ort_outputs(ort_outs, args, pal_rgb, base_path, left_bgr)


def main():
    parser = argparse.ArgumentParser(
        description='SegStereo ONNX：左/右可为单图或目录（目录时按同名配对批量推理）')
    parser.add_argument('onnx_path', type=str, help='ONNX 模型路径')
    parser.add_argument(
        'left',
        type=str,
        help='左图文件路径，或左图所在目录（与 right 同时为目录则批量）')
    parser.add_argument(
        'right',
        type=str,
        help='右图文件路径，或与左目录结构对应的右图目录')
    parser.add_argument(
        '--out-dir',
        type=str,
        default='.',
        help='保存目录；批量时所有结果平铺在该目录下（左图主名作前缀）')
    parser.add_argument(
        '--recursive',
        action='store_true',
        help='左/右均为目录时，递归遍历左目录并按相对路径在右目录找同名文件')
    parser.add_argument('--input-h', type=int, default=None)
    parser.add_argument('--input-w', type=int, default=None)
    parser.add_argument(
        '--mean',
        type=str,
        default='123.675 116.28 103.53',
        help='RGB 均值（与训练 data_preprocessor 一致）')
    parser.add_argument(
        '--std',
        type=str,
        default='58.395 57.12 57.375',
        help='RGB 方差')
    parser.add_argument('--no-bgr-to-rgb', action='store_true')
    parser.add_argument('--swap-outputs', action='store_true')
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'])
    parser.add_argument('--palette', type=str, default=None)
    parser.add_argument('--overlay-alpha', type=float, default=0.5)
    parser.add_argument(
        '--vis-sep',
        type=int,
        default=0,
        help='拼接图中列间白色分隔条宽度（像素），默认 0 不加；需要时设为正整数')
    parser.add_argument(
        '--vis-suffix',
        type=str,
        default='_vis.png',
        help='拼接图文件名后缀（接在左图主名后）')
    parser.add_argument(
        '--save-disp-npy',
        action='store_true',
        help='除拼接图外另存浮点视差 <stem>_disp.npy')
    parser.add_argument(
        '--disp-vis-mode',
        type=str,
        choices=['fixed', 'adaptive'],
        default='fixed',
        help='视差伪彩映射：fixed=按 vmin~vmax 固定色标（默认 0~192，与 max_disp 一致，抑制乱码感）；'
        'adaptive=每图 min-max（易放大近零噪声）')
    parser.add_argument(
        '--disp-vis-vmin',
        type=float,
        default=0.0,
        help='fixed 模式下色标下限（像素视差）')
    parser.add_argument(
        '--disp-vis-vmax',
        type=float,
        default=192.0,
        help='fixed 模式色标上限；拼接图中灰度视差条为 视差/vmax*255，建议与训练 max_disp 一致')
    parser.add_argument(
        '--disp-vis-mask-below',
        type=float,
        default=None,
        help='可选：预测视差低于该值的像素在伪彩上显示为灰色（如 1.0 抑制不可靠近零）')
    args = parser.parse_args()

    # 默认 fixed，避免 adaptive 的 per-image min-max 把近零噪声拉成「彩噪乱码」
    dmb = args.disp_vis_mask_below
    dmb_s = f', mask_below={dmb}' if dmb is not None else ''
    print(
        f'视差伪彩: mode={args.disp_vis_mode}, '
        f'range=[{args.disp_vis_vmin}, {args.disp_vis_vmax}]{dmb_s}')

    pairs = resolve_stereo_pairs(args.left, args.right, args.recursive)
    if not pairs:
        print('未找到可推理的左右图对。', file=sys.stderr)
        sys.exit(1)

    mean = np.array(_parse_float_list(args.mean, 3), dtype=np.float32)
    std = np.array(_parse_float_list(args.std, 3), dtype=np.float32)
    bgr_to_rgb = not args.no_bgr_to_rgb

    os.makedirs(args.out_dir, exist_ok=True)

    session = _build_session(args.onnx_path, args.device)
    inps = session.get_inputs()
    outs = session.get_outputs()
    onnx_h, onnx_w = _static_hw_from_onnx(session)
    h = args.input_h if args.input_h is not None else onnx_h
    w = args.input_w if args.input_w is not None else onnx_w
    if h is None or w is None:
        raise ValueError(
            '无法确定输入分辨率：请在 ONNX 中使用静态 H、W，或通过 --input-h / --input-w 指定。')

    pal_rgb = _DEFAULT_SEG_PALETTE_RGB.copy()
    if args.palette:
        loaded = np.load(args.palette)
        pal_rgb = loaded.astype(np.uint8) if loaded.dtype != np.uint8 else loaded

    mode = '单对' if len(pairs) == 1 else f'批量 {len(pairs)} 对'
    print(f'模式: {mode}')

    for lp, rp in pairs:
        infer_one_pair(
            args, lp, rp, session, inps, outs, h, w, mean, std, bgr_to_rgb,
            pal_rgb)

    print('输入节点:', [(i.name, i.shape) for i in inps])
    print('输出节点:', [(o.name, o.shape) for o in outs])


if __name__ == '__main__':
    main()
