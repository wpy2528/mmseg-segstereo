# Copyright (c) OpenMMLab. All rights reserved.
"""SegStereo PyTorch 推理（``.pth``）：语义分割 + 左视差。

经 ``model.test_step`` → ``SegDataPreProcessor``，与训练时 val 路径一致（6 通道 BGR 0～255 再按 config 做
``bgr_to_rgb`` 与 ``(x-mean)/std``）。**必须**提供 ``--config`` 与 checkpoint。

ONNX 推理请使用 ``infer_onnx_segstereo.py``（手写 resize+归一化，需与 config 对齐）。

用法示例::

    python utils/stereo_matching/infer_segstereo.py \\
        configs/segstereo/stdc2_grass-c4-320x288-penalty_fp_bg_0919_segstereo.py \\
        work_dirs/.../epoch_5.pth \\
        /path/left.jpg /path/right.jpg \\
        --out-dir work_dirs/segstereo/pth_val/

仅左图（右支输入复制左图，视差无意义、默认只画分割）::

    python utils/stereo_matching/infer_segstereo.py cfg.py ckpt.pth left_dir/ --left-only --out-dir out/
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Optional

import cv2
import numpy as np
import torch

from mmseg.apis import init_model
from mmseg.structures import SegDataSample

# ---------------------------------------------------------------------------
# 与 ``infer_onnx_segstereo`` 一致的默认调色板 RGB (K,3) uint8
# ---------------------------------------------------------------------------
_DEFAULT_SEG_PALETTE_RGB = np.asarray(
    [
        [128, 0, 128],
        [0, 255, 0],
        [0, 255, 255],
        [0, 0, 255],
    ],
    dtype=np.uint8,
)

_IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.JPG', '.JPEG', '.PNG')


def _is_image_path(p: str) -> bool:
    return p.lower().endswith(_IMG_EXTS)


def infer_default_resize_backend() -> str:
    try:
        import mmcv  # noqa: F401
        return 'mmcv'
    except Exception:
        return 'cv2'


def num_classes_from_decode_config(config_path: str) -> int:
    from mmengine.config import Config

    cfg = Config.fromfile(config_path)
    model = cfg.model
    dh = model.get('decode_head', None)
    if dh is None:
        raise KeyError('model.decode_head')
    if isinstance(dh, dict):
        return int(dh['num_classes'])
    if isinstance(dh, (list, tuple)):
        return int(dh[0]['num_classes'])
    raise TypeError(type(dh))


def input_hw_from_config(config_path: str) -> tuple[int, int]:
    """``data_preprocessor.size`` 为 ``(H, W)``，与训练 config 一致。"""
    from mmengine.config import Config

    cfg = Config.fromfile(config_path)
    dp = cfg.model.get('data_preprocessor', {})
    size = dp.get('size', None)
    if size is None:
        raise ValueError(
            'config.model.data_preprocessor 未设置 size，请用 --input-h / --input-w')
    h, w = int(size[0]), int(size[1])
    return h, w


def resolve_left_only_paths(left: str, recursive: bool) -> list[str]:
    left = os.path.abspath(os.path.expanduser(left))
    out: list[str] = []
    if os.path.isfile(left):
        return [left]
    if not os.path.isdir(left):
        return []
    if recursive:
        for dp, _, fn in os.walk(left):
            for f in fn:
                if _is_image_path(f):
                    out.append(os.path.join(dp, f))
    else:
        for f in sorted(os.listdir(left)):
            if not _is_image_path(f):
                continue
            p = os.path.join(left, f)
            if os.path.isfile(p):
                out.append(p)
    return sorted(out)


def resolve_stereo_pairs(
        left: str,
        right: str,
        recursive: bool,
) -> list[tuple[str, str]]:
    left = os.path.abspath(os.path.expanduser(left))
    right = os.path.abspath(os.path.expanduser(right))
    if os.path.isfile(left) and os.path.isfile(right):
        return [(left, right)]
    if os.path.isfile(left) or os.path.isfile(right):
        raise ValueError('左右须同为文件或同为目录')
    pairs: list[tuple[str, str]] = []
    if recursive:
        for dp, _, fn in os.walk(left):
            for f in fn:
                if not _is_image_path(f):
                    continue
                lp = os.path.join(dp, f)
                rel = os.path.relpath(lp, left)
                rp = os.path.join(right, rel)
                if os.path.isfile(rp):
                    pairs.append((lp, rp))
    else:
        for f in sorted(os.listdir(left)):
            if not _is_image_path(f):
                continue
            lp = os.path.join(left, f)
            if not os.path.isfile(lp):
                continue
            rp = os.path.join(right, f)
            if os.path.isfile(rp):
                pairs.append((lp, rp))
    return sorted(pairs)


def _resize_bgr_hwc(
        img_bgr: np.ndarray,
        h: int,
        w: int,
        resize_backend: str,
) -> np.ndarray:
    if resize_backend == 'mmcv':
        import mmcv

        return mmcv.imresize(img_bgr, (w, h), interpolation='bilinear', backend='cv2')
    return cv2.resize(img_bgr, (w, h), interpolation=cv2.INTER_LINEAR)


def stack_sixch_bgr_chw(
        left_bgr: np.ndarray,
        right_bgr: np.ndarray,
        h: int,
        w: int,
        resize_backend: str = 'cv2',
) -> np.ndarray:
    """双图 BGR 缩放到 ``h×w``，拼成 ``(6, H, W)`` float32，供 ``SegDataPreProcessor`` 使用。"""
    L = _resize_bgr_hwc(left_bgr, h, w, resize_backend)
    R = _resize_bgr_hwc(right_bgr, h, w, resize_backend)
    hwc6 = np.concatenate([L, R], axis=2)
    return hwc6.transpose(2, 0, 1).astype(np.float32)


def _pred_to_color_bgr(pred: np.ndarray, pal_rgb: np.ndarray) -> np.ndarray:
    k = pal_rgb.shape[0]
    safe = np.clip(pred, 0, k - 1)
    rgb = pal_rgb[safe.reshape(-1)].reshape(*pred.shape, 3)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def make_seg_overlay_bgr(
        img_bgr: np.ndarray,
        pred_hw: np.ndarray,
        pal_rgb: np.ndarray,
        alpha: float,
) -> np.ndarray:
    oh, ow = img_bgr.shape[:2]
    pred_u8 = pred_hw.astype(np.uint8)
    if pred_u8.shape[0] != oh or pred_u8.shape[1] != ow:
        pred_u8 = cv2.resize(pred_u8, (ow, oh), interpolation=cv2.INTER_NEAREST)
    color = _pred_to_color_bgr(pred_u8, pal_rgb)
    a = float(np.clip(alpha, 0.0, 1.0))
    return cv2.addWeighted(img_bgr, 1.0 - a, color, a, 0)


def _hstack_panels(panels: list, sep_w: int) -> np.ndarray:
    oh = panels[0].shape[0]
    if sep_w <= 0:
        return np.hstack(panels)
    sep = np.full((oh, sep_w, 3), 255, dtype=np.uint8)
    out = panels[0]
    for p in panels[1:]:
        out = np.hstack([out, sep, p])
    return out


def _disp_to_color_bgr(disp_2d: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    d = disp_2d.astype(np.float32)
    if d.ndim > 2:
        d = np.squeeze(d)
    vmin = float(args.disp_vis_vmin)
    vmax = float(args.disp_vis_vmax)
    if args.disp_vis_mode == 'adaptive':
        vmin = float(np.min(d))
        vmax = float(np.max(d))
        if vmax <= vmin:
            vmax = vmin + 1e-6
    clipped = np.clip(d, vmin, vmax)
    norm = ((clipped - vmin) / (vmax - vmin + 1e-12) * 255.0).astype(np.uint8)
    colored = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    mb = getattr(args, 'disp_vis_mask_below', None)
    if mb is not None:
        gray = np.full_like(colored, 192)
        mask = d < float(mb)
        colored[mask] = gray[mask]
    return colored


def _pred_sem_seg_to_numpy(result: SegDataSample) -> np.ndarray:
    t = result.pred_sem_seg.data
    if hasattr(t, 'detach'):
        t = t.detach().cpu().numpy()
    else:
        t = np.asarray(t)
    t = np.squeeze(t)
    return t.astype(np.int64)


def save_pth_prediction_vis(
        result: SegDataSample,
        args: argparse.Namespace,
        pal_rgb: np.ndarray,
        base_path: str,
        left_bgr: np.ndarray,
        seg_only_vis: bool = False,
) -> None:
    pred_hw = _pred_sem_seg_to_numpy(result)

    oh, ow = left_bgr.shape[:2]
    if pred_hw.shape[0] != oh or pred_hw.shape[1] != ow:
        pred_vis = cv2.resize(
            pred_hw.astype(np.uint8), (ow, oh), interpolation=cv2.INTER_NEAREST)
    else:
        pred_vis = pred_hw.astype(np.uint8)

    overlay = make_seg_overlay_bgr(left_bgr, pred_vis, pal_rgb, args.overlay_alpha)

    disp_2d: Optional[np.ndarray] = None
    pd = result.get('pred_disp', None)
    if pd is not None:
        d = pd.data
        if hasattr(d, 'detach'):
            d = d.detach().cpu().numpy()
        else:
            d = np.asarray(d)
        disp_2d = np.squeeze(d).astype(np.float32)

    if seg_only_vis or disp_2d is None:
        panels = [left_bgr, overlay]
    else:
        disp_color = _disp_to_color_bgr(disp_2d, args)
        panels = [left_bgr, overlay, disp_color]

    strip = _hstack_panels(panels, int(args.vis_sep))
    cv2.imwrite(base_path + str(args.vis_suffix), strip)

    if getattr(args, 'save_disp_npy', False) and disp_2d is not None:
        np.save(base_path + '_disp.npy', disp_2d)


def _resolve_checkpoint(config: str, ckpt: str) -> str:
    ckpt = ckpt.strip()
    if '/' not in ckpt:
        cfg_rel = config.split('configs/')[-1]
        ckpt = os.path.join('work_dirs', os.path.splitext(cfg_rel)[0], ckpt)
        print(f'checkpoint 已扩展为: {ckpt}')
        time.sleep(0.05)
    if not ckpt.endswith('.pth'):
        with open(ckpt, 'r', encoding='utf-8') as f:
            ckpt = f.read().strip()
    if os.path.basename(ckpt) == 'last_checkpoint':
        with open(ckpt, 'r', encoding='utf-8') as f:
            ckpt = f.read().strip()
    return ckpt


def infer_one_pair(
        args,
        model,
        left_path: str,
        right_path: Optional[str],
        h: int,
        w: int,
        pal_rgb: np.ndarray):
    left_bgr = cv2.imread(left_path)
    if left_bgr is None:
        raise FileNotFoundError(f'读取失败: {left_path}')
    if right_path is None:
        right_bgr = left_bgr
    else:
        right_bgr = cv2.imread(right_path)
        if right_bgr is None:
            raise FileNotFoundError(f'读取失败: {right_path}')

    chw = stack_sixch_bgr_chw(
        left_bgr,
        right_bgr,
        h,
        w,
        resize_backend=args.resize_backend,
    )
    inp = torch.from_numpy(chw).to(args.device)
    _, hh, ww = chw.shape
    oh, ow = left_bgr.shape[:2]
    w_scale = ww / float(ow) if ow else 1.0
    h_scale = hh / float(oh) if oh else 1.0
    sample = SegDataSample()
    sample.set_metainfo(
        dict(
            img_path=left_path,
            ori_shape=(oh, ow),
            img_shape=(hh, ww),
            pad_shape=(hh, ww),
            padding_size=[0, 0, 0, 0],
            scale_factor=(w_scale, h_scale),
            flip=False,
        ))
    data = dict(inputs=[inp], data_samples=[sample])

    with torch.inference_mode():
        out = model.test_step(data)
    if not out:
        raise RuntimeError('test_step 无输出')
    result = out[0]

    stem = os.path.splitext(os.path.basename(left_path))[0]
    base_path = os.path.join(args.out_dir, stem)
    seg_only_vis = bool(getattr(args, 'left_only', False))
    save_pth_prediction_vis(
        result,
        args,
        pal_rgb,
        base_path,
        left_bgr,
        seg_only_vis=seg_only_vis)


def main():
    parser = argparse.ArgumentParser(
        description='SegStereo .pth：经 DataPreprocessor 与训练 val 一致')
    parser.add_argument(
        'config',
        type=str,
        help='训练配置 .py（须含 SegStereo + data_preprocessor）')
    parser.add_argument(
        'checkpoint',
        type=str,
        help='权重 .pth 或 work_dirs 下短名 / last_checkpoint 文件')
    parser.add_argument('left', type=str, help='左图路径或目录')
    parser.add_argument(
        'right',
        type=str,
        nargs='?',
        default=None,
        help='右图路径或目录；省略时需 --left-only')
    parser.add_argument(
        '--left-only',
        action='store_true',
        help='仅左图：右输入为左图复制（无视差）')
    parser.add_argument(
        '--out-dir',
        type=str,
        default='.',
        help='输出目录')
    parser.add_argument(
        '--recursive',
        action='store_true',
        help='目录模式下递归左目录')
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help='如 cuda:0 或 cpu')
    parser.add_argument('--input-h', type=int, default=None)
    parser.add_argument('--input-w', type=int, default=None)
    parser.add_argument('--palette', type=str, default=None)
    parser.add_argument('--overlay-alpha', type=float, default=0.5)
    parser.add_argument('--vis-sep', type=int, default=0)
    parser.add_argument('--vis-suffix', type=str, default='_vis.png')
    parser.add_argument('--save-disp-npy', action='store_true')
    parser.add_argument(
        '--disp-vis-mode',
        type=str,
        choices=['fixed', 'adaptive'],
        default='fixed')
    parser.add_argument('--disp-vis-vmin', type=float, default=0.0)
    parser.add_argument('--disp-vis-vmax', type=float, default=192.0)
    parser.add_argument('--disp-vis-mask-below', type=float, default=None)
    parser.add_argument('--seg-num-classes', type=int, default=None)
    parser.add_argument(
        '--resize-backend',
        type=str,
        choices=['cv2', 'mmcv'],
        default=None,
        help='默认 mmcv（与训练 ResizeStereoImages 一致），无 mmcv 时自动 cv2')
    args = parser.parse_args()
    if args.resize_backend is None:
        args.resize_backend = infer_default_resize_backend()

    if args.left_only:
        if args.right is not None:
            print('提示: --left-only 下忽略 right。', file=sys.stderr)
        left_paths = resolve_left_only_paths(args.left, args.recursive)
        if not left_paths:
            print('未找到左图。', file=sys.stderr)
            sys.exit(1)
        pairs = [(lp, None) for lp in left_paths]
    else:
        if args.right is None:
            print('请提供 right 或使用 --left-only。', file=sys.stderr)
            sys.exit(2)
        pairs = resolve_stereo_pairs(args.left, args.right, args.recursive)
        if not pairs:
            print('未找到左右图对。', file=sys.stderr)
            sys.exit(1)

    h = args.input_h
    w = args.input_w
    if h is None or w is None:
        ih, iw = input_hw_from_config(args.config)
        h = h if h is not None else ih
        w = w if w is not None else iw
    print(f'输入分辨率 H×W = {h}×{w}（与 config data_preprocessor.size 或 --input-h/w 一致）')

    ckpt = _resolve_checkpoint(args.config, args.checkpoint)
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(f'checkpoint 不存在: {ckpt}')

    if args.seg_num_classes is not None:
        num_classes = int(args.seg_num_classes)
    else:
        try:
            num_classes = num_classes_from_decode_config(args.config)
        except Exception:
            num_classes = 4
    print(f'分割 num_classes={num_classes}')

    model = init_model(args.config, ckpt, device=args.device)
    model.eval()

    os.makedirs(args.out_dir, exist_ok=True)
    pal_rgb = _DEFAULT_SEG_PALETTE_RGB.copy()
    if args.palette:
        loaded = np.load(args.palette)
        pal_rgb = loaded.astype(np.uint8) if loaded.dtype != np.uint8 else loaded

    for lp, rp in pairs:
        infer_one_pair(args, model, lp, rp, h, w, pal_rgb)

    print('完成。')


if __name__ == '__main__':
    main()
