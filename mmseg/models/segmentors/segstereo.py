# Copyright (c) OpenMMLab. All rights reserved.
"""SegStereo：语义分割 + 双目视差联合训练（EncoderDecoder + SegStereoDisparityBranch）。"""

from typing import List

import torch
import torch.nn.functional as F
from mmengine.logging import print_log
from mmengine.structures import PixelData
from torch import Tensor

from mmseg.models.utils import resize
from mmseg.registry import MODELS
from mmseg.utils import OptSampleList, SampleList

from .encoder_decoder import EncoderDecoder


def _warp_right_logits_to_left(right_logits: Tensor, disp: Tensor,
                               align_corners: bool) -> tuple:
    """将右视图像素沿水平视差warp到左视图坐标系（SegStereo 论文 L_seg）。

    Args:
        right_logits: (B, C, H, W)
        disp: (B, 1, H, W)，与 ``right_logits`` 同空间尺寸；约定 ``x_left - x_right``。
        align_corners: 与 ``decode_head.align_corners`` 一致。

    Returns:
        tuple: (warped_logits, valid_mask)，``valid_mask`` 为 (B, 1, H, W) bool，
        表示采样落在右图水平范围内。
    """
    b, _, h, w = right_logits.shape
    device = right_logits.device
    dtype = disp.dtype
    y_idx, x_idx = torch.meshgrid(
        torch.arange(h, device=device, dtype=dtype),
        torch.arange(w, device=device, dtype=dtype),
        indexing='ij')
    x_idx = x_idx.unsqueeze(0).expand(b, -1, -1)
    y_idx = y_idx.unsqueeze(0).expand(b, -1, -1)
    disp_2d = disp.squeeze(1)
    x_src = x_idx - disp_2d
    valid = (x_src >= 0) & (x_src <= w - 1)
    valid = valid.unsqueeze(1)
    if align_corners:
        x_norm = 2.0 * x_src / (w - 1) - 1.0
        y_norm = 2.0 * y_idx / (h - 1) - 1.0
    else:
        x_norm = 2.0 * (x_src + 0.5) / w - 1.0
        y_norm = 2.0 * (y_idx + 0.5) / h - 1.0
    grid = torch.stack([x_norm, y_norm], dim=-1)
    warped = F.grid_sample(
        right_logits,
        grid,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=align_corners)
    return warped, valid


@MODELS.register_module()
class SegStereo(EncoderDecoder):
    """联合优化语义分割与视差估计。

    - **输入**：``inputs`` 为 6 通道 ``[左 RGB | 右 RGB]``（与 ``PackSegStereoInputs`` 一致）；
      仅语义分支时也可传入 3 通道左图（此时不计算视差损失）。
    - **标注**：``SegDataSample`` 需含 ``gt_sem_seg``；联合训练时还需 ``gt_disp``（左视差，形状与图一致）。

    Args:
        disparity_branch (dict): ``SegStereoDisparityBranch`` 的配置。
        fuse_semantic_to_disp (bool): 是否把分割 logits 送入视差分支作语义融合。
            为 True 时需 ``disparity_branch['semantic_channels'] == decode_head['num_classes']``。
        loss_disp_weight (float): 视差 Smooth L1 损失权重。
        loss_semantic_warp_weight (float): 论文 **L_seg** 权重：用预测视差将**右**视语义 logits
            warp 到左图后与 ``gt_sem_seg`` 算交叉熵，梯度经可微采样回传至视差。``0`` 表示关闭。
        loss_disp_smooth_weight (float): 视差**一阶平滑**（|∂d/∂x|+|∂d/∂y|）权重；仅在
            与 ``loss_disp`` 相同的 ``gt_disp`` 有效掩码内求平均。``0`` 表示关闭。
        fuse_semantic_warmup_epochs (int): 训练前若干个 epoch **不**做语义融合（仍训视差）；
            从第 ``fuse_semantic_warmup_epochs`` 个 epoch 起启用（与 MMEngine 一致：**0-based**，
            即 ``runner.epoch >= fuse_semantic_warmup_epochs`` 时开启）。默认 ``0`` 表示一开始就融合。
            若使用 :class:`SegStereoSemanticFuseHook` 的平台检测提前打开融合，可与本项 **或** 关系生效。
        fuse_semantic_level_match (bool): IGEV 是否在 ``match`` 上拼接分割 logits（需 ``semantic_channels``）。
        fuse_semantic_level_hg8 (bool): 是否将 ``extract_feat[x][stdc_index_hg8]`` 残差注入 IGEV ``features[1]``（约 1/8）。
        fuse_semantic_level_hg16 (bool): 是否将 ``extract_feat[x][stdc_index_hg16]`` 残差注入 ``features[2]``（约 1/16）。
        stdc_index_hg8 / stdc_index_hg16 (int): STDC 多尺度输出列表下标，默认 ``2`` / ``1``（两路 ARM，128 通道）。
    """

    def __init__(self,
                 disparity_branch: dict,
                 fuse_semantic_to_disp: bool = False,
                 loss_disp_weight: float = 1.0,
                 loss_semantic_warp_weight: float = 0.0,
                 loss_disp_smooth_weight: float = 0.0,
                 fuse_semantic_warmup_epochs: int = 0,
                 fuse_semantic_level_match: bool = True,
                 fuse_semantic_level_hg8: bool = True,
                 fuse_semantic_level_hg16: bool = True,
                 stdc_index_hg8: int = 2,
                 stdc_index_hg16: int = 1,
                 **kwargs):
        super().__init__(**kwargs)
        self.disparity_branch = MODELS.build(disparity_branch)
        self.fuse_semantic_to_disp = fuse_semantic_to_disp
        self.loss_disp_weight = loss_disp_weight
        self.loss_semantic_warp_weight = float(loss_semantic_warp_weight)
        self.loss_disp_smooth_weight = float(loss_disp_smooth_weight)
        self.fuse_semantic_warmup_epochs = max(0, int(fuse_semantic_warmup_epochs))
        self.fuse_semantic_level_match = bool(fuse_semantic_level_match)
        self.fuse_semantic_level_hg8 = bool(fuse_semantic_level_hg8)
        self.fuse_semantic_level_hg16 = bool(fuse_semantic_level_hg16)
        self.stdc_index_hg8 = int(stdc_index_hg8)
        self.stdc_index_hg16 = int(stdc_index_hg16)
        if hasattr(self.disparity_branch, 'fuse_level_match'):
            self.disparity_branch.fuse_level_match = self.fuse_semantic_level_match
            self.disparity_branch.fuse_level_hg8 = self.fuse_semantic_level_hg8
            self.disparity_branch.fuse_level_hg16 = self.fuse_semantic_level_hg16
        if self.fuse_semantic_to_disp and self.fuse_semantic_level_match:
            sc = self.disparity_branch.semantic_channels
            nc = self.decode_head.num_classes
            if sc <= 0 or sc != nc:
                raise ValueError(
                    'fuse_semantic_to_disp 且 fuse_semantic_level_match=True 时需要 '
                    'disparity_branch.semantic_channels 与 num_classes 一致且大于 0，'
                    f'当前 semantic_channels={sc}, num_classes={nc}')
        # 由 SegStereoSemanticFuseHook 在训练中更新；无 Hook 且 warmup>0 时融合会一直关闭
        self._semantic_fuse_active = (
            self.fuse_semantic_to_disp
            and self.fuse_semantic_warmup_epochs == 0)
        if (self.fuse_semantic_to_disp and self.fuse_semantic_warmup_epochs > 0):
            print_log(
                'SegStereo: fuse_semantic_warmup_epochs>0 时请在 config `default_hooks` 中注册 '
                '`SegStereoSemanticFuseHook`（必要时配合 auto_plateau）；否则语义融合不会按 epoch 开启。',
                logger='current',
                level='WARNING')

    def set_semantic_fuse_active(self, active: bool) -> None:
        """由 ``SegStereoSemanticFuseHook`` 调用；``fuse_semantic_to_disp=False`` 时无效。"""
        if not self.fuse_semantic_to_disp:
            self._semantic_fuse_active = False
            return
        self._semantic_fuse_active = bool(active)

    def _use_semantic_fuse(self) -> bool:
        return self.fuse_semantic_to_disp and self._semantic_fuse_active

    def _semantic_tensors_for_disp(self, x: List[Tensor]):
        """视差分支三级语义输入：match logits + STDC ``x[index]`` 两路特征。"""
        if not self._use_semantic_fuse():
            return None, None, None
        left_sem = None
        if self.fuse_semantic_level_match and self.disparity_branch.semantic_channels > 0:
            left_sem = self.decode_head.forward(x)
        fh8 = x[self.stdc_index_hg8] if self.fuse_semantic_level_hg8 else None
        fh16 = x[self.stdc_index_hg16] if self.fuse_semantic_level_hg16 else None
        return left_sem, fh8, fh16

    def forward(
            self,
            inputs: Tensor,
            data_samples: OptSampleList = None,
            mode: str = 'tensor'):
        """导出 ONNX 时使用 ``mode='export_segstereo'``：``inputs`` 为 ``(left, right)`` 或 6 通道拼接张量，返回 ``(seg_logits, disp)``。"""
        if mode == 'export_segstereo':
            if isinstance(inputs, (tuple, list)):
                left, right = inputs[0], inputs[1]
            else:
                if inputs.shape[1] != 6:
                    raise ValueError(
                        'export_segstereo 下单张量输入须为 6 通道 [左|右]')
                left, right = inputs[:, :3], inputs[:, 3:6]
            b = left.shape[0]
            batch_img_metas = [
                dict(
                    ori_shape=left.shape[2:],
                    img_shape=left.shape[2:],
                    pad_shape=left.shape[2:],
                    padding_size=[0, 0, 0, 0])
            ] * b
            seg_logits = self.inference(left, batch_img_metas)
            inputs6 = torch.cat([left, right], dim=1)
            x = self.extract_feat(inputs6)
            left_sem, fh8, fh16 = self._semantic_tensors_for_disp(x)
            disp = self.disparity_branch(
                left,
                right,
                left_semantic=left_sem,
                stdc_feat_hg8=fh8,
                stdc_feat_hg16=fh16)
            return seg_logits, disp
        return super().forward(inputs, data_samples, mode)

    def extract_feat(self, inputs: Tensor) -> List[Tensor]:
        if inputs.shape[1] == 6:
            inputs = inputs[:, :3, :, :]
        return super().extract_feat(inputs)

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        """语义损失 +（6 通道输入时）视差损失。"""
        if inputs.shape[1] != 6:
            return super().loss(inputs, data_samples)

        x = self.extract_feat(inputs)
        losses = dict()
        loss_decode = self._decode_head_forward_train(x, data_samples)
        losses.update(loss_decode)
        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(x, data_samples)
            losses.update(loss_aux)

        left, right = inputs[:, :3], inputs[:, 3:6]
        left_sem, fh8, fh16 = self._semantic_tensors_for_disp(x)
        disp_pred = self.disparity_branch(
            left,
            right,
            left_semantic=left_sem,
            stdc_feat_hg8=fh8,
            stdc_feat_hg16=fh16)
        loss_disp = self._disp_loss(disp_pred, data_samples)
        losses['loss_disp'] = loss_disp * self.loss_disp_weight

        if self.loss_disp_smooth_weight > 0:
            loss_ds = self._disp_smoothness_loss(disp_pred, data_samples)
            losses['loss_disp_smooth'] = loss_ds * self.loss_disp_smooth_weight

        if self.loss_semantic_warp_weight > 0:
            x_r = self.backbone(right)
            right_logits = self.decode_head.forward(x_r)
            loss_sw = self._semantic_warp_loss(disp_pred, right_logits,
                                               data_samples)
            losses['loss_sem_warp'] = loss_sw * self.loss_semantic_warp_weight
        return losses

    def _semantic_warp_loss(self, disp_pred: Tensor, right_logits: Tensor,
                            data_samples: SampleList) -> Tensor:
        """L_seg：warp 右语义 logits，与左图语义真值交叉熵（有效像素内）。"""
        head = self.decode_head
        align = head.align_corners
        if disp_pred.shape[2:] != right_logits.shape[2:]:
            disp_use = resize(
                disp_pred,
                size=right_logits.shape[2:],
                mode='bilinear',
                align_corners=align)
        else:
            disp_use = disp_pred
        warped, valid = _warp_right_logits_to_left(right_logits, disp_use,
                                                   align)
        seg_label = head._stack_batch_gt(data_samples)
        warped = resize(
            warped,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=align)
        valid = resize(
            valid.float(),
            size=seg_label.shape[2:],
            mode='nearest',
            align_corners=None) > 0.5
        seg_label = seg_label.squeeze(1).long()
        ignore = head.ignore_index
        ce_mask = valid.squeeze(1) & (seg_label != ignore)
        if not ce_mask.any():
            return warped.sum() * 0.0
        loss_map = F.cross_entropy(
            warped, seg_label, ignore_index=ignore, reduction='none')
        return (loss_map * ce_mask.float()).sum() / ce_mask.float().sum()

    def _disp_valid_mask(self, data_samples: SampleList) -> tuple:
        """与 ``_disp_loss`` 一致的有效视差掩码 ``(gt, valid)``。"""
        maxd = float(self.disparity_branch.max_disp)
        gts = []
        for s in data_samples:
            if 'gt_disp' not in s:
                raise ValueError(
                    '联合训练 6 通道输入时，每个 SegDataSample 需包含 gt_disp')
            gts.append(s.gt_disp.data)
        gt = torch.stack(gts, dim=0)
        valid = (gt > 0) & (gt < maxd)
        return gt, valid

    def _disp_smoothness_loss(self, pred: Tensor,
                              data_samples: SampleList) -> Tensor:
        """水平 + 垂直一阶差分 L1，仅在相邻两点均 gt 有效处平均。"""
        gt, valid = self._disp_valid_mask(data_samples)
        if pred.shape[2:] != gt.shape[2:]:
            pred = resize(
                pred,
                size=gt.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        if not valid.any():
            return pred.sum() * 0.0
        # 相邻像素都有效才计入，避免掩码边界拉虚假大梯度
        vh = valid[:, :, :, 1:] & valid[:, :, :, :-1]
        vv = valid[:, :, 1:, :] & valid[:, :, :-1, :]
        dx = (pred[:, :, :, 1:] - pred[:, :, :, :-1]).abs()
        dy = (pred[:, :, 1:, :] - pred[:, :, :-1, :]).abs()
        if vh.any():
            lx = (dx * vh.float()).sum() / vh.float().sum()
        else:
            lx = pred.sum() * 0.0
        if vv.any():
            ly = (dy * vv.float()).sum() / vv.float().sum()
        else:
            ly = pred.sum() * 0.0
        return lx + ly

    def _disp_loss(self, pred: Tensor, data_samples: SampleList) -> Tensor:
        """Smooth L1，仅在 gt_disp>0 且 < max_disp 处有效。"""
        gt, valid = self._disp_valid_mask(data_samples)
        if pred.shape[2:] != gt.shape[2:]:
            pred = resize(
                pred,
                size=gt.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        if valid.any():
            return F.smooth_l1_loss(pred[valid], gt[valid], reduction='mean')
        return pred.sum() * 0.0

    def predict(self,
                inputs: Tensor,
                data_samples: OptSampleList = None) -> SampleList:
        """分割预测；若输入为 6 通道，额外写入 ``pred_disp``。"""
        seg_inputs = inputs[:, :3] if inputs.shape[1] == 6 else inputs
        if data_samples is not None:
            batch_img_metas = [
                data_sample.metainfo for data_sample in data_samples
            ]
        else:
            batch_img_metas = [
                dict(
                    ori_shape=seg_inputs.shape[2:],
                    img_shape=seg_inputs.shape[2:],
                    pad_shape=seg_inputs.shape[2:],
                    padding_size=[0, 0, 0, 0])
            ] * seg_inputs.shape[0]

        seg_logits = self.inference(seg_inputs, batch_img_metas)
        out = self.postprocess_result(seg_logits, data_samples)

        if inputs.shape[1] == 6:
            left, right = inputs[:, :3], inputs[:, 3:6]
            x = self.extract_feat(inputs)
            left_sem, fh8, fh16 = self._semantic_tensors_for_disp(x)
            disp = self.disparity_branch(
                left,
                right,
                left_semantic=left_sem,
                stdc_feat_hg8=fh8,
                stdc_feat_hg16=fh16)
            for i, ds in enumerate(out):
                # PixelData 要求 2D 或 3D（如 (1,H,W)），不能为 (1,1,H,W)
                d = disp[i:i + 1].squeeze(0)
                ds.set_data(dict(pred_disp=PixelData(data=d)))

        return out

    def _forward(self,
                 inputs: Tensor,
                 data_samples: OptSampleList = None) -> Tensor:
        """与 EncoderDecoder 一致，仅返回分割 logits。"""
        seg_inputs = inputs[:, :3] if inputs.shape[1] == 6 else inputs
        x = self.extract_feat(inputs)
        return self.decode_head.forward(x)
