import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from .update import BasicUpdateBlock, _autocast_cuda_disabled
from .extractor import Feature
from .geometry import Geo_Encoding_Volume
from .submodule import *
from .submodule import build_gwc_volume_no_scatternd
from mmseg.registry import MODELS

try:
    _amp_autocast = torch.amp.autocast  # PyTorch 2.x; avoids cuda.amp deprecation warning
except AttributeError:
    _amp_autocast = None

if _amp_autocast is not None:

    def autocast(enabled=True, dtype=torch.float16):
        return _amp_autocast('cuda', enabled=enabled, dtype=dtype)

else:
    try:
        autocast = torch.cuda.amp.autocast
    except AttributeError:

        class autocast:
            """No-op when AMP is unavailable."""

            def __init__(self, enabled=True, dtype=None):
                pass

            def __enter__(self):
                pass

            def __exit__(self, *args):
                pass

class hourglass(nn.Module):
    def __init__(self, in_channels):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(BasicConv(in_channels, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1))
                                    
        self.conv2 = nn.Sequential(BasicConv(in_channels*2, in_channels*4, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1))                             

        self.conv3 = nn.Sequential(BasicConv(in_channels*4, in_channels*6, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*6, in_channels*6, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1)) 


        self.conv3_up = BasicConv(in_channels*6, in_channels*4, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2), block_index=3)

        self.conv2_up = BasicConv(in_channels*4, in_channels*2, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2), block_index=2)

        self.conv1_up = BasicConv(in_channels*2, 8, deconv=True, is_3d=True, bn=False,
                                  relu=False, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2), block_index=1)

        self.agg_0 = nn.Sequential(BasicConv(in_channels*8, in_channels*4, is_3d=True, kernel_size=1, padding=0, stride=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, kernel_size=3, padding=1, stride=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, kernel_size=3, padding=1, stride=1),)

        self.agg_1 = nn.Sequential(BasicConv(in_channels*4, in_channels*2, is_3d=True, kernel_size=1, padding=0, stride=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, kernel_size=3, padding=1, stride=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, kernel_size=3, padding=1, stride=1))

        self.feature_att_8 = FeatureAtt(in_channels*2, 64)
        self.feature_att_16 = FeatureAtt(in_channels*4, 192)
        self.feature_att_32 = FeatureAtt(in_channels*6, 160)
        self.feature_att_up_16 = FeatureAtt(in_channels*4, 192)
        self.feature_att_up_8 = FeatureAtt(in_channels*2, 64)

    def forward(self, x, features):

        conv1 = self.conv1(x)
        conv1 = self.feature_att_8(conv1, features[1])

        conv2 = self.conv2(conv1)
        conv2 = self.feature_att_16(conv2, features[2])

        conv3 = self.conv3(conv2)
        conv3 = self.feature_att_32(conv3, features[3])

        conv3_up = self.conv3_up(conv3)
        conv2 = torch.cat((conv3_up, conv2), dim=1)
        conv2 = self.agg_0(conv2)
        conv2 = self.feature_att_up_16(conv2, features[2])

        conv2_up = self.conv2_up(conv2)
        conv1 = torch.cat((conv2_up, conv1), dim=1)
        conv1 = self.agg_1(conv1)
        conv1 = self.feature_att_up_8(conv1, features[1])

        conv = self.conv1_up(conv1)

        return conv


@MODELS.register_module()
class IGEVStereo(nn.Module):
    def __init__(self):
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--name', default='rt-igev-stereo', help="name your experiment")
        parser.add_argument('--restore_ckpt', default=None, help='load the weights from a specific checkpoint')
        parser.add_argument('--logdir', default='./checkpoints_rt', help='the directory to save logs and checkpoints')
        parser.add_argument('--mixed_precision', default=True, action='store_true', help='use mixed precision')
        parser.add_argument('--precision_dtype', default='float16', choices=['float16', 'bfloat16', 'float32'], help='Choose precision type: float16 or bfloat16 or float32')


        # Training parameters
        parser.add_argument('--batch_size', type=int, default=8, help="batch size used during training.")
        parser.add_argument('--train_datasets', default='sceneflow', choices=['sceneflow', 'kitti'], help="training datasets.")
        parser.add_argument('--lr', type=float, default=0.0002, help="max learning rate.")
        parser.add_argument('--num_steps', type=int, default=200000, help="length of training schedule.")
        parser.add_argument('--image_size', type=int, nargs='+', default=[320, 768], help="size of the random image crops used during training.")
        parser.add_argument(
            '--gru_iters',
            type=int,
            default=4,
            help='number of GRU refinement updates per forward (0 = init disp + spx upsample only)')
        parser.add_argument('--wdecay', type=float, default=.00001, help="Weight decay in optimizer.")

        # Validation parameters
        parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during validation forward pass')

        # Architecure choices
        parser.add_argument('--corr_levels', type=int, default=2, help="number of levels in the correlation pyramid")
        parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
        parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
        parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
        parser.add_argument('--n_gru_layers', type=int, default=1, help="number of hidden GRU levels")
        parser.add_argument('--hidden_dim', nargs='+', type=int, default=96, help="hidden state and context dimensions")
        parser.add_argument('--max_disp', type=int, default=192, help="max disp of geometry encoding volume")

        # Data augmentation
        parser.add_argument('--img_gamma', type=float, nargs='+', default=None, help="gamma range")
        parser.add_argument('--saturation_range', type=float, nargs='+', default=[0, 1.4], help='color saturation')
        parser.add_argument('--do_flip', default=False, choices=['h', 'v'], help='flip the images horizontally or vertically')
        parser.add_argument('--spatial_scale', type=float, nargs='+', default=[-0.4, 0.8], help='re-scale the images randomly')
        parser.add_argument('--noyjitter', action='store_true', help='don\'t simulate imperfect rectification')
        args = parser.parse_args([])


        super().__init__()
        self.args = args       
        context_dim = args.hidden_dim
        self.update_block = BasicUpdateBlock(self.args, hidden_dim=args.hidden_dim)
        self.hnet = nn.Sequential(BasicConv(96, args.hidden_dim, kernel_size=3, stride=1, padding=1),
                                     nn.Conv2d(args.hidden_dim, args.hidden_dim, 3, 1, 1, bias=False))

        self.cnet = BasicConv(96, context_dim, kernel_size=3, stride=1, padding=1)
        self.context_zqr_conv = nn.Conv2d(context_dim, context_dim*3, 3, padding=3//2)
        self.feature = Feature()

        self.stem_2 = nn.Sequential(
            BasicConv_IN(3, 32, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(32), nn.ReLU()
            )
        self.stem_4 = nn.Sequential(
            BasicConv_IN(32, 48, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(48, 48, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(48), nn.ReLU()
            )

        self.spx = nn.Sequential(nn.ConvTranspose2d(2*32, 9, kernel_size=4, stride=2, padding=1),)
        self.spx_2 = Conv2x_IN(24, 32, True)
        self.spx_4 = nn.Sequential(
            BasicConv_IN(96, 24, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(24, 24, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(24), nn.ReLU()
            )

        self.spx_2_gru = Conv2x(32, 32, True)
        self.spx_gru = nn.Sequential(nn.ConvTranspose2d(2*32, 9, kernel_size=4, stride=2, padding=1),)

        self.conv = BasicConv_IN(96, 96, kernel_size=3, padding=1, stride=1)
        self.desc = nn.Conv2d(96, 96, kernel_size=1, padding=0, stride=1)

        self.cost_agg = hourglass(8)
        self.classifier = nn.Conv3d(8, 1, 3, 1, 1, bias=False)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def upsample_disp(self, disp, mask_feat_4, stem_2x):
        """1/4 分辨率视差 + GRU 分支的 mask 特征 → 全分辨率（与 init 路径的 spx 并行一套 head）。"""
        with autocast(enabled=self.args.mixed_precision, dtype=getattr(torch, self.args.precision_dtype, torch.float16)):
            xspx = self.spx_2_gru(mask_feat_4, stem_2x)
            spx_pred = self.spx_gru(xspx)
            spx_pred = F.softmax(spx_pred, dim=1)
            up_disp = context_upsample(disp * 4.0, spx_pred.float())
        return up_disp

    
    def forward(self, inputs):
        image1, image2 = inputs.split(3, dim=1)
        return self.forward_inner((image1, image2))

    def forward_inner(self,
                      inputs,
                      iters=None,
                      flow_init=None,
                      test_mode=False,
                      left_semantic=None,
                      sem_proj=None,
                      align_corners=False,
                      fuse_level_match: bool = True,
                      fuse_level_hg8: bool = True,
                      fuse_level_hg16: bool = True,
                      stdc_feat_hg8=None,
                      stdc_feat_hg16=None,
                      sem_add_hg8=None,
                      sem_add_hg16=None):
        """Estimate disparity between pair of frames.

        Args:
            iters: GRU 更新步数；``None`` 时用 ``self.args.gru_iters``。为 0 时仅 init + spx 上采样。
            left_semantic: 可选，左视语义 logits ``(B, C, H, W)``，空间尺寸建议已与输入左图对齐
                （含 pad）。与 ``sem_proj`` 同时给出时，在 ``match`` 特征上拼接投影后的语义
                （与 SegStereo 代价体分支融合方式一致；通道总数须能被 GWC 的 ``num_groups`` 整除）。
            sem_proj: 将 ``left_semantic`` 投影到 ``fused_channels`` 的 ``nn.Module``（如 1x1 Conv+BN）。
            align_corners: 将语义双线性缩放到 ``match`` 分辨率时的 ``align_corners``。
            fuse_level_match: 是否在 ``match`` 上拼接分割 logits。
            fuse_level_hg8 / fuse_level_hg16: 是否将 STDC 特征以残差形式注入 ``features_left[1]`` / ``[2]``。
            stdc_feat_hg8 / stdc_feat_hg16: STDC 多尺度特征，空间尺寸任意，内部双线性对齐到 IGEV 金字塔。
            sem_add_hg8 / sem_add_hg16: ``Conv+BN``，将 STDC 通道数映射到 IGEV ``features[1]``（64）/ ``[2]``（192）。
        """
        image1, image2 = inputs
        image1 = image1.float()
        image2 = image2.float()
        assert image1.shape[2] % 32 == 0
        assert image1.shape[3] % 32 == 0
        if iters is None:
            iters = int(self.args.gru_iters)
        hd = self.args.hidden_dim
        if isinstance(hd, (list, tuple)):
            hd = int(hd[0])

        # SegStereo 在 MMEngine AmpOptimWrapper 下整段 loss 处于全局 autocast；decode_head /
        # backbone 输出常为 FP16。IGEV 在 mixed_precision=False 时在内层使用 autocast(False)，
        # 若仍传入 FP16 张量会与 FP32 权重冲突（HalfTensor vs FloatTensor）。此处统一升到 FP32，
        # 并在下方用 _autocast_cuda_disabled 包住整条 IGEV，避免代价体等在「内层已关 autocast、
        # 外层 AMP 仍开」的缝隙里被半精度化。
        use_fp32_igev = not self.args.mixed_precision
        if left_semantic is not None:
            # (B,1,C,H,W) 等导出/包装产生的多余阶，先压成 (B,C,H,W) 再参与 2D 分支
            while left_semantic.dim() > 4 and left_semantic.shape[1] == 1:
                left_semantic = left_semantic.squeeze(1)
        if use_fp32_igev:
            if left_semantic is not None:
                left_semantic = left_semantic.float()
            if stdc_feat_hg8 is not None:
                stdc_feat_hg8 = stdc_feat_hg8.float()
            if stdc_feat_hg16 is not None:
                stdc_feat_hg16 = stdc_feat_hg16.float()

        igev_fp32_guard = (
            _autocast_cuda_disabled() if use_fp32_igev else contextlib.nullcontext())

        with igev_fp32_guard:
            with autocast(enabled=self.args.mixed_precision, dtype=getattr(torch, self.args.precision_dtype, torch.float16)):
                features_left = self.feature(image1)
                features_right = self.feature(image2)
                stem_2x = self.stem_2(image1)
                stem_4x = self.stem_4(stem_2x)
                stem_2y = self.stem_2(image2)
                stem_4y = self.stem_4(stem_2y)
                features_left[0] = torch.cat((features_left[0], stem_4x), 1)
                features_right[0] = torch.cat((features_right[0], stem_4y), 1)

                # STDC → IGEV 金字塔残差（仅左塔；与 hourglass 内 FeatureAtt 通道一致）
                if fuse_level_hg8 and stdc_feat_hg8 is not None and sem_add_hg8 is not None:
                    s8 = F.interpolate(
                        stdc_feat_hg8,
                        size=features_left[1].shape[2:],
                        mode='bilinear',
                        align_corners=align_corners)
                    features_left[1] = features_left[1] + sem_add_hg8(s8)
                if fuse_level_hg16 and stdc_feat_hg16 is not None and sem_add_hg16 is not None:
                    s16 = F.interpolate(
                        stdc_feat_hg16,
                        size=features_left[2].shape[2:],
                        mode='bilinear',
                        align_corners=align_corners)
                    features_left[2] = features_left[2] + sem_add_hg16(s16)

                match_left = self.desc(self.conv(features_left[0]))
                match_right = self.desc(self.conv(features_right[0]))
                if fuse_level_match and left_semantic is not None:
                    if sem_proj is None:
                        raise ValueError(
                            'forward_inner: left_semantic 需与 sem_proj 同时传入')
                    sem = F.interpolate(
                        left_semantic,
                        size=match_left.shape[2:],
                        mode='bilinear',
                        align_corners=align_corners)
                    sem = sem_proj(sem)
                    # 与在 (B,C,H,W) 上沿通道 cat 等价；中间在 (B,C,H*W) 上拼接，将参与 cat 的张量降为 3 阶
                    b_m, c_ml, h_m, w_m = match_left.shape
                    c_sem = sem.shape[1]
                    hw = h_m * w_m
                    match_left = torch.cat(
                        (match_left.reshape(b_m, c_ml, hw),
                         sem.reshape(b_m, c_sem, hw)),
                        dim=1).view(b_m, c_ml + c_sem, h_m, w_m)
                    match_right = torch.cat(
                        (match_right.reshape(b_m, c_ml, hw),
                         sem.reshape(b_m, c_sem, hw)),
                        dim=1).view(b_m, c_ml + c_sem, h_m, w_m)
                gwc_volume = build_gwc_volume_no_scatternd(
                    match_left, match_right, self.args.max_disp // 4, 8)
                geo_encoding_volume = self.cost_agg(gwc_volume, features_left)

                prob = F.softmax(
                    self.classifier(geo_encoding_volume).squeeze(1), dim=1)
                init_disp = disparity_regression(prob, self.args.max_disp // 4,
                                                 1)

                del prob, gwc_volume

                spx_pred = None
                if iters == 0:
                    xspx = self.spx_4(features_left[0])
                    xspx = self.spx_2(xspx, stem_2x)
                    spx_pred = F.softmax(self.spx(xspx), dim=1)

                hidden = self.hnet(features_left[0])
                net = torch.tanh(hidden)
                context = self.cnet(features_left[0])
                context = list(self.context_zqr_conv(context).split(hd, dim=1))

            # mixed_precision=True 且外层为训练 AMP 时：离开内层 autocast 后仍会回到全局
            # autocast，须单独在 FP32 下构造 Geo_Encoding_Volume 与采样。
            if use_fp32_igev:
                geo_fn = Geo_Encoding_Volume(
                    geo_encoding_volume.float(),
                    radius=self.args.corr_radius,
                    num_levels=self.args.corr_levels)
            else:
                with _autocast_cuda_disabled():
                    geo_fn = Geo_Encoding_Volume(
                        geo_encoding_volume.float(),
                        radius=self.args.corr_radius,
                        num_levels=self.args.corr_levels)
            disp = init_disp
            disp_up = None

            if iters > 0:
                for itr in range(iters):
                    disp = disp.detach()
                    if use_fp32_igev:
                        geo_feat = geo_fn(disp)
                    else:
                        with _autocast_cuda_disabled():
                            geo_feat = geo_fn(disp)
                    with autocast(
                            enabled=self.args.mixed_precision,
                            dtype=getattr(torch, self.args.precision_dtype,
                                          torch.float16)):
                        net, mask_feat_4, delta_disp = self.update_block(
                            net, context, geo_feat, disp)
                    disp = disp + delta_disp
                    if test_mode and itr < iters - 1:
                        continue
                    disp_up = self.upsample_disp(disp, mask_feat_4, stem_2x)
                if disp_up is None:
                    raise RuntimeError('gru_iters>0 but disp_up was not computed')
                return disp_up

            assert spx_pred is not None
            return context_upsample(init_disp * 4.0, spx_pred.float())
