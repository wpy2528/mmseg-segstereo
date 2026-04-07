import torch
import torch.nn as nn
from torch import Tensor, nn
import torch.nn.functional as F
from .stereoplus.update import BasicUpdateBlock
from .stereoplus.extractor import Feature
# from stereoplus.geometry import Combined_Geo_Encoding_Volume
from .stereoplus.submodule import *
import logging
import math
import os
from mmseg.registry import MODELS
logger = logging.getLogger(__name__)

__all__ = ["DStereoPlus"]

# ONNX 导出选项；勿在 import 时使用 input()，否则会阻塞训练。需要合并时设环境变量 MMSEG_ONNX_MERGE_DISP_WEIGHT=1
_ONNX_MERGE = os.environ.get('MMSEG_ONNX_MERGE_DISP_WEIGHT', '0').lower() in ('1', 'true', 'y', 'yes')
ONNX_EXPORT_MERGE_DISP_AND_WEIGHT = _ONNX_MERGE
if _ONNX_MERGE:
    logger.debug('ONNX export: merge disp and weight')

# try:
#     autocast = torch.cuda.amp.autocast
# except:
#     class autocast:
#         def __init__(self, enabled):
#             pass
#         def __enter__(self):
#             pass
#         def __exit__(self, *args):
#             pass
class Conv2DInterpolate(nn.Module):
    def __init__(self, inputs_channel=1, scale_factor=2) -> None:
        super().__init__()
        self.conv=nn.Conv2d(
            in_channels=inputs_channel,
            out_channels=inputs_channel * (scale_factor**2),
            kernel_size=3,
            bias=False,
            padding=1,
        )
        self.scale_factor=scale_factor
        self.inputs_channel = inputs_channel
        self.depth2space = torch.nn.PixelShuffle(scale_factor)
        self._init_weights()
        self.freeze()

    def _init_weights(self):
        conv_weight = torch.zeros(
            self.conv.weight.size(),
            dtype=self.conv.weight.dtype,
        )
        num_conv = conv_weight.shape[0]
        for i_N in range(num_conv):
            i_c = i_N // (self.scale_factor**2)
            conv_weight[i_N, i_c, 1, 1] = 1
        self.conv.weight=torch.nn.Parameter(
            conv_weight, requires_grad=False
        )

    def forward(self, x):
        x=self.conv(x)
        out=self.depth2space(x)
        return out

    def freeze(self):
        for param in self.conv.parameters():
            param.requires_grad = False

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super(Conv2DInterpolate, self).train(mode)
        self.freeze()

class UnfoldConv(nn.Module):
    """
    A unfold module using conv.

    Args:
        in_channels: The channels of inputs.
        kernel_size: The kernel_size of unfold.
    """

    def __init__(self, in_channels: int = 1, kernel_size: int = 3):
        super(UnfoldConv, self).__init__()
        self.kernel_size = kernel_size
        self.unflod_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=self.kernel_size ** 2,
            kernel_size=self.kernel_size,
            stride=1,
            bias=False,
        )
        self.pad = nn.ZeroPad2d(padding=(1, 1, 1, 1))
        self.init_weights()

    def init_weights(self) -> None:
        """Initialize the weights of head module."""

        weight_new = torch.zeros(
            self.unflod_conv.weight.size(), dtype=self.unflod_conv.weight.dtype
        )
        for i in range(self.kernel_size ** 2):
            wx = i % self.kernel_size
            wy = i // self.kernel_size

            weight_new[i, :, wy, wx] = 1

        self.unflod_conv.weight = torch.nn.Parameter(weight_new, requires_grad=False)
        self.freeze()

    def forward(self, x: Tensor) -> Tensor:
        """Perform the forward pass of the model."""

        x = self.pad(x)
        x = self.unflod_conv(x)
        return x

    def freeze(self):
        for param in self.unflod_conv.parameters():
            param.requires_grad = False

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super(UnfoldConv, self).train(mode)
        self.freeze()

class hourglass(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(BasicConv(in_channels, in_channels*2, is_3d=False, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=False, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1))
                                    
        self.conv2 = nn.Sequential(BasicConv(in_channels*2, in_channels*4, is_3d=False, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=False, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1))                             

        self.conv3 = nn.Sequential(BasicConv(in_channels*4, in_channels*6, is_3d=False, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*6, in_channels*6, is_3d=False, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1)) 


        self.conv3_up = BasicConv(in_channels*6, in_channels*4, deconv=True, is_3d=False, bn=True,
                                  relu=True, kernel_size=(4, 4), padding=(1, 1), stride=(2, 2))

        self.conv2_up = BasicConv(in_channels*4, in_channels*2, deconv=True, is_3d=False, bn=True,
                                  relu=True, kernel_size=(4, 4), padding=(1, 1), stride=(2, 2))

        self.conv1_up = BasicConv(in_channels*2, out_channels, deconv=True, is_3d=False, bn=False,
                                  relu=False, kernel_size=(4, 4), padding=(1, 1), stride=(2, 2))

        self.agg_0 = nn.Sequential(BasicConv(in_channels*8, in_channels*4, is_3d=False, kernel_size=1, padding=0, stride=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=False, kernel_size=3, padding=1, stride=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=False, kernel_size=3, padding=1, stride=1),)

        self.agg_1 = nn.Sequential(BasicConv(in_channels*4, in_channels*2, is_3d=False, kernel_size=1, padding=0, stride=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=False, kernel_size=3, padding=1, stride=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=False, kernel_size=3, padding=1, stride=1))

        self.feature_att_8 = FeatureAtt(in_channels*2, 128)
        self.feature_att_16 = FeatureAtt(in_channels*4, 192)
        self.feature_att_32 = FeatureAtt(in_channels*6, 160)
        self.feature_att_up_16 = FeatureAtt(in_channels*4, 192)
        self.feature_att_up_8 = FeatureAtt(in_channels*2, 128)

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
    
def build_gwc_volume_onnx(refimg_fea, targetimg_fea, maxdisp):
    tmp_volume = []
    for i in range(maxdisp):
        if i > 0:
            cost = refimg_fea[:, :, :, i:] * targetimg_fea[:, :, :, :-i]
            cost = cost.mean(dim=1, keepdim=True)
            # cost = cost.unsqueeze(1)
            cost = torch.nn.functional.pad(cost, (i, 0, 0, 0), 'constant', 0)
            tmp_volume.append(cost)
        else:
            cost = refimg_fea * targetimg_fea
            cost = cost.mean(dim=1, keepdim=True)
            # cost = cost.unsqueeze(1)
            tmp_volume.append(cost)

    return torch.cat(tmp_volume, dim=1)     # 709 us (0.2% of model)	1254 us (0.4% of model)

class refinement(nn.Module):    
    def __init__(self, gru_iters, hidden_dim):
        super().__init__()
        self.gru_iters = gru_iters
        self.update_block = BasicUpdateBlock(hidden_dim=hidden_dim)

        self.interp_conv = Conv2DInterpolate(inputs_channel=9)
        self.unfold_conv = UnfoldConv(in_channels=1, kernel_size=3)
        self.spx_2_gru = Conv2x(32, 32, deconv=True, concat=True)
        self.spx_gru = nn.Sequential(nn.ConvTranspose2d(2*32, 9, kernel_size=4, stride=2, padding=1),)


    def context_upsample(self, disp_low, up_weights):
        ###
        # cv (b,1,h,w)
        # sp (b,9,4*h,4*w)
        ###
        b, c, h, w = disp_low.shape       
        if (not ONNX_EXPORT_MERGE_DISP_AND_WEIGHT) and torch.onnx.is_in_onnx_export():
            disp_unfold = self.unfold_conv(disp_low)
            return disp_unfold, up_weights
        else:
            disp_unfold = F.unfold(disp_low,3,1,1).reshape(b,-1,h,w)    # disp_low.shape:[1, 1, 56, 56], unfold: [1, 9, 3136] --> [1, 9, 56, 56]    
            disp_unfold = F.interpolate(disp_unfold,(h*4,w*4),mode='nearest').reshape(b,9,h*4,w*4)
            disp = (disp_unfold*up_weights).sum(dim=1,keepdim=False)      
            return disp
    
    def upsample_disp(self, disp, mask_feat_4, stem_2x):

        # with autocast(enabled=self.args.mixed_precision):
        xspx = self.spx_2_gru(mask_feat_4, stem_2x)
        spx_pred = self.spx_gru(xspx)
        spx_pred = F.softmax(spx_pred, 1)
        up_disp = self.context_upsample(disp*4., spx_pred)
        return up_disp
    
    def forward(self, disp, net, context, geo_encoding_volume, stem_2x):
        disp_preds = []
        # GRUs iterations to update disparity
        for itr in range(self.gru_iters):
            disp = disp.detach()
            # geo_feat = geo_fn(disp, coords)
            # with autocast(enabled=self.args.mixed_precision):
            net, mask_feat_4, delta_disp = self.update_block(net, context, geo_encoding_volume, disp)
            disp = disp + delta_disp
            # if not self.training and itr < (self.gru_iters-1):
            #     continue

            # upsample predictions
            disp_up = self.upsample_disp(disp, mask_feat_4, stem_2x)
            disp_preds.append(disp_up)
        return disp_preds, disp
    
class prepare_forrefinement(nn.Module):    
    def __init__(self, hidden_dim, context_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.hnet = nn.Sequential(BasicConv(64, self.hidden_dim, kernel_size=3, stride=1, padding=1),
                                     nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, 1, 1, bias=False))

        self.cnet = BasicConv(64, context_dim, kernel_size=3, stride=1, padding=1)
        self.context_zqr_conv = nn.Conv2d(context_dim, context_dim*3, 3, padding=3//2)

    def forward(self, features_left):
        hidden = self.hnet(features_left[0])
        net = torch.tanh(hidden)
        context = self.cnet(features_left[0])
        context = list(self.context_zqr_conv(context).split(split_size=self.hidden_dim, dim=1))
        return net, context
    
class get_initdisp(nn.Module):    
    def __init__(self, maxdisp):
        super().__init__()
        self.maxdisp = maxdisp
        self.classifier = BasicConv(maxdisp//4, maxdisp//4, kernel_size=3, stride=1, padding=1)

    def forward(self, geo_encoding_volume):
        # Init disp from geometry encoding volume
        prob = F.softmax(self.classifier(geo_encoding_volume), dim=1)
        init_disp = disparity_regression(prob, self.maxdisp//4, 1)
        return init_disp
    
class get_costvolum(nn.Module):
    def __init__(self, maxdisp):
        super().__init__()
        self.maxdisp = maxdisp

    def forward(self, match_left, match_right):
        if torch.onnx.is_in_onnx_export():
            gwc_volume = build_gwc_volume_onnx(match_left, match_right, self.maxdisp//4)
        else:
            gwc_volume = build_gwc_volume(match_left, match_right, self.maxdisp//4, 1)
        return gwc_volume

class before_costvolum(nn.Module):    
    def __init__(self,):
        super().__init__()
        # self.stem_4 = nn.Sequential(
        #     BasicConv(16, 24, kernel_size=3, stride=2, padding=1),
        #     nn.Conv2d(24, 24, 3, 1, 1, bias=False),
        #     nn.InstanceNorm2d(24), nn.ReLU()
        #     )
        self.desc = nn.Conv2d(48, 48, kernel_size=1, padding=0, stride=1)
        # self.stem_2 = nn.Sequential(
        #     BasicConv(3, 16, kernel_size=3, stride=2, padding=1),
        #     nn.Conv2d(16, 16, 3, 1, 1, bias=False),
        #     nn.InstanceNorm2d(16), nn.ReLU()
        #     )
        self.conv = BasicConv(64, 48, kernel_size=3, padding=1, stride=1)

    def forward(self, features_left, features_right):
        # if not torch.onnx.is_in_onnx_export():
        #     B, _, _, _ = data['img'].shape
        #     B = B // 2
        #     stem_2x = self.stem_2(data['img'][:B, ...])
        #     stem_2y = self.stem_2(data['img'][B:, ...])
        # else:
        # stem_2x = self.stem_2(data['infra1'])
        # stem_2y = self.stem_2(data['infra2'])
        # stem_4x = self.stem_4(stem_2x)
        # stem_4y = self.stem_4(stem_2y)
        # features_left[0] = torch.cat((features_left[0], stem_4x), 1)
        # features_right[0] = torch.cat((features_right[0], stem_4y), 1)

        match_left = self.desc(self.conv(features_left[0]))
        match_right = self.desc(self.conv(features_right[0]))

        return match_left, match_right

@MODELS.register_module()   
class DStereoPlus(nn.Module):
    def __init__(self, backbone, gru_iters, maxdisp):
        super().__init__()
        self.backbone = MODELS.build(backbone)
        self.maxdisp = maxdisp
        self.hidden_dim=48
        context_dim = self.hidden_dim

        self.spx = nn.Sequential(nn.ConvTranspose2d(2*32, 9, kernel_size=4, stride=2, padding=1),)
        self.spx_2 = Conv2x(24, 32, deconv=True)
        self.spx_4 = nn.Sequential(
            BasicConv(64, 24, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(24, 24, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(24), nn.ReLU()
            )
        self.feature = Feature()
        self.cost_agg = hourglass(self.maxdisp // 4, self.maxdisp // 4)
        logger.info("###################### init DStereoPlus done ######################")
        self.get_costvolum = get_costvolum(self.maxdisp)
        self.before_costvolum = before_costvolum()
        self.get_initdisp = get_initdisp(self.maxdisp)
        self.prepare_forrefinement = prepare_forrefinement(hidden_dim=32, context_dim=32)
        self.refinement = refinement(gru_iters, hidden_dim=32)

    def forward(self, data):
        """ Estimate disparity between pair of frames """

        # image1 = (2 * (image1 / 255.0) - 1.0).contiguous()
        # image2 = (2 * (image2 / 255.0) - 1.0).contiguous()
        # with autocast(enabled=self.args.mixed_precision):
        if not torch.onnx.is_in_onnx_export():
            if False:
                features_list = self.backbone(data['img'])
                B, _, _, _ = data['img'].shape
                B = B // 2
                features_left = [i[:B, ...] for i in features_list]
                features_right = [i[B:, ...] for i in features_list]
            else:
                image1, image2 = data.split(3, dim=1)
                features_left = self.backbone(image1)
                features_right = self.backbone(image2)
        else:
            image1, image2 = data
            features_left = self.backbone(image1)
            features_right = self.backbone(image2)
        stem_2x = features_left[0]
        features_left = self.feature(*features_left)
        features_right = self.feature(*features_right)

        match_left, match_right = self.before_costvolum(features_left, features_right)
        gwc_volume = self.get_costvolum(match_left, match_right)
        geo_encoding_volume = self.cost_agg(gwc_volume, features_left)
        init_disp = self.get_initdisp(geo_encoding_volume)

        # if self.training:
        xspx = self.spx_4(features_left[0])
        xspx = self.spx_2(xspx, stem_2x)
        spx_pred = self.spx(xspx)
        spx_pred = F.softmax(spx_pred, 1)

        net, context = self.prepare_forrefinement(features_left)
        disp = init_disp
        disp_preds, disp_4x = self.refinement(disp, net, context, geo_encoding_volume, stem_2x)

        if not self.training:
            if ONNX_EXPORT_MERGE_DISP_AND_WEIGHT:
                return disp_preds[-1].unsqueeze(1)
            else:
                return disp_preds[-1]

        init_disp_pred = self.refinement.context_upsample(init_disp*4., spx_pred.float())
        disp_preds = [init_disp_pred] + disp_preds
        disp_preds = [e.unsqueeze(1) for e in disp_preds]
        disp_preds = torch.cat(disp_preds, dim=1)
        return disp_preds
    
    def sequence_loss(self, agg_pred, iter_preds, disp_gt, loss_gamma=0.9):
        """ Loss function defined over sequence of flow predictions """

        n_predictions = len(iter_preds)
        assert n_predictions >= 1
        disp_loss = []
        # mag = torch.sum(disp_gt**2, dim=1).sqrt()
        valid = ((disp_gt > 0.) & (disp_gt < self.maxdisp))
        assert valid.shape == disp_gt.shape, [valid.shape, disp_gt.shape]
        assert not torch.isinf(disp_gt[valid.bool()]).any()

        disp_loss.append(1.0 * F.smooth_l1_loss(agg_pred[valid.bool()], disp_gt[valid.bool()], reduction='mean'))
        for i in range(n_predictions):
            adjusted_loss_gamma = loss_gamma**(15/(n_predictions - 1))
            i_weight = adjusted_loss_gamma**(n_predictions - i - 1)
            i_loss = (iter_preds[i] - disp_gt).abs()
            assert i_loss.shape == valid.shape, [i_loss.shape, valid.shape, disp_gt.shape, iter_preds[i].shape]
            disp_loss.append(i_weight * i_loss[valid.bool()].mean())

        return disp_loss