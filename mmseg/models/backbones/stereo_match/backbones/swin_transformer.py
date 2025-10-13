# Copyright (c) Horizon Robotics. All rights reserved.

from copy import deepcopy
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from horizon_plugin_pytorch.quantization import QuantStub
from torch.quantization import DeQuantStub

from .models.base_modules.basic_swin_module import (
    BasicLayer,
    PatchEmbedding,
    PatchMerging,
)
from .models.weight_init import trunc_normal
from .registry import OBJECT_REGISTRY
from .utils.checkpoint import load_state_dict

__all__ = ["SwinTransformer", "update_state_dict_from_official"]


def update_state_dict_from_official(state_dict):
    """Update state_dict in official swin transformer.

    Pre-trained weights can be found in
    https://github.com/microsoft/Swin-Transformer.
    """
    new_state_dict = deepcopy(state_dict)
    for k, v in state_dict.items():
        if k.startswith("patch_embed"):
            del new_state_dict[k]
            new_k = k.replace("patch_embed", "patch_embedding")
            new_state_dict[new_k] = v
        elif "mlp" in k:
            del new_state_dict[k]
            if "fc1" in k:
                new_k = k.replace("fc1", "0")
            else:
                new_k = k.replace("fc2", "2")
            new_state_dict[new_k] = v
        elif k.startswith("head"):
            del new_state_dict[k]
            new_k = k.replace("head", "output")
            new_state_dict[new_k] = v
    return new_state_dict


@OBJECT_REGISTRY.register
class SwinTransformer(nn.Module):
    """A module of swin transformer.

    Args:
        depth_list: Depths of each Swin Transformer stage.
            for swin_T, the numbers could be [2, 2, 6, 2].
            for swin_S, swin_B, or swin_L, the numbers could be [2, 2, 18, 2].
        num_heads: Number of attention head of each stage.
            for swin_T or swin_S, the numbers could be [3, 6, 12, 24].
            for swin_B, the numbers could be [4, 8, 16, 32].
            for swin_L, the numbers could be [6, 12, 24, 48].
        num_classes: Num classes of output layer.
        patch_size: Patch size. Default: 4.
        in_channels: Number of input image channels. Default: 3.
        embedding_dims: Number of linear projection output channels.
            for swin_T or swin_S, the numbers could be 96.
            for swin_B, the number could be 128.
            for swin_L, the number could be 192.
        window_size: Window size. Default: 7.
        mlp_ratio: Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias: Whether to add a learnable bias to query, key, value.
            Default: True.
        qk_scale: Override default qk scale of head_dim ** -0.5 if set.
        dropout_ratio: Dropout rate. Default: 0.
        attention_dropout_ratio: Attention dropout rate. Default: 0.
        drop_path_ratio: Stochastic depth rate. Default: 0.0.
        patch_norm: Whether to add normalization after patch embedding.
            Default: True.
        out_indices: Output from which stages.
        frozen_stages: Stages to be frozen (stop grad and set eval mode).
            Default: -1. -1 means not freezing any parameters.
        include_top: Whether to include output layer. Default: True.
        flat_output: Whether to view the output tensor. Default: True.
        pretrained_path: Path to load pre_trained weights.
        use_native_op: Use torch's native operators when compiling
            models with trt and torchdynamo. Default: False
    """

    def __init__(
        self,
        depth_list: List[int],
        num_heads: List[int],
        num_classes: int = 1000,
        patch_size: Union[int, Tuple[int, int]] = 4,
        in_channels: int = 3,
        embedding_dims: int = 96,
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        dropout_ratio: float = 0.0,
        attention_dropout_ratio: float = 0.0,
        drop_path_ratio: float = 0.0,
        patch_norm: bool = True,
        out_indices: Sequence[int] = (0, 1, 2, 3),
        frozen_stages: int = -1,
        include_top: bool = True,
        flat_output: bool = True,
        pretrained_path: Optional[str] = None,
        use_native_op: bool = False,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.depth_list = depth_list
        self.num_layers = len(depth_list)
        self.embedding_dims = embedding_dims
        self.window_size = window_size

        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.dropout_ratio = dropout_ratio
        self.attention_drop_ratio = attention_dropout_ratio
        self.include_top = include_top
        self.flat_output = flat_output
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.pretrained_path = pretrained_path

        self.quant = QuantStub(scale=1.0 / 128.0)
        self.dequant = DeQuantStub()

        # split image into non-overlapping patches
        self.patch_embedding = PatchEmbedding(
            patch_size=patch_size,
            in_channels=in_channels,
            embedding_dims=embedding_dims,
            norm_layer=nn.LayerNorm if patch_norm else None,
        )

        self.pos_drop = nn.Dropout(p=dropout_ratio)

        # stochastic depth
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_ratio, sum(depth_list))
        ]

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embedding_dims * 2 ** i_layer),
                depth=depth_list[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=dropout_ratio,
                attn_drop=attention_dropout_ratio,
                drop_path=dpr[
                    sum(depth_list[:i_layer]) : sum(depth_list[: i_layer + 1])
                ],
                norm_layer=nn.LayerNorm,
                downsample=PatchMerging
                if (i_layer < self.num_layers - 1)
                else None,
                use_native_op=use_native_op,
            )
            self.layers.append(layer)

        num_features = [
            int(embedding_dims * 2 ** i) for i in range(self.num_layers)
        ]
        self.num_features = num_features

        # add a norm layer for each output
        for i_layer in out_indices:
            layer = nn.LayerNorm(num_features[i_layer])
            layer_name = f"norm{i_layer}"
            self.add_module(layer_name, layer)

        if self.include_top:
            self.norm = nn.LayerNorm(self.num_features[-1])
            self.avg_pool = nn.AdaptiveAvgPool1d(1)
            self.output = nn.Linear(self.num_features[-1], num_classes)
        else:
            self.norm, self.avg_pool, self.output = None, None, None

        self._freeze_stages()
        self.apply(self.init_weights)

        if self.pretrained_path is not None:
            load_state_dict(
                self,
                self.pretrained_path,
                map_location="cpu",
                allow_miss=True,
                ignore_extra=True,
                state_dict_update_func=update_state_dict_from_official,
                verbose=True,
            )

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def init_weights(self, m):
        """Initialize the weights in backbone."""

        if isinstance(m, nn.Linear):
            trunc_normal(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.quant(x)
        x = self.patch_embedding(x)
        Wh, Ww = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)

        outs = []
        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww)

            if i in self.out_indices:
                norm_layer = getattr(self, f"norm{i}")
                x_out = norm_layer(x_out)

                out = (
                    x_out.view(-1, H, W, self.num_features[i])
                    .permute(0, 3, 1, 2)
                    .contiguous()
                )
                outs.append(out)

        if not self.include_top:
            return outs
        x = self.norm(x)
        x = self.avg_pool(x.transpose(1, 2))
        x = torch.flatten(x, 1)
        x = self.output(x)
        x = self.dequant(x)
        if self.flat_output:
            x = x.view(-1, self.num_classes)
        return x

    def fuse_model(self):
        pass

    def set_qconfig(self):
        from .utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()

        if self.include_top:
            # disable output quantization for last quanti layer.
            self.output.qconfig = qconfig_manager.get_default_qat_out_qconfig()
