# Copyright (c) Horizon Robotics. All rights reserved.
from functools import partial
from typing import Optional

import horizon_plugin_pytorch.nn.quantized as quantized
import torch
import torch.nn as nn
from horizon_plugin_pytorch.quantization import QuantStub
from torch.quantization import DeQuantStub

from .models.base_modules.basic_timesformer_module import (
    PatchEmbed,
    SpaceTimeBlock,
    load_pretrained,
)
from .models.weight_init import trunc_normal
from .registry import OBJECT_REGISTRY

__all__ = ["TimeSformer"]


class VisionTransformer(nn.Module):
    """A module of vision transformer.

    Args:
        img_size: Image size. Default: 224.
        patch_size: Patch size. Default: 16.
        in_chans: Number of input channels. Default: 3.
        num_classes: Num classes of output layer.
        embed_dim: Number of embedding output channels. Default: 768.
        depth: Depths of timesformer blocks. Default: 12.
        num_heads: Number of attention head. Default: 12.
        mlp_ratio: Ratio of mlp hidden dim to input dim. Default: 4.
        qkv_bias: Whether to add a learnable bias to query, key, value.
            Default: True.
        qk_scale: Override default qk scale of head_dim ** -0.5 if set.
        drop_rate: Dropout rate. Default: 0.
        attn_drop_rate: Attention dropout rate. Default: 0.
        drop_path_rate: Stochastic depth rate. Default: 0.1.
        norm_layer: Normalization layer. Default: nn.LayerNorm.
        num_frames: Number of frames. Default: 8.
        attention_type: Attention type, can be divided_space_time
            or space_only or joint_space_time.
        dropout: Dropout rate. Default: 0.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: Optional[int] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        norm_layer: nn.Module = nn.LayerNorm,
        num_frames: int = 8,
        attention_type: str = "divided_space_time",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.attention_type = attention_type
        self.depth = depth
        self.dropout = nn.Dropout(dropout)
        self.num_classes = num_classes
        self.num_features = (
            self.embed_dim
        ) = embed_dim  # num_features for consistency with other models
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        # Positional Embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim)
        )
        self.pos_drop = nn.Dropout(p=drop_rate)
        if self.attention_type != "space_only":
            self.time_embed = nn.Parameter(
                torch.zeros(1, num_frames, embed_dim)
            )
            self.time_drop = nn.Dropout(p=drop_rate)

        # Attention Blocks
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, self.depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                SpaceTimeBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    attention_type=self.attention_type,
                )
                for i in range(self.depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = (
            nn.Linear(embed_dim, num_classes)
            if num_classes > 0
            else nn.Identity()
        )

        self.cls_token_cat = quantized.FloatFunctional()
        self.pos_embed_add = quantized.FloatFunctional()
        self.time_embed_add = quantized.FloatFunctional()
        self.cls_token_time_cat = quantized.FloatFunctional()

        self.cls_token_quant = QuantStub(scale=None)
        self.pos_embed_quant = QuantStub(scale=None)
        self.time_embed_quant = QuantStub(scale=None)

        self.quant = QuantStub(scale=None)
        self.dequant = DeQuantStub()

        # Init weights
        trunc_normal(self.pos_embed, std=0.02)
        trunc_normal(self.cls_token, std=0.02)
        self.apply(self._init_weights)

        # initialization of temporal attention weights
        if self.attention_type == "divided_space_time":
            i = 0
            for m in self.blocks.modules():
                m_str = str(m)
                if "Block" in m_str:
                    if i > 0:
                        nn.init.constant_(m.temporal_fc.weight, 0)
                        nn.init.constant_(m.temporal_fc.bias, 0)
                    i += 1

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "time_embed"}

    def forward_features(self, x):
        B = x.shape[0]
        x, T, W = self.patch_embed(x)
        N = x.shape[0]
        cls_tokens = self.cls_token.expand(N, -1, -1)
        cls_tokens = self.cls_token_quant(cls_tokens)
        x = self.cls_token_cat.cat((cls_tokens, x), 1)

        pos_embed = self.pos_embed_quant(self.pos_embed)
        x = self.pos_embed_add.add(x, pos_embed)
        x = self.pos_drop(x)

        # Time Embeddings
        if self.attention_type != "space_only":
            cls_tokens = x[:B, 0, :].unsqueeze(1)
            x = x[:, 1:]
            S, C = x.shape[-2:]
            x = x.reshape(B, T, S, C).permute(0, 2, 1, 3)
            x = x.reshape(-1, T, C)
            time_embed = self.time_embed_quant(self.time_embed)
            x = self.time_embed_add.add(x, time_embed)
            x = self.time_drop(x)
            x = x.reshape(B, -1, C)
            x = self.cls_token_time_cat.cat((cls_tokens, x), 1)

        # Attention blocks
        for blk in self.blocks:
            x = blk(x, B, T, W)

        # Predictions for space-only baseline
        if self.attention_type == "space_only":
            x = x.reshape(B, T, S, C)
            x = torch.mean(x, 1)  # averaging predictions for every frame

        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        x = self.quant(x)
        x = self.forward_features(x)
        x = self.head(x)
        x = self.dequant(x)
        return x

    def fuse_model(self):
        pass


@OBJECT_REGISTRY.register
class TimeSformer(nn.Module):
    """A module of TimeSformer.

    Args:
        img_size: Image size. Default: 224.
        patch_size: Patch size. Default: 16.
        num_classes: Num classes of output layer.
        num_frames: Number of frames. Default: 8.
        test_mode: If true, use 3 spatial crops from clip and
            average the scores for these 3 crops. you should set
            UniformCropVideo transform in val_data_loader simultaneously.
        pretrained: Wether to load pretrained model.
        attention_type: Attention type, can be divided_space_time
            or space_only or joint_space_time.
        pretrained_model: Path to pretrained model checkpoint.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        num_classes: int = 400,
        num_frames: int = 8,
        test_mode: bool = False,
        pretrained: bool = False,
        attention_type: str = "divided_space_time",
        pretrained_model: str = "",
        **kwargs
    ):
        super(TimeSformer, self).__init__()
        self.pretrained = pretrained
        self.test_mode = test_mode
        self.model = VisionTransformer(
            img_size=img_size,
            num_classes=num_classes,
            patch_size=patch_size,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.1,
            num_frames=num_frames,
            attention_type=attention_type,
            **kwargs
        )
        self.attention_type = attention_type
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)
        if self.pretrained:
            load_pretrained(
                self.model,
                num_classes=num_classes,
                num_frames=num_frames,
                num_patches=self.num_patches,
                attention_type=self.attention_type,
                pretrained_model=pretrained_model,
            )

    def forward(self, x):
        if self.test_mode:
            x_split = x.chunk(3, dim=2)
            pred = [self.model(item).unsqueeze(1) for item in x_split]
            pred = torch.cat(pred, dim=1)
            x = torch.mean(pred, dim=1)
        else:
            x = self.model(x)
        return x

    def fuse_model(self):
        pass

    def set_qconfig(self):
        from .utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()

        self.model.head.qconfig = qconfig_manager.get_default_qat_out_qconfig()
        # self.model.blocks.qconfig = None
