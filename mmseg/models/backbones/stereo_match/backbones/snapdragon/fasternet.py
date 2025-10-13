from functools import partial
from typing import List

import horizon_plugin_pytorch.nn.quantized as quantized
import torch
import torch.nn as nn
from horizon_plugin_pytorch.quantization import QuantStub
from torch import Tensor
from torch.quantization import DeQuantStub

from .models.base_modules.basic_swin_module import DropPath
from .models.base_modules.conv_module import ConvModule2d
from .registry import OBJECT_REGISTRY

__all__ = ["SNDRFasterNet"]


class Partial_conv3(nn.Module):
    def __init__(self, dim, n_div):
        super().__init__()
        self.dim_conv3 = int(dim / n_div)
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = ConvModule2d(
            self.dim_conv3,
            self.dim_conv3,
            3,
            1,
            1,
            bias=False,
            norm_layer=None,
            act_layer=None,
        )
        self.cat_op = quantized.FloatFunctional()

    def forward(self, x: Tensor) -> Tensor:
        if self.dim_untouched == 0:
            x = self.partial_conv3(x)
        else:
            x1, x2 = torch.split(
                x,
                [self.dim_conv3, self.dim_untouched],
                dim=1,
            )
            x1 = self.partial_conv3(x1)
            x = self.cat_op.cat((x1, x2), 1)

        return x

    def fuse_model(self) -> None:
        self.partial_conv3.fuse_model()

    def set_qconfig(self) -> None:
        from .utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()


class MLPBlock(nn.Module):
    def __init__(
        self,
        dim,
        n_div,
        mlp_ratio,
        drop_path,
        act_layer,
        norm_layer,
    ):

        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.add_op = quantized.FloatFunctional()
        self.drop_path = (
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )
        self.n_div = n_div
        mlp_hidden_dim = int(dim * mlp_ratio)

        mlp_layer: List[nn.Module] = [
            ConvModule2d(
                dim,
                mlp_hidden_dim,
                1,
                bias=False,
                norm_layer=norm_layer(mlp_hidden_dim),
                act_layer=act_layer(),
            ),
            ConvModule2d(
                mlp_hidden_dim,
                dim,
                1,
                bias=False,
                norm_layer=None,
                act_layer=None,
            ),
        ]
        self.mlp = nn.Sequential(*mlp_layer)

        self.spatial_mixing = Partial_conv3(
            dim,
            n_div,
        )

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.spatial_mixing(x)
        x = self.add_op.add(shortcut, self.drop_path(self.mlp(x)))
        return x

    def fuse_model(self) -> None:
        self.spatial_mixing.fuse_model()

        for mod in self.mlp:
            if hasattr(mod, "fuse_model"):
                mod.fuse_model()

    def set_qconfig(self) -> None:
        from .utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()

        for mod in self.mlp:
            if mod is not None and hasattr(mod, "set_qconfig"):
                mod.set_qconfig()

        if hasattr(self.spatial_mixing, "set_qconfig"):
            self.spatial_mixing.set_qconfig()


class BasicStage(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        n_div,
        mlp_ratio,
        drop_path,
        norm_layer,
        act_layer,
    ):

        super().__init__()

        blocks_list = [
            MLPBlock(
                dim=dim,
                n_div=n_div,
                mlp_ratio=mlp_ratio,
                drop_path=drop_path[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
            )
            for i in range(depth)
        ]

        self.blocks = nn.Sequential(*blocks_list)

    def forward(self, x: Tensor) -> Tensor:
        x = self.blocks(x)
        return x

    def fuse_model(self) -> None:
        for mod in self.blocks:
            if hasattr(mod, "fuse_model"):
                mod.fuse_model()

    def set_qconfig(self) -> None:
        from .utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()

        for mod in self.blocks:
            if mod is not None and hasattr(mod, "set_qconfig"):
                mod.set_qconfig()


class PatchEmbed(nn.Module):
    def __init__(
        self, patch_size, patch_stride, in_chans, embed_dim, norm_layer
    ):
        super().__init__()
        self.proj = ConvModule2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_stride,
            bias=False,
            norm_layer=norm_layer(embed_dim)
            if norm_layer is not None
            else nn.Identity(),
            act_layer=None,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)

        return x

    def fuse_model(self) -> None:
        self.proj.fuse_model()

    def set_qconfig(self, is_out=False) -> None:
        from .utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()
        if hasattr(self.proj, "set_qconfig"):
            self.proj.set_qconfig()


class PatchMerging(nn.Module):
    def __init__(self, patch_size2, patch_stride2, dim, norm_layer):
        super().__init__()
        self.reduction = ConvModule2d(
            dim,
            2 * dim,
            kernel_size=patch_size2,
            stride=patch_stride2,
            bias=False,
            norm_layer=norm_layer(2 * dim)
            if norm_layer is not None
            else nn.Identity(),
            act_layer=None,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.reduction(x)
        return x

    def fuse_model(self) -> None:
        self.reduction.fuse_model()

    def set_qconfig(self, is_out=False) -> None:
        from .utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()
        if hasattr(self.reduction, "set_qconfig"):
            self.reduction.set_qconfig()


@OBJECT_REGISTRY.register
class SNDRFasterNet(nn.Module):
    """
    A module of fasternet for snapdragon SoC.

    Implments the network structure of FasterNet:
    <https://arxiv.org/pdf/2303.03667.pdf>

    Adapted from GitHub:
    <https://github.com/JierunChen/FasterNet>

    Args:
        in_chans (int):
            Number of input channels.
            Set 3 for rgb images.
        embed_dim (int):
            Embedding dimension:
            {'t0': 40, 't1': 64, 't2': 96, 's': 128, 'm': 144, 'l': 192},
            recommended setting is 32(small) or 48(large).
        depths (tuple):
            Tuple of integers representing the number of layers in each stage.
            {
                't0': [1, 2, 8, 2],
                't1': [1, 2, 8, 2],
                't2': [1, 2, 8, 2],
                's':  [1, 2, 13, 2],
                'm':  [3, 4, 18, 3],
                'l':  [3, 4, 18, 3],
            },
            recommended setting is [1, 2, 8, 2].
        mlp_ratio (float):
            Ratio of MLP hidden dimension to embedding dimension,
            recommended setting is 2.
        n_div (int):
            Number of divisions in each Partial_conv3 layer,
            recommended setting is [1, 2, 4, 4] when embed_dim is 32,
            and [1.5, 3, 6, 6] when embed_dim is 48.
        patch_size (int):
            Patch size for the first PatchEmbed layer,
            recommended setting is 4.
        patch_stride (int):
            Patch stride for the first PatchEmbed layer,
            recommended setting is 4.
        patch_size2 (int):
            Patch size for subsequent PatchMerging layers,
            recommended setting is 2.
        patch_stride2 (int):
            Patch stride for subsequent PatchMerging layers,
            recommended setting is 2.
        patch_norm (bool):
            Whether to apply normalization to the PatchEmbed layer,
            recommended setting is True.
        drop_path_rate (float):
            Drop path rate for stochastic depth:
            {'t0': 0, 't1': 0.02, 't2': 0.05, 's': 0.1, 'm': 0.2, 'l': 0.3},
            recommended setting is 0.
        norm_layer (str):
            Type of normalization layer to use,
            recommended setting is 'BN'.
        act_layer (str):
            Type of activation layer to use,
            recommended setting is 'RELU'.
    """

    def __init__(
        self,
        in_chans=3,
        embed_dim=32,
        depths=(1, 2, 8, 2),
        mlp_ratio=2.0,
        n_div=(1, 2, 4, 4),
        patch_size=4,
        patch_stride=4,
        patch_size2=2,  # for subsequent layers
        patch_stride2=2,
        patch_norm=True,
        drop_path_rate=0.0,
        norm_layer="BN",
        act_layer="RELU",
        **kwargs
    ):
        super().__init__()

        norm_layer = nn.BatchNorm2d
        act_layer = partial(nn.ReLU, inplace=True)

        self.num_stages = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_stages - 1))
        self.mlp_ratio = mlp_ratio
        self.depths = depths

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            patch_stride=patch_stride,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )

        # stochastic depth decay rule
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]

        # build BasicStage layers
        self.stage0 = BasicStage(
            dim=int(embed_dim * 2 ** 0),
            n_div=n_div[0],
            depth=depths[0],
            mlp_ratio=self.mlp_ratio,
            drop_path=dpr[sum(depths[:0]) : sum(depths[: 0 + 1])],
            norm_layer=norm_layer,
            act_layer=act_layer,
        )
        self.stage2 = BasicStage(
            dim=int(embed_dim * 2 ** 1),
            n_div=n_div[1],
            depth=depths[1],
            mlp_ratio=self.mlp_ratio,
            drop_path=dpr[sum(depths[:1]) : sum(depths[: 1 + 1])],
            norm_layer=norm_layer,
            act_layer=act_layer,
        )
        self.stage4 = BasicStage(
            dim=int(embed_dim * 2 ** 2),
            n_div=n_div[2],
            depth=depths[2],
            mlp_ratio=self.mlp_ratio,
            drop_path=dpr[sum(depths[:2]) : sum(depths[: 2 + 1])],
            norm_layer=norm_layer,
            act_layer=act_layer,
        )
        self.stage6 = BasicStage(
            dim=int(embed_dim * 2 ** 3),
            n_div=n_div[3],
            depth=depths[3],
            mlp_ratio=self.mlp_ratio,
            drop_path=dpr[sum(depths[:3]) : sum(depths[: 3 + 1])],
            norm_layer=norm_layer,
            act_layer=act_layer,
        )

        # build PatchMerging layers
        self.stage1 = PatchMerging(
            patch_size2=patch_size2,
            patch_stride2=patch_stride2,
            dim=int(embed_dim * 2 ** 0),
            norm_layer=norm_layer,
        )
        self.stage3 = PatchMerging(
            patch_size2=patch_size2,
            patch_stride2=patch_stride2,
            dim=int(embed_dim * 2 ** 1),
            norm_layer=norm_layer,
        )
        self.stage5 = PatchMerging(
            patch_size2=patch_size2,
            patch_stride2=patch_stride2,
            dim=int(embed_dim * 2 ** 2),
            norm_layer=norm_layer,
        )

        self.quant = QuantStub(scale=1.0 / 128.0)
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)

        x1 = self.stage0(self.patch_embed(x))  # [4, 32, 32, 32]
        x2 = self.stage2(self.stage1(x1))  # [4, 64, 16, 16]
        x3 = self.stage4(self.stage3(x2))  # [4, 128, 8, 8]
        x4 = self.stage6(self.stage5(x3))  # [4, 256, 4, 4]

        return [x1, x2, x3, x4]

    def fuse_model(self) -> None:
        for mod in [
            self.stage0,
            self.stage1,
            self.stage2,
            self.stage3,
            self.stage4,
            self.stage5,
            self.stage6,
            self.patch_embed,
        ]:
            if hasattr(mod, "fuse_model"):
                mod.fuse_model()

    def set_qconfig(self) -> None:
        from .utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()

        for mod in [
            self.stage0,
            self.stage1,
            self.stage2,
            self.stage3,
            self.stage4,
            self.stage5,
            self.stage6,
            self.patch_embed,
        ]:
            if mod is not None and hasattr(mod, "set_qconfig"):
                mod.set_qconfig()
