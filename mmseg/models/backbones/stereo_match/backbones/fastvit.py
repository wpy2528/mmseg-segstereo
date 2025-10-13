from functools import partial
from typing import Callable, List, Tuple, Union

import torch
import torch.nn as nn
from horizon_plugin_pytorch.nn import quantized
from horizon_plugin_pytorch.quantization import QuantStub
from torch.nn.init import trunc_normal_
from torch.quantization import DeQuantStub

from .models.base_modules.basic_repvgg_module import (
    MultiBranchConvModule,
    MultiBranchModule,
    RepBlock,
)
from .models.base_modules.basic_swin_module import DropPath
from .models.base_modules.conv_module import ConvModule2d
from .registry import OBJECT_REGISTRY
from .utils.model_helpers import fx_wrap


def get_mobileone_block(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int = 1,
    dilation: int = 1,
    groups: int = 1,
    use_se: bool = False,
    act_layer: nn.Module = None,
    use_scale_branch: bool = True,
    num_conv_branches: int = 1,
    post_bn: bool = False,
    deploy: bool = False,
):
    """Get Mobileone Block with RepBlock.

    Mobileone block is typlically a reparameterized block, which has multiple
    conv branches in training stage, and the block could reparameterize to a
    single branch in inference stage.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Size of the largest convolutional kernel.
        stride: Stride for the convolution operation.
        dilation: Dilation rate for the convolution operation.
        groups: Number of groups for grouped convolution.
        use_se: Whether to use the Squeeze-and-Excitation block.
        act_layer: Activation function to apply.
        use_scale_branch: Whether to use a scaling branch.
        num_conv_branches: Number of convolution branches.
        post_bn: Whether to apply batch normalization after the convolution.
            default False for original repvgg and mobileone;
            True for quantization friendly qa version. (see repvgg-qa)
        deploy: Whether the block is used in deployment mode
            (reparameterize to a single branch).

    Returns:
        nn.Module: A MobileOne block as a PyTorch module.
    """

    if kernel_size > 1 and use_scale_branch:
        k_size_list = [0, 1] + [kernel_size] * num_conv_branches
    else:
        k_size_list = [0] + [kernel_size] * num_conv_branches
    if post_bn:
        if kernel_size > 1 and use_scale_branch:
            bn_flag_list = [False, False] + [True] * num_conv_branches
        elif kernel_size > 1:
            bn_flag_list = [False] + [True] * num_conv_branches
        else:
            bn_flag_list = [False] + [False] * num_conv_branches
    else:
        bn_flag_list = [True] * len(k_size_list)
    return RepBlock(
        module=MultiBranchConvModule(
            in_channels=in_channels,
            out_channels=out_channels,
            k_size_list=k_size_list,
            bn_flag_list=bn_flag_list,
            stride=stride,
            dilation=dilation,
            groups=groups,
        ),
        norm_layer=nn.BatchNorm2d(out_channels) if post_bn else None,
        act_layer=act_layer,
        use_se=use_se,
        deploy=deploy,
    )


def get_repLKConv(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int,
    groups: int,
    small_kernel: int,
    deploy: bool = False,
):
    """`RepLKConv` block which rep small kenel conv into large kernel conv.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Size of the convolutional kernel.
        stride: Stride for the convolution operation.
        groups: Number of groups for grouped convolution.
        small_kernel: Size of the small kernel conv.
        deploy: Whether the block is used in deployment mode
            (whether reparameterize the model).

    Returns:
        nn.Module: A repetitive Local Kernel Convolution block as a PyTorch
            module.
    """
    return RepBlock(
        module=MultiBranchConvModule(
            in_channels=in_channels,
            out_channels=out_channels,
            k_size_list=[kernel_size, small_kernel],
            bn_flag_list=[False, False],
            stride=stride,
            groups=groups,
        ),
        norm_layer=nn.BatchNorm2d(out_channels),
        act_layer=nn.ReLU(),
        deploy=deploy,
    )


def convolutional_stem(
    in_channels: int,
    out_channels: int,
    post_bn: bool = False,
    deploy: bool = False,
) -> nn.Sequential:
    """Build convolutional stem with MobileOne blocks.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        deploy: Flag to instantiate model in inference mode.

    Returns:
        nn.Sequential object with stem elements.
    """
    return nn.Sequential(
        get_mobileone_block(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            groups=1,
            act_layer=nn.GELU(),
            use_se=False,
            num_conv_branches=1,
            post_bn=post_bn,
            deploy=deploy,
        ),
        get_mobileone_block(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            groups=out_channels,
            act_layer=nn.GELU(),
            use_se=False,
            num_conv_branches=1,
            post_bn=post_bn,
            deploy=deploy,
        ),
        get_mobileone_block(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            groups=1,
            act_layer=nn.GELU(),
            use_se=False,
            num_conv_branches=1,
            post_bn=post_bn,
            deploy=deploy,
        ),
    )


class MultiheadSelfAttention(nn.Module):
    """Multi-headed Self Attention module.

    Args:
        dim: Number of embedding dimensions.
        head_dim: Number of hidden dimensions per head.
        qkv_bias: Use bias or not.
        attn_drop: Dropout rate for attention tensor.
        proj_drop: Dropout rate for projection tensor.
    """

    def __init__(
        self,
        dim: int,
        head_dim: int = 32,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        assert dim % head_dim == 0, "dim should be divisible by head_dim"
        self.head_dim = head_dim
        self.num_heads = dim // head_dim
        self.scale = head_dim ** -0.5

        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=qkv_bias)
        self.attn_drop = nn.Dropout2d(attn_drop)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.proj_drop = nn.Dropout2d(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

        self.mul = quantized.FloatFunctional()
        self.matmul = quantized.FloatFunctional()
        self.add = quantized.FloatFunctional()
        self.attn_matmul = quantized.FloatFunctional()

        self.scale_quant = QuantStub()

    @fx_wrap()
    def _gen_qkv(self, qkv, B, C, H, W):
        # new 4dims
        qkv = qkv.permute(0, 2, 3, 1)  # BxHxWx3C
        q, k, v = qkv[..., :C], qkv[..., C : C * 2], qkv[..., C * 2 :]  # BHWC
        q = q.reshape(B, H * W, self.num_heads, C // self.num_heads)
        q = q.permute(0, 2, 1, 3)
        k = k.reshape(B, H * W, self.num_heads, C // self.num_heads)
        k = k.permute(0, 2, 1, 3)
        v = v.reshape(B, H * W, self.num_heads, C // self.num_heads)
        v = v.permute(0, 2, 1, 3)
        return q, k, v

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)

        # new 4dims
        q, k, v = self._gen_qkv(qkv, B, C, H, W)  # B, h, N, C/h

        # trick here to make q@k.t more stable
        scale = self.scale_quant(self.scale)
        q = self.mul.mul(q, scale)
        attention = self.matmul.matmul(q, k, x_trans=False, y_trans=True)

        attention = self.softmax(attention)

        attention = self.attn_drop(attention)
        x = self.attn_matmul.matmul(attention, v)
        # attn = (q * self.scale) @ k.transpose(-2, -1)

        x = x.permute(0, 2, 1, 3).reshape(B, N, C, 1)
        x = x.permute(0, 2, 1, 3)

        x = x.reshape(B, C, H, W).contiguous()
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class RepLKPatchEmbed(nn.Module):
    """Convolutional patch embedding layer.

    Args:
        patch_size: Patch size for embedding computation.
        stride: Stride for convolutional embedding layer.
        in_channels: Number of channels of input tensor.
        embed_dim: Number of embedding dimensions.
        deploy: Flag to instantiate model in inference mode.
    """

    def __init__(
        self,
        patch_size: int,
        stride: int,
        in_channels: int,
        embed_dim: int,
        post_bn: bool = False,
        deploy: bool = False,
    ) -> None:
        super().__init__()
        block = []
        block.append(
            get_repLKConv(
                in_channels=in_channels,
                out_channels=embed_dim,
                kernel_size=patch_size,
                stride=stride,
                groups=in_channels,
                small_kernel=3,
                deploy=deploy,
            )
        )
        block.append(
            get_mobileone_block(
                in_channels=embed_dim,
                out_channels=embed_dim,
                kernel_size=1,
                stride=1,
                groups=1,
                act_layer=nn.GELU(),
                use_se=False,
                num_conv_branches=1,
                post_bn=post_bn,
                deploy=deploy,
            )
        )
        self.proj = nn.Sequential(*block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x


def get_repMixer(
    dim: int,
    kernel_size: int = 3,
    use_layer_scale: bool = True,
    layer_scale_init_value: float = 1e-5,
    post_bn: bool = False,
    deploy: bool = False,
):
    """Get a RepMixer block configuration for use in a neural network.

    Args:
        dim: The number of input channels and output channels.
        kernel_size: Size of the convolutional kernel.
        use_layer_scale: If True, apply layer scale before the output.
        layer_scale_init_value: Initial value of the layer scale.
        post_bn: If True, apply batch normalization after the convolution.
        deploy: If True, return a configuration suitable for deployment.

    Returns:
        The RepMixer block.
    """
    norm = nn.BatchNorm2d(dim)
    mixer = get_mobileone_block(
        in_channels=dim,
        out_channels=dim,
        kernel_size=kernel_size,
        groups=dim,
        act_layer=None,
        post_bn=post_bn,
        deploy=deploy,
    )
    if use_layer_scale:
        layer_scale = layer_scale_init_value * torch.ones((dim, 1, 1))
    else:
        layer_scale = 1
    branches = MultiBranchModule(
        branches=[nn.Identity(), mixer, norm],
        scale_list=[1, layer_scale, -1 * layer_scale],
    )
    repmixer = RepBlock(
        module=branches,
        deploy=deploy,
    )
    return repmixer


class ConvFFN(nn.Module):
    """Convolutional FFN Module.

    Args:
        in_channels: Number of input channels.
        hidden_channels: Number of channels after expansion.
        out_channels: Number of output channels.
        act_layer: Activation layer.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = None,
        out_channels: int = None,
        act_layer: nn.Module = nn.GELU,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.conv = ConvModule2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=7,
            padding=3,
            groups=in_channels,
            bias=False,
            norm_layer=nn.BatchNorm2d(num_features=out_channels),
        )
        self.fc1 = ConvModule2d(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=1,
            act_layer=act_layer(),
        )
        self.fc2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


def get_repCPE(
    in_channels: int,
    embed_dim: int = 768,
    spatial_shape: int = 7,
    deploy: bool = False,
):
    """Reparameterized conditional positional encoding.

    For more details refer to paper:
    `Conditional Positional Encodings for Vision Transformers
    <https://arxiv.org/pdf/2102.10882.pdf>`_

    In this implementation, we can reparameterize this module to eliminate
        a skip connection.
    """
    return RepBlock(
        module=MultiBranchConvModule(
            in_channels=in_channels,
            out_channels=embed_dim,
            k_size_list=[0, spatial_shape],
            bn_flag_list=[False, False],
        ),
        deploy=deploy,
    )


class RepMixerBlock(nn.Module):
    """Implementation of Metaformer block with RepMixer as token mixer.

    For more details on Metaformer structure, please refer to:
    `MetaFormer Is Actually What You Need for Vision
    <https://arxiv.org/pdf/2111.11418.pdf>`_

     Args:
        dim: Number of embedding dimensions.
        kernel_size: Kernel size for repmixer.
        mlp_ratio: MLP expansion ratio.
        act_layer: Activation layer.
        dropout: Dropout rate.
        drop_path: Drop path rate.
        use_layer_scale: Flag to turn on layer scale.
        layer_scale_init_value: Layer scale value at initialization.
        deploy: Flag to instantiate block in inference mode.
    """

    def __init__(
        self,
        dim: int,
        kernel_size: int = 3,
        mlp_ratio: float = 4.0,
        act_layer: nn.Module = nn.GELU,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        use_layer_scale: bool = True,
        layer_scale_init_value: float = 1e-5,
        post_bn: bool = False,
        deploy: bool = False,
    ):
        super().__init__()

        self.token_mixer = get_repMixer(
            dim,
            kernel_size=kernel_size,
            use_layer_scale=use_layer_scale,
            layer_scale_init_value=layer_scale_init_value,
            post_bn=post_bn,
            deploy=deploy,
        )

        assert (
            mlp_ratio > 0
        ), "MLP ratio should be greater than 0, found: {}".format(mlp_ratio)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.convffn = ConvFFN(
            in_channels=dim,
            hidden_channels=mlp_hidden_dim,
            act_layer=act_layer,
            dropout=dropout,
        )

        # Drop Path
        self.drop_path = (
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )

        # Layer Scale
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale = nn.Parameter(
                layer_scale_init_value * torch.ones((dim, 1, 1)),
                requires_grad=True,
            )
            self.layer_quant = QuantStub()

    def switch_to_deploy(self):
        for module in self.children():
            if hasattr(module, "switch_to_deploy"):
                module.switch_to_deploy()
        self.deploy = True

    def forward(self, x):
        if self.use_layer_scale:
            layer_scale = self.layer_quant(self.layer_scale)
            x = self.token_mixer(x)
            x = x + self.drop_path(layer_scale * self.convffn(x))
        else:
            x = self.token_mixer(x)
            x = x + self.drop_path(self.convffn(x))
        return x


class AttentionBlock(nn.Module):
    """Metaformer block with MultiheadSelfAttention as token mixer.

    For more details on Metaformer structure, please refer to:
    `MetaFormer Is Actually What You Need for Vision
     <https://arxiv.org/pdf/2111.11418.pdf>`_

     Args:
        dim: Number of embedding dimensions.
        mlp_ratio: MLP expansion ratio.
        act_layer: Activation layer.
        norm_layer: Normalization layer.
        dropout: Dropout rate.
        drop_path: Drop path rate.
        use_layer_scale: Flag to turn on layer scale.
        layer_scale_init_value: Layer scale value at initialization.
    """

    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 4.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.BatchNorm2d,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        use_layer_scale: bool = True,
        layer_scale_init_value: float = 1e-5,
    ):
        super().__init__()

        self.norm = norm_layer(dim)
        self.token_mixer = MultiheadSelfAttention(dim=dim)

        assert (
            mlp_ratio > 0
        ), "MLP ratio should be greater than 0, found: {}".format(mlp_ratio)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.convffn = ConvFFN(
            in_channels=dim,
            hidden_channels=mlp_hidden_dim,
            act_layer=act_layer,
            dropout=dropout,
        )

        # Drop path
        self.drop_path = (
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )

        # Layer Scale
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim, 1, 1)),
                requires_grad=True,
            )
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim, 1, 1)),
                requires_grad=True,
            )
        self.add1 = quantized.FloatFunctional()
        self.add2 = quantized.FloatFunctional()

    def forward(self, x):
        if self.use_layer_scale:
            x = self.add1.add(
                x,
                self.drop_path(
                    self.layer_scale_1 * self.token_mixer(self.norm(x))
                ),
            )
            x = self.add2.add(
                x, self.drop_path(self.layer_scale_2 * self.convffn(x))
            )
        else:
            x = self.add1.add(
                x, self.drop_path(self.token_mixer(self.norm(x)))
            )
            x = self.add2.add(x, self.drop_path(self.convffn(x)))
        return x


def basic_blocks(
    dim: int,
    block_index: int,
    num_blocks: List[int],
    token_mixer_type: str,
    kernel_size: int = 3,
    mlp_ratio: float = 4.0,
    act_layer: nn.Module = nn.GELU,
    norm_layer: nn.Module = nn.BatchNorm2d,
    dropout: float = 0.0,
    drop_path_rate: float = 0.0,
    use_layer_scale: bool = True,
    layer_scale_init_value: float = 1e-5,
    post_bn: bool = False,
    deploy: bool = False,
) -> nn.Sequential:
    """Build FastViT blocks within a stage.

    Args:
        dim: Number of embedding dimensions.
        block_index: block index.
        num_blocks: List containing number of blocks per stage.
        token_mixer_type: Token mixer type.
        kernel_size: Kernel size for repmixer.
        mlp_ratio: MLP expansion ratio.
        act_layer: Activation layer.
        norm_layer: Normalization layer.
        dropout: Dropout rate.
        drop_path_rate: Drop path rate.
        use_layer_scale: Flag to turn on layer scale regularization.
        layer_scale_init_value: Layer scale value at initialization.
        deploy: Flag to instantiate block in inference mode.

    Returns:
        nn.Sequential object of all the blocks within the stage.
    """
    blocks = []
    for block_idx in range(num_blocks[block_index]):
        block_dpr = (
            drop_path_rate
            * (block_idx + sum(num_blocks[:block_index]))
            / (sum(num_blocks) - 1)
        )
        if token_mixer_type == "repmixer":
            blocks.append(
                RepMixerBlock(
                    dim,
                    kernel_size=kernel_size,
                    mlp_ratio=mlp_ratio,
                    act_layer=act_layer,
                    dropout=dropout,
                    drop_path=block_dpr,
                    use_layer_scale=use_layer_scale,
                    layer_scale_init_value=layer_scale_init_value,
                    post_bn=post_bn,
                    deploy=deploy,
                )
            )
        elif token_mixer_type == "attention":
            blocks.append(
                AttentionBlock(
                    dim,
                    mlp_ratio=mlp_ratio,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    dropout=dropout,
                    drop_path=block_dpr,
                    use_layer_scale=use_layer_scale,
                    layer_scale_init_value=layer_scale_init_value,
                )
            )
        else:
            raise ValueError(
                "Token mixer type: {} not supported".format(token_mixer_type)
            )
    blocks = nn.Sequential(*blocks)

    return blocks


class FastViTBase(nn.Module):
    """FastViT architecture <https://arxiv.org/pdf/2303.14189.pdf>`.

    Args:
        layers: List specifying the number of layers in each stage.
        token_mixers: Token mixer types for each stage.
        embed_dims: List of embedding dimensions for each stage.
            If None, it will be determined automatically.
        mlp_ratios: List of MLP expansion ratios for each stage.
            If None, default values are used.
        downsamples: List of downsampling flags for each stage.
        repmixer_kernel_size: Kernel size for the RepMixer operation.
        norm_layer: Normalization layer.
        act_layer: Activation function layer.
        num_classes: Number of output classes.
        pos_embs: List of function that return positional embedding modules
            for each stage. if None, no pos_embed for this stage.
        down_patch_size: Patch size for downsampling.
        down_stride: Stride for downsampling.
        dropout: Dropout rate.
        drop_path_rate: Drop path rate.
        use_layer_scale: Use layer scale before the output.
        layer_scale_init_value: Initial value of the layer scale.
        cls_ratio: Ratio for classifier expansion.
        post_bn:  Whether to apply batch normalization after the convolution.
            default False for original repvgg and mobileone;
            True for quantization friendly qa version. (see repvgg-qa)
        deploy: Flag to instantiate model in inference mode.
        include_top: Whether to include output layer.
        flat_output: Whether to view the output tensor.
    """

    def __init__(
        self,
        layers: List[int],
        token_mixers: Tuple[str, ...],
        embed_dims: List[int] = None,
        mlp_ratios: List[int] = None,
        downsamples: List[bool] = None,
        repmixer_kernel_size=3,
        norm_layer: nn.Module = nn.BatchNorm2d,
        act_layer: nn.Module = nn.GELU,
        num_classes: int = 1000,
        pos_embs: List[Union[None, Callable]] = None,
        down_patch_size: int = 7,
        down_stride: int = 2,
        dropout: float = 0.0,
        drop_path_rate: float = 0.0,
        use_layer_scale: bool = True,
        layer_scale_init_value: float = 1e-5,
        cls_ratio: float = 2.0,
        post_bn: bool = False,
        deploy: bool = False,
        include_top: bool = True,
        flat_output: bool = True,
        **kwargs,
    ) -> None:

        super().__init__()

        if include_top:
            self.num_classes = num_classes
        self.include_top = include_top
        self.flat_output = flat_output

        self.quant = QuantStub(scale=1 / 128.0)
        self.dequant = DeQuantStub()

        if pos_embs is None:
            pos_embs = [None] * len(layers)

        # Convolutional stem
        self.patch_embed = convolutional_stem(
            3, embed_dims[0], post_bn, deploy
        )

        # Build the main stages of the network architecture
        network = []
        for i in range(len(layers)):
            # Add position embeddings if requested
            if pos_embs[i] is not None:
                network.append(
                    pos_embs[i](embed_dims[i], embed_dims[i], deploy=deploy)
                )
            stage = basic_blocks(
                embed_dims[i],
                i,
                layers,
                token_mixer_type=token_mixers[i],
                kernel_size=repmixer_kernel_size,
                mlp_ratio=mlp_ratios[i],
                act_layer=act_layer,
                norm_layer=norm_layer,
                dropout=dropout,
                drop_path_rate=drop_path_rate,
                use_layer_scale=use_layer_scale,
                layer_scale_init_value=layer_scale_init_value,
                post_bn=post_bn,
                deploy=deploy,
            )
            network.append(stage)
            if i >= len(layers) - 1:
                break

            # Patch merging/downsampling between stages.
            if downsamples[i] or embed_dims[i] != embed_dims[i + 1]:
                network.append(
                    RepLKPatchEmbed(
                        patch_size=down_patch_size,
                        stride=down_stride,
                        in_channels=embed_dims[i],
                        embed_dim=embed_dims[i + 1],
                        post_bn=post_bn,
                        deploy=deploy,
                    )
                )

        self.network = nn.ModuleList(network)

        # For segmentation and detection, extract intermediate output
        if not self.include_top:
            # add a norm layer for each output
            self.out_indices = [0, 2, 4, 6]
            for i_emb, i_layer in enumerate(self.out_indices):
                if i_emb == 0:
                    layer = nn.Identity()
                else:
                    layer = norm_layer(embed_dims[i_emb])
                layer_name = f"norm{i_layer}"
                self.add_module(layer_name, layer)
        else:
            # Classifier head
            self.conv_exp = get_mobileone_block(
                in_channels=embed_dims[-1],
                out_channels=int(embed_dims[-1] * cls_ratio),
                kernel_size=3,
                stride=1,
                groups=embed_dims[-1],
                deploy=deploy,
                act_layer=nn.GELU(),
                use_se=True,
                post_bn=post_bn,
                num_conv_branches=1,
            )
            self.output = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=1),
                ConvModule2d(
                    int(embed_dims[-1] * cls_ratio),
                    num_classes,
                    1,
                ),
            )
        self.out_quant = QuantStub()
        self.apply(self.cls_init_weights)
        self.deploy = deploy
        if deploy:
            self.switch_to_deploy()

    def cls_init_weights(self, m: nn.Module) -> None:
        # """Init. for classification"""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def _scrub_checkpoint(checkpoint, model):
        sterile_dict = {}
        for k1, v1 in checkpoint.items():
            if k1 not in model.state_dict():
                continue
            if v1.shape == model.state_dict()[k1].shape:
                sterile_dict[k1] = v1
        return sterile_dict

    def forward_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        return x

    def forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        outs = []
        for idx, block in enumerate(self.network):
            x = block(x)
            if not self.include_top and idx in self.out_indices:
                norm_layer = getattr(self, f"norm{idx}")
                x_out = norm_layer(x)
                outs.append(x_out)
        if not self.include_top:
            return outs
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quant(x)
        x = self.forward_embeddings(x)
        x = self.forward_tokens(x)
        if not self.include_top:
            return x
        x = self.conv_exp(x)
        out = self.output(x)
        out = self.dequant(out)
        if self.flat_output:
            out = out.view(x.size(0), self.num_classes)
        return out

    def switch_to_deploy(self):
        training = self.training
        self.eval()
        for module in self.modules():
            if module != self and hasattr(module, "switch_to_deploy"):
                module.switch_to_deploy()
        if training:
            self.train()

        self.deploy = True

    def set_qconfig(self):
        from .utils import qconfig_manager

        if self.include_top:
            getattr(
                self.output, "1"
            ).qconfig = qconfig_manager.get_default_qat_out_qconfig()


Model_cfgs = {
    "t8": {
        "layers": [2, 2, 4, 2],
        "embed_dims": [48, 96, 192, 384],
        "mlp_ratios": [3, 3, 3, 3],
        "downsamples": [True, True, True, True],
        "token_mixers": ("repmixer", "repmixer", "repmixer", "repmixer"),
    },
    "t12": {
        "layers": [2, 2, 6, 2],
        "embed_dims": [64, 128, 256, 512],
        "mlp_ratios": [3, 3, 3, 3],
        "downsamples": [True, True, True, True],
        "token_mixers": ("repmixer", "repmixer", "repmixer", "repmixer"),
    },  # noqa
    "s12": {
        "layers": [2, 2, 6, 2],
        "embed_dims": [64, 128, 256, 512],
        "mlp_ratios": [4, 4, 4, 4],
        "downsamples": [True, True, True, True],
        "token_mixers": ("repmixer", "repmixer", "repmixer", "repmixer"),
    },  # noqa
    "sa12": {
        "layers": [2, 2, 6, 2],
        "embed_dims": [64, 128, 256, 512],
        "mlp_ratios": [4, 4, 4, 4],
        "downsamples": [True, True, True, True],
        "pos_embs": [
            None,
            None,
            None,
            partial(get_repCPE, spatial_shape=7),
        ],
        "token_mixers": ("repmixer", "repmixer", "repmixer", "attention"),
    },  # noqa
    "sa24": {
        "layers": [4, 4, 12, 4],
        "embed_dims": [64, 128, 256, 512],
        "mlp_ratios": [4, 4, 4, 4],
        "downsamples": [True, True, True, True],
        "pos_embs": [
            None,
            None,
            None,
            partial(get_repCPE, spatial_shape=(7, 7)),
        ],
        "token_mixers": ("repmixer", "repmixer", "repmixer", "attention"),
    },  # noqa
    "sa36": {
        "layers": [6, 6, 18, 6],
        "embed_dims": [64, 128, 256, 512],
        "mlp_ratios": [4, 4, 4, 4],
        "downsamples": [True, True, True, True],
        "pos_embs": [
            None,
            None,
            None,
            partial(get_repCPE, spatial_shape=(7, 7)),
        ],
        "token_mixers": ("repmixer", "repmixer", "repmixer", "attention"),
    },  # noqa
    "ma36": {
        "layers": [6, 6, 18, 6],
        "embed_dims": [76, 152, 304, 608],
        "mlp_ratios": [4, 4, 4, 4],
        "downsamples": [True, True, True, True],
        "pos_embs": [
            None,
            None,
            None,
            partial(get_repCPE, spatial_shape=(7, 7)),
        ],
        "token_mixers": ("repmixer", "repmixer", "repmixer", "attention"),
    },  # noqa
}


@OBJECT_REGISTRY.register
class FastViT(FastViTBase):
    """FastViT model that specify the model type.

    Args:
        model_type: The type of FastViT model to use.
        num_classes: The number of output classes.
        post_bn: If True, apply batch normalization after the final
            classification layer.
        deploy: If True, configure the model for deployment.
        include_top: If True, include the final classification layer.
        flat_output: If True, return a flat output tensor instead of class
            probabilities.
    """

    def __init__(
        self,
        model_type: str,
        num_classes: int = 1000,
        post_bn: bool = False,
        deploy: bool = False,
        include_top: bool = True,
        flat_output: bool = True,
    ):
        super(FastViT, self).__init__(
            **Model_cfgs[model_type],
            num_classes=num_classes,
            post_bn=post_bn,
            deploy=deploy,
            include_top=include_top,
            flat_output=flat_output,
        )
