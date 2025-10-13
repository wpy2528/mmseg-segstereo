from collections import OrderedDict

import torch
import torch.nn.functional as ff
from torch import nn

from .registry import OBJECT_REGISTRY

__all__ = ["Bottleneck", "AttentionPool2d", "WorkConditionResNet"]


class Bottleneck(nn.Module):
    """
    A bottle neck for work condition resnet.

    Args:
        inplanes: Input channels.
        planes: Output channels.
    """

    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1  # noqa
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1  # noqa
            self.downsample = nn.Sequential(
                OrderedDict(
                    [
                        ("-1", nn.AvgPool2d(stride)),
                        (
                            "0",
                            nn.Conv2d(
                                inplanes,
                                planes * self.expansion,
                                1,
                                stride=1,
                                bias=False,
                            ),
                        ),
                        ("1", nn.BatchNorm2d(planes * self.expansion)),
                    ]
                )
            )

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out


class AttentionPool2d(nn.Module):
    """Add attention module to the output feature.

    Args:
        spacial_dim: Dimension to calculate patches of input.
        embed_dim: Number of linear projection output channels.
        num_heads: Number of attention head.
        output_dim: Out dimension of attention module.
    """

    def __init__(
        self,
        spacial_dim: list,
        embed_dim: int,
        num_heads: int,
        output_dim: int = None,
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            torch.randn(spacial_dim[0] * spacial_dim[1] + 1, embed_dim)
            / embed_dim ** 0.5
        )
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(
            2, 0, 1
        )  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = ff.multi_head_attention_forward(
            query=x,
            key=x,
            value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat(
                [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]
            ),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False,
        )

        return x[0]


@OBJECT_REGISTRY.register
class WorkConditionResNet(nn.Module):
    """A ResNet class that is similar to torchvision's but contains the following changes.

    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool

    Args:
        layers: Number of layers for workconditionresnet.
        output_dim: Out dimension of attention module, useful when use_attnpool is True.
        heads: Number of attention head, useful when use_attnpool is True.
        channel_list: Channel dimension of each stage.
        bn_kwargs: Dict for BN layer.
        input_resolution: Input size.
        use_attnpool: Whether to use attention module.
    """  # noqa

    def __init__(
        self,
        layers,
        output_dim,
        heads,
        channel_list,
        bn_kwargs,
        input_resolution=(224, 224),
        use_attnpool=False,
    ):
        super(WorkConditionResNet, self).__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution
        self.use_attnpool = use_attnpool

        # the 3-layer stem
        self.conv1 = nn.Conv2d(
            3, channel_list[0], kernel_size=3, stride=2, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(channel_list[0], **bn_kwargs)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            channel_list[0],
            channel_list[0],
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(channel_list[0], **bn_kwargs)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(
            channel_list[0],
            channel_list[1],
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(channel_list[1], **bn_kwargs)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = channel_list[
            1
        ]  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(channel_list[1], layers[0])
        self.layer2 = self._make_layer(channel_list[2], layers[1], stride=2)
        self.layer3 = self._make_layer(channel_list[3], layers[2], stride=2)
        self.layer4 = self._make_layer(channel_list[4], layers[3], stride=2)

        embed_dim = (
            channel_list[1] * 32
        )  # width * 32  # the ResNet feature dimension
        if self.use_attnpool:
            self.attnpool = AttentionPool2d(
                (input_resolution[0] // 32, input_resolution[1] // 32),
                embed_dim,
                heads,
                output_dim,
            )

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        output = []

        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            output.append(x)
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        output.append(x)
        x = self.layer2(x)
        output.append(x)
        x = self.layer3(x)
        output.append(x)
        x = self.layer4(x)
        output.append(x)
        if self.use_attnpool:
            x = self.attnpool(x)
            output.append(x)

        return output
