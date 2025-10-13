# Copyright (c) Horizon Robotics. All rights reserved.

import torch
import torch.nn as nn
from horizon_plugin_pytorch.quantization import QuantStub
from torch.quantization import DeQuantStub

try:
    from torchcrf import CRF
except ImportError:
    CRF = None

from .models.base_modules.conv_module import ConvModule2d
from .registry import OBJECT_REGISTRY
from .utils.package_helper import require_packages

__all__ = ["NluMultiTaskBackbone"]


class BasicBlock(nn.Module):
    """Basic block for Tcn.

    Example:
        # N(batch size) , C(embedding dim) , H , W(seq length)
        N, C, H, W = 40, 512, 1, 10
        input_tensor = torch.randn(N, C, H, W)
        model = BasicBlock(C)
        output_tensor = model(input_tensor)

    """

    def __init__(self, input_dim, dilation_size=1, dropout_rate=0.2):
        """Init BasicBlock class.

        Args:
            input_dim (int): Input dimensions of BasicBlock module.
            dilation_size (int): Dilation size of Conv2d layer.
            dropout_rate (float): Dropout rate in the module.

        """
        super(BasicBlock, self).__init__()
        self.basic_conv = nn.Sequential(
            ConvModule2d(
                input_dim,
                input_dim,
                (1, 3),  # 卷积尺寸
                padding=(0, dilation_size),
                dilation=dilation_size,
                norm_layer=nn.BatchNorm2d(input_dim),
                act_layer=nn.ReLU(inplace=True),
            ),
            nn.Dropout(dropout_rate),
        )
        self.mod = torch.nn.quantized.FloatFunctional()

    def forward(self, x):
        # QTensor不支持torch.add
        out = self.mod.add(self.basic_conv(x), x)
        return out


class Tcn(nn.Module):
    """Tcn model.

    Example:
        # N(batch_size) , C(embedding dim) , H , W(seq length)
        N, C, H, W = 40, 512, 1, 10
        input_tensor = torch.randn(N, C, H, W)
        model = Tcn(C, 5)
        output_tensor = model(input_tensor) # same shape as input tensor

    """

    def __init__(self, input_dim, levels_num, dropout_rate=0.2):
        """Init Tcn model. Tcn model consists of serveral layers of BasicBlock.

        Args:
            input_dim (int): Input dimensions of BasicBlock.
            levels_num (int): Number of BasicBlock in Tcn Model.
            dropout_rate (float): Dropout rate of BasicBlock.

        """
        super(Tcn, self).__init__()
        self.encoder = nn.Sequential()
        for i in range(levels_num):
            self.encoder.add_module(  # 依次往nn.Sequential中加入模块，添加到计算图中
                "tcn_layer_" + str(i + 1),
                BasicBlock(
                    input_dim, 2 ** i, dropout_rate
                ),  # 参数依次是：输入维度、膨胀系数、dropout参数
            )

    def forward(self, x):
        """Model forward.

        Args:
            x (torch.tensor): Shape [N, C, H, W].

        Returns:
            torch.tensor: Shape [N, C, H, W].

        """
        out = self.encoder(x)
        return out

    def fuse_model(self):
        for module in self.encoder:
            if hasattr(module, "fuse_model"):
                module.fuse_model()


class BasicDecoder(nn.Module):
    def __init__(self, embed_dim, label_num, dropout_rate=0.2):
        super(BasicDecoder, self).__init__()
        self.basic_decoder = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            ConvModule2d(embed_dim, label_num, (1, 1)),
        )

    def forward(self, x):
        out = self.basic_decoder(x)
        return out


@OBJECT_REGISTRY.register
class NluMultiTaskBackbone(nn.Module):
    @require_packages(
        "torchcrf==0.7.2", raise_msg="Please `pip3 install pytorch-crf==0.7.2`"
    )
    def __init__(
        self,
        tokenizer,
        label_processor,
        embed_dim=512,
        tcn_levels=5,
        dropout_rate=0.2,
    ):
        """Init NluMultiTaskBackbone class.

        Args:
            tokenizer: instance of NluBasicTokenizer.
            label_processor: instance of NluLabelProcessor.
            embed_dim (int): Embedding dimension for input tokens.
            tcn_levels (int): Number of BasicBlock in TCN model.
            dropout_rate (float): Dropout rate of classification layer.

        """
        super(NluMultiTaskBackbone, self).__init__()

        self.tokenizer = tokenizer
        self.label_processor = label_processor

        self.embedding = nn.Embedding(
            tokenizer.vocab_size, embed_dim, padding_idx=0
        )  # 随机初始化词向量
        self.encoder = Tcn(embed_dim, tcn_levels)
        self.domain_cls = BasicDecoder(
            embed_dim, label_processor.domain_label_size, dropout_rate
        )
        self.intent_cls = BasicDecoder(
            embed_dim, label_processor.intent_label_size, dropout_rate
        )
        self.slots_cls = BasicDecoder(
            embed_dim, label_processor.slots_label_size, dropout_rate
        )
        self.crf = CRF(label_processor.slots_label_size, batch_first=True)
        self.quant = QuantStub(scale=1 / 128.0, zero_point=0.0)
        self.dequant = DeQuantStub()

    def forward(self, input_emb):
        input_emb = self.quant(input_emb)
        out = self.encoder(input_emb)  # (N,C,1,W)

        # HAT的编译不支持torch.select,即切片操作,用torch.split改写
        domain_out, intent_out, slots_out = torch.split(
            out, [1, 1, out.shape[3] - 2], dim=3
        )

        # domain 取句子的第1个token
        domain_out = self.domain_cls(domain_out)  # (N,domain_label_num,1,1)
        domain_out = domain_out.squeeze(3).squeeze(2)  # (N,domain_label_num)

        # intent 取句子的第2个token
        intent_out = self.intent_cls(intent_out)  # (N,intent_label_num,1,1)
        intent_out = intent_out.squeeze(3).squeeze(2)  # (N,intent_label_num)

        # slots  取句子除了前2个token之后的token
        slots_out = self.slots_cls(slots_out)  # (N,slots_label_num,1,W-2)
        slots_out = (
            slots_out.squeeze(2).permute(0, 2, 1).contiguous()
        )  # (N,W-2,slots_label_num)

        domain_out = self.dequant(domain_out)
        intent_out = self.dequant(intent_out)
        slots_out = self.dequant(slots_out)
        return (domain_out, intent_out, slots_out)

    def fuse_model(self):
        if hasattr(self.encoder, "fuse_model"):
            self.encoder.fuse_model()
