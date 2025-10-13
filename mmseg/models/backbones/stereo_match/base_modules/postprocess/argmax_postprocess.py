# Copyright (c) Horizon Robotics. All rights reserved.

from typing import Mapping

import torch
from horizon_plugin_pytorch.quantization import March

from .registry import OBJECT_REGISTRY

__all__ = ["ArgmaxPostprocess", "HorizonAdasClsPostProcessor"]


# TODO(mengao.zhao, HDLT-235): refactor argmax_postprocess #
@OBJECT_REGISTRY.register
class ArgmaxPostprocess(torch.nn.Module):
    """Apply argmax of data in pred_dict.

    Args:
        data_name (str): name of data to apply argmax.
        dim (int): the dimension to reduce.
        keepdim (bool): whether the output tensor has dim retained or not.

    """

    def __init__(self, data_name: str, dim: int, keepdim: bool = False):
        super(ArgmaxPostprocess, self).__init__()
        self.data_name = data_name
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, pred_dict: Mapping, *args):
        if isinstance(pred_dict[self.data_name], list):
            argmax_datas = []
            for each_data in pred_dict[self.data_name]:
                argmax_datas.append(each_data.argmax(self.dim, self.keepdim))
            pred_dict[self.data_name] = argmax_datas
        elif isinstance(pred_dict[self.data_name], torch.Tensor):
            pred_dict[self.data_name] = pred_dict[self.data_name].argmax(
                self.dim, self.keepdim
            )
        else:
            raise TypeError("only support torch.tensor or list[torch.tensor]")
        return pred_dict

    def set_qconfig(self):
        from .utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()


@OBJECT_REGISTRY.register
class HorizonAdasClsPostProcessor(torch.nn.Module):
    """Apply argmax of data in pred_dict.

    Args:
        data_name (str): name of data to apply argmax.
        dim (int): the dimension to reduce.
        keepdim (bool): whether the output tensor has dim retained or not.

    """

    def __init__(
        self,
        data_name: str,
        dim: int,
        keep_dim: bool = True,
        march: str = March.BAYES,
    ):
        super(HorizonAdasClsPostProcessor, self).__init__()
        self.data_name = data_name
        self.dim = dim
        self.keep_dim = keep_dim
        self.march = march

    @torch.no_grad()
    def forward(self, pred_cls: Mapping, *args):
        assert isinstance(pred_cls, torch.Tensor), "only support torch.Tensor"
        batch_scores, batch_cls_idxs = pred_cls.max(
            dim=self.dim, keepdim=self.keep_dim
        )

        if self.march == March.BAYES:
            return torch.cat((batch_cls_idxs, batch_scores), dim=self.dim)
        else:
            return batch_cls_idxs

    def set_qconfig(self):
        from .utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()
