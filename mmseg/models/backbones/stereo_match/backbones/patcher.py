from typing import Tuple

import torch.nn as nn
from horizon_plugin_pytorch.nn import Interpolate

from .models.base_modules.extend_container import ExtSequential
from .registry import OBJECT_REGISTRY

__all__ = ["ZeroPad2DPatcher", "ResizePatcher"]


@OBJECT_REGISTRY.register
class ZeroPad2DPatcher(nn.Module):
    """ZeroPad2D Patcher.

    Args:
        backbone: Backbone module.
        input_padding: Input padding, in (w_left, w_right, h_top, h_bottom) format.  # noqa E501

    """

    def __new__(cls, backbone: nn.Module, input_padding: Tuple[int]):

        assert len(input_padding) == 4
        if not all([p == 0 for p in input_padding]):
            pad = nn.ZeroPad2d(input_padding)
            if hasattr(backbone, "quant"):
                backbone.quant = ExtSequential(modules=[backbone.quant, pad])

                def hook(state_dict, prefix, *args):
                    backbone_keys = list(state_dict.keys())
                    for key in backbone_keys:
                        name = key[len(prefix) :]
                        if (
                            name.startswith("quant.")
                            and name not in backbone.state_dict()
                        ):
                            new_key0 = prefix + "quant.0." + name[6:]
                            new_key1 = prefix + "quant.1." + name[6:]
                            state_dict[new_key0] = state_dict[key]
                            state_dict[new_key1] = state_dict[key]
                            state_dict.pop(key)

                backbone._register_load_state_dict_pre_hook(hook)
            else:
                raise ValueError(
                    "`backbone` has no `quant` module, please check your `backbone`."  # noqa E501
                )
        return backbone


@OBJECT_REGISTRY.register
class ResizePatcher(nn.Module):
    """Resize2D Patcher.

    Args:
        backbone: Backbone module.
        input_scale: Input scale, in (h_scale, w_scale) format.  # noqa E501

    """

    def __new__(cls, backbone: nn.Module, input_scale: Tuple[float]):

        assert len(input_scale) == 2
        if not all([p == 1 for p in input_scale]):
            op = Interpolate(
                scale_factor=input_scale,
                mode="bilinear",
                align_corners=False,
                recompute_scale_factor=True,
            )
            if hasattr(backbone, "quant"):
                backbone.quant = ExtSequential(modules=[backbone.quant, op])

                def hook(state_dict, prefix, *args):
                    backbone_keys = list(state_dict.keys())
                    for key in backbone_keys:
                        name = key[len(prefix) :]
                        if (
                            name.startswith("quant.")
                            and name not in backbone.state_dict()
                        ):
                            new_key0 = prefix + "quant.0." + name[6:]
                            new_key1 = prefix + "quant.1." + name[6:]
                            state_dict[new_key0] = state_dict[key]
                            state_dict[new_key1] = state_dict[key]
                            state_dict.pop(key)

                backbone._register_load_state_dict_pre_hook(hook)
            else:
                raise ValueError(
                    "`backbone` has no `quant` module, please check your `backbone`."  # noqa E501
                )
        return backbone
