# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_pixel_sampler import BasePixelSampler
from .builder import PIXEL_SAMPLERS


@PIXEL_SAMPLERS.register_module()
class PixelRangeSampler(BasePixelSampler):
    """Pixel Range Sampler for segmentation.
    
    Set loss weight to zero for pixels with seg_label values outside the given range.

    Args:
        context (nn.Module): The context of sampler, subclass of
            :obj:`BaseDecodeHead`.
        min_value (float, optional): Minimum value. Pixels with value < min_value
            will have zero loss weight. Default: 0.0.
        max_value (float, optional): Maximum value. Pixels with value > max_value
            will have zero loss weight. Default: 10000.0.
    """

    def __init__(self, context, min_value=0.0, max_value=10000.0):
        super().__init__()
        self.context = context
        self.min_value = min_value
        self.max_value = max_value

    def sample(self, seg_logit, seg_label):
        """Sample pixels based on pixel range.
        
        Set loss weight to zero for pixels with pixel values outside the specified range.

        Args:
            seg_logit (torch.Tensor): segmentation logits, shape (N, C, H, W)
            seg_label (torch.Tensor): segmentation label containing pixel values, 
                shape (N, C, H, W) where C >= 1

        Returns:
            torch.Tensor: segmentation weight, shape (N, H, W)
        """
        with torch.no_grad():
            assert seg_label.shape[1] == 1
            assert seg_logit.shape == seg_label.shape
            
            # Create weight tensor, initialize with ones
            seg_weight = seg_label.new_ones(size=seg_label.size())
            
            # Create mask for valid pixels (not ignore_index)
            valid_mask = seg_label != self.context.ignore_index
            
            # Create mask for pixel range
            pixel_in_range = (seg_label >= self.min_value) & (seg_label < self.max_value)
            
            # Set weight to zero for pixels outside pixel range or invalid pixels
            seg_weight[~valid_mask] = 0.0
            seg_weight[valid_mask & ~pixel_in_range] = 0.0
            
            return seg_weight
