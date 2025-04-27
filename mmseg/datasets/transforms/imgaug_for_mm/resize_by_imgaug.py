from mmengine.registry import TRANSFORMS
from mmcv.transforms import BaseTransform
import imgaug.augmenters as iaa
import numpy as np

@TRANSFORMS.register_module()
class ResizeByImgaug(BaseTransform):
    """Resize image and segmentation map using imgaug.

    Required Keys:
    - img
    - gt_seg_map (optional)

    Modified Keys:
    - img
    - gt_seg_map (if present)

    Args:
        size (tuple[int, int]): Desired output size (height, width).
        keep_ratio (bool): If True, keep aspect ratio and resize the shorter
                           side to match; otherwise force resize to exact size.
    """

    def __init__(self, size, keep_ratio=False):
        assert isinstance(size, (tuple, list)) and len(size) == 2
        self.size = tuple(size)
        self.keep_ratio = keep_ratio

    def transform(self, results: dict) -> dict:
        img = results['img']
        original_shape = img.shape[:2]  # H, W

        if self.keep_ratio:
            # Resize with aspect ratio preserved
            h, w = original_shape
            scale = min(self.size[0] / h, self.size[1] / w)
            new_h, new_w = int(h * scale), int(w * scale)
        else:
            new_h, new_w = self.size

        resize_aug = iaa.Resize({"height": new_h, "width": new_w})
        results['img'] = resize_aug(image=img)

        # Handle segmentation maps
        for key in results.get('seg_fields', []):
            results[key] = resize_aug(image=results[key])

        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}(size={self.size}, '
                f'keep_ratio={self.keep_ratio})')
