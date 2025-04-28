import cv2
import numpy as np
from mmcv.transforms import BaseTransform
from mmengine.registry import TRANSFORMS


@TRANSFORMS.register_module()
class MaskEdgeIgnore(BaseTransform):
    def __init__(self, 
                 target_class, 
                 ignore_value=255, 
                 edge_width=5):
        self.target_class = target_class
        self.ignore_value = ignore_value
        self.edge_width = edge_width

    def transform(self, results):
        mask = results['gt_seg_map']  # (H, W), dtype=np.uint8 or np.int32

        # 创建一个只关注目标类别的mask
        class_mask = (mask == self.target_class).astype(np.uint8)  # 0/1

        # 用opencv找边界
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(class_mask, kernel, iterations=self.edge_width)
        eroded = cv2.erode(class_mask, kernel, iterations=self.edge_width)

        # 边界区域 = 膨胀 - 腐蚀
        boundary = ((dilated - eroded) > 0)

        # 在原mask上，把这些边界区域设成ignore_value
        mask[boundary] = self.ignore_value

        results['gt_seg_map'] = mask

        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}(target_class={self.target_class}, '
                f'ignore_value={self.ignore_value}, edge_width={self.edge_width})')
