import numpy as np
import cv2
from mmcv.transforms import BaseTransform
from mmseg.registry import TRANSFORMS

SOIL_ID = 2
ANIMAL_ID = 3


@TRANSFORMS.register_module()
class DilateSoilMask(BaseTransform):
    """对分割 mask 中泥土类别进行膨胀操作的增强类。

    Args:
        kernel_size (int): 膨胀核大小（必须为奇数，如 3、5）。
        iterations (int): 膨胀次数。
    """

    def __init__(self, kernel_size=3, iterations=1):
        self.soil_id = SOIL_ID
        self.kernel_size = kernel_size
        self.iterations = iterations
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (self.kernel_size, self.kernel_size))

    def transform(self, results):
        """对 mask 执行膨胀处理"""
        gt_seg = results['gt_seg_map']  # shape: (H, W)，numpy array
        mask = (gt_seg == self.soil_id).astype(np.uint8)  # 二值化类别区域

        # 形态学膨胀
        dilated = cv2.dilate(mask, self.kernel, iterations=self.iterations)

        animal_seg = gt_seg == ANIMAL_ID
        # 将膨胀结果重新标记为目标类别，覆盖原始 gt
        gt_seg[dilated == 1] = self.soil_id
        # 动物类别不会被膨胀区域覆盖
        gt_seg[animal_seg] = ANIMAL_ID
        results['gt_seg_map'] = gt_seg

        return results
