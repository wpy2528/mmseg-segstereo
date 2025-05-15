import numpy as np
import cv2
import random
from mmcv.transforms import BaseTransform
from mmengine.registry import TRANSFORMS


@TRANSFORMS.register_module()
class CropTop(BaseTransform):
    """裁切图像上方的指定比例部分。
    
    Args:
        prop (float): 要裁切的上方部分的比例，范围[0,1]。默认为0.2。
        prob (float): 应用此增广的概率。默认为1.0。
    """

    def __init__(self, prop=0.2, prob=1.0):
        super().__init__()
        self.prop = prop
        self.prob = prob

    def transform(self, results):
        """应用裁切增广。
        
        Args:
            results (dict): 包含'img'和'gt_seg_map'的字典。
            
        Returns:
            dict: 增广后的结果字典。
        """
        img = results['img']
        gt_seg_map = results.get('gt_seg_map', None)

        if not isinstance(img, np.ndarray):
            raise TypeError(f'输入图像必须是np.ndarray类型，但得到的是{type(img)}')

        # 根据概率决定是否执行增广
        if random.random() > self.prob:
            return results

        h, w = img.shape[:2]
        crop_h = int(h * self.prop)
        
        # 裁切图像
        img = img[crop_h:, :]
        
        # 如果存在分割图，也进行相同的裁切
        if gt_seg_map is not None:
            gt_seg_map = gt_seg_map[crop_h:, :]
            results['gt_seg_map'] = gt_seg_map

        results['img'] = img
        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(prop={self.prop}, prob={self.prob})'
