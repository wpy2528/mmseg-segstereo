import os
from mmengine.registry import TRANSFORMS
from mmcv.transforms import BaseTransform
import cv2
import numpy as np

@TRANSFORMS.register_module()
class ResizeToFrontOrSideImageOriginalSize(BaseTransform):
    """把给定图像缩放到前目或侧目图像原始尺寸"""

    def __init__(self):
        super().__init__()
        self.front_hw = (272, 320)  # 前目图目标尺寸
        self.side_hw = (320, 272)  # 侧目图目标尺寸

    def transform(self, results: dict) -> dict:
        src_image_np = results['img']
        src_image_path = results["img_path"]

        is_front = self.should_resize_to_front(src_image_path, src_image_np)
        if is_front:
            target_hw = self.front_hw
        else:
            target_hw = self.side_hw

        results['img'] = cv2.resize(src_image_np, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_LINEAR)
        results['img_shape'] = results['img'].shape[:2]
        
        # 处理分割图
        for key in results.get('seg_fields', []):
            results[key] = cv2.resize(results[key], (target_hw[1], target_hw[0]), interpolation=cv2.INTER_NEAREST)

        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}(size={self.size}, '
                f'keep_ratio={self.keep_ratio})')

    
    # 根据图像路径和图像原始尺寸判断是否应该缩放到前目图像尺寸
    @staticmethod
    def should_resize_to_front(src_image_path, src_image_np=None):
        ts = ["/home/mck/datasets/grass_seg_data_c3_reassigned/train/0058_0076/images/2025-05-08-095916-110.jpg_crop1.jpg",
        "/home/mck/datasets/grass_seg_data_c3_reassigned/train/2024/2024-07/0724_seg_special/images/left_101114180.jpg",
        "/home/mck/datasets/grass_seg_data_c3_reassigned/train/2024/2024-07/0724_seg_special/images/right_side_3263643641.jpg"
        ]
        
        src_image_name = os.path.basename(src_image_path)
        if "outside" not in src_image_name and "side" in src_image_name:
            return False
        if '_' in src_image_name:
            postfix = os.path.splitext(src_image_name)[0].split("_")[-1]
            if postfix in ["1", "crop1"]:
                return True
        if "front" in src_image_name:
            return True
        if src_image_np is None:
            src_image_np = cv2.imread(src_image_path)
        h, w = src_image_np.shape[:2]
        if w > h:
            return True
        else:
            return False
        
        
if __name__ == "__main__":
    transform = ResizeToFrontOrSideImageOriginalSize()
    with open("work_dirs/stdc2_grass-320x320_ft/20250519_104643/train_samples.txt", "r") as f:
        lines = [line.strip() for line in f.readlines()]
    s_count = 0
    f_count = 0
    from tqdm import tqdm
    for line in tqdm(lines):
        s = ResizeToFrontOrSideImageOriginalSize.should_resize_to_front(line)
        if s:
            s_count += 1
        else:
            f_count += 1
    print(f"s_count: {s_count}, f_count: {f_count}")
