import cv2
import numpy as np
from mmcv.transforms import BaseTransform
from mmengine.registry import TRANSFORMS


@TRANSFORMS.register_module()
class RandomMasking(BaseTransform):
    def __init__(self, mask_ratio=0.75, patch_size=16):
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size

    def transform(self, results):
        img = results['img']
        h, w, c = img.shape

        # 确保图像可以整除 patch_size
        assert h % self.patch_size == 0 and w % self.patch_size == 0, \
            f"Image size ({h}, {w}) must be divisible by patch size {self.patch_size}"

        gh, gw = h // self.patch_size, w // self.patch_size  # grid size
        num_patches = gh * gw
        num_mask = int(num_patches * self.mask_ratio)

        # 随机选哪些 patch 被遮挡
        mask_indices = np.random.choice(num_patches, num_mask, replace=False)
        mask_grid = np.zeros((gh, gw), dtype=bool)
        mask_grid[np.unravel_index(mask_indices, (gh, gw))] = True

        # 执行遮挡（置零）
        img = img.copy()
        for i in range(gh):
            for j in range(gw):
                if mask_grid[i, j]:
                    y1, y2 = i * self.patch_size, (i + 1) * self.patch_size
                    x1, x2 = j * self.patch_size, (j + 1) * self.patch_size
                    img[y1:y2, x1:x2, :] = 0

        results['img'] = img
        return results