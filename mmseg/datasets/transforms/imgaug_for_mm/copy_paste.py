from mmengine.registry import TRANSFORMS
from mmcv.transforms import BaseTransform
import imgaug.augmenters as iaa
import numpy as np
import random


@TRANSFORMS.register_module()
class CopyPasteTop(BaseTransform):
    """从图池中随机选图，将其下半部分粘贴到当前图上半部分。
    
    图池始终保持固定大小，每次替换池中旧图以保证多样性。
    """

    def __init__(self, pool_size=16, prob=1.0):
        self.pool_size = pool_size
        self.pool = []
        self.prob = prob
    def transform(self, results: dict) -> dict:
        if random.random() > self.prob:
            return results

        img = results['img']
        h, w = img.shape[:2]
        half = h // 2

        # 更新图池：
        current_copy = {
            'img': img.copy(),
            'gt_seg_map': results.get('gt_seg_map', None).copy() if 'gt_seg_map' in results else None
        }

        if half == 0:
            return results  # 图太小

        do_paste = len(self.pool) > 0
        paste_idx = None

        if do_paste:
            # 随机选一个池中图
            paste_idx = random.randint(0, len(self.pool) - 1)
            paste_entry = self.pool[paste_idx]
            paste_img = paste_entry['img']
            paste_seg = paste_entry.get('gt_seg_map')

            # 裁出其底部1/2
            paste_half = paste_img.shape[0] // 2
            paste_c = paste_img[paste_half:].copy()

            assert paste_c.shape[0] == half
            if paste_c.shape[0] != half:
                paste_c = np.resize(paste_c, (half, w, img.shape[2]))
            # 贴到原图的顶部1/2
            img[0:half] = paste_c
            results['img'] = img

            if 'gt_seg_map' in results and paste_seg is not None:
                seg = results['gt_seg_map']
                c_seg = paste_seg[paste_half:].copy()
                if c_seg.shape[0] != half:
                    c_seg = np.resize(c_seg, (half, seg.shape[1]))
                seg[0:half] = c_seg
                results['gt_seg_map'] = seg



        if len(self.pool) < self.pool_size:
            self.pool.append(current_copy)
        else:
            # 替换旧图（使用刚才参与拼贴的图的位置）
            if paste_idx is not None:
                self.pool[paste_idx] = current_copy
            else:
                replace_idx = random.randint(0, self.pool_size - 1)
                self.pool[replace_idx] = current_copy

        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(pool_size={self.pool_size})'
