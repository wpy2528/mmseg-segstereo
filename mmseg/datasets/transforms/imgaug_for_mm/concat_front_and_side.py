from mmengine.registry import TRANSFORMS
from mmcv.transforms import BaseTransform
import imgaug.augmenters as iaa
import numpy as np
import random
import cv2

@TRANSFORMS.register_module()
class ConcatFrontAndSide(BaseTransform):
    """从池中选一张图作为 side_image，当前图作为 front_image，
    按指定规则拼接成一张新图。
    
    逻辑同 concat_front_and_side_images，生成融合图作为增强。

    Args:
        pool_size (int): 池子大小。保存最近见过的一些图以供 side_image 使用。
        prob (float): 执行增广的概率。默认 0.5。
    """

    def __init__(self, pool_size=16, prob=0.5):
        self.pool_size = pool_size
        self.prob = prob
        self.pool = []

    def transform(self, results: dict) -> dict:
        """执行拼接操作。"""
        if np.random.rand() > self.prob:
            return results  # 按概率跳过增广

        img = results['img']
        dst_size = (img.shape[1], img.shape[0])  # (w, h)

        # 需要至少一张图才能拼
        if len(self.pool) == 0:
            # 池子为空，先把当前图存进去
            self._add_to_pool(results)
            return results

        # 从池子随机选一张 side_image
        side_idx = random.randint(0, len(self.pool) - 1)
        side_entry = self.pool[side_idx]
        side_img = side_entry['img']
        
        self._replace_in_pool(results, side_idx)

        # 执行拼接
        # 随机生成一个上图（侧目）的占比
        upper_height_proportion = np.random.uniform(1.0 / 3, 1.0 / 3 + 2.0 / 10)
        # 随机生成一个上图（侧目）的裁切位置
        src_upper_tly_proportion = 1 - upper_height_proportion
        # 随机生成一个下图（正视）的裁切位置
        src_lower_tly_proportion = 1.0 / 3
        new_img = self.concat_front_and_side_images(front_image=img, side_image=side_img, dst_size=dst_size, 
                                                    upper_height_proportion=upper_height_proportion,
                                                    src_upper_tly_proportion=src_upper_tly_proportion,
                                                    src_lower_tly_proportion=src_lower_tly_proportion)
        results['img'] = new_img

        # 同步处理 segmentation
        if 'gt_seg_map' in results and side_entry.get('gt_seg_map') is not None:
            front_seg = results['gt_seg_map']
            side_seg = side_entry['gt_seg_map']
            new_seg = self.concat_front_and_side_images(front_image=front_seg[..., np.newaxis],
                                                        side_image=side_seg[..., np.newaxis],
                                                        dst_size=dst_size,
                                                        upper_height_proportion=upper_height_proportion,
                                                        src_upper_tly_proportion=src_upper_tly_proportion,
                                                        src_lower_tly_proportion=src_lower_tly_proportion)
            results['gt_seg_map'] = new_seg.squeeze(-1)


        return results

    def concat_front_and_side_images(self, front_image, side_image, dst_size, 
                                     upper_height_proportion, src_upper_tly_proportion, src_lower_tly_proportion):
        """实现具体拼接逻辑"""
        dst_image = np.zeros((dst_size[1], dst_size[0], front_image.shape[2]), dtype=front_image.dtype)

        src_upper_tly = round(side_image.shape[0] * src_upper_tly_proportion)
        src_upper_height = round(side_image.shape[0] * upper_height_proportion)

        src_lower_tly = round(front_image.shape[0] * src_lower_tly_proportion)
        src_lower_height = round(front_image.shape[0] * (1.0 - upper_height_proportion))

        # 防止超界
        src_upper_tly = min(src_upper_tly, side_image.shape[0] - 1)
        src_lower_tly = min(src_lower_tly, front_image.shape[0] - 1)

        src_upper_roi = side_image[src_upper_tly : src_upper_tly + src_upper_height, :, :]
        src_lower_roi = front_image[src_lower_tly : src_lower_tly + src_lower_height, :, :]

        dst_upper_height = round(dst_size[1] * upper_height_proportion)
        dst_lower_height = dst_size[1] - dst_upper_height

        dst_upper_roi = cv2.resize(src_upper_roi, (dst_size[0], dst_upper_height))
        dst_lower_roi = cv2.resize(src_lower_roi, (dst_size[0], dst_lower_height))
        if len(dst_upper_roi.shape) == 2:
            dst_upper_roi = dst_upper_roi[..., np.newaxis]
        if len(dst_lower_roi.shape) == 2:
            dst_lower_roi = dst_lower_roi[..., np.newaxis]

        dst_image[0 : dst_upper_height, :, :] = dst_upper_roi
        dst_image[dst_upper_height : dst_size[1], :, :] = dst_lower_roi

        return dst_image

    def _add_to_pool(self, results):
        """把当前图存入池子（深拷贝）"""
        entry = {
            'img': results['img'].copy(),
            'gt_seg_map': results.get('gt_seg_map', None).copy() if 'gt_seg_map' in results else None
        }
        if len(self.pool) < self.pool_size:
            self.pool.append(entry)
        else:
            replace_idx = random.randint(0, self.pool_size - 1)
            self.pool[replace_idx] = entry

    def _replace_in_pool(self, results, idx):
        """用当前图替换掉池中指定 idx 的图"""
        entry = {
            'img': results['img'].copy(),
            'gt_seg_map': results.get('gt_seg_map', None).copy() if 'gt_seg_map' in results else None
        }
        self.pool[idx] = entry

    def __repr__(self):
        return f'{self.__class__.__name__}(pool_size={self.pool_size}, prob={self.prob})'


if __name__ == '__main__':
    import cv2
    import numpy as np
    from pathlib import Path
    import random

    # 创建测试目录
    test_dir = Path("test_concat_front_and_side")
    test_dir.mkdir(exist_ok=True)

    # 从val.txt读取图像路径
    with open("data/perception_segmentation/0427/val.txt", "r") as f:
        img_paths = f.readlines()
    img_paths = [p.strip() for p in img_paths]
    # 随机分成两组
    random.shuffle(img_paths)
    img_paths1 = img_paths[:len(img_paths)//2]
    img_paths2 = img_paths[len(img_paths)//2:]
    
    
