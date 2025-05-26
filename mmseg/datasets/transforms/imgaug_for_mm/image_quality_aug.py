import numpy as np
import cv2
import random
import imgaug.augmenters as iaa
from mmcv.transforms import BaseTransform
from mmengine.registry import TRANSFORMS


@TRANSFORMS.register_module()
class ImageQualityAug(BaseTransform):
    """Simulate ISP color style variations for data augmentation, with probability control."""

    def __init__(self, prob=1.0):
        """
        Args:
            prob (float): Probability of applying the augmentation. Default 1.0 (always apply).
        """
        super().__init__()
        self.prob = prob

        # 定义 imgaug 增强器组合
        self.augmentor = iaa.Sequential([
            iaa.Sometimes(
                0.5,  # 50% 的概率加噪声
                iaa.AdditiveGaussianNoise(scale=(1, 8))  # 加随机高斯噪声
            )
        ])

    def transform(self, results):
        """Apply ISP style augmentation."""
        img = results['img']

        if not isinstance(img, np.ndarray):
            raise TypeError(f'Input img must be np.ndarray, but got {type(img)}')

        # 根据设定概率决定是否执行增强
        if random.random() > self.prob:
            return results  # 不增强，直接返回原图

        # OpenCV默认是BGR，imgaug期望RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 应用 imgaug 增强器
        augmented = self.augmentor(image=img_rgb)

        # 转回 BGR
        img_bgr = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)

        results['img'] = img_bgr
        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(prob={self.prob})'


if __name__ == '__main__':
    # 测试ISP风格增强
    import cv2
    import numpy as np

    # 读取测试图像
    img = cv2.imread('pgs/left_686193389.jpg')
    
    # 创建增强器实例
    isp_aug = ImageQualityAug(prob=1.0) # 设置prob=1确保一定会增强
    
    # 生成多次增强结果
    n_aug = 8  # 增强次数
    aug_imgs = []
    aug_imgs.append(img)  # 添加原图
    
    # 进行多次增强
    for i in range(n_aug):
        results = {'img': img.copy()}
        aug_results = isp_aug.transform(results)
        aug_imgs.append(aug_results['img'])
    
    # 计算拼接图像的尺寸
    img_h, img_w = img.shape[:2]
    n_cols = 3  # 每行放3张图
    n_rows = (len(aug_imgs) + n_cols - 1) // n_cols
    
    # 创建拼接图像
    canvas = np.zeros((img_h * n_rows, img_w * n_cols, 3), dtype=np.uint8)
    
    # 将图像拼接到画布上
    for idx, aug_img in enumerate(aug_imgs):
        row = idx // n_cols
        col = idx % n_cols
        canvas[row*img_h:(row+1)*img_h, col*img_w:(col+1)*img_w] = aug_img
    
    # 保存拼接后的图像
    cv2.imwrite('demo_augmented_grid.jpg', canvas)
    print('已保存拼接后的增强图像到 demo_augmented_grid.jpg')