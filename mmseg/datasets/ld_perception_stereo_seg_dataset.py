# Copyright (c) OpenMMLab. All rights reserved.
"""乐动感知双目分割数据集：在 LDPerceptionSegDataset 扫描逻辑上生成立体训练所需字段。"""
import os.path as osp
from typing import List

from mmengine.logging import print_log

from mmseg.registry import DATASETS

from .ld_perception_seg_dataset import LDPerceptionSegDataset

__all__ = ['LDPerceptionStereoSegDataset']


@DATASETS.register_module()
class LDPerceptionStereoSegDataset(LDPerceptionSegDataset):
    """与 :class:`LDPerceptionSegDataset` 相同的样本枚举方式，但每条样本输出 SegStereo 所需双目字段。

    将单目 ``img_path``（视为**左图**）按规则映射为右图路径与视差路径：

    .. code-block:: none

        data_root/.../images/foo.jpg          -> left_img_path
        data_root/.../{right_images_subdir}/foo.jpg -> right_img_path
        data_root/.../{disp_subdir}/foo{disp_suffix} -> left_disp_path

    ``seg_map_path`` 仍与单目版一致（``images`` -> ``labels`` 及后缀替换）。

    Args:
        right_images_subdir (str): 右图所在子目录名，替换左图路径中的 ``/{images_dirname}/``。
            默认 ``images_right``。
        disp_subdir (str): 视差图所在子目录名。默认 ``disparity``。
        disp_suffix (str): 视差文件扩展名，默认 ``.png``（亦常见 ``.pfm``）。
        images_dirname (str): 左图目录名片段，默认 ``images``，用于路径替换定位。
    """

    def __init__(self,
                 right_images_subdir: str = 'images_right',
                 disp_subdir: str = 'disparity',
                 disp_suffix: str = '.png',
                 images_dirname: str = 'images',
                 **kwargs) -> None:
        self.right_images_subdir = right_images_subdir.strip('/')
        self.disp_subdir = disp_subdir.strip('/')
        self.disp_suffix = disp_suffix
        self.images_dirname = images_dirname.strip('/')
        super().__init__(**kwargs)

    def _needle(self) -> str:
        return '/' + self.images_dirname + '/'

    def _left_to_right(self, left_path: str) -> str:
        n = self._needle()
        if n not in left_path:
            raise ValueError(
                f'左图路径中应包含 {n} 以便替换为右图目录，当前: {left_path}')
        rep = '/' + self.right_images_subdir + '/'
        return left_path.replace(n, rep, 1)

    def _left_to_disp(self, left_path: str) -> str:
        n = self._needle()
        if n not in left_path:
            raise ValueError(
                f'左图路径中应包含 {n} 以便替换为视差目录，当前: {left_path}')
        rep = '/' + self.disp_subdir + '/'
        body = left_path.replace(n, rep, 1)
        return osp.splitext(body)[0] + self.disp_suffix

    def load_data_list(self) -> List[dict]:
        mono = super().load_data_list()
        out: List[dict] = []
        for info in mono:
            left = info['img_path']
            try:
                right = self._left_to_right(left)
                disp = self._left_to_disp(left)
            except ValueError as e:
                print_log(
                    f'跳过样本（路径规则不匹配）: {left}\n{e}',
                    logger='current',
                    level='WARNING')
                continue
            out.append(
                dict(
                    left_img_path=left,
                    right_img_path=right,
                    seg_map_path=info['seg_map_path'],
                    left_disp_path=disp,
                    mask_path=None,
                    label_map=info['label_map'],
                    reduce_zero_label=info['reduce_zero_label'],
                    seg_fields=[]))
        if self.test_mode:
            print_log(
                f'双目测试集 有效样本 {len(out)} / 单目枚举 {len(mono)}',
                logger='current')
        else:
            print_log(
                f'双目训练集 有效样本 {len(out)} / 单目枚举 {len(mono)}',
                logger='current')
        return out
