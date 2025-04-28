# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmengine.fileio as fileio

from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset
from typing import Callable, Dict, List

@DATASETS.register_module()
class LDPerceptionSegDataset(BaseSegDataset):
    """乐动割草机分割数据集

    Args:
        split (str): Split txt file for LDMower.
    """
    METAINFO = dict(
        classes=('background', 'grass', 'water', 'soil', 'animal'),
        palette=[[128, 0, 128], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 0]]
    )
    def __init__(self,
                 ann_file,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 seg_water_and_animal=False,
                 **kwargs) -> None:
        self.seg_water_and_animal = seg_water_and_animal
        if not seg_water_and_animal:
            self.METAINFO['classes'] = ('background', 'grass', 'soil')
            self.METAINFO['palette'] = [[128, 0, 128], [0, 255, 0], [255, 255, 0]]
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            ann_file=ann_file,
            **kwargs)

    # 加载yolo风格的数据集
    def load_data_list(self) -> List[dict]:
        data_list = []
        if isinstance(self.ann_file, list):
            ann_files = self.ann_file
        elif osp.isfile(self.ann_file):
            ann_files = [self.ann_file]
        else:
            raise ValueError(f'你这鸟玩意既不是一个文件也不是一个列表，你搁这逗我玩呢？ {self.ann_file}')

        if not self.seg_water_and_animal:
            self.label_map = {
                0: 0,
                1: 1,
                2: 0,
                3: 2,
                4: 0
            }
            # 这个反向映射的作用是将移位后的泥土2还原为原来的3
            self.reverse_label_map = {0: 0, 1: 1, 2: 3}

        lines = []
        for ann_file in ann_files:
            with open(ann_file, 'r') as f:
                lines.extend([line.strip() for line in f.readlines()])
        for line in lines:
            src_image_path = line
            data_info = dict(
                img_path=src_image_path)
            data_info['seg_map_path'] = src_image_path.replace("/images/", "/labels/").replace(self.img_suffix, self.seg_map_suffix)
            data_info['label_map'] = self.label_map
            data_info['reduce_zero_label'] = self.reduce_zero_label
            data_info['seg_fields'] = []
            data_list.append(data_info)
        return data_list