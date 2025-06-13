# Copyright (c) OpenMMLab. All rights reserved.
import sys
import glob
import time
import os
import os.path as osp

from typing import List

from mmengine.logging import MMLogger, print_log

from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset

@DATASETS.register_module()
class LDPerceptionSegDataset(BaseSegDataset):
    """乐动割草机分割数据集

    Args:
        split (str): Split txt file for LDMower.
    """
    METAINFO = dict(
        classes = ('background', 'grass', 'soil'),
        palette = [[128, 0, 128], [0, 255, 0], [255, 255, 0]]
    )
    def __init__(self,
                 data_root,
                 num_classes=3,
                 ann_file=None,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 include=None,
                 repeat=None,
                 exclude=None,
                 **kwargs) -> None:
        if num_classes == 4:
            LDPerceptionSegDataset.METAINFO = dict(
                classes = ('background', 'grass', 'soil', 'animal'),
                palette = [[128, 0, 128], [0, 255, 0], [255, 255, 0], [0, 0, 255]]
            )
        else:
            LDPerceptionSegDataset.METAINFO = dict(
                classes = ('background', 'grass', 'soil'),
                palette = [[128, 0, 128], [0, 255, 0], [255, 255, 0]]
            )
        # include和exclude不能同时存在，如果非None，则必须为list
        assert include is None or exclude is None, "include和exclude不能同时存在"
        if include is not None:
            assert isinstance(include, list), "include必须为list"
            include = ['/' + e.rstrip('/').lstrip('/') + '/' for e in include]
        if exclude is not None:
            assert isinstance(exclude, list), "exclude必须为list"
            exclude = ['/' + e.rstrip('/').lstrip('/') + '/' for e in exclude]
        if repeat is not None:
            assert isinstance(repeat, dict), "repeat必须为int"
            self.repeat = repeat
        self.include = include
        self.exclude = exclude
        self.repeat = repeat
        print(f"include: {self.include}, exclude: {self.exclude}, repeat: {self.repeat}")
        
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            ann_file=ann_file,
            data_root=data_root,
            **kwargs)

    # 加载coco风格的数据集
    def load_data_list(self) -> List[dict]:
        data_list = []
        lines = []
        if self.ann_file is not None:
            if isinstance(self.ann_file, list):
                ann_files = self.ann_file
            elif osp.isfile(self.ann_file):
                ann_files = [self.ann_file]
            else:
                raise ValueError(f'你这鸟玩意既不是一个文件也不是一个列表，你搁这逗我玩呢？ {self.ann_file}')

            for ann_file in ann_files:
                with open(ann_file, 'r') as f:
                    lines.extend([line.strip() for line in f.readlines()])
        else:
            assert os.path.isdir(self.data_root), self.data_root
            lines = glob.glob(os.path.join(self.data_root, "**", "images", "*.jpg"), recursive=True)

        if self.include is not None:
            lines = [line for line in lines if any(include in line for include in self.include)]
        if self.exclude is not None:
            lines = [line for line in lines if not any(exclude in line for exclude in self.exclude)]
        if self.repeat is not None:
            # 根据repeat的dict，对lines进行重复
            repeat_lines = []
            for line in lines:
                for key, value in self.repeat.items():
                    if key in line:
                        repeat_lines.extend([line] * value)
                        break

            lines += repeat_lines

        src_log_path = MMLogger.get_current_instance().log_file
        # 如果当前是调试模式，则不写入样本路径
        if src_log_path is not None and not sys.gettrace():
            if self.test_mode:
                src_dataset_log_path = os.path.join(os.path.dirname(src_log_path), f"val_samples.txt")
                print_log(f"测试集 样本路径写入到 {src_dataset_log_path} 中", logger="current")
            else:
                src_dataset_log_path = os.path.join(os.path.dirname(src_log_path), f"train_samples.txt")
                print_log(f"训练集 样本路径写入到 {src_dataset_log_path} 中", logger="current")
            f = open(src_dataset_log_path, "w")
            for line in lines:
                f.write(line + "\n")
            f.close()
            
        for line in lines:
            src_image_path = line
            data_info = dict(
                img_path=src_image_path)
            data_info['seg_map_path'] = src_image_path.replace("/images/", "/labels/").replace(self.img_suffix, self.seg_map_suffix)
            data_info['label_map'] = self.label_map
            data_info['reduce_zero_label'] = self.reduce_zero_label
            data_info['seg_fields'] = []
            data_list.append(data_info)
        
        if self.test_mode:
            print_log(f"测试集 共计 {len(data_list)} 个样本", logger="current")
        else:
            print_log(f"训练集 共计 {len(data_list)} 个样本", logger="current")
        time.sleep(1)
        return data_list
