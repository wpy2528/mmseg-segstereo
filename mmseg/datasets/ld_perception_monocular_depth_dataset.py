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

PALETTE = [[128, 0, 128], [0, 255, 0], [0, 255, 255], [0, 0, 255]]
@DATASETS.register_module()
class LDPerceptionMonocularDepthDataset(BaseSegDataset):
    """乐动割草机单目深度数据集

    Args:
        split (str): Split txt file for LDMower.
    """
    METAINFO = dict(
        classes = ('background', 'grass', 'soil'),
        palette = PALETTE
    )
    def __init__(self,
                 data_root,
                 class_names,
                 ann_file=None,
                 img_suffix='.jpg',
                 depth_suffix='.png',
                 include=None,
                 repeat=None,
                 exclude=None,
                 **kwargs) -> None:
        assert isinstance(class_names, (tuple, list)), type(class_names)
        LDPerceptionMonocularDepthDataset.METAINFO = dict(
            classes = class_names,
            palette = PALETTE
        )
        # include和exclude不能同时存在，如果非None，则必须为list
        assert include is None or exclude is None, "include和exclude不能同时存在"
        if include is not None:
            assert isinstance(include, list), "include必须为list"
            include = ['/' + e.rstrip('/').lstrip('/') + '/' if (not e.endswith('.txt')) else e for e in include]
        if exclude is not None:
            assert isinstance(exclude, list), "exclude必须为list"
            exclude = ['/' + e.rstrip('/').lstrip('/') + '/' if (not e.endswith('.txt')) else e for e in exclude]
        if repeat is not None:
            assert isinstance(repeat, dict), "repeat必须为int"
            r_ = {}
            for key, value in repeat.items():
                if not key.endswith('.txt'):
                    r_['/' + key.rstrip('/').lstrip('/') + '/'] = value
                else:
                    r_[key] = value
            self.repeat = r_
        self.include = include
        self.exclude = exclude
        self.repeat = repeat
        print(f"include: {self.include}, exclude: {self.exclude}, repeat: {self.repeat}")
        
        super().__init__(
            img_suffix=img_suffix,
            depth_suffix=depth_suffix,
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

        # 进行数据集增删时，允许给定的关键字是数据路径中的子串，也可以是txt文件路径。
        # 如果是子串，那么对包含子串的数据路径进行处理；如果是txt文件路径，那么直接读取txt中的内容，对这些内容进行对应的处理。
        if self.include is not None:
            txt_paths = [e for e in self.include if e.endswith('.txt')]
            keywords = [e for e in self.include if not e.endswith('.txt')]
            lines = [line for line in lines if any(include in line for include in keywords)]
            for txt_path in txt_paths:
                with open(txt_path, 'r') as f:
                    lines.extend([line.strip() for line in f.readlines()])
        if self.exclude is not None:
            txt_paths = [e for e in self.exclude if e.endswith('.txt')]
            keywords = [e for e in self.exclude if not e.endswith('.txt')]
            lines = set([line for line in lines if not any(exclude in line for exclude in keywords)])
            exclude_paths = set()
            for txt_path in txt_paths:
                with open(txt_path, 'r') as f:
                    exclude_paths.update([line.strip() for line in f.readlines()])
            # 这里需要转set加速筛除
            lines = list(set(lines) - exclude_paths)
        if self.repeat is not None:
            # 根据repeat的dict，对lines进行重复
            repeat_lines = []
            for line in lines:
                for key, value in self.repeat.items():
                    if (not key.endswith('.txt')) and (key in line):
                        repeat_lines.extend([line] * value)
                        break
            txt_paths = [e for e in self.repeat.keys() if e.endswith('.txt')]
            for txt_path in txt_paths:
                with open(txt_path, 'r') as f:
                    repeat_lines.extend([line.strip() for line in f.readlines()] * self.repeat[txt_path])
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
            data_info['depth_path'] = src_image_path.replace("/images/", "/depth/").replace(self.img_suffix, self.depth_suffix)
            data_list.append(data_info)
        
        if self.test_mode:
            print_log(f"测试集 共计 {len(data_list)} 个样本", logger="current")
        else:
            print_log(f"训练集 共计 {len(data_list)} 个样本", logger="current")
        time.sleep(1)
        return data_list
