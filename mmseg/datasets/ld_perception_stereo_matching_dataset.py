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
class LDPerceptionStereoMatchingDataset(BaseSegDataset):
    """乐动割草机双目匹配数据集

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
            LDPerceptionStereoMatchingDataset.METAINFO = dict(
                classes = ('background', 'grass', 'soil', 'animal'),
                palette = [[128, 0, 128], [0, 255, 0], [255, 255, 0], [0, 0, 255]]
            )
        else:
            LDPerceptionStereoMatchingDataset.METAINFO = dict(
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

    # 加载双目匹配数据集
    def load_data_list(self) -> List[dict]:
        data_list = []
        left_image_paths = []
        if self.ann_file is not None:
            if isinstance(self.ann_file, list):
                ann_files = self.ann_file
            elif osp.isfile(self.ann_file):
                ann_files = [self.ann_file]
            else:
                raise ValueError(f'你这鸟玩意既不是一个文件也不是一个列表，你搁这逗我玩呢？ {self.ann_file}')

            for ann_file in ann_files:
                with open(ann_file, 'r') as f:
                    left_image_paths.extend([line.strip() for line in f.readlines()])
        else:
            assert os.path.isdir(self.data_root), self.data_root
            left_image_paths = self.glob_all_left_image_paths(self.data_root)

        if self.include is not None:
            left_image_paths = [line for line in left_image_paths if any(include in line for include in self.include)]
        if self.exclude is not None:
            left_image_paths = [line for line in left_image_paths if not any(exclude in line for exclude in self.exclude)]
        if self.repeat is not None:
            # 根据repeat的dict，对lines进行重复
            repeat_lines = []
            for line in left_image_paths:
                for key, value in self.repeat.items():
                    if key in line:
                        repeat_lines.extend([line] * value)
                        break

            left_image_paths += repeat_lines

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
            for line in left_image_paths:
                f.write(line + "\n")
            f.close()
            
        for line in left_image_paths:
            left_img_path, right_img_path, left_disp_path = self.parse_right_and_disp_paths_by_left_path(line)
            data_info = dict(
                left_img_path=left_img_path,
                right_img_path=right_img_path,
                left_disp_path=left_disp_path
            )
            data_info['left_disp_path'] = left_disp_path
            data_info['label_map'] = None
            data_info['reduce_zero_label'] = False
            data_info['seg_fields'] = []
            data_list.append(data_info)
        
        if self.test_mode:
            print_log(f"测试集 共计 {len(data_list)} 个样本", logger="current")
        else:
            print_log(f"训练集 共计 {len(data_list)} 个样本", logger="current")
        assert len(data_list) > 0, "数据集为空"
        time.sleep(1)
        return data_list
    
    def glob_all_left_image_paths(self, data_root: str) -> List[str]:
        if "flyingthings3d" in data_root:
            left_img_paths = glob.glob(os.path.join(data_root, "**", "left", "*.png"), recursive=True)
        elif "driving" in data_root:
            left_img_paths = glob.glob(os.path.join(data_root, "**","**","**", "**","left", "*.png"), recursive=True)
        elif "monkaa" in data_root:
            left_img_paths = glob.glob(os.path.join(data_root, "**","**", "left","*.png"), recursive=True)
        elif "ETH3d" in data_root:
            left_img_paths = glob.glob(os.path.join(data_root, "**", "**","im0.png"), recursive=True)
        elif "kitti" in data_root:
            left_img_paths = glob.glob(os.path.join(data_root, "raw_img", "**","**","image_02","data", "*.jpg"), recursive=True)
        elif "HR-VS" in data_root:
            left_img_paths = glob.glob(os.path.join(data_root, "**","trainingF", "**","im0.png"), recursive=True)
        elif "instereo_2k" in data_root:
            left_img_paths = glob.glob(os.path.join(data_root, "**", "**", "left.png"), recursive=True)
        elif "crestereo" in data_root:
            left_img_paths = glob.glob(os.path.join(data_root, "**", "*_left.jpg"), recursive=True)
        elif "falling_things" in data_root:
            left_img_paths = glob.glob(os.path.join(data_root, "**", "*.left.jpg"), recursive=True)
        elif "generate_isaac" in data_root:
            left_img_paths = glob.glob(os.path.join(data_root, "**", "Replicator", "**", "rgb_*.png"), recursive=True)
        else:
            raise ValueError(f"不支持的数据集: {data_root}")
        return left_img_paths
    
    def parse_right_and_disp_paths_by_left_path(self, left_img_path: str) -> tuple:
        if "flyingthings3d" in left_img_path:
            right_img_path = left_img_path.replace("/left/", "/right/")
            left_disp_path = left_img_path.replace("/images/", "/disparity/").replace(".png", ".pfm")
        elif "driving" in left_img_path:
            right_img_path = left_img_path.replace("/left/", "/right/")
            left_disp_path = left_img_path.replace("/images/", "/disparity/").replace(".png", ".pfm")
        elif "monkaa" in left_img_path:
            right_img_path = left_img_path.replace("/left/", "/right/")
            left_disp_path = left_img_path.replace("/images/", "/disparity/").replace(".png", ".pfm")
        elif "ETH3d" in left_img_path:
            right_img_path = left_img_path.replace("im0.png", "im1.png")
            left_disp_path = left_img_path.replace("test", "ground_truth").replace("train", "ground_truth").replace("im0.png", "disp0GT.pfm")
        elif "kitti" in left_img_path:
            right_img_path = left_img_path.replace("image_02", "image_03")
            left_disp_path = left_img_path.replace("raw_img", "eigen_disp").replace(".jpg", ".png")
        elif "HR-VS" in left_img_path:
            right_img_path = left_img_path.replace("im0.png", "im1.png")
            left_disp_path = left_img_path.replace("im0.png", "disp0GT.pfm")
        elif "instereo_2k" in left_img_path:
            right_img_path = left_img_path.replace("left.png", "right.png")
            left_disp_path = left_img_path.replace("left.png", "left_disp.png")
        elif "crestereo/hole" in left_img_path:
            right_img_path = left_img_path.replace("_left.jpg", "_right.jpg")
            left_disp_path = left_img_path.replace("_left.jpg", "_left.disp.png")
        elif "falling_things" in left_img_path:
            right_img_path = left_img_path.replace(".left.jpg", ".right.jpg")
            left_disp_path = left_img_path.replace(".left.jpg", ".left.depth.png")
        elif "generate_isaac" in left_img_path:
            right_img_path = left_img_path.replace("Replicator", "Replicator_01")
            depth_path = left_img_path.replace("/rgb/", "/distance_to_image_plane/").replace(".png", ".npy")
            assert 0, "待计算视差"
        else:
            raise ValueError(f"不支持的数据集: {left_img_path}")
        
        # print(f"left: {left_img_path}, right: {right_img_path}, disp: {left_disp_path}")
        return left_img_path, right_img_path, left_disp_path


if __name__ == "__main__":
    dataset = LDPerceptionStereoMatchingDataset(
        data_root="stereo_datasets/SceneFlow_driving",
        ann_file="stereo_datasets/SceneFlow_driving/train.txt",
        num_classes=3,
        test_mode=True
    )
    data_list = dataset.load_data_list()
    print(data_list)