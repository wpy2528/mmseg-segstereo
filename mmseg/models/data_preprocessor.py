# Copyright (c) OpenMMLab. All rights reserved.
from numbers import Number
from typing import Any, Dict, List, Optional, Sequence

import torch
from mmengine.model import BaseDataPreprocessor

from mmseg.registry import MODELS
from mmseg.utils import stack_batch


@MODELS.register_module()
class SegDataPreProcessor(BaseDataPreprocessor):
    """Image pre-processor for segmentation tasks.

    Comparing with the :class:`mmengine.ImgDataPreprocessor`,

    1. It won't do normalization if ``mean`` is not specified.
    2. It does normalization and color space conversion after stacking batch.
    3. It supports batch augmentations like mixup and cutmix.


    It provides the data pre-processing as follows

    - Collate and move data to the target device.
    - Pad inputs to the input size with defined ``pad_val``, and pad seg map
        with defined ``seg_pad_val``.
    - Stack inputs to batch_inputs.
    - Convert inputs from bgr to rgb if the shape of input is (3, H, W).
    - Normalize image with defined std and mean.
    - Do batch augmentations like Mixup and Cutmix during training.

    Args:
        mean (Sequence[Number], optional): The pixel mean of R, G, B channels.
            Defaults to None.
        std (Sequence[Number], optional): The pixel standard deviation of
            R, G, B channels. Defaults to None.
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value. Default: 0.
        seg_pad_val (float, optional): Padding value of segmentation map.
            Default: 255.
        padding_mode (str): Type of padding. Default: constant.
            - constant: pads with a constant value, this value is specified
              with pad_val.
        bgr_to_rgb (bool): whether to convert image from BGR to RGB.
            Defaults to False.
        rgb_to_bgr (bool): whether to convert image from RGB to RGB.
            Defaults to False.
        batch_augments (list[dict], optional): Batch-level augmentations
        test_cfg (dict, optional): The padding size config in testing, if not
            specify, will use `size` and `size_divisor` params as default.
            Defaults to None, only supports keys `size` or `size_divisor`.
    """

    def __init__(
        self,
        mean: Sequence[Number] = None,
        std: Sequence[Number] = None,
        size: Optional[tuple] = None,
        size_divisor: Optional[int] = None,
        pad_val: Number = 0,
        seg_pad_val: Number = 255,
        bgr_to_rgb: bool = False,
        rgb_to_bgr: bool = False,
        batch_augments: Optional[List[dict]] = None,
        test_cfg: dict = None,
    ):
        super().__init__()
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        self.seg_pad_val = seg_pad_val

        assert not (bgr_to_rgb and rgb_to_bgr), (
            '`bgr2rgb` and `rgb2bgr` cannot be set to True at the same time')
        self.channel_conversion = rgb_to_bgr or bgr_to_rgb

        if mean is not None:
            assert std is not None, 'To enable the normalization in ' \
                                    'preprocessing, please specify both ' \
                                    '`mean` and `std`.'
            # Enable the normalization in preprocessing.
            self._enable_normalize = True
            self.register_buffer('mean',
                                 torch.tensor(mean).view(-1, 1, 1), False)
            self.register_buffer('std',
                                 torch.tensor(std).view(-1, 1, 1), False)
        else:
            self._enable_normalize = False

        # TODO: support batch augmentations.
        self.batch_augments = batch_augments

        # Support different padding methods in testing
        self.test_cfg = test_cfg

    def forward(self, data: dict, training: bool = False) -> Dict[str, Any]:
        """Perform normalization、padding and bgr2rgb conversion based on
        ``BaseDataPreProcessor``.

        Args:
            data (dict): data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.

        Returns:
            Dict: Data in the same format as the model input.
        """
        data = self.cast_data(data)  # type: ignore
        inputs = data['inputs']
        data_samples = data.get('data_samples', None)
        # TODO: whether normalize should be after stack_batch
        if self.channel_conversion and inputs[0].size(0) == 3:
            inputs = [_input[[2, 1, 0], ...] for _input in inputs]

        inputs = [_input.float() for _input in inputs]
        if self._enable_normalize:
            inputs = [(_input - self.mean) / self.std for _input in inputs]

        if training:
            assert data_samples is not None, ('During training, ',
                                              '`data_samples` must be define.')
            inputs, data_samples = stack_batch(
                inputs=inputs,
                data_samples=data_samples,
                size=self.size,
                size_divisor=self.size_divisor,
                pad_val=self.pad_val,
                seg_pad_val=self.seg_pad_val)

            if self.batch_augments is not None:
                inputs, data_samples = self.batch_augments(
                    inputs, data_samples)
        else:
            img_size = inputs[0].shape[1:]
            assert all(input_.shape[1:] == img_size for input_ in inputs),  \
                'The image size in a batch should be the same.'
            # pad images when testing
            if self.test_cfg:
                inputs, padded_samples = stack_batch(
                    inputs=inputs,
                    size=self.test_cfg.get('size', None),
                    size_divisor=self.test_cfg.get('size_divisor', None),
                    pad_val=self.pad_val,
                    seg_pad_val=self.seg_pad_val)
                for data_sample, pad_info in zip(data_samples, padded_samples):
                    data_sample.set_metainfo({**pad_info})
            else:
                inputs = torch.stack(inputs, dim=0)

        return dict(inputs=inputs, data_samples=data_samples)


@MODELS.register_module()
class StereoDataPreProcessor(BaseDataPreprocessor):
    """立体视觉数据预处理器，用于处理双目图像数据。

    相比 :class:`SegDataPreProcessor`，这个预处理器专门处理立体视觉数据：

    1. 支持左右目图像的分别处理
    2. 保持左右目图像的空间对应关系
    3. 支持立体视觉特有的数据增强
    4. 处理视差图（disparity map）的预处理

    提供的数据预处理功能：

    - 将数据移动到目标设备
    - 对左右目图像进行填充到指定尺寸
    - 对分割图和视差图进行相应的填充
    - 将输入堆叠成批次
    - 进行颜色空间转换（BGR到RGB）
    - 使用指定的均值和标准差进行归一化
    - 支持立体视觉特有的批次增强

    Args:
        mean (Sequence[Number], optional): R, G, B通道的像素均值。
            Defaults to None.
        std (Sequence[Number], optional): R, G, B通道的像素标准差。
            Defaults to None.
        size (tuple, optional): 固定的填充尺寸。
        size_divisor (int, optional): 填充尺寸的除数。
        pad_val (float, optional): 填充值。Default: 0.
        seg_pad_val (float, optional): 分割图的填充值。Default: 255.
        disp_pad_val (float, optional): 视差图的填充值。Default: 0.
        bgr_to_rgb (bool): 是否将图像从BGR转换为RGB。
            Defaults to False.
        rgb_to_bgr (bool): 是否将图像从RGB转换为BGR。
            Defaults to False.
        batch_augments (list[dict], optional): 批次级增强
        test_cfg (dict, optional): 测试时的填充尺寸配置，如果不指定，
            将使用 `size` 和 `size_divisor` 参数作为默认值。
            Defaults to None，仅支持 `size` 或 `size_divisor` 键。
        stereo_mode (str): 立体视觉模式，支持 'left_right'（左右目分别）
            或 'concat'（左右目拼接）。Defaults to 'left_right'.
    """

    def __init__(
        self,
        mean: Sequence[Number] = None,
        std: Sequence[Number] = None,
        size: Optional[tuple] = None,
        size_divisor: Optional[int] = None,
        pad_val: Number = 0,
        seg_pad_val: Number = 255,
        disp_pad_val: Number = 0,
        bgr_to_rgb: bool = False,
        rgb_to_bgr: bool = False,
        batch_augments: Optional[List[dict]] = None,
        test_cfg: dict = None,
        stereo_mode: str = 'left_right',
    ):
        super().__init__()
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        self.seg_pad_val = seg_pad_val
        self.disp_pad_val = disp_pad_val
        self.stereo_mode = stereo_mode

        assert stereo_mode in ['left_right', 'concat'], \
            f'stereo_mode must be "left_right" or "concat", got {stereo_mode}'

        assert not (bgr_to_rgb and rgb_to_bgr), (
            '`bgr2rgb` and `rgb2bgr` cannot be set to True at the same time')
        self.channel_conversion = rgb_to_bgr or bgr_to_rgb

        if mean is not None:
            assert std is not None, 'To enable the normalization in ' \
                                    'preprocessing, please specify both ' \
                                    '`mean` and `std`.'
            # Enable the normalization in preprocessing.
            self._enable_normalize = True
            self.register_buffer('mean',
                                 torch.tensor(mean).view(-1, 1, 1), False)
            self.register_buffer('std',
                                 torch.tensor(std).view(-1, 1, 1), False)
        else:
            self._enable_normalize = False

        # TODO: support batch augmentations.
        self.batch_augments = batch_augments

        # Support different padding methods in testing
        self.test_cfg = test_cfg

    def _process_stereo_inputs(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """处理立体视觉输入，根据模式进行相应处理。

        Args:
            inputs (List[torch.Tensor]): 输入张量列表

        Returns:
            List[torch.Tensor]: 处理后的输入张量列表
        """
        if self.stereo_mode == 'left_right':
            # 左右目分别处理，保持原有格式
            return inputs
        elif self.stereo_mode == 'concat':
            # 将左右目图像在通道维度拼接
            processed_inputs = []
            for i in range(0, len(inputs), 2):
                if i + 1 < len(inputs):
                    # 拼接左右目图像
                    left_img = inputs[i]
                    right_img = inputs[i + 1]
                    concat_img = torch.cat([left_img, right_img], dim=0)
                    processed_inputs.append(concat_img)
                else:
                    # 如果只有左目图像，用零填充右目
                    left_img = inputs[i]
                    right_img = torch.zeros_like(left_img)
                    concat_img = torch.cat([left_img, right_img], dim=0)
                    processed_inputs.append(concat_img)
            return processed_inputs
        else:
            raise ValueError(f'Unsupported stereo_mode: {self.stereo_mode}')

    def forward(self, data: dict, training: bool = False) -> Dict[str, Any]:
        """基于 ``BaseDataPreProcessor`` 执行归一化、填充和颜色空间转换。

        Args:
            data (dict): 从数据加载器采样的数据。
            training (bool): 是否启用训练时增强。

        Returns:
            Dict: 与模型输入相同格式的数据。
        """
        data = self.cast_data(data)  # type: ignore
        inputs = data['inputs']
        data_samples = data.get('data_samples', None)

        # 颜色空间转换
        if self.channel_conversion and inputs[0].size(0) == 3:
            inputs = [_input[[2, 1, 0], ...] for _input in inputs]

        # 转换为浮点数
        inputs = [_input.float() for _input in inputs]

        # 归一化
        if self._enable_normalize:
            inputs = [(_input - self.mean) / self.std for _input in inputs]

        # 立体视觉输入处理
        inputs = self._process_stereo_inputs(inputs)

        if training:
            assert data_samples is not None, ('During training, ',
                                              '`data_samples` must be define.')
            inputs, data_samples = stack_batch(
                inputs=inputs,
                data_samples=data_samples,
                size=self.size,
                size_divisor=self.size_divisor,
                pad_val=self.pad_val,
                seg_pad_val=self.seg_pad_val)

            if self.batch_augments is not None:
                inputs, data_samples = self.batch_augments(
                    inputs, data_samples)
        else:
            img_size = inputs[0].shape[1:]
            assert all(input_.shape[1:] == img_size for input_ in inputs),  \
                'The image size in a batch should be the same.'
            # 测试时填充图像
            if self.test_cfg:
                inputs, padded_samples = stack_batch(
                    inputs=inputs,
                    size=self.test_cfg.get('size', None),
                    size_divisor=self.test_cfg.get('size_divisor', None),
                    pad_val=self.pad_val,
                    seg_pad_val=self.seg_pad_val)
                for data_sample, pad_info in zip(data_samples, padded_samples):
                    data_sample.set_metainfo({**pad_info})
            else:
                inputs = torch.stack(inputs, dim=0)

        return dict(inputs=inputs, data_samples=data_samples)

