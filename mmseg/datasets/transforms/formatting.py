# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import numpy as np
from mmcv.transforms import to_tensor
from mmcv.transforms.base import BaseTransform
from mmengine.structures import PixelData

from mmseg.registry import TRANSFORMS
from mmseg.structures import SegDataSample
from mmseg.structures import StereoMatchingDataSample


@TRANSFORMS.register_module()
class PackSegInputs(BaseTransform):
    """Pack the inputs data for the semantic segmentation.

    The ``img_meta`` item is always populated.  The contents of the
    ``img_meta`` dictionary depends on ``meta_keys``. By default this includes:

        - ``img_path``: filename of the image

        - ``ori_shape``: original shape of the image as a tuple (h, w, c)

        - ``img_shape``: shape of the image input to the network as a tuple \
            (h, w, c).  Note that images may be zero padded on the \
            bottom/right if the batch tensor is larger than this shape.

        - ``pad_shape``: shape of padded images

        - ``scale_factor``: a float indicating the preprocessing scale

        - ``flip``: a boolean indicating if image flip transform was used

        - ``flip_direction``: the flipping direction

    Args:
        meta_keys (Sequence[str], optional): Meta keys to be packed from
            ``SegDataSample`` and collected in ``data[img_metas]``.
            Default: ``('img_path', 'ori_shape',
            'img_shape', 'pad_shape', 'scale_factor', 'flip',
            'flip_direction')``
    """

    def __init__(self,
                 meta_keys=('img_path', 'seg_map_path', 'ori_shape',
                            'img_shape', 'pad_shape', 'scale_factor', 'flip',
                            'flip_direction', 'reduce_zero_label')):
        self.meta_keys = meta_keys

    def transform(self, results: dict) -> dict:
        """Method to pack the input data.

        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict:

            - 'inputs' (obj:`torch.Tensor`): The forward data of models.
            - 'data_sample' (obj:`SegDataSample`): The annotation info of the
                sample.
        """
        packed_results = dict()
        if 'img' in results:
            img = results['img']
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            if not img.flags.c_contiguous:
                img = to_tensor(np.ascontiguousarray(img.transpose(2, 0, 1)))
            else:
                img = img.transpose(2, 0, 1)
                img = to_tensor(img).contiguous()
            packed_results['inputs'] = img

        data_sample = SegDataSample()
        if 'gt_seg_map' in results:
            if len(results['gt_seg_map'].shape) == 2:
                data = to_tensor(results['gt_seg_map'][None,
                                                       ...].astype(np.int64))
            else:
                warnings.warn('Please pay attention your ground truth '
                              'segmentation map, usually the segmentation '
                              'map is 2D, but got '
                              f'{results["gt_seg_map"].shape}')
                if len(results['gt_seg_map'].shape) == 3:
                    warnings.warn('3通道 视为普通图像处理')
                    data = to_tensor(results['gt_seg_map'].transpose(2, 0, 1).astype(np.int64))
                else:
                    raise ValueError(f'3通道我都忍你了，还搞个 {results["gt_seg_map"].shape} 差不多得了')
            gt_sem_seg_data = dict(data=data)
            data_sample.gt_sem_seg = PixelData(**gt_sem_seg_data)

        if 'gt_edge_map' in results:
            gt_edge_data = dict(
                data=to_tensor(results['gt_edge_map'][None,
                                                      ...].astype(np.int64)))
            data_sample.set_data(dict(gt_edge_map=PixelData(**gt_edge_data)))

        if 'gt_depth_map' in results:
            gt_depth_data = dict(
                data=to_tensor(results['gt_depth_map'][None, ...]))
            data_sample.set_data(dict(gt_depth_map=PixelData(**gt_depth_data)))

        img_meta = {}
        for key in self.meta_keys:
            if key in results:
                img_meta[key] = results[key]
        data_sample.set_metainfo(img_meta)
        packed_results['data_samples'] = data_sample

        return packed_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(meta_keys={self.meta_keys})'
        return repr_str



@TRANSFORMS.register_module()
class PackStereoMatchingInputs(BaseTransform):
    """打包双目匹配任务的输入数据。

    img_meta 字典内容依赖于 meta_keys，默认包括：

        - ``left_img_path``: 左图文件路径
        - ``right_img_path``: 右图文件路径
        - ``ori_shape``: 原始图像尺寸 (h, w, c)
        - ``img_shape``: 网络输入图像尺寸 (h, w, c)
        - ``pad_shape``: 填充后图像尺寸
        - ``scale_factor``: 预处理缩放因子
        - ``flip``: 是否翻转
        - ``flip_direction``: 翻转方向

    Args:
        meta_keys (Sequence[str], optional): 需要从 results 中收集的元信息键。
            默认: ('left_img_path', 'right_img_path', 'disp_map_path', 'ori_shape',
                  'img_shape', 'pad_shape', 'scale_factor', 'flip',
                  'flip_direction')
    """
    def __init__(self,
                 meta_keys=('left_img_path', 'right_img_path', 'disp_map_path', 'ori_shape',
                            'img_shape', 'pad_shape', 'scale_factor', 'flip',
                            'flip_direction', 'reduce_zero_label')):
        self.meta_keys = meta_keys

    def transform(self, results: dict) -> dict:
        """Method to pack the input data.

        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict:

            - 'inputs' (obj:`torch.Tensor`): The forward data of models.
            - 'data_sample' (obj:`SegDataSample`): The annotation info of the
                sample.
        """
        packed_results = dict()
        assert 'img' not in results, "截至执行打包之前 img 不能在results里"
        left_img = results['left_img']
        right_img = results['right_img']
        if len(left_img.shape) == 3:
            results['img'] = np.concatenate([left_img, right_img], axis=2)
        elif len(left_img.shape) == 2:
            results['img'] = np.concatenate([left_img[..., np.newaxis], right_img[..., np.newaxis]], axis=2)
        else:
            raise ValueError(f'left_img 和 right_img 的形状不合法: {left_img.shape} 和 {right_img.shape}')
        del results['left_img']
        del results['right_img']

        if 'img' in results:
            img = results['img']
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            if not img.flags.c_contiguous:
                img = to_tensor(np.ascontiguousarray(img.transpose(2, 0, 1)))
            else:
                img = img.transpose(2, 0, 1)
                img = to_tensor(img).contiguous()
            packed_results['inputs'] = img

        data_sample = StereoMatchingDataSample()
        if 'left_disp' in results:
            if len(results['left_disp'].shape) == 2:
                data = to_tensor(results['left_disp'][None,
                                                       ...])
            else:
                warnings.warn('Please pay attention your ground truth '
                              'segmentation map, usually the segmentation '
                              'map is 2D, but got '
                              f'{results["left_disp"].shape}')
                if len(results['left_disp'].shape) == 3:
                    warnings.warn('3通道 视为普通图像处理')
                    data = to_tensor(results['left_disp'].transpose(2, 0, 1).astype(np.int64))
                else:
                    raise ValueError(f'3通道我都忍你了，还搞个 {results["left_disp"].shape} 差不多得了')
            gt_disp_data = dict(data=data)
            data_sample.gt_disp = PixelData(**gt_disp_data)

        if 'disp_mask' in results:
            gt_disp_mask_data = dict(
                data=to_tensor(results['disp_mask'][None,
                                                      ...].astype(np.int64)))
            data_sample.set_data(dict(disp_mask=PixelData(**gt_disp_mask_data)))

        img_meta = {}
        for key in self.meta_keys:
            if key in results:
                img_meta[key] = results[key]
        data_sample.set_metainfo(img_meta)
        packed_results['data_samples'] = data_sample

        return packed_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(meta_keys={self.meta_keys})'
        return repr_str


@TRANSFORMS.register_module()
class PackSegStereoInputs(BaseTransform):
    """打包 SegStereo 联合训练所需数据：6 通道输入 + ``gt_sem_seg`` + ``gt_disp``。

    需在 pipeline 中先加载 ``left_img``、``right_img``、``gt_seg_map``、``left_disp`` 等，
    与 ``LoadStereoImages`` / ``LoadAnnotations`` / ``LoadStereoMatchingAnnotations`` 等组合使用。

    - ``inputs``：左右图在通道维拼接为 6 通道 ``(6, H, W)``。
    - ``data_samples``：``SegDataSample``，含 ``gt_sem_seg``；若存在 ``left_disp`` 则写入 ``gt_disp``。
    """

    def __init__(self,
                 meta_keys=('left_img_path', 'right_img_path', 'seg_map_path',
                            'left_disp_path', 'disp_map_path', 'ori_shape',
                            'img_shape', 'pad_shape', 'scale_factor', 'flip',
                            'flip_direction', 'reduce_zero_label')):
        self.meta_keys = meta_keys

    def transform(self, results: dict) -> dict:
        packed_results = dict()
        assert 'img' not in results, '执行打包前 results 中不应含 img'
        left_img = results['left_img']
        right_img = results['right_img']
        if len(left_img.shape) == 3:
            results['img'] = np.concatenate([left_img, right_img], axis=2)
        elif len(left_img.shape) == 2:
            results['img'] = np.concatenate(
                [left_img[..., np.newaxis], right_img[..., np.newaxis]], axis=2)
        else:
            raise ValueError(
                f'left_img/right_img 形状非法: {left_img.shape}, {right_img.shape}')
        del results['left_img']
        del results['right_img']

        img = results['img']
        if len(img.shape) < 3:
            img = np.expand_dims(img, -1)
        if not img.flags.c_contiguous:
            img = to_tensor(np.ascontiguousarray(img.transpose(2, 0, 1)))
        else:
            img = img.transpose(2, 0, 1)
            img = to_tensor(img).contiguous()
        packed_results['inputs'] = img

        data_sample = SegDataSample()
        if 'gt_seg_map' in results:
            if len(results['gt_seg_map'].shape) == 2:
                data = to_tensor(results['gt_seg_map'][None,
                                                       ...].astype(np.int64))
            else:
                warnings.warn(
                    '语义分割图通常为 2D，当前为 '
                    f'{results["gt_seg_map"].shape}')
                if len(results['gt_seg_map'].shape) == 3:
                    data = to_tensor(
                        results['gt_seg_map'].transpose(2, 0,
                                                        1).astype(np.int64))
                else:
                    raise ValueError(f'不支持的 gt_seg_map 形状: '
                                     f'{results["gt_seg_map"].shape}')
            data_sample.gt_sem_seg = PixelData(data=data)

        if 'left_disp' in results:
            if len(results['left_disp'].shape) == 2:
                disp_t = to_tensor(results['left_disp'][None, ...].astype(
                    np.float32))
            else:
                disp_t = to_tensor(results['left_disp'][None, ...])
            data_sample.gt_disp = PixelData(data=disp_t)

        if 'disp_mask' in results:
            m = to_tensor(results['disp_mask'][None, ...].astype(np.int64))
            data_sample.set_data(dict(disp_mask=PixelData(data=m)))

        img_meta = {}
        for key in self.meta_keys:
            if key in results:
                img_meta[key] = results[key]
        data_sample.set_metainfo(img_meta)
        packed_results['data_samples'] = data_sample

        return packed_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(meta_keys={self.meta_keys})'
        return repr_str
