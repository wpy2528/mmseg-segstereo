# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.structures import BaseDataElement, PixelData


class StereoMatchingDataSample(BaseDataElement):
    """A data structure interface for stereo matching tasks. They are used as
    interfaces between different components.

    The attributes in ``StereoMatchingDataSample`` are divided into several parts:

        - ``left_img``(PixelData): Left image data.
        - ``right_img``(PixelData): Right image data.
        - ``gt_disp``(PixelData): Ground truth disparity map.
        - ``pred_disp``(PixelData): Predicted disparity map.
        - ``disp_logits``(PixelData): Predicted logits of disparity.
        - ``disp_mask``(PixelData): Mask for valid disparity regions.
        - ``left_disp``(PixelData): Left disparity map (alias for gt_disp).
        - ``right_disp``(PixelData): Right disparity map (if available).

    Examples:
         >>> import torch
         >>> import numpy as np
         >>> from mmengine.structures import PixelData
         >>> from mmseg.structures import StereoMatchingDataSample

         >>> data_sample = StereoMatchingDataSample()
         >>> img_meta = dict(img_shape=(384, 512, 3),
         ...                 pad_shape=(384, 512, 3))
         >>> left_img_data = PixelData(metainfo=img_meta)
         >>> left_img_data.data = torch.rand(3, 384, 512)
         >>> data_sample.left_img = left_img_data
         >>> assert 'img_shape' in data_sample.left_img.metainfo_keys()
         >>> data_sample.left_img.shape
         (3, 384, 512)
         >>> print(data_sample)
        <StereoMatchingDataSample(

            META INFORMATION

            DATA FIELDS
            left_img: <PixelData(

                    META INFORMATION
                    img_shape: (384, 512, 3)
                    pad_shape: (384, 512, 3)

                    DATA FIELDS
                    data: tensor([[[0.1234, 0.5678, ...],
                                 [0.9012, 0.3456, ...],
                                 ...]])
                ) at 0x1c2b4156460>
        ) at 0x1c2aae44d60>

        >>> data_sample = StereoMatchingDataSample()
        >>> gt_disp_data = dict(disp=torch.rand(1, 384, 512))
        >>> gt_disp = PixelData(**gt_disp_data)
        >>> data_sample.gt_disp = gt_disp
        >>> assert 'gt_disp' in data_sample
        >>> assert 'disp' in data_sample.gt_disp
    """

    @property
    def left_img(self) -> PixelData:
        return self._left_img

    @left_img.setter
    def left_img(self, value: PixelData) -> None:
        self.set_field(value, '_left_img', dtype=PixelData)

    @left_img.deleter
    def left_img(self) -> None:
        del self._left_img

    @property
    def right_img(self) -> PixelData:
        return self._right_img

    @right_img.setter
    def right_img(self, value: PixelData) -> None:
        self.set_field(value, '_right_img', dtype=PixelData)

    @right_img.deleter
    def right_img(self) -> None:
        del self._right_img

    @property
    def gt_disp(self) -> PixelData:
        return self._gt_disp

    @gt_disp.setter
    def gt_disp(self, value: PixelData) -> None:
        self.set_field(value, '_gt_disp', dtype=PixelData)

    @gt_disp.deleter
    def gt_disp(self) -> None:
        del self._gt_disp

    @property
    def pred_disp(self) -> PixelData:
        return self._pred_disp

    @pred_disp.setter
    def pred_disp(self, value: PixelData) -> None:
        self.set_field(value, '_pred_disp', dtype=PixelData)

    @pred_disp.deleter
    def pred_disp(self) -> None:
        del self._pred_disp

    @property
    def disp_logits(self) -> PixelData:
        return self._disp_logits

    @disp_logits.setter
    def disp_logits(self, value: PixelData) -> None:
        self.set_field(value, '_disp_logits', dtype=PixelData)

    @disp_logits.deleter
    def disp_logits(self) -> None:
        del self._disp_logits

    @property
    def disp_mask(self) -> PixelData:
        return self._disp_mask

    @disp_mask.setter
    def disp_mask(self, value: PixelData) -> None:
        self.set_field(value, '_disp_mask', dtype=PixelData)

    @disp_mask.deleter
    def disp_mask(self) -> None:
        del self._disp_mask

    @property
    def left_disp(self) -> PixelData:
        """Alias for gt_disp, commonly used in stereo matching literature."""
        return self._gt_disp

    @left_disp.setter
    def left_disp(self, value: PixelData) -> None:
        self.set_field(value, '_gt_disp', dtype=PixelData)

    @left_disp.deleter
    def left_disp(self) -> None:
        del self._gt_disp

    @property
    def right_disp(self) -> PixelData:
        return self._right_disp

    @right_disp.setter
    def right_disp(self, value: PixelData) -> None:
        self.set_field(value, '_right_disp', dtype=PixelData)

    @right_disp.deleter
    def right_disp(self) -> None:
        del self._right_disp 