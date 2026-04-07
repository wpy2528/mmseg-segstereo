# Copyright (c) OpenMMLab. All rights reserved.
from .formatting import PackSegInputs, PackSegStereoInputs
from .loading import (LoadAnnotations, LoadBiomedicalAnnotation,
                      LoadBiomedicalData, LoadBiomedicalImageFromFile,
                      LoadDepthAnnotation, LoadImageFromNDArray,
                      LoadMultipleRSImageFromFile, LoadSingleRSImageFromFile)
# yapf: disable
from .transforms import (CLAHE, AdjustGamma, Albu, BioMedical3DPad,
                         BioMedical3DRandomCrop, BioMedical3DRandomFlip,
                         BioMedicalGaussianBlur, BioMedicalGaussianNoise,
                         BioMedicalRandomGamma, ConcatCDInput, GenerateEdge,
                         PhotoMetricDistortion, RandomCrop, RandomCutOut,
                         RandomDepthMix, RandomFlip, RandomMosaic,
                         RandomRotate, RandomRotFlip, RemapSegLabels, Rerange,
                         Resize, ResizeShortestEdge, ResizeToMultiple, RGB2Gray,
                         SegRescale)
from .loading import LoadStereoImages, LoadStereoMatchingAnnotations

from .imgaug_for_mm.resize_to_front_or_side_camera_original_size import ResizeToFrontOrSideImageOriginalSize
from .imgaug_for_mm.copy_paste import CopyPasteTop
from .imgaug_for_mm.isp_stype_aug import ISPStyleAug
from .imgaug_for_mm.concat_front_and_side import ConcatFrontAndSide
from .imgaug_for_mm.crop import CropTop
from .imgaug_for_mm.image_quality_aug import ImageQualityAug
from .imgaug_for_mm.dilate_soil import DilateSoilMask
from .imgaug_for_mm.random_masking import RandomMasking

from .stereo_aug.cre_stereo_augmentor import CREStereoAugmentor
from .stereo_aug.igevpp_stereo_augmentor import FlowAugmentor, SparseFlowAugmentor
from .stereo_aug.resize_stereo_images import ResizeStereoImages
# yapf: enable
__all__ = [
    'LoadAnnotations', 'RandomCrop', 'BioMedical3DRandomCrop', 'SegRescale',
    'PhotoMetricDistortion', 'RandomRotate', 'AdjustGamma', 'CLAHE', 'Rerange',
    'RGB2Gray', 'RandomCutOut', 'RandomMosaic', 'PackSegInputs',
    'PackSegStereoInputs',
    'ResizeToMultiple', 'LoadImageFromNDArray', 'LoadBiomedicalImageFromFile',
    'LoadBiomedicalAnnotation', 'LoadBiomedicalData', 'GenerateEdge',
    'ResizeShortestEdge', 'BioMedicalGaussianNoise', 'BioMedicalGaussianBlur',
    'BioMedical3DRandomFlip', 'BioMedicalRandomGamma', 'BioMedical3DPad',
    'RandomRotFlip', 'Albu', 'LoadSingleRSImageFromFile', 'ConcatCDInput',
    'LoadMultipleRSImageFromFile', 'LoadDepthAnnotation', 'RandomDepthMix',
    'RandomFlip', 'RemapSegLabels', 'Resize',
    'ResizeToFrontOrSideImageOriginalSize', 'CopyPasteTop', 'ISPStyleAug', 'ConcatFrontAndSide', 'CropTop', 'ImageQualityAug', 'DilateSoilMask', 'RandomMasking', 'LoadStereoImages', 'LoadStereoMatchingAnnotations',
    'CREStereoAugmentor', 'FlowAugmentor', 'SparseFlowAugmentor', 'ResizeStereoImages'
]
