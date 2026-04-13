# Copyright (c) OpenMMLab. All rights reserved.
"""惰性加载各 Hook，避免 ``mmseg.registry`` 与 hooks 循环导入。"""

__all__ = [
    'SegVisualizationHook',
    'SegStereoSemanticFuseHook',
    'SessionCheckpointHook',
    'ValPredictionSaveHook',
]


def __getattr__(name):
    if name == 'SegVisualizationHook':
        from .visualization_hook import SegVisualizationHook

        return SegVisualizationHook
    if name == 'ValPredictionSaveHook':
        from .val_prediction_save_hook import ValPredictionSaveHook

        return ValPredictionSaveHook
    if name == 'SegStereoSemanticFuseHook':
        from .segstereo_semantic_fuse_hook import SegStereoSemanticFuseHook

        return SegStereoSemanticFuseHook
    if name == 'SessionCheckpointHook':
        from .session_checkpoint_hook import SessionCheckpointHook

        return SessionCheckpointHook
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
