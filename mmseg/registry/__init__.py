# Copyright (c) OpenMMLab. All rights reserved.
from .registry import (DATA_SAMPLERS, DATASETS, EVALUATOR, HOOKS, INFERENCERS,
                       LOG_PROCESSORS, LOOPS, METRICS, MODEL_WRAPPERS, MODELS,
                       OPTIM_WRAPPER_CONSTRUCTORS, OPTIM_WRAPPERS, OPTIMIZERS,
                       PARAM_SCHEDULERS, RUNNER_CONSTRUCTORS, RUNNERS,
                       TASK_UTILS, TRANSFORMS, VISBACKENDS, VISUALIZERS,
                       WEIGHT_INITIALIZERS)

__all__ = [
    'HOOKS', 'DATASETS', 'DATA_SAMPLERS', 'TRANSFORMS', 'MODELS',
    'WEIGHT_INITIALIZERS', 'OPTIMIZERS', 'OPTIM_WRAPPER_CONSTRUCTORS',
    'TASK_UTILS', 'PARAM_SCHEDULERS', 'METRICS', 'MODEL_WRAPPERS',
    'VISBACKENDS', 'VISUALIZERS', 'RUNNERS', 'RUNNER_CONSTRUCTORS', 'LOOPS',
    'EVALUATOR', 'LOG_PROCESSORS', 'OPTIM_WRAPPERS', 'INFERENCERS'
]

import importlib

# Eager-import hook modules so @HOOKS.register_module() runs before Runner
# builds default_hooks (lazy mmseg.engine.hooks.__init__ would skip these).
for _hook_mod in (
        'mmseg.engine.hooks.session_checkpoint_hook',
        'mmseg.engine.hooks.segstereo_semantic_fuse_hook',
        'mmseg.engine.hooks.visualization_hook',
):
    importlib.import_module(_hook_mod)
