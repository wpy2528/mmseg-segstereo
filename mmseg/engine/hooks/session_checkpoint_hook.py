# Copyright (c) OpenMMLab. All rights reserved.
"""当次训练会话目录下保存权重（与 Runner 日志子目录 ``work_dir/<timestamp>/`` 对齐）。"""

import os.path as osp
from collections import deque

from mmengine.dist import is_main_process
from mmengine.fileio import FileClient, get_file_backend
from mmengine.hooks import CheckpointHook
from mmengine.utils import mkdir_or_exist

from mmseg.registry import HOOKS


@HOOKS.register_module()
class SessionCheckpointHook(CheckpointHook):
    """继承 :class:`mmengine.hooks.CheckpointHook`，将 ``.pth`` 写入
    ``<runner.work_dir>/<runner.timestamp>/``。

    MMEngine 默认把 checkpoint 放在 ``work_dir`` 根目录；本 Hook 与
    ``Runner`` 为日志创建的 ``work_dir/<timestamp>/`` 使用同一时间戳子目录，
    便于按「当次训练」归档模型文件。

    若配置里显式设置了 ``out_dir`` 且不等于 ``work_dir``，仍沿用父类对
    ``out_dir`` 的拼接规则（在自定义根目录下再加 ``work_dir`` 末级名），
    不再强制套一层 ``timestamp``。
    """

    def before_train(self, runner) -> None:
        user_set_out = self.out_dir is not None
        redirected_to_session = False

        if not user_set_out:
            self.out_dir = osp.join(runner.work_dir, runner.timestamp)
            mkdir_or_exist(self.out_dir)
            redirected_to_session = True
        elif self.out_dir == runner.work_dir:
            self.out_dir = osp.join(runner.work_dir, runner.timestamp)
            mkdir_or_exist(self.out_dir)
            redirected_to_session = True

        self.file_client = FileClient.infer_client(self.file_client_args,
                                                   self.out_dir)

        if self.file_client_args is None:
            self.file_backend = get_file_backend(
                self.out_dir, backend_args=self.backend_args)
        else:
            self.file_backend = self.file_client

        if (user_set_out and not redirected_to_session
                and self.out_dir != runner.work_dir):
            basename = osp.basename(runner.work_dir.rstrip(osp.sep))
            self.out_dir = self.file_backend.join_path(
                self.out_dir, basename)  # type: ignore

        runner.logger.info(f'Checkpoints will be saved to {self.out_dir}.')

        if self.save_best is not None:
            if len(self.key_indicators) == 1:
                if 'best_ckpt' not in runner.message_hub.runtime_info:
                    self.best_ckpt_path = None
                else:
                    self.best_ckpt_path = runner.message_hub.get_info(
                        'best_ckpt')
            else:
                for key_indicator in self.key_indicators:
                    best_ckpt_name = f'best_ckpt_{key_indicator}'
                    if best_ckpt_name not in runner.message_hub.runtime_info:
                        self.best_ckpt_path_dict[key_indicator] = None
                    else:
                        self.best_ckpt_path_dict[
                            key_indicator] = runner.message_hub.get_info(
                                best_ckpt_name)

        if self.max_keep_ckpts > 0:
            keep_ckpt_ids = []
            if 'keep_ckpt_ids' in runner.message_hub.runtime_info:
                keep_ckpt_ids = runner.message_hub.get_info('keep_ckpt_ids')

            while len(keep_ckpt_ids) > self.max_keep_ckpts:
                step = keep_ckpt_ids.pop(0)
                if is_main_process():
                    path = self.file_backend.join_path(
                        self.out_dir, self.filename_tmpl.format(step))
                    if self.file_backend.isfile(path):
                        self.file_backend.remove(path)
                    elif self.file_backend.isdir(path):
                        self.file_backend.rmtree(path)

            self.keep_ckpt_ids = deque(keep_ckpt_ids, self.max_keep_ckpts)
