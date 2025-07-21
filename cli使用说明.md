通过browse_dataset将pipeline增广后的图像创建为数据集
python tools/analysis_tools/browse_dataset.py configs/ld/generate_augmented_val_dataset.py --create_augmented_dataset --output-dir /data/mck/grass_seg_data/val_concat

一键拉取给定的labelstudio project
python utils/export_labelstudio_project.py missseg_common pgs/kk_c3


模型pth导出为onnx然后导出为nb

 python utils/export_onnx.py configs/ld/stdc2_grass-c4-320x272-penalty_fp_bg_0702.py  work_dirs/stdc2_grass-c4-320x272-penalty_fp_bg_0702/last_checkpoint --input_hw 272 320

    ./utils/convert_onnx_to_nb.zsh work_dirs/stdc2_grass-c4-320x272-penalty_fp_bg_0702/epoch_70.onnx