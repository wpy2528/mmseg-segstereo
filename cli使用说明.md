通过browse_dataset将pipeline增广后的图像创建为数据集
python tools/analysis_tools/browse_dataset.py configs/ld/generate_augmented_val_dataset.py --create_augmented_dataset --output-dir /data/mck/grass_seg_data/val_concat

一键拉取给定的labelstudio project
python utils/export_labelstudio_project.py missseg_common pgs/kk_c3


