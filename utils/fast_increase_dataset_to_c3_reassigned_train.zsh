set -e

project_name=$1
dst_dataset_dir=$1

echo "开始导出labelstudio项目"
python utils/export_labelstudio_project.py $project_name ~/datasets/grass_seg_data_c3/train/$dst_dataset_dir
echo "导出labelstudio项目完成"

echo "开始软链接数据集"
python utils/soft_link_dataset.py ~/datasets/grass_seg_data_c3/train/$dst_dataset_dir ~/datasets/grass_seg_data_c3_reassigned/train
echo "软链接数据集完成"

echo "开始可视化"
rm -r pgs/vis_gt; python utils/visualize_gt.py ~/datasets/grass_seg_data_c3_reassigned/train/$dst_dataset_dir pgs/vis_gt --concat
echo "可视化完成"
