set -e

for project_name in "$@"
do
    dst_dataset_dir=$project_name

    echo "开始导出labelstudio项目: $project_name"
    python utils/export_labelstudio_project.py $project_name ~/datasets/occlusion_seg_data/$dst_dataset_dir occlusion
    echo "导出labelstudio项目完成: $project_name"

    echo "开始可视化: $project_name"
    rm -rf pgs/vis_gt_$project_name
    python utils/visualize_gt.py ~/datasets/occlusion_seg_data/$dst_dataset_dir pgs/vis_gt_$project_name --concat
    echo "可视化完成: $project_name"
done
