#!/bin/zsh

set -e

src_onnx_path=$1
src_calibration_dataset_dir=$2

mean=0.0
std=255.0

# 检查mean和std是否为0.0和255.0
if [[ $mean != 0.0 || $std != 255.0 ]]; then
    echo "mean和std必须为0.0和255.0"
    exit 1
fi

# 检查onnx文件以.onnx结尾
if [[ $src_onnx_path != *.onnx ]]; then
    echo "onnx文件必须以.onnx结尾"
    exit 1
fi

# 检查onnx文件是否存在
if [ ! -f "$src_onnx_path" ]; then
    echo "onnx文件不存在"
    exit 1
fi

# 检查calibration_dataset是不是一个目录
if [ ! -d "$src_calibration_dataset_dir" ]; then
    echo "calibration_dataset不存在或不是一个目录"
    exit 1
fi

# 检查docker容器是否已启动，如果未启动则启动
container_id="3a47b3feb6abee0c8d3e8b311a3fb3680ee7be96b220749235f3212c9ffea5e3"
container_status=$(docker inspect -f '{{.State.Running}}' $container_id 2>/dev/null)

if [[ "$container_status" != "true" ]]; then
    echo "容器 $container_id 未启动，正在启动..."
    docker start $container_id
    if [[ $? -ne 0 ]]; then
        echo "启动容器 $container_id 失败，请检查容器ID是否正确"
        exit 1
    fi
else
    echo "容器 $container_id 已经启动"
fi


dir_to_delete="/rdk_model_zoo/demos/stereo_matching/DStereo_X5/ptq_mm_bgr/model_output_bgr"
echo "是否要删除 $dir_to_delete 目录？(y/n)"
read answer
if [[ $answer == "y" || $answer == "Y" ]]; then
    echo "正在删除目录..."
    docker exec $container_id rm -rf $dir_to_delete
    # echo "正在重新创建目录..."
    # docker exec $container_id mkdir -p $dir_to_delete
fi

# 打印onnx文件的md5
echo "onnx文件的md5: $(md5sum $src_onnx_path)"

# 把onnx文件拷贝到docker中
docker cp $src_onnx_path $container_id:/rdk_model_zoo/demos/stereo_matching/DStereo_X5/ptq_mm_bgr/float.onnx

# 检查标定数据集目录是否合法
python utils/x5/check_x5_stereo_matching_calibration_dataset_validation.py $src_calibration_dataset_dir
# 删除旧的标定数据集目录，并生成新的标定数据集目录
rm -r rdk_model_zoo/demos/stereo_matching/DStereo_X5/ptq_mm_bgr/calib_data
cp -r $src_calibration_dataset_dir rdk_model_zoo/demos/stereo_matching/DStereo_X5/ptq_mm_bgr/calib_data

# 进入docker执行模型量化，生成bin文件
docker exec -i $container_id bash <<'EOF'
cd /rdk_model_zoo/demos/stereo_matching/DStereo_X5
echo "替换mul和reducesum算子为gemm算子..."
python3.10 ptq_mm_bgr/replace_mul_reducesum.py ptq_mm_bgr/float.onnx ptq_mm_bgr/float_modify.onnx
echo "量化模型..."
hb_mapper makertbin -c ptq_mm_bgr/D-StereoPlus_bgr.yaml --model-type onnx
echo "模型量化完成！"

# 推理bin文件并可视化验证正确性
python3.10 ptq_mm_bgr/infer_quant_onnx.py --onnx_path ptq_mm_bgr/model_output_bgr/DStereoV23_quantized_model.onnx --input_dir /rdk_model_zoo/resource/stereo/ --result_path vis_bgr_quant
echo "推理完成！请检查vis_bgr_quant目录下的结果是否正确"
EOF

# 把生成的bin文件拷贝出来
dst_dir=$(dirname $src_onnx_path)
dst_bin_path=${dst_dir}/model_stereo_matching.bin
docker cp $container_id:/rdk_model_zoo/demos/stereo_matching/DStereo_X5/ptq_mm_bgr/model_output_bgr/DStereoV23.bin $dst_bin_path

# 打印nb文件的md5
echo "bin文件的md5: $(md5sum $dst_bin_path)"
