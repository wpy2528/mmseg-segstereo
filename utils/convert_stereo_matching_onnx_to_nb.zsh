#!/bin/zsh

set -e

src_onnx_path=$1
src_calibration_dataset_dir=$2

mean=0.0
std=255.0

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

# 检查docker容器f744897436592是否已启动，如果未启动则启动
container_id="f744897436592"
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


# 询问用户是否要删除目录
echo "是否要删除并重新创建目录？(y/n)"
read answer
if [[ $answer == "y" || $answer == "Y" ]]; then
    echo "正在删除目录..."
    docker exec $container_id rm -rf /workspace/board-demo-T527/docker_images_v1.8.x/model-convert/stereo_matching/model_stereo_matching
    echo "正在重新创建目录..."
    docker exec $container_id mkdir -p /workspace/board-demo-T527/docker_images_v1.8.x/model-convert/stereo_matching/model_stereo_matching
fi

# 打印onnx文件的md5
echo "onnx文件的md5: $(md5sum $src_onnx_path)"

docker exec $container_id mkdir -p /workspace/board-demo-T527/docker_images_v1.8.x/model-convert/stereo_matching/model_stereo_matching
# 把onnx文件拷贝到docker中
docker cp $src_onnx_path $container_id:/workspace/board-demo-T527/docker_images_v1.8.x/model-convert/stereo_matching/model_stereo_matching/model_stereo_matching.onnx

# 删除 calibration_dataset 目录
docker exec $container_id rm -rf /workspace/board-demo-T527/docker_images_v1.8.x/model-convert/stereo_matching/calibration_dataset

# 将 $2 复制为 calibration_dataset 目录
docker cp "$src_calibration_dataset_dir" $container_id:/workspace/board-demo-T527/docker_images_v1.8.x/model-convert/stereo_matching/calibration_dataset


# 进入docker执行模型转换，生成预处理和后处理yml文件
docker exec -i $container_id bash <<'EOF'
cd /workspace/board-demo-T527/docker_images_v1.8.x/model-convert/stereo_matching

pushd model_stereo_matching
find ../calibration_dataset -name "*.jpg" > dataset.txt
popd

export ACUITY_PATH=/root/acuity-toolkit-binary-6.21.14/bin
export VIV_SDK=/root/Vivante_IDE/VivanteIDE5.8.2/cmdtools
alias pegasus='/root/acuity-toolkit-binary-6.21.14/bin/pegasus'
alias nbinfo='/root/nbinfo'

ACUITY_PATH=/root/acuity-toolkit-binary-6.21.14/bin VIV_SDK=/root/Vivante_IDE/VivanteIDE5.8.2/cmdtools ./pegasus_import.sh model_stereo_matching/
EOF

# 把生成的预处理yml文件从docker中拷贝出来，将其默认的均值和方差修改为我们指定的值，然后再拷贝回docker中
docker cp $container_id:/workspace/board-demo-T527/docker_images_v1.8.x/model-convert/stereo_matching/model_stereo_matching/model_stereo_matching_inputmeta.yml .
python utils/modify_inputmeta_inplace.py model_stereo_matching_inputmeta.yml $mean $std
docker cp model_stereo_matching_inputmeta.yml $container_id:/workspace/board-demo-T527/docker_images_v1.8.x/model-convert/stereo_matching/model_stereo_matching/

# 进入docker执行模型量化，生成nb文件
docker exec -i $container_id bash <<'EOF'
cd /workspace/board-demo-T527/docker_images_v1.8.x/model-convert/stereo_matching
ACUITY_PATH=/root/acuity-toolkit-binary-6.21.14/bin VIV_SDK=/root/Vivante_IDE/VivanteIDE5.8.2/cmdtools ./quan_infer_export_pipeline.sh model_stereo_matching/
EOF

# 把生成的nb文件拷贝到src_onnx_path的目录下
dst_dir=$(dirname $src_onnx_path)
dst_nb_path=${dst_dir}/model_grass_recognize.nb
docker cp $container_id:/workspace/board-demo-T527/docker_images_v1.8.x/model-convert/stereo_matching/model_stereo_matching/wksp/model_stereo_matching_uint8_nbg_unify/model_stereo_matching_uint8_x527.nb $dst_nb_path

# 打印nb文件的md5
echo "nb文件的md5: $(md5sum $dst_nb_path)"

