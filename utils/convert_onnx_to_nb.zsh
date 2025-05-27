#!/bin/zsh

set -e

src_onnx_path=$1

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

# 询问用户是否要删除目录
echo "是否要删除并重新创建目录？(y/n)"
read answer
if [[ $answer == "y" || $answer == "Y" ]]; then
    echo "正在删除目录..."
    docker exec f744897436592 rm -rf /workspace/board-demo-T527/docker_images_v1.8.x/model-convert/grass_segmentation/model_grass_segmentation
    echo "正在重新创建目录..."
    docker exec f744897436592 mkdir -p /workspace/board-demo-T527/docker_images_v1.8.x/model-convert/grass_segmentation/model_grass_segmentation
fi

# 打印onnx文件的md5
echo "onnx文件的md5: $(md5sum $src_onnx_path)"

docker exec f744897436592 mkdir -p /workspace/board-demo-T527/docker_images_v1.8.x/model-convert/grass_segmentation/model_grass_segmentation
# 把onnx文件拷贝到docker中
docker cp $src_onnx_path f744897436592:/workspace/board-demo-T527/docker_images_v1.8.x/model-convert/grass_segmentation/model_grass_segmentation/model_grass_segmentation.onnx

# 进入docker执行模型转换
docker exec -i f744897436592 bash <<'EOF'
cd /workspace/board-demo-T527/docker_images_v1.8.x/model-convert/grass_segmentation

pushd model_grass_segmentation
find ../calibration_dataset -name "*.jpg" > dataset.txt
popd

export ACUITY_PATH=/root/acuity-toolkit-binary-6.21.14/bin
export VIV_SDK=/root/Vivante_IDE/VivanteIDE5.8.2/cmdtools
alias pegasus='/root/acuity-toolkit-binary-6.21.14/bin/pegasus'
alias nbinfo='/root/nbinfo'

ACUITY_PATH=/root/acuity-toolkit-binary-6.21.14/bin VIV_SDK=/root/Vivante_IDE/VivanteIDE5.8.2/cmdtools ./pegasus_import.sh model_grass_segmentation/
EOF

docker cp f744897436592:/workspace/board-demo-T527/docker_images_v1.8.x/model-convert/grass_segmentation/model_grass_segmentation/model_grass_segmentation_inputmeta.yml .
python utils/modify_inputmeta_inplace.py model_grass_segmentation_inputmeta.yml $mean $std
docker cp model_grass_segmentation_inputmeta.yml f744897436592:/workspace/board-demo-T527/docker_images_v1.8.x/model-convert/grass_segmentation/model_grass_segmentation/

docker exec -i f744897436592 bash <<'EOF'
cd /workspace/board-demo-T527/docker_images_v1.8.x/model-convert/grass_segmentation
ACUITY_PATH=/root/acuity-toolkit-binary-6.21.14/bin VIV_SDK=/root/Vivante_IDE/VivanteIDE5.8.2/cmdtools ./quan_infer_export_pipeline.sh model_grass_segmentation/
EOF

# 把nb文件拷贝到src_onnx_path的目录下
dst_dir=$(dirname $src_onnx_path)
dst_nb_path=${dst_dir}/model_grass_recognize.nb
docker cp f744897436592:/workspace/board-demo-T527/docker_images_v1.8.x/model-convert/grass_segmentation/model_grass_segmentation/wksp/model_grass_segmentation_uint8_nbg_unify/model_grass_segmentation_uint8_x527.nb $dst_nb_path

# 打印nb文件的md5
echo "nb文件的md5: $(md5sum $dst_nb_path)"

