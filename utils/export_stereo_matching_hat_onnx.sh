#!/usr/bin/env zsh
set -e

src_pth_path=$1
dst_onnx_path=$2

echo "开始转换mmseg模型为hat模型（修改state_dict键名），模型路径： $src_pth_path"
python utils/stereo_matching/convert_mmseg_model_to_hat.py $src_pth_path

current_dir=$(pwd)

cd /data_SSD2/mck/DStereoV23 && pwd

/mount/docker_data/conda_env/envs/stereonet_test/bin/python -c "import hat"

PYTHONPATH=/data_SSD2/mck/DStereoV23 \
/mount/docker_data/conda_env/envs/stereonet_test/bin/python tools/deploy/export_onnx.py \
  --config /data_SSD2/mck/DStereoV23/DStereo/DStereoPlus.py

if [[ "$dst_onnx_path" == /* ]]; then
  cp ptq/float.onnx "$dst_onnx_path"
else
  cp ptq/float.onnx "$current_dir/$dst_onnx_path"
fi

echo "地平线环境下的onnx导出完成，onnx文件路径： $dst_onnx_path"

cd - && pwd