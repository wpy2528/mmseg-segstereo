import argparse
import os
import glob
import cv2
import numpy as np
import onnxruntime

def main():
    parser = argparse.ArgumentParser(description="ONNX双输入（left/right）单输出模型推理")
    parser.add_argument("onnx_path", type=str, help="onnx模型文件路径")
    parser.add_argument("left_image_path", type=str, help="左图像路径")
    parser.add_argument("right_image_path", type=str, help="右图像路径")
    parser.add_argument("output_path", type=str, help="输出结果保存路径")
    args = parser.parse_args()

    # 读取左、右图像
    left_image_np = cv2.imread(args.left_image_path)
    right_image_np = cv2.imread(args.right_image_path)

    left_image_np = cv2.resize(left_image_np, (480, 256))
    right_image_np = cv2.resize(right_image_np, (480, 256))
    if left_image_np is None or right_image_np is None:
        raise FileNotFoundError("左图像或右图像读取失败，请检查路径。")

    # 转为float32，归一化到[0,1]，并转为NCHW
    left_image_np = left_image_np.astype(np.float32) / 255.0
    right_image_np = right_image_np.astype(np.float32) / 255.0
    left_image_np = np.transpose(left_image_np, (2, 0, 1))[np.newaxis, ...]
    right_image_np = np.transpose(right_image_np, (2, 0, 1))[np.newaxis, ...]

    # 加载onnx模型
    session = onnxruntime.InferenceSession(args.onnx_path, providers=['CPUExecutionProvider'])
    input_names = [inp.name for inp in session.get_inputs()]
    if len(input_names) != 2:
        raise RuntimeError("模型输入数量不是2，请检查onnx模型。")
    output_name = session.get_outputs()[0].name

    # 推理
    ort_inputs = {
        input_names[0]: left_image_np,
        input_names[1]: right_image_np
    }
    ort_outs = session.run([output_name], ort_inputs)
    output_np = ort_outs[0]

    # 保存输出（假设输出为单通道图像，保存为16位png）
    if output_np.ndim == 4:
        output_np = output_np[0]
    if output_np.shape[0] == 1:
        output_np = output_np[0]
    mask_np = np.squeeze(output_np)
    print(mask_np.max(), mask_np.min())
    
    
    # 归一化到0-255
    disp_min = np.min(mask_np)
    disp_max = np.max(mask_np)
    disp_normalized = ((mask_np - disp_min) / (disp_max - disp_min) * 255).astype(np.uint8)
    
    # 应用颜色映射
    vis_np = cv2.applyColorMap(disp_normalized, cv2.COLORMAP_JET)
    print(vis_np.shape)
    
    cv2.imwrite(args.output_path, vis_np)
    print(f"推理结果已保存到: {args.output_path}")

if __name__ == "__main__":
    main()
