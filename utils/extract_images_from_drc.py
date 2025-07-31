import struct
import glob
import os
import cv2
import numpy as np
import argparse
from pathlib import Path


import re
from datetime import datetime, timedelta
from typing import Generator, Tuple

import patoolib

def extract_file(src_zip_path, dst_dir):
    """
    解压缩文件到指定目录
    :param src_zip_path: 压缩文件路径
    :param dst_dir: 解压目标目录
    """
    os.makedirs(dst_dir, exist_ok=True)
    try:
        patoolib.extract_archive(str(src_zip_path), outdir=str(dst_dir))
        print(f"[ok] 已解压: {src_zip_path} -> {dst_dir}")
    except patoolib.util.PatoolError as e:
        print(f"[ERR] 解压失败 {src_zip_path}: {e}")

def get_time_str_local(timestamp_ms: int) -> str:
    # 将毫秒时间戳转为 datetime（本地时区）
    dt = datetime.fromtimestamp(timestamp_ms / 1000.0)
    return dt.strftime("%Y-%m-%d-%H%M%S-%f")[:-3]  # 保留毫秒（去掉最后3位微秒）

def get_final_time_str(start_real_unix_ts: int, head_timestamp_us: int, start_sys_ts: int) -> str:
    adjusted_ts_ms = start_real_unix_ts + (head_timestamp_us) - start_sys_ts
    return get_time_str_local(adjusted_ts_ms)

def extract_unix_timestamp(value: str) -> int:
    if value is None:
        return 0  # 或者 raise 异常
    
    pattern = r"(\d{8}_\d{6})"
    match = re.search(pattern, value)
    
    date_str = "19700101_000000"  # 默认值（注意补足6位秒数）
    if match:
        date_str = match.group(1)
    
    try:
        dt = datetime.strptime(date_str, "%Y%m%d_%H%M%S")
        unix_ts = int(dt.timestamp() * 1000)  # 转为毫秒
    except Exception as e:
        print("⚠️ 时间解析失败:", e)
        unix_ts = 0

    return unix_ts


# === 结构定义 ===
LOG_HEAD_STRUCT = struct.Struct('<BBHI BBB b i 16s')    # LogHead
PERCEIVE_HEAD_STRUCT = struct.Struct('<BBIQ')           # PerceiveHead (14 bytes)

def extract_images_from_drc(file_path: str, save_jpg: bool = False) -> Generator[Tuple[np.ndarray, str], None, None]:
    """解析drc文件，返回图像数据和对应的文件名"""
    start_real_unix_ts = extract_unix_timestamp(os.path.splitext(os.path.basename(file_path))[0])
    _start_sys_ts = 0

    with open(file_path, 'rb') as f:
        index = 0
        while True:
            head_bytes = f.read(LOG_HEAD_STRUCT.size)
            if len(head_bytes) < LOG_HEAD_STRUCT.size:
                break

            # 解包 LogHead
            (
                start_chip_id,
                target_chip_id,
                length,
                systick_ms,
                mod_type_id,
                mod_offset,
                mod_len,
                data_type,
                real_len,
                robot_sn
            ) = LOG_HEAD_STRUCT.unpack(head_bytes)

            # 读取 data 段
            try:
                data = f.read(real_len)
            except Exception as e:
                print(f"⚠️ 读取数据失败: {e}")
                break

            if len(data) < real_len:
                print("⚠️ 文件中 data 长度不足，提前结束")
                break

            # === 感知结果解析 ===
            if data_type == 7:
                if len(data) < PERCEIVE_HEAD_STRUCT.size:
                    print("⚠️ 感知头不足，跳过该块")
                    continue

                perceive_head = data[:PERCEIVE_HEAD_STRUCT.size]
                sys_count, p_type, p_len, time_stamp = PERCEIVE_HEAD_STRUCT.unpack(perceive_head)

                # 图像类型判断
                if p_type <= 30:
                    image_data = data[PERCEIVE_HEAD_STRUCT.size:PERCEIVE_HEAD_STRUCT.size + p_len]
                    if len(image_data) != p_len:
                        print("⚠️ 图像数据长度不足")
                        continue
                    
                    if _start_sys_ts == 0:
                        _start_sys_ts = time_stamp // 1000
                    
                    # 解码图像数据
                    img_array = np.frombuffer(image_data, dtype=np.uint8)
                    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    
                    if img is not None:
                        # 生成文件名
                        dst_image_name = get_final_time_str(start_real_unix_ts, time_stamp // 1000, _start_sys_ts) + (".jpg" if save_jpg else ".png")
                        yield img, dst_image_name
                    else:
                        print("❌ 图像解码失败")

            index += 1

def process_drc_files(src_path: str, save_dir: str, reconcat_3x3: bool = False, save_jpg: bool = False):
    """处理drc文件或目录"""
    # 获取所有drc文件路径
    if os.path.isfile(src_path):
        drc_paths = [src_path]
    else:
        drc_zip_paths = glob.glob(os.path.join(src_path, "**", "*.drc.zip"), recursive=True)
        for drc_zip_path in drc_zip_paths:
            extract_file(drc_zip_path, os.path.dirname(drc_zip_path))
        drc_paths = glob.glob(os.path.join(src_path, "**", "*.drc"), recursive=True)
    if not drc_paths:
        print(f"⚠️ 没有找到drc文件: {src_path}")
        return
    
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"找到 {len(drc_paths)} 个drc文件")
    total_images = 0
    
    for drc_path in drc_paths:
        print(f"\n处理文件: {drc_path}")
        drc_name = Path(drc_path).stem
        save_subdir = save_dir
        os.makedirs(save_subdir, exist_ok=True)
        
        # 直接处理生成器，实现流式处理
        image_count = 0
        for img, filename in extract_images_from_drc(drc_path, save_jpg):
            if reconcat_3x3:
                if img.shape[1] == 2880:
                    # 将img横向切成3份,然后竖着拼接
                    src_image_patches = []
                    for i in range(3):
                        src_image_patches.append(img[:, i*img.shape[1]//3:(i+1)*img.shape[1]//3])
                    dst_image = np.concatenate(src_image_patches, axis=0)
                else:
                    continue
            else:
                dst_image = img
            out_path = os.path.join(save_subdir, filename)
            cv2.imwrite(out_path, dst_image)
            image_count += 1
            print(f"✅ 图像保存为: {out_path}")
        
        total_images += image_count
        print(f"✅ 文件 {drc_name} 解析了 {image_count} 张图像")
    
    print(f"\n✅ 总共解析了 {total_images} 张图像")

def main():
    parser = argparse.ArgumentParser(description='解析drc文件中的图像数据')
    parser.add_argument('src_path', help='drc文件或目录的路径')
    parser.add_argument('--save_dir', default=None, help='保存解析出的图像的目录')
    parser.add_argument('--reconcat_3x3', action='store_true', help='是否重组九宫格')
    parser.add_argument('--save_jpg', action='store_true', help='是否保存为png格式')
    args = parser.parse_args()
    if args.save_dir is None:
        if os.path.isdir(args.src_path):
            save_dir = args.src_path
        else:
            save_dir = os.path.dirname(args.src_path)
    else:
        save_dir = args.save_dir
    
    process_drc_files(args.src_path, save_dir, args.reconcat_3x3, args.save_jpg)

if __name__ == "__main__":
    main()
