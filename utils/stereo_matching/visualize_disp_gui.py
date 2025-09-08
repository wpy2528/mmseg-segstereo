import cv2
import numpy as np
import argparse
import glob
import os

def read_pfm(path):
    with open(path, 'rb') as f:
        header = f.readline().decode().strip()
        if header == 'PF':
            color = True
        elif header == 'Pf':
            color = False
        else:
            raise Exception('Not a PFM file.')

        dim = f.readline().decode().strip()
        width, height = map(int, dim.split())
        scale = float(f.readline().decode().strip())
        endian = '<' if scale < 0 else '>'
        data = np.fromfile(f, endian + 'f')
        shape = (height, width, 3) if color else (height, width)
        data = np.reshape(data, shape)
        data = np.flipud(data)  # PFM文件通常是倒置的
        return data

def get_pfm_files(directory):
    """获取目录中所有的PFM文件"""
    pattern = os.path.join(directory, "*.pfm")
    pfm_files = glob.glob(pattern)
    return sorted(pfm_files)

def visualize_pfm(pfm_path, window_name):
    """可视化单个PFM文件"""
    data = read_pfm(pfm_path)
    
    # 可视化图（使用jet颜色映射）
    min_val = np.min(data)
    max_val = np.max(data)
    vis = (data - min_val) / (max_val - min_val + 1e-8)
    vis_img = (vis * 255).astype(np.uint8)

    # 应用jet颜色映射
    if vis_img.ndim == 2:
        vis_img = cv2.applyColorMap(vis_img, cv2.COLORMAP_JET)
    else:
        # 如果是3通道数据，取第一个通道进行颜色映射
        vis_img = cv2.applyColorMap(vis_img[:, :, 0], cv2.COLORMAP_JET)
    
    return vis_img, data

def mouse_move(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        data, vis_img, window_name = param
        value = data[y, x]
        if isinstance(value, np.ndarray):
            txt = f"({x}, {y}): {value[0]:.6f}, {value[1]:.6f}, {value[2]:.6f}"
        else:
            txt = f"({x}, {y}): {value:.6f}"

        img_copy = vis_img.copy()
        cv2.putText(img_copy, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2)
        cv2.imshow(window_name, img_copy)

def main():
    parser = argparse.ArgumentParser(description="PFM文件查看器 - 支持目录遍历")
    parser.add_argument("path", help="PFM文件路径或包含PFM文件的目录路径")
    parser.add_argument("--grep", default="", help="过滤文件名")
    args = parser.parse_args()
    
    path = args.path
    
    # 判断是文件还是目录
    if os.path.isfile(path):
        # 单个文件
        pfm_files = [path]
    elif os.path.isdir(path):
        # 目录
        pfm_files = glob.glob(os.path.join(path, "**", "*.pfm"), recursive=True)
        if args.grep:
            pfm_files = [file for file in pfm_files if args.grep in file]
        if not pfm_files:
            print(f"在目录 {path} 中没有找到PFM文件")
            return
        print(f"找到 {len(pfm_files)} 个PFM文件")
    else:
        print(f"路径 {path} 不存在")
        return
    
    current_index = 0
    
    while True:
        if current_index >= len(pfm_files):
            print("已浏览完所有文件")
            break
            
        current_file = pfm_files[current_index]
        print(f"正在显示: {os.path.basename(current_file)} ({current_index + 1}/{len(pfm_files)})")
        
        try:
            # 重新创建窗口，避免窗口关闭问题
            cv2.destroyAllWindows()
            # 使用当前pfm文件的路径作为窗口标题
            window_name = current_file
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            
            vis_img, data = visualize_pfm(current_file, window_name)
            
            # 设置鼠标回调
            cv2.setMouseCallback(window_name, mouse_move, (data, vis_img, window_name))
            
            # 显示图片
            cv2.imshow(window_name, vis_img)
            
            # 等待按键
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord('q'):
                # 切换到下一张图片
                current_index += 1
            elif key == ord('a'):
                # 切换到上一张图片
                current_index = max(0, current_index - 1)
            elif key == 27:  # ESC键
                break
            elif key == ord('r'):
                # 重新加载当前图片
                continue
                
        except Exception as e:
            print(f"处理文件 {current_file} 时出错: {e}")
            current_index += 1
            continue
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
