import os
import shutil
from pathlib import Path

def count_mmengine_info(log_file):
    """统计日志文件中'mmengine - INFO -'出现的次数"""
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
            return content.count('mmengine - INFO -')
    except Exception as e:
        print(f"读取文件 {log_file} 时出错: {e}")
        return 0

def clean_empty_logs(base_dir='work_dirs', min_info_count=50, dry_run=True):
    """清理无效的日志目录"""
    base_path = Path(base_dir)
    empty_dirs = []
    
    # 遍历work_dirs下的所有目录
    for exp_dir in base_path.iterdir():
        if not exp_dir.is_dir():
            continue
            
        # 遍历实验目录下的所有子目录
        for log_dir in exp_dir.iterdir():
            if not log_dir.is_dir():
                continue
                
            # 检查同名log文件
            log_file = log_dir / f"{log_dir.name}.log"
            if not log_file.exists() or count_mmengine_info(log_file) < min_info_count:
                empty_dirs.append(log_dir)
    
    # 输出结果
    print(f"找到 {len(empty_dirs)} 个无效的日志目录")
    for dir_path in empty_dirs:
        if dry_run:
            print(f"[DRY RUN] 将删除: {dir_path}")
        else:
            try:
                shutil.rmtree(dir_path)
                print(f"已删除: {dir_path}")
            except Exception as e:
                print(f"删除 {dir_path} 时出错: {e}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='清理无效的日志目录')
    parser.add_argument('--base-dir', type=str, default='work_dirs',
                      help='要清理的基准目录')
    parser.add_argument('--min-info-count', type=int, default=50,
                      help='日志文件中mmengine - INFO -的最小出现次数')
    parser.add_argument('--dry-run', action='store_true',
                      help='只显示将要删除的目录，不实际删除')
    
    args = parser.parse_args()
    
    clean_empty_logs(args.base_dir, args.min_info_count, args.dry_run)

if __name__ == "__main__":
    main() 