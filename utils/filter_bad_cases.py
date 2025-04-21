import os

def load_file_list(file_path):
    """加载文件列表"""
    with open(file_path, 'r') as f:
        return set(line.strip() for line in f)

def extract_filename(full_path):
    """从完整路径中提取文件名"""
    return os.path.basename(full_path)

def main():
    # 加载训练集文件列表
    train_file = "data/perception_segmentation/0107/train.txt"
    train_files = load_file_list(train_file)
    # 提取训练集中的文件名
    train_filenames = {extract_filename(path) for path in train_files}
    
    # 加载bad cases文件列表
    bad_cases_file = "pgs/bad_cases_from_over_exposure_3w.txt"
    bad_cases = load_file_list(bad_cases_file)
    
    # 找出不在训练集中的bad cases
    not_in_train = bad_cases - train_filenames
    
    # 输出结果
    print(f"训练集文件数: {len(train_files)}")
    print(f"Bad cases文件数: {len(bad_cases)}")
    print(f"不在训练集中的bad cases数: {len(not_in_train)}")
    
    # 将结果保存到文件
    output_file = "pgs/bad_cases_not_in_train.txt"
    with open(output_file, 'w') as f:
        for file_name in sorted(not_in_train):
            f.write(f"{file_name}\n")
    
    print(f"\n结果已保存到: {output_file}")

if __name__ == "__main__":
    main() 