import argparse
import os

def read_file_list(txt_path):
    with open(txt_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    return set(lines)

def write_list_to_txt(path_list, out_txt_path):
    with open(out_txt_path, 'w') as f:
        for path in path_list:
            f.write(f"{path}\n")


def main():
    parser = argparse.ArgumentParser(description="对比两个txt文件中的文件路径差异，并将差异写入txt文件")
    parser.add_argument('a_txt_path', type=str, help='第一个txt文件路径')
    parser.add_argument('b_txt_path', type=str, help='第二个txt文件路径')
    args = parser.parse_args()
    args.only_in_a_txt_path = os.path.join("pgs", "only_in_" + (args.a_txt_path).split('/')[-3] + ".txt")
    args.only_in_b_txt_path = os.path.join("pgs", "only_in_" + (args.b_txt_path).split('/')[-3] + ".txt")
    

    a_set = read_file_list(args.a_txt_path)
    b_set = read_file_list(args.b_txt_path)

    only_in_a = sorted(a_set - b_set)
    only_in_b = sorted(b_set - a_set)

    print(f"A中有但B中没有的文件路径（共{len(only_in_a)}个），已写入: {args.only_in_a_txt_path}")
    print(f"B中有但A中没有的文件路径（共{len(only_in_b)}个），已写入: {args.only_in_b_txt_path}")

    write_list_to_txt(only_in_a, args.only_in_a_txt_path)
    write_list_to_txt(only_in_b, args.only_in_b_txt_path)

if __name__ == '__main__':
    main()


