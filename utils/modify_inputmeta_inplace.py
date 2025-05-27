import sys
import yaml
import argparse

def update_mean_scale(yml_path, mean_val, std_val):
    with open(yml_path, 'r') as f:
        data = yaml.safe_load(f)

    # 计算 scale = 1 / std
    scale_val = 1.0 / std_val

    try:
        ports = data['input_meta']['databases'][0]['ports']
        for port in ports:
            port['preprocess']['mean'] = [float(mean_val)] * 3
            port['preprocess']['scale'] = [float(scale_val)] * 3
    except KeyError as e:
        print(f"KeyError: {e}")
        return

    with open(yml_path, 'w') as f:
        yaml.safe_dump(data, f, sort_keys=False)

    print(f"Updated mean to {mean_val} and scale to {scale_val} in '{yml_path}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("yml_path", help="yml文件路径")
    parser.add_argument("mean_val", type=float, help="mean值")
    parser.add_argument("std_val", type=float, help="std值") 
    args = parser.parse_args()

    update_mean_scale(args.yml_path, args.mean_val, args.std_val)
