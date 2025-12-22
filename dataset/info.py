#打印训练样本详细信息（pkl文件）
import pathlib
import pickle
from pathlib import Path
def print_dataset_info(file_path: Path):
    # 加载pkl文件
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    # 打印数据集信息
    print(f"Dataset Information for {file_path.name}:")
    print(f"Total samples: {len(data)}")
    
    for i, sample in enumerate(data):
        print(f"\nSample {i + 1}:")
        if isinstance(sample, dict):
            for key, value in sample.items():
                if isinstance(value, list):
                    print(f"  {key}: List of length {len(value)}")
                else:
                    print(f"  {key}: {value}")
        else:
            print(f"  Sample is not a dictionary: {sample}")
# 示例用法
basedir = pathlib.Path(__file__).parent
print_dataset_info(basedir / 'simulation_data_cleaned' / 'ode_sc_ei_sample_0.pkl')
