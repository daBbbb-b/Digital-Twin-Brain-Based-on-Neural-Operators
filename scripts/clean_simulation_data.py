import os
import pickle
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'dataset' / 'simulation_data'
OUTPUT_DIR = ROOT / 'dataset' / 'simulation_data_cleaned'

# 创建输出目录
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def process_single_file(p):
    """处理单个文件的函数，供并行调用"""
    print(f"Processing {p.name}...")
    try:
        with open(p, 'rb') as f:
            obj = pickle.load(f)
        
        if isinstance(obj, dict) and 'neural_activity' in obj:
            del obj['neural_activity']
            
            out_path = OUTPUT_DIR / p.name
            with open(out_path, 'wb') as f:
                # 使用最高协议版本以加快速度
                pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
            return True
        return False
    except Exception as e:
        print(f"Error processing {p.name}: {e}")
        return None

def main():
    pkl_files = sorted(DATA_DIR.glob('*.pkl'))
    print(f"Found {len(pkl_files)} files. Processing in parallel...")

    # 使用进程池并行处理
    # max_workers 建议设为 CPU 核心数，如果内存压力大可适当调小
    modified = 0
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(process_single_file, pkl_files))
    
    modified = sum(1 for r in results if r is True)
    errors = sum(1 for r in results if r is None)
    
    print(f"Scanned: {len(pkl_files)}")
    print(f"Cleaned and saved to {OUTPUT_DIR}: {modified}")
    if errors > 0:
        print(f"Errors encountered: {errors}")

if __name__ == "__main__":
    main()
