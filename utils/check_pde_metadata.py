"""
检查PDE仿真数据的metadata结构
"""
import pickle
from pathlib import Path
import json

def check_pde_metadata(file_path):
    """检查PDE数据文件的metadata"""
    file_path = Path(file_path)
    
    print(f"\n{'='*60}")
    print(f"检查文件: {file_path.name}")
    print(f"{'='*60}")
    
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        print("\n数据文件包含的键:")
        print(f"  {list(data.keys())}")
        
        print("\n--- Metadata 内容 ---")
        metadata = data.get('metadata', {})
        if metadata:
            print("Metadata 键:")
            for key, value in metadata.items():
                if isinstance(value, (int, float, str, bool, type(None))):
                    print(f"  {key}: {value}")
                elif isinstance(value, np.ndarray):
                    print(f"  {key}: array shape={value.shape}, dtype={value.dtype}")
                else:
                    print(f"  {key}: {type(value).__name__}")
        else:
            print("  无 metadata")
        
        print("\n--- Stimulus Config 内容 ---")
        stim_config = data.get('stimulus_config', {})
        if stim_config:
            print("Stimulus Config 键:")
            for key, value in stim_config.items():
                if isinstance(value, (int, float, str, bool, type(None))):
                    print(f"  {key}: {value}")
                elif isinstance(value, (list, dict)):
                    print(f"  {key}: {type(value).__name__} (len={len(value) if hasattr(value, '__len__') else 'N/A'})")
                else:
                    print(f"  {key}: {type(value).__name__}")
        else:
            print("  无 stimulus_config")
        
        print("\n--- 数据形状信息 ---")
        if 'bold_signal' in data:
            bold = data['bold_signal']
            print(f"  bold_signal: shape={bold.shape if hasattr(bold, 'shape') else 'N/A'}")
        
        if 'neural_activity' in data:
            neural = data['neural_activity']
            print(f"  neural_activity: shape={neural.shape if hasattr(neural, 'shape') else 'N/A'}")
        
        if 'time_points' in data:
            time_pts = data['time_points']
            print(f"  time_points: shape={time_pts.shape if hasattr(time_pts, 'shape') else 'N/A'}")
            if hasattr(time_pts, '__len__') and len(time_pts) > 0:
                print(f"    时间范围: {time_pts[0]:.2f} - {time_pts[-1]:.2f} ms")
        
        print("\n--- 完整的 Metadata (JSON格式) ---")
        # 将metadata转换为可序列化的格式
        metadata_serializable = {}
        for key, value in metadata.items():
            if isinstance(value, (int, float, str, bool, type(None))):
                metadata_serializable[key] = value
            elif isinstance(value, np.ndarray):
                metadata_serializable[key] = f"array(shape={value.shape}, dtype={value.dtype})"
            else:
                metadata_serializable[key] = str(type(value).__name__)
        
        print(json.dumps(metadata_serializable, indent=2, ensure_ascii=False))
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import numpy as np
    import sys
    import os
    
    # 添加项目根目录到路径
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    
    # 检查数据目录中的PDE文件
    data_dir = Path('dataset/simulation_data')
    
    if len(sys.argv) > 1:
        # 如果提供了文件路径
        file_path = Path(sys.argv[1])
        check_pde_metadata(file_path)
    else:
        # 检查第一个找到的PDE文件
        pde_files = list(data_dir.glob('pde_*.pkl'))
        if pde_files:
            print(f"找到 {len(pde_files)} 个PDE文件")
            print(f"检查第一个文件: {pde_files[0].name}\n")
            check_pde_metadata(pde_files[0])
            
            if len(pde_files) > 1:
                print(f"\n提示: 可以指定其他文件进行检查，例如:")
                print(f"  python utils/check_pde_metadata.py {pde_files[1]}")
        else:
            print(f"在 {data_dir} 中未找到PDE文件")

