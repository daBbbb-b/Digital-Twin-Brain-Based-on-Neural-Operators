import os
import numpy as np
import pickle
import pathlib
from nilearn.maskers import NiftiLabelsMasker
import pandas as pd

# ================= 路径配置 =================
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
BOLD_DIR = PROJECT_ROOT / "dataset" / "bold_nii"              # 存放 sub-01_xxx.nii.gz
ATLAS_PATH = PROJECT_ROOT / "dataset" / "BN_Atlas_246_1mm.nii.gz" # 脑图谱
OUTPUT_DIR = PROJECT_ROOT / "results" / "inference_data"      # 输出结果
# ===========================================

def preprocess_nifti():
    if not ATLAS_PATH.exists():
        print(f"错误：找不到图谱文件 {ATLAS_PATH}")
        return
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    nii_files = sorted(list(BOLD_DIR.glob("*.nii.gz")))
    
    print(f"初始化 Masker (Atlas: BN_246)...")
    # 队友代码没做标准化，所以这里 standardized=False
    masker = NiftiLabelsMasker(labels_img=str(ATLAS_PATH), standardized=False, verbose=1)

    for nii_path in nii_files:
        try:
            print(f"正在提取信号: {nii_path.name}")
            # 1. 提取时间序列 (Time, Regions)
            time_series = masker.fit_transform(str(nii_path))
            
            # 2. 处理 NaN (参考 data_loader.py 第 59 行)
            if np.isnan(time_series).any():
                time_series = np.nan_to_num(time_series, nan=0.0)
            
            # 3. 构造数据字典
            # 我们直接把 BOLD 存为 "x"，这样逻辑上它是模型的 Input
            data_dict = {
                "x": time_series.astype(np.float32),  # Shape: (Time, 246)
                "filename": nii_path.name,
                "n_regions": time_series.shape[1]
            }
            
            # 4. 保存
            save_name = nii_path.name.replace(".nii.gz", ".pkl")
            with open(OUTPUT_DIR / save_name, "wb") as f:
                pickle.dump(data_dict, f)
                
            print(f"已保存: {save_name} | Shape: {time_series.shape}")

        except Exception as e:
            print(f"处理失败 {nii_path.name}: {e}")

if __name__ == "__main__":
    preprocess_nifti()