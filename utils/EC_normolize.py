import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union, Optional


def load_ec_matrix(file_path: Union[str, Path]) -> np.ndarray:
    """
    加载EC（有效连接）矩阵CSV文件
    
    参数：
        file_path: CSV文件路径
        
    返回：
        np.ndarray: EC矩阵 (N, N)
    """
    file_path = Path(file_path)
    try:
        df = pd.read_csv(file_path, header=None)
        return df.values.astype(np.float32)
    except Exception as e:
        raise ValueError(f"Failed to load EC matrix from {file_path}: {e}")


def normalize_ec_matrix(ec_matrix: np.ndarray, 
                        method: str = 'max_row_sum',
                        return_stats: bool = False) -> Union[np.ndarray, tuple]:
    """
    归一化EC（有效连接）矩阵
    
    参数：
        ec_matrix: EC矩阵 (N, N)，可以包含正负值
        method: 归一化方法
            - 'max_row_sum': 按最大行和归一化（类似SC矩阵归一化）
            - 'max_abs': 按最大绝对值归一化（保持相对强度比例）
            - 'spectral': 按谱范数（最大奇异值）归一化
            - 'row_wise': 按行归一化（每行独立归一化到[-1, 1]）
        return_stats: 是否返回归一化统计信息
        
    返回：
        normalized_matrix: 归一化后的矩阵
        如果 return_stats=True，返回 (normalized_matrix, stats_dict)
    """
    ec_matrix = np.asarray(ec_matrix, dtype=np.float32)
    
    if method == 'max_row_sum':
        # 按最大行和归一化（类似SC矩阵的处理方式）
        # 计算每行的绝对值和（因为EC可以有正负值）
        row_sums = np.sum(np.abs(ec_matrix), axis=1)
        max_row_sum = np.max(row_sums)
        if max_row_sum > 0:
            normalized = ec_matrix / max_row_sum
            stats = {'max_row_sum': max_row_sum, 'method': method}
        else:
            normalized = ec_matrix.copy()
            stats = {'max_row_sum': 0.0, 'method': method, 'warning': 'All rows sum to zero'}
            
    elif method == 'max_abs':
        # 按最大绝对值归一化（保持相对强度比例和符号）
        max_abs = np.max(np.abs(ec_matrix))
        if max_abs > 0:
            normalized = ec_matrix / max_abs
            stats = {'max_abs': max_abs, 'method': method}
        else:
            normalized = ec_matrix.copy()
            stats = {'max_abs': 0.0, 'method': method, 'warning': 'Matrix is all zeros'}
            
    elif method == 'spectral':
        # 按谱范数（最大奇异值）归一化
        # 这对于保持矩阵的动态稳定性很有用
        try:
            from scipy.linalg import svd
            U, s, Vh = svd(ec_matrix)
            max_singular = s[0] if len(s) > 0 else 1.0
            if max_singular > 0:
                normalized = ec_matrix / max_singular
                stats = {'max_singular': max_singular, 'method': method}
            else:
                normalized = ec_matrix.copy()
                stats = {'max_singular': 0.0, 'method': method, 'warning': 'Matrix is singular'}
        except ImportError:
            raise ImportError("scipy is required for spectral normalization")
            
    elif method == 'row_wise':
        # 按行归一化：每行独立归一化到[-1, 1]范围
        # 保持每行内部的相对比例
        normalized = np.zeros_like(ec_matrix)
        for i in range(ec_matrix.shape[0]):
            row = ec_matrix[i, :]
            row_max_abs = np.max(np.abs(row))
            if row_max_abs > 0:
                normalized[i, :] = row / row_max_abs
            else:
                normalized[i, :] = row
        stats = {'method': method, 'note': 'Each row normalized independently'}
        
    else:
        raise ValueError(f"Unknown normalization method: {method}. "
                        f"Choose from: 'max_row_sum', 'max_abs', 'spectral', 'row_wise'")
    
    if return_stats:
        return normalized, stats
    return normalized


def normalize_ec_matrix_from_file(file_path: Union[str, Path],
                                  method: str = 'max_row_sum',
                                  output_path: Optional[Union[str, Path]] = None,
                                  return_stats: bool = False) -> Union[np.ndarray, tuple]:
    """
    从CSV文件加载EC矩阵并归一化
    
    参数：
        file_path: EC矩阵CSV文件路径
        method: 归一化方法（见 normalize_ec_matrix）
        output_path: 可选，保存归一化后矩阵的路径
        return_stats: 是否返回统计信息
        
    返回：
        normalized_matrix: 归一化后的矩阵
        如果 return_stats=True，返回 (normalized_matrix, stats_dict)
    """
    # 加载矩阵
    ec_matrix = load_ec_matrix(file_path)
    
    # 归一化
    if return_stats:
        normalized, stats = normalize_ec_matrix(ec_matrix, method=method, return_stats=True)
    else:
        normalized = normalize_ec_matrix(ec_matrix, method=method, return_stats=False)
        stats = None
    
    # 保存（如果指定了输出路径）
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(normalized).to_csv(output_path, index=False, header=False)
        print(f"Normalized EC matrix saved to {output_path}")
    
    if return_stats:
        return normalized, stats
    return normalized


base_dir = Path(__file__).resolve().parent.parent
normalize_ec_matrix_from_file(file_path=base_dir / "dataset" /  "sub-01_EC_mean.csv", method="spectral", output_path=base_dir  / "EC_normalized.csv" , return_stats=True)
#normalize_ec_matrix_from_file(file_path=base_dir / "dataset" /  "task_001_EC.csv", method="max_row_sum", output_path=base_dir / "dataset" / "EC_normalized" , return_stats=True)