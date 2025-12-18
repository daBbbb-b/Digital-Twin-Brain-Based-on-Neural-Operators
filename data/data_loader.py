"""
数据加载器模块

功能说明：
    负责加载各种类型的大脑数据，包括fMRI数据、结构连接、有效连接等。

主要类：
    DataLoader: 数据加载器类

输入：
    - 数据路径配置
    - 数据类型（fMRI、结构连接、有效连接等）
    - 任务ID（可选）

输出：
    - 加载的数据数组（numpy.ndarray或字典格式）
    - 元数据信息

使用示例：
    loader = DataLoader(data_root='/path/to/data')
    fmri_data = loader.load_fmri(task_id='task001', subject_id='sub001')
    structural_conn = loader.load_structural_connectivity()
"""

import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Dict, Optional, Union, List


class DataLoader:
    """
    数据加载器类
    
    功能：
        - 加载MNI空间的fMRI数据（103个任务）
        - 加载T1图像和分割标签
        - 加载灰白质surface数据
        - 加载结构连接矩阵（皮层、白质）
        - 加载有效连接矩阵
        - 加载246分区模板
        
    属性：
        data_root (Path): 数据根目录路径
        task_file (Path): 任务列表tsv文件路径
        
    方法：
        load_fmri: 加载fMRI时间序列数据
        load_structural_connectivity: 加载结构连接矩阵
        load_effective_connectivity: 加载有效连接矩阵
        load_white_matter_connectivity: 加载白质连接矩阵
        load_t1_image: 加载T1图像
        load_surface: 加载surface数据
        load_parcellation: 加载分区模板
    """
    
    def __init__(self, data_root: str):
        """
        初始化数据加载器
        
        参数：
            data_root (str): 数据根目录路径
        """
        pass
    
    def load_fmri(self, task_id: str, subject_id: Optional[str] = None) -> np.ndarray:
        """
        加载fMRI时间序列数据
        
        参数：
            task_id (str): 任务ID（103个任务之一）
            subject_id (str, optional): 被试ID，如果为None则加载组平均数据
            
        返回：
            np.ndarray: fMRI时间序列数据，形状为 (n_timepoints, n_regions)
            注意：时间间隔为2秒
        """
        pass
    
    def load_structural_connectivity(self, connectivity_type: str = 'cortical') -> np.ndarray:
        """
        加载结构连接矩阵
        
        参数：
            connectivity_type (str): 连接类型
                - 'cortical': 皮层的结构连接
                - 'white_matter': 组平均的白质结构连接
                
        返回：
            np.ndarray: 结构连接矩阵，形状为 (n_regions, n_regions)
        """
        pass
    
    def load_effective_connectivity(self, task_id: str) -> np.ndarray:
        """
        加载有效连接矩阵
        
        使用文献《Mapping effective connectivity by virtually perturbing a surrogate brain》
        中的方法计算得到的有效连接
        
        参数：
            task_id (str): 任务ID
            
        返回：
            np.ndarray: 有效连接矩阵，形状为 (n_regions, n_regions)
        """
        pass
    
    def load_t1_image(self, subject_id: Optional[str] = None) -> nib.Nifti1Image:
        """
        加载MNI空间的T1图像及其分割标签
        
        参数：
            subject_id (str, optional): 被试ID，如果为None则加载模板
            
        返回：
            nib.Nifti1Image: T1图像对象
        """
        pass
    
    def load_surface(self, hemisphere: str = 'both') -> Dict[str, np.ndarray]:
        """
        加载MNI空间的灰白质surface数据
        
        参数：
            hemisphere (str): 半球选择
                - 'left': 左半球
                - 'right': 右半球
                - 'both': 双半球
                
        返回：
            Dict[str, np.ndarray]: surface数据字典，包含顶点坐标和面片信息
        """
        pass
    
    def load_parcellation(self, n_parcels: int = 246) -> np.ndarray:
        """
        加载脑区分区模板
        
        参数：
            n_parcels (int): 分区数量，默认246
            
        返回：
            np.ndarray: 分区标签数组
        """
        pass
