"""
有效连接模块

功能说明：
    计算和处理有效连接（Effective Connectivity）。

主要类：
    EffectiveConnectivity: 有效连接计算类

输入：
    - fMRI时间序列数据
    - 脑区定义

输出：
    - 有效连接矩阵
    - 连接强度和方向

参考：
    Luo et al. (2025) "Mapping effective connectivity by virtually perturbing 
    a surrogate brain" Nature Methods
"""

import numpy as np
from typing import Dict, Optional, Tuple, List


class EffectiveConnectivity:
    """
    有效连接计算类
    
    功能：
        - 基于虚拟扰动方法计算有效连接
        - 实现动态因果模型（DCM）
        - 分析因果影响关系
        
    有效连接定义：
        有效连接描述一个神经元群体对另一个群体的因果影响
        
    方法：
        - 虚拟扰动法（Virtual Perturbation）
        - 动态因果模型（Dynamic Causal Modeling）
        - Granger因果分析
        
    应用：
        - ODE动力学：有效连接作为ODE系统的连接矩阵
        - 任务态分析：揭示任务相关的定向连接
        
    属性：
        method (str): 计算方法
        n_regions (int): 脑区数量
        
    方法：
        compute_from_fmri: 从fMRI数据计算有效连接
        virtual_perturbation: 虚拟扰动方法
        granger_causality: Granger因果分析
        dcm_estimation: DCM估计
    """
    
    def __init__(self, method: str = 'virtual_perturbation'):
        """
        初始化有效连接计算
        
        参数：
            method (str): 计算方法
                - 'virtual_perturbation': 虚拟扰动法（推荐）
                - 'dcm': 动态因果模型
                - 'granger': Granger因果
                - 'transfer_entropy': 转移熵
        """
        pass
    
    def compute_from_fmri(self,
                         fmri_timeseries: np.ndarray,
                         **kwargs) -> np.ndarray:
        """
        从fMRI时间序列计算有效连接
        
        参数：
            fmri_timeseries (np.ndarray): fMRI数据 (n_timepoints, n_regions)
            **kwargs: 方法特定的参数
            
        返回：
            np.ndarray: 有效连接矩阵 (n_regions, n_regions)
                EC[i,j] 表示脑区j对脑区i的因果影响
        """
        pass
    
    def virtual_perturbation(self,
                            timeseries: np.ndarray,
                            structural_connectivity: Optional[np.ndarray] = None,
                            perturbation_strength: float = 0.1) -> np.ndarray:
        """
        虚拟扰动方法计算有效连接
        
        核心思想：
        1. 在代理大脑模型上进行虚拟扰动
        2. 观察扰动如何传播
        3. 推断因果连接
        
        参数：
            timeseries (np.ndarray): 时间序列数据
            structural_connectivity (np.ndarray, optional): 结构连接（作为先验）
            perturbation_strength (float): 扰动强度
            
        返回：
            np.ndarray: 有效连接矩阵
            
        参考：
            Luo et al. (2025) Nature Methods
        """
        pass
    
    def granger_causality(self,
                         timeseries: np.ndarray,
                         max_lag: int = 5) -> np.ndarray:
        """
        Granger因果分析
        
        如果X的历史信息能改进Y的预测，则X Granger因果Y
        
        参数：
            timeseries (np.ndarray): 时间序列 (n_timepoints, n_regions)
            max_lag (int): 最大滞后阶数
            
        返回：
            np.ndarray: Granger因果矩阵 (n_regions, n_regions)
        """
        pass
    
    def dcm_estimation(self,
                      timeseries: np.ndarray,
                      task_design: Optional[np.ndarray] = None,
                      model_structure: Optional[np.ndarray] = None) -> Dict:
        """
        动态因果模型估计
        
        参数：
            timeseries (np.ndarray): 时间序列数据
            task_design (np.ndarray, optional): 任务设计矩阵
            model_structure (np.ndarray, optional): 模型结构（先验连接）
            
        返回：
            Dict: DCM估计结果，包含：
                - 'A': 内在连接矩阵
                - 'B': 任务调制矩阵
                - 'C': 输入矩阵
                
        参考：
            Friston et al. (2003) "Dynamic causal modelling" NeuroImage
        """
        pass
    
    def transfer_entropy(self,
                        timeseries: np.ndarray,
                        lag: int = 1) -> np.ndarray:
        """
        转移熵分析
        
        参数：
            timeseries (np.ndarray): 时间序列
            lag (int): 时间滞后
            
        返回：
            np.ndarray: 转移熵矩阵
        """
        pass
    
    def task_modulated_connectivity(self,
                                   rest_timeseries: np.ndarray,
                                   task_timeseries: np.ndarray) -> Dict[str, np.ndarray]:
        """
        计算任务调制的有效连接
        
        对比静息态和任务态的有效连接差异
        
        参数：
            rest_timeseries (np.ndarray): 静息态时间序列
            task_timeseries (np.ndarray): 任务态时间序列
            
        返回：
            Dict[str, np.ndarray]: 包含
                - 'rest_ec': 静息态有效连接
                - 'task_ec': 任务态有效连接
                - 'modulation': 任务调制（差异）
        """
        pass
    
    def validate_connectivity(self,
                             ec_matrix: np.ndarray,
                             ground_truth: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        验证有效连接的质量
        
        参数：
            ec_matrix (np.ndarray): 估计的有效连接
            ground_truth (np.ndarray, optional): 真实有效连接（如果有）
            
        返回：
            Dict[str, float]: 验证指标
        """
        pass
