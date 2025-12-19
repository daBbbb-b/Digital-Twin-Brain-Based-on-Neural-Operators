"""
偏微分方程（PDE）模型模块

功能说明：
    实现基于PDE的神经动力学方程，主要用于描述皮层上的神经活动扩散。

主要类：
    PDEModel: PDE模型基类
    WaveEquationModel: 波动方程模型 (Damped Wave Equation)

输入：
    - 初始状态：皮层上的神经活动分布
    - 皮层结构连接：定义扩散路径的几何结构 (Graph Laplacian)
    - 刺激函数：空间和时间相关的外部刺激
    - 扩散参数：扩散系数等

输出：
    - 时空序列：皮层活动随时间和空间的演化
    - 活动分布：各个皮层位置的活动强度

参考：
    Pang et al. (2023) "Geometric constraints on human brain function"
    公式(9)：PDE扩散/波动方程
"""

import numpy as np
from typing import Callable, Optional, Tuple, Dict, Union
from scipy.sparse import csr_matrix, diags
import scipy.sparse as sparse

class PDEModel:
    """
    PDE模型基类
    """
    
    def __init__(self, n_nodes: int, params: Optional[Dict] = None):
        self.n_nodes = n_nodes
        self.params = params if params is not None else {}
        self.laplacian = None
        
    def set_laplacian(self, adjacency_matrix: Union[np.ndarray, csr_matrix]):
        """
        计算图拉普拉斯矩阵 L = D - A
        支持稀疏矩阵
        """
        if sparse.issparse(adjacency_matrix):
            degree = np.array(adjacency_matrix.sum(axis=1)).flatten()
            D = sparse.diags(degree)
            self.laplacian = D - adjacency_matrix
        else:
            degree = np.sum(adjacency_matrix, axis=1)
            D = np.diag(degree)
            self.laplacian = D - adjacency_matrix
        
        # 归一化 (可选)
        # d_inv_sqrt = np.power(degree, -0.5)
        # d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0
        # D_inv_sqrt = np.diag(d_inv_sqrt)
        # self.laplacian = np.eye(self.n_nodes) - D_inv_sqrt @ adjacency_matrix @ D_inv_sqrt
        
    def dynamics(self, t: float, state: np.ndarray, stimulus: Optional[np.ndarray] = None) -> np.ndarray:
        raise NotImplementedError

class WaveEquationModel(PDEModel):
    """
    阻尼波动方程模型 (Damped Wave Equation)
    
    d^2u/dt^2 + gamma * du/dt + c^2 * L * u = F(u) + stimulus
    
    状态变量 state: [u, v] 其中 v = du/dt
    """
    
    def __init__(self, n_nodes: int, params: Optional[Dict] = None):
        super().__init__(n_nodes, params)
        
        self.default_params = {
            'gamma': 0.5,    # 阻尼系数
            'c': 10.0,       # 传播速度
            'L': None,       # 拉普拉斯矩阵
            'activation': 'sigmoid' # 非线性激活函数
        }
        if params:
            self.default_params.update(params)
        self.params = self.default_params
        
        if self.params['L'] is not None:
            self.laplacian = self.params['L']
            
    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))
            
    def dynamics(self, t: float, state: np.ndarray, stimulus: Optional[np.ndarray] = None) -> np.ndarray:
        """
        计算导数
        state: [u, v] (2 * n_nodes)
        """
        n = self.n_nodes
        u = state[:n]
        v = state[n:]
        
        gamma = self.params['gamma']
        c = self.params['c']
        
        if self.laplacian is None:
            raise ValueError("Laplacian matrix not set. Call set_laplacian() first.")
            
        # 外部输入
        inp = stimulus if stimulus is not None else np.zeros(n)
        
        # 波动方程
        # du/dt = v
        # dv/dt = -gamma * v - c^2 * L * u + sigmoid(u) + inp
        
        du_dt = v
        
        # 计算 Laplacian * u
        Lu = self.laplacian @ u
        
        # 非线性项 (可选，模拟神经元激活)
        nonlinear_term = self.sigmoid(u)
        
        dv_dt = -gamma * v - (c**2) * Lu + nonlinear_term + inp
        
        return np.concatenate([du_dt, dv_dt])
