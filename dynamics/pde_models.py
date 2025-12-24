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
        计算图拉普拉斯矩阵
        优化：使用对称归一化拉普拉斯矩阵 (Symmetric Normalized Laplacian)
        L_sys = D^-1/2 * L * D^-1/2 = I - D^-1/2 * A * D^-1/2
        这消除了节点度数不均匀对扩散的影响，更接近几何拉普拉斯算子。
        """
        if sparse.issparse(adjacency_matrix):
            # 计算度数
            degree = np.array(adjacency_matrix.sum(axis=1)).flatten()
            
            # 避免除零
            d_inv_sqrt = np.power(degree, -0.5)
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
            
            # 构建对角矩阵 D^-1/2
            D_inv_sqrt = sparse.diags(d_inv_sqrt)
            
            # L_sym = I - D^-1/2 * A * D^-1/2
            ident = sparse.eye(self.n_nodes)
            self.laplacian = ident - D_inv_sqrt @ adjacency_matrix @ D_inv_sqrt
            
            # 强制转换为 CSR 格式，优化后续的矩阵向量乘法 (SpMV)
            self.laplacian = self.laplacian.tocsr()
        else:
            degree = np.sum(adjacency_matrix, axis=1)
            d_inv_sqrt = np.power(degree, -0.5)
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
            D_inv_sqrt = np.diag(d_inv_sqrt)
            
            ident = np.eye(self.n_nodes)
            self.laplacian = ident - D_inv_sqrt @ adjacency_matrix @ D_inv_sqrt
        
    def dynamics(self, t: float, state: np.ndarray, stimulus: Optional[np.ndarray] = None) -> np.ndarray:
        raise NotImplementedError

class WaveEquationModel(PDEModel):
    """
    阻尼波动方程模型 (Damped Wave Equation)
    
    d^2u/dt^2 + gamma * du/dt + c^2 * L * u = alpha * F(u) + stimulus
    
    状态变量 state: [u, v] 其中 v = du/dt
    """
    
    def __init__(self, n_nodes: int, params: Optional[Dict] = None):
        super().__init__(n_nodes, params)
        
        # 注意：时间单位统一为毫秒 (ms)，空间单位为毫米 (mm)
        self.default_params = {
            'gamma': 0.23,   # 阻尼系数 (ms^-1)
                             # 对应 Friston 2003 的 2*gamma_s, gamma_s ~ 116 s^-1 = 0.116 ms^-1
            'c': 2.0,        # 传播速度 (mm/ms) 
            'alpha': 100.0,  # 非线性项增益 (最大发放率，单位 Hz 或 arbitrary)
            'L': None,       # 拉普拉斯矩阵 (基于 mm 单位的网格计算)
            'activation': 'sigmoid' # 非线性激活函数
        }
        if params:
            self.default_params.update(params)
        self.params = self.default_params
        
        if self.params['L'] is not None:
            self.laplacian = self.params['L']
            
    @staticmethod
    def _stable_sigmoid(x: np.ndarray, clip: float = 60.0) -> np.ndarray:
        """
        数值稳定 sigmoid/logistic：
        - 对输入裁剪避免 exp 溢出
        - 分段公式避免极端情况下 overflow/underflow
        """
        x = np.asarray(x, dtype=np.float64)
        x = np.clip(x, -clip, clip)
        return np.where(
            x >= 0,
            1.0 / (1.0 + np.exp(-x)),
            np.exp(x) / (1.0 + np.exp(x)),
        )

    def sigmoid(self, x):
        """Sigmoid函数（带数值稳定保护）"""
        return self._stable_sigmoid(x)
            
    def dynamics(self, t: float, state: np.ndarray, stimulus: Optional[np.ndarray] = None) -> np.ndarray:
        """
        计算导数
        state: [u, v] (2 * n_nodes)
        """
        n = self.n_nodes
        state = np.asarray(state, dtype=np.float64)
        u = state[:n]
        v = state[n:]

        # 数值保护：避免 u/v 在显式 Euler 积分中发散后进入 sigmoid 触发 exp overflow
        u = np.clip(u, -60.0, 60.0)
        v = np.clip(v, -200.0, 200.0)
        
        gamma = self.params['gamma']
        c = self.params['c']
        alpha = self.params.get('alpha', 1.0) # 获取非线性增益，默认1.0
        
        if self.laplacian is None:
            raise ValueError("Laplacian matrix not set. Call set_laplacian() first.")
            
        # 外部输入
        inp = stimulus if stimulus is not None else np.zeros(n, dtype=np.float64)
        
        # 波动方程
        # du/dt = v
        # dv/dt = -gamma * v - c^2 * L * u + alpha * sigmoid(u) + inp
        
        du_dt = v
        
        # 计算 Laplacian * u
        Lu = self.laplacian @ u
        
        # 非线性项 (可选，模拟神经元激活)
        nonlinear_term = self.sigmoid(u)
        
        dv_dt = -gamma * v - (c**2) * Lu + alpha * nonlinear_term + inp
        
        return np.concatenate([du_dt, dv_dt])
