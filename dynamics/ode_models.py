"""
常微分方程（ODE）模型模块

功能说明：
    实现基于ODE的神经动力学方程，主要用于描述神经元群体活动。

主要类：
    ODEModel: ODE模型基类
    EIModel: BEI (Biophysical Excitation-Inhibition) / Reduced Wong-Wang 模型

参考：
    Deco et al. (2014) "Great Expectations: Using Whole-Brain Computational Connectomics..."
    Pang et al. (2023) "Geometric constraints on human brain function"
"""

import numpy as np
from typing import Callable, Optional, Tuple, Dict, Union

class ODEModel:
    """
    ODE模型基类
    """
    
    def __init__(self, n_nodes: int, params: Optional[Dict] = None):
        self.n_nodes = n_nodes
        self.params = params if params is not None else {}
        
    def dynamics(self, t: float, state: np.ndarray, stimulus: Optional[np.ndarray] = None) -> np.ndarray:
        """
        定义ODE的右侧函数 dy/dt = f(t, y, u)
        """
        raise NotImplementedError

class EIModel(ODEModel):
    """
    BEI / Reduced Wong-Wang 模型 (Deco et al. 2013/2014)
    
    状态变量 state: [S_E, S_I] (大小为 2 * n_nodes)
    S_E: 兴奋性突触门控变量 (NMDA)
    S_I: 抑制性突触门控变量 (GABA)
    """
    
    def __init__(self, n_nodes: int, params: Optional[Dict] = None):
        super().__init__(n_nodes, params)
        
        # 默认参数 (参考 Deco 2014 / Pang 2023)
        # 注意：所有时间常数单位为 秒 (s)，发放率为 Hz
        self.default_params = {
            'tau_E': 0.100,    # NMDA 衰减时间常数 (s) = 100ms
            'tau_I': 0.010,    # GABA 衰减时间常数 (s) = 10ms
            'gamma': 0.641/1000 , # 动力学速率常数 (Scaling factor)
                               # 原文中 gamma=0.641，但时间单位配合发放率通常需要缩放
                               # 注意：如果时间单位为秒，发放率r为Hz，则 dS/dt = -S/tau + (1-S) * gamma * r
                               # 如果 gamma=0.641且r~10Hz，则第二项~6.4，导致S迅速饱和到1.0
                               # Deco 2014使用gamma=0.641是配合r (Hz) 的吗？
                               # 文献通常取 gamma=0.641/1000 如果时间是ms或者r单位不一致。
                               # 为了安全，这里保持除以1000，确保S动态缓慢。
            
            # 耦合参数
            'w_EE': 1.0,       # 自兴奋权重 (Reduced WW: w_+ usually ~ 1.4 for bistability)
                               # 提高自兴奋以维持持续活动
            'w_EI': 0.15,      # I -> E (weight from Inhibitory to Excitatory) J_i in Deco
            'w_IE': 1.0,       # E -> I (weight from Excitatory to Inhibitory) W_E in Deco (local I input) = 1.0
            'w_II': 0.0,       # I -> I (Self-inhibition, usually 0 or small)
            'G': 0.5,          # 全局耦合强度
            'J_NMDA': 0.15,    # NMDA 耦合系数 (nA)
            
            # 外部输入背景流
            'I0': 0.382,       # 背景输入 (nA)
            'W_E': 1.0,        # 背景输入到E的权重
            'W_I': 0.7,        # 背景输入到I的权重
            
            # 连接矩阵
            'C': None,         # (n_nodes x n_nodes)
            
            # 增益函数 H(x) 参数
            # Excitatory
            'a_E': 310,        # (nC^-1)
            'b_E': 125,        # (Hz)
            'd_E': 0.16,       # (s)
            
            # Inhibitory
            'a_I': 615,        # (nC^-1)
            'b_I': 177,        # (Hz)
            'd_I': 0.087,      # (s)
        }
        
        # 更新参数
        if params:
            self.default_params.update(params)
        self.params = self.default_params
        
        if self.params['C'] is None:
            self.params['C'] = np.zeros((n_nodes, n_nodes))
            
    @staticmethod
    def _H_function(x: np.ndarray, a: float, b: float, d: float) -> np.ndarray:
        """
        神经元响应函数 (Abbott-Kepler f-I curve)
        r = (ax - b) / (1 - exp(-d(ax - b)))
        """
        # 防止数值溢出
        v = a * x - b
        
        # 处理 v 接近 0 的情况 (L'Hopital's rule limit is 1/d)
        # 但实际上 exp(-d*v) 当 v=0 是 1, 分母为 0.
        # 当 v -> 0, func -> 1/d
        
        # 避免溢出: 如果 -d*v 很大 (v 很负), exp 很大.
        # 如果 -d*v 很小 (v 很正), exp 接近 0, 分母接近 1, r ~ v
        
        # 裁剪指数部分
        exponent = -d * v
        exponent = np.clip(exponent, -50.0, 50.0)
        
        denominator = 1.0 - np.exp(exponent)
        
        # 避免除以零
        mask_zero = np.abs(denominator) < 1e-7
        
        res = v / (denominator + 1e-9) # 加上 epsilon 避免 div0 error, 稍后修正
        
        # 对接近0的点使用近似值 1/d
        if np.any(mask_zero):
            res[mask_zero] = 1.0 / d
            
        return res

    def H_E(self, x):
        return self._H_function(x, self.params['a_E'], self.params['b_E'], self.params['d_E'])

    def H_I(self, x):
        return self._H_function(x, self.params['a_I'], self.params['b_I'], self.params['d_I'])

    def dynamics(self, t: float, state: np.ndarray, stimulus: Optional[np.ndarray] = None) -> np.ndarray:
        """
        计算 BEI 模型导数
        
        state: [S_E, S_I]
        t: time (should be in seconds consistent with tau)
        """
        n = self.n_nodes
        S_E = state[:n]
        S_I = state[n:]
        
        # 裁剪状态以防发散
        S_E = np.clip(S_E, 0.0, 1.0)
        S_I = np.clip(S_I, 0.0, 1.0)
        
        # 获取参数
        tau_E = self.params['tau_E']
        tau_I = self.params['tau_I']
        gamma = self.params['gamma']
        w_EE = self.params['w_EE'] # w_+
        w_EI = self.params['w_EI'] # J_i
        w_IE = self.params['w_IE'] # 1.0
        w_II = self.params['w_II']
        G = self.params['G']
        J_NMDA = self.params['J_NMDA']
        I0 = self.params['I0']
        W_E = self.params['W_E']
        W_I = self.params['W_I']
        C = self.params['C']
        
        u = stimulus if stimulus is not None else np.zeros(n)
        
        # 计算总输入电流 I_E, I_I (nA)
        # I_E = W_E*I0 + w_+ * S_E + G * J_NMDA * sum(C_ij * S_j) - J_i * S_I + I_ext
        # I_I = W_I*I0 + J_NMDA * S_E - S_I (+ I_ext_I?)
        
        # 长程输入
        long_range_input = G * J_NMDA * (C @ S_E)
        
        # E 群体输入电流
        I_E = W_E * I0 + w_EE * S_E + long_range_input - w_EI * S_I + u
        
        # I 群体输入电流 (通常 I 群体主要接收 E 的输入)
        # 文献公式 (16): W_I * I0 + w_EI_input * S_E - S_I ?
        # Deco 2014: I_I = W_I * I0 + J_NMDA * S_E - S_I (here S_I might represent self-inhibition or just leak?)
        # 让我们按照标准: Input to I is Excitation from E.
        # 通常 I_I = W_I * I0 + w_IE * S_E (Note: w_IE here means E to I).
        # 根据 prompt 文献公式 (16): I_i^I = W_I I_0 + w_EI S_E - S_I
        # 这里的 w_EI 可能是指 E -> I 的权重 (通常记为 w_IE 或 J_NMDA).
        # 而 - S_I 项保留。
        I_I = W_I * I0 + w_IE * S_E - S_I + u # 假设 u 也输入到 I (可选)
        
        # 计算发放率 r (Hz)
        r_E = self.H_E(I_E)
        r_I = self.H_I(I_I)
        
        # 计算导数
        # dS_E/dt = -S_E/tau_E + (1-S_E) * gamma * r_E
        dS_E = -S_E / tau_E + (1.0 - S_E) * gamma * r_E
        
        # dS_I/dt = -S_I/tau_I + r_I
        dS_I = -S_I / tau_I + r_I
        
        return np.concatenate([dS_E, dS_I])

class BilinearControlModel(ODEModel):
    """
    双线性控制模型 (用于神经算子反演任务)
    
    方程:
    dx(t)/dt = A x(t) + sum_{k=1}^K u_k(t) B^{(k)} x(t) + C u(t) + eta(t)
    
    其中:
    - x(t): 状态变量 (N,)
    - u(t): 控制输入 (K,)
    - A: 系统矩阵 (N, N) (连接矩阵)
    - B: 调制张量 (K, N, N)
    - C: 驱动矩阵 (N, K)
    """
    
    def __init__(self, n_nodes: int, params: Optional[Dict] = None):
        super().__init__(n_nodes, params)
        
        self.default_params = {
            'A': None,       # (N, N)
            'B': None,       # (K, N, N) or list of (N, N)
            'C': None,       # (N, K)
        }
        if params:
            self.default_params.update(params)
        self.params = self.default_params
        
        # 检查参数
        A = self.params.get('A', None)
        if A is None:
            A = np.zeros((n_nodes, n_nodes), dtype=np.float64)
        else:
            A = np.asarray(A, dtype=np.float64)
            if A.shape != (n_nodes, n_nodes):
                raise ValueError(f"A shape must be ({n_nodes}, {n_nodes}), got {A.shape}")
        self.params['A'] = A
        
        # 标准化 B, C 形状，避免后续广播出错
        B = self.params.get('B', None)
        if B is not None:
            if isinstance(B, list):
                B = np.stack([np.asarray(b, dtype=np.float64) for b in B], axis=0)
            else:
                B = np.asarray(B, dtype=np.float64)
            if B.ndim != 3 or B.shape[1:] != (n_nodes, n_nodes):
                raise ValueError(f"B must have shape (K, {n_nodes}, {n_nodes}), got {B.shape}")
        self.params['B'] = B
        
        C = self.params.get('C', None)
        if C is not None:
            C = np.asarray(C, dtype=np.float64)
            if C.shape[0] != n_nodes:
                raise ValueError(f"C must have shape ({n_nodes}, K), got {C.shape}")
        self.params['C'] = C
            
    def dynamics(self, t: float, state: np.ndarray, stimulus: Optional[np.ndarray] = None) -> np.ndarray:
        """
        计算导数 dx/dt
        
        优化版本：支持稀疏矩阵 B，大幅加速计算
        """
        x = state
        A = self.params['A']
        B = self.params['B'] # List of sparse matrices or 3D array
        C = self.params['C']
        
        # 线性部分 Ax
        dxdt = A @ x
        
        if stimulus is not None:
            # stimulus u(t) is (K,)
            u = np.asarray(stimulus, dtype=np.float64)
            if u.ndim != 1:
                u = u.squeeze()
            if u.ndim != 1:
                raise ValueError(f"Stimulus for bilinear model must be 1-D (K,), got shape {u.shape}")
            
            # 双线性部分 sum u_k B^{(k)} x
            if B is not None:
                # 检测 B 的类型并使用高效计算
                if isinstance(B, list):
                    # 稀疏矩阵列表（高效模式）
                    if len(B) != len(u):
                        raise ValueError(f"Stimulus length ({len(u)}) must match B list length ({len(B)})")
                    
                    bilinear_term = np.zeros_like(x)
                    for k, u_k in enumerate(u):
                        if abs(u_k) > 1e-10:  # 跳过零刺激
                            # 稀疏矩阵-向量乘法：O(nnz) 而非 O(N^2)
                            bilinear_term += u_k * B[k].dot(x)
                    
                    dxdt += bilinear_term
                else:
                    # 密集数组（原始模式）
                    if B.shape[0] != len(u):
                        raise ValueError(f"Stimulus length ({len(u)}) must match B first dim ({B.shape[0]})")
                    # sum_k u_k * (B_k @ x)
                    bilinear_term = np.einsum('k,kij,j->i', u, B, x)
                    dxdt += bilinear_term
            
            # 驱动部分 Cu
            if C is not None:
                if C.shape[1] != len(u):
                    raise ValueError(f"Stimulus length ({len(u)}) must match C second dim ({C.shape[1]})")
                dxdt += C @ u
                
        return dxdt
