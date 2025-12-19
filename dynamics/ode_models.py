"""
常微分方程（ODE）模型模块

功能说明：
    实现基于ODE的神经动力学方程，主要用于描述神经元群体活动。

主要类：
    ODEModel: ODE模型基类
    EIModel: 兴奋-抑制（E-I）模型

输入：
    - 初始状态：神经元活动的初始值
    - 连接矩阵：有效连接或白质结构连接
    - 刺激函数：外部输入刺激
    - 时间参数：仿真时长、时间步长等

输出：
    - 时间序列：神经元活动随时间的演化
    - 状态变量：E群体和I群体的活动

参考：
    Pang et al. (2023) "Geometric constraints on human brain function" 
    公式(10-16)：基于EI模型的ODE方程
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
    兴奋-抑制（E-I）模型
    
    基于Wilson-Cowan模型或类似的神经质量模型。
    每个节点包含一个兴奋性群体(E)和一个抑制性群体(I)。
    
    状态变量 state: [E_1, ..., E_n, I_1, ..., I_n] (大小为 2 * n_nodes)
    """
    
    def __init__(self, n_nodes: int, params: Optional[Dict] = None):
        super().__init__(n_nodes, params)
        
        # 默认参数 (参考 Deco et al. 或 Pang et al.)
        self.default_params = {
            'tau_E': 10.0,   # 兴奋性时间常数 (ms)
            'tau_I': 20.0,   # 抑制性时间常数 (ms)
            'w_EE': 1.5,     # E -> E 自连接权重
            'w_EI': 1.0,     # I -> E 连接权重
            'w_IE': 1.0,     # E -> I 连接权重
            'w_II': 0.0,     # I -> I 连接权重
            'G': 1.0,        # 全局耦合强度 (用于长程连接)
            'C': None,       # 连接矩阵 (n_nodes x n_nodes), 默认为零矩阵
            'a_E': 310,      # 增益函数参数
            'b_E': 125,      # 增益函数参数
            'd_E': 0.16,     # 增益函数参数
            'a_I': 615,      # 增益函数参数
            'b_I': 177,      # 增益函数参数
            'd_I': 0.087,    # 增益函数参数
        }
        
        # 更新参数
        if params:
            self.default_params.update(params)
        self.params = self.default_params
        
        # 初始化连接矩阵
        if self.params['C'] is None:
            self.params['C'] = np.zeros((n_nodes, n_nodes))
            
    def sigmoid_E(self, x):
        """兴奋性群体的激活函数"""
        # H(x) = (a*x - b) / (1 - exp(-d*(a*x - b)))
        # 这是一个常用的非线性传递函数
        # 为了数值稳定性，可以使用简单的sigmoid: 1 / (1 + exp(-x))
        # 这里使用简化的sigmoid形式，或者参考Pang et al.的具体公式
        # 假设使用简单的sigmoid:
        return 1.0 / (1.0 + np.exp(-x))

    def sigmoid_I(self, x):
        """抑制性群体的激活函数"""
        return 1.0 / (1.0 + np.exp(-x))

    def dynamics(self, t: float, state: np.ndarray, stimulus: Optional[np.ndarray] = None) -> np.ndarray:
        """
        计算微分方程的导数
        
        state: 形状为 (2 * n_nodes,)
        stimulus: 形状为 (n_nodes,)，仅作用于E群体（通常假设）
        """
        n = self.n_nodes
        E = state[:n]
        I = state[n:]
        
        # 获取参数
        tau_E = self.params['tau_E']
        tau_I = self.params['tau_I']
        w_EE = self.params['w_EE']
        w_EI = self.params['w_EI']
        w_IE = self.params['w_IE']
        w_II = self.params['w_II']
        G = self.params['G']
        C = self.params['C'] # 连接矩阵
        
        # 外部输入
        u = stimulus if stimulus is not None else np.zeros(n)
        
        # 计算输入电流
        # E群体的输入: 自兴奋 + 长程兴奋(来自其他节点的E) - 局部抑制 + 外部刺激
        # 注意：长程连接通常是 E -> E
        # C @ E 表示来自其他节点的输入
        input_E = w_EE * E - w_EI * I + G * (C @ E) + u
        
        # I群体的输入: 局部兴奋 - 自抑制
        input_I = w_IE * E - w_II * I
        
        # 计算导数
        dE_dt = (-E + self.sigmoid_E(input_E)) / tau_E
        dI_dt = (-I + self.sigmoid_I(input_I)) / tau_I
        
        return np.concatenate([dE_dt, dI_dt])
