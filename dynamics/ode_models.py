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
from typing import Callable, Optional, Tuple, Dict
from scipy.integrate import odeint, solve_ivp


class ODEModel:
    """
    ODE模型基类
    
    功能：
        - 定义ODE系统的通用接口
        - 提供数值求解方法
        - 支持参数化刺激函数
        
    属性：
        n_nodes (int): 节点数量（脑区数量）
        params (Dict): 模型参数
        
    方法：
        dynamics: 定义ODE的右侧函数
        solve: 求解ODE系统
        set_stimulus: 设置刺激函数
    """
    
    def __init__(self, n_nodes: int, params: Optional[Dict] = None):
        """
        初始化ODE模型
        
        参数：
            n_nodes (int): 节点数量
            params (Dict, optional): 模型参数字典
        """
        pass
    
    def dynamics(self, t: float, state: np.ndarray, 
                connectivity: np.ndarray,
                stimulus: Optional[Callable] = None) -> np.ndarray:
        """
        定义ODE右侧函数 dx/dt = f(x, t)
        
        参数：
            t (float): 当前时间
            state (np.ndarray): 当前状态，形状为 (n_nodes * n_variables,)
            connectivity (np.ndarray): 连接矩阵，形状为 (n_nodes, n_nodes)
            stimulus (Callable, optional): 刺激函数 s(t, node_idx)
            
        返回：
            np.ndarray: 状态导数 dx/dt
        """
        pass
    
    def solve(self, 
             initial_state: np.ndarray,
             connectivity: np.ndarray,
             stimulus: Optional[Callable] = None,
             t_span: Tuple[float, float] = (0, 100),
             dt: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
        """
        求解ODE系统
        
        参数：
            initial_state (np.ndarray): 初始状态
            connectivity (np.ndarray): 连接矩阵
            stimulus (Callable, optional): 刺激函数
            t_span (Tuple[float, float]): 时间范围 (t_start, t_end)
            dt (float): 时间步长
            
        返回：
            Tuple[np.ndarray, np.ndarray]: 
                - 时间数组，形状为 (n_timepoints,)
                - 状态轨迹，形状为 (n_timepoints, n_nodes * n_variables)
        """
        pass


class EIModel(ODEModel):
    """
    兴奋-抑制（E-I）模型
    
    功能：
        - 实现E-I神经元群体动力学
        - 考虑兴奋性和抑制性神经元的相互作用
        - 支持多尺度刺激（神经递质层面、平均发放率层面）
        
    状态变量：
        - E: 兴奋性神经元群体活动
        - I: 抑制性神经元群体活动
        
    参数说明：
        - w_ee: E到E的连接权重
        - w_ei: E到I的连接权重
        - w_ie: I到E的连接权重
        - w_ii: I到I的连接权重
        - tau_e: E群体的时间常数
        - tau_i: I群体的时间常数
        - gamma_e: E群体的增益参数
        - gamma_i: I群体的增益参数
        
    参考：
        Pang et al. (2023) 公式(10-16)
    """
    
    def __init__(self, n_nodes: int, params: Optional[Dict] = None):
        """
        初始化E-I模型
        
        参数：
            n_nodes (int): 节点数量
            params (Dict, optional): 模型参数，包含：
                - w_ee, w_ei, w_ie, w_ii: 连接权重
                - tau_e, tau_i: 时间常数
                - gamma_e, gamma_i: 增益参数
        """
        pass
    
    def dynamics(self, t: float, state: np.ndarray,
                connectivity: np.ndarray,
                stimulus: Optional[Callable] = None) -> np.ndarray:
        """
        E-I模型动力学方程
        
        dE/dt = (-E + f(w_ee * E - w_ei * I + s_E + coupling_E)) / tau_e
        dI/dt = (-I + f(w_ie * E - w_ii * I + s_I + coupling_I)) / tau_i
        
        其中 f 是激活函数，coupling是通过连接矩阵传递的耦合项
        
        参数：
            t (float): 当前时间
            state (np.ndarray): 当前状态 [E1, ..., En, I1, ..., In]
            connectivity (np.ndarray): 连接矩阵
            stimulus (Callable, optional): 刺激函数
            
        返回：
            np.ndarray: 状态导数
        """
        pass
    
    def activation_function(self, x: np.ndarray) -> np.ndarray:
        """
        激活函数（如sigmoid或tanh）
        
        参数：
            x (np.ndarray): 输入
            
        返回：
            np.ndarray: 激活后的输出
        """
        pass
