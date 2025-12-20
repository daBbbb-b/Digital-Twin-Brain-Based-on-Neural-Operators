"""
刺激生成器模块

功能说明：
    生成多样化的刺激函数，用于ODE和PDE仿真。

主要类：
    StimulationGenerator: 刺激函数生成器

输入：
    - 刺激类型（脉冲、正弦、阶跃等）
    - 刺激参数（幅度、频率、持续时间等）
    - 目标区域

输出：
    - 刺激函数（时间或时空函数）

说明：
    支持多尺度刺激：
    - 神经递质层面：模拟神经调质的缓慢变化
    - 平均发放率层面：模拟神经元群体的快速活动
"""

import numpy as np
from typing import Callable, Dict, List, Optional, Tuple, Union


class StimulationGenerator:
    """
    刺激函数生成器
    """
    
    def __init__(self, n_nodes: int, dt: float, duration: float):
        self.n_nodes = n_nodes
        self.dt = dt
        self.duration = duration
        self.time_points = np.arange(0, duration, dt)
        self.n_time_steps = len(self.time_points)
        
    def generate_boxcar(self, 
                        onset: float, 
                        duration: float, 
                        amplitude: float = 1.0, 
                        nodes: Optional[List[int]] = None) -> Tuple[np.ndarray, Dict]:
        """
        生成Boxcar（矩形波）刺激
        
        参数:
            onset: 开始时间 (ms)
            duration: 持续时间 (ms)
            amplitude: 幅度
            nodes: 受刺激的节点索引列表。如果为None，则所有节点受刺激。
            
        返回:
            stimulus: (n_time_steps, n_nodes)
            config: 刺激配置字典
        """
        stimulus = np.zeros((self.n_time_steps, self.n_nodes))
        
        start_idx = int(onset / self.dt)
        end_idx = int((onset + duration) / self.dt)
        
        # 边界检查
        start_idx = max(0, start_idx)
        end_idx = min(self.n_time_steps, end_idx)
        
        if nodes is None:
            stimulus[start_idx:end_idx, :] = amplitude
        else:
            stimulus[start_idx:end_idx, nodes] = amplitude
            
        config = {
            'type': 'boxcar',
            'params': {
                'onset': onset,
                'duration': duration,
                'amplitude': amplitude,
                'nodes': nodes
            }
        }
            
        return stimulus, config

    def generate_noise(self, 
                       sigma: float = 0.01, 
                       color: str = 'white', 
                       tau_noise: float = 50.0,
                       seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """
        生成噪声刺激
        
        参数:
            sigma: 噪声标准差
            color: 'white' (白噪声) 或 'ou' (Ornstein-Uhlenbeck过程/红噪声)
            tau_noise: OU过程的时间常数 (ms)
            seed: 随机种子
            
        返回:
            noise: (n_time_steps, n_nodes)
            config: 噪声配置字典
        """
        if seed is not None:
            np.random.seed(seed)
            
        if color == 'white':
            noise = np.random.normal(0, sigma, (self.n_time_steps, self.n_nodes))
        
        elif color == 'ou':
            # Ornstein-Uhlenbeck process
            # dx = -x/tau * dt + sigma * sqrt(2/tau) * dW
            noise = np.zeros((self.n_time_steps, self.n_nodes))
            x = np.zeros(self.n_nodes)
            
            sqrt_dt = np.sqrt(self.dt)
            factor = sigma * np.sqrt(2.0 / tau_noise)
            decay = 1.0 - self.dt / tau_noise
            
            for i in range(self.n_time_steps):
                x = decay * x + factor * sqrt_dt * np.random.normal(0, 1, self.n_nodes)
                noise[i] = x
        
        config = {
            'type': 'noise',
            'params': {
                'sigma': sigma,
                'color': color,
                'tau_noise': tau_noise,
                'seed': seed
            }
        }
                
        return noise, config            

    def combine_stimuli(self, stimuli_list: List[np.ndarray]) -> np.ndarray:
        """
        组合多个刺激（叠加）
        """
        if not stimuli_list:
            return np.zeros((self.n_time_steps, self.n_nodes))
        return sum(stimuli_list)
