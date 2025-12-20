"""
ODE仿真器模块

功能说明：
    生成基于ODE动力学方程的仿真数据集，用于训练神经算子。

主要类：
    ODESimulator: ODE仿真数据生成器

输入：
    - 动力学模型配置
    - 连接矩阵（有效连接、白质连接）
    - 刺激参数
    - 仿真参数（时间范围、时间步长等）

输出：
    - 仿真数据集：包含输入（连接矩阵、刺激函数）和输出（时间序列）
    - 元数据：仿真参数、数据统计信息
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import sys
import os

# 添加父目录到路径以导入其他模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynamics.ode_models import ODEModel, EIModel
from dynamics.balloon_model import BalloonModel
from simulation.stimulation_generator import StimulationGenerator

class ODESimulator:
    """
    ODE仿真数据生成器
    """
    
    def __init__(self, 
                 n_nodes: int = 246, 
                 dt: float = 0.1, 
                 duration: float = 20000.0, # 20 seconds
                 model_type: str = 'EI',
                 model_params: Optional[Dict] = None):
        
        self.n_nodes = n_nodes
        self.dt = dt
        self.duration = duration
        self.time_points = np.arange(0, duration, dt)
        self.n_time_steps = len(self.time_points)
        
        # 初始化模型
        if model_type == 'EI':
            self.model = EIModel(n_nodes, model_params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
        # 初始化血氧动力学模型
        self.balloon_model = BalloonModel()
        
        # 初始化刺激生成器
        self.stim_generator = StimulationGenerator(n_nodes, dt, duration)
        
    def run_simulation(self, 
                       connectivity: np.ndarray, 
                       stimulus: Optional[np.ndarray] = None,
                       stimulus_config: Optional[Dict] = None,
                       noise_level: float = 0.01,
                       noise_seed: Optional[int] = None,
                       initial_state: Optional[np.ndarray] = None,
                       sampling_interval: float = 50.0) -> Dict:
        """
        运行单次仿真
        
        参数:
            connectivity: 连接矩阵 (n_nodes, n_nodes)
            stimulus: 外部刺激 (n_time_steps, n_nodes)
            stimulus_config: 刺激配置字典 (用于复现)
            noise_level: 噪声水平
            noise_seed: 噪声随机种子
            initial_state: 初始状态
            sampling_interval: 采样时间间隔 (ms), 默认为50ms
            
        返回:
            results: 包含神经活动、BOLD信号等的字典
        """
        # 设置连接矩阵
        self.model.params['C'] = connectivity
        
        # 生成噪声
        noise, noise_config = self.stim_generator.generate_noise(sigma=noise_level, color='ou', seed=noise_seed)
        
        # 组合输入 (刺激 + 噪声)
        total_input = noise
        if stimulus is not None:
            total_input += stimulus
            
        # 初始状态
        if initial_state is None:
            # 随机初始化
            # EI模型状态大小为 2 * n_nodes
            initial_state = np.random.rand(2 * self.n_nodes) * 0.1
            
        # 欧拉积分
        # 注意：为了支持时变输入，我们需要手动积分而不是使用odeint
        # odeint通常假设参数是常数，或者需要传入函数来插值输入
        
        # 策略：只保存降采样后的神经活动以节省内存
        downsample_factor = int(sampling_interval / self.dt)
        if downsample_factor < 1: downsample_factor = 1
        
        n_saved_steps = (self.n_time_steps + downsample_factor - 1) // downsample_factor
        saved_neural_activity = np.zeros((n_saved_steps, self.n_nodes), dtype=np.float32)
        
        current_state = initial_state
        saved_idx = 0
        
        # 保存初始状态
        if 0 % downsample_factor == 0:
            saved_neural_activity[0] = current_state[:self.n_nodes]
            saved_idx += 1
            
        print(f"Starting ODE simulation: {self.n_time_steps} steps.")
        for i in range(1, self.n_time_steps):
            if i % 10000 == 0:
                print(f"Progress: {i}/{self.n_time_steps} ({i/self.n_time_steps*100:.1f}%)", end='\r')
                
            t = self.time_points[i-1]
            u = total_input[i-1]
            
            # 计算导数
            d_state = self.model.dynamics(t, current_state, stimulus=u)
            
            # 更新状态 (Euler method)
            current_state = current_state + d_state * self.dt
            
            # 简单的边界限制防止发散 (可选)
            current_state = np.clip(current_state, 0, 1)
            
            # 降采样保存 (只保存兴奋性群体 E)
            if i % downsample_factor == 0:
                if saved_idx < n_saved_steps:
                    saved_neural_activity[saved_idx] = current_state[:self.n_nodes]
                    saved_idx += 1
            
        print(f"Progress: {self.n_time_steps}/{self.n_time_steps} (100.0%)")
        print("Simulation finished. Computing BOLD signal...")
            
        # 提取兴奋性群体活动作为BOLD模型的输入
        neural_activity_down = saved_neural_activity
        time_points_down = self.time_points[::downsample_factor]
        
        # 计算BOLD信号
        # 注意：BalloonModel期望时间单位为秒(s)，而仿真器使用毫秒(ms)
        # 因此需要将时间点转换为秒
        time_points_sec = time_points_down / 1000.0
        bold_signal = self.balloon_model.compute_bold(neural_activity_down, time_points_sec)
        
        return {
            'time_points': time_points_down,
            'neural_activity': neural_activity_down,
            'bold_signal': bold_signal,
            'stimulus_config': {
                'task_stimulus': stimulus_config,
                'noise_config': noise_config
            },
            'metadata': {
                'model_type': 'EI_ODE',
                'dt': self.dt,
                'sampling_interval': sampling_interval
            }
        }
