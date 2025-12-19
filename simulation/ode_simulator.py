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
                 duration: float = 60000.0, # 1 minute
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
                       noise_level: float = 0.01,
                       initial_state: Optional[np.ndarray] = None) -> Dict:
        """
        运行单次仿真
        
        参数:
            connectivity: 连接矩阵 (n_nodes, n_nodes)
            stimulus: 外部刺激 (n_time_steps, n_nodes)
            noise_level: 噪声水平
            initial_state: 初始状态
            
        返回:
            results: 包含神经活动、BOLD信号等的字典
        """
        # 设置连接矩阵
        self.model.params['C'] = connectivity
        
        # 生成噪声
        noise = self.stim_generator.generate_noise(sigma=noise_level, color='ou')
        
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
        
        states = np.zeros((self.n_time_steps, 2 * self.n_nodes))
        states[0] = initial_state
        
        current_state = initial_state
        
        # 预计算参数以加速
        # 这里直接调用model.dynamics，虽然稍微慢一点但更通用
        
        for i in range(1, self.n_time_steps):
            t = self.time_points[i-1]
            u = total_input[i-1]
            
            # 计算导数
            d_state = self.model.dynamics(t, current_state, stimulus=u)
            
            # 更新状态 (Euler method)
            current_state = current_state + d_state * self.dt
            
            # 简单的边界限制防止发散 (可选)
            current_state = np.clip(current_state, 0, 1)
            
            states[i] = current_state
            
        # 提取兴奋性群体活动作为BOLD模型的输入
        # 假设前n_nodes是E群体
        neural_activity_E = states[:, :self.n_nodes]
        
        # 下采样神经活动以匹配BOLD模型的时间步长 (如果需要)
        # 这里假设BOLD模型可以处理相同的时间步长，或者我们在BOLD模型内部处理
        # BalloonModel通常需要较小的时间步长以保持稳定，这里dt=0.1ms是足够的
        # 但是BOLD信号本身变化很慢，我们通常只需要每秒保存一个点(TR)
        # 但为了计算准确，我们先计算全部分辨率的BOLD，然后下采样
        
        # 计算BOLD信号
        # 注意：这步可能很慢，如果时间步长很小。
        # 实际fMRI TR通常为0.72s或2s。
        # 我们可以先对神经活动进行降采样，然后再输入Balloon模型，但这会丢失高频信息。
        # 更好的方法是使用原始分辨率计算BOLD，然后降采样。
        
        # 为了演示速度，我们可能需要对神经活动进行降采样，比如每10ms一个点
        downsample_factor = int(10 / self.dt) # 假设目标是10ms分辨率
        if downsample_factor < 1: downsample_factor = 1
        
        neural_activity_down = neural_activity_E[::downsample_factor]
        time_points_down = self.time_points[::downsample_factor]
        
        # 转换单位：Balloon模型通常期望输入已经归一化或处于某种范围内
        # 简单的缩放
        neural_input = neural_activity_down
        
        bold_signal = self.balloon_model.compute_bold(neural_input, time_points_down)
        
        return {
            'time_points': time_points_down,
            'neural_activity': neural_activity_down,
            'bold_signal': bold_signal,
            'stimulus': total_input[::downsample_factor] if stimulus is not None else noise[::downsample_factor]
        }
