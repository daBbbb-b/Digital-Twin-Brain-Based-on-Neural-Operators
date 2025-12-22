"""
PDE仿真器模块

功能说明：
    生成基于PDE动力学方程的仿真数据集，用于训练神经算子。

主要类：
    PDESimulator: PDE仿真数据生成器

输入：
    - PDE动力学模型配置
    - 皮层结构连接
    - 空间刺激参数
    - 仿真参数

输出：
    - 仿真数据集：包含输入（皮层连接、空间刺激）和输出（时空序列）
    - 元数据
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union,Callable
import sys
import os

# 添加父目录到路径以导入其他模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynamics.pde_models import PDEModel, WaveEquationModel
from dynamics.balloon_model import BalloonModel
from simulation.stimulation_generator import StimulationGenerator

class PDESimulator:
    """
    PDE仿真数据生成器
    """
    
    def __init__(self, 
                 n_nodes: int = 246, 
                 dt: float = 0.05, 
                 duration: float = 200.0, 
                 model_type: str = 'wave',
                 model_params: Optional[Dict] = None):
        
        self.n_nodes = n_nodes
        self.dt = dt
        self.duration = duration
        self.time_points = np.arange(0, duration, dt)
        self.n_time_steps = len(self.time_points)
        
        if model_type == 'wave':
            self.model = WaveEquationModel(n_nodes, model_params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
        self.balloon_model = BalloonModel()
        self.stim_generator = StimulationGenerator(n_nodes, dt, duration)
        
    def run_simulation(self, 
                       connectivity: Union[np.ndarray, object], 
                       vertices: Optional[np.ndarray] = None,
                       faces: Optional[np.ndarray] = None,
                       stimulus: Optional[Union[np.ndarray, Callable]] = None,
                       stimulus_config: Optional[Dict] = None,
                       noise_level: float = 0.01,
                       noise_seed: Optional[int] = None,
                       initial_state: Optional[np.ndarray] = None,
                       sampling_interval: float = 0.05) -> Dict:
        """
        运行单次PDE仿真
        
        参数:
            connectivity: 连接矩阵或拉普拉斯矩阵
            vertices: (N, 3) 顶点坐标 (用于生成空间刺激)
            faces: (M, 3) 面索引 (可选，用于测地距离)
            stimulus: 外部刺激
            stimulus_config: 刺激配置字典 (用于复现)
            noise_level: 噪声水平
            noise_seed: 噪声随机种子
            initial_state: 初始状态
            sampling_interval: 采样时间间隔 (s), 默认为0.05s
            
        返回:
            results: 包含神经活动、BOLD信号等的字典
        """
        # 设置拉普拉斯矩阵
        self.model.set_laplacian(connectivity)
        
        if noise_seed is not None:
            np.random.seed(noise_seed)
            
        # 自动生成刺激 (如果未提供且提供了顶点信息)
        if stimulus is None and vertices is not None:
            # 使用 TaskSchedule 生成刺激
            # 假设 PDE 刺激与 ODE 任务结构类似，或者独立生成
            # 这里我们独立生成 PDE 任务序列
            tasks = self.stim_generator.generate_task_schedule(n_channels=0, n_vertices_pde=self.n_nodes)
            stimulus, stimulus_config = self.stim_generator.generate_pde_stimulus(tasks, vertices, faces)
        
        # 预生成噪声 (内存允许的情况下)
        # PDE 噪声: 空间-时间白噪声
        # noise = np.random.normal(0, 1, (self.n_time_steps, self.n_nodes)) * noise_level
        
        if initial_state is None:
            # Wave equation state: [u, v]
            initial_state = np.zeros(2 * self.n_nodes)
            
        # 运行积分
        state = initial_state
        states = [] # 只保存 u (神经活动)
        
        sampling_steps = int(sampling_interval / self.dt)
        
        for i in range(self.n_time_steps):
            t = self.time_points[i]
            
            # 获取当前时刻刺激
            u_t = None
            if stimulus is not None:
                if callable(stimulus):
                    u_t = stimulus(t)
                else:
                    u_t = stimulus[i]
            
            # 生成当前步噪声
            noise_t = np.random.normal(0, 1, self.n_nodes) * noise_level
            
            # 组合输入
            total_input = noise_t
            if u_t is not None:
                total_input += u_t
                
            # 计算导数
            dydt = self.model.dynamics(t, state, total_input)
            
            # Euler step
            state = state + dydt * self.dt
            
            # 简单的边界限制 (可选)
            # state = np.clip(state, -10, 10)
            
            if i % sampling_steps == 0:
                # 只保存 u (前 n_nodes 个状态)
                states.append(state[:self.n_nodes].copy())
                
        states = np.array(states)
        
        # 计算 BOLD 信号
        # BalloonModel.compute_bold 期望 t_span 是一个时间点数组，而不是单个 float
        time_points_downsampled = self.time_points[::sampling_steps]
        bold = self.balloon_model.compute_bold(states, time_points_downsampled)
        
        return {
            'time_points': time_points_downsampled,
            #'neural_activity': states,
            'bold_signal': bold,
            'stimulus_config': stimulus_config,
            'initial_state': initial_state,
            'metadata': {
                'model_type': 'Wave_PDE',
                'dt': self.dt,
                'duration': self.duration,
                'sampling_interval': sampling_interval,
                'noise_level': noise_level,
                'noise_seed': noise_seed
            }
        }



