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
                 dt: float = 0.1, 
                 duration: float = 60000.0, 
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
                       stimulus: Optional[Union[np.ndarray, Callable]] = None,
                       noise_level: float = 0.01,
                       initial_state: Optional[np.ndarray] = None) -> Dict:
        """
        运行单次PDE仿真
        """
        # 设置拉普拉斯矩阵
        self.model.set_laplacian(connectivity)
        
        # 内存优化：不预先生成所有噪声
        # noise = self.stim_generator.generate_noise(sigma=noise_level, color='white')
        
        if initial_state is None:
            # Wave equation state: [u, v]
            initial_state = np.zeros(2 * self.n_nodes)
            
        # 内存优化：如果节点数太多，不保存所有状态？
        # 但我们需要计算BOLD，需要历史。
        # 如果N很大(32k)，T=40k，states大小为 40000 * 64000 * 8 bytes = 20GB!
        # 我们必须在线计算BOLD或者降采样保存。
        
        # 策略：只保存降采样后的神经活动
        downsample_factor = int(10 / self.dt) # 10ms resolution
        if downsample_factor < 1: downsample_factor = 1
        
        n_saved_steps = (self.n_time_steps + downsample_factor - 1) // downsample_factor
        saved_neural_activity = np.zeros((n_saved_steps, self.n_nodes), dtype=np.float32)
        
        current_state = initial_state
        saved_idx = 0
        
        # 保存初始状态
        if 0 % downsample_factor == 0:
             saved_neural_activity[0] = current_state[:self.n_nodes]
             saved_idx += 1
        
        for i in range(1, self.n_time_steps):
            t = self.time_points[i-1]
            
            # 生成当前步噪声
            noise_t = np.random.normal(0, noise_level, self.n_nodes)
            
            # 获取当前步刺激
            u = noise_t
            if stimulus is not None:
                if callable(stimulus):
                    u += stimulus(t)
                elif isinstance(stimulus, np.ndarray):
                    u += stimulus[i-1]
            
            d_state = self.model.dynamics(t, current_state, stimulus=u)
            current_state = current_state + d_state * self.dt
            
            # 简单的数值稳定性限制
            current_state = np.clip(current_state, -10, 10)
            
            # 降采样保存
            if i % downsample_factor == 0:
                saved_neural_activity[saved_idx] = current_state[:self.n_nodes]
                saved_idx += 1
            
        # 提取u作为神经活动
        neural_activity = saved_neural_activity
        
        # 归一化到0-1之间以便输入Balloon模型 (假设u代表膜电位偏差)
        # sigmoid已经在模型里了，但这里我们取出来的值可能需要调整
        neural_activity_norm = 1.0 / (1.0 + np.exp(-neural_activity))
        
        # Balloon模型计算
        # 注意：Balloon模型内部也会消耗内存，如果N很大。
        # BalloonModel.compute_bold 需要 (T_down, N)
        # T_down ~ 2000, N ~ 32000 -> 64M floats -> 256MB. 这是可以接受的。
        
        time_points_down = self.time_points[::downsample_factor]
        # 确保长度匹配
        time_points_down = time_points_down[:len(neural_activity_norm)]
        
        bold_signal = self.balloon_model.compute_bold(neural_activity_norm, time_points_down)
        
        return {
            'time_points': time_points_down,
            'neural_activity': neural_activity_norm, # 已经是降采样的
            'bold_signal': bold_signal,
            # 'stimulus': ... # 刺激太大，不保存完整历史
        }
