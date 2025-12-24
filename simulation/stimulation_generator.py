"""
刺激生成器模块

功能说明：
    生成多样化的刺激函数，用于ODE和PDE仿真。
    支持基于Task的复杂刺激生成，满足神经算子反演需求。

主要类：
    StimulationGenerator: 刺激函数生成器

输入：
    - 刺激类型（脉冲、正弦、阶跃等）
    - 刺激参数（幅度、频率、持续时间等）
    - 目标区域

输出：
    - 刺激函数（时间或时空函数）
"""

import numpy as np
from typing import Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import scipy.sparse as sparse

@dataclass
class TaskEvent:
    """定义一个Task事件"""
    index: int
    start_time: float
    end_time: float
    duration: float
    # ODE 参数
    active_channels: List[int] = field(default_factory=list)
    amplitudes_ode: List[float] = field(default_factory=list)
    # PDE 参数
    seed_vertices: List[int] = field(default_factory=list)
    amplitude_pde: float = 0.0
    sigma_s: float = 10.0

class PDEStimulus:
    """PDE刺激函数包装器，避免生成巨大的 (T, N) 数组"""
    def __init__(self, n_verts, dt, duration, task_data):
        self.n_verts = n_verts
        self.dt = dt
        self.duration = duration
        self.task_data = task_data # List of dicts with 'envelope', 'spatial_map', 'amp'

    def __call__(self, t):
        idx = int(round(t / self.dt))
        # 简单的边界保护
        if idx < 0: idx = 0
        
        u_t = np.zeros(self.n_verts)
        for data in self.task_data:
            # envelope 是 (T,) 数组
            if idx < len(data['envelope']):
                val = data['envelope'][idx]
                if abs(val) > 1e-6:
                    u_t += data['amp'] * val * data['spatial_map']
        return u_t

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
        
    def _smooth_boxcar(self, t: np.ndarray, t0: float, t1: float, rise_time: float = 500.0) -> np.ndarray:
        """
        生成平滑的Boxcar函数 (Sigmoid边界)
        
        s(t) = sigmoid((t - t0)/tau) * (1 - sigmoid((t - t1)/tau))
        """
        # tau 使得 rise_time 对应约 10% 到 90%
        # sigmoid(x) = 1 / (1 + exp(-x))
        # x goes from -3 to 3 covers most transition
        tau = rise_time / 6.0

        def stable_sigmoid(x: np.ndarray, clip: float = 60.0) -> np.ndarray:
            # 数值稳定 sigmoid：先裁剪输入，再用分段公式避免 exp 溢出
            x = np.asarray(x, dtype=np.float64)
            x = np.clip(x, -clip, clip)
            return np.where(
                x >= 0,
                1.0 / (1.0 + np.exp(-x)),
                np.exp(x) / (1.0 + np.exp(x)),
            )

        x1 = (t - t0) / tau
        x2 = (t - t1) / tau
        s1 = stable_sigmoid(x1)
        s2 = stable_sigmoid(x2)
        
        return s1 * (1.0 - s2)

    def generate_task_schedule(self, 
                             n_channels: int, 
                             n_vertices_pde: Optional[int] = None) -> List[TaskEvent]:
        """
        生成 Run 的 Task 调度序列
        
        参数:
            n_channels: ODE刺激通道数 K
            n_vertices_pde: PDE顶点数 (用于随机选择seed)
            
        返回:
            task_list: TaskEvent 列表
        """
        tasks = []
        current_time = 0.0
        task_idx = 0
        
        # 预留开始的一段静息时间
        current_time += np.random.uniform(1000, 3000)
        
        while current_time < self.duration - 5000: # 留出结尾余量
            # Task duration: 5s -15s
            dur = np.random.uniform(5000, 15000)
            if current_time + dur > self.duration:
                dur = self.duration - current_time - 1000
                if dur < 5000: break
            
            t_start = current_time
            t_end = current_time + dur
            
            # ODE Params
            # 随机选择 1-3 个通道
            # 小巧思：ei的n_channels为节点数，可以刺激所有节点,实际上只会选择最多4个节点
            n_active = np.random.randint(1, min(4, n_channels + 1))
            active_chs = np.random.choice(n_channels, size=n_active, replace=False).tolist()
            
            # 幅度采样: [-2.0, -0.5] U [0.5, 2.0]
            amps_ode = []
            for _ in range(n_active):
                if np.random.rand() > 0.5:
                    amp = np.random.uniform(0.5, 2.0)
                else:
                    amp = np.random.uniform(-2.0, -0.5)
                amps_ode.append(amp)
                
            # PDE Params
            seeds = []
            amp_pde = 0.0
            sigma_s = 10.0
            if n_vertices_pde is not None:
                n_seeds = np.random.randint(1, 4)
                seeds = np.random.choice(n_vertices_pde, size=n_seeds, replace=False).tolist()
                if np.random.rand() > 0.5:
                    amp_pde = np.random.uniform(0.5, 2.0)
                else:
                    amp_pde = np.random.uniform(-2.0, -0.5)
                sigma_s = np.random.uniform(5.0, 15.0)
            
            task = TaskEvent(
                index=task_idx,
                start_time=t_start,
                end_time=t_end,
                duration=dur,
                active_channels=active_chs,
                amplitudes_ode=amps_ode,
                seed_vertices=seeds,
                amplitude_pde=amp_pde,
                sigma_s=sigma_s
            )
            tasks.append(task)
            
            current_time = t_end
            task_idx += 1
            
        return tasks

    def generate_ode_stimulus(self, 
                            tasks: List[TaskEvent], 
                            n_channels: int) -> Tuple[np.ndarray, Dict]:
        """
        生成 ODE 刺激 u(t) (K维)
        
        返回:
            u: (n_time_steps, n_channels)
            config: 配置信息
        """
        u = np.zeros((self.n_time_steps, n_channels))
        
        for task in tasks:
            # 时间包络
            envelope = self._smooth_boxcar(self.time_points, task.start_time, task.end_time)
            
            for ch_idx, amp in zip(task.active_channels, task.amplitudes_ode):
                u[:, ch_idx] += amp * envelope
                
        # 添加弱慢噪声 (可选)
        # 这里不添加，保持确定性部分，噪声在外部添加或作为独立项
        
        config = {
            'type': 'task_based_ode',
            'n_channels': n_channels,
            'tasks': [
                {
                    'index': t.index,
                    'range': (t.start_time, t.end_time),
                    'channels': t.active_channels,
                    'amplitudes': t.amplitudes_ode
                }
                for t in tasks
            ]
        }
        return u, config

    def generate_pde_stimulus(self, 
                            tasks: List[TaskEvent], 
                            vertices: np.ndarray,
                            faces: Optional[np.ndarray] = None) -> Tuple[Union[np.ndarray, Callable], Dict]:
        """
        生成 PDE 刺激 u_pde(s, t)
        
        参数:
            vertices: (N, 3) 顶点坐标
            faces: (M, 3) 面索引
            
        返回:
            u_pde: Callable (t -> np.ndarray) or np.ndarray
            config: 配置信息
        """
        n_verts = vertices.shape[0]
        
        # 预计算任务的空间分布和时间包络
        task_data = []
        
        for task in tasks:
            if not task.seed_vertices:
                continue
                
            # 时间包络 (T,)
            envelope = self._smooth_boxcar(self.time_points, task.start_time, task.end_time)
            
            # 空间分布 phi(s) (N,)
            spatial_map = np.zeros(n_verts)
            
            for seed in task.seed_vertices:
                seed_pos = vertices[seed]
                # 计算所有点到 seed 的距离 (欧氏距离近似)
                dists = np.linalg.norm(vertices - seed_pos, axis=1)
                
                # Gaussian patch
                patch = np.exp(-dists**2 / (2 * task.sigma_s**2))
                patch[patch < 0.01] = 0
                
                spatial_map += patch
            
            task_data.append({
                'envelope': envelope,
                'spatial_map': spatial_map,
                'amp': task.amplitude_pde
            })

        # 返回 Callable 对象
        u_pde = PDEStimulus(n_verts, self.dt, self.duration, task_data)
            
        config = {
            'type': 'task_based_pde',
            'tasks': [
                {
                    'index': t.index,
                    'range': (t.start_time, t.end_time),
                    'seeds': t.seed_vertices,
                    'amplitude': t.amplitude_pde,
                    'sigma_s': t.sigma_s
                }
                for t in tasks
            ]
        }
        return u_pde, config

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
            # 精确离散 OU 过程更新
            # 连续 SDE: dX = -X/tau_noise dt + sigma * sqrt(2/tau_noise) dW
            theta = 1.0 / tau_noise
            decay = np.exp(-theta * self.dt)  # e^{-theta dt} = e^{-dt/tau}
            # 增量标准差遵循精确解的方差：Var = sigma^2 * (1 - e^{-2 theta dt})
            incr_std = sigma * np.sqrt(1.0 - decay**2)
            # 初始状态采用平稳分布 N(0, sigma^2)
            x = np.random.normal(0.0, sigma, size=self.n_nodes)
            
            for i in range(self.n_time_steps):
                x = decay * x + incr_std * np.random.normal(0.0, 1.0, size=self.n_nodes)
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
