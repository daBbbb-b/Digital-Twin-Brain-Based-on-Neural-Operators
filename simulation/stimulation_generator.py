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
    rng_seed: int = 0  # 用于可复现实验的随机种子（生成波形内部随机性）
    # ODE 参数
    active_channels: List[int] = field(default_factory=list)
    amplitudes_ode: List[float] = field(default_factory=list)
    # PDE 参数
    seed_vertices: List[int] = field(default_factory=list)
    amplitude_pde: float = 0.0
    sigma_s: float = 10.0
    # 波形类型 (新增)
    waveform_type: str = 'boxcar' # 'boxcar', 'impulse', 'continuous'

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

    def _impulse_train(self, t: np.ndarray, t0: float, t1: float, interval_mean: float = 2000.0, rng: Optional[np.random.RandomState] = None) -> np.ndarray:
        """
        生成随机脉冲序列
        """
        envelope = np.zeros_like(t)
        rng = rng or np.random.RandomState()
        
        current_t = t0
        while current_t < t1:
            # 找到对应的时间索引
            idx = int(current_t / self.dt)
            if 0 <= idx < self.n_time_steps:
                # 脉冲宽度 100ms
                width_idx = int(100.0 / self.dt)
                envelope[idx : min(idx + width_idx, self.n_time_steps)] = 1.0
            
            # 下一个脉冲
            current_t += rng.exponential(interval_mean)
            
        return envelope

    def _continuous_signal(self, t: np.ndarray, t0: float, t1: float, rng: Optional[np.random.RandomState] = None) -> np.ndarray:
        """
        生成连续变化信号 (滤波噪声或正弦组合)
        """
        rng = rng or np.random.RandomState()
        # 只在 t0 到 t1 之间非零
        mask = (t >= t0) & (t <= t1)
        
        # 使用正弦波组合模拟自然信号
        n_freqs = 5
        signal = np.zeros_like(t)
        
        # 基频 0.05Hz - 0.5Hz
        freqs = rng.uniform(0.05, 0.5, n_freqs)
        phases = rng.uniform(0, 2*np.pi, n_freqs)
        amps = rng.exponential(1.0, n_freqs)
        amps /= np.sum(amps) # 归一化
        
        for f, phi, a in zip(freqs, phases, amps):
            signal += a * np.sin(2 * np.pi * f * t / 1000.0 + phi)
            
        # 加上包络使得两端平滑
        window = self._smooth_boxcar(t, t0, t1, rise_time=1000.0)
        return signal * window

    def generate_task_schedule(self, 
                             n_channels: int, 
                             n_vertices_pde: Optional[int] = None,
                             seed: int = 42) -> List[TaskEvent]:
        """
        生成 Run 的 Task 调度序列
        
        1. 每个刺激模块时长 12.8s
        2. 每个模块根据剩余时间动态插入尽可能多的任务
        3. 任务随机为 2s-10s
        4. 混合三种波形
        
        Added seed for reproducibility.
        """
        # Set seed locally if provided, otherwise rely on global state (not recommended for strict reproducibility)
        # Better: create a local RandomState
        rng = np.random.RandomState(seed)
        
        tasks = []
        module_duration = 12800.0 # 12.8s
        
        n_modules = int(self.duration / module_duration)
        
        for mod_idx in range(n_modules):
            module_start = mod_idx * module_duration
            
            current_time = module_start + rng.uniform(0, 1000) # 初始随机延迟
            
            # 动态生成任务：根据剩余时间尽可能多地插入任务
            while True:
                # 剩余可用时间
                time_left = (module_start + module_duration) - current_time
                if time_left < 5000: break # 时间不够，跳过
                
                # 任务时长 2s - 12s (但不超过剩余时间)
                max_dur = min(12000.0, time_left - 1000) # 留1s间隔
                if max_dur < 2000: break
                
                dur = rng.uniform(2000.0, max_dur)
                t_start = current_time
                t_end = t_start + dur
                
                # 随机选择波形类型
                wf_type = rng.choice(['boxcar', 'impulse', 'continuous'])
                
                # ODE Params (随机刺激多个脑区)
                active_chs: List[int] = []
                amps_ode: List[float] = []
                if n_channels > 0:
                    # 刺激更多脑区，模拟复杂任务
                    n_active = rng.randint(1, min(5, n_channels + 1))
                    active_chs = rng.choice(n_channels, size=n_active, replace=False).tolist()

                    for _ in range(n_active):
                        # 幅度随机，可正可负
                        amp = rng.uniform(0.5, 2.0) * rng.choice([1, -1])
                        amps_ode.append(amp)
                
                # PDE Params
                seeds = []
                amp_pde = 0.0
                sigma_s = 10.0
                if n_vertices_pde is not None:
                    n_seeds = rng.randint(1, 5)
                    seeds = rng.choice(n_vertices_pde, size=n_seeds, replace=False).tolist()
                    amp_pde = rng.uniform(0.5, 2.0) * rng.choice([1, -1])
                    sigma_s = rng.uniform(5.0, 15.0)
                
                task = TaskEvent(
                    index=len(tasks),
                    start_time=t_start,
                    end_time=t_end,
                    duration=dur,
                    rng_seed=rng.randint(1_000_000_000),
                    active_channels=active_chs,
                    amplitudes_ode=amps_ode,
                    seed_vertices=seeds,
                    amplitude_pde=amp_pde,
                    sigma_s=sigma_s,
                    waveform_type=wf_type
                )
                tasks.append(task)
                
                current_time = t_end + rng.uniform(500, 2000) # 任务间隔
                
        return tasks

    def generate_ode_stimulus(self, 
                            tasks: List[TaskEvent], 
                            n_channels: int,
                            seed: int = 42) -> Tuple[np.ndarray, Dict]:
        """
        生成 ODE 刺激 u(t) (K维)
        """
        # Local RNG for reproducibility (global-level)
        rng_global = np.random.RandomState(seed)
        u = np.zeros((self.n_time_steps, n_channels), dtype=np.float32)
        
        task_configs = []
        
        for task in tasks:
            rng_task = np.random.RandomState(task.rng_seed)
            
            if task.waveform_type == 'boxcar':
                # Boxcar 随机截断
                min_dur = max(1000.0, task.duration * 0.5)
                actual_dur = rng_task.uniform(min_dur, task.duration)
                actual_end_time = task.start_time + actual_dur
                envelope = self._smooth_boxcar(self.time_points, task.start_time, actual_end_time)
                
                # Record specific params for reproducibility
                task_specific_params = {'actual_end_time': actual_end_time}
                
            elif task.waveform_type == 'impulse':
                envelope = self._impulse_train(self.time_points, task.start_time, task.end_time, rng=rng_task)
                task_specific_params = {'interval_mean': 2000.0}
                
            elif task.waveform_type == 'continuous':
                envelope = self._continuous_signal(self.time_points, task.start_time, task.end_time, rng=rng_task)
                task_specific_params = {'n_freqs': 5}
                
            else:
                envelope = self._smooth_boxcar(self.time_points, task.start_time, task.end_time)
                task_specific_params = {}
            
            for ch_idx, amp in zip(task.active_channels, task.amplitudes_ode):
                u[:, ch_idx] += amp * envelope
            
            task_configs.append({
                'index': task.index,
                'range': (task.start_time, task.end_time),
                'type': task.waveform_type,
                'channels': task.active_channels,
                'amplitudes': task.amplitudes_ode,
                'task_seed': task.rng_seed, # Crucial for reproduction
                'specific_params': task_specific_params
            })
                
        # 添加背景噪声（可复现）
        noise_level = 0.05
        background_noise_seed = rng_global.randint(1_000_000_000)
        noise_params = {'sigma': noise_level, 'color': 'ou', 'tau_noise': 100.0, 'seed': background_noise_seed}
        background_noise, noise_cfg = self.generate_noise(**noise_params)
        u += background_noise
        
        config = {
            'type': 'mixed_task_ode',
            'n_channels': n_channels,
            'tasks': task_configs,
            'noise': noise_params,
            'global_seed': seed
        }
        return u, config

    def generate_pde_stimulus(self, 
                            tasks: List[TaskEvent], 
                            vertices: np.ndarray,
                            faces: Optional[np.ndarray] = None) -> Tuple[Union[np.ndarray, Callable], Dict]:
        """
        生成 PDE 刺激 u_pde(s, t)
        """
        n_verts = vertices.shape[0]
        task_data = []
        task_configs = []
        
        for task in tasks:
            if not task.seed_vertices:
                continue
                
            # 波形生成
            if task.waveform_type == 'boxcar':
                envelope = self._smooth_boxcar(self.time_points, task.start_time, task.end_time)
            elif task.waveform_type == 'impulse':
                envelope = self._impulse_train(self.time_points, task.start_time, task.end_time, rng=np.random.RandomState(task.rng_seed))
            elif task.waveform_type == 'continuous':
                envelope = self._continuous_signal(self.time_points, task.start_time, task.end_time, rng=np.random.RandomState(task.rng_seed))
            else:
                envelope = self._smooth_boxcar(self.time_points, task.start_time, task.end_time)
            
            # 空间分布
            spatial_map = np.zeros(n_verts)
            for seed in task.seed_vertices:
                seed_pos = vertices[seed]
                dists = np.linalg.norm(vertices - seed_pos, axis=1)
                patch = np.exp(-dists**2 / (2 * task.sigma_s**2))
                patch[patch < 0.01] = 0
                spatial_map += patch
            
            task_data.append({
                'envelope': envelope,
                'spatial_map': spatial_map,
                'amp': task.amplitude_pde
            })
            
            task_configs.append({
                'index': task.index,
                'range': (task.start_time, task.end_time),
                'type': task.waveform_type,
                'seeds': task.seed_vertices,
                'amplitude': task.amplitude_pde,
                'rng_seed': task.rng_seed,
                'sigma_s': task.sigma_s
            })

        # 对于 PDE，我们无法像 ODE 那样简单叠加全脑噪声并存储（太大了）
        # 所以这里的 PDEStimulus 只包含确定的任务成分
        # 噪声会在 PDESimulator 内部积分时添加
        
        u_pde = PDEStimulus(n_verts, self.dt, self.duration, task_data)
            
        config = {
            'type': 'mixed_task_pde',
            'tasks': task_configs
        }
        return u_pde, config

    def generate_boxcar(self, 
                        onset: float, 
                        duration: float, 
                        amplitude: float = 1.0, 
                        nodes: Optional[List[int]] = None) -> Tuple[np.ndarray, Dict]:

        """
        生成Boxcar（矩形波）刺激
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
                       seed: Optional[int] = None,
                       return_rng: bool = False) -> Tuple[np.ndarray, Dict]:
        """
        生成噪声刺激
        """
        rng = np.random.RandomState(seed) if seed is not None else np.random
            
        if color == 'white':
            noise = rng.normal(0, sigma, (self.n_time_steps, self.n_nodes))
        
        elif color == 'ou':
            # Ornstein-Uhlenbeck process
            noise = np.zeros((self.n_time_steps, self.n_nodes))
            theta = 1.0 / tau_noise
            decay = np.exp(-theta * self.dt)
            incr_std = sigma * np.sqrt(1.0 - decay**2)
            x = rng.normal(0.0, sigma, size=self.n_nodes)
            
            for i in range(self.n_time_steps):
                x = decay * x + incr_std * rng.normal(0.0, 1.0, size=self.n_nodes)
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
        
        if return_rng:
            return noise, config, rng
        return noise, config            

    def combine_stimuli(self, stimuli_list: List[np.ndarray]) -> np.ndarray:
        """
        组合多个刺激（叠加）
        """
        if not stimuli_list:
            return np.zeros((self.n_time_steps, self.n_nodes))
        return sum(stimuli_list)
