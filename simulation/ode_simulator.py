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

from dynamics.ode_models import ODEModel, EIModel, BilinearControlModel
from dynamics.balloon_model import BalloonModel
from simulation.stimulation_generator import StimulationGenerator

class ODESimulator:
    """
    ODE仿真数据生成器
    """
    
    def __init__(self, 
                 n_nodes: int = 246, 
                 dt: float = 50.0,    # 50 ms
                 duration: float = 20000.0, # 20 seconds(ms)
                 model_type: str = 'EI',
                 model_params: Optional[Dict] = None):
        
        self.n_nodes = n_nodes
        self.dt = dt
        self.duration = duration
        self.time_points = np.arange(0, duration, dt) #时间点数组，单位: ms
        self.n_time_steps = len(self.time_points) #时间步数
        self.model_type = model_type
        
        # 初始化模型
        if model_type == 'EI':
            self.model = EIModel(n_nodes, model_params)
        elif model_type == 'bilinear':
            self.model = BilinearControlModel(n_nodes, model_params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
        # 初始化血氧动力学模型
        self.balloon_model = BalloonModel()
        
        # 初始化刺激生成器 (单位: ms)
        self.stim_generator = StimulationGenerator(n_nodes, dt, duration)
        
    def _generate_bilinear_matrices(self, n_channels: int) -> Tuple[Union[np.ndarray, List], np.ndarray]:
        """
        生成双线性控制模型的 B 和 C 矩阵
        
        当 n_channels == n_nodes 时（与 EI 模型刺激通道数一致）：
        - B: 使用稀疏矩阵列表 [scipy.sparse.csr_matrix] * K，极大加速计算
        - C: (N, N) 对角矩阵，表示刺激直接输入到对应脑区
        
        当 n_channels < n_nodes 时（少量控制通道）：
        - B: (K, N, N) 稀疏随机
        - C: (N, K) 稀疏，每个通道作用于少数脑区
        """
        import scipy.sparse as sp
        
        # 简化：当通道数等于节点数时，使用对角驱动矩阵
        if n_channels == self.n_nodes:
            # 关键优化：使用稀疏矩阵列表而非密集 3D 数组
            # B: 每个脑区的刺激可以调制局部连接（非常稀疏，约 1%）
            B_list = []
            sparsity = 0.01  # 1% 非零
            for k in range(n_channels):
                # 生成稀疏随机矩阵
                nnz = int(self.n_nodes * self.n_nodes * sparsity)
                row_idx = np.random.randint(0, self.n_nodes, nnz)
                col_idx = np.random.randint(0, self.n_nodes, nnz)
                values = np.random.uniform(-0.1, 0.1, nnz)
                
                # 使用 CSR 格式（行压缩，矩阵-向量乘法最快）
                B_k = sp.csr_matrix((values, (row_idx, col_idx)), 
                                    shape=(self.n_nodes, self.n_nodes),
                                    dtype=np.float64)
                B_list.append(B_k)
            
            # C: 对角矩阵，刺激直接输入到对应脑区
            # 权重随机化以模拟不同脑区的刺激敏感度
            C = np.diag(np.random.uniform(0.5, 1.5, self.n_nodes))
            
            return B_list, C
        else:
            # 原有的少通道逻辑（保持密集数组，因为 K 很小）
            # 生成 B
            B = np.zeros((n_channels, self.n_nodes, self.n_nodes))
            for k in range(n_channels):
                # 稀疏随机连接
                # 假设每个通道调制约 5% 的连接
                mask = np.random.rand(self.n_nodes, self.n_nodes) < 0.05
                values = np.random.uniform(-0.3, 0.3, (self.n_nodes, self.n_nodes))
                B[k] = values * mask
                
            # 生成 C
            C = np.zeros((self.n_nodes, n_channels))
            for k in range(n_channels):
                # 每个通道作用于 1-2 个脑区
                n_targets = np.random.randint(1, 3)
                targets = np.random.choice(self.n_nodes, size=n_targets, replace=False)
                values = np.random.uniform(0.5, 1.5, n_targets)
                C[targets, k] = values
            
            return B, C

    def run_simulation(self, 
                       connectivity: np.ndarray, 
                       stimulus: Optional[np.ndarray] = None,
                       stimulus_config: Optional[Dict] = None,
                       noise_level: float = 0.01,
                       noise_seed: Optional[int] = None,
                       initial_state: Optional[np.ndarray] = None,
                       sampling_interval: float = 50.0,
                       n_stim_channels: int = 5,
                       clip_state: bool = True,
                       state_clip_value: float = 1.0,
                       fail_on_nan: bool = False) -> Dict:
        """
        运行单次仿真
        
        参数:
            connectivity: 连接矩阵 (n_nodes, n_nodes)
            stimulus: 外部刺激 (n_time_steps, n_nodes) 或 (n_time_steps, n_channels)
            stimulus_config: 刺激配置字典
            noise_level: 噪声水平
            noise_seed: 噪声随机种子
            initial_state: 初始状态
            sampling_interval: 采样时间间隔 (ms)
            n_stim_channels: 刺激通道数 (仅用于 bilinear 模型且未提供 stimulus 时)
            clip_state: 是否裁剪状态变量 (防止数值发散)
            state_clip_value: 裁剪阈值 (默认1.0，因为 BEI 模型变量 S_E/S_I 是门控变量，范围 [0, 1])
            fail_on_nan: 遇到 NaN 时是否报错
            
        返回:
            results: 包含神经活动、BOLD信号等的字典
        """
        # 设置随机种子 (确保整个过程可复现)
        if noise_seed is not None:
            np.random.seed(noise_seed)

        # 设置连接矩阵
        if self.model_type == 'EI':
            self.model.params['C'] = connectivity
        elif self.model_type == 'bilinear':
            self.model.params['A'] = connectivity
            
            # 如果是 bilinear 模型，检查是否需要生成 B 和 C
            if self.model.params['B'] is None or self.model.params['C'] is None:
                B, C = self._generate_bilinear_matrices(n_stim_channels)
                self.model.params['B'] = B
                self.model.params['C'] = C
        
        # 自动生成刺激 (如果未提供)
        if stimulus is None:
            # 为确保可复现，若未提供 noise_seed 则使用固定种子
            stim_seed = noise_seed if noise_seed is not None else 42
            # 统一使用 n_nodes 作为刺激通道数，适用于 EI 和 bilinear 模型
            # 这样刺激生成逻辑完全通用
            n_channels_for_stim = n_stim_channels if self.model_type == 'bilinear' else self.n_nodes
            
            tasks = self.stim_generator.generate_task_schedule(n_channels=n_channels_for_stim, seed=stim_seed)
            stimulus, stimulus_config = self.stim_generator.generate_ode_stimulus(tasks, n_channels_for_stim, seed=stim_seed)

        # 生成噪声
        # ODE 噪声: eta(t) = sigma * epsilon(t)
        # 注意：EI模型状态维度是 2*n_nodes (E和I)，而Bilinear模型是 n_nodes
        # 噪声通常加在所有状态变量上，或者只加在E上？
        # 题目描述: eta(t) = sigma * epsilon(t), epsilon ~ N(0, I).
        # 对于EI模型，通常E和I都有噪声输入。
        
        noise_dim = self.n_nodes
        if self.model_type == 'EI':
            noise_dim = 2 * self.n_nodes
            
        noise = np.random.normal(0, 1, (self.n_time_steps, noise_dim)) * noise_level

        
        # 初始状态
        if initial_state is None:
            if self.model_type == 'EI':
                initial_state = np.random.rand(2 * self.n_nodes) * 0.1
            else:
                initial_state = np.random.rand(self.n_nodes) * 0.1
        
        # 运行积分
        # 使用简单的 Euler 或 RK4
        state = initial_state
        states = []
        
        # 降采样记录
        sampling_steps = int(sampling_interval / self.dt)
        if sampling_steps < 1:
            sampling_steps = 1
        
        # 进度输出（每 10% 输出一次）
        progress_interval = max(1, self.n_time_steps // 10)
        
        for i in range(self.n_time_steps):

            t = self.time_points[i]
            
            # 获取当前时刻刺激
            u_t = None
            if stimulus is not None:
                u_t = stimulus[i]
            
            # 计算导数
            # dy/dt = f(t, y, u) + noise
            # 注意：噪声是加性的，且通常在积分步中处理为 sqrt(dt) * noise (SDE)
            # 但题目要求 eta(t) = sigma * epsilon(t)，这看起来像是直接加在导数上的白噪声项
            # 如果是 SDE: dy = f dt + sigma dW. dW ~ N(0, dt). -> dy/dt = f + sigma * N(0, 1)/sqrt(dt)
            # 题目给出的形式是 eta(t) = sigma * epsilon(t), epsilon ~ N(0, I).
            # 这通常意味着在离散化时: x(t+dt) = x(t) + f(x, u)*dt + sigma * epsilon * dt
            # 或者 x(t+dt) = x(t) + f(x, u)*dt + sigma * sqrt(dt) * epsilon'
            # 按照题目 "eta(t) = sigma * epsilon(t)" 且 "每个时间步...独立采样"，
            # 且 sigma ~ 0.01-0.05.
            # 我们假设这是加在导数上的项，离散化时乘以 dt。
            
            # 兼容时间单位：模型参数假设单位为秒(s)，但dt是ms
            # 因此这里传入 t (ms) 实际上只是为了查找刺激，动力学方程应该基于 s
            # ODESimulator 的 dt (ms) 在积分时应该转为 s: dt_s = self.dt / 1000.0
            
            # 注意: dynamics 内部参数如 tau=0.1s，意味着 dS/dt 单位是 1/s
            # 所以更新时 delta_S = (dS/dt) * dt_s
            
            dt_s = self.dt / 1000.0
            # dynamics 的 t 参数在模型内部主要用于非自主系统（依赖时间的参数）
            # 目前模型是自主的（不显式依赖 t），但为了接口一致性，传入秒单位的 t
            dydt = self.model.dynamics(t / 1000.0, state, u_t)
            
            # Euler step
            # state = state + dydt * dt_s + noise[i] * np.sqrt(dt_s) # SDE standard if noise is variance 1
            # 按照题目描述，直接加噪声项
            # 假设 noise[i] 已经是 eta(t)
            # state = state + (dydt + noise[i]) * dt_s
            
            # BEI 模型对步长敏感，使用 Euler-Maruyama 可能会有数值问题
            # 这里简单使用 Euler，注意 dt 必须足够小 (e.g. 0.1-1.0 ms)
            state = state + dydt * dt_s + noise[i] * np.sqrt(dt_s)
            
            # 物理约束：神经门控变量 S_E, S_I 必须非负
            state = np.maximum(state, 0.0)

            # 数值健壮性检查：一旦出现 NaN/Inf，后续会迅速污染整段序列
            if not np.all(np.isfinite(state)):
                if fail_on_nan:
                    raise FloatingPointError(f"Non-finite state encountered at step {i} (t={t}).")
                state = np.nan_to_num(state, nan=0.0, posinf=state_clip_value, neginf=-state_clip_value)

            if clip_state:
                state = np.clip(state, -state_clip_value, state_clip_value)
            
            if i % sampling_steps == 0:
                states.append(state.copy())
            
            # 进度输出（每 10% 输出一次）
            if (i + 1) % progress_interval == 0:
                progress = (i + 1) / self.n_time_steps * 100
                print(f"ODE仿真进度: {progress:.1f}% ({i+1}/{self.n_time_steps})", end='\r')
                
        print()  # 换行
        states = np.array(states)
        
        # 计算 BOLD 信号
        # 注意：EI模型状态是 [E, I]，BOLD通常只基于 E
        # Bilinear模型状态是 x
        neural_activity_for_bold = states
        if self.model_type == 'EI':
            neural_activity_for_bold = states[:, :self.n_nodes]
            
        # BalloonModel.compute_bold 期望 t_span 是一个时间点数组，而不是单个 float
        # 我们需要传入降采样后的时间点序列
        time_points_downsampled = self.time_points[::sampling_steps]
        bold = self.balloon_model.compute_bold(neural_activity_for_bold, time_points_downsampled)
        
        return {
            'time_points': time_points_downsampled,
            'neural_activity': states,
            'bold_signal': bold,
            #'stimulus': stimulus, # 返回完整的刺激时间序列 (T, N)
            'stimulus_config': stimulus_config,
            'model_params': self.model.params, # 包含生成的 B 和 C
            'initial_state': initial_state, # 保存初始状态以支持完全复现
            'metadata': {
                'model_type': self.model_type,
                'dt': self.dt,
                'duration': self.duration,
                'sampling_interval': sampling_interval,
                'noise_level': noise_level,
                'noise_seed': noise_seed # 保存种子
            }
        }


