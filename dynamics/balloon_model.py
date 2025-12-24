"""
气球-卷积模型模块

功能说明：
    实现血氧动力学模型（气球模型），用于将神经活动转换为fMRI BOLD信号。

主要类：
    BalloonModel: 气球-卷积模型

输入：
    - 神经活动时间序列
    - 血氧动力学参数

输出：
    - BOLD信号时间序列
    - 中间血氧动力学状态变量

参考：
    Pang et al. (2023) "Geometric constraints on human brain function"
    公式(17-21)：气球-卷积方程
"""

import numpy as np
from typing import Dict, Optional, Tuple
from scipy.integrate import odeint


class BalloonModel:
    """
    气球-卷积模型（Balloon-Windkessel Model）
    
    功能：
        - 将神经活动转换为BOLD信号
        - 模拟血流量、血容量、脱氧血红蛋白的动态过程
        - 考虑血氧动力学响应函数（HRF）
        
    模型方程（公式17-21）：
        ds/dt = z - κs - γ(f - 1)
        df/dt = s
        dv/dt = (f - v^(1/α)) / τ
        dq/dt = (f * E(f, E0) / E0 - q * v^(1/α-1)) / τ
        
        BOLD信号：
        y = V0 * (k1*(1-q) + k2*(1-q/v) + k3*(1-v))
        
        其中：
        - s: 血流量诱导信号
        - f: 血流量（flow）
        - v: 血容量（volume）
        - q: 脱氧血红蛋白含量
    """
    
    def __init__(self, params: Optional[Dict] = None):
        # 默认参数 (Friston et al., 2003)
        # 注意：Balloon 模型内部参数单位均为 秒 (s)
        self.default_params = {
            'kappa': 0.65,  # 信号衰减率 (s^-1)
            'gamma': 0.41,  # 流量依赖性消除率 (s^-1)
            'tau': 0.98,    # 血流动力学传递时间 (s)
            'alpha': 0.32,  # Grubb指数 (无量纲)
            'E0': 0.4,      # 静息氧摄取分数 (无量纲)
            'V0': 0.04,     # 静息血容量分数 (无量纲)
            'k1': 7 * 0.4,  # BOLD常数 k1 = 7 * E0
            'k2': 2.0,      # BOLD常数
            'k3': 2 * 0.4 - 0.2, # BOLD常数 k3 = 2 * E0 - 0.2
            'dt': 0.001     # 积分时间步长 (s)
        }
        if params:
            self.default_params.update(params)
        self.params = self.default_params

    def compute_bold(self, neural_activity: np.ndarray, t_span: np.ndarray) -> np.ndarray:
        """
        计算BOLD信号
        
        参数:
            neural_activity: 神经活动时间序列 (n_time_steps, n_nodes)
            t_span: 时间点数组 (n_time_steps,) [单位: ms]
            
        返回:
            bold_signal: BOLD信号 (n_time_steps, n_nodes)
        """
        n_time_steps, n_nodes = neural_activity.shape
        
        # 初始状态 (s=0, f=1, v=1, q=1)
        # 状态向量: [s, f, v, q] * n_nodes
        # 为了方便odeint，我们将状态展平: [s_1...s_n, f_1...f_n, v_1...v_n, q_1...q_n]
        initial_state = np.concatenate([
            np.zeros(n_nodes), # s
            np.ones(n_nodes),  # f
            np.ones(n_nodes),  # v
            np.ones(n_nodes)   # q
        ])
        
        # 定义ODE系统
        def balloon_dynamics(state, t):
            # 找到当前时间对应的神经活动 (简单的插值或最近邻)
            # 这里的 t 输入为 秒 (s)，需要转换为 ms 才能在 t_span 中查找
            t_ms = t * 1000.0
            idx = int(np.clip(np.searchsorted(t_span, t_ms), 0, n_time_steps - 1))
            z = neural_activity[idx]
            
            s = state[0*n_nodes : 1*n_nodes]
            f = state[1*n_nodes : 2*n_nodes]
            v = state[2*n_nodes : 3*n_nodes]
            q = state[3*n_nodes : 4*n_nodes]
            
            kappa = self.params['kappa']
            gamma = self.params['gamma']
            tau = self.params['tau']
            alpha = self.params['alpha']
            E0 = self.params['E0']
            
            # 限制f和v为正值以避免数值错误
            f = np.maximum(f, 1e-6)
            v = np.maximum(v, 1e-6)
            
            # 氧摄取分数 E(f) = 1 - (1-E0)^(1/f)
            E_f = 1 - (1 - E0)**(1.0 / f)
            
            ds = z - kappa * s - gamma * (f - 1)
            df = s
            dv = (f - v**(1/alpha)) / tau
            dq = (f * E_f / E0 - q * v**(1/alpha - 1)) / tau
            
            return np.concatenate([ds, df, dv, dq])
        
        # 求解ODE
        # 传入的 t_span 单位为 ms，但 Balloon 模型参数 (tau, kappa, gamma) 单位为 s
        # 因此我们需要将 dt 转换为秒 (s)
        
        dt_ms = t_span[1] - t_span[0]
        dt_s = dt_ms / 1000.0
        
        states = np.zeros((n_time_steps, 4 * n_nodes))
        states[0] = initial_state
        
        current_state = initial_state
        
        # 欧拉积分循环
        # 为了稳定性，内部积分步长设为 10ms (0.01s) 或更小
        # 但是，如果 dt_s 很大（例如 > 0.5s），我们可以适当增大 internal_dt_s 以提高效率
        # 因为 Balloon 模型的时间常数 tau ~ 0.98s，步长 0.05s 仍然足够稳定
        if dt_s <= 0.01:
            internal_dt_s = dt_s
        elif dt_s <= 0.1:
            internal_dt_s = 0.01
        else:
            # 对于更大的 dt_s，使用自适应步长，但不超过 dt_s/10 或 0.05s
            internal_dt_s = min(dt_s / 10.0, 0.05)
            
        steps_per_sample = int(np.ceil(dt_s / internal_dt_s))
        real_dt_s = dt_s / steps_per_sample # 实际积分步长 (s)
        
        for i in range(1, n_time_steps):
            z = neural_activity[i-1]
            
            for _ in range(steps_per_sample):
                s = current_state[0*n_nodes : 1*n_nodes]
                f = current_state[1*n_nodes : 2*n_nodes]
                v = current_state[2*n_nodes : 3*n_nodes]
                q = current_state[3*n_nodes : 4*n_nodes]
                
                kappa = self.params['kappa']
                gamma = self.params['gamma']
                tau = self.params['tau']
                alpha = self.params['alpha']
                E0 = self.params['E0']
                
                # 限制范围防止溢出
                f = np.clip(f, 1e-4, 100.0)
                v = np.clip(v, 1e-4, 100.0)
                q = np.clip(q, 1e-4, 100.0)
                s = np.clip(s, -100.0, 100.0)
                
                E_f = 1 - (1 - E0)**(1.0 / f)
                
                ds = z - kappa * s - gamma * (f - 1)
                df = s
                dv = (f - v**(1/alpha)) / tau
                dq = (f * E_f / E0 - q * v**(1/alpha - 1)) / tau
                
                # 更新状态 (注意这里必须用秒单位的步长)
                s_new = s + ds * real_dt_s
                f_new = f + df * real_dt_s
                v_new = v + dv * real_dt_s
                q_new = q + dq * real_dt_s
                
                current_state = np.concatenate([s_new, f_new, v_new, q_new])
            
            states[i] = current_state
            
        # 计算BOLD信号
        V0 = self.params['V0']
        k1 = self.params['k1']
        k2 = self.params['k2']
        k3 = self.params['k3']
        
        v_all = states[:, 2*n_nodes : 3*n_nodes]
        q_all = states[:, 3*n_nodes : 4*n_nodes]
        
        # 避免除零
        v_all = np.maximum(v_all, 1e-6)
        
        y = V0 * (k1 * (1 - q_all) + k2 * (1 - q_all / v_all) + k3 * (1 - v_all))
        
        return y
