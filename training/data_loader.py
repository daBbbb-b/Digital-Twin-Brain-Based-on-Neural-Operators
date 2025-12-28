import pickle  # 用于加载 .pkl 文件
import numpy as np  # 用于处理数值数据
import torch  # 用于张量操作和深度学习
import sys
import os

# 增加项目根目录到路径，以便导入 simulation 模块
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from simulation.stimulation_generator import StimulationGenerator
except ImportError:
    print("Warning: Could not import StimulationGenerator. ODE stimulus reconstruction may fail.")


"""
本模块实现了从 .pkl 文件中加载数据并转换为适合深度学习模型训练的格式。
主要实现的功能包括：
1. 将原始数据（如时间序列信号和刺激配置）转换为 PyTorch 张量。
2. 如果数据是时间序列信号，切分为多个固定长度的样本。
3. 如果数据包含刺激配置，生成刺激矩阵并与时间序列信号对齐。

数据转换的具体过程：
- 输入数据格式：
  1. 如果数据包含键 "x" 和 "u"，直接加载为输入和输出张量。
  2. 如果数据包含键 "bold_signal" 和 "stimulus_config"，则需要进一步处理：
     - "bold_signal" 是时间序列信号，形状为 (Time, Channels)。
     - "stimulus_config" 包含刺激任务的时间范围、通道和幅度。
     - 根据刺激任务生成刺激矩阵，形状为 (Time, Channels)。
- 数据切分：
  - 将时间序列数据切分为多个样本，每个样本的长度为 T。
  - 切分后的数据形状为 (num_samples, T, num_channels)。
- 使用的公式：
  - 刺激矩阵生成公式：
    stimulus_matrix[mask, ch] += amp
    其中，mask 是时间范围的布尔掩码，ch 是通道索引，amp 是刺激幅度。
  - 数据切分公式：
    x_full[:num_samples*T].view(num_samples, T, n_channels)
    将时间序列数据切分为多个样本，并调整形状。

输出数据格式：
- x (torch.Tensor): 输入数据张量，形状为 (num_samples, T, num_channels)。
- u (torch.Tensor): 输出数据张量，形状为 (num_samples, T, num_channels)。
"""

def load_data_from_pkl(pkl_path, T=512, normalize=False, sim_type="auto"):
    """
    从 .pkl 文件中加载数据，并将其处理为适合训练的格式。

    参数:
    - pkl_path (str or Path): .pkl 文件的路径。
    - T (int): 每个样本的时间步长（即序列长度）。数据将被切分为多个长度为 T 的样本。
    - normalize (bool): 是否对数据进行归一化处理（Z-score）。默认为 False。
    - sim_type (str): 模拟类型，"ode" | "pde" | "auto"。推荐显式传入，避免依赖数据中的字段。

    返回:
    - x (torch.Tensor): 输入数据张量，形状为 (num_samples, T, num_channels)。
    - u (torch.Tensor): 输出数据张量，形状为 (num_samples, T, num_channels)。
      如果加载失败或数据格式不正确，返回 (None, None)。
    """
    try:
        # 打开 .pkl 文件并加载数据
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        
        # 检查数据是否包含 "x" 和 "u" 键
        if "x" in data and "u" in data:
            # 如果数据包含 "x" 和 "u"，直接将其转换为 PyTorch 张量
            x = torch.as_tensor(data["x"], dtype=torch.float32)  # 输入数据
            u = torch.as_tensor(data["u"], dtype=torch.float32)  # 输出数据
            
            if normalize:
                # 对 x 进行归一化
                # 计算除了最后一个维度（通道维度）之外的所有维度的均值和标准差
                dim = tuple(range(x.ndim - 1))
                mean = torch.mean(x, dim=dim, keepdim=True)
                std = torch.std(x, dim=dim, keepdim=True)
                std[std == 0] = 1.0
                x = (x - mean) / std

        # 检查数据是否包含 "bold_signal" 或 "neural_activity" 和 "stimulus_config" 键
        elif ("bold_signal" in data or "neural_activity" in data) and "stimulus_config" in data:
            # 处理仿真数据的情况
            if "bold_signal" in data:
                raw_x = data["bold_signal"]
            else:
                raw_x = data["neural_activity"]
                
            if np.isnan(raw_x).any():
                # 如果数据中存在 NaN 值，将其替换为 0
                raw_x = np.nan_to_num(raw_x, nan=0.0)
            
            if normalize:
                # 对数据进行归一化处理 (Z-score normalization)
                mean = np.mean(raw_x, axis=0)
                std = np.std(raw_x, axis=0)
                std[std == 0] = 1.0  # 防止除以零
                raw_x = (raw_x - mean) / std
            
            # 获取时间步数和通道数
            n_time, n_channels = raw_x.shape
            
            # 尝试直接获取刺激矩阵
            if "stimulus" in data and data["stimulus"] is not None:
                stimulus_matrix = data["stimulus"]
                # 确保形状匹配
                if stimulus_matrix.shape[0] != n_time:
                    # 如果长度不匹配，尝试截断或填充
                    if stimulus_matrix.shape[0] > n_time:
                        stimulus_matrix = stimulus_matrix[:n_time]
                    else:
                        # 填充
                        padding = np.zeros((n_time - stimulus_matrix.shape[0], stimulus_matrix.shape[1]))
                        stimulus_matrix = np.vstack([stimulus_matrix, padding])
            else:
                # 如果没有保存刺激矩阵，则根据配置重建
                stimulus_matrix = np.zeros((n_time, n_channels), dtype=np.float32)
                config = data["stimulus_config"]
                metadata = data.get("metadata", {})
                dt = metadata.get("dt", 0.1)
                duration = metadata.get("duration", 600000)
                
                # 获取时间点数组
                if "time_points" in data:
                    time_points = data["time_points"]
                else:
                    time_points = np.arange(n_time) * dt

                # 判断是 ODE 还是 PDE 任务
                if sim_type == "ode":
                    is_ode = True
                elif sim_type == "pde":
                    is_ode = False
                else:
                    # auto: 兼容旧数据，按配置推断
                    is_ode = config.get('type') == 'mixed_task_ode'
                
                # ODE 刺激重建
                if is_ode:
                    try:
                        stim_gen = StimulationGenerator(n_nodes=n_channels, dt=dt, duration=duration)
                        # 使用生成器的时间点，可能需要对齐
                        gen_time_points = stim_gen.time_points
                        
                        # 重建背景噪声
                        noise_cfg = config.get('noise', None)
                        if noise_cfg:
                            noise, _ = stim_gen.generate_noise(
                                sigma=noise_cfg.get('sigma', 0.05),
                                color=noise_cfg.get('color', 'ou'),
                                tau_noise=noise_cfg.get('tau_noise', 100.0),
                                seed=noise_cfg.get('seed', None)
                            )
                            # 确保噪声长度与 raw_x 一致
                            if len(noise) > n_time:
                                noise = noise[:n_time]
                            elif len(noise) < n_time:
                                noise = np.pad(noise, ((0, n_time - len(noise)), (0, 0)))
                            stimulus_matrix += noise

                        # 重建任务刺激
                        tasks = config.get('tasks', [])
                        for task in tasks:
                            task_seed = task.get('task_seed', 0)
                            rng = np.random.RandomState(task_seed)
                            t0, t1 = task['range']
                            wf_type = task.get('type', 'boxcar')

                            if wf_type == 'boxcar':
                                actual_end = task.get('specific_params', {}).get('actual_end_time', t1)
                                envelope = stim_gen._smooth_boxcar(gen_time_points, t0, actual_end)
                            elif wf_type == 'impulse':
                                interval_mean = task.get('specific_params', {}).get('interval_mean', 2000.0)
                                envelope = stim_gen._impulse_train(gen_time_points, t0, t1, interval_mean=interval_mean, rng=rng)
                            elif wf_type == 'continuous':
                                envelope = stim_gen._continuous_signal(gen_time_points, t0, t1, rng=rng)
                            else:
                                envelope = stim_gen._smooth_boxcar(gen_time_points, t0, t1)
                            
                            # 对齐 envelope 长度
                            if len(envelope) > n_time:
                                envelope = envelope[:n_time]
                            elif len(envelope) < n_time:
                                envelope = np.pad(envelope, (0, n_time - len(envelope)))

                            for ch_idx, amp in zip(task.get('channels', []), task.get('amplitudes', [])):
                                if ch_idx < n_channels:
                                    stimulus_matrix[:, ch_idx] += amp * envelope
                    except Exception as e:
                        print(f"Error reconstructing ODE stimulus: {e}")
                        # Fallback to simple reconstruction
                        for task in config.get('tasks', []):
                            t_start, t_end = task['range']
                            mask = (time_points >= t_start) & (time_points <= t_end)
                            for ch, amp in zip(task.get('channels', []), task.get('amplitudes', [])):
                                if ch < n_channels:
                                    stimulus_matrix[mask, ch] += amp

                # PDE 刺激重建 (简化版，因为可能缺少几何信息)
                else:
                    # 尝试重建 PDE 噪声
                    if 'noise_level' in metadata and 'noise_seed' in metadata:
                        noise_level = metadata['noise_level']
                        noise_seed = metadata['noise_seed']
                        rng = np.random.default_rng(noise_seed)
                        # 生成噪声
                        noise_full = np.zeros((n_time, n_channels))
                        # 注意：这里假设噪声是独立生成的，或者我们需要知道采样间隔
                        # 简单起见，直接生成
                        for i in range(n_time):
                            noise_full[i] = rng.normal(0.0, noise_level, size=n_channels)
                        stimulus_matrix += noise_full

                    # 重建 PDE 任务 (仅时间包络，空间分布如果缺少 vertices 很难精确重建)
                    # 如果数据中有 vertices，可以尝试重建空间分布
                    vertices = data.get("vertices")
                    if vertices is None and "surface" in data:
                        vertices = data["surface"].get("vertices")
                    
                    tasks = config.get('tasks', [])
                    for task in tasks:
                        t_start, t_end = task['range']
                        amplitude = task.get('amplitude', 0.0)
                        seeds = task.get('seeds', [])
                        sigma_s = task.get('sigma_s', 10.0)
                        
                        # 计算时间包络
                        rise_time = 500.0
                        # 向量化计算包络
                        envelope = np.zeros(n_time)
                        # 简单梯形包络
                        mask_rise = (time_points >= t_start) & (time_points < t_start + rise_time)
                        mask_plateau = (time_points >= t_start + rise_time) & (time_points <= t_end - rise_time)
                        mask_fall = (time_points > t_end - rise_time) & (time_points <= t_end)
                        
                        envelope[mask_rise] = (time_points[mask_rise] - t_start) / rise_time
                        envelope[mask_plateau] = 1.0
                        envelope[mask_fall] = (t_end - time_points[mask_fall]) / rise_time
                        
                        if vertices is not None:
                            # 如果有顶点信息，计算空间分布
                            spatial_pattern = np.zeros(n_channels)
                            for seed_idx in seeds:
                                if seed_idx < len(vertices):
                                    seed_pos = vertices[seed_idx]
                                    dists = np.linalg.norm(vertices - seed_pos, axis=1)
                                    spatial_pattern += np.exp(-dists**2 / (2 * sigma_s**2))
                            
                            if np.max(np.abs(spatial_pattern)) > 0:
                                spatial_pattern = spatial_pattern / np.max(np.abs(spatial_pattern))
                                
                            # 叠加到刺激矩阵: outer product
                            stimulus_matrix += np.outer(envelope, spatial_pattern) * amplitude
                        else:
                            # 如果没有顶点信息，仅在 seeds 处添加刺激 (忽略扩散)
                            # 这是一个近似
                            for seed_idx in seeds:
                                if seed_idx < n_channels:
                                    stimulus_matrix[:, seed_idx] += envelope * amplitude

            # 将刺激矩阵和输出信号转换为 PyTorch 张量
            u_full = torch.tensor(stimulus_matrix, dtype=torch.float32)  # 输入数据
            x_full = torch.tensor(raw_x, dtype=torch.float32)  # 输出数据
            
            # 将数据切分为多个样本，每个样本长度为 T
            num_samples = n_time // T  # 样本数量
            if num_samples > 0:
                # 切分数据并调整形状为 (num_samples, T, n_channels)
                x = x_full[:num_samples*T].view(num_samples, T, n_channels)
                u = u_full[:num_samples*T].view(num_samples, T, n_channels)
            else:
                # 如果数据不足以切分出一个样本，返回 None
                return None, None
        else:
            # 如果数据格式未知，打印警告并返回 None
            print(f"跳过 {pkl_path}: 未知的数据格式")
            return None, None

        # 返回处理后的输入和输出数据
        return x, u
    except Exception as e:
        # 如果加载过程中发生错误，打印错误信息并返回 None
        print(f"加载 {pkl_path} 失败: {e}")
        return None, None