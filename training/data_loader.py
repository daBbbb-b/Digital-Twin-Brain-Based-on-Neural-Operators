import pickle  # 用于加载 .pkl 文件
import numpy as np  # 用于处理数值数据
import torch  # 用于张量操作和深度学习

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

def load_data_from_pkl(pkl_path, T=512):
    """
    从 .pkl 文件中加载数据，并将其处理为适合训练的格式。

    参数:
    - pkl_path (str or Path): .pkl 文件的路径。
    - T (int): 每个样本的时间步长（即序列长度）。数据将被切分为多个长度为 T 的样本。

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
        # 检查数据是否包含 "bold_signal" 和 "stimulus_config" 键
        elif "bold_signal" in data and "stimulus_config" in data:
            # 处理仿真数据的情况
            raw_x = data["bold_signal"]  # 输出信号数据，形状为 (Time, Channels)
            if np.isnan(raw_x).any():
                # 如果数据中存在 NaN 值，将其替换为 0
                raw_x = np.nan_to_num(raw_x, nan=0.0)
            
            # 获取时间步数和通道数
            n_time, n_channels = raw_x.shape
            # 初始化刺激矩阵，形状为 (Time, Channels)
            stimulus_matrix = np.zeros((n_time, n_channels))
            # 获取时间点数组
            time_points = data["time_points"]
            # 获取刺激配置
            config = data["stimulus_config"]
            
            # 遍历每个任务，生成刺激矩阵
            for task in config['tasks']:
                t_start, t_end = task['range']  # 刺激的时间范围
                channels = task['channels']  # 刺激的通道列表
                amplitudes = task['amplitudes']  # 刺激的幅度列表
                # 创建时间范围的掩码
                mask = (time_points >= t_start) & (time_points <= t_end)
                # 遍历通道和对应的幅度
                for ch, amp in zip(channels, amplitudes):
                    if ch < n_channels:  # 确保通道索引合法
                        stimulus_matrix[mask, ch] += amp  # 在掩码范围内添加刺激幅度
            
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