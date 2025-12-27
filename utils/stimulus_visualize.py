import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os
import re

# 增加项目根目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from simulation.stimulation_generator import StimulationGenerator

def extract_number_from_filename(file_path):
    """
    从文件名中提取数字，用于排序
    例如: 'ode_0.pkl' -> 0, 'ode_10.pkl' -> 10, 'ode_100.pkl' -> 100
    
    参数:
        file_path: 文件路径（Path对象或字符串）
    
    返回:
        int: 提取的数字，如果没有找到则返回 0
    """
    file_path = Path(file_path)
    # 使用正则表达式查找文件名中的所有数字
    numbers = re.findall(r'\d+', file_path.stem)
    if numbers:
        # 返回最后一个数字（通常是样本编号）
        return int(numbers[-1])
    return 0

def plot_sample_stimulus(file_path, output_dir):
    """
    读取单个样本文件并可视化其刺激函数
    """
    file_path = Path(file_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return

    # 优先使用保存的真实刺激时间序列；若缺失则按 stimulus_config 完全重建
    stimulus = data.get('stimulus')
    stim_config = data.get('stimulus_config')
    
    metadata = data.get('metadata', {})
    dt = metadata.get('dt', 0.1)
    duration = metadata.get('duration', 600000)
    
    # 构建时间轴
    # 注意：stimulus shape 是 (T, N)，时间步数 T 应该与 dt, duration 对应
    # 但实际 T = len(stimulus)
    if stimulus is not None:
        n_steps = stimulus.shape[0]
        time_points = np.arange(n_steps) * dt
    else:
        # 如果 stimulus 缺失，尝试基于 config 重建
        if stim_config and stim_config.get('type') == 'mixed_task_ode':
            n_channels = stim_config.get('n_channels', 1)
            # 重建刺激矩阵
            stim_gen = StimulationGenerator(n_nodes=n_channels, dt=dt, duration=duration)
            # 模拟 generate_ode_stimulus 的复现流程
            # 1) 全局种子（用于背景噪声）：暂不需要修改时间轴
            # 2) time_points 与 duration 一致
            time_points = stim_gen.time_points
            time_s = time_points / 1000.0
            u = np.zeros((len(time_points), n_channels), dtype=np.float32)

            # 背景噪声
            noise_cfg = stim_config.get('noise', None)
            if noise_cfg:
                # generate_noise 接口: sigma, color, tau_noise, seed
                noise, _ = stim_gen.generate_noise(
                    sigma=noise_cfg.get('sigma', 0.05),
                    color=noise_cfg.get('color', 'ou'),
                    tau_noise=noise_cfg.get('tau_noise', 100.0),
                    seed=noise_cfg.get('seed', None)
                )
                u += noise

            tasks = stim_config.get('tasks', [])
            for task in tasks:
                task_seed = task.get('task_seed', 0)
                rng = np.random.RandomState(task_seed)
                t0, t1 = task['range']
                wf_type = task.get('type', 'boxcar')

                if wf_type == 'boxcar':
                    actual_end = task.get('specific_params', {}).get('actual_end_time', t1)
                    envelope = stim_gen._smooth_boxcar(time_points, t0, actual_end)
                elif wf_type == 'impulse':
                    interval_mean = task.get('specific_params', {}).get('interval_mean', 2000.0)
                    envelope = stim_gen._impulse_train(time_points, t0, t1, interval_mean=interval_mean, rng=rng)
                elif wf_type == 'continuous':
                    envelope = stim_gen._continuous_signal(time_points, t0, t1, rng=rng)
                else:
                    envelope = stim_gen._smooth_boxcar(time_points, t0, t1)

                for ch_idx, amp in zip(task.get('channels', []), task.get('amplitudes', [])):
                    u[:, ch_idx] += amp * envelope

            stimulus = u
        else:
            time_points = np.arange(0, duration, dt)

    time_s = time_points / 1000.0

    # 绘图
    fig, ax = plt.subplots(figsize=(15, 6))
    
    if stimulus is not None:
        # 可视化真实刺激矩阵
        # 由于脑区太多 (246)，只画出有显著活动的脑区，或者画热力图
        
        # 1. 找出活跃脑区 (Abs Max > 0.01)
        # 注意：刺激可能包含全脑噪声，所以简单的 > 0 可能选中所有。
        # 我们使用标准差或范围来判断主要受刺激脑区
        
        # 简单策略：根据 config 里的 active_channels 标注
        active_channels = set()
        if stim_config and 'tasks' in stim_config:
            for task in stim_config['tasks']:
                if 'channels' in task:
                    active_channels.update(task['channels'])
        
        # 如果没有 config 或者找不到，就画 Top 5 active channels
        if not active_channels:
            std_activity = np.std(stimulus, axis=0)
            active_channels = set(np.argsort(std_activity)[-5:])

        # 绘图
        for ch in sorted(list(active_channels)):
            ax.plot(time_s, stimulus[:, ch], label=f'Node {ch}', alpha=0.8)
            
        # 标注任务区域 (如果 Config 存在)
        if stim_config and 'tasks' in stim_config:
            colors = plt.cm.get_cmap('Set3', len(stim_config['tasks']))
            for i, task in enumerate(stim_config['tasks']):
                t0, t1 = task['range']
                ax.axvspan(t0/1000.0, t1/1000.0, color=colors(i), alpha=0.2)
                
                # 标注类型
                wf_type = task.get('type', 'unknown')
                ax.text((t0+t1)/2000.0, ax.get_ylim()[1]*0.95, f"T{task['index']}\n({wf_type})", 
                        ha='center', fontsize=8, fontweight='bold')

        ax.set_ylabel('Stimulus Amplitude (u)')
        ax.set_title(f"Stimulus Waveforms (Active Nodes): {file_path.name}")
        if active_channels:
            ax.legend(loc='upper right', fontsize='small', ncol=min(5, len(active_channels)))
            
    else:
        print(f"Warning: No stimulus and no usable stimulus_config in {file_path}.")
        ax.text(0.5, 0.5, "No stimulus found.", ha='center', va='center', transform=ax.transAxes)

    plt.xlabel('Time (s)')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    save_path = output_dir / f"{file_path.stem}_stimulus.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Successfully saved stimulus plot to {save_path}")

def main():
    # 配置
    data_dir = Path('dataset/simulation_data')
    output_dir = data_dir / 'plots'
    
    # 获取所有 .pkl 文件，并按文件名中的数字排序
    pkl_files = sorted(list(data_dir.glob('*.pkl')), key=extract_number_from_filename)
    
    if not pkl_files:
        print(f"No .pkl files found in {data_dir}")
        return

    print(f"Found {len(pkl_files)} samples. Starting visualization...")
    
    # 为了演示，只处理前 5 个样本
    max_samples = 5
    for i, pkl_file in enumerate(pkl_files[:max_samples]):
        print(f"[{i+1}/{min(max_samples, len(pkl_files))}] Processing {pkl_file.name}...")
        plot_sample_stimulus(pkl_file, output_dir)

if __name__ == "__main__":
    main()
