import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# 增加项目根目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from simulation.stimulation_generator import StimulationGenerator

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

    stim_config = data.get('stimulus_config')
    if not stim_config:
        print(f"No stimulus_config found in {file_path}")
        return

    metadata = data.get('metadata', {})
    dt = metadata.get('dt', 0.1)
    duration = metadata.get('duration', 600000)
    
    # 检查是 ODE 还是 PDE 还是两者都有
    # 结构可能是 {'ode': config_ode, 'pde': config_pde} 或直接是 config
    configs = {}
    if 'ode' in stim_config or 'pde' in stim_config:
        configs = stim_config
    else:
        # 兼容旧格式
        if stim_config.get('type') == 'task_based_ode':
            configs['ode'] = stim_config
        elif stim_config.get('type') == 'task_based_pde':
            configs['pde'] = stim_config

    if not configs:
        print(f"Unknown stimulus_config format in {file_path}")
        return

    n_plots = len(configs)
    fig, axes = plt.subplots(n_plots, 1, figsize=(15, 5 * n_plots), sharex=True)
    if n_plots == 1:
        axes = [axes]

    plot_idx = 0
    for stim_type, config in configs.items():
        ax = axes[plot_idx]
        
        if stim_type == 'ode':
            n_channels = config.get('n_channels', 246)
            stim_gen = StimulationGenerator(n_nodes=n_channels, dt=dt, duration=duration)
            time_points = stim_gen.time_points
            time_s = time_points / 1000.0
            
            tasks = config.get('tasks', [])
            u = np.zeros((len(time_points), n_channels))
            
            active_channels = set()
            for task in tasks:
                t0, t1 = task['range']
                envelope = stim_gen._smooth_boxcar(time_points, t0, t1)
                for ch_idx, amp in zip(task['channels'], task['amplitudes']):
                    u[:, ch_idx] += amp * envelope
                    active_channels.add(ch_idx)
            
            # 绘图
            for ch in sorted(list(active_channels)):
                ax.plot(time_s, u[:, ch], label=f'Node {ch}')
            
            # 标注任务区域
            colors = plt.cm.get_cmap('Set3', len(tasks))
            for i, task in enumerate(tasks):
                t0, t1 = task['range']
                ax.axvspan(t0/1000.0, t1/1000.0, color=colors(i), alpha=0.2)
                # 在上方标注任务索引
                ax.text((t0+t1)/2000.0, ax.get_ylim()[1]*0.9, f"T{task['index']}", 
                        ha='center', fontsize=8, fontweight='bold')

            ax.set_ylabel('ODE Amplitude')
            ax.set_title(f"ODE Stimulus Waveforms (Active Nodes)")
            if active_channels:
                ax.legend(loc='upper right', fontsize='small', ncol=min(5, len(active_channels)))

        elif stim_type == 'pde':
            # PDE 刺激通常是高维的 (T, N_vertices)
            # 我们可视化其时间包络或种子点的幅度
            tasks = config.get('tasks', [])
            # 假设 duration 和 dt 一致
            time_points = np.arange(0, duration, dt)
            time_s = time_points / 1000.0
            
            # 模拟生成器用于 smooth_boxcar
            stim_gen = StimulationGenerator(n_nodes=1, dt=dt, duration=duration)
            
            # 绘制每个任务的包络 * 幅度
            for i, task in enumerate(tasks):
                t0, t1 = task['range']
                amp = task['amplitude']
                envelope = stim_gen._smooth_boxcar(time_points, t0, t1)
                ax.plot(time_s, amp * envelope, label=f"Task {task['index']} (Amp={amp:.2f})")
                
                # 标注任务区域
                ax.axvspan(t0/1000.0, t1/1000.0, alpha=0.1)

            ax.set_ylabel('PDE Amplitude (Envelope)')
            ax.set_title(f"PDE Stimulus Envelopes")
            ax.legend(loc='upper right', fontsize='small')

        plot_idx += 1

    plt.xlabel('Time (s)')
    plt.suptitle(f"Stimulus Visualization: {file_path.name}")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    save_path = output_dir / f"{file_path.stem}_stimulus.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Successfully saved stimulus plot to {save_path}")

def main():
    # 配置
    data_dir = Path('dataset/simulation_data')
    output_dir = data_dir / 'plots'
    
    # 获取所有 .pkl 文件
    pkl_files = sorted(list(data_dir.glob('*.pkl')))
    
    if not pkl_files:
        print(f"No .pkl files found in {data_dir}")
        return

    print(f"Found {len(pkl_files)} samples. Starting visualization...")
    
    # 为了演示，只处理前 5 个样本，或者用户可以修改范围
    max_samples = 10 
    for i, pkl_file in enumerate(pkl_files[:max_samples]):
        print(f"[{i+1}/{min(max_samples, len(pkl_files))}] Processing {pkl_file.name}...")
        plot_sample_stimulus(pkl_file, output_dir)

if __name__ == "__main__":
    main()
