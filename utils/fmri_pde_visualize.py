import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os
import re

# 增加项目根目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def extract_number_from_filename(file_path):
    """
    从文件名中提取数字，用于排序
    例如: 'pde_surf_sample_0.pkl' -> 0
    """
    file_path = Path(file_path)
    numbers = re.findall(r'\d+', file_path.stem)
    if numbers:
        return int(numbers[-1])
    return 0

def plot_sample_pde_bold(file_path, output_dir, alpha=0.1, linewidth=0.5, max_display_nodes=200, debug=False):
    """
    读取单个PDE样本文件并可视化其 BOLD 信号
    
    参数:
        file_path: pickle 文件路径
        output_dir: 输出目录
        alpha: 单个节点线条的透明度
        linewidth: 线条宽度
        max_display_nodes: 背景节点的最大显示数量
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

    # 检查是否有 BOLD 信号
    bold_signal = data.get('bold_signal')
    if bold_signal is None:
        print(f"No bold_signal found in {file_path}")
        return
    
    # 获取metadata（用于后续噪声复原）
    metadata = data.get('metadata', {})
    
    # 获取时间点
    time_points = data.get('time_points')
    if time_points is None:
        sampling_interval = metadata.get('sampling_interval', 50.0)
        duration = metadata.get('duration', 20000.0)
        if bold_signal.ndim > 0:
            n_steps = bold_signal.shape[0]
            time_points = np.arange(n_steps) * sampling_interval
        else:
            time_points = np.arange(0, duration, sampling_interval)
    
    # 转换为秒
    time_s = time_points / 1000.0
    
    # 确保 BOLD 信号是 2D 数组
    bold_signal = np.asarray(bold_signal)
    if bold_signal.ndim == 1:
        bold_signal = bold_signal.reshape(-1, 1)
    
    n_time_steps, n_nodes = bold_signal.shape
    
    # 计算平均值
    bold_mean = np.mean(bold_signal, axis=1)
    
    # 复原噪声序列（如果metadata中有噪声信息）
    noise_signal = None
    if 'noise_level' in metadata and 'noise_seed' in metadata:
        noise_level = metadata['noise_level']
        noise_seed = metadata['noise_seed']
        dt = metadata.get('dt', 50.0)
        duration = metadata.get('duration', 20000.0)
        sampling_interval = metadata.get('sampling_interval', 50.0)
        
        if debug:
            print(f"[调试] 噪声复原参数: dt={dt}ms, duration={duration}ms, sampling_interval={sampling_interval}ms")
            print(f"[调试] noise_level={noise_level}, noise_seed={noise_seed}")
        
        # 计算总时间步数和采样步数
        n_total_steps = int(np.round(duration / dt))
        sampling_steps = int(np.round(sampling_interval / dt))
        if sampling_steps < 1:
            sampling_steps = 1
        
        if debug:
            print(f"[调试] 总时间步数: {n_total_steps}, 采样步数: {sampling_steps}")
            print(f"[调试] 预期采样后点数: {n_total_steps // sampling_steps}, 实际BOLD点数: {n_time_steps}")
        
        # 使用相同的随机种子重新生成噪声
        # 注意：PDE仿真中使用的是 np.random.default_rng，需要保持一致
        rng = np.random.default_rng(noise_seed)
        
        # 生成完整的噪声序列（每个时间步）
        # 注意：PDE中噪声是在每个时间步独立生成的，形状是 (n_nodes,)
        noise_full = np.zeros((n_total_steps, n_nodes))
        for i in range(n_total_steps):
            noise_full[i] = rng.normal(0.0, noise_level, size=n_nodes)
        
        # 按采样间隔降采样（与BOLD信号对齐）
        # 注意：采样应该从第一个时间点开始，与BOLD信号的采样保持一致
        noise_signal = noise_full[::sampling_steps]
        
        if debug:
            print(f"[调试] 降采样后噪声形状: {noise_signal.shape}")
        
        # 确保长度匹配
        if noise_signal.shape[0] != n_time_steps:
            if debug:
                print(f"[警告] 噪声长度 ({noise_signal.shape[0]}) 与BOLD长度 ({n_time_steps}) 不匹配，进行截断")
            # 如果长度不匹配，截断到较短的长度
            min_len = min(noise_signal.shape[0], n_time_steps)
            noise_signal = noise_signal[:min_len]
            if noise_signal.ndim == 1:
                noise_signal = noise_signal.reshape(-1, 1)
        
        if debug:
            print(f"[调试] 最终噪声形状: {noise_signal.shape}")
    
    # 获取任务信息用于背景标注和受刺激节点识别
    stim_config = data.get('stimulus_config')
    tasks = []
    stimulated_nodes = set()
    
    if stim_config and isinstance(stim_config, dict):
        if 'tasks' in stim_config:
            tasks = stim_config['tasks']
            for task in tasks:
                if 'seeds' in task:
                    stimulated_nodes.update(task['seeds'])
    
    # 确定要绘制的背景节点
    background_candidates = [i for i in range(n_nodes) if i not in stimulated_nodes]
    
    if len(background_candidates) > max_display_nodes:
        rng_bg = np.random.RandomState(42)  # 用于背景节点采样，避免与噪声rng冲突
        nodes_to_plot_bg = rng_bg.choice(background_candidates, size=max_display_nodes, replace=False).tolist()
    else:
        nodes_to_plot_bg = background_candidates
    
    # 创建图形
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    
    # 获取颜色映射
    try:
        colors = plt.colormaps.get_cmap('Set3')
    except AttributeError:
        colors = plt.get_cmap('Set3')
    
    # 绘制任务背景区域
    if tasks:
        for i, task in enumerate(tasks):
            if 'range' in task:
                t0, t1 = task['range']
                if np.isfinite(t0) and np.isfinite(t1):
                    ax.axvspan(t0/1000.0, t1/1000.0, color=colors(i), alpha=0.15, 
                              label=f"Task {task.get('index', i)}" if i == 0 else "")
    
    # 1. 绘制背景节点 (灰色，半透明)
    for node_idx in nodes_to_plot_bg:
        ax.plot(time_s, bold_signal[:, node_idx], 
               alpha=alpha, linewidth=linewidth, color='gray', zorder=1)
        
    # 2. 绘制受刺激节点 (橙色，较明显)
    stimulated_nodes_list = sorted(list(stimulated_nodes))
    for i, node_idx in enumerate(stimulated_nodes_list):
        if node_idx < n_nodes:
            label = 'Stimulated Nodes' if i == 0 else None
            ax.plot(time_s, bold_signal[:, node_idx], 
                   alpha=0.6, linewidth=linewidth*1.5, color='orange', label=label, zorder=5)
    
    # 3. 绘制噪声信号（如果可用）
    if noise_signal is not None:
        # 计算噪声的平均值
        noise_mean = np.mean(noise_signal, axis=1)
        # 确保时间轴长度匹配
        min_len = min(len(time_s), len(noise_mean))
        ax.plot(time_s[:min_len], noise_mean[:min_len], 
               linewidth=0.8, color='blue', alpha=0.5, 
               linestyle='--', label='Mean Noise', zorder=8)
    
    # 4. 绘制平均值线 (红色，细线)
    ax.plot(time_s, bold_mean, 
           linewidth=1.0, color='red', label='Mean BOLD', zorder=10)
    
    # 添加任务文本标注
    if tasks:
        y_lim = ax.get_ylim()
        y_max = y_lim[1] * 0.95
        
        for i, task in enumerate(tasks):
            if 'range' in task:
                t0, t1 = task['range']
                if np.isfinite(t0) and np.isfinite(t1):
                    text_x = (t0+t1)/2000.0
                    if np.isfinite(text_x) and np.isfinite(y_max):
                        ax.text(text_x, y_max, f"T{task.get('index', i)}", 
                               ha='center', fontsize=9, fontweight='bold', 
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7),
                               zorder=11)
    
    # 设置标签和标题
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('BOLD Signal', fontsize=12)
    
    bg_note = f"(showing {len(nodes_to_plot_bg)}/{len(background_candidates)} background nodes)" if len(background_candidates) > max_display_nodes else ""
    ax.set_title(f'PDE BOLD Visualization: {file_path.name}\n'
                f'({n_nodes} nodes, {n_time_steps} time points) {bg_note}', 
                fontsize=13, fontweight='bold')
    
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', fontsize=10)
    
    # 添加统计信息
    stats_text = (f'BOLD Mean: {np.mean(bold_mean):.4f}\n'
                 f'BOLD Std: {np.std(bold_mean):.4f}\n'
                 f'BOLD Min: {np.min(bold_mean):.4f}\n'
                 f'BOLD Max: {np.max(bold_mean):.4f}')
    
    if noise_signal is not None:
        noise_mean = np.mean(noise_signal, axis=1)
        stats_text += (f'\n\nNoise Mean: {np.mean(noise_mean):.4f}\n'
                      f'Noise Std: {np.std(noise_mean):.4f}\n'
                      f'Noise Level: {metadata.get("noise_level", "N/A")}')
    
    ax.text(0.02, 0.98, stats_text, 
           transform=ax.transAxes, fontsize=9,
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    save_path = output_dir / f"{file_path.stem}_bold.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Successfully saved PDE BOLD plot to {save_path}")

def main(max_samples=10, alpha=0.1, linewidth=0.5, data_dir=None, output_dir=None, max_display_nodes=200):
    """
    主函数：批量可视化 PDE BOLD 信号
    """
    if data_dir is None:
        data_dir = Path('dataset/simulation_data')
    else:
        data_dir = Path(data_dir)
    
    if output_dir is None:
        output_dir = data_dir / 'plots' / 'pde_bold'
    else:
        output_dir = Path(output_dir)
    
    # 获取所有 PDE 相关的 .pkl 文件 (通常是 pde_surf_sample_*.pkl)
    pkl_files = sorted(list(data_dir.glob('pde_*.pkl')), key=extract_number_from_filename)
    
    if not pkl_files:
        print(f"No PDE .pkl files found in {data_dir}")
        return
    
    print(f"Found {len(pkl_files)} PDE samples. Starting BOLD visualization...")
    print(f"Output directory: {output_dir}")
    
    processed = 0
    for i, pkl_file in enumerate(pkl_files[:max_samples]):
        print(f"[{i+1}/{min(max_samples, len(pkl_files))}] Processing {pkl_file.name}...")
        try:
            plot_sample_pde_bold(pkl_file, output_dir, alpha=alpha, linewidth=linewidth, max_display_nodes=max_display_nodes, debug=False)
            processed += 1
        except Exception as e:
            print(f"Error processing {pkl_file.name}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\nSuccessfully processed {processed} samples.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize BOLD signals from PDE simulation data')
    parser.add_argument('--max_samples', type=int, default=10, help='Maximum number of samples')
    parser.add_argument('--alpha', type=float, default=0.1, help='Transparency')
    parser.add_argument('--linewidth', type=float, default=0.5, help='Line width')
    parser.add_argument('--max_display_nodes', type=int, default=200, help='Max background nodes to display')
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    
    args = parser.parse_args()
    
    main(max_samples=args.max_samples, 
         alpha=args.alpha, 
         linewidth=args.linewidth,
         max_display_nodes=args.max_display_nodes,
         data_dir=args.data_dir,
         output_dir=args.output_dir)

