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

def plot_sample_bold(file_path, output_dir, alpha=0.1, linewidth=0.5):
    """
    读取单个样本文件并可视化其 BOLD 信号
    
    参数:
        file_path: pickle 文件路径
        output_dir: 输出目录
        alpha: 单个脑区线条的透明度（默认 0.1，避免过于密集）
        linewidth: 线条宽度（默认 0.5）
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
    
    # 获取时间点
    time_points = data.get('time_points')
    if time_points is None:
        metadata = data.get('metadata', {})
        sampling_interval = metadata.get('sampling_interval', 50.0)
        duration = metadata.get('duration', 200000.0)
        time_points = np.arange(0, duration, sampling_interval)
    
    # 转换为秒
    time_s = time_points / 1000.0
    
    # 确保 BOLD 信号是 2D 数组 (n_time_steps, n_nodes)
    bold_signal = np.asarray(bold_signal)
    if bold_signal.ndim == 1:
        bold_signal = bold_signal.reshape(-1, 1)
    
    n_time_steps, n_nodes = bold_signal.shape
    
    # 计算平均值
    bold_mean = np.mean(bold_signal, axis=1)
    
    # 获取任务信息用于背景标注
    stim_config = data.get('stimulus_config')
    tasks = []
    if stim_config:
        # 兼容不同格式
        if isinstance(stim_config, dict):
            if 'tasks' in stim_config:
                tasks = stim_config['tasks']
            elif 'ode' in stim_config and 'tasks' in stim_config['ode']:
                tasks = stim_config['ode']['tasks']
    
    # 从stimulus_config中提取所有受刺激的脑区
    stimulated_regions = set()
    if tasks:
        for task in tasks:
            if 'channels' in task:
                stimulated_regions.update(task['channels'])
    
    # 创建图形
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    
    # 先绘制任务背景区域（如果有任务信息）
    if tasks:
        # 修复：使用新的 colormap API（兼容 Matplotlib 3.7+）
        try:
            # Matplotlib 3.7+ 新 API
            colors = plt.colormaps.get_cmap('Set3')
        except AttributeError:
            # 兼容旧版本
            colors = plt.get_cmap('Set3')
        
        for i, task in enumerate(tasks):
            if 'range' in task:
                t0, t1 = task['range']
                # 确保时间值是有限的
                if not (np.isfinite(t0) and np.isfinite(t1)):
                    continue
                    
                ax.axvspan(t0/1000.0, t1/1000.0, color=colors(i), alpha=0.15, 
                          label=f"Task {task.get('index', i)}" if i == 0 else "")
    
    # 绘制所有脑区的 BOLD 信号
    # 先绘制未受刺激的脑区（灰色，半透明）
    for node_idx in range(n_nodes):
        if node_idx not in stimulated_regions:
            ax.plot(time_s, bold_signal[:, node_idx], 
                   alpha=alpha, linewidth=linewidth, color='gray', zorder=1)
    
    # 再绘制受刺激的脑区（橙色/红色，更明显）
    for node_idx in stimulated_regions:
        if node_idx < n_nodes:  # 确保索引有效
            ax.plot(time_s, bold_signal[:, node_idx], 
                   alpha=0.6, linewidth=linewidth*2, color='orange',
                   label='Stimulated regions' if node_idx == min(stimulated_regions) else '', zorder=5)
    
    # 绘制平均值线（细线）
    ax.plot(time_s, bold_mean, 
           linewidth=1.0, color='red', label='Mean BOLD', zorder=10)
    
    # 在绘制数据后，添加任务文本标注（此时 y 轴范围已设置）
    if tasks:
        # 获取当前 y 轴范围（数据已绘制，范围已确定）
        y_lim = ax.get_ylim()
        y_max = y_lim[1] * 0.95  # 使用 y 轴上限的 95% 位置
        
        for i, task in enumerate(tasks):
            if 'range' in task:
                t0, t1 = task['range']
                # 确保时间值是有限的
                if not (np.isfinite(t0) and np.isfinite(t1)):
                    continue
                
                # 确保文本位置是有限的
                text_x = (t0+t1)/2000.0
                if np.isfinite(text_x) and np.isfinite(y_max):
                    ax.text(text_x, y_max, f"T{task.get('index', i)}", 
                           ha='center', fontsize=9, fontweight='bold', 
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7),
                           zorder=11)
    
    # 设置标签和标题
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('BOLD Signal', fontsize=12)
    ax.set_title(f'BOLD Signal Visualization: {file_path.name}\n'
                f'({n_nodes} brain regions, {n_time_steps} time points)', 
                fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', fontsize=10)
    
    # 添加统计信息文本框
    stats_text = (f'Mean: {np.mean(bold_mean):.4f}\n'
                 f'Std: {np.std(bold_mean):.4f}\n'
                 f'Min: {np.min(bold_mean):.4f}\n'
                 f'Max: {np.max(bold_mean):.4f}')
    ax.text(0.02, 0.98, stats_text, 
           transform=ax.transAxes, fontsize=9,
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    save_path = output_dir / f"{file_path.stem}_bold.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Successfully saved BOLD plot to {save_path}")

def plot_bold_comparison(file_paths, output_dir, max_regions=50, alpha=0.15):
    """
    对比多个样本的 BOLD 信号平均值
    
    参数:
        file_paths: 文件路径列表
        output_dir: 输出目录
        max_regions: 最多显示多少个脑区的对比（避免过于密集）
        alpha: 透明度
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    
    # 修复：使用新的 colormap API（兼容 Matplotlib 3.7+）
    try:
        # Matplotlib 3.7+ 新 API
        colors = plt.colormaps.get_cmap('tab10')
    except AttributeError:
        # 兼容旧版本
        colors = plt.get_cmap('tab10')
    
    for idx, file_path in enumerate(file_paths):
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            bold_signal = data.get('bold_signal')
            if bold_signal is None:
                continue
            
            time_points = data.get('time_points')
            if time_points is None:
                metadata = data.get('metadata', {})
                sampling_interval = metadata.get('sampling_interval', 50.0)
                duration = metadata.get('duration', 200000.0)
                time_points = np.arange(0, duration, sampling_interval)
            
            time_s = time_points / 1000.0
            bold_signal = np.asarray(bold_signal)
            if bold_signal.ndim == 1:
                bold_signal = bold_signal.reshape(-1, 1)
            
            bold_mean = np.mean(bold_signal, axis=1)
            
            ax.plot(time_s, bold_mean, 
                   linewidth=2, color=colors(idx), 
                   label=Path(file_path).stem, alpha=0.8)
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Mean BOLD Signal', fontsize=12)
    ax.set_title('BOLD Signal Comparison (Mean across regions)', 
                fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    
    save_path = output_dir / "bold_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Successfully saved comparison plot to {save_path}")

def main(max_samples=10, alpha=0.1, linewidth=0.5, data_dir=None, output_dir=None):
    """
    主函数：批量可视化 BOLD 信号
    
    参数:
        max_samples: 最多处理多少个样本（默认 10）
        alpha: 单个脑区线条的透明度（默认 0.1）
        linewidth: 线条宽度（默认 0.5）
        data_dir: 数据目录（默认 'dataset/simulation_data'）
        output_dir: 输出目录（默认 'dataset/simulation_data/plots'）
    """
    # 配置路径
    if data_dir is None:
        data_dir = Path('dataset/simulation_data')
    else:
        data_dir = Path(data_dir)
    
    if output_dir is None:
        output_dir = data_dir / 'plots' / 'bold'
    else:
        output_dir = Path(output_dir)
    
    # 获取所有 ODE 相关的 .pkl 文件，并按文件名中的数字排序
    pkl_files = sorted(list(data_dir.glob('ode_*.pkl')), key=extract_number_from_filename)
    
    if not pkl_files:
        print(f"No ODE .pkl files found in {data_dir}")
        print("Looking for any .pkl files...")
        pkl_files = sorted(list(data_dir.glob('*.pkl')), key=extract_number_from_filename)
        if not pkl_files:
            print(f"No .pkl files found in {data_dir}")
            return
    
    print(f"Found {len(pkl_files)} samples. Starting BOLD visualization...")
    print(f"Will process up to {max_samples} samples.")
    print(f"Output directory: {output_dir}")
    
    # 处理样本
    processed = 0
    for i, pkl_file in enumerate(pkl_files[:max_samples]):
        print(f"[{i+1}/{min(max_samples, len(pkl_files))}] Processing {pkl_file.name}...")
        try:
            plot_sample_bold(pkl_file, output_dir, alpha=alpha, linewidth=linewidth)
            processed += 1
        except Exception as e:
            print(f"Error processing {pkl_file.name}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\nSuccessfully processed {processed} samples.")
    
    # 如果处理了多个样本，生成对比图
    if processed > 1:
        print("\nGenerating comparison plot...")
        try:
            plot_bold_comparison(pkl_files[:max_samples], output_dir)
        except Exception as e:
            print(f"Error generating comparison plot: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize BOLD signals from ODE simulation data')
    parser.add_argument('--max_samples', type=int, default=5,
                       help='Maximum number of samples to visualize (default: 10)')
    parser.add_argument('--alpha', type=float, default=0.1,
                       help='Transparency of individual region lines (default: 0.1)')
    parser.add_argument('--linewidth', type=float, default=0.5,
                       help='Line width for individual regions (default: 0.5)')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Data directory (default: dataset/simulation_data)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (default: dataset/simulation_data/plots/bold)')
    
    args = parser.parse_args()
    
    main(max_samples=args.max_samples, 
         alpha=args.alpha, 
         linewidth=args.linewidth,
         data_dir=args.data_dir,
         output_dir=args.output_dir)

