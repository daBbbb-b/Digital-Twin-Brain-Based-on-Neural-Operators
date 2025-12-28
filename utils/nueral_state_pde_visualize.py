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

def plot_sample_pde_neural_activity(file_path, output_dir, alpha=0.1, linewidth=0.5, max_display_nodes=200):
    """
    读取单个PDE样本文件并可视化其神经活动数据 (u state)
    
    参数:
        file_path: pickle 文件路径
        output_dir: 输出目录
        alpha: 单个节点线条的透明度
        linewidth: 线条宽度
        max_display_nodes: 背景节点的最大显示数量（避免大规模网络绘图过慢）
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

    # 检查是否有神经活动数据
    neural_activity = data.get('neural_activity')
    if neural_activity is None:
        print(f"No neural_activity found in {file_path}. (Make sure saving is enabled in simulation)")
        return
    
    # 获取时间点
    time_points = data.get('time_points')
    if time_points is None:
        metadata = data.get('metadata', {})
        sampling_interval = metadata.get('sampling_interval', 50.0)
        duration = metadata.get('duration', 20000.0)
        # 估算时间步数
        if neural_activity.ndim > 0:
            n_steps = neural_activity.shape[0]
            time_points = np.arange(n_steps) * sampling_interval
        else:
            time_points = np.arange(0, duration, sampling_interval)
    
    # 转换为秒
    time_s = time_points / 1000.0
    
    # 确保神经活动数据是 2D 数组
    neural_activity = np.asarray(neural_activity)
    if neural_activity.ndim == 1:
        neural_activity = neural_activity.reshape(-1, 1)
    
    n_time_steps, n_nodes = neural_activity.shape
    
    # 计算平均值
    mean_activity = np.mean(neural_activity, axis=1)
    
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
    
    # 确定要绘制的背景节点 (未受刺激的节点)
    # 如果节点总数很大，则随机采样一部分背景节点进行绘制，以避免图像过于密集
    background_candidates = [i for i in range(n_nodes) if i not in stimulated_nodes]
    
    if len(background_candidates) > max_display_nodes:
        # 使用固定种子以保证一致性
        rng = np.random.RandomState(42)
        nodes_to_plot_bg = rng.choice(background_candidates, size=max_display_nodes, replace=False).tolist()
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
        ax.plot(time_s, neural_activity[:, node_idx], 
               alpha=alpha, linewidth=linewidth, color='gray', zorder=1)
        
    # 2. 绘制受刺激节点 (橙色，较明显)
    stimulated_nodes_list = sorted(list(stimulated_nodes))
    for i, node_idx in enumerate(stimulated_nodes_list):
        if node_idx < n_nodes:
            label = 'Stimulated Nodes' if i == 0 else None
            ax.plot(time_s, neural_activity[:, node_idx], 
                   alpha=0.6, linewidth=linewidth*1.5, color='orange', label=label, zorder=5)
    
    # 3. 绘制平均值线 (红色，细线)
    ax.plot(time_s, mean_activity, 
           linewidth=1.0, color='red', label='Mean Activity', zorder=10)
    
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
    ax.set_ylabel('Neural Activity (u)', fontsize=12)
    
    bg_note = f"(showing {len(nodes_to_plot_bg)}/{len(background_candidates)} background nodes)" if len(background_candidates) > max_display_nodes else ""
    ax.set_title(f'PDE Neural Activity Visualization: {file_path.name}\n'
                f'({n_nodes} nodes, {n_time_steps} time points) {bg_note}', 
                fontsize=13, fontweight='bold')
    
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', fontsize=10)
    
    # 添加统计信息
    stats_text = (f'Mean: {np.mean(mean_activity):.4f}\n'
                 f'Std: {np.std(mean_activity):.4f}\n'
                 f'Min: {np.min(mean_activity):.4f}\n'
                 f'Max: {np.max(mean_activity):.4f}')
    ax.text(0.02, 0.98, stats_text, 
           transform=ax.transAxes, fontsize=9,
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    save_path = output_dir / f"{file_path.stem}_neural.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Successfully saved PDE neural plot to {save_path}")

def main(max_samples=10, alpha=0.1, linewidth=0.5, data_dir=None, output_dir=None, max_display_nodes=200):
    """
    主函数：批量可视化 PDE 神经活动数据
    """
    if data_dir is None:
        data_dir = Path('dataset/simulation_data')
    else:
        data_dir = Path(data_dir)
    
    if output_dir is None:
        output_dir = data_dir / 'plots' / 'pde_neural'
    else:
        output_dir = Path(output_dir)
    
    # 获取所有 PDE 相关的 .pkl 文件 (通常是 pde_surf_sample_*.pkl)
    pkl_files = sorted(list(data_dir.glob('pde_*.pkl')), key=extract_number_from_filename)
    
    if not pkl_files:
        print(f"No PDE .pkl files found in {data_dir}")
        return
    
    print(f"Found {len(pkl_files)} PDE samples. Starting neural activity visualization...")
    print(f"Output directory: {output_dir}")
    
    processed = 0
    for i, pkl_file in enumerate(pkl_files[:max_samples]):
        print(f"[{i+1}/{min(max_samples, len(pkl_files))}] Processing {pkl_file.name}...")
        try:
            plot_sample_pde_neural_activity(pkl_file, output_dir, alpha=alpha, linewidth=linewidth, max_display_nodes=max_display_nodes)
            processed += 1
        except Exception as e:
            print(f"Error processing {pkl_file.name}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\nSuccessfully processed {processed} samples.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize neural activity from PDE simulation data')
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

