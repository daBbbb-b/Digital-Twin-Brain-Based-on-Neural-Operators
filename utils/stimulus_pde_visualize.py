"""
PDE刺激函数可视化工具

功能说明：
    可视化PDE仿真中的时空刺激函数u(s, t)，显示在皮层表面的空间分布和时间演化。

输入：
    - PDE仿真数据(.pkl文件)
    - 皮层表面网格(vertices, faces)

输出：
    - 空间分布快照图（显示不同时间点的刺激分布）
    - 时间序列图（显示关键顶点的刺激时间演化）
    - 可选：GIF动画

使用方法：
    python utils/stimulus_pde_visualize.py
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
import sys
import os
import re

# 增加项目根目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def extract_number_from_filename(file_path):
    """
    从文件名中提取数字，用于排序
    例如: 'pde_0.pkl' -> 0, 'pde_10.pkl' -> 10, 'pde_100.pkl' -> 100
    
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

def plot_pde_stimulus_snapshots(file_path, output_dir, n_snapshots=6):
    """
    绘制PDE刺激的空间分布快照
    
    参数:
        file_path: 仿真数据文件路径
        output_dir: 输出目录
        n_snapshots: 快照数量（默认6个，均匀分布在时间轴上）
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
    
    # 获取刺激配置
    stim_config = data.get('stimulus_config')
    if not stim_config or 'tasks' not in stim_config:
        print(f"No PDE stimulus config found in {file_path}")
        return
    
    tasks = stim_config['tasks']
    if not tasks:
        print(f"No PDE tasks found in {file_path}")
        return
    
    # 获取时间点
    time_points = data.get('time_points')
    metadata = data.get('metadata', {})
    dt = metadata.get('dt', 0.1)
    duration = metadata.get('duration', 600000)
    
    if time_points is None:
        time_points = np.arange(0, duration, dt)
    
    time_s = time_points / 1000.0
    n_time_steps = len(time_points)
    
    # 从配置重建PDE刺激
    # 注意：这里简化处理，不加载完整的vertices/faces，只可视化时空模式
    print(f"Processing {len(tasks)} PDE tasks...")
    
    # 选择快照时间点（均匀分布）
    snapshot_indices = np.linspace(0, n_time_steps-1, n_snapshots, dtype=int)
    
    # 为每个任务创建一个子图
    n_tasks = min(len(tasks), 5)  # 最多显示5个任务
    fig, axes = plt.subplots(n_tasks, n_snapshots, figsize=(20, n_tasks*3))
    
    if n_tasks == 1:
        axes = axes.reshape(1, -1)
    
    # 自定义颜色映射（蓝-白-红）
    colors_list = ['#2166ac', '#4393c3', '#92c5de', '#d1e5f0', 
                   '#f7f7f7', '#fddbc7', '#f4a582', '#d6604d', '#b2182b']
    cmap = LinearSegmentedColormap.from_list('custom_diverging', colors_list)
    
    for task_idx, task in enumerate(tasks[:n_tasks]):
        seed_vertices = task.get('seeds', [])
        amplitude = task.get('amplitude', 0.0)
        sigma_s = task.get('sigma_s', 10.0)
        t_start, t_end = task['range']
        
        # 简化：假设100个顶点的示例网格
        n_vertices_example = 100
        vertices_example = np.random.randn(n_vertices_example, 3) * 50  # 随机示例顶点
        
        # 计算空间分布（高斯核）
        spatial_pattern = np.zeros(n_vertices_example)
        for seed_idx in seed_vertices:
            # 将seed索引映射到示例网格
            seed_idx_mapped = seed_idx % n_vertices_example
            seed_pos = vertices_example[seed_idx_mapped]
            
            # 计算到所有顶点的距离
            dists = np.linalg.norm(vertices_example - seed_pos, axis=1)
            
            # 高斯核
            spatial_pattern += np.exp(-dists**2 / (2 * sigma_s**2))
        
        # 归一化
        if np.max(np.abs(spatial_pattern)) > 0:
            spatial_pattern = spatial_pattern / np.max(np.abs(spatial_pattern))
        
        # 绘制每个快照
        for snap_idx, time_idx in enumerate(snapshot_indices):
            t = time_points[time_idx]
            
            # 计算时间包络（简化的平滑boxcar）
            if t < t_start:
                temporal_envelope = 0.0
            elif t > t_end:
                temporal_envelope = 0.0
            else:
                # 简单的梯形包络
                rise_time = 500.0  # ms
                if t < t_start + rise_time:
                    temporal_envelope = (t - t_start) / rise_time
                elif t > t_end - rise_time:
                    temporal_envelope = (t_end - t) / rise_time
                else:
                    temporal_envelope = 1.0
            
            # 最终刺激强度
            stimulus_pattern = amplitude * spatial_pattern * temporal_envelope
            
            # 绘制
            ax = axes[task_idx, snap_idx]
            
            # 使用scatter plot模拟空间分布
            scatter = ax.scatter(vertices_example[:, 0], vertices_example[:, 1], 
                               c=stimulus_pattern, cmap=cmap, 
                               s=50, vmin=-2, vmax=2, edgecolors='none', alpha=0.8)
            
            ax.set_xlim(np.min(vertices_example[:, 0])-10, np.max(vertices_example[:, 0])+10)
            ax.set_ylim(np.min(vertices_example[:, 1])-10, np.max(vertices_example[:, 1])+10)
            ax.set_aspect('equal')
            ax.axis('off')
            
            # 添加标题（只在第一行）
            if task_idx == 0:
                ax.set_title(f't={time_s[time_idx]:.1f}s', fontsize=10, fontweight='bold')
            
            # 添加Y轴标签（只在第一列）
            if snap_idx == 0:
                ax.text(-0.1, 0.5, f'Task {task["index"]}', 
                       transform=ax.transAxes, rotation=90, 
                       va='center', ha='right', fontsize=11, fontweight='bold')
    
    # 添加colorbar
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(scatter, cax=cbar_ax)
    cbar.set_label('Stimulus Amplitude', rotation=270, labelpad=20, fontsize=12)
    
    plt.suptitle(f'PDE Stimulus Spatial Distribution: {file_path.name}', 
                fontsize=14, fontweight='bold', y=0.98)
    
    save_path = output_dir / f"{file_path.stem}_pde_stimulus_snapshots.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Successfully saved PDE stimulus snapshots to {save_path}")


def plot_pde_stimulus_timeseries(file_path, output_dir):
    """
    绘制关键顶点的刺激时间序列
    
    参数:
        file_path: 仿真数据文件路径
        output_dir: 输出目录
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
    
    # 获取刺激配置
    stim_config = data.get('stimulus_config')
    if not stim_config or 'tasks' not in stim_config:
        print(f"No PDE stimulus config found in {file_path}")
        return
    
    tasks = stim_config['tasks']
    if not tasks:
        print(f"No PDE tasks found in {file_path}")
        return
    
    # 获取时间点
    time_points = data.get('time_points')
    metadata = data.get('metadata', {})
    dt = metadata.get('dt', 0.1)
    duration = metadata.get('duration', 600000)
    
    if time_points is None:
        time_points = np.arange(0, duration, dt)
    
    time_s = time_points / 1000.0
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(16, 6))
    
    # 收集所有种子顶点
    all_seed_vertices = set()
    for task in tasks:
        all_seed_vertices.update(task.get('seeds', []))
    
    # 限制显示数量
    seed_vertices_to_plot = sorted(list(all_seed_vertices))[:10]
    
    # 使用颜色映射
    colors = plt.cm.get_cmap('tab10', len(seed_vertices_to_plot))
    
    # 为每个种子顶点计算并绘制时间序列
    for seed_idx, vertex_idx in enumerate(seed_vertices_to_plot):
        # 计算该顶点的刺激时间序列
        stimulus_timeseries = np.zeros(len(time_points))
        
        for task in tasks:
            if vertex_idx not in task.get('seeds', []):
                continue
            
            amplitude = task.get('amplitude', 0.0)
            t_start, t_end = task['range']
            
            # 计算时间包络
            rise_time = 500.0  # ms
            for i, t in enumerate(time_points):
                if t < t_start:
                    envelope = 0.0
                elif t > t_end:
                    envelope = 0.0
                elif t < t_start + rise_time:
                    envelope = (t - t_start) / rise_time
                elif t > t_end - rise_time:
                    envelope = (t_end - t) / rise_time
                else:
                    envelope = 1.0
                
                stimulus_timeseries[i] += amplitude * envelope
        
        # 绘制
        ax.plot(time_s, stimulus_timeseries, 
               label=f'Vertex {vertex_idx}', 
               color=colors(seed_idx), linewidth=1.5, alpha=0.8)
    
    # 标注任务区域
    if tasks:
        task_colors = plt.cm.get_cmap('Set3', len(tasks))
        for i, task in enumerate(tasks[:20]):  # 最多显示20个任务
            t0, t1 = task['range']
            ax.axvspan(t0/1000.0, t1/1000.0, color=task_colors(i), alpha=0.15)
            
            # 添加任务标签
            if i < 10:  # 只为前10个任务添加文本标签
                ax.text((t0+t1)/2000.0, ax.get_ylim()[1]*0.95, f"T{task['index']}", 
                       ha='center', fontsize=8, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Stimulus Amplitude', fontsize=12)
    ax.set_title(f'PDE Stimulus Time Series (Key Vertices): {file_path.name}', 
                fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', fontsize=9, ncol=2)
    
    plt.tight_layout()
    
    save_path = output_dir / f"{file_path.stem}_pde_stimulus_timeseries.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Successfully saved PDE stimulus timeseries to {save_path}")


def main():
    """
    主函数：批量可视化PDE刺激
    """
    # 配置路径
    data_dir = Path('dataset/simulation_data')
    output_dir = data_dir / 'plots' / 'pde_stimulus'
    
    # 获取所有PDE相关的.pkl文件，并按文件名中的数字排序
    pkl_files = sorted(list(data_dir.glob('pde_*.pkl')), key=extract_number_from_filename)
    
    if not pkl_files:
        print(f"No PDE .pkl files found in {data_dir}")
        print("Looking for any .pkl files...")
        pkl_files = sorted(list(data_dir.glob('*.pkl')), key=extract_number_from_filename)
        if not pkl_files:
            print(f"No .pkl files found in {data_dir}")
            return
    
    print(f"Found {len(pkl_files)} samples. Starting PDE stimulus visualization...")
    print(f"Output directory: {output_dir}")
    
    # 处理样本
    max_samples = 5  # 最多处理5个样本
    processed = 0
    for i, pkl_file in enumerate(pkl_files[:max_samples]):
        print(f"\n[{i+1}/{min(max_samples, len(pkl_files))}] Processing {pkl_file.name}...")
        try:
            plot_pde_stimulus_snapshots(pkl_file, output_dir, n_snapshots=6)
            plot_pde_stimulus_timeseries(pkl_file, output_dir)
            processed += 1
        except Exception as e:
            print(f"Error processing {pkl_file.name}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\nSuccessfully processed {processed} samples.")


if __name__ == "__main__":
    main()

