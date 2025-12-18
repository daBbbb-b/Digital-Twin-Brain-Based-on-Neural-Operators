"""
示例4：可视化结果

功能说明：
    可视化刺激图谱与功能图谱的对比，生成图表和报告。

输入：
    - 预测的刺激图谱
    - 真实的功能图谱
    - 评估指标

输出：
    - 可视化图表
    - 对比分析图
    - 综合报告

使用方法：
    python examples/04_visualize_results.py
"""

import sys
sys.path.append('..')

import numpy as np
from pathlib import Path

from data import TaskManager
from visualization import BrainVisualizer, StimulusMapper, ComparisonPlots
from utils import IOUtils, Logger


def main():
    """
    主函数：可视化结果
    
    步骤：
    1. 加载预测结果和真实数据
    2. 绘制刺激图谱
    3. 绘制对比图表
    4. 生成综合可视化报告
    """
    
    # 设置日志
    logger = Logger('Visualization')
    logger.info("开始可视化结果...")
    
    # 加载数据
    logger.info("加载预测结果...")
    predicted_stimuli = IOUtils.load_pickle('results/predicted_stimuli.pkl')
    true_activations = IOUtils.load_pickle('results/true_activations.pkl')
    task_metrics = IOUtils.load_pickle('results/task_metrics.pkl')
    
    # 加载任务信息
    task_manager = TaskManager(task_file='data/tasks.tsv')
    task_categories = {}
    for task_id in predicted_stimuli.keys():
        task_info = task_manager.get_task_info(task_id)
        task_categories[task_id] = task_info['category']
    
    # 初始化可视化器
    brain_visualizer = BrainVisualizer(
        surface_file='data/surfaces/brain_surface.npz'
    )
    stimulus_mapper = StimulusMapper()
    comparison_plots = ComparisonPlots()
    
    # 1. 绘制刺激图谱
    logger.info("绘制刺激图谱...")
    
    # 选择几个代表性任务
    representative_tasks = list(predicted_stimuli.keys())[:5]
    
    for task_id in representative_tasks:
        # 刺激时间过程
        stimulus_mapper.plot_stimulus_timecourse(
            stimulus=predicted_stimuli[task_id],
            save_path=f'figures/timecourse/{task_id}_timecourse.png'
        )
        
        # 空间刺激模式
        stimulus_mapper.plot_spatial_pattern(
            stimulus=predicted_stimuli[task_id],
            save_path=f'figures/spatial/{task_id}_spatial.png'
        )
        
        # 刺激热图
        stimulus_mapper.plot_stimulus_heatmap(
            stimulus=predicted_stimuli[task_id],
            save_path=f'figures/heatmap/{task_id}_heatmap.png'
        )
    
    # 2. 绘制对比图表
    logger.info("绘制对比图表...")
    
    for task_id in representative_tasks:
        # 刺激vs激活对比
        comparison_plots.plot_stimulus_vs_activation(
            predicted_stimulus=np.mean(predicted_stimuli[task_id], axis=0),
            true_activation=true_activations[task_id],
            task_name=task_id,
            save_path=f'figures/comparison/{task_id}_comparison.png'
        )
    
    # 3. 一致性总结图
    logger.info("绘制一致性总结...")
    
    # 提取所有任务的相关系数
    task_correlations = {
        task_id: metrics['spatial_correlation']['pearson']
        for task_id, metrics in task_metrics.items()
    }
    
    # 绘制一致性分析
    comparison_plots.plot_consistency_analysis(
        stimuli=predicted_stimuli,
        activations=true_activations,
        metrics=task_metrics,
        save_path='figures/consistency_analysis.png'
    )
    
    # 4. 跨任务对比
    logger.info("绘制跨任务对比...")
    
    comparison_plots.plot_cross_task_comparison(
        task_metrics=task_metrics,
        metric_name='correlation',
        save_path='figures/cross_task_comparison.png'
    )
    
    # 5. 按任务类别分析
    logger.info("按任务类别分析性能...")
    
    comparison_plots.plot_task_category_performance(
        task_categories={
            cat: [tid for tid, c in task_categories.items() if c == cat]
            for cat in set(task_categories.values())
        },
        task_metrics=task_metrics,
        save_path='figures/category_performance.png'
    )
    
    # 6. 在大脑上可视化
    logger.info("在大脑surface上可视化...")
    
    for task_id in representative_tasks[:3]:
        # 平均刺激
        avg_stimulus = np.mean(predicted_stimuli[task_id], axis=0)
        
        brain_visualizer.plot_surface_map(
            surface_data=avg_stimulus,
            save_path=f'figures/brain/{task_id}_brain.png'
        )
    
    logger.info("可视化完成！所有图表已保存到 figures/ 目录")
    
    # 生成可视化报告
    report = f"""
    可视化报告
    ==========
    
    已生成的可视化：
    - 刺激时间过程图：{len(representative_tasks)}个
    - 空间刺激模式图：{len(representative_tasks)}个
    - 刺激热图：{len(representative_tasks)}个
    - 刺激vs激活对比图：{len(representative_tasks)}个
    - 一致性分析图：1个
    - 跨任务对比图：1个
    - 任务类别性能图：1个
    - 大脑surface可视化：{min(3, len(representative_tasks))}个
    
    所有图表保存在：figures/ 目录
    """
    
    print(report)
    
    with open('figures/visualization_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)


if __name__ == '__main__':
    main()
