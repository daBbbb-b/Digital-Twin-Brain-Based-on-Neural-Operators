"""
示例3：在真实fMRI数据上微调

功能说明：
    在103个认知任务的fMRI数据上微调预训练的神经算子模型。

输入：
    - 预训练模型
    - 103个任务的fMRI数据
    - 有效连接数据
    - 微调配置

输出：
    - 微调后的模型
    - 每个任务的预测刺激图谱
    - 评估结果

使用方法：
    python examples/03_finetune_on_real_data.py --pretrained models/saved/fno_simulation.pth
"""

import sys
sys.path.append('..')

import torch
import numpy as np
from pathlib import Path

from models import FNO
from data import DataLoader, TaskManager
from training import FineTuner, StimulusSolver
from evaluation import RealDataMetrics, ConsistencyAnalysis
from visualization import StimulusMapper
from utils import IOUtils, Logger


def main():
    """
    主函数：微调和预测
    
    步骤：
    1. 加载预训练模型
    2. 加载103个任务数据
    3. 微调模型
    4. 预测刺激图谱
    5. 评估和分析一致性
    6. 可视化结果
    """
    
    # 设置日志
    logger = Logger('RealDataFinetuning')
    logger.info("开始在真实数据上微调...")
    
    # 加载预训练模型
    logger.info("加载预训练模型...")
    model = FNO(dim=1)  # 从配置初始化架构
    model.load_state_dict(torch.load('models/saved/fno_simulation.pth'))
    
    # 加载任务管理器
    task_manager = TaskManager(task_file='data/tasks.tsv')
    all_tasks = task_manager.get_all_tasks()
    logger.info(f"共有 {len(all_tasks)} 个任务")
    
    # 加载数据
    logger.info("加载fMRI和连接数据...")
    data_loader = DataLoader(data_root='data/real')
    
    task_fmri = {}
    task_connectivity = {}
    
    for task_id in all_tasks:
        # 加载fMRI数据（TR=2秒）
        fmri_data = data_loader.load_fmri(task_id=task_id)
        task_fmri[task_id] = fmri_data
        
        # 加载有效连接
        connectivity = data_loader.load_effective_connectivity(task_id=task_id)
        task_connectivity[task_id] = connectivity
    
    # 初始化微调器
    logger.info("初始化微调器...")
    fine_tuner = FineTuner(
        pretrained_model=model,
        config=IOUtils.load_config('config/train_config.yaml')['real_data_finetuning']
    )
    
    # 微调模型
    logger.info("在所有任务上微调...")
    finetune_results = fine_tuner.finetune_on_all_tasks(
        task_data=task_fmri,
        connectivity_data=task_connectivity,
        epochs=50,
        task_incremental=True
    )
    
    # 预测刺激图谱
    logger.info("预测刺激图谱...")
    stimulus_solver = StimulusSolver(model=model)
    
    predicted_stimuli = {}
    for task_id in all_tasks:
        stimulus = stimulus_solver.solve(
            connectivity=task_connectivity[task_id],
            observed_fmri=task_fmri[task_id]
        )
        predicted_stimuli[task_id] = stimulus
        logger.info(f"任务 {task_id} 的刺激图谱已预测")
    
    # 加载真实功能激活图谱
    logger.info("加载真实功能图谱...")
    true_activations = {}
    for task_id in all_tasks:
        # 从fMRI计算平均激活
        activation = np.mean(np.abs(task_fmri[task_id]), axis=0)
        true_activations[task_id] = activation
    
    # 评估指标
    logger.info("计算评估指标...")
    real_metrics = RealDataMetrics()
    
    # 评估所有任务
    task_metrics = real_metrics.evaluate_across_tasks(
        task_stimuli=predicted_stimuli,
        task_activations=true_activations
    )
    
    # 一致性分析（核心评价）
    logger.info("进行一致性分析...")
    consistency_analyzer = ConsistencyAnalysis()
    
    consistency_results = consistency_analyzer.analyze_consistency(
        task_stimuli=predicted_stimuli,
        task_activations=true_activations,
        task_metadata=task_manager.tasks_df
    )
    
    # 生成全面报告
    report = consistency_analyzer.generate_comprehensive_report(
        consistency_results,
        save_path='results/consistency_analysis_report.txt'
    )
    
    print("\n" + "="*60)
    print("一致性分析报告")
    print("="*60)
    print(report)
    
    # 可视化刺激图谱
    logger.info("可视化刺激图谱...")
    stimulus_mapper = StimulusMapper()
    
    # 为每个任务绘制刺激图谱
    for task_id in all_tasks[:10]:  # 示例：只可视化前10个任务
        stimulus_mapper.plot_stimulus_map(
            stimulus=predicted_stimuli[task_id]['stimulus'],
            task_name=task_id,
            save_path=f'figures/stimulus_maps/{task_id}_stimulus.png'
        )
    
    logger.info("微调和预测完成！")


if __name__ == '__main__':
    main()
