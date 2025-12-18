"""
示例2：在仿真数据上训练神经算子

功能说明：
    使用生成的仿真数据训练FNO或DeepONet模型。

输入：
    - 仿真数据集
    - 模型配置
    - 训练配置

输出：
    - 训练好的模型
    - 训练日志和指标
    - 评估结果

使用方法：
    python examples/02_train_on_simulation.py --model fno --config config/train_config.yaml
"""

import sys
sys.path.append('..')

import torch
from pathlib import Path

from models import FNO, DeepONet
from training import Trainer
from evaluation import SimulationMetrics
from utils import IOUtils, Logger


def main():
    """
    主函数：训练神经算子
    
    步骤：
    1. 加载仿真数据
    2. 初始化模型
    3. 训练模型
    4. 评估模型
    5. 保存模型和结果
    """
    
    # 设置日志
    logger = Logger('SimulationTraining')
    logger.info("开始训练神经算子...")
    
    # 加载配置
    train_config = IOUtils.load_config('config/train_config.yaml')
    model_config = IOUtils.load_config('config/model_config.yaml')
    
    # 加载仿真数据
    logger.info("加载仿真数据...")
    ode_dataset = IOUtils.load_pickle('data/simulation/ode_dataset.pkl')
    
    # 初始化模型（这里使用FNO1d作为示例）
    logger.info("初始化FNO模型...")
    model = FNO(
        dim=1,
        modes=model_config['fno']['fno1d']['modes'],
        width=model_config['fno']['fno1d']['width'],
        n_layers=model_config['fno']['fno1d']['n_layers']
    )
    
    logger.info(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
    
    # 初始化训练器
    logger.info("初始化训练器...")
    trainer = Trainer(
        model=model,
        config=train_config['simulation_training'],
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # 训练模型
    logger.info("开始训练...")
    training_history = trainer.train(
        train_dataset=ode_dataset['train'],
        val_dataset=ode_dataset['val'],
        epochs=train_config['simulation_training']['hyperparameters']['epochs'],
        save_dir='checkpoints/simulation'
    )
    
    logger.info("训练完成！")
    
    # 评估模型
    logger.info("评估模型性能...")
    metrics_calculator = SimulationMetrics()
    
    # 在测试集上评估
    test_predictions = trainer.predict(ode_dataset['test'])
    test_metrics = metrics_calculator.compute_metrics(
        predicted=test_predictions,
        ground_truth=ode_dataset['test']['target']
    )
    
    # 按条件评估
    condition_metrics = metrics_calculator.evaluate_by_condition(
        predictions=test_predictions,
        ground_truths=ode_dataset['test']['target'],
        conditions=ode_dataset['test']['conditions']
    )
    
    # 生成评估报告
    report = metrics_calculator.generate_report(
        {
            'overall': test_metrics,
            'by_condition': condition_metrics,
            'training_history': training_history
        },
        save_path='results/simulation_training_report.txt'
    )
    
    print("\n" + "="*50)
    print("训练评估结果")
    print("="*50)
    print(report)
    
    # 保存最终模型
    torch.save(model.state_dict(), 'models/saved/fno_simulation.pth')
    logger.info("模型已保存到 models/saved/fno_simulation.pth")


if __name__ == '__main__':
    main()
