"""
示例1：生成仿真数据

功能说明：
    生成用于训练神经算子的仿真数据集。
    包含ODE和PDE的仿真数据，覆盖多种条件。

输入：
    - 配置文件
    - 连接矩阵模板

输出：
    - 仿真数据集
    - 数据统计报告

使用方法：
    python examples/01_generate_simulation_data.py --config config/data_config.yaml
"""

import sys
sys.path.append('..')

import numpy as np
from pathlib import Path

from simulation import ODESimulator, PDESimulator
from connectivity import StructuralConnectivity, WhiteMatterConnectivity
from utils import IOUtils, Logger


def main():
    """
    主函数：生成仿真数据
    
    步骤：
    1. 加载配置
    2. 初始化仿真器
    3. 生成ODE仿真数据
    4. 生成PDE仿真数据
    5. 保存数据集
    6. 生成统计报告
    """
    
    # 设置日志
    logger = Logger('SimulationDataGeneration')
    logger.info("开始生成仿真数据...")
    
    # 加载配置
    config = IOUtils.load_config('config/data_config.yaml')
    
    # 初始化ODE仿真器
    logger.info("生成ODE仿真数据...")
    ode_simulator = ODESimulator(model_type='EI', n_nodes=246)
    
    # 生成ODE数据集
    # 覆盖不同的刺激脑区、刺激方式、有效连接、白质连接
    ode_dataset = ode_simulator.generate_dataset(
        n_samples=1000,
        vary_connectivity=True,
        vary_stimulus=True,
        connectivity_types=['effective', 'white_matter'],
        add_noise=True  # 包含随机ODE
    )
    
    # 保存ODE数据集
    save_path = Path('data/simulation/ode_dataset.pkl')
    IOUtils.create_directory(save_path.parent)
    ode_simulator.save_dataset(ode_dataset, str(save_path))
    logger.info(f"ODE数据集已保存到 {save_path}")
    
    # 初始化PDE仿真器
    logger.info("生成PDE仿真数据...")
    pde_simulator = PDESimulator(model_type='diffusion', n_vertices=10000)
    
    # 生成PDE数据集
    # 覆盖不同的空间刺激位置和模式
    pde_dataset = pde_simulator.generate_dataset(
        n_samples=500,
        vary_cortical_connectivity=True,
        vary_stimulus=True,
        add_noise=True  # 包含随机PDE
    )
    
    # 保存PDE数据集
    save_path = Path('data/simulation/pde_dataset.pkl')
    pde_simulator.save_dataset(pde_dataset, str(save_path))
    logger.info(f"PDE数据集已保存到 {save_path}")
    
    # 生成数据统计报告
    logger.info("生成数据统计报告...")
    report = f"""
    仿真数据生成报告
    ==================
    
    ODE数据集：
    - 样本数量: {ode_dataset['metadata']['n_samples']}
    - 节点数量: {ode_dataset['metadata']['n_nodes']}
    - 时间范围: {ode_dataset['metadata']['t_span']}
    - 包含噪声: {ode_dataset['metadata']['has_noise']}
    
    PDE数据集：
    - 样本数量: {pde_dataset['metadata']['n_samples']}
    - 顶点数量: {pde_dataset['metadata']['n_vertices']}
    - 时间范围: {pde_dataset['metadata']['t_span']}
    - 包含噪声: {pde_dataset['metadata']['has_noise']}
    
    数据集覆盖：
    - ✓ 不同的刺激脑区
    - ✓ 不同的刺激方式
    - ✓ 不同的有效连接
    - ✓ 不同的白质结构连接
    - ✓ 不同的皮层结构连接
    - ✓ ODE和PDE方程
    - ✓ 随机噪声项
    """
    
    print(report)
    
    # 保存报告
    with open('data/simulation/generation_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info("仿真数据生成完成！")


if __name__ == '__main__':
    main()
