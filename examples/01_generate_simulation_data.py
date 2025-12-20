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
    python examples/01_generate_simulation_data.py
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import pickle

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.ode_simulator import ODESimulator
from simulation.pde_simulator import PDESimulator
from simulation.stimulation_generator import StimulationGenerator

# 配置日志
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("simulation_generation.log"),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger("SimulationDataGeneration")

def load_connectivity(file_path):
    """加载连接矩阵"""
    try:
        df = pd.read_csv(file_path, header=None)
        return df.values
    except Exception as e:
        logger.error(f"Failed to load connectivity from {file_path}: {e}")
        return None

def generate_dummy_ec(n_nodes):
    """生成虚拟有效连接矩阵 (EC)"""
    # 随机生成一个稀疏矩阵作为EC
    print("生成虚拟EC矩阵...")
    ec = np.random.randn(n_nodes, n_nodes) * 0.1
    # 稀疏化
    mask = np.random.rand(n_nodes, n_nodes) > 0.8
    ec = ec * mask
    np.fill_diagonal(ec, 0)
    return ec

def main():
    logger.info("开始生成仿真数据...")
    
    # --- 配置区域 ---
    # 设置为 True 以启用对应的仿真生成，设置为 False 以跳过
    ENABLE_ODE_EC = True   # 基于有效连接(EC)的ODE仿真
    ENABLE_ODE_SC = True   # 基于结构连接(SC)的ODE仿真
    ENABLE_PDE_SURF = True # 基于皮层表面的PDE仿真
    # ----------------

    # 路径设置
    project_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dataset_dir = project_root / 'dataset'
    output_dir = dataset_dir / 'simulation_data'
    output_dir.mkdir(exist_ok=True)
    
    # 1. 加载数据
    bna_path = dataset_dir / 'BNA_matrix_binary_246x246.csv'
    sc_matrix = load_connectivity(bna_path)
    
    if sc_matrix is None:
        logger.error("无法加载SC矩阵，使用随机矩阵代替进行测试。")
        sc_matrix = np.random.rand(246, 246) > 0.9
        sc_matrix = sc_matrix.astype(float)
        
    n_nodes = sc_matrix.shape[0]
    logger.info(f"加载SC矩阵，大小: {sc_matrix.shape}")
    
    # 2. 生成EC矩阵 (模拟刘泉影的方法)
    ec_matrix = generate_dummy_ec(n_nodes)
    logger.info("生成虚拟EC矩阵")
    
    # 3. 仿真参数
    dt = 0.1 # ms (神经动力学仿真步长)
    duration = 60000.0 # 60s
    sampling_interval = 50.0 # 50ms (采样间隔)
    n_samples = 5 # 演示用样本数，实际任务建议 1000
    
    # 初始化 ODE 仿真器和刺激生成器 (如果需要)
    ode_sim = None
    stim_gen = None
    if ENABLE_ODE_EC or ENABLE_ODE_SC:
        ode_sim = ODESimulator(n_nodes=n_nodes, dt=dt, duration=duration, model_type='EI')
        stim_gen = StimulationGenerator(n_nodes, dt, duration)

    # 4. ODE仿真 (基于EC)
    if ENABLE_ODE_EC:
        logger.info("开始ODE仿真 (基于EC)...")
        ode_sim = ODESimulator(n_nodes=n_nodes, dt=dt, duration=duration, model_type='EI')
        
        for i in range(n_samples):
            # 随机生成刺激
            onset = np.random.uniform(2000, 15000)
            stim_dur = np.random.uniform(1000, 5000)
            target_nodes = np.random.choice(n_nodes, size=10, replace=False).tolist()
            
            stimulus, stim_config = stim_gen.generate_boxcar(onset, stim_dur, amplitude=0.5, nodes=target_nodes)
            
            # 运行仿真
            results = ode_sim.run_simulation(
                connectivity=ec_matrix, 
                stimulus=stimulus, 
                stimulus_config=stim_config,
                noise_level=0.02, 
                noise_seed=i,
                sampling_interval=sampling_interval
            )
            
            # 保存结果
            save_path = output_dir / f'ode_ec_sample_{i}.pkl'
            with open(save_path, 'wb') as f:
                pickle.dump(results, f)
            logger.info(f"保存ODE样本 {i} 到 {save_path}")
    else:
        logger.info("跳过 ODE仿真 (基于EC)")
        
    # 5. ODE仿真 (基于SC - 白质)
    if ENABLE_ODE_SC:
        logger.info("开始ODE仿真 (基于SC)...")
        ode_sim = ODESimulator(n_nodes=n_nodes, dt=dt, duration=duration, model_type='EI')
        for i in range(n_samples):
            onset = np.random.uniform(2000, 15000)
            stim_dur = np.random.uniform(1000, 5000)
            target_nodes = np.random.choice(n_nodes, size=10, replace=False).tolist()
            
            stimulus, stim_config = stim_gen.generate_boxcar(onset, stim_dur, amplitude=0.5, nodes=target_nodes)
            
            results = ode_sim.run_simulation(
                connectivity=sc_matrix, 
                stimulus=stimulus, 
                stimulus_config=stim_config,
                noise_level=0.02,
                noise_seed=i,
                sampling_interval=sampling_interval
            )
            
            save_path = output_dir / f'ode_sc_sample_{i}.pkl'
            with open(save_path, 'wb') as f:
                pickle.dump(results, f)
            logger.info(f"保存ODE(SC)样本 {i} 到 {save_path}")
    else:
        logger.info("跳过 ODE仿真 (基于SC)")
        
    # 6. PDE仿真 (基于皮层Surface)
    if ENABLE_PDE_SURF:
        logger.info("开始PDE仿真 (基于皮层Surface)...")
        
        from utils import surface_utils
        # 假设只仿真左半球
        surf_file = dataset_dir / 'T103/anat/sub-01_hemi-L_midthickness.surf.gii'
        
        if surf_file.exists() and surface_utils.HAS_NIBABEL:
            try:
                vertices, faces = surface_utils.load_surface(surf_file)
                n_vertices = vertices.shape[0]
                logger.info(f"加载左半球Surface: {n_vertices} 顶点")
                
                surf_adj = surface_utils.get_mesh_adjacency(faces, n_vertices)
                
                pde_sim = PDESimulator(n_nodes=n_vertices, dt=dt, duration=duration, model_type='wave')
                stim_gen_pde = StimulationGenerator(n_nodes=n_vertices, dt=dt, duration=duration)
                
                for i in range(n_samples):
                    # 随机刺激参数
                    onset = np.random.uniform(2000, 15000)
                    stim_dur = np.random.uniform(1000, 5000)
                    target_nodes = np.random.choice(n_vertices, size=5, replace=False).tolist()
                    
                    stimulus, stim_config = stim_gen_pde.generate_boxcar(onset, stim_dur, amplitude=1.0, nodes=target_nodes)
                    
                    results = pde_sim.run_simulation(
                        connectivity=surf_adj, 
                        stimulus=stimulus, 
                        stimulus_config=stim_config,
                        noise_level=0.01, 
                        noise_seed=i,
                        sampling_interval=sampling_interval
                    )
                    
                    save_path = output_dir / f'pde_surf_sample_{i}.pkl'
                    with open(save_path, 'wb') as f:
                        pickle.dump(results, f)
                    logger.info(f"保存PDE(Surface)样本 {i} 到 {save_path}")
                    
            except Exception as e:
                logger.error(f"Surface PDE仿真失败: {e}")
        else:
            logger.warning("无法加载Surface文件或nibabel未安装，跳过Surface PDE仿真。")
    else:
        logger.info("跳过 PDE仿真 (基于皮层Surface)")
        
    logger.info("仿真数据生成完成。")

if __name__ == "__main__":
    main()
